#!/usr/bin/env python3
"""
stages/probes/probes_orb.py — Downstream material-property prediction from
frozen ORB v3 embeddings.

Central question
----------------
ORB v3 is a *conservative interatomic potential* — trained on energy/forces.
Can its frozen embeddings predict Materials Project properties it was never
explicitly trained on, to what accuracy, and which layer's embedding is best?

For each of the 6 ORB layers (encoder, gnn_stacks_0..4; backbone excluded as it
is essentially identical to gnn_stacks_4) and 3
mean-pooled representations (node[256], edge[256], concat[512]) we train an
XGBoost model and a linear baseline (Ridge for regression, Logistic for
classification) to predict each target. A fixed per-target 80/20 split is
reused across every layer/rep/model so layer comparisons are apples-to-apples.

Targets mirror the k-NN purity + LVR analyses in stages/geometry/organisation_orb.py:
  - Regression (LVR_MAT_PROPS + mag_density): formation_energy/atom,
    energy_above_hull, band_gap, density, bulk_modulus, mag_density.
  - Classification (CAT_PROPS): crystal_system, metal_insulator, magnetic_label.
Per-atom (element_identity) and per-bond (bond_type/bond_distance) labels are
OUT OF SCOPE here: pooled structure features predict structure-level targets.

Pipeline (idempotent — every stage guarded by existence checks):
  1. Feature cache : read pooled `structure` / `edge_structure_emb` from every
                     .pt file -> features/{layer}_pooled.npz
  2. Target table  : features/targets.npz (mp_ids + all targets)
  3. Splits + train: splits/{target}.npz, metrics.csv (+ optional models/)
  4. Plots         : layer curves, parity / confusion, summary heatmap,
                     best-layer table, metrics_summary.txt

Reads  $MLIP_PROBES_ROOT/results/orb/{layer}/{nodes,edges}/mp-*.pt
Writes $MLIP_PROBES_ROOT/results/orb/property_prediction/

Examples
--------
  # smoke test on a small subset (separate output dir so it never clobbers a
  # full-run cache):
  uv run python scripts/stages/probes/probes_orb.py \
      --limit 2000 --targets band_gap,crystal_system \
      --output-dir $MLIP_PROBES_ROOT/results/orb/property_prediction_smoke

  # full run (idempotent, resumable) — run inside tmux:
  uv run python scripts/stages/probes/probes_orb.py --num-workers 8
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import argparse
import csv
import math
import multiprocessing as mpr
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["legend.frameon"] = False  # no box border around figure legends
import matplotlib.pyplot as plt
import numpy as np
import torch

import xgboost as xgb
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, mean_absolute_error, r2_score,
                             root_mean_squared_error)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlipprobes import common as qc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Data-flow order (input -> output), matching stages/geometry/organisation_orb.py.
# The ORB "backbone" layer is essentially identical to gnn_stacks_4, so it is
# excluded; gnn_stacks_4 is the final layer.
ALL_LAYERS = [
    "encoder",
    "gnn_stacks_0", "gnn_stacks_1", "gnn_stacks_2",
    "gnn_stacks_3", "gnn_stacks_4",
]
LAYER_LABELS = {
    "encoder": "encoder", "gnn_stacks_0": "gnn_0", "gnn_stacks_1": "gnn_1",
    "gnn_stacks_2": "gnn_2", "gnn_stacks_3": "gnn_3", "gnn_stacks_4": "gnn_4",
}

# Regression targets = mlipprobes.common.CONT_PROPS (LVR_MAT_PROPS + mag_density).
REG_PROPS = list(qc.CONT_PROPS)
# Classification targets = mlipprobes.common.CAT_PROPS.
CAT_PROPS = list(qc.CAT_PROPS)

# Property order used in the k-NN purity + LVR summary tables of
# stages/geometry/organisation_orb.py: the Categorical (k-NN purity) block first
# (crystal_system, metal_insulator, magnetic_label), then the Continuous (LVR)
# block (LVR_MAT_PROPS order, with mag_density appended last). REG_PROPS already
# follows the LVR order and CAT_PROPS the k-NN order, so this is just CAT + REG.
SUMMARY_ORDER = CAT_PROPS + REG_PROPS

# Physical units for regression targets (for interpretable error sizes).
UNITS = {
    "formation_energy/atom": "eV/atom", "energy_above_hull": "eV/atom",
    "band_gap": "eV", "density": "g/cm^3", "bulk_modulus": "GPa",
    "mag_density": "uB/A^3",
}

# Scientific grouping (see module docstring). The energy-derived targets are
# the only ones close to ORB's training objective.
NEAR_IN_DOMAIN    = {"formation_energy/atom", "energy_above_hull"}
GEOMETRIC_CONTROL = {"density"}

REPS = ["node", "edge", "concat"]
EMB_DIM = 256

REP_COLORS  = {"node": "#1f77b4", "edge": "#d62728", "concat": "#2ca02c"}
# Distinguish models by colour intensity (not linestyle): xgb = strong/full
# colour, linear = lighter shade of the same rep colour. lw also scales.
MODEL_LIGHTEN = {"xgb": 0.0, "linear": 0.55, "rf": 0.25, "mlp": 0.75}
MODEL_LW      = {"xgb": 2.3, "linear": 1.4, "rf": 1.9, "mlp": 1.1}


def _lighten(color: str, amount: float):
    """Blend a colour toward white. amount=0 -> original, 1 -> white."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)

# Friendly CLI aliases -> canonical qc prop names.
TARGET_ALIASES = {
    "formation_energy_per_atom": "formation_energy/atom",
    "formation_energy":          "formation_energy/atom",
}

CSV_FIELDS = [
    "target", "task", "rep", "layer", "model", "n_train", "n_test",
    "mae", "rmse", "r2", "acc", "macro_f1",
    "base_mae", "base_rmse", "base_acc", "base_f1",
]


def safe(name: str) -> str:
    """Filesystem-safe token for a target name."""
    return name.replace("/", "_per_")


def is_clf(target: str) -> bool:
    return target in CAT_PROPS


def primary_metric_name(target: str) -> str:
    return "macro_f1" if is_clf(target) else "r2"


def order_targets(targets) -> list[str]:
    """Sort targets into the script-03 k-NN/LVR summary-table order."""
    s = set(targets)
    return [t for t in SUMMARY_ORDER if t in s]


# ---------------------------------------------------------------------------
# Stage 1 — feature cache
# ---------------------------------------------------------------------------

def _read_pooled_chunk(args):
    """Worker: read pooled node + edge vectors for a contiguous slice of ids."""
    layer, emb_dir, mp_ids = args
    node_dir = Path(emb_dir) / layer / "nodes"
    edge_dir = Path(emb_dir) / layer / "edges"
    node = np.full((len(mp_ids), EMB_DIM), np.nan, dtype=np.float32)
    edge = np.full((len(mp_ids), EMB_DIM), np.nan, dtype=np.float32)
    for i, m in enumerate(mp_ids):
        np_p = node_dir / f"{m}.pt"
        if np_p.exists():
            node[i] = torch.load(np_p, weights_only=False)["structure"].float().numpy()
        ep_p = edge_dir / f"{m}.pt"
        if ep_p.exists():
            edge[i] = torch.load(ep_p, weights_only=False)["edge_structure_emb"].float().numpy()
    return node, edge


def build_feature_cache(emb_dir: Path, feat_dir: Path,
                        mp_ids: list[str], num_workers: int) -> None:
    feat_dir.mkdir(parents=True, exist_ok=True)
    mp_arr = np.array(mp_ids, dtype=object)
    n = len(mp_ids)
    for layer in ALL_LAYERS:
        out_p = feat_dir / f"{layer}_pooled.npz"
        if out_p.exists():
            existing = np.load(out_p, allow_pickle=True)["mp_ids"]
            if existing.shape == mp_arr.shape and np.array_equal(existing, mp_arr):
                print(f"  [cache] {layer}: up to date ({n:,} mats), skip")
                continue
            print(f"  [cache] {layer}: stale (row mismatch) -> rebuild")
        t0 = time.time()
        chunk = max(1, math.ceil(n / max(1, num_workers)))
        jobs = [(layer, str(emb_dir), mp_ids[i:i + chunk]) for i in range(0, n, chunk)]
        if num_workers == 1:
            results = [_read_pooled_chunk(j) for j in jobs]
        else:
            with mpr.Pool(num_workers) as pool:
                results = pool.map(_read_pooled_chunk, jobs)
        node = np.concatenate([r[0] for r in results], axis=0)
        edge = np.concatenate([r[1] for r in results], axis=0)
        n_edge_nan = int(np.isnan(edge).any(axis=1).sum())
        np.savez(out_p, mp_ids=mp_arr, node=node, edge=edge)
        print(f"  [cache] {layer}: {n:,} mats, edge-NaN={n_edge_nan:,}, "
              f"{time.time() - t0:.0f}s -> {out_p.name}")


# ---------------------------------------------------------------------------
# Stage 2 — target table
# ---------------------------------------------------------------------------

def build_target_table(structs: dict, feat_dir: Path, mp_ids: list[str]) -> None:
    out_p = feat_dir / "targets.npz"
    mp_arr = np.array(mp_ids, dtype=object)
    if out_p.exists():
        existing = np.load(out_p, allow_pickle=True)["mp_ids"]
        if existing.shape == mp_arr.shape and np.array_equal(existing, mp_arr):
            print(f"  [targets] up to date ({len(mp_ids):,} mats), skip")
            return
        print("  [targets] stale -> rebuild")
    data: dict[str, np.ndarray] = {"mp_ids": mp_arr}
    for p in REG_PROPS:
        data[f"reg__{p}"] = qc.cont_array(structs, mp_arr, p)
    for p in CAT_PROPS:
        data[f"cat__{p}"] = qc.cat_array(structs, mp_arr, p)
    np.savez(out_p, **data)
    print(f"  [targets] wrote {len(REG_PROPS)} regression + {len(CAT_PROPS)} "
          f"classification columns -> {out_p.name}")


# ---------------------------------------------------------------------------
# Stage 3 — splits + train/eval
# ---------------------------------------------------------------------------

def load_canonical_targets(feat_dir: Path, targets: list[str]):
    """Return (canonical_mp_ids, {target -> aligned value array})."""
    # Canonical order = the order stored in any layer cache (all identical).
    layer_p = next(feat_dir.glob("*_pooled.npz"))
    canon = np.load(layer_p, allow_pickle=True)["mp_ids"]
    td = np.load(feat_dir / "targets.npz", allow_pickle=True)
    tgt_ids = td["mp_ids"]
    if np.array_equal(tgt_ids, canon):
        reorder = np.arange(len(canon))
    else:  # defensive: align targets to the feature order by mp_id
        pos = {m: i for i, m in enumerate(tgt_ids)}
        reorder = np.array([pos[m] for m in canon])
    out = {}
    for t in targets:
        key = f"cat__{t}" if is_clf(t) else f"reg__{t}"
        out[t] = td[key][reorder]
    return canon, out


def valid_mask(target: str, y: np.ndarray) -> np.ndarray:
    if is_clf(target):
        return np.array([v is not None for v in y])
    return np.isfinite(y.astype(float))


def get_split(target: str, y: np.ndarray, mp_ids: np.ndarray,
              test_size: float, seed: int, splits_dir: Path):
    splits_dir.mkdir(parents=True, exist_ok=True)
    sp = splits_dir / f"{safe(target)}.npz"
    if sp.exists():
        d = np.load(sp, allow_pickle=True)
        return d["train_idx"], d["test_idx"]
    valid = np.where(valid_mask(target, y))[0]
    strat = y[valid] if is_clf(target) else None
    tr, te = train_test_split(valid, test_size=test_size,
                              random_state=seed, stratify=strat)
    np.savez(sp, train_idx=tr, test_idx=te,
             train_mp_ids=mp_ids[tr], test_mp_ids=mp_ids[te])
    return tr, te


def build_X(node: np.ndarray, edge: np.ndarray, rep: str) -> np.ndarray:
    if rep == "node":
        return node
    if rep == "edge":
        return edge
    return np.concatenate([node, edge], axis=1)


def make_model(task: str, model: str, seed: int, n_jobs: int):
    if task == "reg":
        if model == "xgb":
            return xgb.XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                n_jobs=n_jobs, random_state=seed)
        if model == "linear":
            return make_pipeline(StandardScaler(),
                                 RidgeCV(alphas=np.logspace(-2, 3, 6)))
        if model == "rf":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=200, n_jobs=n_jobs,
                                         random_state=seed)
        if model == "mlp":
            from sklearn.neural_network import MLPRegressor
            return make_pipeline(StandardScaler(),
                                 MLPRegressor(hidden_layer_sizes=(256, 128),
                                              max_iter=300, random_state=seed))
    else:
        if model == "xgb":
            return xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                n_jobs=n_jobs, random_state=seed, eval_metric="mlogloss")
        if model == "linear":
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000, C=1.0,
                                                    n_jobs=n_jobs))
        if model == "rf":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=200, n_jobs=n_jobs,
                                          random_state=seed)
        if model == "mlp":
            from sklearn.neural_network import MLPClassifier
            return make_pipeline(StandardScaler(),
                                 MLPClassifier(hidden_layer_sizes=(256, 128),
                                               max_iter=300, random_state=seed))
    raise ValueError(f"unknown model {model}")


def _prep_rows(X_full: np.ndarray, idx: np.ndarray):
    """Select rows by idx and drop any with non-finite features."""
    X = X_full[idx]
    finite = np.isfinite(X).all(axis=1)
    return X[finite], idx[finite]


def fit_eval(target, task, rep, layer, model, node, edge, y, le,
             train_idx, test_idx, seed, n_jobs):
    """Train one (target, rep, layer, model) and return a metrics row + preds."""
    X_full = build_X(node, edge, rep)
    Xtr, tr = _prep_rows(X_full, train_idx)
    Xte, te = _prep_rows(X_full, test_idx)
    if len(tr) < 10 or len(te) < 5:
        return None, None
    row = {f: "" for f in CSV_FIELDS}
    row.update(target=target, task=task, rep=rep, layer=layer, model=model,
               n_train=len(tr), n_test=len(te))

    if task == "reg":
        ytr, yte = y[tr].astype(float), y[te].astype(float)
        est = make_model(task, model, seed, n_jobs).fit(Xtr, ytr)
        pred = est.predict(Xte)
        base = np.full_like(yte, ytr.mean())
        row.update(
            mae=mean_absolute_error(yte, pred),
            rmse=root_mean_squared_error(yte, pred),
            r2=r2_score(yte, pred),
            base_mae=mean_absolute_error(yte, base),
            base_rmse=root_mean_squared_error(yte, base))
        return row, (yte, pred)

    # classification
    ytr = le.transform(y[tr]); yte = le.transform(y[te])
    est = make_model(task, model, seed, n_jobs).fit(Xtr, ytr)
    pred = est.predict(Xte)
    maj = np.bincount(ytr).argmax()
    base = np.full_like(yte, maj)
    row.update(
        acc=accuracy_score(yte, pred),
        macro_f1=f1_score(yte, pred, average="macro"),
        base_acc=accuracy_score(yte, base),
        base_f1=f1_score(yte, base, average="macro"))
    return row, (yte, pred)


def read_done_keys(metrics_csv: Path) -> set:
    done = set()
    if metrics_csv.exists():
        with metrics_csv.open() as f:
            for r in csv.DictReader(f):
                done.add((r["target"], r["rep"], r["layer"], r["model"]))
    return done


def append_row(metrics_csv: Path, row: dict) -> None:
    new = not metrics_csv.exists()
    with metrics_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def train_all(feat_dir, out_dir, targets, models, test_size, seed, n_jobs):
    splits_dir = out_dir / "splits"
    metrics_csv = out_dir / "metrics.csv"
    canon, ydict = load_canonical_targets(feat_dir, targets)

    # Precompute splits + label encoders (independent of layer/rep/model).
    encoders = {}
    for t in targets:
        get_split(t, ydict[t], canon, test_size, seed, splits_dir)
        if is_clf(t):
            v = valid_mask(t, ydict[t])
            encoders[t] = LabelEncoder().fit(ydict[t][v])

    done = read_done_keys(metrics_csv)
    total = len(targets) * len(ALL_LAYERS) * len(REPS) * len(models)
    print(f"  {total} combos ({len(targets)} targets x {len(ALL_LAYERS)} layers "
          f"x {len(REPS)} reps x {len(models)} models); {len(done)} already done")

    # Layer-outer loop -> each layer cache loaded once (memory efficient).
    for layer in ALL_LAYERS:
        d = np.load(feat_dir / f"{layer}_pooled.npz", allow_pickle=True)
        node, edge = d["node"], d["edge"]
        for t in targets:
            task = "clf" if is_clf(t) else "reg"
            y = ydict[t]
            tr_idx, te_idx = get_split(t, y, canon, test_size, seed, splits_dir)
            le = encoders.get(t)
            for rep in REPS:
                for model in models:
                    if (t, rep, layer, model) in done:
                        continue
                    t0 = time.time()
                    row, _ = fit_eval(t, task, rep, layer, model, node, edge,
                                      y, le, tr_idx, te_idx, seed, n_jobs)
                    if row is None:
                        print(f"    skip {t}/{rep}/{layer}/{model} (too few rows)")
                        continue
                    append_row(metrics_csv, row)
                    pm = primary_metric_name(t)
                    print(f"    {t}/{rep}/{LAYER_LABELS[layer]}/{model}: "
                          f"{pm}={float(row[pm]):.3f}  ({time.time()-t0:.1f}s)")
        del node, edge, d
    print(f"  metrics -> {metrics_csv}")


# ---------------------------------------------------------------------------
# Stage 4 — plots
# ---------------------------------------------------------------------------

def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _ccc(yt, yp):
    """Lin's concordance correlation coefficient: agreement with the y=x parity
    line (penalises bias and scale shifts, not just correlation like R2). 1 =
    perfect parity, 0 = no agreement, <0 = systematic disagreement."""
    yt = np.asarray(yt, dtype=float); yp = np.asarray(yp, dtype=float)
    vt, vp = yt.var(), yp.var()
    mt, mp = yt.mean(), yp.mean()
    cov = np.mean((yt - mt) * (yp - mp))
    den = vt + vp + (mt - mp) ** 2
    return float(2 * cov / den) if den > 0 else float("nan")


def load_metrics(metrics_csv: Path) -> list[dict]:
    with metrics_csv.open() as f:
        return list(csv.DictReader(f))


def _metric_lookup(rows, target, rep, layer, model, metric):
    for r in rows:
        if (r["target"] == target and r["rep"] == rep and r["layer"] == layer
                and r["model"] == model):
            return _to_float(r[metric])
    return float("nan")


def plot_layer_curves(rows, targets, models, plots_dir):
    d = plots_dir / "layer_curves"
    d.mkdir(parents=True, exist_ok=True)
    xs = np.arange(len(ALL_LAYERS))
    xlabels = [LAYER_LABELS[l] for l in ALL_LAYERS]
    for t in targets:
        pm = primary_metric_name(t)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for rep in REPS:
            for model in models:
                ys = [_metric_lookup(rows, t, rep, l, model, pm) for l in ALL_LAYERS]
                ax.plot(xs, ys, "-",
                        color=_lighten(REP_COLORS[rep], MODEL_LIGHTEN[model]),
                        lw=MODEL_LW[model], marker="o", ms=4,
                        label=f"{rep} / {model}")
        ax.set_xticks(xs); ax.set_xticklabels(xlabels, rotation=30, ha="right")
        ax.set_ylabel(pm.upper()); ax.set_xlabel("ORB layer")
        ax.set_ylim(top=1.0)
        ax.set_title(f"{t} — {pm.upper()} across ORB layers")
        ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
        fig.savefig(d / f"{safe(t)}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  [plots] layer curves -> {d}")


def _render_heatmap(rows, targets, plots_dir, out_name, title, value_fn):
    """Targets x layers heatmap of XGB primary metric; `value_fn(t, layer)->float`."""
    ordered = order_targets(targets)
    M = np.full((len(ordered), len(ALL_LAYERS)), np.nan)
    for i, t in enumerate(ordered):
        for j, l in enumerate(ALL_LAYERS):
            M[i, j] = value_fn(t, l)
    fig, ax = plt.subplots(figsize=(1.0 * len(ALL_LAYERS) + 2, 0.5 * len(ordered) + 2))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(ALL_LAYERS)))
    ax.set_xticklabels([LAYER_LABELS[l] for l in ALL_LAYERS], rotation=30, ha="right")
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([f"{t}*" if t in NEAR_IN_DOMAIN else
                        (f"{t} (ctrl)" if t in GEOMETRIC_CONTROL else t)
                        for t in ordered])
    for i in range(len(ordered)):
        for j in range(len(ALL_LAYERS)):
            if not math.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="w" if M[i, j] < 0.6 else "k", fontsize=7)
    ax.set_title(f"{title}\n(* = near in-domain / energy-derived; ctrl = geometric)")
    fig.colorbar(im, ax=ax, label="R2 (reg) / macro-F1 (clf)")
    fig.savefig(plots_dir / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] heatmap -> {plots_dir / out_name}")


def plot_summary_heatmap(rows, targets, plots_dir):
    """Best-over-reps heatmap plus one per representation (node/edge/concat)."""
    def best_over_reps(t, l):
        vals = [_metric_lookup(rows, t, rep, l, "xgb", primary_metric_name(t))
                for rep in REPS]
        vals = [v for v in vals if not math.isnan(v)]
        return max(vals) if vals else float("nan")

    _render_heatmap(rows, targets, plots_dir, "summary_heatmap.png",
                    "Best XGBoost score per target x layer", best_over_reps)
    for rep in REPS:
        _render_heatmap(
            rows, targets, plots_dir, f"summary_heatmap_{rep}.png",
            f"XGBoost score per target x layer — {rep} embedding",
            lambda t, l, rep=rep: _metric_lookup(rows, t, rep, l, "xgb",
                                                 primary_metric_name(t)))


def best_config(rows, target):
    """Best (rep, layer) for the XGB model by primary metric."""
    pm = primary_metric_name(target)
    best, best_v = None, -np.inf
    for rep in REPS:
        for l in ALL_LAYERS:
            v = _metric_lookup(rows, target, rep, l, "xgb", pm)
            if not math.isnan(v) and v > best_v:
                best_v, best = v, (rep, l)
    return best, best_v


def plot_best_layer_table(rows, targets, plots_dir):
    lines = [f"{'target':28s} {'best_layer':12s} {'rep':7s} "
             f"{'metric':>8s} {'score':>7s} {'xgb-lin gap':>11s}"]
    lines.append("-" * 80)
    table = []
    for t in order_targets(targets):
        cfg, v = best_config(rows, t)
        if cfg is None:
            continue
        rep, layer = cfg
        pm = primary_metric_name(t)
        lin = _metric_lookup(rows, t, rep, layer, "linear", pm)
        gap = v - lin if not math.isnan(lin) else float("nan")
        lines.append(f"{t:28s} {LAYER_LABELS[layer]:12s} {rep:7s} "
                     f"{pm:>8s} {v:7.3f} {gap:11.3f}")
        table.append([t, LAYER_LABELS[layer], rep, f"{v:.3f}", f"{gap:+.3f}"])
    (plots_dir / "best_layer_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    if not table:          # no target of this task was requested
        return
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(table) + 1))
    ax.axis("off")
    tbl = ax.table(cellText=table,
                   colLabels=["target", "best layer", "rep",
                              "score (R2/F1)", "xgb-linear gap"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    ax.set_title("Best ORB layer per property (XGBoost)")
    fig.savefig(plots_dir / "best_layer_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] best-layer table -> {plots_dir/'best_layer_summary.txt'}")


def best_layer_for_rep(rows, target, rep):
    """Best layer for a fixed representation (XGB, by primary metric)."""
    pm = primary_metric_name(target)
    best, best_v = None, -np.inf
    for l in ALL_LAYERS:
        v = _metric_lookup(rows, target, rep, l, "xgb", pm)
        if not math.isnan(v) and v > best_v:
            best_v, best = v, l
    return best, best_v


def _render_pred_plots(requests, ydict, feat_dir, splits_dir, pdir, cdir,
                       seed, n_jobs, f1dir=None):
    """Refit XGB and render parity (reg) / confusion (clf); if `f1dir` is given,
    also write a per-class F1 bar chart for classification targets.

    `requests` is an iterable of (target, rep, layer); loads are grouped by
    layer so each feature cache is read once.
    """
    pdir.mkdir(parents=True, exist_ok=True); cdir.mkdir(parents=True, exist_ok=True)
    if f1dir is not None:
        f1dir.mkdir(parents=True, exist_ok=True)
    by_layer: dict[str, list] = {}
    for t, rep, layer in requests:
        if layer is not None:
            by_layer.setdefault(layer, []).append((t, rep))
    for layer, items in by_layer.items():
        d = np.load(feat_dir / f"{layer}_pooled.npz", allow_pickle=True)
        node, edge = d["node"], d["edge"]
        for t, rep in items:
            task = "clf" if is_clf(t) else "reg"
            y = ydict[t]
            sp = np.load(splits_dir / f"{safe(t)}.npz", allow_pickle=True)
            le = LabelEncoder().fit(y[valid_mask(t, y)]) if task == "clf" else None
            row, preds = fit_eval(t, task, rep, layer, "xgb", node, edge, y, le,
                                  sp["train_idx"], sp["test_idx"], seed, n_jobs)
            if preds is None:
                continue
            ytrue, ypred = preds
            if task == "reg":
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(ytrue, ypred, s=4, alpha=0.25, color=REP_COLORS[rep])
                lo = min(ytrue.min(), ypred.min()); hi = max(ytrue.max(), ypred.max())
                ax.plot([lo, hi], [lo, hi], "k--", lw=1)
                ax.set_xlabel("MP value (true)"); ax.set_ylabel("predicted")
                ax.set_title(f"{t}\nXGB / {rep} / {LAYER_LABELS[layer]} — "
                             f"R2={_to_float(row['r2']):.3f}  MAE={_to_float(row['mae']):.3g}")
                fig.savefig(pdir / f"{rep}_{safe(t)}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            else:
                labs = le.classes_
                cm = confusion_matrix(ytrue, ypred, labels=np.arange(len(labs)))
                # Row-normalize (per-true-class recall) so each class gets equal
                # visual weight regardless of class imbalance.
                rs = cm.sum(axis=1, keepdims=True)
                cm_norm = np.divide(cm, rs, out=np.zeros(cm.shape, dtype=float),
                                    where=rs != 0)
                fig, ax = plt.subplots(figsize=(5.8, 5.2))
                im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
                ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=7)
                ax.set_xlabel("predicted"); ax.set_ylabel("true")
                # Percentage in the colour scale + raw count inside each cell.
                for i in range(len(labs)):
                    for j in range(len(labs)):
                        ax.text(j, i, f"{cm_norm[i, j]*100:.0f}%\nn={cm[i, j]}",
                                ha="center", va="center", fontsize=6.5,
                                color="w" if cm_norm[i, j] > 0.5 else "k")
                ax.set_title(f"{t} (row-normalized)\nXGB / {rep} / {LAYER_LABELS[layer]} — "
                             f"acc={_to_float(row['acc']):.3f}  F1={_to_float(row['macro_f1']):.3f}")
                fig.colorbar(im, ax=ax, label="fraction of true class (recall)")
                fig.savefig(cdir / f"{rep}_{safe(t)}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

                if f1dir is not None:
                    per = f1_score(ytrue, ypred, average=None,
                                   labels=np.arange(len(labs)))
                    fig2, ax2 = plt.subplots(figsize=(max(4.5, 0.9 * len(labs)), 4))
                    ax2.bar(range(len(labs)), per, color=REP_COLORS[rep])
                    ax2.axhline(_to_float(row["macro_f1"]), color="k", ls="--", lw=1,
                                label=f"macro-F1={_to_float(row['macro_f1']):.3f}")
                    ax2.set_xticks(range(len(labs)))
                    ax2.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
                    ax2.set_ylim(0, 1); ax2.set_ylabel("F1"); ax2.grid(axis="y", alpha=0.3)
                    for i, vv in enumerate(per):
                        ax2.text(i, vv + 0.02, f"{vv:.2f}", ha="center", fontsize=7)
                    ax2.set_title(f"{t} — per-class F1\nXGB / {rep} / {LAYER_LABELS[layer]}")
                    ax2.legend(fontsize=8)
                    fig2.savefig(f1dir / f"{rep}_{safe(t)}.png", dpi=150, bbox_inches="tight")
                    plt.close(fig2)
        del node, edge, d


def plot_pred_diagnostics(rows, targets, feat_dir, out_dir, seed, n_jobs):
    """Prediction diagnostics for every embedding type, each at that rep's best
    layer. All figures land flat in three shared folders — plots/parity,
    plots/confusion, plots/per_class_f1 — named `{rep}_{property}.png`, so there
    is one folder per figure kind instead of one per (rep × kind)."""
    canon, ydict = load_canonical_targets(feat_dir, targets)
    requests = []
    for rep in REPS:
        for t in targets:
            bl = best_layer_for_rep(rows, t, rep)[0]
            if bl is not None:
                requests.append((t, rep, bl))
    _render_pred_plots(requests, ydict, feat_dir, out_dir / "splits",
                       out_dir / "plots" / "parity", out_dir / "plots" / "confusion",
                       seed, n_jobs, f1dir=out_dir / "plots" / "per_class_f1")
    print(f"  [plots] parity / confusion / per-class F1 (all reps, {{rep}}_{{prop}}) "
          f"-> {out_dir/'plots'}")


def write_metrics_summary(rows, targets, out_dir):
    lines = ["Property prediction summary (best XGBoost config per target)", ""]
    for grp, names in [("Near in-domain (energy-derived)", NEAR_IN_DOMAIN),
                       ("Never explicitly trained on", None),
                       ("Geometric control", GEOMETRIC_CONTROL)]:
        if names is None:
            sel = [t for t in targets if t not in NEAR_IN_DOMAIN
                   and t not in GEOMETRIC_CONTROL]
        else:
            sel = [t for t in targets if t in names]
        sel = order_targets(sel)
        if not sel:
            continue
        lines.append(f"== {grp} ==")
        for t in sel:
            cfg, v = best_config(rows, t)
            if cfg is None:
                continue
            rep, layer = cfg
            pm = primary_metric_name(t)
            base = "base_f1" if is_clf(t) else "base_rmse"
            bval = _metric_lookup(rows, t, rep, layer, "xgb", base)
            lines.append(f"  {t:26s} best={pm}={v:.3f} @ {LAYER_LABELS[layer]}/{rep}"
                         f"  (baseline {base}={bval:.3g})")
        lines.append("")
    (out_dir / "plots" / "metrics_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  [plots] summary -> {out_dir/'plots'/'metrics_summary.txt'}")


def write_full_metrics_table(rows, targets, out_dir):
    """Complete per-(target,rep,layer,model) metrics for the record: a tidy CSV
    plus a formatted text table (too dense to be a figure, kept for reference)."""
    ordered = order_targets(targets)
    model_order = ["xgb", "linear", "rf", "mlp"]

    def sort_key(r):
        return (ordered.index(r["target"]) if r["target"] in ordered else 99,
                REPS.index(r["rep"]) if r["rep"] in REPS else 99,
                ALL_LAYERS.index(r["layer"]) if r["layer"] in ALL_LAYERS else 99,
                model_order.index(r["model"]) if r["model"] in model_order else 99)

    srows = sorted(rows, key=sort_key)
    reg = [r for r in srows if r["task"] == "reg"]
    clf = [r for r in srows if r["task"] == "clf"]

    def f(x, p=4):
        v = _to_float(x)
        return "" if math.isnan(v) else f"{v:.{p}f}"

    # Baseline improvement per row, computed purely from already-stored columns
    # (no refit needed): regression = RMSE reduction vs the predict-mean
    # baseline, classification = macro-F1 gain over the majority-class baseline.
    for r in reg:
        rmse, base = _to_float(r["rmse"]), _to_float(r["base_rmse"])
        r["baseline_improvement"] = ("" if math.isnan(rmse) or math.isnan(base) or base == 0
                                     else f"{1 - rmse / base:.4f}")
    for r in clf:
        f1, base = _to_float(r["macro_f1"]), _to_float(r["base_f1"])
        r["baseline_improvement"] = ("" if math.isnan(f1) or math.isnan(base)
                                     else f"{f1 - base:.4f}")

    L = ["FULL METRICS TABLE — ORB embedding property prediction",
         "All targets x 7 layers x reps x models. Test = held-out 20% (seed 42).",
         "baseline_improvement = 1 - RMSE/baseRMSE (regression) or macro-F1 - baseF1 (classification).",
         "Recorded for completeness (too dense to be a figure).", "",
         "=" * 112,
         "REGRESSION  (metric of record: R2; baseRMSE = predict-mean baseline)",
         "=" * 112,
         f"{'target':24s} {'rep':6s} {'layer':9s} {'model':7s} {'n_test':>7s} "
         f"{'MAE':>10s} {'RMSE':>10s} {'R2':>9s} {'baseRMSE':>10s} {'baseImpr':>10s}",
         "-" * 112]
    for r in reg:
        L.append(f"{r['target']:24s} {r['rep']:6s} {LAYER_LABELS.get(r['layer'], r['layer']):9s} "
                 f"{r['model']:7s} {r['n_test']:>7s} {f(r['mae']):>10s} {f(r['rmse']):>10s} "
                 f"{f(r['r2']):>9s} {f(r['base_rmse']):>10s} {r['baseline_improvement']:>10s}")
    L += ["", "=" * 94,
          "CLASSIFICATION  (metric of record: macro-F1; baseF1 = majority-class baseline)",
          "=" * 94,
          f"{'target':18s} {'rep':6s} {'layer':9s} {'model':7s} {'n_test':>7s} "
          f"{'acc':>8s} {'macroF1':>9s} {'baseF1':>8s} {'baseImpr':>10s}",
          "-" * 94]
    for r in clf:
        L.append(f"{r['target']:18s} {r['rep']:6s} {LAYER_LABELS.get(r['layer'], r['layer']):9s} "
                 f"{r['model']:7s} {r['n_test']:>7s} {f(r['acc']):>8s} {f(r['macro_f1']):>9s} "
                 f"{f(r['base_f1']):>8s} {r['baseline_improvement']:>10s}")
    (out_dir / "plots" / "full_metrics_table.txt").write_text("\n".join(L), encoding="utf-8")

    cols = ["task", "target", "rep", "layer", "model", "n_train", "n_test",
            "mae", "rmse", "r2", "acc", "macro_f1", "base_rmse", "base_f1",
            "baseline_improvement"]
    with (out_dir / "plots" / "full_metrics_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(srows)
    print(f"  [plots] full metrics table ({len(reg)} reg + {len(clf)} clf rows) "
          f"-> {out_dir/'plots'/'full_metrics_table.txt'}")


def _label_unit(t: str) -> str:
    u = UNITS.get(t, "")
    return f"{t} ({u})" if u else t


def _best_config_preds(rows, targets, feat_dir, splits_dir, seed, n_jobs):
    """Refit XGB once per target at its single overall-best (rep, layer) config
    and return {target: (task, ytrue, ypred)}. Cheap (one refit per target) —
    reused to compute the parity/confusion single-number summaries (CCC /
    kappa) without rerunning the full training grid."""
    canon, ydict = load_canonical_targets(feat_dir, targets)
    by_layer: dict[str, list] = {}
    for t in targets:
        cfg, _ = best_config(rows, t)
        if cfg is None:
            continue
        rep, layer = cfg
        by_layer.setdefault(layer, []).append((t, rep))
    out = {}
    for layer, items in by_layer.items():
        d = np.load(feat_dir / f"{layer}_pooled.npz", allow_pickle=True)
        node, edge = d["node"], d["edge"]
        for t, rep in items:
            task = "clf" if is_clf(t) else "reg"
            y = ydict[t]
            sp = np.load(splits_dir / f"{safe(t)}.npz", allow_pickle=True)
            le = LabelEncoder().fit(y[valid_mask(t, y)]) if task == "clf" else None
            _row, preds = fit_eval(t, task, rep, layer, "xgb", node, edge, y, le,
                                   sp["train_idx"], sp["test_idx"], seed, n_jobs)
            if preds is not None:
                out[t] = (task, preds[0], preds[1])
        del node, edge, d
    return out


def plot_regression_error_table(rows, targets, plots_dir, preds=None):
    """Best-config (best-over-reps XGB) physical error sizes per regression target.
    If `preds` (from _best_config_preds) is given, also reports CCC — Lin's
    concordance correlation coefficient, the single-number parity-plot summary
    (agreement with y=x, distinct from R2 which ignores bias/scale shifts)."""
    reg = [t for t in order_targets(targets) if not is_clf(t)]
    # Width the first column to the longest target label so names never clip.
    tgt_w = max(len("target (unit)"), max((len(_label_unit(t)) for t in reg), default=0))
    header = (f"{'target (unit)':{tgt_w}s} {'best_layer':11s} {'rep':7s} "
              f"{'R2':>8s} {'MAE':>11s} {'RMSE':>11s} {'CCC':>8s}")
    lines = [header, "-" * len(header)]
    table = []
    for t in reg:
        cfg, r2 = best_config(rows, t)
        if cfg is None:
            continue
        rep, layer = cfg
        mae = _metric_lookup(rows, t, rep, layer, "xgb", "mae")
        rmse = _metric_lookup(rows, t, rep, layer, "xgb", "rmse")
        ccc = float("nan")
        if preds and t in preds:
            _task, yt, yp = preds[t]
            ccc = _ccc(yt, yp)
        ccc_s = "n/a" if math.isnan(ccc) else f"{ccc:.3f}"
        lines.append(f"{_label_unit(t):{tgt_w}s} {LAYER_LABELS[layer]:11s} {rep:7s} "
                     f"{r2:8.3f} {mae:11.4g} {rmse:11.4g} {ccc_s:>8s}")
        table.append([_label_unit(t), LAYER_LABELS[layer], rep,
                      f"{r2:.3f}", f"{mae:.4g}", f"{rmse:.4g}", ccc_s])
    (plots_dir / "regression_error_table.txt").write_text("\n".join(lines), encoding="utf-8")
    # Give the "target (unit)" column ~5x the width of a numeric column and
    # left-align it so long physical-unit labels fit without clipping.
    col_w = [0.40, 0.12, 0.08, 0.09, 0.10, 0.10, 0.09]
    if not table:          # no target of this task was requested
        return
    fig, ax = plt.subplots(figsize=(13, 0.45 * len(table) + 1.2)); ax.axis("off")
    tbl = ax.table(cellText=table,
                   colLabels=["target (unit)", "best layer", "rep", "R2", "MAE", "RMSE", "CCC"],
                   colWidths=col_w, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    for (_r, _c), _cell in tbl.get_celld().items():
        if _c == 0:
            _cell.set_text_props(ha="left")
            _cell.PAD = 0.02
    ax.set_title("Regression error sizes — best XGBoost config (physical units)\n"
                 "CCC = Lin's concordance correlation coefficient (parity-plot agreement with y=x)")
    fig.savefig(plots_dir / "regression_error_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] regression error table -> {plots_dir/'regression_error_table.txt'}")


def plot_classification_error_table(rows, targets, plots_dir, preds=None):
    """Classification analog of plot_regression_error_table: best-config
    accuracy / macro-F1 + Cohen's kappa — the single-number confusion-matrix
    summary (chance-corrected agreement; distinct from accuracy/macro-F1,
    which don't correct for the chance level set by class imbalance)."""
    clf = [t for t in order_targets(targets) if is_clf(t)]
    tgt_w = max(len("target"), max((len(t) for t in clf), default=0))
    header = (f"{'target':{tgt_w}s} {'best_layer':11s} {'rep':7s} "
              f"{'acc':>8s} {'macroF1':>9s} {'kappa':>8s}")
    lines = [header, "-" * len(header)]
    table = []
    for t in clf:
        cfg, f1 = best_config(rows, t)
        if cfg is None:
            continue
        rep, layer = cfg
        acc = _metric_lookup(rows, t, rep, layer, "xgb", "acc")
        kappa = float("nan")
        if preds and t in preds:
            _task, yt, yp = preds[t]
            kappa = float(cohen_kappa_score(yt, yp))
        kappa_s = "n/a" if math.isnan(kappa) else f"{kappa:.3f}"
        lines.append(f"{t:{tgt_w}s} {LAYER_LABELS[layer]:11s} {rep:7s} "
                     f"{acc:8.3f} {f1:9.3f} {kappa_s:>8s}")
        table.append([t, LAYER_LABELS[layer], rep, f"{acc:.3f}", f"{f1:.3f}", kappa_s])
    (plots_dir / "classification_error_table.txt").write_text("\n".join(lines), encoding="utf-8")
    col_w = [0.32, 0.16, 0.11, 0.13, 0.14, 0.13]
    if not table:          # no target of this task was requested
        return
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(table) + 1.2)); ax.axis("off")
    tbl = ax.table(cellText=table,
                   colLabels=["target", "best layer", "rep", "accuracy", "macro-F1", "kappa"],
                   colWidths=col_w, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    for (_r, _c), _cell in tbl.get_celld().items():
        if _c == 0:
            _cell.set_text_props(ha="left")
            _cell.PAD = 0.02
    ax.set_title("Classification accuracy — best XGBoost config\n"
                 "kappa = Cohen's kappa (confusion-matrix agreement, chance-corrected)")
    fig.savefig(plots_dir / "classification_error_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] classification error table -> {plots_dir/'classification_error_table.txt'}")


def plot_baseline_improvement(rows, targets, plots_dir):
    """How much better than the trivial baseline: RMSE reduction (regression) and
    macro-F1 gain over majority class (classification), for the best config."""
    reg = [t for t in order_targets(targets) if not is_clf(t)]
    clf = [t for t in order_targets(targets) if is_clf(t)]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(3.0, 0.5 * max(len(reg), len(clf)) + 1.5)))

    ax = axes[0]; vals, labs = [], []
    for t in reg:
        cfg, _ = best_config(rows, t)
        if cfg is None:
            continue
        rep, layer = cfg
        rmse = _metric_lookup(rows, t, rep, layer, "xgb", "rmse")
        base = _metric_lookup(rows, t, rep, layer, "xgb", "base_rmse")
        red = 1 - rmse / base if base and not math.isnan(base) else float("nan")
        vals.append(red); labs.append(t)
    yy = np.arange(len(labs))
    disp = [0 if math.isnan(v) else max(-1.0, min(1.0, v)) for v in vals]
    ax.barh(yy, disp, color=["#2ca02c" if (not math.isnan(v) and v >= 0) else "#d62728" for v in vals])
    ax.set_yticks(yy); ax.set_yticklabels(labs, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(-1.08, 1.12); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("RMSE reduction  =  1 - RMSE / baseRMSE")
    ax.set_title("Regression: improvement over predict-mean baseline\n"
                 "(0 = no better than mean, 1 = perfect, <0 = worse)")
    for i, v in enumerate(vals):
        ax.text(disp[i] + (0.02 if disp[i] >= 0 else -0.02), i,
                "n/a" if math.isnan(v) else f"{v:.2f}", va="center",
                ha="left" if disp[i] >= 0 else "right", fontsize=7)

    ax = axes[1]; vals, labs = [], []
    for t in clf:
        cfg, _ = best_config(rows, t)
        if cfg is None:
            continue
        rep, layer = cfg
        f1 = _metric_lookup(rows, t, rep, layer, "xgb", "macro_f1")
        base = _metric_lookup(rows, t, rep, layer, "xgb", "base_f1")
        vals.append(f1 - base if not math.isnan(base) else float("nan")); labs.append(t)
    yy = np.arange(len(labs))
    ax.barh(yy, [0 if math.isnan(v) else v for v in vals], color="#1f77b4")
    ax.set_yticks(yy); ax.set_yticklabels(labs, fontsize=8); ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("macro-F1 gain over majority class")
    ax.set_title("Classification: improvement over majority baseline")
    for i, v in enumerate(vals):
        ax.text((0 if math.isnan(v) else v) + 0.005, i,
                "n/a" if math.isnan(v) else f"{v:.2f}", va="center", fontsize=7)
    fig.savefig(plots_dir / "baseline_improvement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] baseline improvement -> {plots_dir/'baseline_improvement.png'}")


def plot_xgb_linear_gap(rows, targets, plots_dir):
    """XGB - linear (primary metric, best config): large gap = information present
    but nonlinearly encoded; small gap = linearly accessible / or absent."""
    ordered = order_targets(targets)
    vals, labs, colors = [], [], []
    for t in ordered:
        cfg, v = best_config(rows, t)
        if cfg is None:
            continue
        rep, layer = cfg
        lin = _metric_lookup(rows, t, rep, layer, "linear", primary_metric_name(t))
        vals.append(v - lin if not math.isnan(lin) else float("nan"))
        labs.append(t)
        colors.append("#9467bd" if is_clf(t) else "#ff7f0e")
    yy = np.arange(len(labs))
    disp = [0 if math.isnan(v) else max(-0.5, min(0.7, v)) for v in vals]
    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(labs) + 1.5))
    ax.barh(yy, disp, color=colors)
    ax.set_yticks(yy); ax.set_yticklabels(labs, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(-0.55, 0.78); ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("XGBoost - linear  (R2 for reg / macro-F1 for clf, best config)")
    ax.set_title("Nonlinear accessibility of information in ORB embeddings\n"
                 "(orange = regression, purple = classification; values clipped for display)")
    for i, v in enumerate(vals):
        ax.text(disp[i] + (0.01 if disp[i] >= 0 else -0.01), i,
                "n/a" if math.isnan(v) else f"{v:+.2f}", va="center",
                ha="left" if disp[i] >= 0 else "right", fontsize=7)
    fig.savefig(plots_dir / "xgb_linear_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] xgb-linear gap -> {plots_dir/'xgb_linear_gap.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_targets(arg: str | None) -> list[str]:
    if not arg:
        return REG_PROPS + CAT_PROPS
    out = []
    for tok in arg.split(","):
        tok = tok.strip()
        tok = TARGET_ALIASES.get(tok, tok)
        if tok not in REG_PROPS and tok not in CAT_PROPS:
            raise SystemExit(f"unknown target '{tok}'. "
                             f"choices: {REG_PROPS + CAT_PROPS} (+ aliases {list(TARGET_ALIASES)})")
        out.append(tok)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict MP properties from frozen ORB v3 embeddings.")
    parser.add_argument("--embeddings-dir", default=str(P.RESULTS / "orb"))
    parser.add_argument("--structures-dir", default=str(P.STRUCTURES))
    parser.add_argument("--output-dir", default=str(P.RESULTS / "orb/property_prediction"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", default=None,
                        help="comma list; default = the nine structure-level targets")
    parser.add_argument("--extra-models", action="store_true",
                        help="also train RandomForest + MLP")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap number of materials (local testing only)")
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)
    str_dir = Path(args.structures_dir)
    out_dir = Path(args.output_dir)
    feat_dir = out_dir / "features"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    jsonl = str_dir / "index.jsonl"
    if not jsonl.exists():
        raise FileNotFoundError(jsonl)

    targets = resolve_targets(args.targets)
    models = ["xgb", "linear"] + (["rf", "mlp"] if args.extra_models else [])

    print("=== Load manifest ===")
    structs, _ = qc.load_manifest(jsonl)
    mp_ids = list(structs.keys())
    if args.limit:
        mp_ids = mp_ids[:args.limit]
        structs = {m: structs[m] for m in mp_ids}
    print(f"  {len(mp_ids):,} materials; targets={targets}; models={models}")

    print("\n=== Stage 1: feature cache ===")
    build_feature_cache(emb_dir, feat_dir, mp_ids, args.num_workers)

    print("\n=== Stage 2: target table ===")
    build_target_table(structs, feat_dir, mp_ids)

    print("\n=== Stage 3: train + evaluate ===")
    train_all(feat_dir, out_dir, targets, models,
              args.test_size, args.seed, args.num_workers)

    print("\n=== Stage 4: plots ===")
    rows = load_metrics(out_dir / "metrics.csv")
    # Main comparison
    plot_layer_curves(rows, targets, models, plots_dir)
    plot_summary_heatmap(rows, targets, plots_dir)
    plot_best_layer_table(rows, targets, plots_dir)
    # Diagnostics
    best_preds = _best_config_preds(rows, targets, feat_dir, out_dir / "splits",
                                    args.seed, args.num_workers)
    plot_regression_error_table(rows, targets, plots_dir, preds=best_preds)
    plot_classification_error_table(rows, targets, plots_dir, preds=best_preds)
    plot_baseline_improvement(rows, targets, plots_dir)
    plot_xgb_linear_gap(rows, targets, plots_dir)
    plot_pred_diagnostics(rows, targets, feat_dir, out_dir,
                          args.seed, args.num_workers)
    write_metrics_summary(rows, targets, out_dir)
    write_full_metrics_table(rows, targets, out_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
