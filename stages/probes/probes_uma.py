#!/usr/bin/env python3
"""
stages/probes/probes_uma.py — Downstream material-property prediction
from frozen UMA (FairChem / eSCNMD) embeddings.

Central question
----------------
UMA is a *conservative interatomic potential* — trained on energy/forces. Can
its frozen embeddings predict Materials Project properties it was never
explicitly trained on, to what accuracy, and which embedding family × layer is
best? UMA's raw per-atom tensor is equivariant [N, 9, 128] (SO(3) channels
L=0 scalar, L=1 vector, L=2 tensor), so — unlike ORB's single pooled vector —
the analysis pipeline decomposes it into many pooled *families*. This lets us
contrast composition-only (l0), geometry-only (geom_norm) and full information,
and node vs edge vs their concatenation.

Feature families (structure-level, mean-pooled), mirroring
stages/geometry/organisation_uma.py:
  node  (6 layers: embed, block_0..3, backbone):
        full[1152], ang_norm[384], l0[128], l1_polar[128], l2_polar[128], geom_norm[256]
  edge  (5 layers: embed, block_0..3 — no backbone):
        flat[1152], ang_norm[384], l0[128], l1_norm[128], l2_norm[128], geom_norm[256]
  concat (5 shared layers, node_X ⊕ edge_X matched by L-channel):
        full(=node_full⊕edge_flat)[2304], ang_norm[768], l0[256],
        l1(=node_l1_polar⊕edge_l1_norm)[256], l2(=node_l2_polar⊕edge_l2_norm)[256],
        geom_norm[512]
Grid = 36 node + 30 edge + 30 concat = 96 feature sets.

For each feature set we train an XGBoost model and a linear baseline (Ridge for
regression, Logistic for classification). A fixed per-target 80/20 split
(seed 42) is defined once on the canonical material list and reused across every
feature set, so comparisons are apples-to-apples.

Targets mirror the k-NN purity + LVR analyses (structure-level only):
  - Regression (CONT_PROPS): formation_energy/atom, energy_above_hull, band_gap,
    density, bulk_modulus, mag_density.
  - Classification (CAT_PROPS): crystal_system, metal_insulator, magnetic_label.

Features are NOT rebuilt from the .pt files. They are read straight from the
full-size pooled caches that stages/geometry/organisation_uma.py already wrote:
  $MLIP_PROBES_ROOT/results/fairchem/plots/{nodes,edges}/{layer}_{nodes|edges}_pooled_{family}_umap_cache.npz
  (key `X_raw` [~154,875, dim] float32 + aligned `mp_ids`).

Pipeline (idempotent — every stage guarded by existence checks):
  1. Target table  : features/targets.npz (mp_ids + all targets)
  2. Splits + train: splits/{target}.npz, metrics.csv
  3. Plots         : 3-panel layer curves, summary + per-group heatmaps,
                     best-config table, regression error table, baseline
                     improvement, xgb-linear gap, parity/confusion/per-class F1,
                     metrics_summary.txt, full metrics table.

Reads  $MLIP_PROBES_ROOT/results/fairchem/plots/{nodes,edges}/*_umap_cache.npz
       $MLIP_PROBES_ROOT/structures/index.jsonl
Writes $MLIP_PROBES_ROOT/results/fairchem/property_prediction/

Examples
--------
  # smoke test (own output dir so it never clobbers a full-run cache):
  uv run python scripts/stages/probes/probes_uma.py \
      --limit 3000 --targets band_gap,crystal_system \
      --groups node,edge,concat --families l0,ang_norm \
      --output-dir $MLIP_PROBES_ROOT/results/fairchem/property_prediction_smoke

  # full run (idempotent, resumable) — run inside tmux:
  uv run python scripts/stages/probes/probes_uma.py --n-jobs 8
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
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["legend.frameon"] = False  # no box border around figure legends
import matplotlib.pyplot as plt
import numpy as np

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

CACHE_ROOT = (P.RESULTS / "fairchem/plots")

# Data-flow order (input -> output), matching stages/geometry/organisation_uma.py.
NODE_LAYERS = ["embed", "block_0", "block_1", "block_2", "block_3", "backbone"]
EDGE_LAYERS = ["embed", "block_0", "block_1", "block_2", "block_3"]  # no backbone
LAYER_LABELS = {"embed": "embed", "block_0": "b0", "block_1": "b1",
                "block_2": "b2", "block_3": "b3", "backbone": "backbone"}
# Column set for the combined heatmap (union of all groups' layers).
ALL_LAYERS = NODE_LAYERS

GROUPS = ["node", "edge", "concat"]
GROUP_LAYERS = {"node": NODE_LAYERS, "edge": EDGE_LAYERS, "concat": EDGE_LAYERS}
GROUP_FAMILIES = {
    "node":   ["full", "ang_norm", "l0", "l1_polar", "l2_polar", "geom_norm"],
    "edge":   ["flat", "ang_norm", "l0", "l1_norm", "l2_norm", "geom_norm"],
    "concat": ["full", "ang_norm", "l0", "l1", "l2", "geom_norm"],
}
# concat family -> (node family, edge family), matched by L-channel.
CONCAT_MAP = {
    "full":      ("full", "flat"),
    "ang_norm":  ("ang_norm", "ang_norm"),
    "l0":        ("l0", "l0"),
    "l1":        ("l1_polar", "l1_norm"),
    "l2":        ("l2_polar", "l2_norm"),
    "geom_norm": ("geom_norm", "geom_norm"),
}

# Rotation-invariant channel families (scalars / component norms). The raw tensor views
# (full, flat) and the sign-carrying L=1 views rotate with the input frame.
INVARIANT_FAMILIES = {"ang_norm", "geom_norm", "l0", "l2", "l2_polar", "l2_norm"}
# Targets for which we restrict the *reported* config to rotation-invariant channels, because
# the full-tensor peak reads the conventional CIF orientation rather than intrinsic symmetry
# (crystal system; see the rotation control in 06 --rotation and paper §3.5). Restricting the
# selection here propagates ang_norm as UMA's crystal-system config through every downstream
# table (best_config / best_within_group feed the significance pipeline, controls, and figures).
INVARIANT_ONLY_TARGETS = {"crystal_system"}


def _family_allowed(target, fam):
    return target not in INVARIANT_ONLY_TARGETS or fam in INVARIANT_FAMILIES

# Canonical L-family key (for colouring + cross-group comparison).
FAMILY_CANON = {
    "full": "full", "flat": "full", "ang_norm": "ang_norm", "l0": "l0",
    "l1_norm": "l1", "l1_polar": "l1", "l1": "l1",
    "l2_norm": "l2", "l2_polar": "l2", "l2": "l2",
    "geom_norm": "geom_norm",
}
CANON_ORDER  = ["full", "ang_norm", "l0", "l1", "l2", "geom_norm"]
CANON_COLORS = {"full": "#1f77b4", "ang_norm": "#ff7f0e", "l0": "#2ca02c",
                "l1": "#d62728", "l2": "#9467bd", "geom_norm": "#8c564b"}

# Regression / classification targets (model-independent, from mlipprobes.common).
REG_PROPS = list(qc.CONT_PROPS)
CAT_PROPS = list(qc.CAT_PROPS)
SUMMARY_ORDER = CAT_PROPS + REG_PROPS

UNITS = {
    "formation_energy/atom": "eV/atom", "energy_above_hull": "eV/atom",
    "band_gap": "eV", "density": "g/cm^3", "bulk_modulus": "GPa",
    "mag_density": "uB/A^3",
}
NEAR_IN_DOMAIN    = {"formation_energy/atom", "energy_above_hull"}
GEOMETRIC_CONTROL = {"density"}

# Distinguish models by colour intensity: xgb = strong, linear = lighter.
MODEL_LIGHTEN = {"xgb": 0.0, "linear": 0.55}
MODEL_LW      = {"xgb": 2.3, "linear": 1.4}

TARGET_ALIASES = {
    "formation_energy_per_atom": "formation_energy/atom",
    "formation_energy":          "formation_energy/atom",
}

CSV_FIELDS = [
    "target", "task", "group", "family", "layer", "model", "n_train", "n_test",
    "mae", "rmse", "r2", "acc", "macro_f1",
    "base_mae", "base_rmse", "base_acc", "base_f1",
]


def _lighten(color: str, amount: float):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def safe(name: str) -> str:
    return name.replace("/", "_per_")


def is_clf(target: str) -> bool:
    return target in CAT_PROPS


def primary_metric_name(target: str) -> str:
    return "macro_f1" if is_clf(target) else "r2"


def order_targets(targets) -> list[str]:
    s = set(targets)
    return [t for t in SUMMARY_ORDER if t in s]


def feature_sets(groups, families=None):
    """Yield (group, family, layer) for the requested groups (+ optional family
    filter, matched against the canonical L-family key)."""
    fam_filter = None if not families else {FAMILY_CANON.get(f, f) for f in families}
    for g in groups:
        for fam in GROUP_FAMILIES[g]:
            if fam_filter is not None and FAMILY_CANON[fam] not in fam_filter:
                continue
            for layer in GROUP_LAYERS[g]:
                yield g, fam, layer


# ---------------------------------------------------------------------------
# Feature loading — straight from organisation_uma.py's pooled umap caches (no .pt reads)
# ---------------------------------------------------------------------------

def _cache_path(group: str, family: str, layer: str) -> Path:
    sub = "nodes" if group == "node" else "edges"
    return CACHE_ROOT / sub / f"{layer}_{sub}_pooled_{family}_umap_cache.npz"


def _load_one(group: str, family: str, layer: str):
    p = _cache_path(group, family, layer)
    if not p.exists():
        raise SystemExit(
            f"missing pooled cache: {p}\n"
            f"Run stages/geometry/organisation_uma.py first (it writes the "
            f"{{layer}}_{{nodes|edges}}_pooled_{{family}}_umap_cache.npz files).")
    d = np.load(p, allow_pickle=True)
    return d["X_raw"].astype(np.float32, copy=False), d["mp_ids"]


def load_feature_set(group: str, family: str, layer: str):
    """Return (X [n, dim] float32, mp_ids). concat = node_X ⊕ edge_X aligned on
    the mp_id intersection (node order preserved)."""
    if group in ("node", "edge"):
        return _load_one(group, family, layer)
    nfam, efam = CONCAT_MAP[family]
    Xn, idn = _load_one("node", nfam, layer)
    Xe, ide = _load_one("edge", efam, layer)
    pose = {m: i for i, m in enumerate(ide)}
    common = [m for m in idn if m in pose]
    posn = {m: i for i, m in enumerate(idn)}
    rn = np.fromiter((posn[m] for m in common), dtype=np.int64, count=len(common))
    re = np.fromiter((pose[m] for m in common), dtype=np.int64, count=len(common))
    X = np.hstack([Xn[rn], Xe[re]])
    return X.astype(np.float32, copy=False), np.array(common, dtype=object)


# ---------------------------------------------------------------------------
# Stage 1 — target table
# ---------------------------------------------------------------------------

def build_target_table(structs: dict, feat_dir: Path, mp_ids: list[str]) -> np.ndarray:
    feat_dir.mkdir(parents=True, exist_ok=True)
    out_p = feat_dir / "targets.npz"
    mp_arr = np.array(mp_ids, dtype=object)
    if out_p.exists():
        existing = np.load(out_p, allow_pickle=True)["mp_ids"]
        if existing.shape == mp_arr.shape and np.array_equal(existing, mp_arr):
            print(f"  [targets] up to date ({len(mp_ids):,} mats), skip")
            return mp_arr
        print("  [targets] stale -> rebuild")
    data: dict[str, np.ndarray] = {"mp_ids": mp_arr}
    for p in REG_PROPS:
        data[f"reg__{p}"] = qc.cont_array(structs, mp_arr, p)
    for p in CAT_PROPS:
        data[f"cat__{p}"] = qc.cat_array(structs, mp_arr, p)
    np.savez(out_p, **data)
    print(f"  [targets] wrote {len(REG_PROPS)} regression + {len(CAT_PROPS)} "
          f"classification columns -> {out_p.name}")
    return mp_arr


def load_targets(feat_dir: Path, targets: list[str]):
    """Return (canon_mp_ids, {target -> value array aligned to canon})."""
    td = np.load(feat_dir / "targets.npz", allow_pickle=True)
    canon = td["mp_ids"]
    out = {}
    for t in targets:
        key = f"cat__{t}" if is_clf(t) else f"reg__{t}"
        out[t] = td[key]
    return canon, out


# ---------------------------------------------------------------------------
# Stage 2 — splits + train/eval
# ---------------------------------------------------------------------------

def valid_mask(target: str, y: np.ndarray) -> np.ndarray:
    if is_clf(target):
        return np.array([v is not None for v in y])
    return np.isfinite(y.astype(float))


def get_split(target, y, mp_ids, test_size, seed, splits_dir):
    """Fixed per-target split on the canonical valid set; stored as mp_id lists
    so it is reused identically across every feature set."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    sp = splits_dir / f"{safe(target)}.npz"
    if sp.exists():
        d = np.load(sp, allow_pickle=True)
        return d["train_mp_ids"], d["test_mp_ids"]
    valid = np.where(valid_mask(target, y))[0]
    strat = y[valid] if is_clf(target) else None
    tr, te = train_test_split(valid, test_size=test_size,
                              random_state=seed, stratify=strat)
    np.savez(sp, train_mp_ids=mp_ids[tr], test_mp_ids=mp_ids[te])
    return mp_ids[tr], mp_ids[te]


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
    raise ValueError(f"unknown model {model}")


def _align(X, feat_pos, canon_pos, y, split_ids):
    """Select feature rows for the split mp_ids present in this feature set,
    aligned with their target values; drop non-finite feature rows."""
    ids = [m for m in split_ids if m in feat_pos]
    if not ids:
        return np.empty((0, X.shape[1])), np.empty((0,), dtype=y.dtype)
    rx = np.fromiter((feat_pos[m] for m in ids), dtype=np.int64, count=len(ids))
    ry = np.fromiter((canon_pos[m] for m in ids), dtype=np.int64, count=len(ids))
    Xs, ys = X[rx], y[ry]
    finite = np.isfinite(Xs).all(axis=1)
    return Xs[finite], ys[finite]


def fit_eval(target, task, group, family, layer, X, feat_pos, canon_pos,
             y, le, train_ids, test_ids, seed, n_jobs, model):
    """Train one (target, group, family, layer, model); return metrics row + preds."""
    Xtr, ytr_raw = _align(X, feat_pos, canon_pos, y, train_ids)
    Xte, yte_raw = _align(X, feat_pos, canon_pos, y, test_ids)
    if len(Xtr) < 10 or len(Xte) < 5:
        return None, None
    row = {f: "" for f in CSV_FIELDS}
    row.update(target=target, task=task, group=group, family=family,
               layer=layer, model=model, n_train=len(Xtr), n_test=len(Xte))

    if task == "reg":
        ytr, yte = ytr_raw.astype(float), yte_raw.astype(float)
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

    ytr, yte = le.transform(ytr_raw), le.transform(yte_raw)
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
                done.add((r["target"], r["group"], r["family"], r["layer"], r["model"]))
    return done


def append_row(metrics_csv: Path, row: dict) -> None:
    new = not metrics_csv.exists()
    with metrics_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def train_all(feat_dir, out_dir, targets, groups, families, models,
              test_size, seed, n_jobs):
    splits_dir = out_dir / "splits"
    metrics_csv = out_dir / "metrics.csv"
    canon, ydict = load_targets(feat_dir, targets)
    canon_pos = {m: i for i, m in enumerate(canon)}

    splits, encoders = {}, {}
    for t in targets:
        splits[t] = get_split(t, ydict[t], canon, test_size, seed, splits_dir)
        if is_clf(t):
            v = valid_mask(t, ydict[t])
            encoders[t] = LabelEncoder().fit(ydict[t][v])

    fsets = list(feature_sets(groups, families))
    done = read_done_keys(metrics_csv)
    total = len(fsets) * len(targets) * len(models)
    print(f"  {total} combos ({len(fsets)} feature sets x {len(targets)} targets "
          f"x {len(models)} models); {len(done)} already done")

    # Feature-set-outer loop -> each cache loaded once (memory efficient).
    for gi, (group, family, layer) in enumerate(fsets, 1):
        # Skip whole feature set if every combo already done.
        pending = [(t, m) for t in targets for m in models
                   if (t, group, family, layer, m) not in done]
        if not pending:
            continue
        X, feat_ids = load_feature_set(group, family, layer)
        feat_pos = {m: i for i, m in enumerate(feat_ids)}
        tag = f"{group}/{family}/{LAYER_LABELS[layer]}"
        print(f"  [{gi}/{len(fsets)}] {tag}: X{X.shape}")
        for t in targets:
            task = "clf" if is_clf(t) else "reg"
            tr_ids, te_ids = splits[t]
            le = encoders.get(t)
            for model in models:
                if (t, group, family, layer, model) in done:
                    continue
                t0 = time.time()
                row, _ = fit_eval(t, task, group, family, layer, X, feat_pos,
                                  canon_pos, ydict[t], le, tr_ids, te_ids,
                                  seed, n_jobs, model)
                if row is None:
                    print(f"      skip {t}/{model} (too few rows)")
                    continue
                append_row(metrics_csv, row)
                pm = primary_metric_name(t)
                print(f"      {t}/{model}: {pm}={float(row[pm]):.3f}  "
                      f"({time.time()-t0:.1f}s)")
        del X, feat_pos
    print(f"  metrics -> {metrics_csv}")


# ---------------------------------------------------------------------------
# Stage 3 — plots
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


def _lookup(rows, target, group, family, layer, model, metric):
    for r in rows:
        if (r["target"] == target and r["group"] == group
                and r["family"] == family and r["layer"] == layer
                and r["model"] == model):
            return _to_float(r[metric])
    return float("nan")


def plot_layer_curves(rows, targets, models, plots_dir):
    """Per target: 3 panels (node/edge/concat), primary metric vs layer; one line
    per family (colour = L-family), model by colour intensity."""
    d = plots_dir / "layer_curves"
    d.mkdir(parents=True, exist_ok=True)
    for t in targets:
        pm = primary_metric_name(t)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
        for ax, group in zip(axes, GROUPS):
            layers = GROUP_LAYERS[group]
            xs = np.arange(len(layers))
            for family in GROUP_FAMILIES[group]:
                canon = FAMILY_CANON[family]
                for model in models:
                    ys = [_lookup(rows, t, group, family, l, model, pm) for l in layers]
                    ax.plot(xs, ys, "-",
                            color=_lighten(CANON_COLORS[canon], MODEL_LIGHTEN[model]),
                            lw=MODEL_LW[model], marker="o", ms=3)
            ax.set_xticks(xs)
            ax.set_xticklabels([LAYER_LABELS[l] for l in layers], rotation=30, ha="right")
            ax.set_title(group); ax.set_xlabel("UMA layer")
            ax.set_ylim(top=1.0); ax.grid(alpha=0.3)
        axes[0].set_ylabel(pm.upper())
        # Shared legend: family colours + model intensity.
        import matplotlib.lines as ml
        import matplotlib.patches as mp
        fam_handles = [mp.Patch(color=CANON_COLORS[c], label=c) for c in CANON_ORDER]
        mdl_handles = [ml.Line2D([0], [0], color="0.25", lw=MODEL_LW["xgb"], label="XGBoost"),
                       ml.Line2D([0], [0], color="0.62", lw=MODEL_LW["linear"], label="linear")]
        fig.legend(handles=fam_handles + mdl_handles, ncol=8, fontsize=8,
                   loc="lower center", bbox_to_anchor=(0.5, -0.20))
        fig.suptitle(f"{t} — {pm.upper()} across UMA layers "
                     f"(colour = family, darker = XGBoost)")
        fig.subplots_adjust(bottom=0.24)
        fig.savefig(d / f"{safe(t)}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  [plots] layer curves -> {d}")


def _render_heatmap(rows, targets, plots_dir, out_name, title, layers, value_fn):
    ordered = order_targets(targets)
    M = np.full((len(ordered), len(layers)), np.nan)
    for i, t in enumerate(ordered):
        for j, l in enumerate(layers):
            M[i, j] = value_fn(t, l)
    fig, ax = plt.subplots(figsize=(1.0 * len(layers) + 2, 0.5 * len(ordered) + 2))
    im = ax.imshow(M, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([LAYER_LABELS[l] for l in layers], rotation=30, ha="right")
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels([f"{t}*" if t in NEAR_IN_DOMAIN else
                        (f"{t} (ctrl)" if t in GEOMETRIC_CONTROL else t)
                        for t in ordered])
    for i in range(len(ordered)):
        for j in range(len(layers)):
            if not math.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="w" if M[i, j] < 0.6 else "k", fontsize=7)
    ax.set_title(f"{title}\n(* = near in-domain / energy-derived; ctrl = geometric)")
    fig.colorbar(im, ax=ax, label="R2 (reg) / macro-F1 (clf)")
    fig.savefig(plots_dir / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] heatmap -> {plots_dir / out_name}")


def plot_summary_heatmap(rows, targets, plots_dir):
    """Best-over-all-96 heatmap + one 'best family' heatmap per group."""
    def best_all(t, l):
        vals = []
        for g in GROUPS:
            if l not in GROUP_LAYERS[g]:
                continue
            for fam in GROUP_FAMILIES[g]:
                v = _lookup(rows, t, g, fam, l, "xgb", primary_metric_name(t))
                if not math.isnan(v):
                    vals.append(v)
        return max(vals) if vals else float("nan")

    _render_heatmap(rows, targets, plots_dir, "summary_heatmap.png",
                    "Best XGBoost score per target x layer (over all 96 feature sets)",
                    ALL_LAYERS, best_all)

    for g in GROUPS:
        def best_fam(t, l, g=g):
            vals = [_lookup(rows, t, g, fam, l, "xgb", primary_metric_name(t))
                    for fam in GROUP_FAMILIES[g]]
            vals = [v for v in vals if not math.isnan(v)]
            return max(vals) if vals else float("nan")
        _render_heatmap(rows, targets, plots_dir, f"summary_heatmap_{g}.png",
                        f"Best XGBoost score per target x layer — {g} (over its families)",
                        GROUP_LAYERS[g], best_fam)


def best_config(rows, target):
    """Best (group, family, layer) for XGB by primary metric. For INVARIANT_ONLY_TARGETS
    (crystal system) the search is restricted to rotation-invariant channels (§3.5)."""
    pm = primary_metric_name(target)
    best, best_v = None, -np.inf
    for g in GROUPS:
        for fam in GROUP_FAMILIES[g]:
            if not _family_allowed(target, fam):
                continue
            for l in GROUP_LAYERS[g]:
                v = _lookup(rows, target, g, fam, l, "xgb", pm)
                if not math.isnan(v) and v > best_v:
                    best_v, best = v, (g, fam, l)
    return best, best_v


def best_within_group(rows, target, group):
    pm = primary_metric_name(target)
    best, best_v = None, -np.inf
    for fam in GROUP_FAMILIES[group]:
        if not _family_allowed(target, fam):
            continue
        for l in GROUP_LAYERS[group]:
            v = _lookup(rows, target, group, fam, l, "xgb", pm)
            if not math.isnan(v) and v > best_v:
                best_v, best = v, (fam, l)
    return best, best_v


def plot_best_config_table(rows, targets, plots_dir):
    lines = [f"{'target':28s} {'group':7s} {'family':10s} {'layer':9s} "
             f"{'metric':>8s} {'score':>7s} {'xgb-lin':>8s}"]
    lines.append("-" * 84)
    table = []
    for t in order_targets(targets):
        cfg, v = best_config(rows, t)
        if cfg is None:
            continue
        g, fam, layer = cfg
        pm = primary_metric_name(t)
        lin = _lookup(rows, t, g, fam, layer, "linear", pm)
        gap = v - lin if not math.isnan(lin) else float("nan")
        lines.append(f"{t:28s} {g:7s} {fam:10s} {LAYER_LABELS[layer]:9s} "
                     f"{pm:>8s} {v:7.3f} {gap:8.3f}")
        table.append([t, g, fam, LAYER_LABELS[layer], f"{v:.3f}", f"{gap:+.3f}"])
    (plots_dir / "best_config_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    if not table:          # no target of this task was requested
        return
    fig, ax = plt.subplots(figsize=(10, 0.4 * len(table) + 1))
    ax.axis("off")
    tbl = ax.table(cellText=table,
                   colLabels=["target", "group", "family", "best layer",
                              "score (R2/F1)", "xgb-linear gap"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    ax.set_title("Best UMA embedding (group/family/layer) per property (XGBoost)")
    fig.savefig(plots_dir / "best_config_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] best-config table -> {plots_dir/'best_config_summary.txt'}")


def _label_unit(t: str) -> str:
    u = UNITS.get(t, "")
    return f"{t} ({u})" if u else t


def _best_config_preds(rows, targets, feat_dir, splits_dir, seed, n_jobs):
    """Refit XGB once per target at its single overall-best (group, family,
    layer) config and return {target: (task, ytrue, ypred)}. Cheap (one refit
    per target) — reused to compute the parity/confusion single-number
    summaries (CCC / kappa) without rerunning the full 96-feature-set grid."""
    canon, ydict = load_targets(feat_dir, targets)
    canon_pos = {m: i for i, m in enumerate(canon)}
    by_fs: dict[tuple, list] = {}
    for t in targets:
        cfg, _ = best_config(rows, t)
        if cfg is None:
            continue
        g, fam, layer = cfg
        by_fs.setdefault((g, fam, layer), []).append(t)
    out = {}
    for (g, fam, layer), ts in by_fs.items():
        X, feat_ids = load_feature_set(g, fam, layer)
        feat_pos = {m: i for i, m in enumerate(feat_ids)}
        for t in ts:
            task = "clf" if is_clf(t) else "reg"
            y = ydict[t]
            sp = np.load(splits_dir / f"{safe(t)}.npz", allow_pickle=True)
            le = LabelEncoder().fit(y[valid_mask(t, y)]) if task == "clf" else None
            _row, preds = fit_eval(t, task, g, fam, layer, X, feat_pos, canon_pos,
                                   y, le, sp["train_mp_ids"], sp["test_mp_ids"],
                                   seed, n_jobs, "xgb")
            if preds is not None:
                out[t] = (task, preds[0], preds[1])
        del X, feat_pos
    return out


def plot_regression_error_table(rows, targets, plots_dir, preds=None):
    """If `preds` (from _best_config_preds) is given, also reports CCC — Lin's
    concordance correlation coefficient, the single-number parity-plot summary
    (agreement with y=x, distinct from R2 which ignores bias/scale shifts)."""
    reg = [t for t in order_targets(targets) if not is_clf(t)]
    tgt_w = max(len("target (unit)"), max((len(_label_unit(t)) for t in reg), default=0))
    header = (f"{'target (unit)':{tgt_w}s} {'group':7s} {'family':10s} {'layer':9s} "
              f"{'R2':>8s} {'MAE':>11s} {'RMSE':>11s} {'CCC':>8s}")
    lines = [header, "-" * len(header)]
    table = []
    for t in reg:
        cfg, r2 = best_config(rows, t)
        if cfg is None:
            continue
        g, fam, layer = cfg
        mae = _lookup(rows, t, g, fam, layer, "xgb", "mae")
        rmse = _lookup(rows, t, g, fam, layer, "xgb", "rmse")
        ccc = float("nan")
        if preds and t in preds:
            _task, yt, yp = preds[t]
            ccc = _ccc(yt, yp)
        ccc_s = "n/a" if math.isnan(ccc) else f"{ccc:.3f}"
        lines.append(f"{_label_unit(t):{tgt_w}s} {g:7s} {fam:10s} {LAYER_LABELS[layer]:9s} "
                     f"{r2:8.3f} {mae:11.4g} {rmse:11.4g} {ccc_s:>8s}")
        table.append([_label_unit(t), g, fam, LAYER_LABELS[layer],
                      f"{r2:.3f}", f"{mae:.4g}", f"{rmse:.4g}", ccc_s])
    (plots_dir / "regression_error_table.txt").write_text("\n".join(lines), encoding="utf-8")
    col_w = [0.34, 0.09, 0.12, 0.09, 0.08, 0.075, 0.075, 0.08]
    if not table:          # no target of this task was requested
        return
    fig, ax = plt.subplots(figsize=(14, 0.45 * len(table) + 1.2)); ax.axis("off")
    tbl = ax.table(cellText=table,
                   colLabels=["target (unit)", "group", "family", "best layer",
                              "R2", "MAE", "RMSE", "CCC"],
                   colWidths=col_w, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    for (_r, _c), _cell in tbl.get_celld().items():
        if _c == 0:
            _cell.set_text_props(ha="left"); _cell.PAD = 0.02
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
    header = (f"{'target':{tgt_w}s} {'group':7s} {'family':10s} {'layer':9s} "
              f"{'acc':>8s} {'macroF1':>9s} {'kappa':>8s}")
    lines = [header, "-" * len(header)]
    table = []
    for t in clf:
        cfg, f1 = best_config(rows, t)
        if cfg is None:
            continue
        g, fam, layer = cfg
        acc = _lookup(rows, t, g, fam, layer, "xgb", "acc")
        kappa = float("nan")
        if preds and t in preds:
            _task, yt, yp = preds[t]
            kappa = float(cohen_kappa_score(yt, yp))
        kappa_s = "n/a" if math.isnan(kappa) else f"{kappa:.3f}"
        lines.append(f"{t:{tgt_w}s} {g:7s} {fam:10s} {LAYER_LABELS[layer]:9s} "
                     f"{acc:8.3f} {f1:9.3f} {kappa_s:>8s}")
        table.append([t, g, fam, LAYER_LABELS[layer], f"{acc:.3f}", f"{f1:.3f}", kappa_s])
    (plots_dir / "classification_error_table.txt").write_text("\n".join(lines), encoding="utf-8")
    col_w = [0.26, 0.11, 0.15, 0.11, 0.11, 0.12, 0.11]
    if not table:          # no target of this task was requested
        return
    fig, ax = plt.subplots(figsize=(11.5, 0.45 * len(table) + 1.2)); ax.axis("off")
    tbl = ax.table(cellText=table,
                   colLabels=["target", "group", "family", "best layer",
                              "accuracy", "macro-F1", "kappa"],
                   colWidths=col_w, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.4)
    for (_r, _c), _cell in tbl.get_celld().items():
        if _c == 0:
            _cell.set_text_props(ha="left"); _cell.PAD = 0.02
    ax.set_title("Classification accuracy — best XGBoost config\n"
                 "kappa = Cohen's kappa (confusion-matrix agreement, chance-corrected)")
    fig.savefig(plots_dir / "classification_error_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] classification error table -> {plots_dir/'classification_error_table.txt'}")


def plot_baseline_improvement(rows, targets, plots_dir):
    reg = [t for t in order_targets(targets) if not is_clf(t)]
    clf = [t for t in order_targets(targets) if is_clf(t)]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(3.0, 0.5 * max(len(reg), len(clf)) + 1.5)))

    ax = axes[0]; vals, labs = [], []
    for t in reg:
        cfg, _ = best_config(rows, t)
        if cfg is None:
            continue
        g, fam, layer = cfg
        rmse = _lookup(rows, t, g, fam, layer, "xgb", "rmse")
        base = _lookup(rows, t, g, fam, layer, "xgb", "base_rmse")
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
        g, fam, layer = cfg
        f1 = _lookup(rows, t, g, fam, layer, "xgb", "macro_f1")
        base = _lookup(rows, t, g, fam, layer, "xgb", "base_f1")
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
    ordered = order_targets(targets)
    vals, labs, colors = [], [], []
    for t in ordered:
        cfg, v = best_config(rows, t)
        if cfg is None:
            continue
        g, fam, layer = cfg
        lin = _lookup(rows, t, g, fam, layer, "linear", primary_metric_name(t))
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
    ax.set_title("Nonlinear accessibility of information in UMA embeddings\n"
                 "(orange = regression, purple = classification; values clipped for display)")
    for i, v in enumerate(vals):
        ax.text(disp[i] + (0.01 if disp[i] >= 0 else -0.01), i,
                "n/a" if math.isnan(v) else f"{v:+.2f}", va="center",
                ha="left" if disp[i] >= 0 else "right", fontsize=7)
    fig.savefig(plots_dir / "xgb_linear_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plots] xgb-linear gap -> {plots_dir/'xgb_linear_gap.png'}")


def _render_pred_plots(requests, ydict, canon_pos, splits_dir, pdir, cdir,
                       seed, n_jobs, f1dir=None):
    """Refit XGB for each (target, group, family, layer) and render parity (reg) /
    row-normalised confusion (clf) + per-class F1. Loads are grouped by feature
    set so each cache is read once."""
    pdir.mkdir(parents=True, exist_ok=True); cdir.mkdir(parents=True, exist_ok=True)
    if f1dir is not None:
        f1dir.mkdir(parents=True, exist_ok=True)
    by_fs: dict[tuple, list] = {}
    for t, g, fam, layer in requests:
        by_fs.setdefault((g, fam, layer), []).append(t)
    for (g, fam, layer), ts in by_fs.items():
        X, feat_ids = load_feature_set(g, fam, layer)
        feat_pos = {m: i for i, m in enumerate(feat_ids)}
        for t in ts:
            task = "clf" if is_clf(t) else "reg"
            y = ydict[t]
            sp = np.load(splits_dir / f"{safe(t)}.npz", allow_pickle=True)
            le = LabelEncoder().fit(y[valid_mask(t, y)]) if task == "clf" else None
            row, preds = fit_eval(t, task, g, fam, layer, X, feat_pos, canon_pos,
                                  y, le, sp["train_mp_ids"], sp["test_mp_ids"],
                                  seed, n_jobs, "xgb")
            if preds is None:
                continue
            ytrue, ypred = preds
            col = CANON_COLORS[FAMILY_CANON[fam]]
            tag = f"{g}/{fam}/{LAYER_LABELS[layer]}"
            fname = f"{g}_{fam}_{safe(t)}.png"
            if task == "reg":
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.scatter(ytrue, ypred, s=4, alpha=0.25, color=col)
                lo = min(ytrue.min(), ypred.min()); hi = max(ytrue.max(), ypred.max())
                ax.plot([lo, hi], [lo, hi], "k--", lw=1)
                ax.set_xlabel("MP value (true)"); ax.set_ylabel("predicted")
                ax.set_title(f"{t}\nXGB / {tag} — "
                             f"R2={_to_float(row['r2']):.3f}  MAE={_to_float(row['mae']):.3g}")
                fig.savefig(pdir / fname, dpi=150, bbox_inches="tight")
                plt.close(fig)
            else:
                labs = le.classes_
                cm = confusion_matrix(ytrue, ypred, labels=np.arange(len(labs)))
                rs = cm.sum(axis=1, keepdims=True)
                cm_norm = np.divide(cm, rs, out=np.zeros(cm.shape, dtype=float),
                                    where=rs != 0)
                fig, ax = plt.subplots(figsize=(5.8, 5.2))
                im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
                ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=7)
                ax.set_xlabel("predicted"); ax.set_ylabel("true")
                for i in range(len(labs)):
                    for j in range(len(labs)):
                        ax.text(j, i, f"{cm_norm[i, j]*100:.0f}%\nn={cm[i, j]}",
                                ha="center", va="center", fontsize=6.5,
                                color="w" if cm_norm[i, j] > 0.5 else "k")
                ax.set_title(f"{t} (row-normalized)\nXGB / {tag} — "
                             f"acc={_to_float(row['acc']):.3f}  F1={_to_float(row['macro_f1']):.3f}")
                fig.colorbar(im, ax=ax, label="fraction of true class (recall)")
                fig.savefig(cdir / fname, dpi=150, bbox_inches="tight")
                plt.close(fig)

                if f1dir is not None:
                    per = f1_score(ytrue, ypred, average=None, labels=np.arange(len(labs)))
                    fig2, ax2 = plt.subplots(figsize=(max(4.5, 0.9 * len(labs)), 4))
                    ax2.bar(range(len(labs)), per, color=col)
                    ax2.axhline(_to_float(row["macro_f1"]), color="k", ls="--", lw=1,
                                label=f"macro-F1={_to_float(row['macro_f1']):.3f}")
                    ax2.set_xticks(range(len(labs)))
                    ax2.set_xticklabels(labs, rotation=45, ha="right", fontsize=8)
                    ax2.set_ylim(0, 1); ax2.set_ylabel("F1"); ax2.grid(axis="y", alpha=0.3)
                    for i, vv in enumerate(per):
                        ax2.text(i, vv + 0.02, f"{vv:.2f}", ha="center", fontsize=7)
                    ax2.set_title(f"{t} — per-class F1\nXGB / {tag}")
                    ax2.legend(fontsize=8)
                    fig2.savefig(f1dir / fname, dpi=150, bbox_inches="tight")
                    plt.close(fig2)
        del X, feat_pos


def plot_pred_diagnostics(rows, targets, feat_dir, out_dir, seed, n_jobs):
    """Parity / confusion / per-class F1 for the best (family, layer) within each
    group (node/edge/concat) — 3 per target, named {group}_{family}_{prop}.png."""
    canon, ydict = load_targets(feat_dir, targets)
    canon_pos = {m: i for i, m in enumerate(canon)}
    requests = []
    for g in GROUPS:
        for t in targets:
            cfg, _ = best_within_group(rows, t, g)
            if cfg is not None:
                fam, layer = cfg
                requests.append((t, g, fam, layer))
    _render_pred_plots(requests, ydict, canon_pos, out_dir / "splits",
                       out_dir / "plots" / "parity", out_dir / "plots" / "confusion",
                       seed, n_jobs, f1dir=out_dir / "plots" / "per_class_f1")
    print(f"  [plots] parity / confusion / per-class F1 (best per group) "
          f"-> {out_dir/'plots'}")


def write_metrics_summary(rows, targets, out_dir):
    lines = ["UMA property prediction summary (best XGBoost config per target)", ""]
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
            g, fam, layer = cfg
            pm = primary_metric_name(t)
            base = "base_f1" if is_clf(t) else "base_rmse"
            bval = _lookup(rows, t, g, fam, layer, "xgb", base)
            lines.append(f"  {t:26s} best={pm}={v:.3f} @ {g}/{fam}/{LAYER_LABELS[layer]}"
                         f"  (baseline {base}={bval:.3g})")
        lines.append("")
    (out_dir / "plots" / "metrics_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  [plots] summary -> {out_dir/'plots'/'metrics_summary.txt'}")


def write_full_metrics_table(rows, targets, out_dir):
    ordered = order_targets(targets)

    def sort_key(r):
        g = r["group"]
        return (ordered.index(r["target"]) if r["target"] in ordered else 99,
                GROUPS.index(g) if g in GROUPS else 99,
                GROUP_FAMILIES.get(g, []).index(r["family"])
                if r["family"] in GROUP_FAMILIES.get(g, []) else 99,
                GROUP_LAYERS.get(g, []).index(r["layer"])
                if r["layer"] in GROUP_LAYERS.get(g, []) else 99,
                0 if r["model"] == "xgb" else 1)

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

    L = ["FULL METRICS TABLE — UMA embedding property prediction",
         "All targets x groups x families x layers x models. Test = held-out 20% (seed 42).",
         "baseline_improvement = 1 - RMSE/baseRMSE (regression) or macro-F1 - baseF1 (classification).",
         "Recorded for completeness (too dense to be a figure).", "",
         "=" * 120,
         "REGRESSION  (metric of record: R2; baseRMSE = predict-mean baseline)",
         "=" * 120,
         f"{'target':22s} {'group':7s} {'family':10s} {'layer':9s} {'model':7s} "
         f"{'n_test':>7s} {'MAE':>10s} {'RMSE':>10s} {'R2':>9s} {'baseRMSE':>10s} {'baseImpr':>10s}",
         "-" * 120]
    for r in reg:
        L.append(f"{r['target']:22s} {r['group']:7s} {r['family']:10s} "
                 f"{LAYER_LABELS.get(r['layer'], r['layer']):9s} {r['model']:7s} "
                 f"{r['n_test']:>7s} {f(r['mae']):>10s} {f(r['rmse']):>10s} "
                 f"{f(r['r2']):>9s} {f(r['base_rmse']):>10s} {r['baseline_improvement']:>10s}")
    L += ["", "=" * 106,
          "CLASSIFICATION  (metric of record: macro-F1; baseF1 = majority-class baseline)",
          "=" * 106,
          f"{'target':18s} {'group':7s} {'family':10s} {'layer':9s} {'model':7s} "
          f"{'n_test':>7s} {'acc':>8s} {'macroF1':>9s} {'baseF1':>8s} {'baseImpr':>10s}",
          "-" * 106]
    for r in clf:
        L.append(f"{r['target']:18s} {r['group']:7s} {r['family']:10s} "
                 f"{LAYER_LABELS.get(r['layer'], r['layer']):9s} {r['model']:7s} "
                 f"{r['n_test']:>7s} {f(r['acc']):>8s} {f(r['macro_f1']):>9s} "
                 f"{f(r['base_f1']):>8s} {r['baseline_improvement']:>10s}")
    (out_dir / "plots" / "full_metrics_table.txt").write_text("\n".join(L), encoding="utf-8")

    cols = ["task", "target", "group", "family", "layer", "model", "n_train",
            "n_test", "mae", "rmse", "r2", "acc", "macro_f1", "base_rmse", "base_f1",
            "baseline_improvement"]
    with (out_dir / "plots" / "full_metrics_table.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(srows)
    print(f"  [plots] full metrics table ({len(reg)} reg + {len(clf)} clf rows) "
          f"-> {out_dir/'plots'/'full_metrics_table.txt'}")


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


def resolve_groups(arg: str | None) -> list[str]:
    if not arg:
        return list(GROUPS)
    out = []
    for tok in arg.split(","):
        tok = tok.strip()
        if tok not in GROUPS:
            raise SystemExit(f"unknown group '{tok}'. choices: {GROUPS}")
        out.append(tok)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict MP properties from frozen UMA (FairChem) embeddings.")
    parser.add_argument("--cache-root", default=str(P.RESULTS / "fairchem/plots"),
                        help="dir holding organisation_uma.py's {nodes,edges}/*_umap_cache.npz")
    parser.add_argument("--structures-dir", default=str(P.STRUCTURES))
    parser.add_argument("--output-dir", default=str(P.RESULTS / "fairchem/property_prediction"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", default=None,
                        help="comma list; default = the nine structure-level targets")
    parser.add_argument("--groups", default=None,
                        help="comma list of node,edge,concat; default = all")
    parser.add_argument("--families", default=None,
                        help="comma list of L-family keys (full,ang_norm,l0,l1,l2,geom_norm); default = all")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap number of materials (local testing only)")
    args = parser.parse_args()

    global CACHE_ROOT
    CACHE_ROOT = Path(args.cache_root)

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
    groups = resolve_groups(args.groups)
    families = [s.strip() for s in args.families.split(",")] if args.families else None
    models = ["xgb", "linear"]

    print("=== Load manifest ===")
    structs, _ = qc.load_manifest(jsonl)
    mp_ids = list(structs.keys())
    if args.limit:
        mp_ids = mp_ids[:args.limit]
        structs = {m: structs[m] for m in mp_ids}
    print(f"  {len(mp_ids):,} materials; targets={targets}; groups={groups}; "
          f"families={families or 'all'}")

    print("\n=== Stage 1: target table ===")
    build_target_table(structs, feat_dir, mp_ids)

    print("\n=== Stage 2: train + evaluate ===")
    train_all(feat_dir, out_dir, targets, groups, families, models,
              args.test_size, args.seed, args.n_jobs)

    print("\n=== Stage 3: plots ===")
    rows = load_metrics(out_dir / "metrics.csv")
    plot_layer_curves(rows, targets, models, plots_dir)
    plot_summary_heatmap(rows, targets, plots_dir)
    plot_best_config_table(rows, targets, plots_dir)
    best_preds = _best_config_preds(rows, targets, feat_dir, out_dir / "splits",
                                    args.seed, args.n_jobs)
    plot_regression_error_table(rows, targets, plots_dir, preds=best_preds)
    plot_classification_error_table(rows, targets, plots_dir, preds=best_preds)
    plot_baseline_improvement(rows, targets, plots_dir)
    plot_xgb_linear_gap(rows, targets, plots_dir)
    plot_pred_diagnostics(rows, targets, feat_dir, out_dir, args.seed, args.n_jobs)
    write_metrics_summary(rows, targets, out_dir)
    write_full_metrics_table(rows, targets, out_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
