#!/usr/bin/env python3
"""
Label efficiency — how much labelled data does each approach need?

Learning curves for three feature sets on the paper's 154,875 Materials Project crystals and
all nine targets:

  orb     ORB-v3 pooled node embedding, backbone layer (256-d)
  uma     UMA-S pooled node embedding, ang_norm family, backbone layer
  magpie  132 Magpie composition statistics (composition only, no structure)

Probes are XGBoost and a linear model with the paper's hyperparameters. Scores are R^2 for
regression and macro-F1 for classification.

Design (fixed; the seeds and grids are constants below)
  * Rows are in the order of the materials table. Every arm is reindexed to it and must cover
    it completely, so for a given target all three arms draw the SAME frozen test set and the
    SAME training materials: the embedding-minus-Magpie difference is a paired quantity at
    every point of the curve.
  * Frozen test set per target: a seeded permutation of the rows with a label; the first
    FROZEN_TEST_N rows (or half the labelled rows for sparse targets) are the test set, the
    rest the training pool.
  * Training axis: disjoint draws from the pool at each size of TRAIN_GRID; 30 draws up to
    2,000, 10 at 5,000, 7 above, fewer where the pool cannot supply that many disjoint draws.
  * Test axis: one probe fitted on the first min(20,000, pool) training rows; its predictions
    on the frozen test set are re-scored on 200 subsamples at each size of TEST_GRID.
  * Summary per point: median, p10, p90, spread (p10-p90 width estimated as 2.5631 x sd,
    the normal-quantile factor, which is unbiased at few draws where the direct quantile
    range runs narrow), gap to the full-data ceiling (the paper's own score for the same
    probe and feature set).
  * Thresholds per curve: n_stable = smallest n from which the spread stays <= 0.05;
    n_saturated = smallest n from which the gap stays <= 0.05.

Passes (each resumable; `curves` appends per (target, probe, size) and skips what exists)
  curves --arm orb|uma|magpie --axis train|test
  summarise            summary_{arm}.csv, thresholds.csv, curves_train.png, the thresholds table
  gain                 paired_gain.csv  (embedding minus Magpie, XGBoost, paired draws)
  split-fraction       split_fraction.csv (shortfall and test-size noise in quadrature, per
                       dataset size x split fraction)
  gnn                  the from-scratch GNN at each target's saturation size (pipeline only)

Inputs: the paper's public parquet release, --from-dataset DIR with DIR/data/materials.parquet,
DIR/embeddings/node/orb-node-gnn_stacks_4.parquet (ORB's backbone read-out),
DIR/reproduction/magpie_features.parquet, DIR/results/probe_metrics.parquet (the ORB and UMA
ceilings), DIR/results/controls/composition_baseline.parquet (the Magpie ceilings; if it is absent the
Magpie arm's ceiling falls back to its own score at the largest size), and the
UMA arm from the companion gated release (--uma-dataset DIR2, default DIR/uma or the sibling
uma_gated): DIR2/embeddings/node/uma-node-ang_norm-backbone.parquet.

Usage
    <script> curves --arm magpie --axis train --from-dataset DIR --out-dir results
    <script> curves --arm orb --axis test --from-dataset DIR --out-dir results
    <script> summarise --from-dataset DIR --out-dir results
    <script> gain --out-dir results
    <script> split-fraction --out-dir results
    <script> curves --arm orb --axis train --from-dataset DIR --dry-run
"""
# [pipeline-only]
# Inside the paper pipeline this script also reads its own caches when --from-dataset is absent:
#   $MLIP_PROBES_ROOT/results/orb/property_prediction/features/{gnn_stacks_4_pooled.npz,targets.npz}
#   $MLIP_PROBES_ROOT/results/fairchem/plots/nodes/backbone_pooled_ang_norm_umap_cache.npz  (via 05)
#   $MLIP_PROBES_ROOT/results/exp_magpie/magpie_features.npz
#   $MLIP_PROBES_ROOT/results/{orb,fairchem}/property_prediction/metrics.csv          (ceilings)
#   $MLIP_PROBES_ROOT/results/proppred_controls/composition_baseline/composition_baseline.csv
# and writes $MLIP_PROBES_ROOT/results/label_efficiency/. `summarise` regenerates
# config/label_efficiency_thresholds.csv, which the `gnn` pass (mlipprobes/gnn.py, the structures)
# reads. Usage: uv run python stages/downstream/label_efficiency.py <pass> ...
# [/pipeline-only]
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path

import argparse
import csv
import importlib
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import f1_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
# [pipeline-only]
OUT_DEFAULT = (P.RESULTS / "label_efficiency")
THRESHOLDS_CSV = HERE.parents[2] / "config" / "label_efficiency_thresholds.csv"
# [/pipeline-only]
#public: OUT_DEFAULT = Path("results")
#public: THRESHOLDS_CSV = Path("results") / "thresholds_config.csv"

TARGETS = ["formation_energy/atom", "energy_above_hull", "band_gap", "density", "bulk_modulus",
           "mag_density", "crystal_system", "metal_insulator", "magnetic_label"]
CLF = {"crystal_system", "metal_insulator", "magnetic_label"}
ARMS = ["orb", "uma", "magpie"]
METHODS = ["xgboost", "linear"]

DRAW_SEED = 7            # frozen test sets and training draws
PROBE_SEED = 42          # estimator randomness, as in the paper
FROZEN_TEST_N = 20_000
TRAIN_GRID = [50, 100, 200, 500, 1000, 2000, 5000, 7000, 10000, 20000]
TEST_GRID = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
TEST_RESAMPLES = 200
TEST_AXIS_TRAIN_N = 20_000
P90_P10_OVER_SD = 2.5631  # p90 - p10 of a normal distribution, in units of sd
MAX_SPREAD = 0.05
MAX_GAP = 0.05
MIN_PAIRS = 3             # a paired difference needs a spread; single-draw sizes are not reported
GNN_DRAWS = 3
SPLIT_SIZES = [200, 500, 1000, 2000, 5000, 10000, 20000]
SPLIT_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5]

# Dataset-release column names for the targets whose spelling differs.
DATASET_NAMES = {"formation_energy/atom": "formation_energy_per_atom"}


def safe(target: str) -> str:
    return target.replace("/", "_per_")


def task_of(target: str) -> str:
    return "clf" if target in CLF else "reg"


def repeats(n: int) -> int:
    if n <= 2000:
        return 30
    return 10 if n <= 5000 else 7


# ----------------------------------------------------------------------------- inputs
class Inputs:
    """Canonical mp_ids, one arm's feature matrix in that order, the nine target arrays in
    that order, and the full-data ceilings {(target, method): score} for the arm."""

    def __init__(self, mp_ids, X, targets, ceilings, arm):
        self.mp_ids, self.X, self.targets, self.ceilings, self.arm = mp_ids, X, targets, ceilings, arm
        if X is not None and len(X) != len(mp_ids):
            raise SystemExit(f"{arm}: {len(X)} feature rows for {len(mp_ids)} canonical ids")


def _reindex(X, ids, canon, arm):
    ids = np.asarray(ids).astype(str)
    if len(ids) == len(canon) and (ids == canon).all():
        return X
    pos = {m: i for i, m in enumerate(ids)}
    missing = [m for m in canon if m not in pos]
    if missing:
        raise SystemExit(f"{arm}: {len(missing)} canonical materials absent (e.g. {missing[:3]}); "
                         "a subset would change every draw, so the arm is refused")
    idx = np.fromiter((pos[m] for m in canon), dtype=np.int64, count=len(canon))
    print(f"  {arm}: rows reindexed to the canonical order")
    return X[idx]


def _clean_targets(raw: dict) -> dict:
    out = {}
    for t, y in raw.items():
        if task_of(t) == "reg":
            out[t] = np.asarray(y, dtype=float)
        else:
            out[t] = np.array([None if v is None or v == "" or (isinstance(v, float) and np.isnan(v))
                               else str(v) for v in y], dtype=object)
    return out


# [pipeline-only]
def load_pipeline_inputs(arm, orb_dir, uma_dir, magpie_npz, controls_dir, need_features=True):
    O = importlib.import_module("stages.probes.probes_orb")
    feat_dir = orb_dir / "features"
    canon, ydict = O.load_canonical_targets(feat_dir, TARGETS)
    canon = np.asarray(canon).astype(str)
    targets = _clean_targets(ydict)
    # The public release omits a few materials; the experiment is defined on the released corpus
    # so that the draws are identical whether the inputs come from the caches or the release.
    excluded = importlib.import_module("mlipprobes.common").excluded_materials()
    if excluded:
        keep = np.array([m not in excluded for m in canon])
        canon = canon[keep]
        targets = {t: y[keep] for t, y in targets.items()}
        print(f"  {len(excluded)} materials excluded per config/excluded_materials.csv; corpus {len(canon):,}")
    X = None
    if need_features:
        if arm == "orb":
            # ORB's backbone output is byte-identical to gnn_stacks_4; the probe stage
            # writes only the latter.
            d = np.load(feat_dir / "gnn_stacks_4_pooled.npz", allow_pickle=True)
            X = _reindex(d["node"].astype(np.float32), d["mp_ids"], canon, arm)
        elif arm == "uma":
            U = importlib.import_module("stages.probes.probes_uma")
            Xu, idu = U.load_feature_set("node", "ang_norm", "backbone")
            X = _reindex(Xu.astype(np.float32), idu, canon, arm)
        else:
            d = np.load(magpie_npz, allow_pickle=True)
            X = _reindex(d["X"].astype(np.float32), d["mp_ids"], canon, arm)
    return Inputs(canon, X, targets, _pipeline_ceilings(arm, orb_dir, uma_dir, controls_dir), arm)


def _pipeline_ceilings(arm, orb_dir, uma_dir, controls_dir):
    ceil = {}
    est = {"xgboost": "xgb", "linear": "linear"}
    if arm in ("orb", "uma"):
        path = (orb_dir if arm == "orb" else uma_dir) / "metrics.csv"
        for r in csv.DictReader(open(path)):
            ok = (r["layer"] == ("gnn_stacks_4" if arm == "orb" else "backbone")
                  and r["model"] in est.values()
                  and (r.get("rep") == "node" if arm == "orb"
                       else (r.get("group") == "node" and r.get("family") == "ang_norm")))
            if not ok or r["target"] not in TARGETS:
                continue
            v = r["macro_f1"] if task_of(r["target"]) == "clf" else r["r2"]
            if v not in ("", None):
                ceil[(r["target"], {"xgb": "xgboost", "linear": "linear"}[r["model"]])] = float(v)
    else:
        path = controls_dir / "composition_baseline" / "composition_baseline.csv"
        if path.exists():
            for r in csv.DictReader(open(path)):
                if r["target"] in TARGETS:
                    for m, col in (("xgboost", "comp_xgb"), ("linear", "comp_linear")):
                        if r.get(col) not in ("", None):
                            ceil[(r["target"], m)] = float(r[col])
    return ceil
# [/pipeline-only]


def uma_release_dir(dataset_dir, uma_dir=None) -> Path:
    if uma_dir:
        return Path(uma_dir)
    for cand in (dataset_dir / "uma", dataset_dir.parent / "uma_gated"):
        if cand.exists():
            return cand
    raise SystemExit("the UMA arm needs the companion gated release: pass --uma-dataset DIR")


def load_dataset_inputs(arm, dataset_dir, uma_dir=None, need_features=True):
    """The same inputs from the public parquet release (pyarrow required)."""
    import pandas as pd
    mat = pd.read_parquet(dataset_dir / "data" / "materials.parquet")
    canon = mat["mp_id"].astype(str).to_numpy()
    targets = _clean_targets({t: mat[DATASET_NAMES.get(t, t)].to_numpy() for t in TARGETS})
    X = None
    if need_features:
        if arm == "orb":
            # the release stores ORB's backbone read-out once, as gnn_stacks_4 (bit-identical)
            e = pd.read_parquet(dataset_dir / "embeddings" / "node" / "orb-node-gnn_stacks_4.parquet")
            if "level" in e.columns:
                e = e[e["level"] == "node"]
            X = _reindex(np.stack(e["emb"].to_numpy()).astype(np.float32), e["mp_id"], canon, arm)
        elif arm == "uma":
            e = pd.read_parquet(uma_release_dir(dataset_dir, uma_dir) / "embeddings" / "node" / "uma-node-ang_norm-backbone.parquet")
            if "level" in e.columns:
                e = e[e["level"] == "node"]
            X = _reindex(np.stack(e["emb"].to_numpy()).astype(np.float32), e["mp_id"], canon, arm)
        else:
            e = pd.read_parquet(dataset_dir / "reproduction" / "magpie_features.parquet")
            X = _reindex(np.stack(e["features"].to_numpy()).astype(np.float32), e["mp_id"], canon, arm)
    return Inputs(canon, X, targets, _dataset_ceilings(arm, dataset_dir), arm)


def _dataset_ceilings(arm, dataset_dir):
    import pandas as pd
    ceil = {}
    if arm in ("orb", "uma"):
        pm = pd.read_parquet(dataset_dir / "results" / "probe_metrics.parquet")
        sel = (pm["model"].str.upper() == arm.upper()) & (pm["read_out"] == "node") & (pm["layer"] == "backbone")
        if arm == "uma" and "family" in pm.columns:
            sel &= pm["family"] == "ang_norm"
        for _, r in pm[sel].iterrows():
            t = r["target"]
            if t not in TARGETS:
                continue
            m = {"xgb": "xgboost", "linear": "linear"}.get(r["estimator"])
            v = pd.to_numeric(r["macro_f1"] if task_of(t) == "clf" else r["r2"], errors="coerce")
            if m and v == v:
                ceil[(t, m)] = float(v)
    else:
        p = dataset_dir / "results" / "controls" / "composition_baseline.parquet"
        if p.exists():
            cb = pd.read_parquet(p)
            for _, r in cb.iterrows():
                if r["target"] in TARGETS:
                    for m, col in (("xgboost", "comp_xgb"), ("linear", "comp_linear")):
                        v = pd.to_numeric(r[col], errors="coerce") if col in cb.columns else float("nan")
                        if v == v:
                            ceil[(r["target"], m)] = float(v)
    return ceil


def load_inputs(args, arm, need_features=True):
    if args.from_dataset:
        return load_dataset_inputs(arm, Path(args.from_dataset), getattr(args, "uma_dataset", None), need_features)
    if "load_pipeline_inputs" not in globals():
        raise SystemExit("this copy reads the public release only: pass --from-dataset DIR")
    return load_pipeline_inputs(arm, Path(args.orb_dir), Path(args.uma_dir), Path(args.magpie_npz),
                                Path(args.controls_dir), need_features)


# ----------------------------------------------------------------------------- probes
def make_probe(task, method, n_jobs):
    """The paper's probe hyperparameters."""
    if method == "xgboost":
        import xgboost as xgb
        common = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, tree_method="hist", n_jobs=n_jobs, random_state=PROBE_SEED)
        return xgb.XGBRegressor(**common) if task == "reg" else xgb.XGBClassifier(eval_metric="mlogloss", **common)
    if task == "reg":
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 6)))
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))


def fit_predict(task, method, Xtr, ytr, Xte, n_jobs):
    """Predictions on Xte, or None when a classification draw holds a single class."""
    if task == "clf":
        classes, ytr_enc = np.unique(ytr, return_inverse=True)
        if len(classes) < 2:
            return None
        model = make_probe(task, method, n_jobs).fit(Xtr, ytr_enc)
        return classes[model.predict(Xte)]
    return make_probe(task, method, n_jobs).fit(Xtr, ytr).predict(Xte)


def score(task, y_true, y_pred) -> float:
    if task == "clf":
        return float(f1_score(y_true, y_pred, average="macro"))
    return float(r2_score(y_true, y_pred))


# ----------------------------------------------------------------------------- design
def labelled_rows(task, y):
    if task == "reg":
        return np.flatnonzero(np.isfinite(y))
    return np.flatnonzero(np.array([v is not None for v in y]))


def frozen_split(target, task, y):
    """(test_rows, pool_rows): same for every arm because it depends only on the labels."""
    rows = labelled_rows(task, y)
    rng = np.random.default_rng([DRAW_SEED, int.from_bytes(target.encode(), "little") % (2**31)])
    perm = rows[rng.permutation(len(rows))]
    n_test = min(FROZEN_TEST_N, len(rows) // 2)
    return perm[:n_test], perm[n_test:]


def train_draws(target, pool, n):
    """Disjoint training draws of size n from the pool, as consecutive chunks of a permutation
    seeded by (target, n)."""
    k = min(repeats(n), len(pool) // n)
    if k == 0:
        return []
    rng = np.random.default_rng([DRAW_SEED, int.from_bytes(target.encode(), "little") % (2**31), n])
    perm = pool[rng.permutation(len(pool))]
    return [perm[i * n:(i + 1) * n] for i in range(k)]


def design_table(inp: Inputs):
    rows = []
    for t in TARGETS:
        task = task_of(t)
        test, pool = frozen_split(t, task, inp.targets[t])
        sizes = [(n, len(train_draws(t, pool, n))) for n in TRAIN_GRID]
        rows.append(dict(target=t, task=task, labelled=len(test) + len(pool), n_test=len(test), n_pool=len(pool),
                         draws=" ".join(f"{n}x{k}" for n, k in sizes if k)))
    return rows


# ----------------------------------------------------------------------------- raw CSV helpers
TRAIN_COLS = ["arm", "target", "method", "axis", "n", "rep", "score", "n_test", "n_pool"]
TEST_COLS = ["arm", "target", "method", "axis", "n", "rep", "score", "n_train_fit"]


def read_rows(path):
    return list(csv.DictReader(open(path))) if path.exists() else []


def append_rows(path, cols, rows):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerows(rows)


def write_rows(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def _fmt(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.6f}" if isinstance(v, float) else v)


# ----------------------------------------------------------------------------- pass: curves
def record_frozen_ids(out, inp: Inputs):
    """frozen_test_ids.csv: the materials of every target's test set; written once, and
    checked against on every later run so the design cannot drift between arms."""
    path = out / "frozen_test_ids.csv"
    rows = []
    for t in TARGETS:
        test, _ = frozen_split(t, task_of(t), inp.targets[t])
        rows += [dict(target=t, mp_id=m) for m in inp.mp_ids[np.sort(test)]]
    if path.exists():
        old = read_rows(path)
        if [(r["target"], r["mp_id"]) for r in old] != [(r["target"], r["mp_id"]) for r in rows]:
            raise SystemExit(f"{path} differs from the current design; the inputs changed")
        return
    write_rows(path, ["target", "mp_id"], rows)


def run_train_axis(inp: Inputs, out, targets, n_jobs):
    path = out / "raw" / f"curves_train_{inp.arm}.csv"
    done = {(r["target"], r["method"], int(r["n"])) for r in read_rows(path)}
    for t in targets:
        task, y = task_of(t), inp.targets[t]
        test, pool = frozen_split(t, task, y)
        Xte, yte = inp.X[test], y[test]
        print(f"[{inp.arm} train] {t}: {len(test):,} test, {len(pool):,} pool", flush=True)
        for n in TRAIN_GRID:
            draws = train_draws(t, pool, n)
            for method in METHODS:
                if not draws or (t, method, n) in done:
                    continue
                t0, rows = time.time(), []
                for rep, tr in enumerate(draws):
                    pred = fit_predict(task, method, inp.X[tr], y[tr], Xte, n_jobs)
                    s = score(task, yte, pred) if pred is not None else float("nan")
                    rows.append(dict(arm=inp.arm, target=t, method=method, axis="train", n=n, rep=rep,
                                     score=_fmt(s), n_test=len(test), n_pool=len(pool)))
                append_rows(path, TRAIN_COLS, rows)
                med = np.nanmedian([float(r["score"]) for r in rows])
                print(f"    n={n:>6} {method:8s} {len(draws):>2} draws  median={med:.3f}  ({time.time()-t0:.0f}s)", flush=True)


def run_test_axis(inp: Inputs, out, targets, n_jobs):
    path = out / "raw" / f"curves_test_{inp.arm}.csv"
    done = {(r["target"], r["method"], int(r["n"])) for r in read_rows(path)}
    for t in targets:
        task, y = task_of(t), inp.targets[t]
        test, pool = frozen_split(t, task, y)
        sizes = [m for m in TEST_GRID if m <= len(test)]
        for method in METHODS:
            todo = [m for m in sizes if (t, method, m) not in done]
            if not todo:
                continue
            fit_rows = pool[:min(TEST_AXIS_TRAIN_N, len(pool))]
            t0 = time.time()
            pred = fit_predict(task, method, inp.X[fit_rows], y[fit_rows], inp.X[test], n_jobs)
            print(f"[{inp.arm} test] {t} {method}: fitted on {len(fit_rows):,}, full-test score "
                  f"{score(task, y[test], pred):.3f} ({time.time()-t0:.0f}s)", flush=True)
            rng = np.random.default_rng([DRAW_SEED, int.from_bytes(t.encode(), "little") % (2**31), 999])
            yte = y[test]
            for m in todo:
                rows = []
                k = 1 if m == len(test) else TEST_RESAMPLES
                for rep in range(k):
                    sub = rng.choice(len(test), m, replace=False) if m < len(test) else np.arange(len(test))
                    rows.append(dict(arm=inp.arm, target=t, method=method, axis="test", n=m, rep=rep,
                                     score=_fmt(score(task, yte[sub], pred[sub])), n_train_fit=len(fit_rows)))
                append_rows(path, TEST_COLS, rows)


# ----------------------------------------------------------------------------- pass: summarise
def _group(rows, keys):
    g = {}
    for r in rows:
        g.setdefault(tuple(r[k] for k in keys), []).append(r)
    return g


def _scores(rows):
    return np.array([float(r["score"]) for r in rows if r["score"] not in ("", "nan")])


def summarise_arm(arm, out, ceilings):
    rows = []
    for axis in ("train", "test"):
        raw = read_rows(out / "raw" / f"curves_{axis}_{arm}.csv")
        for (t, method), g in sorted(_group(raw, ["target", "method"]).items()):
            key = (t, method)
            by_n = _group(g, ["n"])
            if key in ceilings:
                full, src = ceilings[key], "reference"
            else:
                nmax = max(int(n) for (n,) in by_n)
                full, src = float(np.nanmedian(_scores(by_n[(str(nmax),)]))), "measured"
            for (n,), gg in sorted(by_n.items(), key=lambda kv: int(kv[0][0])):
                s = _scores(gg)
                if len(s) == 0:
                    continue
                sd_spread = P90_P10_OVER_SD * float(np.std(s, ddof=1)) if len(s) > 1 else float("nan")
                rows.append(dict(arm=arm, target=t, method=method, axis=axis, n=int(n), reps=len(s),
                                 median=float(np.median(s)), p10=float(np.quantile(s, 0.10)),
                                 p90=float(np.quantile(s, 0.90)), spread=sd_spread,
                                 spread_direct=float(np.quantile(s, 0.90) - np.quantile(s, 0.10)),
                                 ceiling=full, ceiling_source=src, gap=full - float(np.median(s))))
    return rows


def _first_sustained(points, col, cut):
    """Smallest n from which `col` <= cut holds at every larger measured n; None if never.
    Points where the quantity is undefined (a single draw has no spread) are not evidence
    either way and are skipped."""
    pts = [r for r in sorted(points, key=lambda r: r["n"]) if r[col] == r[col]]
    ok = [r[col] <= cut for r in pts]
    for i in range(len(pts)):
        if all(ok[i:]):
            return pts[i]["n"]
    return None


def thresholds(summary):
    rows = []
    for (arm, t, method), g in sorted(_group(summary, ["arm", "target", "method"]).items()):
        tr = [r for r in g if r["axis"] == "train"]
        te = [r for r in g if r["axis"] == "test"]
        rows.append(dict(arm=arm, target=t, method=method,
                         n_stable=_first_sustained(tr, "spread", MAX_SPREAD) if tr else None,
                         n_saturated=_first_sustained(tr, "gap", MAX_GAP) if tr else None,
                         n_train_max=max(r["n"] for r in tr) if tr else None,
                         test_n_stable=_first_sustained(te, "spread", MAX_SPREAD) if te else None,
                         ceiling=tr[0]["ceiling"] if tr else None, ceiling_source=tr[0]["ceiling_source"] if tr else None))
    return rows


def write_config_table(th, path):
    """config/label_efficiency_thresholds.csv from the XGBoost curves. An unsaturated curve
    takes its largest measured n. n_gnn = the larger of the two embedding arms' saturation
    sizes: the size where BOTH frozen-embedding probes have stopped improving."""
    old = {r["target"]: r for r in read_rows(path)}
    by = {(r["arm"], r["target"]): r for r in th if r["method"] == "xgboost"}
    rows = []
    for t in TARGETS:
        o, u = by.get(("orb", t)), by.get(("uma", t))
        if o is None or u is None:
            print(f"  config: {t} skipped (missing {'orb' if o is None else 'uma'} curve)")
            continue
        sat_o = o["n_saturated"] or o["n_train_max"]
        sat_u = u["n_saturated"] or u["n_train_max"]
        row = dict(target=safe(t), orb_n_stable=o["n_stable"] or "", uma_n_stable=u["n_stable"] or "",
                   n_saturated_orb=sat_o, n_saturated_uma=sat_u, n_gnn=max(sat_o, sat_u))
        prev = old.get(safe(t))
        if prev:
            diff = {k: (prev[k], str(row[k])) for k in row if k != "target" and str(prev.get(k, "")) != str(row[k])}
            if diff:
                print(f"  config: {t} changed: " + ", ".join(f"{k} {a}->{b}" for k, (a, b) in diff.items()))
        rows.append(row)
    if len(rows) == len(TARGETS):
        write_rows(path, ["target", "orb_n_stable", "uma_n_stable", "n_saturated_orb", "n_saturated_uma", "n_gnn"], rows)
        print(f"  wrote {path}")
    else:
        print(f"  config table NOT written: {len(rows)}/{len(TARGETS)} targets have both embedding curves")


def plot_train(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"orb": "#2a78d6", "uma": "#eb6834", "magpie": "#1baf7a"}
    fig, axes = plt.subplots(3, 3, figsize=(10, 8.5))
    for ax, t in zip(axes.flat, TARGETS):
        for arm in ARMS:
            pts = sorted([r for r in summary if r["arm"] == arm and r["target"] == t
                          and r["method"] == "xgboost" and r["axis"] == "train"], key=lambda r: r["n"])
            if not pts:
                continue
            n = [r["n"] for r in pts]
            ax.fill_between(n, [r["p10"] for r in pts], [r["p90"] for r in pts], color=col[arm], alpha=0.15, lw=0)
            ax.plot(n, [r["median"] for r in pts], color=col[arm], lw=2, marker="o", ms=3, label=arm)
        ax.set_xscale("log"); ax.set_title(t, fontsize=9)
        ax.set_ylabel("macro-F1" if t in CLF else "R²"); ax.set_xlabel("training materials")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "curves_train.png", dpi=150); plt.close(fig)


def pass_summarise(args, out):
    allrows = []
    for arm in ARMS:
        if not (out / "raw" / f"curves_train_{arm}.csv").exists():
            print(f"  {arm}: no raw curves yet, skipped")
            continue
        ceilings = load_inputs(args, arm, need_features=False).ceilings
        rows = summarise_arm(arm, out, ceilings)
        cols = ["arm", "target", "method", "axis", "n", "reps", "median", "p10", "p90", "spread",
                "spread_direct", "ceiling", "ceiling_source", "gap"]
        write_rows(out / f"summary_{arm}.csv", cols, [{k: _fmt(v) if isinstance(v, float) else v for k, v in r.items()} for r in rows])
        allrows += rows
        print(f"  {arm}: {len(rows)} summary points")
    th = thresholds(allrows)
    write_rows(out / "thresholds.csv", ["arm", "target", "method", "n_stable", "n_saturated", "n_train_max",
                                        "test_n_stable", "ceiling", "ceiling_source"],
               [{k: _fmt(v) if isinstance(v, float) else ("" if v is None else v) for k, v in r.items()} for r in th])
    for r in th:
        if r["method"] == "xgboost":
            print(f"  {r['arm']:6s} {r['target']:22s} n_stable={r['n_stable']}  n_saturated={r['n_saturated']}  test_n_stable={r['test_n_stable']}")
    if allrows:
        plot_train(allrows, out)
    write_config_table(th, Path(args.config_csv))


# ----------------------------------------------------------------------------- pass: gain
def pass_gain(out):
    def draws(arm):
        raw = read_rows(out / "raw" / f"curves_train_{arm}.csv")
        return {(r["target"], int(r["n"]), int(r["rep"])): float(r["score"])
                for r in raw if r["method"] == "xgboost" and r["score"] not in ("", "nan")}
    base = draws("magpie")
    rows = []
    for arm in ("orb", "uma"):
        emb = draws(arm)
        for (t, n) in sorted({(k[0], k[1]) for k in emb}):
            d = np.array([emb[k] - base[k] for k in emb if k[0] == t and k[1] == n and k in base])
            if len(d) < MIN_PAIRS:
                continue
            rows.append(dict(arm=arm, target=t, n=n, n_pairs=len(d), median=_fmt(float(np.median(d))),
                             p10=_fmt(float(np.quantile(d, 0.1))), p90=_fmt(float(np.quantile(d, 0.9))),
                             frac_positive=_fmt(float((d > 0).mean()))))
    write_rows(out / "paired_gain.csv", ["arm", "target", "n", "n_pairs", "median", "p10", "p90", "frac_positive"], rows)
    print(f"  paired_gain.csv: {len(rows)} rows")


# ----------------------------------------------------------------------------- pass: split-fraction
def _loginterp(xs, ys, x):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if x < xs.min() or x > xs.max():
        return float("nan")
    return float(np.interp(np.log(x), np.log(xs), ys))


def pass_split_fraction(out):
    """Total error of one train/test split of N materials with a fraction f held out:
    the shortfall (ceiling minus the training curve at (1-f)N) and the test-size noise
    added in quadrature. The noise is the test-axis spread at fN, measured for a probe
    near its ceiling and rescaled by (1 - score) / (1 - ceiling) because a probe scoring
    lower scatters more, then divided by 2.5631 to give a standard deviation. Both terms
    are read off the XGBoost summaries by log-linear interpolation; cells outside the
    measured grids are left blank."""
    rows = []
    for arm in ARMS:
        summ = read_rows(out / f"summary_{arm}.csv")
        for t in TARGETS:
            tr = sorted([r for r in summ if r["target"] == t and r["method"] == "xgboost" and r["axis"] == "train"], key=lambda r: int(r["n"]))
            te = sorted([r for r in summ if r["target"] == t and r["method"] == "xgboost" and r["axis"] == "test" and r["spread"] != ""], key=lambda r: int(r["n"]))
            if not tr or not te:
                continue
            ceiling = float(tr[0]["ceiling"])
            for N in SPLIT_SIZES:
                for f in SPLIT_FRACTIONS:
                    n_te = int(round(f * N)); n_tr = N - n_te
                    score = _loginterp([int(r["n"]) for r in tr], [float(r["median"]) for r in tr], n_tr)
                    spread = _loginterp([int(r["n"]) for r in te], [float(r["spread"]) for r in te], n_te)
                    short = max(0.0, ceiling - score)
                    rescaled = spread * (1.0 - score) / max(1e-9, 1.0 - ceiling)
                    sd = rescaled / P90_P10_OVER_SD
                    rows.append(dict(arm=arm, target=t, N=N, fraction=f, n_train=n_tr, n_test=n_te, score=_fmt(score),
                                     shortfall=_fmt(short), spread=_fmt(rescaled), sd=_fmt(sd),
                                     total=_fmt(float(np.hypot(short, sd)))))
    write_rows(out / "split_fraction.csv", ["arm", "target", "N", "fraction", "n_train", "n_test", "score", "shortfall", "spread", "sd", "total"], rows)
    print(f"  split_fraction.csv: {len(rows)} rows")


# [pipeline-only]
# ----------------------------------------------------------------------------- pass: gnn
def gnn_sizes() -> dict[str, int]:
    with open(THRESHOLDS_CSV) as fh:
        return {r["target"]: int(r["n_gnn"]) for r in csv.DictReader(fh)}


def gnn_sweep(target, orb_dir, cache_path, seed, sizes=None, have=(), device="cpu"):
    """GNN scores at each training size in `sizes` (default: the thresholds table's n_gnn), GNN_DRAWS
    draws up to 5,000 and one above; (n, rep) pairs already in `have` are skipped."""
    import importlib
    O = importlib.import_module("stages.probes.probes_orb")
    G = importlib.import_module("mlipprobes.gnn")
    from sklearn.preprocessing import LabelEncoder
    feat_dir = orb_dir / "features"
    canon, ydict = O.load_canonical_targets(feat_dir, [target])
    canon_pos = {m: i for i, m in enumerate(canon)}
    y = ydict[target]
    task = task_of(target)
    le = LabelEncoder().fit(y[O.valid_mask(target, y)]) if task == "clf" else None
    n_classes = len(le.classes_) if task == "clf" else 1
    sp = np.load(orb_dir / "splits" / f"{O.safe(target)}.npz", allow_pickle=True)
    cache = G.GraphCache(cache_path)

    def align(ids):
        keep = [m for m in ids if m in canon_pos and m in cache]
        r = np.fromiter((canon_pos[m] for m in keep), dtype=np.int64, count=len(keep))
        ys = y[r]
        fin = np.isfinite(ys) if task == "reg" else np.array([v is not None for v in ys])
        return [keep[i] for i in range(len(keep)) if fin[i]], ys[fin]

    train_pool_ids, ytr_pool_raw = align(sp["train_mp_ids"])
    test_ids, yte_raw = align(sp["test_mp_ids"])
    if task == "clf":
        ytr_pool, yte = le.transform(ytr_pool_raw), le.transform(yte_raw)
    else:
        ytr_pool, yte = ytr_pool_raw.astype(float), yte_raw.astype(float)
    n_pool = len(ytr_pool)
    rows_out = []
    rng = np.random.default_rng(seed + 2)
    sizes = sizes or [gnn_sizes()[O.safe(target)]]
    for size in sizes:
        if size > n_pool:
            print(f"    gnn {target}: n={size} exceeds the {n_pool:,} labelled training materials, skipped", flush=True)
            continue
        n = size
        for rep in range(GNN_DRAWS if n <= 5000 else 1):
            if (n, rep) in have:
                continue
            rng = np.random.default_rng([seed + 2, n, rep])      # per (n, rep): a resumed run draws the same subsets
            sub = rng.choice(n_pool, n, replace=False) if n < n_pool else np.arange(n_pool)
            rows_out += _gnn_one(target, task, cache, train_pool_ids, ytr_pool, test_ids, yte, n_classes, sub, n, rep, seed, device)
            write_rows(_GNN_PATH[0], ["target", "method", "config", "n_train", "rep", "score"], _GNN_ROWS[0] + rows_out)
    return rows_out


_GNN_PATH, _GNN_ROWS = [None], [None]     # checkpoint target for gnn_sweep (set by pass_gnn)


def _gnn_one(target, task, cache, train_pool_ids, ytr_pool, test_ids, yte, n_classes, sub, n, rep, seed, device):
    import importlib
    G = importlib.import_module("mlipprobes.gnn")
    if True:
        rng = np.random.default_rng([seed + 3, n, rep])
        ids_sub = [train_pool_ids[i] for i in sub]
        y_sub = ytr_pool[sub]
        n_val = max(10, min(len(ids_sub) // 5, 500))
        val_idx = rng.choice(len(ids_sub), n_val, replace=False) if n_val < len(ids_sub) else np.arange(len(ids_sub))
        val_set = set(val_idx.tolist())
        tr_idx = [i for i in range(len(ids_sub)) if i not in val_set]
        if len(tr_idx) < 5:
            tr_idx = list(range(len(ids_sub)))
        t0 = time.time()
        model, _ = G.train_gnn([ids_sub[i] for i in tr_idx], y_sub[tr_idx], [ids_sub[i] for i in val_idx], y_sub[val_idx],
                               task=task, cache=cache, n_classes=n_classes, seed=seed + rep, device=device,
                               max_epochs=200, patience=15)
        pred = G.predict_gnn(model, test_ids, cache, task=task, device=device)
        s = score(task, yte, pred)
        print(f"    gnn {target} n={n} rep={rep}: score={s:.3f} ({time.time()-t0:.1f}s)", flush=True)
        return [dict(target=target, method="gnn", config="mp-gnn-3x64", n_train=n, rep=rep, score=s)]


def pass_gnn(args, out, targets):
    import importlib
    O = importlib.import_module("stages.probes.probes_orb")
    G = importlib.import_module("mlipprobes.gnn")
    orb_dir, cache_path = Path(args.orb_dir), Path(args.graph_cache)
    path = out / "label_efficiency_table.csv"
    rows = read_rows(path)
    for r in rows:
        r["n_train"] = int(r["n_train"]); r["rep"] = int(r["rep"]); r["score"] = float(r["score"])
    sizes = gnn_sizes()
    grid = [int(x) for x in args.gnn_grid.split(",")] if args.gnn_grid else None
    def wanted(r):
        n_sat = sizes[O.safe(r["target"])]
        return r["n_train"] == n_sat or (grid is not None and r["n_train"] in grid)
    keep = [r for r in rows if r["method"] != "gnn" or wanted(r)]
    dropped = len(rows) - len(keep)
    if dropped:
        print(f"  {dropped} GNN rows at sizes outside the thresholds table / grid are dropped")
    rows = keep
    needed = set()
    for t in targets:
        sp = np.load(orb_dir / "splits" / f"{O.safe(t)}.npz", allow_pickle=True)
        needed.update(sp["train_mp_ids"].astype(str)); needed.update(sp["test_mp_ids"].astype(str))
    print(f"[gnn] packing graphs for {len(needed):,} materials (once, cached, resumable)")
    G.pack_graphs(sorted(needed), Path(args.structures_dir), cache_path, n_jobs=args.n_jobs)
    for t in targets:
        have = {(r["n_train"], r["rep"]) for r in rows if r["method"] == "gnn" and r["target"] == t}
        want = sorted(set((grid or []) + [sizes[O.safe(t)]]))
        print(f"[gnn] {t}: sizes {want}", flush=True)
        _GNN_PATH[0], _GNN_ROWS[0] = path, rows
        new = gnn_sweep(t, orb_dir, cache_path, args.seed, sizes=want, have=have)
        rows += new
        write_rows(path, ["target", "method", "config", "n_train", "rep", "score"], rows)
    print(f"  wrote {path}")
# [/pipeline-only]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pass_", metavar="PASS", choices=["curves", "summarise", "gain", "split-fraction", "gnn"])
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--axis", choices=["train", "test"])
    ap.add_argument("--targets", help="comma list; default all nine")
    ap.add_argument("--from-dataset", help="read inputs from the public parquet release at this directory")
    ap.add_argument("--uma-dataset", help="the companion gated release for the UMA arm (default: <dataset>/uma or the sibling uma_gated)")
    # [pipeline-only]
    ap.add_argument("--orb-dir", default=str(P.RESULTS / "orb/property_prediction"))
    ap.add_argument("--uma-dir", default=str(P.RESULTS / "fairchem/property_prediction"))
    ap.add_argument("--magpie-npz", default=str(P.RESULTS / "exp_magpie/magpie_features.npz"))
    ap.add_argument("--controls-dir", default=str(P.RESULTS / "proppred_controls"))
    ap.add_argument("--structures-dir", default=str(P.STRUCTURES))
    ap.add_argument("--graph-cache", default=str(OUT_DEFAULT / "graph_cache.npz"))
    # [/pipeline-only]
    ap.add_argument("--config-csv", default=str(THRESHOLDS_CSV))
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT))
    ap.add_argument("--seed", type=int, default=PROBE_SEED, help="GNN seed only; the draw design is fixed")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--gnn-grid", help="gnn: comma list of training sizes to add to n_gnn (a GNN learning curve)")
    ap.add_argument("--dry-run", action="store_true", help="curves: print the draw design and exit")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    targets = args.targets.split(",") if args.targets else TARGETS
    bad = [t for t in targets if t not in TARGETS]
    if bad:
        raise SystemExit(f"unknown targets {bad}; valid: {TARGETS}")

    if args.pass_ == "curves":
        if not args.arm or not args.axis:
            raise SystemExit("curves needs --arm and --axis")
        inp = load_inputs(args, args.arm, need_features=not args.dry_run)
        if args.dry_run:
            for r in design_table(inp):
                print(f"  {r['target']:22s} {r['task']} labelled={r['labelled']:>7,} test={r['n_test']:>6,} "
                      f"pool={r['n_pool']:>7,}  draws: {r['draws']}")
            return
        record_frozen_ids(out, inp)
        print(f"{args.arm}: {len(inp.mp_ids):,} materials, {inp.X.shape[1]} features; "
              f"ceilings for {len(inp.ceilings)} (target, probe) pairs", flush=True)
        (run_train_axis if args.axis == "train" else run_test_axis)(inp, out, targets, args.n_jobs)
    elif args.pass_ == "summarise":
        pass_summarise(args, out)
    elif args.pass_ == "gain":
        pass_gain(out)
    elif args.pass_ == "split-fraction":
        pass_split_fraction(out)
    else:
        if "pass_gnn" not in globals():
            raise SystemExit("the GNN arm runs inside the paper pipeline only (it needs the raw structures)")
        pass_gnn(args, out, targets)


if __name__ == "__main__":
    main()
