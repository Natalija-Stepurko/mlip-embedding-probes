#!/usr/bin/env python3
"""
mlipprobes/significance_lib.py — Statistical significance for the ORB vs UMA
property-prediction results (stages/probes/probes_{orb,uma}.py).

Why this exists
---------------
Scripts 05 report a single point metric per (target, config) from ONE fixed
80/20 split (seed 42). A single split has no split-to-split variance, so the
only defensible classical tests are *paired, same-test-set* comparisons on the
per-material predictions, plus bootstrap confidence intervals that quantify
sampling noise on that fixed test set. This script does exactly that — it does
NOT retrain over many seeds (that would be needed for "layer b3 > b2"-style
claims and is out of scope here).

What it computes (per target, best XGBoost config in each of ORB / UMA):
  1. Bootstrap 95% CI on the headline metric (R2 for regression, macro-F1 for
     classification) for ORB and for UMA, on each one's own fixed test set.
  2. Paired UMA-minus-ORB difference on the INTERSECTION of their test
     materials (so the comparison is truly paired), with a paired-bootstrap CI
     and a two-sided bootstrap p-value.
  3. Paired XGBoost-minus-linear difference (same feature set / same test set,
     the best XGB config) — "is the nonlinear head worth it".
Multiple-comparison control: Benjamini-Hochberg FDR across targets, per family
of comparisons.

Caveats it prints alongside the numbers:
  * With ~31k test materials almost any non-zero difference is "significant" —
    read the effect sizes and CIs, not just p. Tiny Δ with p<1e-3 is noise-free
    but scientifically trivial.
  * A target whose best R2 < 0 would be reported but flagged UNRELIABLE — a
    bootstrap CI around a failed fit is not meaningful (none of the nine is).

Predictions are recovered by refitting the single best config per target (no
prediction cache exists yet); this reuses the loaders in the two 05 modules and
does not modify them.

Reads  $MLIP_PROBES_ROOT/results/orb/property_prediction/{metrics.csv,features,splits}
       $MLIP_PROBES_ROOT/results/fairchem/property_prediction/{metrics.csv,features,splits}
Writes $MLIP_PROBES_ROOT/results/proppred_significance/

Example
-------
  uv run python scripts/mlipprobes/significance_lib.py --bootstrap 2000
  uv run python scripts/mlipprobes/significance_lib.py --targets band_gap,bulk_modulus --bootstrap 500
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import argparse
import csv
import math
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["legend.frameon"] = False  # no box border around figure legends
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
O = importlib.import_module("stages.probes.probes_orb")
U = importlib.import_module("stages.probes.probes_uma")


# ---------------------------------------------------------------------------
# Prediction recovery (refit the single best config; return mp_id-keyed preds)
# ---------------------------------------------------------------------------

def _refit(task, Xtr, ytr, Xte, seed, n_jobs):
    xgb = O.make_model(task, "xgb", seed, n_jobs).fit(Xtr, ytr)
    lin = O.make_model(task, "linear", seed, n_jobs).fit(Xtr, ytr)
    return xgb.predict(Xte), lin.predict(Xte)


def orb_predictions(orb_dir: Path, target, rep, layer, seed, n_jobs):
    """Return (test_mp_ids, ytrue, ypred_xgb, ypred_linear, is_clf, baseline_pred)
    for ORB's best config on `target`. baseline_pred = constant train-mean (reg)
    or train-majority class (clf) on the test set."""
    feat_dir = orb_dir / "features"
    canon, ydict = O.load_canonical_targets(feat_dir, [target])
    canon_pos = {m: i for i, m in enumerate(canon)}
    d = np.load(feat_dir / f"{layer}_pooled.npz", allow_pickle=True)
    assert np.array_equal(d["mp_ids"], canon), "ORB feature/target mp_id order mismatch"
    X = O.build_X(d["node"], d["edge"], rep)
    y = ydict[target]
    task = "clf" if O.is_clf(target) else "reg"
    le = LabelEncoder().fit(y[O.valid_mask(target, y)]) if task == "clf" else None

    sp = np.load(orb_dir / "splits" / f"{O.safe(target)}.npz", allow_pickle=True)

    def align(ids):
        keep = [m for m in ids if m in canon_pos]
        r = np.fromiter((canon_pos[m] for m in keep), dtype=np.int64, count=len(keep))
        Xs, ys, ms = X[r], y[r], np.array(keep, dtype=object)
        fin = np.isfinite(Xs).all(axis=1)
        return Xs[fin], ys[fin], ms[fin]

    Xtr, ytr_raw, _ = align(sp["train_mp_ids"])
    Xte, yte_raw, te_mp = align(sp["test_mp_ids"])
    if task == "clf":
        ytr, yte = le.transform(ytr_raw), le.transform(yte_raw)
    else:
        ytr, yte = ytr_raw.astype(float), yte_raw.astype(float)
    px, pl = _refit(task, Xtr, ytr, Xte, seed, n_jobs)
    if task == "clf":
        vals, cnts = np.unique(ytr, return_counts=True)
        base = np.full(len(yte), vals[int(cnts.argmax())])   # majority-class baseline
    else:
        base = np.full(len(yte), float(np.mean(ytr)))        # train-mean baseline
    return te_mp, yte, px, pl, task == "clf", base


def uma_predictions(uma_dir: Path, target, group, family, layer, seed, n_jobs):
    """Return (test_mp_ids, ytrue, ypred_xgb, ypred_linear, is_clf, baseline_pred)
    for UMA's best config on `target`. baseline_pred = constant train-mean (reg)
    or train-majority class (clf) on the test set."""
    feat_dir = uma_dir / "features"
    canon, ydict = U.load_targets(feat_dir, [target])
    canon_pos = {m: i for i, m in enumerate(canon)}
    X, feat_ids = U.load_feature_set(group, family, layer)
    feat_pos = {m: i for i, m in enumerate(feat_ids)}
    y = ydict[target]
    task = "clf" if U.is_clf(target) else "reg"
    le = LabelEncoder().fit(y[U.valid_mask(target, y)]) if task == "clf" else None

    sp = np.load(uma_dir / "splits" / f"{U.safe(target)}.npz", allow_pickle=True)

    def align(ids):
        keep = [m for m in ids if m in feat_pos and m in canon_pos]
        rx = np.fromiter((feat_pos[m] for m in keep), dtype=np.int64, count=len(keep))
        ry = np.fromiter((canon_pos[m] for m in keep), dtype=np.int64, count=len(keep))
        Xs, ys, ms = X[rx], y[ry], np.array(keep, dtype=object)
        fin = np.isfinite(Xs).all(axis=1)
        return Xs[fin], ys[fin], ms[fin]

    Xtr, ytr_raw, _ = align(sp["train_mp_ids"])
    Xte, yte_raw, te_mp = align(sp["test_mp_ids"])
    if task == "clf":
        ytr, yte = le.transform(ytr_raw), le.transform(yte_raw)
    else:
        ytr, yte = ytr_raw.astype(float), yte_raw.astype(float)
    px, pl = _refit(task, Xtr, ytr, Xte, seed, n_jobs)
    if task == "clf":
        vals, cnts = np.unique(ytr, return_counts=True)
        base = np.full(len(yte), vals[int(cnts.argmax())])   # majority-class baseline
    else:
        base = np.full(len(yte), float(np.mean(ytr)))        # train-mean baseline
    return te_mp, yte, px, pl, task == "clf", base


# ---------------------------------------------------------------------------
# Bootstrap machinery
# ---------------------------------------------------------------------------

def _metric(ytrue, ypred, is_clf):
    if is_clf:
        return f1_score(ytrue, ypred, average="macro")
    return r2_score(ytrue, ypred)


def boot_ci(ytrue, ypred, is_clf, B, rng, alpha=0.05):
    """Bootstrap point estimate + (lo, hi) CI of the metric on a fixed test set."""
    n = len(ytrue)
    point = _metric(ytrue, ypred, is_clf)
    stats = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        stats[b] = _metric(ytrue[i], ypred[i], is_clf)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, lo, hi


def paired_boot_diff(ytrue, pred_a, pred_b, is_clf, B, rng, alpha=0.05):
    """Paired bootstrap of metric(a) - metric(b) on the SAME resampled indices.
    Returns (point_diff, lo, hi, two_sided_p)."""
    n = len(ytrue)
    point = _metric(ytrue, pred_a, is_clf) - _metric(ytrue, pred_b, is_clf)
    diffs = np.empty(B)
    for k in range(B):
        i = rng.integers(0, n, n)
        diffs[k] = (_metric(ytrue[i], pred_a[i], is_clf)
                    - _metric(ytrue[i], pred_b[i], is_clf))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return point, lo, hi, min(1.0, p)


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        i = n - rank  # 1-based rank from the largest
        val = p[idx] * n / i
        prev = min(prev, val)
        adj[idx] = prev
    return adj


# Full metric set for the summary table. Baseline predictor = train mean (reg) /
# train majority class (clf) evaluated on the test set.
REG_METRIC_COLS = [("mae", "MAE"), ("rmse", "RMSE"), ("std", "σ (target)"),
                   ("rmse_over_std", "RMSE/σ"), ("r2", "R2"),
                   ("base_mae", "baseline MAE"), ("base_rmse", "baseline RMSE"),
                   ("nonlin_gap", "XGB−linear gap (R2)")]
CLF_METRIC_COLS = [("accuracy", "accuracy"), ("macro_f1", "macro-F1"),
                   ("n_classes", "K classes"), ("macro_f1_skill", "macro-F1 skill"),
                   ("base_accuracy", "baseline accuracy"), ("base_macro_f1", "baseline macro-F1"),
                   ("nonlin_gap", "XGB−linear gap (macro-F1)")]


def full_metrics(yt, yp, base, is_clf, B, rng, alpha=0.05):
    """Point estimate + bootstrap 95% CI for the full metric set on a fixed test
    set. Regression: mae, rmse, r2, base_mae, base_rmse. Classification:
    accuracy, macro_f1, base_accuracy, base_macro_f1. `base` is the constant
    baseline prediction vector. Returns {metric: (point, lo, hi)}."""
    yt = np.asarray(yt); yp = np.asarray(yp); base = np.asarray(base)
    n = len(yt)
    if is_clf:
        acc = lambda t, p: float(np.mean(t == p))
        mf1 = lambda t, p: f1_score(t, p, average="macro", zero_division=0)
        specs = [("accuracy", yp, acc), ("macro_f1", yp, mf1),
                 ("base_accuracy", base, acc), ("base_macro_f1", base, mf1)]
    else:
        mae = lambda t, p: float(np.mean(np.abs(t - p)))
        rmse = lambda t, p: float(np.sqrt(np.mean((t - p) ** 2)))
        r2 = lambda t, p: float(r2_score(t, p))
        std = lambda t, p: float(np.std(t))                        # target spread (test set)
        nrmse = lambda t, p: float(np.sqrt(np.mean((t - p) ** 2)) / np.std(t))  # = sqrt(1-R2)
        specs = [("mae", yp, mae), ("rmse", yp, rmse), ("std", yp, std),
                 ("rmse_over_std", yp, nrmse), ("r2", yp, r2),
                 ("base_mae", base, mae), ("base_rmse", base, rmse)]
    names = [s[0] for s in specs]
    point = {nm: fn(yt, pr) for nm, pr, fn in specs}
    boot = {nm: np.empty(B) for nm in names}
    for b in range(B):
        idx = rng.integers(0, n, n)
        t = yt[idx]
        for nm, pr, fn in specs:
            boot[nm][b] = fn(t, pr[idx])
    out = {}
    for nm in names:
        lo, hi = np.percentile(boot[nm], [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[nm] = (point[nm], float(lo), float(hi))
    if is_clf:
        # macro-F1 has no variance-normalisation like R2, but its chance level
        # depends on the number of classes K and the class balance. K sets the
        # baseline; macro_f1_skill = (F1 - F1_base)/(1 - F1_base) is the
        # chance-corrected analog of R2 (0 = majority baseline, 1 = perfect).
        K = int(len(np.unique(yt)))
        out["n_classes"] = (float(K), float(K), float(K))  # constant → rendered without CI
        den = np.clip(1.0 - boot["base_macro_f1"], 1e-9, None)
        sk_boot = (boot["macro_f1"] - boot["base_macro_f1"]) / den
        sk_pt = (point["macro_f1"] - point["base_macro_f1"]) / max(1.0 - point["base_macro_f1"], 1e-9)
        lo, hi = np.percentile(sk_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out["macro_f1_skill"] = (float(sk_pt), float(lo), float(hi))
    return out


REP_PAIRS = [("node", "edge"), ("node", "concat"), ("edge", "concat")]


def _paired_diff_on_common(mp_a, yt_a, px_a, mp_b, yt_b, px_b, is_clf, B, rng):
    pos_a = {m: i for i, m in enumerate(mp_a)}
    common = [m for m in mp_b if m in pos_a]
    pos_b = {m: i for i, m in enumerate(mp_b)}
    ia = np.fromiter((pos_a[m] for m in common), dtype=np.int64, count=len(common))
    ib = np.fromiter((pos_b[m] for m in common), dtype=np.int64, count=len(common))
    d_pt, d_lo, d_hi, d_p = paired_boot_diff(yt_b[ib], px_b[ib], px_a[ia], is_clf, B, rng)
    return d_pt, d_lo, d_hi, d_p, len(common)


def orb_rep_analysis(orb_dir, orb_rows, target, is_clf, o_rep, o_layer, o_mp, o_yt, o_px,
                     B, rng, seed, n_jobs):
    """Bootstrap CI for each of ORB's node/edge/concat (best layer within that
    rep) + pairwise paired-bootstrap diffs among them. Reuses the already-fit
    overall-best predictions when a rep coincides with it (no redundant refit)."""
    data = {}
    for rep in O.REPS:
        layer, _ = O.best_layer_for_rep(orb_rows, target, rep)
        if layer is None:
            continue
        if rep == o_rep and layer == o_layer:
            mp, yt, px = o_mp, o_yt, o_px
        else:
            mp, yt, px, _, _, _ = orb_predictions(orb_dir, target, rep, layer, seed, n_jobs)
        pt, lo, hi = boot_ci(yt, px, is_clf, B, rng)
        data[rep] = dict(mp=mp, yt=yt, px=px, pt=pt, lo=lo, hi=hi)
    pairs = {}
    for a, b in REP_PAIRS:
        if a not in data or b not in data:
            continue
        pairs[(a, b)] = _paired_diff_on_common(data[a]["mp"], data[a]["yt"], data[a]["px"],
                                               data[b]["mp"], data[b]["yt"], data[b]["px"],
                                               is_clf, B, rng)
    return data, pairs


def uma_rep_analysis(uma_dir, uma_rows, target, is_clf, u_group, u_fam, u_layer, u_mp, u_yt, u_px,
                     B, rng, seed, n_jobs):
    data = {}
    for grp in U.GROUPS:
        cfg, _ = U.best_within_group(uma_rows, target, grp)
        if cfg is None:
            continue
        fam, layer = cfg
        if (grp, fam, layer) == (u_group, u_fam, u_layer):
            mp, yt, px = u_mp, u_yt, u_px
        else:
            mp, yt, px, _, _, _ = uma_predictions(uma_dir, target, grp, fam, layer, seed, n_jobs)
        pt, lo, hi = boot_ci(yt, px, is_clf, B, rng)
        data[grp] = dict(mp=mp, yt=yt, px=px, pt=pt, lo=lo, hi=hi)
    pairs = {}
    for a, b in REP_PAIRS:
        if a not in data or b not in data:
            continue
        pairs[(a, b)] = _paired_diff_on_common(data[a]["mp"], data[a]["yt"], data[a]["px"],
                                               data[b]["mp"], data[b]["yt"], data[b]["px"],
                                               is_clf, B, rng)
    return data, pairs


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


# Practical-effect thresholds on |delta| (R2 and macro-F1 are both ~[0,1]
# scores, so one threshold set applies to both). Domain-specific, not derived
# from the data — chosen so that "significant" boilerplate stops masquerading
# as "important" at n ~ 31k test materials.
EFFECT_THRESHOLDS = (0.005, 0.02, 0.05)  # trivial | small | moderate | large


def classify_effect(delta, lo, hi, thresholds=EFFECT_THRESHOLDS, pos="A", neg="B"):
    """Label a paired difference by practical effect size, not just p/q.
    'tie' = CI straddles zero AND |delta| below the trivial threshold (no
    detectable difference); 'trivial' = detectable but too small to matter."""
    a = abs(delta)
    if a < thresholds[0]:
        return "tie" if (lo <= 0 <= hi) else "trivial"
    side = pos if delta >= 0 else neg
    if a < thresholds[1]:
        return f"small {side} advantage"
    if a < thresholds[2]:
        return f"moderate {side} advantage"
    return f"large {side} advantage"


# Fixed model colours: UMA = goldenrod, ORB = turquoise, wherever the two are
# compared. Trivial / tie differences are light grey.
C_UMA = "#864992"
C_ORB = "#48A394"
C_TIE = "lightgray"


def _effect_color(effect: str, pos: str, pos_color=C_UMA, neg_color=C_ORB, neutral=C_TIE):
    if "advantage" not in effect:
        return neutral
    return pos_color if effect.endswith(f"{pos} advantage") else neg_color


def _verdict(effect: str) -> str:
    """Turn a classify_effect() label into the short narrative phrasing used
    in the markdown report ('large UMA advantage' -> 'UMA hugely better')."""
    if effect in ("tie", "trivial"):
        return effect
    mag, side = effect.split()[0], effect.split()[1]
    if mag == "large":
        return f"{side} hugely better"
    if mag == "moderate":
        return f"{side} better, moderate"
    return f"{side} better, tiny"


def _verdict_cell(effect: str, q: float) -> str:
    """Verdict column text: narrative + the ACTUAL q-value (never stars)."""
    note = "ns, " if q >= 0.05 else ""
    return f"{_verdict(effect)} ({note}q={q:.2g})"


def _adaptive_xlim(los, his, default_lo, default_hi, pad=0.02):
    """Sensible fixed default range, expanded only if real data would clip."""
    lo = min([default_lo] + [v - pad for v in los])
    hi = max([default_hi] + [v + pad for v in his])
    return lo, hi


def _stored_split(mdir, mod, target, mp):
    """Train/test index arrays into `mp` from the probes' stored seed-42 split
    (splits/<target>.npz): the same rows Table 1 was scored on."""
    sp = np.load(mdir / "splits" / f"{mod.safe(target)}.npz", allow_pickle=True)
    pos = {m: i for i, m in enumerate(mp)}
    tr = np.array([pos[m] for m in sp["train_mp_ids"].astype(str) if m in pos])
    te = np.array([pos[m] for m in sp["test_mp_ids"].astype(str) if m in pos])
    return tr, te


def _cfg_matrix(mod, name, mdir, mrows, target, seed):
    """(X, y, clf, train_idx, test_idx, cfg_string) for a model's best config on `target`,
    using the probes' stored seed-42 split (same rows as Table 1)."""
    cfg, _ = mod.best_config(mrows, target)
    if cfg is None:
        return None
    if name == "ORB":
        rep, layer = cfg
        canon, yd = mod.load_canonical_targets(mdir / "features", [target])
        d = np.load(mdir / "features" / f"{layer}_pooled.npz", allow_pickle=True)
        X = mod.build_X(d["node"], d["edge"], rep); xids = d["mp_ids"].astype(str)
        cfg_s = f"{rep}/{layer}"
    else:
        g, fam, layer = cfg
        canon, yd = mod.load_targets(mdir / "features", [target])
        X, fids = mod.load_feature_set(g, fam, layer); xids = fids.astype(str)
        cfg_s = f"{g}/{fam}/{layer}"
    cpos = {m: i for i, m in enumerate(canon.astype(str))}
    xpos = {m: i for i, m in enumerate(xids)}
    y = yd[target]; clf = mod.is_clf(target)
    keep = [m for m in xids if m in cpos and str(y[cpos[m]]) not in ("nan", "None", "")]
    rx = np.array([xpos[m] for m in keep]); ry = np.array([cpos[m] for m in keep])
    Xa = X[rx]; fin = np.isfinite(Xa).all(1); Xa = Xa[fin]
    yr = y[ry][fin]; mp = np.array(keep)[fin]
    if clf:
        ya = LabelEncoder().fit_transform(yr)
    else:
        ya = yr.astype(float)
    tr, te = _stored_split(mdir, mod, target, mp)
    return Xa, ya, clf, tr, te, cfg_s


def run_band_gap_subset(orb_dir, uma_dir, orb_rows, uma_rows, out_dir, seed, n_jobs):
    """Band gap is zero-inflated: roughly half of MP is a metal with gap exactly 0, so a
    pooled R^2 partly rewards metal-vs-non-metal separation. Re-score each model's best
    band-gap configuration on the non-metal test subset only (true gap > 0). Same probe
    settings and the same stored seed-42 split as Table 1."""
    from sklearn.metrics import r2_score, mean_squared_error
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, mod, mdir, mrows in (("ORB", O, orb_dir, orb_rows), ("UMA", U, uma_dir, uma_rows)):
        got = _cfg_matrix(mod, name, mdir, mrows, "band_gap", seed)
        if got is None:
            continue
        X, y, _, tr, te, cfg_s = got
        m = O.make_model("reg", "xgb", seed, n_jobs).fit(X[tr], y[tr])
        pred = m.predict(X[te]); true = y[te]
        nz = true > 0
        full_r2 = r2_score(true, pred)
        full_rmse = float(np.sqrt(mean_squared_error(true, pred)))
        sub_r2 = r2_score(true[nz], pred[nz])
        sub_rmse = float(np.sqrt(mean_squared_error(true[nz], pred[nz])))
        rows.append(dict(model=name, cfg=cfg_s, n_test=len(te), n_nonmetal=int(nz.sum()),
                         frac_metal=round(float((~nz).mean()), 4),
                         r2_full=round(full_r2, 4), rmse_full=round(full_rmse, 4),
                         r2_nonmetal=round(sub_r2, 4), rmse_nonmetal=round(sub_rmse, 4),
                         sigma_full=round(float(true.std()), 4),
                         sigma_nonmetal=round(float(true[nz].std()), 4)))
        print(f"  [band-gap] {name} {cfg_s}: full R2={full_r2:.3f} RMSE={full_rmse:.3f} | "
              f"non-metal (n={int(nz.sum()):,}) R2={sub_r2:.3f} RMSE={sub_rmse:.3f}", flush=True)
    with open(out_dir / "band_gap_subset.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  [band-gap] wrote {out_dir/'band_gap_subset.csv'}", flush=True)


def run_knn_predictor(orb_dir, uma_dir, orb_rows, uma_rows, out_dir, seed, k=10):
    """k-NN used as a PREDICTOR, not as a purity statistic, so retrieval and the probe are
    scored in the same units on the same split: majority vote for classification, neighbour
    mean for regression. Run under Euclidean and cosine distance."""
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.metrics import f1_score, r2_score
    out_dir.mkdir(parents=True, exist_ok=True)
    TARGETS = ["crystal_system", "metal_insulator", "magnetic_label",
               "formation_energy/atom", "energy_above_hull", "band_gap",
               "density", "bulk_modulus", "mag_density"]
    rows = []
    for name, mod, mdir, mrows in (("ORB", O, orb_dir, orb_rows), ("UMA", U, uma_dir, uma_rows)):
        for t in TARGETS:
            got = _cfg_matrix(mod, name, mdir, mrows, t, seed)
            if got is None:
                continue
            X, y, clf, tr, te, cfg_s = got
            for metric in ("euclidean", "cosine"):
                if clf:
                    kn = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
                    kn.fit(X[tr], y[tr])
                    sc = f1_score(y[te], kn.predict(X[te]), average="macro"); mname = "macro-F1"
                else:
                    kn = KNeighborsRegressor(n_neighbors=k, metric=metric, n_jobs=-1)
                    kn.fit(X[tr], y[tr])
                    sc = r2_score(y[te], kn.predict(X[te])); mname = "R2"
                rows.append(dict(model=name, target=t, cfg=cfg_s, metric=mname,
                                 distance=metric, knn_score=round(float(sc), 4),
                                 k=k, n_test=len(te)))
                print(f"  [knn-pred] {name:3s} {t:20s} {metric:9s} {mname}={sc:.3f}", flush=True)
    with open(out_dir / "knn_predictor.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  [knn-pred] wrote {out_dir/'knn_predictor.csv'}", flush=True)


def recompute_fdr(out_dir: Path):
    """Re-apply BH-FDR across the nine reported properties to a finished run's p-values. The bootstrap p-values
    of each target do not depend on which other targets were run, so dropping targets from the family and
    recomputing the adjusted q is exact; no refit is needed."""
    import csv as _csv
    keep = set(RELIABLE_TARGETS)
    sig_p = out_dir / "significance.csv"; pairs_p = out_dir / "embedding_pairs.csv"; full_p = out_dir / "full_metrics.csv"
    recs = [r for r in _csv.DictReader(sig_p.open()) if r["target"] in keep]
    for key, padj in [("p_uma_orb", "q_uma_orb"), ("p_orb_xl", "q_orb_xl"), ("p_uma_xl", "q_uma_xl")]:
        for r, qq in zip(recs, bh_fdr([float(r[key]) for r in recs])):
            r[padj] = f"{float(qq):.6g}"
    with sig_p.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    pr = [r for r in _csv.DictReader(pairs_p.open()) if r["target"] in keep]
    for src in ("ORB", "UMA"):
        for a, b in REP_PAIRS:
            grp = [r for r in pr if r["source"] == src and r["pair"] == f"{b}_vs_{a}"]
            for r, qq in zip(grp, bh_fdr([float(r["p"]) for r in grp])):
                r["q"] = f"{float(qq):.6g}"
    with pairs_p.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(pr[0])); w.writeheader(); w.writerows(pr)
    if full_p.exists():
        fr = [r for r in _csv.DictReader(full_p.open()) if r["target"] in keep]
        with full_p.open("w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(fr[0])); w.writeheader(); w.writerows(fr)
    print(f"BH-FDR recomputed over {len(recs)} targets ({len(keep)} properties): {sig_p}, {pairs_p}, {full_p}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Significance tests for ORB vs UMA property prediction.")
    ap.add_argument("--orb-dir", default=str(P.RESULTS / "orb/property_prediction"))
    ap.add_argument("--uma-dir", default=str(P.RESULTS / "fairchem/property_prediction"))
    ap.add_argument("--output-dir", default=str(P.RESULTS / "proppred_significance"))
    ap.add_argument("--targets", default=None, help="comma list; default = the nine properties")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--probe-family", action="store_true",
                    help="four supervised probes (linear, XGBoost, k-NN, MLP) at the test-selected configurations, same rows, with RMSE and paired bootstrap")
    ap.add_argument("--probe-peaks", action="store_true",
                    help="probe-family control on the selection: the linear probe's peak layer against XGBoost's, from the sweep")
    ap.add_argument("--val-selection", action="store_true",
                    help="run the validation-split selection control (addresses "
                         "test-set selection bias / unequal ORB-vs-UMA search spaces)")
    ap.add_argument("--composition-baseline", action="store_true",
                    help="run the composition-only (element-fraction) baseline control")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of the train split held out as validation (val-selection)")
    ap.add_argument("--val-sweep-cap", type=int, default=50000,
                    help="max inner-train rows used to RANK configs on validation "
                         "(the selected config is always refit on the full train split); 0 = no cap")
    ap.add_argument("--log-bulk-modulus", action="store_true",
                    help="re-probe bulk modulus on log10(K_VRH) for Matbench comparability")
    ap.add_argument("--probe-selectivity", action="store_true",
                    help="control-task selectivity: real-label minus permuted-label probe score")
    ap.add_argument("--comp-features", choices=["magpie", "element_fractions"],
                    default="magpie",
                    help="featuriser for the composition-only baseline; magpie is the standard "
                         "reference and supersedes raw element fractions")
    ap.add_argument("--band-gap-subset", action="store_true",
                    help="re-score band gap on the non-metal test subset (true gap > 0), since the "
                         "pooled R2 is inflated by the point mass of metals at exactly 0")
    ap.add_argument("--knn-predictor", action="store_true",
                    help="k-NN as a predictor (majority vote / neighbour mean) under Euclidean and "
                         "cosine distance, so retrieval and the probe are scored in the same units")
    ap.add_argument("--same-material", action="store_true",
                    help="same-material-leakage control on per-atom k-NN purity (categorical targets)")
    ap.add_argument("--rotation", action="store_true",
                    help="rotation control for crystal system: re-probe UMA channels + ORB under "
                         "random rotations (requires model inference via the 02 extraction modules)")
    ap.add_argument("--rotation-sub", type=int, default=1000,
                    help="number of test materials to rotate for the rotation control")
    ap.add_argument("--grouped-split", action="store_true",
                    help="re-evaluate each target's best config under a composition-grouped "
                         "(and chemical-system-grouped) split vs the random split")
    ap.add_argument("--whitening", action="store_true",
                    help="material-level k-NN purity under Euclidean vs PCA-whitened (Mahalanobis) "
                         "metric, to test whether poor retrieval is a metric artefact (§3.6)")
    ap.add_argument("--invariant-crystal", action="store_true",
                    help="bootstrap CI for crystal system on the rotation-invariant footing: UMA "
                         "best invariant channel (concat/ang_norm) vs ORB best config, with the "
                         "paired UMA-ORB difference CI (Table 2/3, §3.5)")
    ap.add_argument("--mlp-probe", action="store_true",
                    help="MLP probe at each target's selected configuration (reviewer: probe family)")
    ap.add_argument("--expt-gap", action="store_true",
                    help="band-gap probes scored against measured gaps (reviewer: PBE labels)")
    ap.add_argument("--rotation-invariance", action="store_true",
                    help="feature-level rotation test: relative change of every read-out under rotation")
    ap.add_argument("--rotinv-sub", type=int, default=300, help="materials for --rotation-invariance")
    ap.add_argument("--rotinv-n", type=int, default=5, help="rotations per material for --rotation-invariance")
    ap.add_argument("--recompute-fdr", action="store_true",
                    help="no refits: re-apply the Benjamini-Hochberg correction to the p-values already in "
                         "significance.csv and embedding_pairs.csv, restricted to the nine reported properties, "
                         "and rewrite the two files (and full_metrics.csv restricted the same way)")
    ap.add_argument("--controls-output-dir", default=str(P.RESULTS / "proppred_controls"))
    args = ap.parse_args()

    orb_dir, uma_dir = Path(args.orb_dir), Path(args.uma_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    full_rng = np.random.default_rng(args.seed + 12345)  # separate stream for the full-metric table

    if args.recompute_fdr:
        return recompute_fdr(out_dir)

    orb_rows = O.load_metrics(orb_dir / "metrics.csv")
    uma_rows = U.load_metrics(uma_dir / "metrics.csv")

    if args.targets:
        targets = [O.TARGET_ALIASES.get(t.strip(), t.strip()) for t in args.targets.split(",")]
    else:
        targets = O.order_targets(O.REG_PROPS + O.CAT_PROPS)

    # Review-response controls short-circuit the default bootstrap pipeline.
    _controls = (args.val_selection or args.composition_baseline or args.log_bulk_modulus
                 or args.probe_selectivity or args.same_material or args.rotation
                 or args.band_gap_subset or args.knn_predictor
                 or args.grouped_split or args.whitening or args.invariant_crystal
                 or args.mlp_probe or args.expt_gap or args.rotation_invariance
                 or args.probe_peaks or args.probe_family)
    if _controls:
        ctl_targets = targets if args.targets else RELIABLE_TARGETS
        ctl_dir = Path(args.controls_output_dir)
        if args.composition_baseline:
            run_composition_baseline(orb_rows, uma_rows, orb_dir, uma_dir,
                                     ctl_dir / "composition_baseline", ctl_targets,
                                     args.seed, args.n_jobs, args.bootstrap, args.comp_features)
        if args.val_selection:
            run_val_selection(orb_rows, uma_rows, orb_dir, uma_dir,
                              ctl_dir / "val_selection", ctl_targets,
                              args.val_frac, args.val_sweep_cap, args.seed,
                              args.n_jobs, args.bootstrap)
        if args.log_bulk_modulus:
            run_log_bulk_modulus(orb_rows, uma_rows, orb_dir, uma_dir,
                                 ctl_dir / "log_bulk_modulus", args.seed, args.n_jobs, args.bootstrap)
        if args.probe_family:
            run_probe_family(orb_rows, uma_rows, orb_dir, uma_dir, ctl_dir / "probe_family", ctl_targets, args.seed, args.n_jobs, args.bootstrap)
        if args.probe_peaks:
            run_probe_peaks(orb_rows, uma_rows, ctl_dir / "probe_peaks", ctl_targets)
        if args.probe_selectivity:
            run_probe_selectivity(orb_rows, uma_rows, orb_dir, uma_dir,
                                  ctl_dir / "probe_selectivity", ctl_targets, args.seed, args.n_jobs)
        if args.band_gap_subset:
            run_band_gap_subset(orb_dir, uma_dir, orb_rows, uma_rows,
                                ctl_dir / "band_gap_subset", args.seed, args.n_jobs)
        if args.knn_predictor:
            run_knn_predictor(orb_dir, uma_dir, orb_rows, uma_rows,
                              ctl_dir / "knn_predictor", args.seed)
        if args.same_material:
            run_same_material(orb_dir, uma_dir, ctl_dir / "same_material", args.seed)
        if args.rotation:
            run_rotation(orb_dir, uma_dir, ctl_dir / "rotation", args.rotation_sub, args.seed, args.n_jobs)
        if args.grouped_split:
            run_grouped_split(orb_rows, uma_rows, orb_dir, uma_dir,
                              ctl_dir / "grouped_split", ctl_targets, args.seed, args.n_jobs)
        if args.whitening:
            run_whitening(orb_rows, uma_rows, orb_dir, uma_dir,
                          ctl_dir / "whitening", args.seed)
        if args.invariant_crystal:
            run_invariant_crystal(orb_rows, uma_rows, orb_dir, uma_dir,
                                  ctl_dir / "invariant_crystal", args.seed,
                                  args.n_jobs, args.bootstrap)
        if args.mlp_probe:
            run_mlp_probe(orb_rows, uma_rows, orb_dir, uma_dir, ctl_dir / "mlp_probe", ctl_targets, args.seed, args.n_jobs)
        if args.expt_gap:
            run_expt_gap(orb_rows, uma_rows, orb_dir, uma_dir, ctl_dir / "expt_gap", args.seed, args.n_jobs)
        if args.rotation_invariance:
            run_rotation_invariance(orb_dir, uma_dir, ctl_dir / "rotation_invariance", args.rotinv_sub, args.rotinv_n, args.seed, args.n_jobs)
        return

    recs = []
    pair_recs = []
    full_recs = []
    for t in targets:
        ocfg, ov = O.best_config(orb_rows, t)
        ucfg, uv = U.best_config(uma_rows, t)
        if ocfg is None or ucfg is None:
            print(f"  [skip] {t}: missing best config (orb={ocfg}, uma={ucfg})")
            continue
        o_rep, o_layer = ocfg
        u_group, u_fam, u_layer = ucfg
        print(f"  {t}: refit ORB {o_rep}/{o_layer} (m={ov:.3f}) + UMA "
              f"{u_group}/{u_fam}/{u_layer} (m={uv:.3f}) ...")
        o_mp, o_yt, o_px, o_pl, is_clf, o_base = orb_predictions(orb_dir, t, o_rep, o_layer,
                                                                 args.seed, args.n_jobs)
        u_mp, u_yt, u_px, u_pl, _, u_base = uma_predictions(uma_dir, t, u_group, u_fam, u_layer,
                                                            args.seed, args.n_jobs)

        # Full metric summary (MAE/RMSE/R2 or accuracy/macro-F1 + baselines,
        # each with a bootstrap 95% CI), best config per source.
        full_recs.append(dict(
            target=t, task="clf" if is_clf else "reg",
            orb_cfg=f"{o_rep}/{o_layer}", uma_cfg=f"{u_group}/{u_fam}/{u_layer}",
            orb=full_metrics(o_yt, o_px, o_base, is_clf, args.bootstrap, full_rng),
            uma=full_metrics(u_yt, u_px, u_base, is_clf, args.bootstrap, full_rng),
        ))

        # Marginal bootstrap CIs (each on its own full test set).
        o_pt, o_lo, o_hi = boot_ci(o_yt, o_px, is_clf, args.bootstrap, rng)
        u_pt, u_lo, u_hi = boot_ci(u_yt, u_px, is_clf, args.bootstrap, rng)

        # XGB vs linear, paired within each source (same test set / config).
        oxl = paired_boot_diff(o_yt, o_px, o_pl, is_clf, args.bootstrap, rng)
        uxl = paired_boot_diff(u_yt, u_px, u_pl, is_clf, args.bootstrap, rng)

        # Non-linearity gap (XGB - linear, primary metric, with its own 95% CI)
        # folded into the full-metrics table alongside the raw metric values.
        full_recs[-1]["orb"]["nonlin_gap"] = (oxl[0], oxl[1], oxl[2])
        full_recs[-1]["uma"]["nonlin_gap"] = (uxl[0], uxl[1], uxl[2])

        # UMA vs ORB, paired on the intersection of test materials.
        o_map = {m: k for k, m in enumerate(o_mp)}
        common = [m for m in u_mp if m in o_map]
        u_map = {m: k for k, m in enumerate(u_mp)}
        oi = np.fromiter((o_map[m] for m in common), dtype=np.int64, count=len(common))
        ui = np.fromiter((u_map[m] for m in common), dtype=np.int64, count=len(common))
        yt_o, yt_u = o_yt[oi], u_yt[ui]
        if is_clf:
            same = bool(np.array_equal(yt_o, yt_u))
        else:
            same = bool(np.allclose(yt_o, yt_u, atol=1e-6))
        if not same:
            print(f"    WARNING: {t}: true values differ on common materials "
                  f"(label-encoding order?) — using UMA truth for paired metric")
        yt = yt_u
        d_pt, d_lo, d_hi, d_p = paired_boot_diff(yt, u_px[ui], o_px[oi], is_clf,
                                                 args.bootstrap, rng)

        # Linear's own bootstrap CI (reuses o_pl/u_pl already fit above — no
        # extra refit) + full node/edge/concat rep analysis (bootstrap CI per
        # rep + pairwise paired-bootstrap diffs among them; refits only the
        # reps that don't already coincide with the overall-best config).
        ol_pt, ol_lo, ol_hi = boot_ci(o_yt, o_pl, is_clf, args.bootstrap, rng)
        ul_pt, ul_lo, ul_hi = boot_ci(u_yt, u_pl, is_clf, args.bootstrap, rng)

        orb_rep, orb_pairs = orb_rep_analysis(orb_dir, orb_rows, t, is_clf, o_rep, o_layer,
                                              o_mp, o_yt, o_px, args.bootstrap, rng,
                                              args.seed, args.n_jobs)
        uma_rep, uma_pairs = uma_rep_analysis(uma_dir, uma_rows, t, is_clf, u_group, u_fam, u_layer,
                                              u_mp, u_yt, u_px, args.bootstrap, rng,
                                              args.seed, args.n_jobs)

        for a, b in REP_PAIRS:
            if (a, b) in orb_pairs:
                d, lo, hi, p, n = orb_pairs[(a, b)]
                pair_recs.append(dict(target=t, source="ORB", pair=f"{b}_vs_{a}", a=a, b=b,
                                      delta=d, lo=lo, hi=hi, p=p, n_common=n))
            if (a, b) in uma_pairs:
                d, lo, hi, p, n = uma_pairs[(a, b)]
                pair_recs.append(dict(target=t, source="UMA", pair=f"{b}_vs_{a}", a=a, b=b,
                                      delta=d, lo=lo, hi=hi, p=p, n_common=n))

        recs.append(dict(
            target=t, task="clf" if is_clf else "reg",
            metric="macro_f1" if is_clf else "r2",
            n_test_orb=len(o_yt), n_test_uma=len(u_yt), n_common=len(common),
            orb_cfg=f"{o_rep}/{o_layer}", uma_cfg=f"{u_group}/{u_fam}/{u_layer}",
            orb=o_pt, orb_lo=o_lo, orb_hi=o_hi,
            uma=u_pt, uma_lo=u_lo, uma_hi=u_hi,
            d_uma_orb=d_pt, d_lo=d_lo, d_hi=d_hi, p_uma_orb=d_p,
            orb_xgb_lin=oxl[0], orb_xl_lo=oxl[1], orb_xl_hi=oxl[2], p_orb_xl=oxl[3],
            uma_xgb_lin=uxl[0], uma_xl_lo=uxl[1], uma_xl_hi=uxl[2], p_uma_xl=uxl[3],
            orb_linear=ol_pt, orb_linear_lo=ol_lo, orb_linear_hi=ol_hi,
            uma_linear=ul_pt, uma_linear_lo=ul_lo, uma_linear_hi=ul_hi,
            orb_node=orb_rep.get("node", {}).get("pt", float("nan")),
            orb_node_lo=orb_rep.get("node", {}).get("lo", float("nan")),
            orb_node_hi=orb_rep.get("node", {}).get("hi", float("nan")),
            orb_edge=orb_rep.get("edge", {}).get("pt", float("nan")),
            orb_edge_lo=orb_rep.get("edge", {}).get("lo", float("nan")),
            orb_edge_hi=orb_rep.get("edge", {}).get("hi", float("nan")),
            orb_concat=orb_rep.get("concat", {}).get("pt", float("nan")),
            orb_concat_lo=orb_rep.get("concat", {}).get("lo", float("nan")),
            orb_concat_hi=orb_rep.get("concat", {}).get("hi", float("nan")),
            uma_node=uma_rep.get("node", {}).get("pt", float("nan")),
            uma_node_lo=uma_rep.get("node", {}).get("lo", float("nan")),
            uma_node_hi=uma_rep.get("node", {}).get("hi", float("nan")),
            uma_edge=uma_rep.get("edge", {}).get("pt", float("nan")),
            uma_edge_lo=uma_rep.get("edge", {}).get("lo", float("nan")),
            uma_edge_hi=uma_rep.get("edge", {}).get("hi", float("nan")),
            uma_concat=uma_rep.get("concat", {}).get("pt", float("nan")),
            uma_concat_lo=uma_rep.get("concat", {}).get("lo", float("nan")),
            uma_concat_hi=uma_rep.get("concat", {}).get("hi", float("nan")),
            unreliable=bool((not is_clf) and (o_pt < 0 or u_pt < 0)),
        ))

    if not recs:
        raise SystemExit("no targets produced results")

    # BH-FDR across targets for each comparison family.
    for key, padj in [("p_uma_orb", "q_uma_orb"), ("p_orb_xl", "q_orb_xl"),
                      ("p_uma_xl", "q_uma_xl")]:
        q = bh_fdr([r[key] for r in recs])
        for r, qq in zip(recs, q):
            r[padj] = float(qq)

    # BH-FDR for the embedding-type pairwise diffs, grouped by (source, pair).
    for src in ("ORB", "UMA"):
        for a, b in REP_PAIRS:
            key = f"{b}_vs_{a}"
            grp = [r for r in pair_recs if r["source"] == src and r["pair"] == key]
            if not grp:
                continue
            q = bh_fdr([r["p"] for r in grp])
            for r, qq in zip(grp, q):
                r["q"] = float(qq)

    _write_outputs(recs, pair_recs, out_dir, args.bootstrap)
    _write_full_metrics(full_recs, out_dir, args.bootstrap)
    print("\nAll done.")


def _fmt_ci(pt, lo, hi):
    return f"{pt:.3f} [{lo:.3f},{hi:.3f}]"


def _write_outputs(recs, pair_recs, out_dir: Path, B: int):
    cols = list(recs[0].keys()) + ["effect_uma_orb"]
    for r in recs:
        r["effect_uma_orb"] = classify_effect(r["d_uma_orb"], r["d_lo"], r["d_hi"],
                                              pos="UMA", neg="ORB")
    with (out_dir / "significance.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(recs)

    if pair_recs:
        with (out_dir / "embedding_pairs.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["target", "source", "pair", "a", "b",
                                              "delta", "lo", "hi", "p", "q", "n_common"])
            w.writeheader(); w.writerows(pair_recs)

    reliable = [r for r in recs if not r["unreliable"]]
    unreliable = [r for r in recs if r["unreliable"]]

    L = [f"ORB vs UMA property prediction — significance (bootstrap B={B}, seed-42 fixed split)",
         "Paired comparisons on shared test materials; BH-FDR across targets (q).",
         "Practical-effect thresholds on |delta| (R2 / macro-F1 scale): "
         f"trivial<{EFFECT_THRESHOLDS[0]}  small<{EFFECT_THRESHOLDS[1]}  "
         f"moderate<{EFFECT_THRESHOLDS[2]}  else large.",
         "CAUTION: n_test ~ 31k for dense targets — read effect size + CI, a tiny Δ can carry",
         "p<1e-5 and still be scientifically meaningless.",
         "See significance_tables.md for the same results as markdown tables (exact q, not stars).", "",
         "=" * 108,
         "RELIABLE TARGETS — UMA vs ORB  (Δ = UMA − ORB, best XGBoost config each)",
         "=" * 108,
         f"{'target':22s} {'metric':8s} {'ORB (95% CI)':20s} {'UMA (95% CI)':20s} "
         f"{'Δ UMA-ORB (95% CI)':24s} {'effect':24s}"]
    for r in reliable:
        L.append(f"{r['target']:22s} {r['metric']:8s} "
                 f"{_fmt_ci(r['orb'], r['orb_lo'], r['orb_hi']):20s} "
                 f"{_fmt_ci(r['uma'], r['uma_lo'], r['uma_hi']):20s} "
                 f"{_fmt_ci(r['d_uma_orb'], r['d_lo'], r['d_hi']):24s} "
                 f"{r['effect_uma_orb']:24s}")

    if unreliable:
        L += ["", "=" * 108,
              "FAILED / UNRELIABLE FITS (best R2 < 0 for ORB and/or UMA — CI not meaningful,",
              "excluded from all comparison plots and from the effect-size judgement above)",
              "=" * 108,
              f"{'target':18s} {'metric':8s} {'ORB (95% CI)':22s} {'UMA (95% CI)':22s} "
              f"{'Δ UMA-ORB (95% CI)':26s}"]
        for r in unreliable:
            L.append(f"{r['target']:18s} {r['metric']:8s} "
                     f"{_fmt_ci(r['orb'], r['orb_lo'], r['orb_hi']):22s} "
                     f"{_fmt_ci(r['uma'], r['uma_lo'], r['uma_hi']):22s} "
                     f"{_fmt_ci(r['d_uma_orb'], r['d_lo'], r['d_hi']):26s}")

    L += ["", "=" * 108,
          "XGBoost vs linear  (Δ = XGB − linear at each source's best config; nonlinearity gain)",
          "=" * 108,
          f"{'target':22s} {'metric':8s} {'ORB Δ (95% CI)':22s} {'ORB effect':20s} "
          f"{'UMA Δ (95% CI)':22s} {'UMA effect':20s}"]
    for r in reliable:
        oe = classify_effect(r["orb_xgb_lin"], r["orb_xl_lo"], r["orb_xl_hi"], pos="XGB", neg="linear")
        ue = classify_effect(r["uma_xgb_lin"], r["uma_xl_lo"], r["uma_xl_hi"], pos="XGB", neg="linear")
        L.append(f"{r['target']:22s} {r['metric']:8s} "
                 f"{_fmt_ci(r['orb_xgb_lin'], r['orb_xl_lo'], r['orb_xl_hi']):22s} {oe:20s} "
                 f"{_fmt_ci(r['uma_xgb_lin'], r['uma_xl_lo'], r['uma_xl_hi']):22s} {ue:20s}")

    (out_dir / "significance_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))

    _write_markdown_tables(reliable, unreliable, pair_recs, out_dir)

    _forest_uma_orb(reliable, out_dir)
    _table_uma_orb(reliable, out_dir)
    _forest_xgb_linear(reliable, out_dir)
    _table_xgb_linear(reliable, out_dir)
    if unreliable:
        _table_unreliable(unreliable, out_dir)
        _forest_unreliable(unreliable, out_dir)
    _scatter_uma_vs_orb(reliable, out_dir)
    _scatter_xgb_vs_linear(reliable, out_dir)
    _scatter_embedding_types(reliable, pair_recs, out_dir)
    print(f"\n  outputs -> {out_dir}")


def _fmt_pt(pt, lo, hi, prec=3):
    if lo == hi:                       # constant (e.g. K classes) → no CI
        return f"{pt:.{prec}g}"
    return f"{pt:.{prec}g} [{lo:.{prec}g}, {hi:.{prec}g}]"


def _render_metric_table(frecs, cols, title, path: Path):
    """Rendered value table: two rows (ORB, UMA) per property, one column per
    metric, each cell = point estimate with its 95% bootstrap CI underneath."""
    rows, shade = [], []
    for fr in frecs:
        for model in ("ORB", "UMA"):
            d = fr[model.lower()]
            cells = [fr["target"], model]
            for key, _ in cols:
                pt, lo, hi = d[key]
                cells.append(f"{pt:.3g}" if lo == hi else f"{pt:.3g}\n[{lo:.3g}, {hi:.3g}]")
            rows.append(cells); shade.append(model == "UMA")
    ncol = 2 + len(cols)
    fig, ax = plt.subplots(figsize=(2.15 * ncol, 0.62 * len(rows) + 1.5)); ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["target", "model"] + [lbl for _, lbl in cols],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 2.0)
    for i, sh in enumerate(shade):
        if sh:
            for c in range(ncol):
                tbl[(i + 1, c)].set_facecolor("#f2f2f2")
    ax.set_title(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_full_metrics(full_recs, out_dir: Path, B: int):
    """Full per-property metric summary for both models' best XGBoost config:
    regression (MAE / RMSE / R2 + train-mean baselines) and classification
    (accuracy / macro-F1 + majority-class baselines), each with a 95% bootstrap
    CI, plus the XGB-vs-linear non-linearity gap (paired bootstrap Δ, same
    metric, with its own 95% CI). Written as a long CSV, two rendered PNG
    tables, and a markdown file."""
    if not full_recs:
        return
    with (out_dir / "full_metrics.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "task", "model", "config", "metric", "point", "ci_lo", "ci_hi"])
        for fr in full_recs:
            for model in ("orb", "uma"):
                cfg = fr[f"{model}_cfg"]
                for nm, (pt, lo, hi) in fr[model].items():
                    w.writerow([fr["target"], fr["task"], model.upper(), cfg, nm,
                                f"{pt:.6g}", f"{lo:.6g}", f"{hi:.6g}"])

    reg = [fr for fr in full_recs if fr["task"] == "reg"]
    clf = [fr for fr in full_recs if fr["task"] == "clf"]
    if reg:
        _render_metric_table(reg, REG_METRIC_COLS,
                             f"Regression — full metric summary (point [95% bootstrap CI], B={B})\n"
                             "baseline = predict train-set mean   |   σ = target std,  "
                             "RMSE/σ = √(1−R²)  (low σ ⇒ small errors still give modest R²; "
                             "high σ ⇒ high R² can hide large absolute errors)",
                             out_dir / "full_metrics_regression.png")
    if clf:
        _render_metric_table(clf, CLF_METRIC_COLS,
                             f"Classification — full metric summary (point [95% bootstrap CI], B={B})\n"
                             "baseline = predict train majority class;  K = n classes sets the chance level;  "
                             "macro-F1 skill = (F1−F1_base)/(1−F1_base), chance-corrected (0 = baseline, 1 = perfect)",
                             out_dir / "full_metrics_classification.png")

    L = ["# Property-prediction — full metric summary", "",
         f"Point estimate with 95% bootstrap CI [lo, hi] (B={B}) on the fixed seed-42 test split, "
         "best XGBoost config per property. Baseline = predict the training-set mean (regression) "
         "or majority class (classification).", "",
         "For regression, σ is the target's standard deviation (spread on the test set) and "
         "RMSE/σ = √(1−R²). A small σ means even a physically small error yields only a modest R² "
         "(e.g. mag_density); a large σ can give a high R² despite practically large absolute "
         "errors (e.g. bulk_modulus, band_gap).", "",
         "For classification, macro-F1 is **not** variance-normalised, so the σ caveat does not "
         "transfer directly. The analogous caveat is that macro-F1's chance level depends on the "
         "number of classes K and the class balance: the majority-class baseline macro-F1 falls "
         "toward ~0 as K grows (e.g. crystal_system K=7, baseline macro-F1 ≈ 0.05), and macro-F1 "
         "can diverge from accuracy under imbalance (a rare class the model fails drags macro-F1 "
         "down while accuracy stays high). Read macro-F1 against its baseline; macro-F1 skill = "
         "(F1−F1_base)/(1−F1_base) is the chance-corrected analog of R² (0 = majority baseline, "
         "1 = perfect).", ""]
    if reg:
        L += ["## Regression", "",
              "| Target | Model | Config | " + " | ".join(l for _, l in REG_METRIC_COLS) + " |",
              "|" + "---|" * (3 + len(REG_METRIC_COLS))]
        for fr in reg:
            for model in ("orb", "uma"):
                d = fr[model]
                cells = " | ".join(_fmt_pt(*d[k]) for k, _ in REG_METRIC_COLS)
                L.append(f"| {fr['target']} | {model.upper()} | {fr[model + '_cfg']} | {cells} |")
        L.append("")
    if clf:
        L += ["## Classification", "",
              "| Target | Model | Config | " + " | ".join(l for _, l in CLF_METRIC_COLS) + " |",
              "|" + "---|" * (3 + len(CLF_METRIC_COLS))]
        for fr in clf:
            for model in ("orb", "uma"):
                d = fr[model]
                cells = " | ".join(_fmt_pt(*d[k]) for k, _ in CLF_METRIC_COLS)
                L.append(f"| {fr['target']} | {model.upper()} | {fr[model + '_cfg']} | {cells} |")
        L.append("")
    (out_dir / "full_metrics.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  [full-metrics] {len(full_recs)} properties -> full_metrics.csv / .md / "
          "full_metrics_regression.png / full_metrics_classification.png")


def _write_markdown_tables(reliable, unreliable, pair_recs, out_dir: Path):
    """Markdown tables mirroring significance_summary.txt, with the actual
    q-value in the Verdict column (never asterisks) and Δ always shown with
    its full 95% CI, never just the point estimate."""
    L = ["# Property-prediction significance — summary tables", "",
         "Bootstrap on the fixed seed-42 test split (or paired-bootstrap of the difference); "
         "BH-FDR-corrected q within each comparison family. Every Δ is reported with its 95% CI — "
         "never a bare point value. Verdict states the practical effect size (see thresholds in "
         "significance_summary.txt) plus the exact q (not stars).", ""]

    L += ["## A. UMA vs ORB", "",
         "| Target | ORB | UMA | Δ (95% CI) | Verdict |",
         "|---|---|---|---|---|"]
    for r in reliable:
        eff = r["effect_uma_orb"]
        L.append(f"| {r['target']} | {r['orb']:.3f} | {r['uma']:.3f} | "
                f"{r['d_uma_orb']:+.3f} [{r['d_lo']:+.3f}, {r['d_hi']:+.3f}] | "
                f"{_verdict_cell(eff, r['q_uma_orb'])} |")
    if unreliable:
        names = " / ".join(r["target"] for r in unreliable)
        L.append(f"| {names} | — | — | — | UNRELIABLE (best R2<0) |")
    L.append("")

    L += ["## B. XGBoost vs linear", "",
         "| Target | ORB XGB | ORB linear | ORB Δ (95% CI) | ORB Verdict | "
         "UMA XGB | UMA linear | UMA Δ (95% CI) | UMA Verdict |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in reliable:
        oe = classify_effect(r["orb_xgb_lin"], r["orb_xl_lo"], r["orb_xl_hi"], pos="XGB", neg="linear")
        ue = classify_effect(r["uma_xgb_lin"], r["uma_xl_lo"], r["uma_xl_hi"], pos="XGB", neg="linear")
        L.append(f"| {r['target']} | {r['orb']:.3f} | {r['orb_linear']:.3f} | "
                f"{r['orb_xgb_lin']:+.3f} [{r['orb_xl_lo']:+.3f}, {r['orb_xl_hi']:+.3f}] | "
                f"{_verdict_cell(oe, r['q_orb_xl'])} | "
                f"{r['uma']:.3f} | {r['uma_linear']:.3f} | "
                f"{r['uma_xgb_lin']:+.3f} [{r['uma_xl_lo']:+.3f}, {r['uma_xl_hi']:+.3f}] | "
                f"{_verdict_cell(ue, r['q_uma_xl'])} |")
    L.append("")

    if pair_recs:
        L.append("## C. Embedding representation comparisons "
                 "(concat / node / edge, best layer, XGBoost)")
        L.append("")
        order = [r["target"] for r in reliable]
        for a, b in REP_PAIRS:
            pair_key = f"{b}_vs_{a}"
            here = {r["target"] for r in pair_recs if r["pair"] == pair_key}
            targets_here = [t for t in order if t in here]
            if not targets_here:
                continue
            L += [f"### {b} vs {a}", "",
                 "| Target | ORB Δ (95% CI) | ORB Verdict | UMA Δ (95% CI) | UMA Verdict |",
                 "|---|---|---|---|---|"]
            for t in targets_here:
                orow = next((r for r in pair_recs if r["pair"] == pair_key
                            and r["source"] == "ORB" and r["target"] == t), None)
                urow = next((r for r in pair_recs if r["pair"] == pair_key
                            and r["source"] == "UMA" and r["target"] == t), None)
                if orow:
                    oe = classify_effect(orow["delta"], orow["lo"], orow["hi"], pos=b, neg=a)
                    ocell = (f"{orow['delta']:+.3f} [{orow['lo']:+.3f}, {orow['hi']:+.3f}] | "
                            f"{_verdict_cell(oe, orow['q'])}")
                else:
                    ocell = "n/a | n/a"
                if urow:
                    ue = classify_effect(urow["delta"], urow["lo"], urow["hi"], pos=b, neg=a)
                    ucell = (f"{urow['delta']:+.3f} [{urow['lo']:+.3f}, {urow['hi']:+.3f}] | "
                            f"{_verdict_cell(ue, urow['q'])}")
                else:
                    ucell = "n/a | n/a"
                L.append(f"| {t} | {ocell} | {ucell} |")
            L.append("")

    (out_dir / "significance_tables.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  [tables] -> {out_dir/'significance_tables.md'}")


def _forest_uma_orb(reliable, out_dir: Path):
    """Reliable targets only; sensible fixed x-range (widened only if data
    would clip); effect-size label instead of significance stars."""
    rr = list(reversed(reliable))
    y = np.arange(len(rr))
    xlim = (_adaptive_xlim([r["d_lo"] for r in rr], [r["d_hi"] for r in rr], -0.05, 0.3)[0], 0.3)
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rr) + 1.8))
    for i, r in enumerate(rr):
        eff = r["effect_uma_orb"]
        c = _effect_color(eff, "UMA")
        ax.plot([r["d_lo"], r["d_hi"]], [i, i], color=c, lw=2.5)
        ax.plot(r["d_uma_orb"], i, "o", color=c, ms=6)
        ax.text(xlim[1], i, f"  {r['d_uma_orb']:+.3f} [{r['d_lo']:.3f},{r['d_hi']:.3f}]  {eff}",
                va="center", fontsize=7.5)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(*xlim)
    ax.set_yticks(y); ax.set_yticklabels([f"{r['target']} ({r['metric']})" for r in rr], fontsize=8)
    ax.set_xlabel("Δ metric  =  UMA − ORB  (best XGBoost config each; 95% bootstrap CI)")
    import matplotlib.lines as ml
    handles = [ml.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_UMA,
                        label="UMA advantage", ms=8),
              ml.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ORB,
                        label="ORB advantage", ms=8),
              ml.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_TIE,
                        label="trivial / tie", ms=8)]
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(out_dir / "forest_uma_vs_orb.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _table_uma_orb(reliable, out_dir: Path):
    """Companion value table for forest_uma_vs_orb.png — ORB and UMA headline
    scores (R2 / macro-F1) with CIs, the paired Δ (UMA−ORB) with CI, and the
    practical-effect label. The effect cell is shaded by the winner (pale
    goldenrod = UMA, pale turquoise = ORB, grey = trivial / tie)."""
    table = []
    for r in reliable:
        table.append([
            f"{r['target']} ({r['metric']})",
            _fmt_ci(r["orb"], r["orb_lo"], r["orb_hi"]),
            _fmt_ci(r["uma"], r["uma_lo"], r["uma_hi"]),
            _fmt_ci(r["d_uma_orb"], r["d_lo"], r["d_hi"]),
            r["effect_uma_orb"],
        ])
    fig, ax = plt.subplots(figsize=(13, 0.5 * len(table) + 1.6)); ax.axis("off")
    tbl = ax.table(cellText=table,
                   colLabels=["target (metric)", "ORB (95% CI)", "UMA (95% CI)",
                              "Δ UMA-ORB (95% CI)", "effect"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.6)
    for i, r in enumerate(reliable):
        fc = _effect_color(r["effect_uma_orb"], "UMA",
                           pos_color="#f5e6c8", neg_color="#cdeeea", neutral="#eeeeee")
        tbl[(i + 1, 4)].set_facecolor(fc)
    ax.set_title("UMA vs ORB per property (values for forest_uma_vs_orb.png)\n"
                 "Δ = UMA − ORB, best XGBoost config each; effect = practical size on the "
                 "R2 / macro-F1 scale.")
    fig.savefig(out_dir / "uma_vs_orb_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [uma-orb table] -> {out_dir/'uma_vs_orb_table.png'}")


def _forest_xgb_linear(reliable, out_dir: Path):
    """X-axis is hard-capped at CAP so one blown-out CI (UMA mag_density, whose
    weak/unstable LINEAR baseline gives Δ up to ~0.87) can't squash every other
    property. Whiskers past the cap are clipped and marked with a '>' plus their
    true upper value; the full numbers live in xgb_vs_linear_table.png."""
    CAP = 0.5
    rr = list(reversed(reliable))
    y = np.arange(len(rr))
    los = [r["orb_xl_lo"] for r in rr] + [r["uma_xl_lo"] for r in rr]
    xlim = (_adaptive_xlim(los, [0.0], -0.05, CAP)[0], CAP)
    fig, ax = plt.subplots(figsize=(11.5, 0.55 * len(rr) + 2.2))
    for i, r in enumerate(rr):
        for key, col, mk, off in (("orb", C_ORB, "o", -0.15), ("uma", C_UMA, "s", 0.15)):
            lo, hi, pt = r[f"{key}_xl_lo"], r[f"{key}_xl_hi"], r[f"{key}_xgb_lin"]
            ax.plot([lo, min(hi, CAP)], [i + off, i + off], color=col, lw=2)
            if pt <= CAP:
                ax.plot(pt, i + off, mk, color=col, ms=5)
            if hi > CAP:
                ax.plot(CAP, i + off, ">", color=col, ms=7)
                ax.annotate(f"{hi:.2f}", (CAP, i + off), color=col, fontsize=6.5,
                            va="center", ha="right", xytext=(-4, 0), textcoords="offset points")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(*xlim)
    ax.set_yticks(y); ax.set_yticklabels([f"{r['target']} ({r['metric']})" for r in rr], fontsize=8)
    ax.set_xlabel("Δ metric  =  XGBoost − linear  (95% bootstrap CI)")
    ax.set_title("Nonlinearity gain per property")
    ax.grid(axis="x", alpha=0.3)
    import matplotlib.lines as ml
    handles = [ml.Line2D([0], [0], color=C_ORB, marker="o", ls="-", label="ORB"),
               ml.Line2D([0], [0], color=C_UMA, marker="s", ls="-", label="UMA"),
               ml.Line2D([0], [0], color="k", marker=">", ls="none", label=f"CI extends past {CAP}")]
    fig.legend(handles=handles, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, -0.08,
            "UMA density & mag_density: the large Δ and wide CI come from an UNSTABLE LINEAR baseline "
            "(its R2 has a heavy low tail), not from XGBoost — the mag_density UMA whisker is clipped at "
            "the axis limit (true upper bound labelled). Exact values: xgb_vs_linear_table.png.",
            ha="center", fontsize=7.3, style="italic")
    fig.savefig(out_dir / "forest_xgb_vs_linear.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _table_xgb_linear(reliable, out_dir: Path):
    """Companion value table for forest_xgb_vs_linear.png — the XGB and linear
    scores plus the XGB-linear gain (with CI) for both sources, so the clipped
    whiskers can be read exactly. Rows where UMA's linear baseline is unstable
    (wide linear CI) are shaded to flag that the gain is a linear-side artefact."""
    table = []
    for r in reliable:
        table.append([
            r["target"],
            f"{r['orb']:.3f}", f"{r['orb_linear']:.3f}",
            _fmt_ci(r["orb_xgb_lin"], r["orb_xl_lo"], r["orb_xl_hi"]),
            f"{r['uma']:.3f}", f"{r['uma_linear']:.3f}",
            _fmt_ci(r["uma_xgb_lin"], r["uma_xl_lo"], r["uma_xl_hi"]),
        ])
    fig, ax = plt.subplots(figsize=(15, 0.5 * len(table) + 1.8)); ax.axis("off")
    tbl = ax.table(cellText=table,
                   colLabels=["target", "ORB XGB", "ORB linear", "ORB XGB-lin Δ (95% CI)",
                              "UMA XGB", "UMA linear", "UMA XGB-lin Δ (95% CI)"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.6)
    for i, r in enumerate(reliable):
        if (r["uma_linear_hi"] - r["uma_linear_lo"]) > 0.1:  # unstable linear fit
            for c in (5, 6):
                tbl[(i + 1, c)].set_facecolor("#fdebd0")
    ax.set_title("Nonlinearity gain — XGBoost vs linear (values for forest_xgb_vs_linear.png)\n"
                 "Shaded UMA cells = unstable linear baseline (wide linear CI); the large Δ there is a "
                 "linear-side artefact, not an XGBoost gain.")
    fig.savefig(out_dir / "xgb_vs_linear_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [xgb-lin table] -> {out_dir/'xgb_vs_linear_table.png'}")


def _table_unreliable(unreliable, out_dir: Path):
    """Targets whose best R2 < 0, so a
    bootstrap CI around the metric is not meaningful. Reported plainly, with no
    effect-size judgement, kept out of every other figure."""
    table = []
    for r in unreliable:
        table.append([
            r["target"],
            _fmt_ci(r["orb"], r["orb_lo"], r["orb_hi"]),
            _fmt_ci(r["uma"], r["uma_lo"], r["uma_hi"]),
            _fmt_ci(r["d_uma_orb"], r["d_lo"], r["d_hi"]),
            _fmt_ci(r["orb_xgb_lin"], r["orb_xl_lo"], r["orb_xl_hi"]),
            _fmt_ci(r["uma_xgb_lin"], r["uma_xl_lo"], r["uma_xl_hi"]),
        ])
    fig, ax = plt.subplots(figsize=(15, 0.5 * len(table) + 1.6)); ax.axis("off")
    tbl = ax.table(cellText=table,
                   colLabels=["target", "ORB R2 (95% CI)", "UMA R2 (95% CI)",
                              "Δ UMA-ORB (95% CI)", "ORB XGB-lin Δ (95% CI)",
                              "UMA XGB-lin Δ (95% CI)"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.6)
    ax.set_title("FAILED / UNRELIABLE FITS — best R2 < 0\n"
                 "Bootstrap CIs shown for the record only; not a meaningful model comparison.")
    fig.savefig(out_dir / "unreliable_targets.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [unreliable] {len(unreliable)} targets -> {out_dir/'unreliable_targets.png'}")


def _forest_unreliable(unreliable, out_dir: Path):
    """Same Δ-forest layout as forest_uma_vs_orb, kept as a separate figure so
    the unreliable targets are reported consistently (alongside the table). All
    points are neutral (lightgray): with best R2 < 0 the effect-size judgement
    is not meaningful, so no advantage colouring is applied."""
    rr = list(reversed(unreliable))
    y = np.arange(len(rr))
    xlim = _adaptive_xlim([r["d_lo"] for r in rr], [r["d_hi"] for r in rr], -0.05, 0.3)
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(rr) + 1.8))
    for i, r in enumerate(rr):
        ax.plot([r["d_lo"], r["d_hi"]], [i, i], color=C_TIE, lw=2.5)
        ax.plot(r["d_uma_orb"], i, "o", color=C_TIE, ms=6)
        ax.text(xlim[1], i, f"  {r['d_uma_orb']:+.3f} [{r['d_lo']:.3f},{r['d_hi']:.3f}]",
                va="center", fontsize=7.5)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(*xlim)
    ax.set_yticks(y); ax.set_yticklabels([f"{r['target']} ({r['metric']})" for r in rr], fontsize=8)
    ax.set_xlabel("Δ metric  =  UMA − ORB  (best XGBoost config each; 95% bootstrap CI)")
    ax.set_title("UMA vs ORB per property — UNRELIABLE fits (best R2 < 0)")
    fig.text(0.5, -0.06, "Best R2 < 0 for ORB and/or UMA: the model does no better than "
            "predicting the mean, so the bootstrap CI around it is not a meaningful comparison. "
            "Shown for the record only.", ha="center", fontsize=7.5, style="italic")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(out_dir / "unreliable_targets_forest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [unreliable-forest] {len(unreliable)} targets -> "
          f"{out_dir/'unreliable_targets_forest.png'}")


# Fixed 9-colour dark/pale palette (colour-blind-friendly, one dark+pale pair per
# property). Labels 1-8 are as specified; label 8's dark tone is grey rather than
# black (unlike labels 1-7, low-saturation so it reads as "muted", not "off").
# Label 9 (magenta) was added to cover the 9th reliable property — same dark/pale
# construction, chosen to sit visually apart from the red (5) and purple (4) hues.
PROPERTY_PALETTE = [
    ("#0057B8", "#DCEEFF"),  # 1
    ("#E66100", "#FFF0DC"),  # 2
    ("#00843D", "#DDF7E3"),  # 3
    ("#6A00A8", "#F0E1FF"),  # 4
    ("#C0002B", "#FFE1E8"),  # 5
    ("#007C89", "#DDF8F8"),  # 6
    ("#8A5A00", "#FFF4D6"),  # 7
    ("#595959", "#F0F0F0"),  # 8 (grey, not black)
    ("#A8006A", "#FFE0F0"),  # 9 (added)
]


def _property_palette(reliable):
    """Stable (dark, pale) colour pair per property, in the given order. Dark =
    ORB / primary series; pale = UMA / secondary series, so the two models are
    distinguishable by brightness without any text labels."""
    return {r["target"]: PROPERTY_PALETTE[i % len(PROPERTY_PALETTE)] for i, r in enumerate(reliable)}


def _prop_legend_handles(reliable, palette):
    import matplotlib.lines as ml
    return [ml.Line2D([0], [0], marker="o", ls="none", markerfacecolor=palette[r["target"]][0],
                      markeredgecolor="none", label=r["target"], ms=8) for r in reliable]


def _struct_handles():
    """Parity + CI key only. The bright(ORB)/pale(UMA) distinction is stated as
    plain text in the caption below each plot rather than a colour swatch (a
    grey marker would misrepresent it, since no property is actually grey)."""
    import matplotlib.lines as ml
    return [ml.Line2D([0], [0], color="k", ls="--", lw=1, label="y = x (parity)"),
            ml.Line2D([0], [0], color="0.3", lw=1.4, label="95% bootstrap CI")]


def _diagonal_scatter(ax, xs, ys, colors, labels, xlabel, ylabel, xerr=None, yerr=None,
                      pad=0.05, annotate=False):
    """Diagonal parity scatter. Whiskers (CI) are drawn AFTER (on top of) the
    dots so they are never hidden by the marker fill. Point labels are opt-in
    (annotate=True); by default points are colour-coded by property instead."""
    lo = min(min(xs), min(ys)) - pad
    hi = max(max(xs), max(ys)) + pad
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=1)
    ax.scatter(xs, ys, c=colors, s=50, zorder=2)
    if xerr is not None or yerr is not None:
        for i, (x, y) in enumerate(zip(xs, ys)):
            if xerr is not None and np.isfinite(xerr[i][0]):
                ax.plot([xerr[i][0], xerr[i][1]], [y, y], color="0.3", lw=1.4, zorder=3)
            if yerr is not None and np.isfinite(yerr[i][0]):
                ax.plot([x, x], [yerr[i][0], yerr[i][1]], color="0.3", lw=1.4, zorder=3)
    if annotate:
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), fontsize=6.5, xytext=(4, 3),
                        textcoords="offset points", zorder=4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3); ax.set_aspect("equal", adjustable="box")


def _add_side_text(ax, lines, fontsize=7.3):
    ax.text(1.06, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=fontsize,
           va="top", ha="left", family="monospace", clip_on=False)


def _scatter_uma_vs_orb(reliable, out_dir: Path):
    """Absolute-score view to complement the Δ forest plot: the same Δ means
    something different near ceiling (both R2~0.9) vs. near floor (both
    R2~0.1). Points above the diagonal = UMA better; below = ORB better."""
    palette = _property_palette(reliable)
    xs = [r["orb"] for r in reliable]; ys = [r["uma"] for r in reliable]
    colors = [palette[r["target"]][0] for r in reliable]
    xerr = [(r["orb_lo"], r["orb_hi"]) for r in reliable]
    yerr = [(r["uma_lo"], r["uma_hi"]) for r in reliable]
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    _diagonal_scatter(ax, xs, ys, colors, None,
                      "ORB score (R2 / macro-F1)", "UMA score (R2 / macro-F1)",
                      xerr=xerr, yerr=yerr)
    ax.set_title("UMA vs ORB — paired absolute scores\n(above diagonal = UMA better)")
    leg1 = ax.legend(handles=_struct_handles(), fontsize=8, loc="upper left", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=_prop_legend_handles(reliable, palette), title="property",
              fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.text(0.5, -0.02, "One point per property (colour). Exact Δ (UMA−ORB), CI and "
             "effect size per property: uma_vs_orb_table.png.",
             ha="center", fontsize=7.5, style="italic")
    fig.savefig(out_dir / "scatter_uma_vs_orb.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _scatter_xgb_vs_linear(reliable, out_dir: Path):
    palette = _property_palette(reliable)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    xs = [r["orb_linear"] for r in reliable] + [r["uma_linear"] for r in reliable]
    ys = [r["orb"] for r in reliable] + [r["uma"] for r in reliable]
    colors = ([palette[r["target"]][0] for r in reliable]
             + [palette[r["target"]][1] for r in reliable])
    xerr = ([(r["orb_linear_lo"], r["orb_linear_hi"]) for r in reliable]
           + [(r["uma_linear_lo"], r["uma_linear_hi"]) for r in reliable])
    yerr = ([(r["orb_lo"], r["orb_hi"]) for r in reliable]
           + [(r["uma_lo"], r["uma_hi"]) for r in reliable])
    _diagonal_scatter(ax, xs, ys, colors, None,
                      "linear score (R2 / macro-F1)", "XGBoost score (R2 / macro-F1)",
                      xerr=xerr, yerr=yerr)
    ax.set_title("XGBoost vs linear — paired absolute scores")
    leg1 = ax.legend(handles=_struct_handles(), fontsize=8, loc="upper left", frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=_prop_legend_handles(reliable, palette), title="property",
              fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.text(0.5, -0.02, "Two points per property (bright = ORB, pale = UMA). Exact XGB / "
             "linear scores and XGB−linear Δ per property: xgb_vs_linear_table.png.",
             ha="center", fontsize=7.5, style="italic")
    fig.savefig(out_dir / "scatter_xgb_vs_linear.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _scatter_embedding_types(reliable, pair_recs, out_dir: Path):
    """Pairwise node/edge/concat comparison, both sources overlaid. All three
    pairs (node-edge, node-concat, edge-concat) so the reader can judge whether
    concatenation genuinely helps or just adds edge-only information. Exact Δ,
    CI, p, q for every target x pair x source are in significance_tables.md
    (Table C) — too many numbers (up to 9 targets x 3 pairs x 2 sources) to
    annotate legibly on the plot itself."""
    palette = _property_palette(reliable)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, (a, b) in zip(axes, REP_PAIRS):
        xs, ys, colors, xerr, yerr = [], [], [], [], []
        for r in reliable:
            ox, oy = r[f"orb_{a}"], r[f"orb_{b}"]
            ux, uy = r[f"uma_{a}"], r[f"uma_{b}"]
            if np.isfinite(ox) and np.isfinite(oy):
                xs.append(ox); ys.append(oy); colors.append(palette[r["target"]][0])
                xerr.append((r["orb_" + a + "_lo"], r["orb_" + a + "_hi"]))
                yerr.append((r["orb_" + b + "_lo"], r["orb_" + b + "_hi"]))
            if np.isfinite(ux) and np.isfinite(uy):
                xs.append(ux); ys.append(uy); colors.append(palette[r["target"]][1])
                xerr.append((r["uma_" + a + "_lo"], r["uma_" + a + "_hi"]))
                yerr.append((r["uma_" + b + "_lo"], r["uma_" + b + "_hi"]))
        _diagonal_scatter(ax, xs, ys, colors, None, f"{a} score", f"{b} score",
                          xerr=xerr, yerr=yerr)
        ax.set_title(f"{b} vs {a}")
    # All legend content (parity/CI key + property key) lives in one legend at
    # the bottom of the whole figure, not tucked in a subplot corner.
    handles = _struct_handles() + _prop_legend_handles(reliable, palette)
    fig.legend(handles=handles, fontsize=8,
               ncol=min(len(handles), 6), loc="lower center",
               bbox_to_anchor=(0.5, -0.14), frameon=False)
    fig.text(0.5, -0.22, "Above diagonal = the y-axis representation is better. "
             "Colour = property; bright = ORB, pale = UMA. "
             "Exact Δ / CI / p / q per target: significance_tables.md (Table C).",
             ha="center", fontsize=8, style="italic")
    fig.suptitle("Embedding representation comparison — node vs edge vs concat (best layer, XGBoost)")
    fig.savefig(out_dir / "scatter_embedding_types.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Review-response controls
# ===========================================================================
# Two analyses requested during review, both reusing the cached pooled features,
# the fixed seed-42 80/20 split, and the XGBoost/linear probe defaults (O.make_model):
#
#   (1) VALIDATION-SPLIT SELECTION.  Scripts 05/06 report, per target, the single
#       configuration that scored highest on the *test* set — over 18 configs for
#       ORB and 96 for UMA. That selection is optimistically biased and the bias
#       grows with the search space, so the small ORB/UMA gaps may be selection
#       artefacts. Here we carve a validation split out of train, select each
#       model's peak on validation, retrain it on the full train split, and score
#       it on the untouched test set — removing the winner's-curse and the
#       unequal-search-space confound.
#
#   (2) COMPOSITION-ONLY BASELINE.  Element-fraction vectors (model-independent)
#       through the identical probe pipeline and split, to measure how much of the
#       MLIP's property signal is already available from elemental composition.

from sklearn.model_selection import train_test_split

RELIABLE_TARGETS = ["crystal_system", "metal_insulator", "magnetic_label",
                    "formation_energy/atom", "energy_above_hull", "band_gap",
                    "density", "bulk_modulus", "mag_density"]


def _align_config(X, xpos, ids, y, canon_pos, le, task):
    """Slice a feature matrix + labels to `ids`, drop non-finite rows, encode y."""
    keep = [m for m in ids if m in xpos and m in canon_pos]
    rx = np.fromiter((xpos[m] for m in keep), dtype=np.int64, count=len(keep))
    ry = np.fromiter((canon_pos[m] for m in keep), dtype=np.int64, count=len(keep))
    Xs, ys, ms = X[rx], y[ry], np.array(keep, dtype=object)
    fin = np.isfinite(Xs).all(axis=1)
    Xs, ys, ms = Xs[fin], ys[fin], ms[fin]
    yenc = le.transform(ys) if task == "clf" else ys.astype(float)
    return Xs, yenc, ms


def _orb_model(orb_dir):
    """Model-level context for ORB: the 18-config search space + a feature loader
    (shared across targets, cached by layer) + a per-target label/split accessor."""
    feat_dir = orb_dir / "features"
    configs = [(rep, layer) for layer in O.ALL_LAYERS for rep in O.REPS]  # grouped by layer
    cache = {"layer": None, "d": None}

    def get_X(cfg):
        rep, layer = cfg
        if cache["layer"] != layer:
            cache["d"] = np.load(feat_dir / f"{layer}_pooled.npz", allow_pickle=True)
            cache["layer"] = layer
        d = cache["d"]
        return O.build_X(d["node"], d["edge"], rep), {m: i for i, m in enumerate(d["mp_ids"])}

    def labels(t):
        canon, ydict = O.load_canonical_targets(feat_dir, [t])
        cp = {m: i for i, m in enumerate(canon)}
        y = ydict[t]
        task = "clf" if O.is_clf(t) else "reg"
        le = LabelEncoder().fit(y[O.valid_mask(t, y)]) if task == "clf" else None
        return cp, y, task, le, orb_dir / "splits" / f"{O.safe(t)}.npz"

    return dict(name="ORB", n_configs=18, configs=configs, get_X=get_X,
                label=lambda c: f"{c[0]}/{c[1]}", labels=labels)


def _uma_model(uma_dir):
    """Model-level context for UMA: the 96-config search space + shared feature
    loader + per-target label/split accessor."""
    feat_dir = uma_dir / "features"
    configs = list(U.feature_sets(U.GROUPS))

    def get_X(cfg):
        g, fam, layer = cfg
        X, feat_ids = U.load_feature_set(g, fam, layer)
        return X, {m: i for i, m in enumerate(feat_ids)}

    def labels(t):
        canon, ydict = U.load_targets(feat_dir, [t])
        cp = {m: i for i, m in enumerate(canon)}
        y = ydict[t]
        task = "clf" if U.is_clf(t) else "reg"
        le = LabelEncoder().fit(y[U.valid_mask(t, y)]) if task == "clf" else None
        return cp, y, task, le, uma_dir / "splits" / f"{U.safe(t)}.npz"

    return dict(name="UMA", n_configs=96, configs=configs, get_X=get_X,
                label=lambda c: f"{c[0]}/{c[1]}/{c[2]}", labels=labels)


def _select_for_model(model, targets, val_frac, sweep_cap, seed, n_jobs):
    """Loop-inverted validation-split selection for all targets of one model. Each
    config's (large) feature matrix is loaded once and scored against every
    target's validation split; the ranking sweep may subsample inner-train to
    `sweep_cap` rows for speed, but the selected config is always refit on the FULL
    train split before its test score is reported. Returns {target: result}."""
    rng = np.random.default_rng(seed + 13)
    state = {}
    for t in targets:
        cp, y, task, le, splitp = model["labels"](t)
        sp = np.load(splitp, allow_pickle=True)
        train_ids = [m for m in sp["train_mp_ids"] if m in cp]
        test_ids = [m for m in sp["test_mp_ids"] if m in cp]
        strat = np.array([y[cp[m]] for m in train_ids]) if task == "clf" else None
        inner, val = train_test_split(np.array(train_ids, dtype=object),
                                      test_size=val_frac, random_state=seed, stratify=strat)
        inner = list(inner)
        if sweep_cap and len(inner) > sweep_cap:
            inner = list(rng.choice(np.array(inner, dtype=object), sweep_cap, replace=False))
        state[t] = dict(cp=cp, y=y, task=task, le=le, train=train_ids, test=test_ids,
                        inner=inner, val=list(val), best=None)
    # ranking sweep: load each config once, score every target's validation split on it
    for ci, cfg in enumerate(model["configs"], 1):
        X, xpos = model["get_X"](cfg)
        for t in targets:
            st = state[t]
            # Respect the invariant-config restriction (crystal system, UMA): the validation
            # sweep must search the same space as best_config, else it can pick the
            # orientation-dependent full tensor and produce a spurious inflation (§3.5).
            if model["name"] == "UMA" and t in U.INVARIANT_ONLY_TARGETS and cfg[1] not in U.INVARIANT_FAMILIES:
                continue
            Xi, yi, _ = _align_config(X, xpos, st["inner"], st["y"], st["cp"], st["le"], st["task"])
            Xv, yv, _ = _align_config(X, xpos, st["val"], st["y"], st["cp"], st["le"], st["task"])
            if len(yi) < 10 or len(yv) < 5:
                continue
            est = O.make_model(st["task"], "xgb", seed, n_jobs).fit(Xi, yi)
            vm = _metric(yv, est.predict(Xv), st["task"] == "clf")
            if st["best"] is None or vm > st["best"][0]:
                st["best"] = (vm, cfg)
        print(f"  [{model['name']}] swept config {ci}/{model['n_configs']}: {model['label'](cfg)}", flush=True)
    # final refit of each target's val-selected config on the FULL train split
    results = {}
    for t in targets:
        st = state[t]
        vm, cfg = st["best"]
        X, xpos = model["get_X"](cfg)
        Xtr, ytr, _ = _align_config(X, xpos, st["train"], st["y"], st["cp"], st["le"], st["task"])
        Xte, yte, te_mp = _align_config(X, xpos, st["test"], st["y"], st["cp"], st["le"], st["task"])
        est = O.make_model(st["task"], "xgb", seed, n_jobs).fit(Xtr, ytr)
        pte = est.predict(Xte)
        results[t] = dict(cfg=model["label"](cfg), val_metric=float(vm),
                          test_metric=float(_metric(yte, pte, st["task"] == "clf")),
                          te_mp=te_mp, yte=yte, pte=pte)
    return results


def _paired_common(o_mp, o_yt, o_px, u_mp, u_px, is_clf, B, rng):
    """UMA-minus-ORB paired-bootstrap diff on the intersection of test materials."""
    omap = {m: k for k, m in enumerate(o_mp)}
    umap = {m: k for k, m in enumerate(u_mp)}
    common = [m for m in u_mp if m in omap]
    oi = np.fromiter((omap[m] for m in common), dtype=np.int64, count=len(common))
    ui = np.fromiter((umap[m] for m in common), dtype=np.int64, count=len(common))
    return paired_boot_diff(o_yt[oi], u_px[ui], o_px[oi], is_clf, B, rng)


def run_val_selection(orb_rows, uma_rows, orb_dir, uma_dir, out_dir, targets,
                      val_frac, sweep_cap, seed, n_jobs, B):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 777)
    print(f"  [val-selection] ORB sweep (18 configs x {len(targets)} targets, "
          f"val_frac={val_frac}, sweep_cap={sweep_cap}) ...", flush=True)
    orb_res = _select_for_model(_orb_model(orb_dir), targets, val_frac, sweep_cap, seed, n_jobs)
    print(f"  [val-selection] UMA sweep (96 configs x {len(targets)} targets) ...", flush=True)
    uma_res = _select_for_model(_uma_model(uma_dir), targets, val_frac, sweep_cap, seed, n_jobs)
    rows, pair_rows = [], []
    for t in targets:
        o_cfg, _ = O.best_config(orb_rows, t)
        u_cfg, _ = U.best_config(uma_rows, t)
        if o_cfg is None or u_cfg is None or t not in orb_res or t not in uma_res:
            print(f"    [skip] {t}: missing config/result"); continue
        o_rep, o_layer = o_cfg
        u_g, u_f, u_l = u_cfg
        o_mp, o_yt, o_px, _, is_clf, _ = orb_predictions(orb_dir, t, o_rep, o_layer, seed, n_jobs)
        u_mp, u_yt, u_px, _, _, _ = uma_predictions(uma_dir, t, u_g, u_f, u_l, seed, n_jobs)
        o_test_sel = _metric(o_yt, o_px, is_clf)
        u_test_sel = _metric(u_yt, u_px, is_clf)
        ov, uv = orb_res[t], uma_res[t]
        rows.append(dict(model="ORB", target=t, n_configs=18, test_cfg=f"{o_rep}/{o_layer}",
                         test_score=round(o_test_sel, 4), val_cfg=ov["cfg"],
                         val_score=round(ov["test_metric"], 4), val_metric=round(ov["val_metric"], 4),
                         inflation=round(o_test_sel - ov["test_metric"], 4)))
        rows.append(dict(model="UMA", target=t, n_configs=96, test_cfg=f"{u_g}/{u_f}/{u_l}",
                         test_score=round(u_test_sel, 4), val_cfg=uv["cfg"],
                         val_score=round(uv["test_metric"], 4), val_metric=round(uv["val_metric"], 4),
                         inflation=round(u_test_sel - uv["test_metric"], 4)))
        d_test = _paired_common(o_mp, o_yt, o_px, u_mp, u_px, is_clf, B, rng)
        d_val = _paired_common(ov["te_mp"], ov["yte"], ov["pte"], uv["te_mp"], uv["pte"], is_clf, B, rng)
        pair_rows.append(dict(target=t, is_clf=is_clf,
                              diff_test=round(d_test[0], 4), diff_test_lo=round(d_test[1], 4),
                              diff_test_hi=round(d_test[2], 4), p_test=round(d_test[3], 4),
                              diff_val=round(d_val[0], 4), diff_val_lo=round(d_val[1], 4),
                              diff_val_hi=round(d_val[2], 4), p_val=round(d_val[3], 4)))
        print(f"    {t}: ORB {o_test_sel:.3f}->{ov['test_metric']:.3f} ({ov['cfg']})  "
              f"UMA {u_test_sel:.3f}->{uv['test_metric']:.3f} ({uv['cfg']})  "
              f"UMA-ORB test={d_test[0]:+.3f} val={d_val[0]:+.3f}", flush=True)
    _write_val_selection(rows, pair_rows, out_dir, val_frac, B)


def _load_formula_map(index_path=str(P.STRUCTURES / "index.jsonl")):
    import json
    fm = {}
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                fm[r["mp_id"]] = r["formula"]
    return fm


MAGPIE_CACHE = str(P.RESULTS / "exp_magpie/magpie_features.npz")


def _build_magpie_cache(mp_ids, formula_map, cache=MAGPIE_CACHE):
    """Featurise every formula with Magpie and cache the matrix.

    matminer is imported here rather than at module scope: it caps pandas below 3, and
    it is only needed the first time the cache is built (~2.5 min for 155k formulas).
    """
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    import time
    ep = ElementProperty.from_preset("magpie")
    cols = ep.feature_labels()
    rows, t0 = [], time.time()
    for i, m in enumerate(mp_ids):
        try:
            rows.append(ep.featurize(Composition(formula_map[m])))
        except Exception:
            rows.append([np.nan] * len(cols))
        if (i + 1) % 20000 == 0:
            print(f"    magpie: {i+1:,}/{len(mp_ids):,} ({time.time()-t0:.0f}s)", flush=True)
    X = np.asarray(rows, dtype=np.float64)
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, mp_ids=np.array(list(mp_ids)), cols=np.array(cols))
    print(f"    magpie: {len(mp_ids):,} formulas -> {X.shape} in {time.time()-t0:.0f}s, cached", flush=True)
    return X, [str(c) for c in cols]


def _magpie_matrix(mp_ids, formula_map=None, cache=MAGPIE_CACHE):
    """Magpie composition features (132 statistics over elemental properties), aligned to mp_ids.

    Magpie is a composition-only featuriser: the descriptors are mean / min / max /
    range / mode statistics of tabulated elemental properties weighted by stoichiometry,
    so two polymorphs of the same formula receive identical vectors and no structural
    information enters. It replaces the raw element-fraction baseline, which is well
    below what the field uses as a composition reference.

    The matrix is cached; it is built on first use, which is the only step that needs
    matminer.
    """
    if not Path(cache).exists():
        if formula_map is None:
            raise SystemExit(f"ERROR: {cache} missing and no formula map supplied.")
        print(f"  [magpie] cache absent, building {cache}", flush=True)
        return _build_magpie_cache(mp_ids, formula_map, cache)
    d = np.load(cache, allow_pickle=True)
    pos = {m: i for i, m in enumerate(d["mp_ids"].astype(str))}
    miss = [m for m in mp_ids if m not in pos]
    if miss:
        if formula_map is None:
            raise SystemExit(f"ERROR: {len(miss):,} materials absent from {cache} (first: {miss[:3]}).")
        print(f"  [magpie] cache stale ({len(miss):,} new materials), rebuilding", flush=True)
        return _build_magpie_cache(mp_ids, formula_map, cache)
    rows = np.asarray([pos[m] for m in mp_ids], dtype=int)
    return d["X"][rows].astype(np.float64), [str(c) for c in d["cols"]]


def _composition_matrix(mp_ids, formula_map):
    """Element-fraction feature matrix (rows = mp_ids, cols = elements present)."""
    from pymatgen.core import Composition
    comps, elset = {}, set()
    for m in mp_ids:
        f = formula_map.get(m)
        c = None
        if f:
            try:
                c = Composition(f).fractional_composition.get_el_amt_dict()
            except Exception:
                c = None
        comps[m] = c
        if c:
            elset.update(c.keys())
    elems = sorted(elset)
    idx = {e: i for i, e in enumerate(elems)}
    X = np.full((len(mp_ids), len(elems)), np.nan)
    for r, m in enumerate(mp_ids):
        c = comps[m]
        if c is None:
            continue
        X[r] = 0.0
        for el, fr in c.items():
            X[r, idx[el]] = fr
    return X, elems


def _paired_gain(comp_mp, comp_pred, emb_mp, emb_yt, emb_px, is_clf, B, rng):
    """Paired bootstrap of metric(embedding) - metric(composition-only) on the
    test materials common to both, resampling the same indices. The composition
    and embedding class encoders coincide (both label-encode the same target's
    globally-present classes), so predictions live in one label space. Returns
    (gain, lo, hi, two_sided_p, n_common)."""
    cpos = {str(m): i for i, m in enumerate(comp_mp)}
    ei, ci = [], []
    for j, m in enumerate(emb_mp):
        k = cpos.get(str(m))
        if k is not None:
            ei.append(j); ci.append(k)
    ei = np.asarray(ei, dtype=np.int64); ci = np.asarray(ci, dtype=np.int64)
    g, lo, hi, p = paired_boot_diff(emb_yt[ei], emb_px[ei], comp_pred[ci], is_clf, B, rng)
    return g, lo, hi, p, len(ei)


def run_composition_baseline(orb_rows, uma_rows, orb_dir, uma_dir, out_dir, targets,
                             seed, n_jobs, B, comp_features="magpie"):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 999)
    fm = _load_formula_map()
    rows = []
    for t in targets:
        feat_dir = orb_dir / "features"
        canon, ydict = O.load_canonical_targets(feat_dir, [t])
        canon_pos = {m: i for i, m in enumerate(canon)}
        y = ydict[t]
        task = "clf" if O.is_clf(t) else "reg"
        is_clf = task == "clf"
        le = LabelEncoder().fit(y[O.valid_mask(t, y)]) if is_clf else None
        Xc, elems = (_magpie_matrix(list(canon), fm) if comp_features == "magpie"
                     else _composition_matrix(list(canon), fm))
        sp = np.load(orb_dir / "splits" / f"{O.safe(t)}.npz", allow_pickle=True)
        Xtr, ytr, _ = _align_config(Xc, canon_pos, list(sp["train_mp_ids"]), y, canon_pos, le, task)
        Xte, yte, te_mp = _align_config(Xc, canon_pos, list(sp["test_mp_ids"]), y, canon_pos, le, task)
        est = O.make_model(task, "xgb", seed, n_jobs).fit(Xtr, ytr)
        lin = O.make_model(task, "linear", seed, n_jobs).fit(Xtr, ytr)
        comp_pred = est.predict(Xte)
        pt, lo, hi = boot_ci(yte, comp_pred, is_clf, B, rng)
        comp_lin = _metric(yte, lin.predict(Xte), is_clf)
        # Recover each model's best-config per-sample predictions, then paired-test
        # the gain (embedding minus composition) on the common test materials.
        ocfg, o_best = O.best_config(orb_rows, t)
        ucfg, u_best = U.best_config(uma_rows, t)
        o_rep, o_layer = ocfg
        u_group, u_fam, u_layer = ucfg
        o_mp, o_yt, o_px, _, _, _ = orb_predictions(orb_dir, t, o_rep, o_layer, seed, n_jobs)
        u_mp, u_yt, u_px, _, _, _ = uma_predictions(uma_dir, t, u_group, u_fam, u_layer, seed, n_jobs)
        og, oglo, oghi, ogp, o_n = _paired_gain(te_mp, comp_pred, o_mp, o_yt, o_px, is_clf, B, rng)
        ug, uglo, ughi, ugp, u_n = _paired_gain(te_mp, comp_pred, u_mp, u_yt, u_px, is_clf, B, rng)
        # Physical-error reduction: how the score gain translates into real units.
        # regression -> RMSE (and MAE) in the target's units; classification ->
        # 1 - macro-F1. Lower is better; the cut is 1 - emb_err/comp_err.
        # The classification residual is 1 - macro-F1, not the error rate: this is the
        # definition Table 1 reports, and the error rate is dominated by the largest
        # class, so it understates the gain on the imbalanced targets.
        def _err(yt, pred):
            if is_clf:
                return float(1.0 - f1_score(np.asarray(yt), np.asarray(pred),
                                            average="macro", zero_division=0)), float("nan")
            d = np.asarray(yt, float) - np.asarray(pred, float)
            return float(np.sqrt(np.mean(d ** 2))), float(np.mean(np.abs(d)))
        sigma = float(np.std(np.asarray(yte, float))) if not is_clf else float("nan")
        c_e, c_m = _err(yte, comp_pred)
        o_e, o_m = _err(o_yt, o_px)
        u_e, u_m = _err(u_yt, u_px)
        best_e = min(o_e, u_e)
        rows.append(dict(target=t, task=task, features=comp_features, n_feat=len(elems), n_test=len(yte),
                         comp_xgb=round(pt, 4), comp_xgb_lo=round(lo, 4), comp_xgb_hi=round(hi, 4),
                         comp_linear=round(comp_lin, 4), orb_best=round(o_best, 4),
                         uma_best=round(u_best, 4),
                         orb_gain=round(og, 4), orb_gain_lo=round(oglo, 4), orb_gain_hi=round(oghi, 4),
                         orb_gain_p=round(ogp, 6), orb_gain_n=o_n,
                         uma_gain=round(ug, 4), uma_gain_lo=round(uglo, 4), uma_gain_hi=round(ughi, 4),
                         uma_gain_p=round(ugp, 6), uma_gain_n=u_n,
                         err_metric=("1-macro_f1" if is_clf else "RMSE"),
                         sigma=("" if is_clf else round(sigma, 6)),
                         comp_err=round(c_e, 6), orb_err=round(o_e, 6), uma_err=round(u_e, 6),
                         comp_mae=("" if is_clf else round(c_m, 6)),
                         orb_mae=("" if is_clf else round(o_m, 6)),
                         uma_mae=("" if is_clf else round(u_m, 6)),
                         err_cut=(round(1 - best_e / c_e, 4) if c_e > 0 else ""),
                         # Table 1 reports a cut per model, not one for the better of
                         # the two; err_cut above is min(ORB, UMA) and cannot reproduce
                         # either column on its own.
                         orb_cut=(round(1 - o_e / c_e, 4) if c_e > 0 else ""),
                         uma_cut=(round(1 - u_e / c_e, 4) if c_e > 0 else "")))
        print(f"  [composition] {t:22s} comp={pt:.3f}   "
              f"ORB gain {og:+.3f} [{oglo:+.3f},{oghi:+.3f}] p={ogp:.1e}   "
              f"UMA gain {ug:+.3f} [{uglo:+.3f},{ughi:+.3f}] p={ugp:.1e}", flush=True)
    # Benjamini-Hochberg across all 2*len(targets) gain tests.
    ps = [r["orb_gain_p"] for r in rows] + [r["uma_gain_p"] for r in rows]
    qs = bh_fdr(ps); nt = len(rows)
    for i, r in enumerate(rows):
        r["orb_gain_q"] = round(float(qs[i]), 6)
        r["uma_gain_q"] = round(float(qs[i + nt]), 6)
    _write_composition(rows, out_dir, B)


# ---- writers + figures -----------------------------------------------------

def _write_val_selection(rows, pair_rows, out_dir, val_frac, B):
    with open(out_dir / "val_selection_by_model.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(out_dir / "val_selection_uma_vs_orb.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys())); w.writeheader(); w.writerows(pair_rows)
    L = [f"Validation-split selection control (val_frac={val_frac}, bootstrap B={B})", "=" * 72, "",
         "Selection inflation = test-selected minus validation-selected test score.", ""]
    for r in rows:
        L.append(f"  {r['model']}  {r['target']:22s} test-sel={r['test_score']:.3f} ({r['test_cfg']})"
                 f"  ->  val-sel={r['val_score']:.3f} ({r['val_cfg']})   inflation={r['inflation']:+.3f}")
    L += ["", "ORB vs UMA (UMA-ORB paired diff, test set) under each selection rule:", ""]
    for r in pair_rows:
        L.append(f"  {r['target']:22s} test-sel={r['diff_test']:+.3f} "
                 f"[{r['diff_test_lo']:+.3f},{r['diff_test_hi']:+.3f}]   "
                 f"val-sel={r['diff_val']:+.3f} [{r['diff_val_lo']:+.3f},{r['diff_val_hi']:+.3f}]")
    (out_dir / "val_selection_summary.txt").write_text("\n".join(L), encoding="utf-8")
    _fig_val_inflation(rows, out_dir / "val_selection_inflation.png")
    _fig_val_uma_orb(pair_rows, out_dir / "val_selection_uma_vs_orb.png")
    print(f"  [val-selection] wrote outputs to {out_dir}", flush=True)


def _write_composition(rows, out_dir, B):
    with open(out_dir / "composition_baseline.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    L = [f"Composition-only baseline — element-fraction features, XGBoost (bootstrap B={B})", "=" * 72, "",
         "Gain = best MLIP score minus composition-only score (how much the embedding adds),",
         "with a paired-bootstrap 95% CI and two-sided p on the common test set; q = BH-FDR",
         "across all 2xtargets gain tests. CI excluding 0 => the gain is significant.", ""]
    for r in rows:
        L.append(f"  {r['target']:22s} comp={r['comp_xgb']:.3f} "
                 f"[{r['comp_xgb_lo']:.3f},{r['comp_xgb_hi']:.3f}]   "
                 f"ORB gain {r['orb_gain']:+.3f} [{r['orb_gain_lo']:+.3f},{r['orb_gain_hi']:+.3f}] "
                 f"q={r['orb_gain_q']:.1e}   "
                 f"UMA gain {r['uma_gain']:+.3f} [{r['uma_gain_lo']:+.3f},{r['uma_gain_hi']:+.3f}] "
                 f"q={r['uma_gain_q']:.1e}")
    (out_dir / "composition_baseline_summary.txt").write_text("\n".join(L), encoding="utf-8")
    _fig_composition(rows, out_dir / "composition_baseline.png")
    print(f"  [composition] wrote outputs to {out_dir}", flush=True)


def _fig_composition(rows, path):
    rows = sorted(rows, key=lambda r: r["comp_xgb"])
    labels = [r["target"] for r in rows]
    yp = np.arange(len(rows)); h = 0.26
    fig, ax = plt.subplots(figsize=(9, 0.62 * len(rows) + 1.6))
    ax.barh(yp + h, [r["comp_xgb"] for r in rows], height=h, color="#8a8a8a", label="composition-only (XGB)")
    ax.barh(yp, [r["orb_best"] for r in rows], height=h, color=C_ORB, label="ORB best embedding")
    ax.barh(yp - h, [r["uma_best"] for r in rows], height=h, color=C_UMA, label="UMA best embedding")
    for r, y in zip(rows, yp):
        ax.text(r["comp_xgb"] + 0.006, y + h, f'{r["comp_xgb"]:.2f}', va="center", fontsize=7)
        ax.text(max(r["orb_best"], r["uma_best"]) + 0.006, y, f'+{max(r["orb_gain"], r["uma_gain"]):.2f}',
                va="center", fontsize=7, color="#333")
    ax.set_yticks(yp); ax.set_yticklabels(labels)
    ax.set_xlabel("R² (regression) / macro-F1 (classification)")
    ax.set_xlim(0, 1.02)
    ax.set_title("Composition-only baseline vs best MLIP embedding\n(label = gain of best MLIP over composition)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _fig_val_inflation(rows, path):
    by = {}
    for r in rows:
        by.setdefault(r["target"], {})[r["model"]] = r
    tgts = list(by.keys())
    yp = np.arange(len(tgts)); h = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(tgts) + 1.6))
    ax.barh(yp + h / 2, [by[t]["ORB"]["inflation"] for t in tgts], height=h, color=C_ORB, label="ORB (18 configs)")
    ax.barh(yp - h / 2, [by[t]["UMA"]["inflation"] for t in tgts], height=h, color=C_UMA, label="UMA (96 configs)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(yp); ax.set_yticklabels(tgts)
    ax.set_xlabel("selection inflation  =  test-selected − validation-selected test score")
    ax.set_title("Selection bias by search-space size")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def _fig_val_uma_orb(pair_rows, path):
    tgts = [r["target"] for r in pair_rows]
    yp = np.arange(len(tgts))
    fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(tgts) + 1.6))
    ax.errorbar([r["diff_test"] for r in pair_rows], yp + 0.15,
                xerr=[[r["diff_test"] - r["diff_test_lo"] for r in pair_rows],
                      [r["diff_test_hi"] - r["diff_test"] for r in pair_rows]],
                fmt="o", color="#555", capsize=2, label="test-selected (as reported)")
    ax.errorbar([r["diff_val"] for r in pair_rows], yp - 0.15,
                xerr=[[r["diff_val"] - r["diff_val_lo"] for r in pair_rows],
                      [r["diff_val_hi"] - r["diff_val"] for r in pair_rows]],
                fmt="s", color=C_UMA, capsize=2, label="validation-selected")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(yp); ax.set_yticklabels(tgts)
    ax.set_xlabel("UMA − ORB  (paired difference on the test set)")
    ax.set_title("ORB vs UMA advantage: test-selected vs validation-selected")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ===========================================================================
# Review-response controls, part 2: log-bulk-modulus, probe-selectivity,
# same-material-leakage, rotation.  Each reuses the cached features / embeddings
# and the fixed seed-42 split; the rotation control additionally runs model
# inference through the 02 extraction modules (imported lazily).
# ===========================================================================

def run_log_bulk_modulus(orb_rows, uma_rows, orb_dir, uma_dir, out_dir, seed, n_jobs, B):
    """Re-probe bulk modulus on log10(K_VRH) (Matbench convention) at each model's
    best config, to show the raw-GPa R2 is only mildly inflated by the high-K tail."""
    from sklearn.metrics import r2_score
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 5)

    def _boot(yt, yp):
        n = len(yt); pt = r2_score(yt, yp)
        st = np.array([r2_score(yt[i], yp[i]) for i in (rng.integers(0, n, n) for _ in range(B))])
        return pt, float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))

    def _one(name, mod, mdir, rows):
        (cfg), _ = (mod.best_config(rows, "bulk_modulus"))
        if name == "ORB":
            rep, layer = cfg
            fd = mdir / "features"; canon, yd = mod.load_canonical_targets(fd, ["bulk_modulus"])
            d = np.load(fd / f"{layer}_pooled.npz", allow_pickle=True)
            X = mod.build_X(d["node"], d["edge"], rep); xpos = {m: i for i, m in enumerate(d["mp_ids"].astype(str))}
            cfg_s = f"{rep}/{layer}"
        else:
            g, fam, layer = cfg
            canon, yd = mod.load_targets(mdir / "features", ["bulk_modulus"])
            X, fids = mod.load_feature_set(g, fam, layer); xpos = {m: i for i, m in enumerate(fids.astype(str))}
            cfg_s = f"{g}/{fam}/{layer}"
        cpos = {m: i for i, m in enumerate(canon.astype(str))}; y = yd["bulk_modulus"]
        sp = np.load(mdir / "splits" / f"{mod.safe('bulk_modulus')}.npz", allow_pickle=True)
        def al(ids):
            keep = [m for m in ids.astype(str) if m in xpos and m in cpos]
            rx = np.array([xpos[m] for m in keep]); ry = np.array([cpos[m] for m in keep])
            Xs = X[rx]; ys = y[ry].astype(float); fin = np.isfinite(Xs).all(1) & np.isfinite(ys)
            return Xs[fin], ys[fin]
        Xtr, ytr = al(sp["train_mp_ids"]); Xte, yte = al(sp["test_mp_ids"])
        m = O.make_model("reg", "xgb", seed, n_jobs).fit(Xtr, ytr); r2_raw, lo_r, hi_r = _boot(yte, m.predict(Xte))
        tr = ytr > 0; te = yte > 0
        ml = O.make_model("reg", "xgb", seed, n_jobs).fit(Xtr[tr], np.log10(ytr[tr]))
        r2_log, lo_l, hi_l = _boot(np.log10(yte[te]), ml.predict(Xte[te]))
        print(f"  [log-bulk] {name} {cfg_s}: raw R2={r2_raw:.3f} [{lo_r:.3f},{hi_r:.3f}]  "
              f"log10 R2={r2_log:.3f} [{lo_l:.3f},{hi_l:.3f}]", flush=True)
        return dict(model=name, cfg=cfg_s, r2_raw=round(r2_raw, 4), r2_raw_ci=f"[{lo_r:.3f},{hi_r:.3f}]",
                    r2_log=round(r2_log, 4), r2_log_ci=f"[{lo_l:.3f},{hi_l:.3f}]", n_test=int(te.sum()))

    rows = [_one("ORB", O, orb_dir, orb_rows), _one("UMA", U, uma_dir, uma_rows)]
    with open(out_dir / "log_bulk_modulus.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  [log-bulk] wrote {out_dir/'log_bulk_modulus.csv'}", flush=True)


def run_probe_selectivity(orb_rows, uma_rows, orb_dir, uma_dir, out_dir, targets, seed, n_jobs):
    """Control-task selectivity (Hewitt & Liang): real-label score minus permuted-label
    score at each target's best config. High selectivity => the score reflects the
    representation, not probe capacity."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 11)
    rows = []
    for name, mod, mdir, mrows in (("ORB", O, orb_dir, orb_rows), ("UMA", U, uma_dir, uma_rows)):
        for t in targets:
            cfg, _ = mod.best_config(mrows, t)
            if cfg is None:
                continue
            if name == "ORB":
                rep, layer = cfg; fd = mdir / "features"
                canon, yd = mod.load_canonical_targets(fd, [t])
                d = np.load(fd / f"{layer}_pooled.npz", allow_pickle=True)
                X = mod.build_X(d["node"], d["edge"], rep); xpos = {m: i for i, m in enumerate(d["mp_ids"].astype(str))}
                cfg_s = f"{rep}/{layer}"
            else:
                g, fam, layer = cfg
                canon, yd = mod.load_targets(mdir / "features", [t])
                X, fids = mod.load_feature_set(g, fam, layer); xpos = {m: i for i, m in enumerate(fids.astype(str))}
                cfg_s = f"{g}/{fam}/{layer}"
            cpos = {m: i for i, m in enumerate(canon.astype(str))}; y = yd[t]
            clf = mod.is_clf(t); task = "clf" if clf else "reg"
            le = LabelEncoder().fit(y[mod.valid_mask(t, y)]) if clf else None
            sp = np.load(mdir / "splits" / f"{mod.safe(t)}.npz", allow_pickle=True)
            Xtr, ytr, _ = _align_config(X, xpos, list(sp["train_mp_ids"].astype(str)), y, cpos, le, task)
            Xte, yte, _ = _align_config(X, xpos, list(sp["test_mp_ids"].astype(str)), y, cpos, le, task)
            real = _metric(yte, O.make_model(task, "xgb", seed, n_jobs).fit(Xtr, ytr).predict(Xte), clf)
            pr, pe = rng.permutation(len(ytr)), rng.permutation(len(yte))
            ctrl = _metric(yte[pe], O.make_model(task, "xgb", seed, n_jobs).fit(Xtr, ytr[pr]).predict(Xte), clf)
            rows.append(dict(model=name, target=t, cfg=cfg_s, metric=("F1" if clf else "R2"),
                             real=round(real, 4), control=round(ctrl, 4), selectivity=round(real - ctrl, 4)))
            print(f"  [selectivity] {name} {t:22s} real={real:.3f} control={ctrl:+.3f} sel={real-ctrl:.3f}", flush=True)
    with open(out_dir / "probe_selectivity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  [selectivity] wrote {out_dir/'probe_selectivity.csv'}", flush=True)


def run_same_material(orb_dir, uma_dir, out_dir, seed):
    """Same-material-leakage control on the per-atom k-NN purity used in §3.6 / Table S4.
    Recompute chance-corrected purity (full ~499k-atom index, K=10) excluding neighbours
    from the query atom's own material. Uses the cached per-atom node embeddings."""
    import re
    from sklearn.neighbors import NearestNeighbors
    out_dir.mkdir(parents=True, exist_ok=True)
    K, QSUB, BUF = 10, 30000, 60
    rng = np.random.default_rng(seed)
    tg = np.load(orb_dir / "features" / "targets.npz", allow_pickle=True)
    labmaps = {t: dict(zip(tg["mp_ids"].astype(str), tg[f"cat__{t}"].astype(str)))
               for t in ["crystal_system", "metal_insulator", "magnetic_label"]}
    # (root, per-atom-cache-name-fn, layer-label map).  ORB: one file per layer; UMA: per channel.
    ORBL = {"encoder": "enc", "gnn_stacks_0": "g0", "gnn_stacks_1": "g1", "gnn_stacks_2": "g2",
            "gnn_stacks_3": "g3", "gnn_stacks_4": "g4"}
    UMAL = {"embed": "emb", "block_0": "b0", "block_1": "b1", "block_2": "b2", "block_3": "b3", "backbone": "bb"}

    def purity(X, lab, mp, exclude):
        nn = NearestNeighbors(n_neighbors=K + BUF + 1, algorithm="brute", n_jobs=-1).fit(X)
        q = rng.choice(len(X), min(QSUB, len(X)), replace=False)
        _, idx = nn.kneighbors(X[q]); idx = idx[:, 1:]
        agree = np.empty(len(q)); sf = np.empty(len(q))
        for a, qi in enumerate(q):
            nb = idx[a]; sf[a] = (mp[nb[:K]] == mp[qi]).mean()
            nb = nb[mp[nb] != mp[qi]][:K] if exclude else nb[:K]
            agree[a] = np.mean(lab[nb] == lab[qi]) if len(nb) else np.nan
        vals, cnts = np.unique(lab, return_counts=True); base = float(((cnts / len(lab)) ** 2).sum())
        return (np.nanmean(agree) - base) / (1 - base), float(np.nanmean(sf))

    def peak_std(txt_glob, target):
        best = None
        for f in sorted(Path().glob(txt_glob)):
            for line in open(f):
                if line.strip().startswith(target):
                    m = re.search(r"norm=([0-9.]+)", line)
                    if m and (best is None or float(m.group(1)) > best[1]):
                        best = (f, float(m.group(1)))
        return best

    rows = []
    # ORB: per-atom node cache is {layer}_nodes_umap_cache.npz; knn file {layer}_nodes_knn_lvr.txt
    for t, lm in labmaps.items():
        pk = None
        for L in ORBL:
            f = orb_dir.parent / "plots" / "nodes" / f"{L}_nodes_knn_lvr.txt"
            if f.exists():
                for line in open(f):
                    if line.strip().startswith(t):
                        m = re.search(r"norm=([0-9.]+)", line)
                        if m and (pk is None or float(m.group(1)) > pk[1]): pk = (L, float(m.group(1)))
        if pk is None: continue
        L, paper = pk
        d = np.load(orb_dir.parent / "plots" / "nodes" / f"{L}_nodes_umap_cache.npz", allow_pickle=True)
        X = d["X_raw"]; mp = d["mp_ids"].astype(str); lb = np.array([lm.get(m, "?") for m in mp]); ok = lb != "?"
        X, mp, lb = X[ok], mp[ok], lb[ok]
        std, sf = purity(X, lb, mp, False); excl, _ = purity(X, lb, mp, True)
        rows.append(dict(model="ORB", target=t, layer=ORBL[L], paper_norm=round(paper, 3),
                         purity_std=round(std, 4), purity_excl=round(excl, 4),
                         delta=round(std - excl, 4), same_nbr_frac=round(sf, 4)))
        print(f"  [same-material] ORB {t:16s} std={std:.3f} excl={excl:.3f} d={std-excl:+.3f}", flush=True)
    # UMA: per-channel per-atom caches {layer}_nodes_{fam}_umap_cache.npz; best per-atom channel for crystal system
    for t, lm in labmaps.items():
        best = None
        for fam in ["geom_norm", "ang_norm", "l2_norm", "flat"]:
            for L in UMAL:
                f = uma_dir.parent / "plots" / "nodes" / f"{L}_nodes_{fam}_knn_lvr.txt"
                if f.exists():
                    for line in open(f):
                        if line.strip().startswith(t):
                            m = re.search(r"norm=([0-9.]+)", line)
                            if m and (best is None or float(m.group(1)) > best[2]): best = (fam, L, float(m.group(1)))
        if best is None: continue
        fam, L, paper = best
        d = np.load(uma_dir.parent / "plots" / "nodes" / f"{L}_nodes_{fam}_umap_cache.npz", allow_pickle=True)
        X = d["X_raw"]; mp = d["mp_ids"].astype(str); lb = np.array([lm.get(m, "?") for m in mp]); ok = lb != "?"
        X, mp, lb = X[ok], mp[ok], lb[ok]
        std, sf = purity(X, lb, mp, False); excl, _ = purity(X, lb, mp, True)
        rows.append(dict(model="UMA", target=t, layer=f"{fam}/{UMAL[L]}", paper_norm=round(paper, 3),
                         purity_std=round(std, 4), purity_excl=round(excl, 4),
                         delta=round(std - excl, 4), same_nbr_frac=round(sf, 4)))
        print(f"  [same-material] UMA {t:16s} ({fam}) std={std:.3f} excl={excl:.3f} d={std-excl:+.3f}", flush=True)
    with open(out_dir / "same_material.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  [same-material] wrote {out_dir/'same_material.csv'}", flush=True)


# ---- UMA channel feature builders (match load_feature_set / 03; edges are L2-normalised).
# Module-level so stages/transfer/transfer.py's rotation control reuses the same builders. ----
def _uma_node_feat(d, fam):
    if fam == "full":     return np.asarray(d["structure_full_mean"], np.float32)
    if fam == "l0":       return np.asarray(d["structure_l0_mean"], np.float32)
    if fam == "ang_norm": return np.asarray(d["structure_ang_norm"], np.float32)
    node = np.asarray(d["node"], np.float32)
    if fam == "l1_polar": return np.linalg.norm(node[:, 1:4, :].mean(0), axis=0)
    if fam == "l2_polar": return np.linalg.norm(node[:, 4:9, :].mean(0), axis=0)
    if fam == "geom_norm":
        return np.concatenate([np.linalg.norm(node[:, 1:4, :], axis=1).mean(0),
                               np.linalg.norm(node[:, 4:9, :], axis=1).mean(0)])
    raise ValueError(fam)

def _uma_edge_feat(ep, fam):
    ep = np.asarray(ep, np.float32)
    if   fam == "flat":      v = ep.flatten()
    elif fam == "l0":        v = ep[0, :]
    elif fam == "ang_norm":  v = np.concatenate([ep[0, :], np.linalg.norm(ep[1:4, :], axis=0), np.linalg.norm(ep[4:9, :], axis=0)])
    elif fam == "l1_norm":   v = np.linalg.norm(ep[1:4, :], axis=0)
    elif fam == "l2_norm":   v = np.linalg.norm(ep[4:9, :], axis=0)
    elif fam == "geom_norm": v = np.concatenate([np.linalg.norm(ep[1:4, :], axis=0), np.linalg.norm(ep[4:9, :], axis=0)])
    else: raise ValueError(fam)
    nrm = np.linalg.norm(v); return (v / nrm if nrm > 0 else v).astype(np.float32)

def _uma_build(nres, eres, g, f, l):
    if g == "node": return _uma_node_feat(nres[l], f)
    if g == "edge": return _uma_edge_feat(eres[l]["edge_pooled"], f)
    nf, ef = U.CONCAT_MAP[f]; return np.concatenate([_uma_node_feat(nres[l], nf), _uma_edge_feat(eres[l]["edge_pooled"], ef)])


def _orb_rot_build(rep, layer):
    """Pooled ORB read-out from one forward pass. Eo._extract returns (layer_embs, edge_meta);
    edge_meta is bond geometry, not embeddings, so BOTH node and edge features come from layer_embs."""
    def build(nres, _edge_meta):
        lay = nres[layer]
        nd = np.asarray(lay["node"], np.float32).mean(0)
        ed = np.asarray(lay["edge_emb"], np.float32).mean(0)
        return {"node": nd, "edge": ed, "concat": np.concatenate([nd, ed])}[rep]
    return build


def run_rotation(orb_dir, uma_dir, out_dir, sub_n, seed, n_jobs, uma_random_init=False):
    """Rotation control for crystal system. Re-probe under random per-material rotations:
    (a) every UMA (group, family) channel at its best layer, (b) ORB concat at its best
    layer. Probes trained on cached (unrotated) train features; rotated test features are
    recomputed from a fresh forward pass. Shows only the raw (full/flat) channels collapse.
    uma_random_init: the same test for the random-weight UMA network (weights re-initialised with
    the extraction's fixed seed; probes from the random-init tree passed as uma_dir); ORB skipped."""
    import time as _time
    from sklearn.metrics import f1_score
    from scipy.spatial.transform import Rotation
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    import importlib as _il
    Eu = _il.import_module("stages.extract.extract_uma")
    Eo = _il.import_module("stages.extract.extract_orb")
    if uma_random_init:
        U.CACHE_ROOT = uma_dir.parent / "plots"          # the random-init tree's pooled caches
    out_dir.mkdir(parents=True, exist_ok=True)
    STR = P.STRUCTURES; TGT = "crystal_system"
    rng = np.random.default_rng(seed); adaptor = AseAtomsAdaptor()

    node_feat, edge_feat, uma_build = _uma_node_feat, _uma_edge_feat, _uma_build

    # labels / split (shared)
    canon, yd = U.load_targets(uma_dir / "features", [TGT]); cpos = {m: i for i, m in enumerate(canon.astype(str))}
    yall = yd[TGT].astype(str); le = LabelEncoder().fit(yall[yall != "nan"])
    sp = np.load(uma_dir / "splits" / f"{U.safe(TGT)}.npz", allow_pickle=True)
    test_ids = [str(m) for m in sp["test_mp_ids"]]; train_ids = [str(m) for m in sp["train_mp_ids"]]

    # ---- UMA: best layer per (group, family) from metrics; train a probe each ----
    umr = U.load_metrics(uma_dir / "metrics.csv"); best = {}
    for r in umr:
        if r["target"] == TGT and r["model"] == "xgb":
            k = (r["group"], r["family"])
            try: v = float(r["macro_f1"])
            except: continue
            if k not in best or v > best[k][0]: best[k] = (v, r["layer"])
    EDGE_L = set(Eu.EDGE_LAYERS)
    configs = [(g, f, l) for (g, f), (v, l) in best.items() if not (g in ("edge", "concat") and l not in EDGE_L)]
    probes = {}; unrot = {}; sub = None
    # Kept so Table S11 can be re-scored without re-running the models: the rotated
    # features live only inside this function and were previously discarded.
    unrot_feat = {}; unrot_ids = {}
    for g, f, l in configs:
        X, ids = U.load_feature_set(g, f, l); pos = {m: i for i, m in enumerate(ids.astype(str))}
        def rf(idl):
            keep = [m for m in idl if m in pos and m in cpos and yall[cpos[m]] != "nan"]
            rx = np.array([pos[m] for m in keep]); ry = np.array([cpos[m] for m in keep])
            Xs = X[rx]; fin = np.isfinite(Xs).all(1)
            return np.array(keep)[fin], Xs[fin], le.transform(yall[ry][fin])
        _, Xtr, ytr = rf(train_ids); te_ids, Xte, yte = rf(test_ids)
        probes[(g, f, l)] = U.make_model("clf", "xgb", seed, n_jobs).fit(Xtr, ytr)
        if sub is None: sub = list(rng.choice(te_ids, min(sub_n, len(te_ids)), replace=False))
        tep = {m: i for i, m in enumerate(te_ids)}; idx = [tep[m] for m in sub if m in tep]
        unrot[(g, f, l)] = f1_score(yte[idx], probes[(g, f, l)].predict(Xte[idx]), average="macro")
        unrot_feat[(g, f, l)] = np.asarray(Xte[idx], np.float32)
        unrot_ids[(g, f, l)] = [m for m in sub if m in tep]
    suby = le.transform(np.array([yall[cpos[m]] for m in sub]))

    # ---- ORB: every read-out, each at its own best layer ----
    # One rotated forward pass yields node and edge activations at every layer, so
    # scoring all three read-outs costs two extra probe fits and no extra extraction.
    orows = O.load_metrics(orb_dir / "metrics.csv"); opm = O.primary_metric_name(TGT)
    ocfgs = {}
    for orep in ([] if uma_random_init else O.REPS):
        bl, bv = None, -np.inf
        for l in O.ALL_LAYERS:
            v = O._metric_lookup(orows, TGT, orep, l, "xgb", opm)
            if not np.isnan(v) and v > bv:
                bv, bl = v, l
        if bl is not None:
            ocfgs[orep] = bl
    oprobes, orb_unrot, orb_builders = {}, {}, {}
    orb_unrot_feat, orb_unrot_ids = {}, {}
    for orep, olayer in ocfgs.items():
        od = np.load(orb_dir / "features" / f"{olayer}_pooled.npz", allow_pickle=True)
        oX = O.build_X(od["node"], od["edge"], orep)
        opos = {m: i for i, m in enumerate(od["mp_ids"].astype(str))}
        def orf(idl, oX=oX, opos=opos):
            keep = [m for m in idl if m in opos and m in cpos and yall[cpos[m]] != "nan"]
            rx = np.array([opos[m] for m in keep]); ry = np.array([cpos[m] for m in keep])
            Xs = oX[rx]; fin = np.isfinite(Xs).all(1)
            return np.array(keep)[fin], Xs[fin], le.transform(yall[ry][fin])
        _, oXtr, oytr = orf(train_ids); ote_ids, oXte, oyte = orf(test_ids)
        oprobes[orep] = O.make_model("clf", "xgb", seed, n_jobs).fit(oXtr, oytr)
        otep = {m: i for i, m in enumerate(ote_ids)}; oidx = [otep[m] for m in sub if m in otep]
        orb_unrot[orep] = f1_score(oyte[oidx], oprobes[orep].predict(oXte[oidx]), average="macro")
        orb_unrot_feat[orep] = np.asarray(oXte[oidx], np.float32)
        orb_unrot_ids[orep] = [m for m in sub if m in otep]
        orb_builders[orep] = _orb_rot_build(orep, olayer)
        print(f"  [rotation] ORB {orep}/{olayer}: unrotated F1 = {orb_unrot[orep]:.4f}", flush=True)

    # ---- rotated forward passes ----
    print(f"  [rotation] loading models; rotating {len(sub)} materials ...", flush=True)
    ucalc = Eu._load_fairchem(random_init=uma_random_init)
    orbff, ocalc = (None, None) if uma_random_init else Eo._load_orb()
    upred = {k: [] for k in probes}; opred = {r: [] for r in oprobes}; ry_out = []; t0 = _time.time(); n = 0
    ufeat = {k: [] for k in probes}; ofeat = {r: [] for r in oprobes}; rot_ids = []
    for j, m in enumerate(sub):
        try:
            pmg = Structure.from_file(str(STR / f"{m}.cif")); els = [str(s) for s in pmg.species]
            R = Rotation.random(random_state=int(rng.integers(1e9))).as_matrix()
            at = adaptor.get_atoms(pmg); at.set_positions(at.get_positions() @ R.T); at.set_cell(np.array(at.get_cell()) @ R.T)
            nres, eres = Eu._extract(ucalc, at, els)
            # uma_build(nres, eres, g, f, l): eres is required for the edge and concat
            # read-outs and was missing here, so every material raised and was skipped
            for k, clf in probes.items():
                _fv = np.asarray(uma_build(nres, eres, *k), np.float32)
                ufeat[k].append(_fv)
                upred[k].append(int(clf.predict(_fv[None])[0]))
            if oprobes:
                ao = adaptor.get_atoms(pmg); ao.set_positions(ao.get_positions() @ R.T); ao.set_cell(np.array(ao.get_cell()) @ R.T); ao.calc = ocalc
                nro, ero = Eo._extract(orbff, ocalc, ao, els)
            for orep, oclf in oprobes.items():
                _ov = np.asarray(orb_builders[orep](nro, ero), np.float32)
                ofeat[orep].append(_ov)
                opred[orep].append(int(oclf.predict(_ov[None])[0]))
            ry_out.append(suby[j]); rot_ids.append(str(m)); n += 1
        except Exception as e:
            print(f"    [skip] {m}: {e}", flush=True)
        if (j + 1) % 100 == 0: print(f"    {j+1}/{len(sub)} ({(_time.time()-t0)/(j+1):.2f}s/mat)", flush=True)
    ry_out = np.array(ry_out)
    res = []
    for (g, f, l) in probes:
        fr = f1_score(ry_out, np.array(upred[(g, f, l)]), average="macro")
        res.append(dict(model="UMA", group=g, family=f, layer=l, f1_unrot=round(unrot[(g, f, l)], 4),
                        f1_rot=round(fr, 4), drop=round(unrot[(g, f, l)] - fr, 4)))
    for orep in oprobes:
        fr = f1_score(ry_out, np.array(opred[orep]), average="macro")
        res.append(dict(model="ORB", group=orep, family="-", layer=ocfgs[orep],
                        f1_unrot=round(orb_unrot[orep], 4), f1_rot=round(fr, 4),
                        drop=round(orb_unrot[orep] - fr, 4)))
    res.sort(key=lambda r: -r["f1_unrot"])
    with open(out_dir / "rotation.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
    print(f"\n  [rotation] crystal system, n={n}:", flush=True)
    for r in res:
        print(f"    {r['model']:3s} {r['group']:6s}/{r['family']:9s} unrot={r['f1_unrot']:.3f} "
              f"rot={r['f1_rot']:.3f} drop={r['drop']:+.3f}", flush=True)
    print(f"  [rotation] wrote {out_dir/'rotation.csv'}", flush=True)
    _save_rotation_features(out_dir, sub, rot_ids, ry_out, le,
                            unrot_feat, unrot_ids, ufeat,
                            orb_unrot_feat, orb_unrot_ids, ofeat, ocfgs)


def _save_rotation_features(out_dir, sub, rot_ids, ry_out, le,
                            unrot_feat, unrot_ids, ufeat,
                            orb_unrot_feat, orb_unrot_ids, ofeat, ocfgs):
    """Persist the rotated and unrotated features behind the rotation control.

    The rotated vectors come from a fresh forward pass over randomly rotated structures, so
    no stored embedding reproduces them; writing them lets the control be re-scored without
    the model weights. They are written next to the control's own results,
    <out_dir>/features/{uma,orb}/, and never into results/hf_dataset/, where a reader may
    keep the downloaded datasets. UMA-derived vectors stay under the FAIR Chemistry License.
    """
    try:
        import pyarrow as pa, pyarrow.parquet as pq
    except ImportError:
        print("  [rotation] pyarrow missing; features not persisted", flush=True); return

    def _lc(a):
        a = np.ascontiguousarray(a, dtype=np.float16)
        n, d = a.shape
        offs = pa.array(np.arange(0, (n + 1) * d, d, dtype=np.int32))
        return pa.ListArray.from_arrays(offs, pa.array(a.reshape(-1), type=pa.float16()))

    UMA_DIR = out_dir / "features" / "uma"
    ORB_DIR = out_dir / "features" / "orb"
    for d in (UMA_DIR, ORB_DIR):
        d.mkdir(parents=True, exist_ok=True)
    NOTE = ("Features behind the rotation control on the fixed 1,000-material "
            "test subsample. state=unrotated is the cached embedding; state=rotated is a "
            "fresh forward pass through the frozen model after an independent random "
            "rotation of each structure (seed 42). float16.")

    def _write(path, rows_id, X, state, model, group, family, layer, note):
        t = pa.table({"mp_id": [str(m) for m in rows_id],
                      "state": [state] * len(rows_id), "model": [model] * len(rows_id),
                      "group": [group] * len(rows_id), "family": [family] * len(rows_id),
                      "layer": [layer] * len(rows_id), "emb": _lc(X)})
        pq.write_table(t.replace_schema_metadata({"note": note}), path, compression="zstd")

    n_u = n_o = 0
    for (g, f, l), Xr in ufeat.items():
        if not Xr:
            continue
        _write(UMA_DIR / f"uma-{g}-{f}-{l}-rotated.parquet", rot_ids, np.vstack(Xr),
               "rotated", "UMA", g, f, l, NOTE); n_u += 1
        _write(UMA_DIR / f"uma-{g}-{f}-{l}-unrotated.parquet", unrot_ids[(g, f, l)],
               unrot_feat[(g, f, l)], "unrotated", "UMA", g, f, l, NOTE); n_u += 1
    for orep, Xr in ofeat.items():
        if not Xr:
            continue
        lay = ocfgs[orep]
        _write(ORB_DIR / f"orb-{orep}-{lay}-rotated.parquet", rot_ids, np.vstack(Xr),
               "rotated", "ORB", orep, "-", lay, NOTE); n_o += 1
        _write(ORB_DIR / f"orb-{orep}-{lay}-unrotated.parquet", orb_unrot_ids[orep],
               orb_unrot_feat[orep], "unrotated", "ORB", orep, "-", lay, NOTE); n_o += 1
    # the label key, needed to score either set
    lab = pa.table({"mp_id": [str(m) for m in rot_ids],
                    "crystal_system": [str(c) for c in le.inverse_transform(np.asarray(ry_out))]})
    pq.write_table(lab.replace_schema_metadata(
        {"note": "True crystal system for the rotated subsample, in row order."}),
        ORB_DIR / "_labels.parquet", compression="zstd")
    print(f"  [rotation] persisted features: {n_u} UMA files, {n_o} ORB files, "
          f"{len(rot_ids)} materials -> {out_dir / 'features'}", flush=True)


# Families that are rotation-invariant by construction (scalars / component norms), as
# opposed to the raw tensor views (full, flat) and the sign-carrying L=1 polar/norm views.
INVARIANT_FAMILIES = {"ang_norm", "geom_norm", "l0", "l2", "l2_polar", "l2_norm"}


def run_invariant_crystal(orb_rows, uma_rows, orb_dir, uma_dir, out_dir, seed, n_jobs, B):
    """Bootstrap CI for crystal system on the rotation-invariant footing. The default
    significance pipeline picks UMA's full tensor (concat/full/block_0) as the peak config,
    but that view reads input orientation and collapses under rotation (run_rotation). Here we
    re-score UMA's best rotation-invariant channel (highest macro-F1 among INVARIANT_FAMILIES)
    against ORB's best config, on the full test set, and bootstrap the paired UMA-ORB
    difference. Feeds Table 2 (invariant crystal-system row), Table 3, and §3.5."""
    from sklearn.metrics import f1_score
    out_dir.mkdir(parents=True, exist_ok=True)
    TGT = "crystal_system"; rng = np.random.default_rng(seed)

    canon, yd = U.load_targets(uma_dir / "features", [TGT]); cpos = {m: i for i, m in enumerate(canon.astype(str))}
    yall = yd[TGT].astype(str); le = LabelEncoder().fit(yall[yall != "nan"])
    sp = np.load(uma_dir / "splits" / f"{U.safe(TGT)}.npz", allow_pickle=True)
    train_ids = [str(m) for m in sp["train_mp_ids"]]; test_ids = [str(m) for m in sp["test_mp_ids"]]

    # UMA: best rotation-invariant config from metrics (group, family, layer)
    best = None
    for r in uma_rows:
        if r["target"] == TGT and r["model"] == "xgb" and r["family"] in INVARIANT_FAMILIES:
            try: v = float(r["macro_f1"])
            except: continue
            if best is None or v > best[0]:
                best = (v, r["group"], r["family"], r["layer"])
    _, ug, uf, ul = best
    print(f"  [invariant] UMA best invariant config: {ug}/{uf}/{ul} (cached F1={best[0]:.4f})", flush=True)

    def rows_uma(X, ids, idl):
        pos = {m: i for i, m in enumerate(ids.astype(str))}
        keep = [m for m in idl if m in pos and m in cpos and yall[cpos[m]] != "nan"]
        rx = np.array([pos[m] for m in keep]); ry = np.array([cpos[m] for m in keep])
        Xs = X[rx]; fin = np.isfinite(Xs).all(1)
        return np.array(keep)[fin], Xs[fin], le.transform(yall[ry][fin])
    uX, uids = U.load_feature_set(ug, uf, ul)
    _, uXtr, uytr = rows_uma(uX, uids, train_ids); u_te, uXte, uyte = rows_uma(uX, uids, test_ids)
    u_pred = U.make_model("clf", "xgb", seed, n_jobs).fit(uXtr, uytr).predict(uXte)

    # ORB: best config (as in the main pipeline)
    orb_best, _ = O.best_config(orb_rows, TGT); orep, olayer = orb_best
    od = np.load(orb_dir / "features" / f"{olayer}_pooled.npz", allow_pickle=True)
    oX = O.build_X(od["node"], od["edge"], orep); oids = od["mp_ids"].astype(str)
    o_te, oXte, oyte = None, None, None
    opos = {m: i for i, m in enumerate(oids)}
    def rows_orb(idl):
        keep = [m for m in idl if m in opos and m in cpos and yall[cpos[m]] != "nan"]
        rx = np.array([opos[m] for m in keep]); ry = np.array([cpos[m] for m in keep])
        Xs = oX[rx]; fin = np.isfinite(Xs).all(1)
        return np.array(keep)[fin], Xs[fin], le.transform(yall[ry][fin])
    _, oXtr, oytr = rows_orb(train_ids); o_te, oXte, oyte = rows_orb(test_ids)
    o_pred = O.make_model("clf", "xgb", seed, n_jobs).fit(oXtr, oytr).predict(oXte)

    # align on common test materials for the paired difference
    umap = {m: i for i, m in enumerate(u_te)}; omap = {m: i for i, m in enumerate(o_te)}
    common = [m for m in u_te if m in omap]
    ui = np.array([umap[m] for m in common]); oi = np.array([omap[m] for m in common])
    up, uy = u_pred[ui], uyte[ui]; op, oy = o_pred[oi], oyte[oi]
    n = len(common)
    u_pts = np.empty(B); o_pts = np.empty(B); d_pts = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        fu = f1_score(uy[idx], up[idx], average="macro"); fo = f1_score(oy[idx], op[idx], average="macro")
        u_pts[b] = fu; o_pts[b] = fo; d_pts[b] = fu - fo
    def ci(a): return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
    uf1 = f1_score(uy, up, average="macro"); of1 = f1_score(oy, op, average="macro")
    ulo, uhi = ci(u_pts); olo, ohi = ci(o_pts); dlo, dhi = ci(d_pts)
    res = [dict(config=f"UMA {ug}/{uf}/{ul}", macro_f1=round(uf1, 4), ci_lo=round(ulo, 4), ci_hi=round(uhi, 4)),
           dict(config=f"ORB {orep}/{olayer}", macro_f1=round(of1, 4), ci_lo=round(olo, 4), ci_hi=round(ohi, 4)),
           dict(config="diff UMA-ORB", macro_f1=round(uf1 - of1, 4), ci_lo=round(dlo, 4), ci_hi=round(dhi, 4))]
    with open(out_dir / "invariant_crystal.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys())); w.writeheader(); w.writerows(res)
    print(f"  [invariant] UMA {uf1:.4f} [{ulo:.4f},{uhi:.4f}]  ORB {of1:.4f} [{olo:.4f},{ohi:.4f}]  "
          f"diff {uf1-of1:+.4f} [{dlo:+.4f},{dhi:+.4f}] (n={n})", flush=True)
    print(f"  [invariant] wrote {out_dir/'invariant_crystal.csv'}", flush=True)


def run_grouped_split(orb_rows, uma_rows, orb_dir, uma_dir, out_dir, targets, seed, n_jobs):
    """Re-evaluate each target's best config under grouped splits that keep related
    materials together (composition-grouped = same reduced formula; chemsys-grouped =
    same element set), versus the random 80/20 split. Isolates train/test leakage from
    polymorphs / near-duplicate stoichiometries. Reuses cached features."""
    import json
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    from pymatgen.core import Composition
    out_dir.mkdir(parents=True, exist_ok=True)

    fm = {}
    with open(str(P.STRUCTURES / "index.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line); fm[r["mp_id"]] = r["formula"]

    _gcache = {}
    def gkeys(mp_list):
        red = np.empty(len(mp_list), dtype=object); sysk = np.empty(len(mp_list), dtype=object)
        for i, m in enumerate(mp_list):
            if m not in _gcache:
                f = fm.get(m); rf = sy = None
                if f:
                    try:
                        c = Composition(f); rf = c.reduced_formula
                        sy = "-".join(sorted(set(str(e) for e in c.elements)))
                    except Exception:
                        pass
                _gcache[m] = (rf, sy)
            red[i], sysk[i] = _gcache[m]
        return red, sysk

    def _score(Xtr, ytr, Xte, yte, clf):
        task = "clf" if clf else "reg"
        est = O.make_model(task, "xgb", seed, n_jobs).fit(Xtr, ytr)
        return _metric(yte, est.predict(Xte), clf)

    rows = []
    for name, mod, mdir, mrows in (("ORB", O, orb_dir, orb_rows), ("UMA", U, uma_dir, uma_rows)):
        for t in targets:
            cfg, _ = mod.best_config(mrows, t)
            if cfg is None:
                continue
            if name == "ORB":
                rep, layer = cfg; fd = mdir / "features"
                canon, yd = mod.load_canonical_targets(fd, [t])
                d = np.load(fd / f"{layer}_pooled.npz", allow_pickle=True)
                X = mod.build_X(d["node"], d["edge"], rep); xids = d["mp_ids"].astype(str)
                cfg_s = f"{rep}/{layer}"
            else:
                g, fam, layer = cfg
                canon, yd = mod.load_targets(mdir / "features", [t])
                X, fids = mod.load_feature_set(g, fam, layer); xids = fids.astype(str)
                cfg_s = f"{g}/{fam}/{layer}"
            xpos = {m: i for i, m in enumerate(xids)}
            cpos = {m: i for i, m in enumerate(canon.astype(str))}
            y = yd[t]; clf = mod.is_clf(t)
            le = LabelEncoder().fit(y[mod.valid_mask(t, y)]) if clf else None
            # valid materials (finite features + valid label)
            keep = [m for m in xids if m in cpos and (str(y[cpos[m]]) not in ("nan", "None", ""))]
            rx = np.array([xpos[m] for m in keep]); ry = np.array([cpos[m] for m in keep])
            Xa = X[rx]; fin = np.isfinite(Xa).all(1)
            Xa = Xa[fin]; mp = np.array(keep)[fin]
            ya_raw = y[ry][fin]
            ya = le.transform(ya_raw) if clf else ya_raw.astype(float)
            red, sysk = gkeys(mp)

            def split_score(kind):
                if kind == "random":
                    # the probes' stored seed-42 split: this column is the Table 1 run
                    tr, te = _stored_split(mdir, mod, t, mp)
                    ng = len(np.unique(mp))
                else:
                    grp = red if kind == "composition" else sysk
                    ok = np.array([gg is not None for gg in grp])
                    idx = np.where(ok)[0]
                    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                    tri, tei = next(gss.split(idx, groups=grp[ok]))
                    tr, te = idx[tri], idx[tei]; ng = len(np.unique(grp[ok]))
                s = _score(Xa[tr], ya[tr], Xa[te], ya[te], clf)
                return s, len(te), ng

            s_rand, n_te, _ = split_score("random")
            s_comp, _, ng_comp = split_score("composition")
            s_sys, _, ng_sys = split_score("chemsys")
            rows.append(dict(model=name, target=t, cfg=cfg_s, metric=("F1" if clf else "R2"),
                             random=round(s_rand, 4), composition_grouped=round(s_comp, 4),
                             chemsys_grouped=round(s_sys, 4),
                             drop_composition=round(s_rand - s_comp, 4),
                             drop_chemsys=round(s_rand - s_sys, 4),
                             n_materials=len(ya), n_comp_groups=ng_comp, n_chemsys=ng_sys))
            print(f"  [grouped] {name} {t:22s} random={s_rand:.3f} comp={s_comp:.3f} "
                  f"(Δ{s_rand-s_comp:+.3f}) chemsys={s_sys:.3f} (Δ{s_rand-s_sys:+.3f})", flush=True)
    with open(out_dir / "grouped_split.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  [grouped] wrote {out_dir/'grouped_split.csv'}", flush=True)


def run_whitening(orb_rows, uma_rows, orb_dir, uma_dir, out_dir, seed):
    """Material-level k-NN purity (chance-corrected, K=10) under the raw Euclidean metric
    vs a PCA-whitened metric (= Mahalanobis on the raw features), for the categorical
    targets at each model's best config. Tests whether the poor unsupervised retrieval of
    §3.6 is a metric artefact (whitening recovers it) or genuine (it does not)."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
    out_dir.mkdir(parents=True, exist_ok=True)
    K, QSUB = 10, 30000
    rng = np.random.default_rng(seed + 3)

    def purity(X, lab):
        n = len(X)
        nn = NearestNeighbors(n_neighbors=K + 1, algorithm="auto", n_jobs=-1).fit(X)
        q = rng.choice(n, min(QSUB, n), replace=False)
        _, idx = nn.kneighbors(X[q]); idx = idx[:, 1:]
        agree = np.mean([np.mean(lab[idx[a]] == lab[qi]) for a, qi in enumerate(q)])
        vals, cnts = np.unique(lab, return_counts=True); base = float(((cnts / n) ** 2).sum())
        return (agree - base) / (1 - base)

    rows = []
    for name, mod, mdir, mrows in (("ORB", O, orb_dir, orb_rows), ("UMA", U, uma_dir, uma_rows)):
        for t in ["crystal_system", "metal_insulator", "magnetic_label"]:
            cfg, _ = mod.best_config(mrows, t)
            if cfg is None:
                continue
            if name == "ORB":
                rep, layer = cfg; fd = mdir / "features"
                canon, yd = mod.load_canonical_targets(fd, [t])
                d = np.load(fd / f"{layer}_pooled.npz", allow_pickle=True)
                X = mod.build_X(d["node"], d["edge"], rep); xids = d["mp_ids"].astype(str); cfg_s = f"{rep}/{layer}"
            else:
                g, fam, layer = cfg
                canon, yd = mod.load_targets(mdir / "features", [t])
                X, fids = mod.load_feature_set(g, fam, layer); xids = fids.astype(str); cfg_s = f"{g}/{fam}/{layer}"
            cpos = {m: i for i, m in enumerate(canon.astype(str))}; y = yd[t]
            keep = [m for m in xids if m in cpos and str(y[cpos[m]]) not in ("nan", "None", "")]
            xpos = {m: i for i, m in enumerate(xids)}
            rx = np.array([xpos[m] for m in keep]); ry = np.array([cpos[m] for m in keep])
            Xa = X[rx].astype(np.float64); fin = np.isfinite(Xa).all(1)
            Xa = Xa[fin]; lab = y[ry][fin].astype(str)
            p_raw = purity(Xa, lab)
            Xw = PCA(n_components=min(Xa.shape[1], Xa.shape[0]) - 1, whiten=True,
                     random_state=seed).fit_transform(Xa)
            p_white = purity(Xw, lab)
            rows.append(dict(model=name, target=t, cfg=cfg_s, n=len(lab),
                             purity_euclidean=round(p_raw, 4), purity_whitened=round(p_white, 4),
                             gain=round(p_white - p_raw, 4)))
            print(f"  [whitening] {name} {t:16s} euclid={p_raw:.3f} whitened={p_white:.3f} "
                  f"(gain {p_white-p_raw:+.3f})", flush=True)
    with open(out_dir / "whitening.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"  [whitening] wrote {out_dir/'whitening.csv'}", flush=True)



# =============================================================================== reviewer-driven controls (2026-10)
def _xy_for(model, mdir, cfg, target, seed=None):
    """Feature matrix, labels and the paper's seed-42 split for one configuration.
    Returns (Xtr, ytr, Xte, yte, te_ids, task, label_encoder)."""
    if model == "orb":
        rep, layer = cfg
        canon, ydict = O.load_canonical_targets(mdir / "features", [target])
        d = np.load(mdir / "features" / f"{layer}_pooled.npz", allow_pickle=True)
        X, ids = O.build_X(d["node"], d["edge"], rep), np.asarray(canon).astype(str)
        y = ydict[target]
        task = "clf" if O.is_clf(target) else "reg"
        safe = O.safe
    else:
        group, fam, layer = cfg
        canon, ydict = U.load_targets(mdir / "features", [target])
        X, ids = U.load_feature_set(group, fam, layer)
        ids = np.asarray(ids).astype(str)
        pos = {str(m): i for i, m in enumerate(canon)}
        y = np.array([ydict[target][pos[m]] for m in ids], dtype=object)
        task = "clf" if U.is_clf(target) else "reg"
        safe = U.safe
    ok = (np.array([v is not None and str(v) != "nan" for v in y]) if task == "clf"
          else np.isfinite(np.asarray(y, dtype=float)))
    sp = np.load(mdir / "splits" / f"{safe(target)}.npz", allow_pickle=True)
    pos = {m: i for i, m in enumerate(ids)}

    def take(idl):
        r = np.array([pos[str(m)] for m in idl if str(m) in pos])
        r = r[ok[r] & np.isfinite(X[r]).all(axis=1)]
        return X[r], y[r], ids[r]
    Xtr, ytr, _ = take(sp["train_mp_ids"]); Xte, yte, te_ids = take(sp["test_mp_ids"])
    le = None
    if task == "clf":
        le = LabelEncoder().fit(np.concatenate([ytr, yte]).astype(str))
        ytr, yte = le.transform(ytr.astype(str)), le.transform(yte.astype(str))
    else:
        ytr, yte = ytr.astype(float), yte.astype(float)
    return Xtr, ytr, Xte, yte, te_ids, task, le


def _score_of(task, y, p):
    from sklearn.metrics import f1_score, r2_score
    return float(f1_score(y, p, average="macro")) if task == "clf" else float(r2_score(y, p))


def _selected_cfg(model, rows, target):
    """The validation-selected configuration from the val-selection control, else the test-selected one."""
    S10 = importlib.import_module("stages.downstream.screening_enrichment")
    got = S10.val_configs().get((model, target))
    if got:
        cfg = S10.parse_cfg(model, got[1])
        if cfg:
            return cfg, "validation"
    cfg, _ = (O if model == "orb" else U).best_config(rows, target)
    return cfg, "test"



def _peak_for(model, rows, target, probe):
    """Best configuration and score for one probe family over the same search space as best_config."""
    pm = primary_metric_name_of(model, target)
    best, best_v = None, -np.inf
    if model == "orb":
        for rep in O.REPS:
            for l in O.ALL_LAYERS:
                v = O._metric_lookup(rows, target, rep, l, probe, pm)
                if not math.isnan(v) and v > best_v:
                    best_v, best = v, (rep, l)
    else:
        for g in U.GROUPS:
            for fam in U.GROUP_FAMILIES[g]:
                if not U._family_allowed(target, fam):
                    continue
                for l in U.GROUP_LAYERS[g]:
                    v = U._lookup(rows, target, g, fam, l, probe, pm)
                    if not math.isnan(v) and v > best_v:
                        best_v, best = v, (g, fam, l)
    return best, best_v


def primary_metric_name_of(model, target):
    return (O if model == "orb" else U).primary_metric_name(target)


def run_probe_peaks(orb_rows, uma_rows, out_dir, targets):
    """Probe-family control on the SELECTION (reviewer U7LE 4): the sweep already scores a linear probe
    at every layer and read-out, so the layer the linear probe would select can be compared with the
    layer XGBoost selected, without a new fit. Writes probe_peaks.csv with both peaks, their scores,
    the layer distance between them (in layers of that model's sequence) and each probe's score at
    the other's peak."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for model, rows in (("orb", orb_rows), ("uma", uma_rows)):
        layers = O.ALL_LAYERS if model == "orb" else None
        for t in targets:
            pm = primary_metric_name_of(model, t)
            xc, xv = _peak_for(model, rows, t, "xgb")
            lc, lv = _peak_for(model, rows, t, "linear")
            if xc is None or lc is None:
                continue
            if model == "orb":
                look = lambda cfg, probe: O._metric_lookup(rows, t, cfg[0], cfg[1], probe, pm)
                seq = layers
            else:
                look = lambda cfg, probe: U._lookup(rows, t, cfg[0], cfg[1], cfg[2], probe, pm)
                seq = U.GROUP_LAYERS[xc[0]]
            xl, ll = xc[-1], lc[-1]
            dist = abs(seq.index(xl) - seq.index(ll)) if xl in seq and ll in seq else float("nan")
            rows_out.append(dict(model=model.upper(), target=t, metric=pm,
                                 xgb_cfg="/".join(xc), xgb_score=round(xv, 4),
                                 linear_cfg="/".join(lc), linear_score=round(lv, 4),
                                 same_layer=xl == ll, layer_distance=dist,
                                 linear_at_xgb_cfg=round(float(look(xc, "linear")), 4),
                                 xgb_at_linear_cfg=round(float(look(lc, "xgb")), 4),
                                 n_layers=len(seq)))
    with open(out_dir / "probe_peaks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    same = sum(r["same_layer"] for r in rows_out); adj = sum(r["layer_distance"] <= 1 for r in rows_out)
    print(f"  [probe-peaks] {len(rows_out)} model-target cases: same layer {same}, within one layer {adj}")
    for r in rows_out:
        print(f"    {r['model']} {r['target']:22s} xgb {r['xgb_cfg']:26s} {r['xgb_score']:.3f}  linear {r['linear_cfg']:26s} {r['linear_score']:.3f}  dist {r['layer_distance']}")


def run_mlp_probe(orb_rows, uma_rows, orb_dir, uma_dir, out_dir, targets, seed, n_jobs):
    """Probe-family control (reviewer U7LE): a multilayer perceptron at each target's selected
    configuration, two capacities x two weight decays chosen on a held-out tenth of the training
    split, scored on the paper's test split next to XGBoost and the linear probe."""
    import time as _time
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    out_dir.mkdir(parents=True, exist_ok=True)
    GRID = [((256,), 1e-4), ((256,), 1e-2), ((512, 256), 1e-4), ((512, 256), 1e-2)]
    rows_out = []
    path = out_dir / "mlp_probe.csv"
    done = set()
    if path.exists():
        import csv as _csv
        done = {(r["model"], r["target"]) for r in _csv.DictReader(open(path))}
        rows_out = list(_csv.DictReader(open(path)))
    for target in targets:
        for model, rows, mdir in (("orb", orb_rows, orb_dir), ("uma", uma_rows, uma_dir)):
            if (model, target) in done:
                continue
            cfg, how = _selected_cfg(model, rows, target)
            if cfg is None:
                continue
            Xtr, ytr, Xte, yte, _, task, _ = _xy_for(model, mdir, cfg, target)
            sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
            strat = ytr if task == "clf" else None
            a, b = train_test_split(np.arange(len(ytr)), test_size=0.1, random_state=seed, stratify=strat)
            best = None
            for hidden, alpha in GRID:
                kw = dict(hidden_layer_sizes=hidden, alpha=alpha, batch_size=512, learning_rate_init=1e-3,
                          max_iter=200, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                          random_state=seed)
                t0 = _time.time()
                est = (MLPClassifier(**kw) if task == "clf" else MLPRegressor(**kw)).fit(Xtr_s[a], ytr[a])
                v = _score_of(task, ytr[b], est.predict(Xtr_s[b]))
                print(f"  [mlp] {target} {model} {hidden} alpha={alpha:g}: val={v:.4f} ({_time.time()-t0:.0f}s, {est.n_iter_} epochs)", flush=True)
                if best is None or v > best[0]:
                    best = (v, hidden, alpha, est, _time.time() - t0)
            v, hidden, alpha, est, dt = best
            te = _score_of(task, yte, est.predict(Xte_s))
            pm = O.primary_metric_name(target)
            if model == "orb":
                xgb_v = O._metric_lookup(rows, target, cfg[0], cfg[1], "xgb", pm)
                lin_v = O._metric_lookup(rows, target, cfg[0], cfg[1], "linear", pm)
            else:
                xgb_v = U._lookup(rows, target, cfg[0], cfg[1], cfg[2], "xgb", pm)
                lin_v = U._lookup(rows, target, cfg[0], cfg[1], cfg[2], "linear", pm)
            rows_out.append(dict(model=model, target=target, config="/".join(cfg), selected_by=how, metric=pm,
                                 mlp_hidden="x".join(map(str, hidden)), mlp_alpha=alpha, mlp_val=round(v, 4),
                                 mlp_test=round(te, 4), xgb_test=round(float(xgb_v), 4), linear_test=round(float(lin_v), 4),
                                 mlp_minus_xgb=round(te - float(xgb_v), 4), n_train=len(a), n_test=len(yte),
                                 fit_seconds=round(dt)))
            print(f"  [mlp] {target} {model}: test MLP={te:.4f}  XGB={float(xgb_v):.4f}  linear={float(lin_v):.4f}", flush=True)
            import csv as _csv
            with open(path, "w", newline="") as fh:
                w = _csv.DictWriter(fh, fieldnames=list(rows_out[0].keys())); w.writeheader(); w.writerows(rows_out)
    print(f"  wrote {path}")


def run_probe_family(orb_rows, uma_rows, orb_dir, uma_dir, out_dir, targets, seed, n_jobs, B=1000):
    """Four supervised probes on the same rows (reviewer U7LE 4): at each target's test-selected
    configuration, the one Table 1 and Table S16 report, the linear probe, XGBoost, the K = 10
    nearest-neighbour predictor and a multilayer perceptron (two capacities x two weight decays
    chosen on a held-out tenth of the training split, winner refit on the full training split) are
    fitted on the paper's seed-42 training split and scored on its test split. Each probe gets its
    primary score, RMSE and MAE (regression) or accuracy and 1 - macro-F1 (classification), fit and
    predict wall time, the number of fitted parameters, and a paired bootstrap over test materials of
    its difference from XGBoost in the primary metric and in RMSE. Resumable per (model, target)."""
    import time as _time
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, root_mean_squared_error
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler
    out_dir.mkdir(parents=True, exist_ok=True); (out_dir / "predictions").mkdir(exist_ok=True)
    GRID = [((256,), 1e-4), ((256,), 1e-2), ((512, 256), 1e-4), ((512, 256), 1e-2)]
    path = out_dir / "probe_family.csv"
    rows_out = list(csv.DictReader(open(path))) if path.exists() else []
    done = {(r["model"], r["target"]) for r in rows_out}
    rng = np.random.default_rng(seed)

    def n_params(est, probe, d):
        if probe == "linear":
            m = est[-1]; return int(np.size(m.coef_) + np.size(m.intercept_))
        if probe == "xgb":
            return int(sum(int(t.count("leaf=")) for t in est.get_booster().get_dump()))
        if probe == "mlp":
            return int(sum(w.size for w in est.coefs_) + sum(b.size for b in est.intercepts_))
        return 0

    for target in targets:
        for model, rows, mdir in (("orb", orb_rows, orb_dir), ("uma", uma_rows, uma_dir)):
            if (model, target) in done:
                continue
            cfg, _ = (O if model == "orb" else U).best_config(rows, target)
            if cfg is None:
                continue
            Xtr, ytr, Xte, yte, te_ids, task, _ = _xy_for(model, mdir, cfg, target)
            d = Xtr.shape[1]
            preds, timing, params, hyper = {}, {}, {}, {}
            # linear and XGBoost: the paper's own probe definitions
            for probe in ("linear", "xgb"):
                est = O.make_model(task, probe, seed, n_jobs)
                t0 = _time.time(); est.fit(Xtr, ytr); t1 = _time.time(); preds[probe] = est.predict(Xte); t2 = _time.time()
                timing[probe] = (t1 - t0, t2 - t1); params[probe] = n_params(est, probe, d)
                hyper[probe] = "alpha (RidgeCV)" if probe == "linear" and task == "reg" else ("C fixed" if probe == "linear" else "fixed (300 trees, depth 6)")
            # k-NN predictor as in Table S14: K = 10, Euclidean, majority vote or neighbour mean
            kn = (KNeighborsClassifier if task == "clf" else KNeighborsRegressor)(n_neighbors=10, n_jobs=n_jobs)
            t0 = _time.time(); kn.fit(Xtr, ytr); t1 = _time.time(); preds["knn"] = kn.predict(Xte); t2 = _time.time()
            timing["knn"] = (t1 - t0, t2 - t1); params["knn"] = 0; hyper["knn"] = "K fixed (10)"
            # MLP: grid chosen on a held-out tenth, winner refit on the full training split
            sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
            a, b = train_test_split(np.arange(len(ytr)), test_size=0.1, random_state=seed, stratify=ytr if task == "clf" else None)
            best, t_grid = None, _time.time()
            for hidden, alpha in GRID:
                kw = dict(hidden_layer_sizes=hidden, alpha=alpha, batch_size=512, learning_rate_init=1e-3, max_iter=200,
                          early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, random_state=seed)
                est = (MLPClassifier(**kw) if task == "clf" else MLPRegressor(**kw)).fit(Xtr_s[a], ytr[a])
                v = _score_of(task, ytr[b], est.predict(Xtr_s[b]))
                print(f"  [probe-family] {target} {model} mlp {hidden} alpha={alpha:g}: val={v:.4f}", flush=True)
                if best is None or v > best[0]:
                    best = (v, hidden, alpha, kw)
            t_grid = _time.time() - t_grid
            v, hidden, alpha, kw = best
            t0 = _time.time(); est = (MLPClassifier(**kw) if task == "clf" else MLPRegressor(**kw)).fit(Xtr_s, ytr); t1 = _time.time()
            preds["mlp"] = est.predict(Xte_s); t2 = _time.time()
            timing["mlp"] = (t1 - t0, t2 - t1); params["mlp"] = n_params(est, "mlp", d)
            hyper["mlp"] = f"{'x'.join(map(str, hidden))}, alpha {alpha:g} (grid of 4 on held-out tenth, {t_grid:.0f}s)"
            np.savez(out_dir / "predictions" / f"{model}_{O.safe(target)}.npz", mp_ids=te_ids, y=yte,
                     **{f"pred_{k}": v for k, v in preds.items()})
            # scores and the paired bootstrap against XGBoost
            idx = [rng.integers(0, len(yte), len(yte)) for _ in range(B)]
            def prim(y, p):
                return f1_score(y, p, average="macro") if task == "clf" else r2_score(y, p)
            def err(y, p):
                return (1 - f1_score(y, p, average="macro")) if task == "clf" else root_mean_squared_error(y, p)
            px = preds["xgb"]
            for probe in ("linear", "xgb", "knn", "mlp"):
                p = preds[probe]
                row = dict(model=model.upper(), target=target, config="/".join(cfg), task=task, probe=probe,
                           score=round(prim(yte, p), 4), n_test=len(yte), n_train=len(ytr), dim=d,
                           fit_seconds=round(timing[probe][0], 1), predict_seconds=round(timing[probe][1], 1),
                           n_params=params[probe], hyperparameters=hyper[probe])
                if task == "reg":
                    row.update(rmse=round(root_mean_squared_error(yte, p), 5), mae=round(mean_absolute_error(yte, p), 5))
                else:
                    row.update(accuracy=round(accuracy_score(yte, p), 4), one_minus_f1=round(1 - prim(yte, p), 4))
                if probe != "xgb":
                    ds = np.array([prim(yte[i], p[i]) - prim(yte[i], px[i]) for i in idx])
                    de = np.array([err(yte[i], p[i]) - err(yte[i], px[i]) for i in idx])
                    row.update(d_score_vs_xgb=round(prim(yte, p) - prim(yte, px), 4),
                               d_score_lo=round(float(np.percentile(ds, 2.5)), 4), d_score_hi=round(float(np.percentile(ds, 97.5)), 4),
                               d_err_vs_xgb=round(err(yte, p) - err(yte, px), 5),
                               d_err_lo=round(float(np.percentile(de, 2.5)), 5), d_err_hi=round(float(np.percentile(de, 97.5)), 5))
                rows_out.append(row)
                print(f"  [probe-family] {target} {model} {probe:6s} score={row['score']:.4f} fit={row['fit_seconds']}s params={row['n_params']}", flush=True)
            keys = []
            for r in rows_out:
                for k in r:
                    if k not in keys: keys.append(k)
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows_out)
    print(f"  wrote {path}")



def run_expt_gap(orb_rows, uma_rows, orb_dir, uma_dir, out_dir, seed, n_jobs):
    """Label-provenance control (reviewers QUD5, tHGj): the band-gap probes, trained on MP's PBE
    gaps, scored against measured gaps (matminer `expt_gap_kingsbury`, matched to MP ids) on the
    test-split materials that have one. The PBE label itself is scored the same way, so the probe's
    error can be compared with the label's own error against experiment."""
    from matminer.datasets import load_dataset
    from scipy.stats import spearmanr
    from sklearn.metrics import f1_score, mean_absolute_error, r2_score, root_mean_squared_error
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load_dataset("expt_gap_kingsbury")
    d = d[d.likely_mpid.notna()].copy(); d["likely_mpid"] = d.likely_mpid.astype(str)
    expt = d.groupby("likely_mpid").expt_gap.mean().to_dict()
    target = "band_gap"
    preds = {}
    cfg_o, _ = O.best_config(orb_rows, target)
    te_mp, yte, px, pl, _, _ = orb_predictions(orb_dir, target, cfg_o[0], cfg_o[1], seed, n_jobs)
    te_mp = np.asarray(te_mp).astype(str)
    preds["pbe_label"] = dict(zip(te_mp, yte)); preds["orb_xgb"] = dict(zip(te_mp, px)); preds["orb_linear"] = dict(zip(te_mp, pl))
    cfg_u, _ = U.best_config(uma_rows, target)
    te_mu, yte_u, pxu, plu, _, _ = uma_predictions(uma_dir, target, cfg_u[0], cfg_u[1], cfg_u[2], seed, n_jobs)
    te_mu = np.asarray(te_mu).astype(str)
    preds["uma_xgb"] = dict(zip(te_mu, pxu)); preds["uma_linear"] = dict(zip(te_mu, plu))
    # Magpie, same split
    canon, ydict = O.load_canonical_targets(orb_dir / "features", [target]); canon = np.asarray(canon).astype(str)
    Xm, _ = _magpie_matrix(list(canon), _load_formula_map())
    cpos = {m: i for i, m in enumerate(canon)}; y = np.asarray(ydict[target], dtype=float)
    sp = np.load(orb_dir / "splits" / f"{O.safe(target)}.npz", allow_pickle=True)
    tr = np.array([cpos[str(m)] for m in sp["train_mp_ids"] if str(m) in cpos]); tr = tr[np.isfinite(y[tr]) & np.isfinite(Xm[tr]).all(1)]
    te = np.array([cpos[str(m)] for m in sp["test_mp_ids"] if str(m) in cpos]); te = te[np.isfinite(Xm[te]).all(1)]
    mg = O.make_model("reg", "xgb", seed, n_jobs).fit(Xm[tr], y[tr]).predict(Xm[te])
    preds["magpie_xgb"] = dict(zip(canon[te], mg))
    common = sorted(set(te_mp) & set(expt))
    e = np.array([expt[m] for m in common])
    rows = []
    for name, pm in preds.items():
        p = np.array([pm.get(m, np.nan) for m in common]); ok = np.isfinite(p)
        pe, ee = p[ok], e[ok]
        nonmetal = (ee > 0) & (np.array([preds["pbe_label"].get(m, np.nan) for m in common])[ok] > 0)
        rows.append(dict(predictor=name, n=int(ok.sum()), mae=round(mean_absolute_error(ee, pe), 4),
                         rmse=round(root_mean_squared_error(ee, pe), 4), r2=round(r2_score(ee, pe), 4),
                         spearman=round(float(spearmanr(ee, pe).correlation), 4), mean_bias=round(float(np.mean(pe - ee)), 4),
                         mae_nonmetals=round(mean_absolute_error(ee[nonmetal], pe[nonmetal]), 4) if nonmetal.sum() > 1 else "",
                         n_nonmetals=int(nonmetal.sum()),
                         metal_f1=round(f1_score(ee == 0, pe <= 0.05, average="macro"), 4)))
        print(f"  [expt-gap] {name:12s} n={ok.sum():4d}  MAE={rows[-1]['mae']:.3f}  R2={rows[-1]['r2']:.3f}  bias={rows[-1]['mean_bias']:+.3f}  metal F1={rows[-1]['metal_f1']:.3f}", flush=True)
    import csv as _csv
    with open(out_dir / "expt_gap.csv", "w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(out_dir / "expt_gap_rows.csv", "w", newline="") as fh:
        w = _csv.writer(fh); w.writerow(["mp_id", "expt_gap"] + list(preds))
        for m in common:
            w.writerow([m, expt[m]] + [preds[k].get(m, "") for k in preds])
    (out_dir / "README.md").write_text(
        "# Experimental band gaps\n\nMeasured gaps from matminer `expt_gap_kingsbury` (Kingsbury et al.; Matbench expt_gap with MP ids), "
        f"restricted to the {len(common)} materials of the paper's band-gap test split that carry one. Every predictor is trained on MP's "
        "PBE gaps; `pbe_label` is MP's own gap scored against experiment (the label's error), `metal_f1` treats a measured gap of 0 as a metal "
        "and a predicted gap <= 0.05 eV as a predicted metal. `mae_nonmetals` restricts to materials with both measured and PBE gaps above zero.\n")
    print(f"  wrote {out_dir}")


def run_rotation_invariance(orb_dir, uma_dir, out_dir, sub_n, n_rot, seed, n_jobs):
    """Feature-level rotation test (reviewer u3fv): for `sub_n` crystal-system test materials and
    `n_rot` random rotations each, the relative change of every pooled read-out (UMA node and edge
    families at every layer, ORB node and edge at every layer) against an identity-rotation pass of
    the same extraction, the identity pass against the stored cache, the number of graph edges per
    pass, and the crystal-system F1 of the best probes across the rotations."""
    import time as _time
    from sklearn.metrics import f1_score
    from scipy.spatial.transform import Rotation
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    Eu = importlib.import_module("stages.extract.extract_uma"); Eo = importlib.import_module("stages.extract.extract_orb")
    out_dir.mkdir(parents=True, exist_ok=True)
    STR = P.STRUCTURES; TGT = "crystal_system"
    rng = np.random.default_rng(seed); adaptor = AseAtomsAdaptor()
    UN = ["full", "l0", "ang_norm", "l1_polar", "l2_polar", "geom_norm"]
    UE = ["flat", "l0", "ang_norm", "l1_norm", "l2_norm", "geom_norm"]
    ucfgs = [("node", f, l) for f in UN for l in U.NODE_LAYERS] + [("edge", f, l) for f in UE for l in U.EDGE_LAYERS]
    ocfgs = [(rep, l) for rep in O.REPS for l in O.ALL_LAYERS]          # concat too: the ORB probe's read-out

    canon, yd = U.load_targets(uma_dir / "features", [TGT]); cpos = {str(m): i for i, m in enumerate(canon)}
    yall = yd[TGT].astype(str); le = LabelEncoder().fit(yall[yall != "nan"])
    sp = np.load(uma_dir / "splits" / f"{U.safe(TGT)}.npz", allow_pickle=True)
    test_ids = [str(m) for m in sp["test_mp_ids"] if str(m) in cpos and yall[cpos[str(m)]] != "nan"]
    sub = [str(m) for m in rng.choice(test_ids, min(sub_n, len(test_ids)), replace=False)]
    suby = le.transform(np.array([yall[cpos[m]] for m in sub]))

    # cached features for the identity-vs-cache comparison and the probes
    cache = {}
    for g, f, l in ucfgs:
        try:
            X, ids = U.load_feature_set(g, f, l); pos = {str(m): i for i, m in enumerate(ids)}
            cache[("uma", g, f, l)] = (X, pos)
        except SystemExit:
            pass
    for rep, l in ocfgs:
        od = np.load(orb_dir / "features" / f"{l}_pooled.npz", allow_pickle=True)
        cache[("orb", rep, l)] = (O.build_X(od["node"], od["edge"], rep), {str(m): i for i, m in enumerate(od["mp_ids"])})
    # probes: UMA concat/ang_norm and node/full and node/ang_norm at their best layers, ORB concat at its best
    umr = U.load_metrics(uma_dir / "metrics.csv"); pm = U.primary_metric_name(TGT) if hasattr(U, "primary_metric_name") else "macro_f1"
    def ubest(g, f):
        b, bv = None, -np.inf
        for l in U.GROUP_LAYERS[g]:
            v = U._lookup(umr, TGT, g, f, l, "xgb", pm)
            if not np.isnan(v) and v > bv:
                bv, b = v, l
        return b
    probe_cfgs = [("uma", g, f, ubest(g, f)) for g, f in (("concat", "ang_norm"), ("node", "full"), ("node", "ang_norm"), ("node", "l0"))]
    for k in list(probe_cfgs):
        if k[3] is None:
            probe_cfgs.remove(k)
    orows = O.load_metrics(orb_dir / "metrics.csv"); ocfg, _ = O.best_config(orows, TGT)
    probe_cfgs.append(("orb", ocfg[0], ocfg[1]))
    probes = {}
    train_ids = [str(m) for m in sp["train_mp_ids"] if str(m) in cpos and yall[cpos[str(m)]] != "nan"]
    for k in probe_cfgs:
        if k[0] == "uma":
            X, ids = U.load_feature_set(k[1], k[2], k[3]); pos = {str(m): i for i, m in enumerate(ids)}
            mk = U.make_model
        else:
            X, pos = cache[("orb", k[1], k[2])]; mk = O.make_model
        tr = [m for m in train_ids if m in pos]
        Xtr = X[[pos[m] for m in tr]]; ytr = le.transform(np.array([yall[cpos[m]] for m in tr]))
        fin = np.isfinite(Xtr).all(1)
        probes[k] = (mk("clf", "xgb", seed, n_jobs).fit(Xtr[fin], ytr[fin]), X, pos)
        print(f"  [rotinv] probe {k} trained", flush=True)

    ucalc = Eu._load_fairchem(); orbff, ocalc = Eo._load_orb()
    deltas = {k: [] for k in list(cache) }                 # relative change rotation vs identity
    id_vs_cache = {k: [] for k in cache}
    edge_counts = {"uma": [], "orb": []}
    probe_pred = {k: {r: [] for r in range(n_rot + 1)} for k in probe_cfgs}   # r=0 identity
    kept_y = []
    t0 = _time.time()
    for j, m in enumerate(sub):
        try:
            pmg = Structure.from_file(str(STR / f"{m}.cif")); els = [str(s) for s in pmg.species]
            feats_id = {}
            ecount = {"uma": [], "orb": []}
            for r in range(n_rot + 1):
                R = np.eye(3) if r == 0 else Rotation.random(random_state=int(rng.integers(1e9))).as_matrix()
                at = adaptor.get_atoms(pmg); at.set_positions(at.get_positions() @ R.T); at.set_cell(np.array(at.get_cell()) @ R.T)
                nres, eres = Eu._extract(ucalc, at, els)
                ao = adaptor.get_atoms(pmg); ao.set_positions(ao.get_positions() @ R.T); ao.set_cell(np.array(ao.get_cell()) @ R.T); ao.calc = ocalc
                nro, ero = Eo._extract(orbff, ocalc, ao, els)
                ecount["uma"].append(int(next(iter(eres.values()))["n_edges_total"])); ecount["orb"].append(len(ero["distance"]))
                for k in cache:
                    if k[0] == "uma":
                        v = np.asarray(_uma_build(nres, eres, k[1], k[2], k[3]), np.float32)
                    else:
                        v = np.asarray(_orb_rot_build(k[1], k[2])(nro, ero), np.float32)
                    if r == 0:
                        feats_id[k] = v
                        X, pos = cache[k]
                        if m in pos:
                            c = X[pos[m]]; id_vs_cache[k].append(float(np.linalg.norm(v - c) / (np.linalg.norm(c) + 1e-12)))
                    else:
                        deltas[k].append(float(np.linalg.norm(v - feats_id[k]) / (np.linalg.norm(feats_id[k]) + 1e-12)))
                for k, (clf, X, pos) in probes.items():
                    if k[0] == "uma":
                        v = np.asarray(_uma_build(nres, eres, k[1], k[2], k[3]), np.float32)
                    else:
                        v = np.asarray(_orb_rot_build(k[1], k[2])(nro, ero), np.float32)
                    probe_pred[k][r].append(int(clf.predict(v[None])[0]))
            edge_counts["uma"].append(ecount["uma"]); edge_counts["orb"].append(ecount["orb"]); kept_y.append(suby[j])
        except Exception as ex:
            print(f"    [skip] {m}: {ex}", flush=True); continue
        if (j + 1) % 25 == 0:
            print(f"  [rotinv] {j+1}/{len(sub)} ({(_time.time()-t0)/(j+1):.1f}s/mat)", flush=True)
    import csv as _csv
    with open(out_dir / "feature_deltas.csv", "w", newline="") as fh:
        w = _csv.writer(fh); w.writerow(["model", "group", "family", "layer", "n_pairs", "rel_change_median", "rel_change_p90", "rel_change_max",
                                        "identity_vs_cache_median", "identity_vs_cache_max"])
        for k in cache:
            dd, ic = np.array(deltas[k]), np.array(id_vs_cache[k])
            row = list(k) if k[0] == "uma" else [k[0], k[1], "", k[2]]
            w.writerow(row + [len(dd), f"{np.median(dd):.3e}", f"{np.quantile(dd, .9):.3e}", f"{dd.max():.3e}",
                              f"{np.median(ic):.3e}" if len(ic) else "", f"{ic.max():.3e}" if len(ic) else ""])
    kept_y = np.array(kept_y)
    with open(out_dir / "f1_by_rotation.csv", "w", newline="") as fh:
        w = _csv.writer(fh); w.writerow(["model", "config", "n", "f1_identity", "f1_rotations_mean", "f1_rotations_sd", "f1_rotations_min", "f1_rotations_max"])
        for k, pr in probe_pred.items():
            f1s = [f1_score(kept_y, np.array(pr[r]), average="macro") for r in range(1, n_rot + 1)]
            f0 = f1_score(kept_y, np.array(pr[0]), average="macro")
            w.writerow([k[0], "/".join(k[1:]), len(kept_y), f"{f0:.4f}", f"{np.mean(f1s):.4f}", f"{np.std(f1s, ddof=1):.4f}", f"{min(f1s):.4f}", f"{max(f1s):.4f}"])
            print(f"  [rotinv] F1 {k}: identity {f0:.3f}, rotated {np.mean(f1s):.3f} ± {np.std(f1s, ddof=1):.3f}", flush=True)
    eu, eo = np.array(edge_counts["uma"]), np.array(edge_counts["orb"])
    with open(out_dir / "edge_counts.csv", "w", newline="") as fh:
        w = _csv.writer(fh); w.writerow(["model", "n", "frac_structures_edge_count_changes", "median_abs_change", "max_abs_change"])
        for name, ec in (("uma", eu), ("orb", eo)):
            ch = np.abs(ec[:, 1:] - ec[:, :1])
            w.writerow([name, len(ec), f"{(ch.max(1) > 0).mean():.3f}", f"{np.median(ch):.1f}", f"{ch.max():.0f}"])
    (out_dir / "README.md").write_text(
        f"# Rotation invariance at the feature level\n\n{len(kept_y)} crystal-system test materials, {n_rot} random rotations each, one identity pass. "
        "`rel_change` = ||f(Rx) - f(x)|| / ||f(x)|| for the pooled read-out; `identity_vs_cache` compares a fresh identity-rotation pass with the stored "
        "feature cache; `edge_counts` reports whether the graph (number of edges) changes under rotation; `f1_by_rotation` scores the paper's probes "
        "on the identity pass and on each rotation.\n")
    print(f"  wrote {out_dir}")

if __name__ == "__main__":
    main()
