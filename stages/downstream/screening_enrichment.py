#!/usr/bin/env python3
"""
stages/downstream/screening_enrichment.py — does the probe's ranking actually help you screen candidates?

Sections 3.2/5 of the paper argue the best probes are accurate enough (in physical
units, via RMSE = sigma*sqrt(1-R2)) "to rank and filter candidates for screening" —
but nothing in the pipeline tests that directly. This script does, using metrics a
materials-screening reader actually cares about: does the model's predicted top-k%
overlap the true top-k%, and by how much more than a random draw would.

Reuses mlipprobes/significance_lib.py's best-config prediction recovery (which itself
reuses probes_*.py's loaders/best_config and significance_lib's orb_predictions/uma_predictions) --
no new model fitting, just new metrics computed on the same (y_true, y_pred) pairs
already used for every other significance result in the paper.

Scope: the 6 RELIABLE regression targets. Each has a scientifically motivated
"interesting tail" (TARGET_DIRECTION below) for top-k enrichment / recall-at-decile,
except band_gap, which instead gets a directly interpretable window-retrieval
scenario (precision/recall for the 0.5-3.0 eV semiconductor range) since a single
top/bottom tail isn't the natural screening query for it. Classification targets
(crystal_system, metal_insulator, magnetic_label) get the class analogue: rank the
test set by the probe's predicted probability of a class and ask how enriched that
class is in the top-k% -- precision@k against the class base rate (lift), recall@k,
and one-vs-rest AUROC as the ranking-quality summary. Every class of every target is
reported, so the nine paper targets are all covered.

Reads  $MLIP_PROBES_ROOT/results/{orb,fairchem}/property_prediction/{metrics.csv,features,splits}
Writes $MLIP_PROBES_ROOT/results/screening_enrichment/

Usage:
    uv run python stages/downstream/screening_enrichment.py
    uv run python stages/downstream/screening_enrichment.py --targets bulk_modulus,band_gap
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import argparse
import csv
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["legend.frameon"] = False
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
O = importlib.import_module("stages.probes.probes_orb")
U = importlib.import_module("stages.probes.probes_uma")
S = importlib.import_module("mlipprobes.significance_lib")

# direction of the "interesting tail" for top-k enrichment: "low" = most negative/
# smallest values are the screening target (e.g. formation energy: more negative =
# more stable), "high" = largest values are (e.g. bulk modulus: stiffest materials).
# density's direction is illustrative (lightweight-materials screening), not the
# only valid choice -- the enrichment/recall numbers hold either way by symmetry.
TARGET_DIRECTION = {
    "formation_energy/atom": "low",   # most stable (most negative formation energy)
    "density":                "low",   # lightweight-materials screening (illustrative)
    "bulk_modulus":           "high",  # stiffest materials
    "mag_density":            "high",  # strongest magnetic moment density
}
# energy_above_hull is NOT a percentile-tail target: 22% of the seed-42 test set sits at
# exactly 0 eV/atom (on the hull), so the bottom 1%, 5% and 10% cutoffs are all 0.0000 and
# "recall of the true bottom k%" measures tie-breaking, not screening. It is screened by a
# physical threshold instead: rank by predicted E-hull, ask how many of the shortlist are
# truly within `thr` of the hull. 0.0 = on the hull; 0.05 / 0.10 eV/atom are the usual
# synthesisability windows (Sun et al. 2016 put the 90th percentile of metastable
# inorganic phases at 67 meV/atom).
THRESHOLD_TARGETS = {"energy_above_hull": [0.0, 0.05, 0.10]}
BAND_GAP_WINDOW = (0.5, 3.0)  # eV -- the semiconductor range a screening query would use
# the full lift curve, 1% .. 100% (lift -> 1 at 100% by construction); 1/5/10 are the
# headline points. Random selection has expected lift 1 at every k.
TOP_FRACS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
HEADLINE_FRACS = [0.01, 0.05, 0.10]
BOOT_B = 2000            # paired bootstrap resamples, as everywhere else in the paper
MAGPIE_NPZ = (P.RESULTS / "exp_magpie/magpie_features.npz")
# The validation-selection control (stages/controls/baselines.py): per (model, target) the configuration chosen on the
# test split (the paper's) and the one chosen on a validation split carved from train. For 15 of
# the 18 pairs they coincide; where they differ, every screening metric is recomputed at the
# validation-selected configuration and reported with selection = "val", plus the paired
# test-minus-val difference. The paper's val-selection control bounds the effect on *scores*
# at <= 0.004; a top-1% tail is ~300 materials and can be flattered where the mean is not.
VAL_SELECTION_CSV = (P.RESULTS / "proppred_controls/val_selection/val_selection_by_model.csv")


def parse_cfg(model_name, s):
    """'concat/gnn_stacks_2' -> ('concat','gnn_stacks_2'); 'node/ang_norm/block_3' -> 3-tuple."""
    parts = tuple(s.split("/"))
    return parts if len(parts) == (2 if model_name == "orb" else 3) else None


def val_configs():
    """{(model, target): (test_cfg_str, val_cfg_str)} from the validation-selection control; {} if absent."""
    if not VAL_SELECTION_CSV.exists():
        print(f"  [val-selection] {VAL_SELECTION_CSV} not found; skipping", file=sys.stderr); return {}
    out = {}
    with open(VAL_SELECTION_CSV) as f:
        for r in csv.DictReader(f):
            out[(r["model"].lower(), r["target"])] = (r["test_cfg"], r["val_cfg"])
    return out


def _config_and_predictions(model_name, mod, mdir, rows, target, seed, n_jobs, cfg=None):
    if cfg is None:
        cfg, score = mod.best_config(rows, target)
    else:
        score = float("nan")
    if cfg is None:
        return None
    if model_name == "orb":
        rep, layer = cfg
        mp, yt, px, _pl, is_clf, _base = S.orb_predictions(mdir, target, rep, layer, seed, n_jobs)
        cfg_str = f"{rep}/{layer}"
    else:
        group, fam, layer = cfg
        mp, yt, px, _pl, is_clf, _base = S.uma_predictions(mdir, target, group, fam, layer, seed, n_jobs)
        cfg_str = f"{group}/{fam}/{layer}"
    assert not is_clf, f"{target} is classification; screening enrichment is regression-only"
    return dict(mp=mp, yt=yt.astype(float), yp=px.astype(float), cfg=cfg_str, score=score)


CLF_TARGETS = ["crystal_system", "metal_insulator", "magnetic_label"]


def _fit_clf_proba(model_name, mod, mdir, rows, target, seed, n_jobs, cfg=None):
    """Refit the top XGBoost config on the fixed seed-42 train split and return
    (test_mp_ids, y_true_int, proba [n, C], classes, cfg_str, mp_score). Mirrors the
    alignment in 06_significance_proppred.orb_predictions / uma_predictions, but keeps
    the model so predict_proba is available."""
    from sklearn.preprocessing import LabelEncoder
    if cfg is None:
        cfg, mp_score = mod.best_config(rows, target)
    else:
        mp_score = float("nan")
    if cfg is None:
        return None
    feat_dir = mdir / "features"
    if model_name == "orb":
        rep, layer = cfg
        canon, ydict = O.load_canonical_targets(feat_dir, [target])
        d = np.load(feat_dir / f"{layer}_pooled.npz", allow_pickle=True)
        X, ids = O.build_X(d["node"], d["edge"], rep), canon
        y = ydict[target]; cfg_str = f"{rep}/{layer}"
    else:
        group, fam, layer = cfg
        canon, ydict = U.load_targets(feat_dir, [target])
        X, ids = U.load_feature_set(group, fam, layer)
        pos = {m: i for i, m in enumerate(canon)}
        y = np.array([ydict[target][pos[m]] for m in ids], dtype=object); cfg_str = f"{group}/{fam}/{layer}"
    sp = np.load(mdir / "splits" / f"{mod.safe(target)}.npz", allow_pickle=True)
    pos = {m: i for i, m in enumerate(ids)}

    def align(want):
        keep = [m for m in want if m in pos]
        r = np.fromiter((pos[m] for m in keep), dtype=np.int64, count=len(keep))
        Xs, ys, ms = X[r], y[r], np.array(keep, dtype=object)
        ok = np.isfinite(Xs).all(axis=1) & np.array([v is not None for v in ys])
        return Xs[ok], ys[ok], ms[ok]

    Xtr, ytr_raw, _ = align(sp["train_mp_ids"]); Xte, yte_raw, te_mp = align(sp["test_mp_ids"])
    le = LabelEncoder().fit(ytr_raw)
    known = np.isin(yte_raw, le.classes_)
    Xte, yte_raw, te_mp = Xte[known], yte_raw[known], te_mp[known]
    est = mod.make_model("clf", "xgb", seed, n_jobs).fit(Xtr, le.transform(ytr_raw))
    return te_mp, le.transform(yte_raw), est.predict_proba(Xte), list(le.classes_), cfg_str, mp_score


def class_enrichment_at_k(y_int, proba, c, frac):
    """Rank by P(class c); top-k = round(n*frac) rows. Returns (k, precision, recall, lift, base_rate)."""
    n = len(y_int); k = max(1, int(round(n * frac)))
    top = np.argsort(-proba[:, c])[:k]
    is_c = (y_int == c)
    hits = int(is_c[top].sum())
    base = float(is_c.mean())
    precision = hits / k
    recall = hits / int(is_c.sum()) if is_c.sum() else float("nan")
    lift = precision / base if base > 0 else float("nan")
    return k, precision, recall, lift, base


def enrichment_at_k(yt, yp, frac, direction):
    """Overlap between the true top-frac and predicted top-frac (same-size sets, so
    precision == recall here). Returns (n_overlap, n_set, recall, lift) where lift
    is enrichment over a random draw of the same size (1.0 = no better than chance)."""
    n = len(yt)
    k = max(1, int(round(n * frac)))
    order_t = np.argsort(yt) if direction == "low" else np.argsort(-yt)
    order_p = np.argsort(yp) if direction == "low" else np.argsort(-yp)
    true_top = set(order_t[:k].tolist())
    pred_top = set(order_p[:k].tolist())
    overlap = len(true_top & pred_top)
    recall = overlap / k
    lift = recall / frac  # frac of true-top captured, divided by the random-draw expectation (frac)
    return overlap, k, recall, lift


def window_retrieval(yt, yp, lo, hi):
    """Precision/recall/F1 for identifying materials whose TRUE value falls in
    [lo, hi], using the PREDICTED value to make the call."""
    true_in = (yt >= lo) & (yt <= hi)
    pred_in = (yp >= lo) & (yp <= hi)
    tp = int(np.sum(true_in & pred_in))
    fp = int(np.sum(~true_in & pred_in))
    fn = int(np.sum(true_in & ~pred_in))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return dict(n_true_in_window=int(true_in.sum()), tp=tp, fp=fp, fn=fn,
                precision=precision, recall=recall, f1=f1)



def _magpie_predictions(orb_dir, target, seed, n_jobs):
    """Composition-only XGBoost on the SAME seed-42 split as ORB's probe: the third arm of
    every screening comparison. Returns (test_mp_ids, ytrue, ypred, is_clf, proba|None, classes|None)."""
    from sklearn.preprocessing import LabelEncoder
    z = np.load(MAGPIE_NPZ, allow_pickle=True)
    Xm, mids = z["X"].astype(np.float32), z["mp_ids"]
    mpos = {m: i for i, m in enumerate(mids)}
    feat_dir = orb_dir / "features"
    canon, ydict = O.load_canonical_targets(feat_dir, [target]); y = ydict[target]
    cpos = {m: i for i, m in enumerate(canon)}
    sp = np.load(orb_dir / "splits" / f"{O.safe(target)}.npz", allow_pickle=True)
    is_clf = O.is_clf(target)

    def align(want):
        keep = [m for m in want if m in mpos and m in cpos]
        X = Xm[[mpos[m] for m in keep]]; yy = y[[cpos[m] for m in keep]]
        ok = np.isfinite(X).all(axis=1) & (np.array([v is not None for v in yy]) if is_clf else np.isfinite(yy.astype(float)))
        return X[ok], yy[ok], np.array(keep, dtype=object)[ok]

    Xtr, ytr, _ = align(sp["train_mp_ids"]); Xte, yte, te = align(sp["test_mp_ids"])
    task = "clf" if is_clf else "reg"
    if is_clf:
        le = LabelEncoder().fit(ytr); known = np.isin(yte, le.classes_)
        Xte, yte, te = Xte[known], yte[known], te[known]
        est = O.make_model(task, "xgb", seed, n_jobs).fit(Xtr, le.transform(ytr))
        return te, le.transform(yte), est.predict(Xte), True, est.predict_proba(Xte), list(le.classes_)
    est = O.make_model(task, "xgb", seed, n_jobs).fit(Xtr, ytr.astype(float))
    return te, yte.astype(float), est.predict(Xte), False, None, None


def recall_at_k(yt, yp, frac, direction):
    """Recall of the true top-frac tail by the predicted top-frac shortlist (same sizes, so
    precision == recall). argpartition, so a bootstrap resample costs O(n)."""
    n = len(yt); k = max(1, int(round(n * frac)))
    if k >= n:
        return 1.0
    sgn = 1.0 if direction == "low" else -1.0
    top_true = np.argpartition(sgn * yt, k - 1)[:k]
    top_pred = np.argpartition(sgn * yp, k - 1)[:k]
    return len(np.intersect1d(top_true, top_pred, assume_unique=True)) / k


def threshold_at_k(yt, yp, frac, thr):
    """Shortlist the predicted-lowest top-frac; precision/recall for TRUE y <= thr."""
    n = len(yt); k = max(1, int(round(n * frac)))
    top = np.argpartition(yp, min(k, n) - 1)[:k] if k < n else np.arange(n)
    hit = yt[top] <= thr; n_true = int((yt <= thr).sum())
    prec = float(hit.mean()); rec = float(hit.sum() / n_true) if n_true else float("nan")
    base = n_true / n
    return prec, rec, (prec / base if base > 0 else float("nan")), base


def paired_bootstrap(fn, arrays, B, seed):
    """fn(*arrays_resampled) -> scalar. Returns (point, lo, hi, boots) over B paired resamples
    of the material index. Percentile 95% interval."""
    rng = np.random.default_rng(seed); n = len(arrays[0]); point = fn(*arrays)
    boots = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boots[b] = fn(*[a[idx] for a in arrays])
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)), boots


def tie_diagnostics(yt, frac, direction):
    """How ambiguous is the true top-frac? Share of the test set tied at the k-th value."""
    n = len(yt); k = max(1, int(round(n * frac)))
    v = np.sort(yt) if direction == "low" else -np.sort(-yt)
    cut = v[k - 1]
    return float(cut), float(np.mean(yt == cut))


def lift_curve_plot(curve_rows, out_dir):
    """Lift vs shortlist fraction, one panel per target, ORB/UMA/Magpie + random=1."""
    import pandas as pd
    df = pd.DataFrame(curve_rows)
    tgts = [t for t in list(TARGET_DIRECTION) + list(THRESHOLD_TARGETS) if t in set(df.target)]
    if not tgts:
        return
    cols = 3; rows_ = int(np.ceil(len(tgts) / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(4.2 * cols, 3.3 * rows_), squeeze=False)
    colours = {"uma": "#4C72B0", "orb": "#55A868", "magpie": "#DD8452"}
    labels = {"uma": "UMA embeddings", "orb": "ORB embeddings", "magpie": "Magpie composition baseline"}
    for ax, tg in zip(axes.ravel(), tgts):
        d = df[df.target == tg]
        for model in ("uma", "orb", "magpie"):
            g = d[d.model == model].sort_values("frac")
            if len(g):
                ax.plot(g.frac * 100, g.lift, "o-", ms=4, color=colours[model], label=labels[model])
                if "ci_lo" in g and g.ci_lo.notna().any():
                    ax.fill_between(g.frac * 100, g.ci_lo, g.ci_hi, color=colours[model], alpha=0.15, lw=0)
        ax.axhline(1.0, color="#C9A227", lw=1.4, label="Random selection")
        ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
        sub = d.iloc[0]["criterion"] if len(d) else ""
        ax.set_title(f"{tg}  ({sub})", fontsize=9); ax.set_xlabel("shortlist size (% of test set)"); ax.set_ylabel("lift over random")
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(tgts):]:
        ax.axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(out_dir / "lift_curves.png", dpi=180); plt.close(fig)


def evaluate_regression(t, preds, yt, boot, seed, selection, curve_rows, pair_rows, tie_rows, pairs=(("uma","orb"),("orb","magpie"),("uma","magpie"))):
    """Percentile-tail / threshold / window metrics for one target given aligned predictions."""
    if t in TARGET_DIRECTION:
        direction = TARGET_DIRECTION[t]
        for frac in TOP_FRACS:
            if selection == "test":
                cut, tied = tie_diagnostics(yt, frac, direction)
                tie_rows.append(dict(target=t, frac=frac, cutoff_value=cut, share_tied_at_cutoff=round(tied, 4), ambiguous=tied > frac))
            for name, rec in preds.items():
                f = lambda a, b, fr=frac, d=direction: recall_at_k(a, b, fr, d)
                if boot:
                    pt, lo, hi, _ = paired_bootstrap(f, [rec["yt"], rec["yp"]], boot, seed)
                else:
                    pt, lo, hi = f(rec["yt"], rec["yp"]), float("nan"), float("nan")
                curve_rows.append(dict(target=t, model=name, selection=selection, config=rec["cfg"],
                                       criterion=f"true top {int(frac*100)}% ({direction})", frac=frac,
                                       k=max(1, int(round(len(yt) * frac))), recall=round(pt, 4), lift=round(pt / frac, 2),
                                       ci_lo=round(lo / frac, 2), ci_hi=round(hi / frac, 2), spearman_rho=round(rec["rho"], 4)))
            for a, b in pairs:
                if a in preds and b in preds and boot:
                    f = lambda ya, pa, pb, fr=frac, d=direction: recall_at_k(ya, pa, fr, d) - recall_at_k(ya, pb, fr, d)
                    pt, lo, hi, boots = paired_bootstrap(f, [yt, preds[a]["yp"], preds[b]["yp"]], boot, seed)
                    pair_rows.append(dict(target=t, selection=selection, frac=frac, pair=f"{a}-{b}", delta_recall=round(pt, 4),
                                          ci_lo=round(lo, 4), ci_hi=round(hi, 4), p_boot=float(2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))))
    if t in THRESHOLD_TARGETS:
        for thr in THRESHOLD_TARGETS[t]:
            for frac in TOP_FRACS:
                for name, rec in preds.items():
                    f = lambda a, b, fr=frac, th=thr: threshold_at_k(a, b, fr, th)[0]
                    prec, recl, lift, base = threshold_at_k(rec["yt"], rec["yp"], frac, thr)
                    lo, hi = (paired_bootstrap(f, [rec["yt"], rec["yp"]], boot, seed)[1:3] if boot else (float("nan"), float("nan")))
                    curve_rows.append(dict(target=t, model=name, selection=selection, config=rec["cfg"],
                                           criterion=f"true E-hull <= {thr:.2f} eV/atom", frac=frac, k=max(1, int(round(len(yt) * frac))),
                                           threshold=thr, base_rate=round(base, 4), precision=round(prec, 4), recall=round(recl, 4),
                                           lift=round(lift, 2), ci_lo=round(lo / base, 2) if base else float("nan"),
                                           ci_hi=round(hi / base, 2) if base else float("nan"), spearman_rho=round(rec["rho"], 4)))
        if selection == "test":
            tie_rows.append(dict(target=t, frac=None, cutoff_value=0.0, share_tied_at_cutoff=round(float(np.mean(yt == 0)), 4),
                                 ambiguous=True, note="percentile tail undefined: bottom 1/5/10% cutoffs all 0.0"))


def align_common(preds):
    common = sorted(set.intersection(*[set(r["mp"]) for r in preds.values()]))
    for rec in preds.values():
        pos = {m: i for i, m in enumerate(rec["mp"])}
        idx = np.array([pos[m] for m in common])
        rec["yt"], rec["yp"], rec["mp"] = rec["yt"][idx], rec["yp"][idx], np.array(common, dtype=object)
        rec["rho"] = float(spearmanr(rec["yt"], rec["yp"])[0])
    return next(iter(preds.values()))["yt"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orb-dir", default=str(P.RESULTS / "orb/property_prediction"))
    ap.add_argument("--uma-dir", default=str(P.RESULTS / "fairchem/property_prediction"))
    ap.add_argument("--output-dir", default=str(P.RESULTS / "screening_enrichment"))
    ap.add_argument("--targets", default=None, help="comma list; default = all nine")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--boot", type=int, default=BOOT_B, help="paired bootstrap resamples (0 = off)")
    ap.add_argument("--skip-magpie", action="store_true")
    ap.add_argument("--skip-val-selection", action="store_true")
    ap.add_argument("--uma-cache-root", default=None,
                    help="UMA pooled-feature cache for --uma-dir when it is not the paper's tree "
                         "(e.g. $MLIP_PROBES_ROOT/results/fairchem_random_init/plots for the random-initialisation control)")
    args = ap.parse_args()
    if args.uma_cache_root:
        U.CACHE_ROOT = Path(args.uma_cache_root)

    orb_dir, uma_dir = Path(args.orb_dir), Path(args.uma_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    targets = args.targets.split(",") if args.targets else (list(TARGET_DIRECTION) + list(THRESHOLD_TARGETS) + ["band_gap"] + CLF_TARGETS)
    orb_rows = O.load_metrics(orb_dir / "metrics.csv"); uma_rows = U.load_metrics(uma_dir / "metrics.csv")
    vsel = {} if args.skip_val_selection else val_configs()

    curve_rows, pair_rows, window_rows, tie_rows, clf_rows, sel_rows = [], [], [], [], [], []
    models = [("orb", O, orb_dir, orb_rows), ("uma", U, uma_dir, uma_rows)]

    for t in targets:
        # ---------------- classification targets
        if t in CLF_TARGETS:
            arms = []
            for name, mod, mdir, rows in models:
                res = _fit_clf_proba(name, mod, mdir, rows, t, args.seed, args.n_jobs)
                if res is None:
                    print(f"  [skip] {t} ({name}): no best config"); continue
                arms.append((name, "test", res))
                tc, vc = vsel.get((name, t), (None, None))
                if vc and vc != tc and parse_cfg(name, vc):
                    arms.append((name, "val", _fit_clf_proba(name, mod, mdir, rows, t, args.seed, args.n_jobs, cfg=parse_cfg(name, vc))))
            if not args.skip_magpie and MAGPIE_NPZ.exists():
                mp, y_int, _, _, proba, classes = _magpie_predictions(orb_dir, t, args.seed, args.n_jobs)
                arms.append(("magpie", "test", (mp, y_int, proba, classes, "magpie/xgb", float("nan"))))
            for name, sel, (mp, y_int, proba, classes, cfg_str, mp_score) in arms:
                print(f"  {t:24s} {name:6s} [{sel}] cfg={cfg_str:24s} macro-F1(MP)={mp_score:.3f}  n_test={len(y_int):,}")
                for ci, cname in enumerate(classes):
                    is_c = (y_int == ci).astype(int)
                    auroc = float(roc_auc_score(is_c, proba[:, ci])) if 0 < is_c.sum() < len(is_c) else float("nan")
                    for frac in TOP_FRACS:
                        k, prec, rec_, lift, base = class_enrichment_at_k(y_int, proba, ci, frac)
                        clf_rows.append(dict(target=t, model=name, selection=sel, config=cfg_str, **{"class": cname},
                                             base_rate=round(base, 4), frac=frac, k=k, precision=round(prec, 4),
                                             recall=round(rec_, 4), lift=round(lift, 2), auroc=round(auroc, 4)))
            continue

        # ---------------- regression targets: test-selected arms (+ Magpie), aligned and paired
        preds = {}
        for name, mod, mdir, rows in models:
            rec = _config_and_predictions(name, mod, mdir, rows, t, args.seed, args.n_jobs)
            if rec is None:
                print(f"  [skip] {t} ({name}): no best config"); continue
            preds[name] = rec
        if not args.skip_magpie and MAGPIE_NPZ.exists():
            mp, yt_m, yp_m, _, _, _ = _magpie_predictions(orb_dir, t, args.seed, args.n_jobs)
            preds["magpie"] = dict(mp=mp, yt=yt_m.astype(float), yp=yp_m.astype(float), cfg="magpie/xgb", score=float("nan"))
        if not preds:
            continue
        yt = align_common(preds)
        for name, rec in preds.items():
            print(f"  {t:24s} {name:6s} [test] cfg={rec['cfg']:24s} spearman_rho={rec['rho']:.3f}  n_test={len(yt):,}")
        evaluate_regression(t, preds, yt, args.boot, args.seed, "test", curve_rows, pair_rows, tie_rows)
        if t == "band_gap":
            for name, rec in preds.items():
                w = window_retrieval(rec["yt"], rec["yp"], *BAND_GAP_WINDOW)
                window_rows.append(dict(target=t, model=name, selection="test", config=rec["cfg"], lo=BAND_GAP_WINDOW[0], hi=BAND_GAP_WINDOW[1],
                                        spearman_rho=round(rec["rho"], 4), **w))

        # ---------------- validation-selected arms, where the configuration differs
        preds_val = {}
        for name, mod, mdir, rows in models:
            tc, vc = vsel.get((name, t), (None, None))
            if not vc or vc == tc or name not in preds:
                continue
            cfg = parse_cfg(name, vc)
            rec = _config_and_predictions(name, mod, mdir, rows, t, args.seed, args.n_jobs, cfg=cfg) if cfg else None
            if rec is not None:
                preds_val[name] = rec
        if preds_val:
            # align the val arm onto the test arm's materials so test-vs-val is a paired difference
            for name, rec in preds_val.items():
                pos = {m: i for i, m in enumerate(rec["mp"])}
                idx = np.array([pos[m] for m in preds[name]["mp"] if m in pos])
                rec["yt"], rec["yp"], rec["mp"] = rec["yt"][idx], rec["yp"][idx], preds[name]["mp"][:len(idx)]
                rec["rho"] = float(spearmanr(rec["yt"], rec["yp"])[0])
                print(f"  {t:24s} {name:6s} [val]  cfg={rec['cfg']:24s} spearman_rho={rec['rho']:.3f}  (test-selected was {preds[name]['cfg']})")
            evaluate_regression(t, preds_val, yt, args.boot, args.seed, "val", curve_rows, pair_rows, tie_rows, pairs=())
            for name, rec in preds_val.items():
                if t == "band_gap":            # a window target: no percentile tail or hull threshold applies
                    wt = window_retrieval(yt, preds[name]["yp"], *BAND_GAP_WINDOW)
                    wv = window_retrieval(yt, rec["yp"], *BAND_GAP_WINDOW)
                    window_rows.append(dict(target=t, model=name, selection="val", config=rec["cfg"], lo=BAND_GAP_WINDOW[0], hi=BAND_GAP_WINDOW[1],
                                            spearman_rho=round(rec["rho"], 4), **wv))
                    for k in ("precision", "recall"):
                        f = lambda ya, pt_, pv_, kk=k: (window_retrieval(ya, pt_, *BAND_GAP_WINDOW)[kk]
                                                        - window_retrieval(ya, pv_, *BAND_GAP_WINDOW)[kk])
                        if args.boot:
                            pt, lo, hi, _ = paired_bootstrap(f, [yt, preds[name]["yp"], rec["yp"]], args.boot, args.seed)
                        else:
                            pt, lo, hi = wt[k] - wv[k], float("nan"), float("nan")
                        sel_rows.append(dict(target=t, model=name, frac=float("nan"), criterion=f"window {k}", test_cfg=preds[name]["cfg"],
                                             val_cfg=rec["cfg"], test_minus_val=round(pt, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))
                    continue
                for frac in HEADLINE_FRACS:
                    if t in TARGET_DIRECTION:
                        d = TARGET_DIRECTION[t]
                        f = lambda ya, pt_, pv_, fr=frac, dd=d: recall_at_k(ya, pt_, fr, dd) - recall_at_k(ya, pv_, fr, dd)
                        crit = "recall@k"
                    else:
                        thr = THRESHOLD_TARGETS[t][1]
                        f = lambda ya, pt_, pv_, fr=frac, th=thr: threshold_at_k(ya, pt_, fr, th)[0] - threshold_at_k(ya, pv_, fr, th)[0]
                        crit = f"precision@k, E-hull <= {thr:.2f}"
                    if args.boot:
                        pt, lo, hi, boots = paired_bootstrap(f, [yt, preds[name]["yp"], rec["yp"]], args.boot, args.seed)
                    else:
                        pt, lo, hi = f(yt, preds[name]["yp"], rec["yp"]), float("nan"), float("nan")
                    sel_rows.append(dict(target=t, model=name, frac=frac, criterion=crit, test_cfg=preds[name]["cfg"], val_cfg=rec["cfg"],
                                         test_minus_val=round(pt, 4), ci_lo=round(lo, 4), ci_hi=round(hi, 4)))

    # ---- write ----
    def dump(rows, name):
        if rows:
            keys = []
            for r in rows:
                for k in r:
                    if k not in keys: keys.append(k)
            with open(out_dir / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    dump(curve_rows, "enrichment_curve.csv"); dump(pair_rows, "enrichment_pairs.csv"); dump(window_rows, "band_gap_window_retrieval.csv")
    dump(tie_rows, "ties.csv"); dump(clf_rows, "enrichment_classification.csv"); dump(sel_rows, "selection_sensitivity.csv")
    lift_curve_plot([r for r in curve_rows if r["selection"] == "test"], out_dir)

    lines = ["# Screening enrichment", "",
             "Fixed seed-42 test set, top XGBoost config per model (selected on MP's test split, the paper's rule),",
             "plus the Magpie composition baseline refit on the same split. `lift` = recall@k / k for percentile",
             "tails (random = 1) and precision / base-rate for the E-hull threshold criterion. 95% intervals: paired",
             f"bootstrap over test materials, B = {args.boot}. Full curves in `lift_curves.png`.", "",
             "## Headline points (1 / 5 / 10 %), test-selected configurations", "",
             "| Target | Criterion | Model | Frac | Recall or precision | Lift [95% CI] | Spearman rho |", "|---|---|---|---|---|---|---|"]
    for r in curve_rows:
        if r["frac"] in HEADLINE_FRACS and r["selection"] == "test":
            val = r["precision"] if "precision" in r else r["recall"]
            lines.append(f"| {r['target']} | {r['criterion']} | {r['model']} | {int(r['frac']*100)}% | {val:.3f} | {r['lift']:.2f}x [{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] | {r['spearman_rho']:.3f} |")
    if pair_rows:
        lines += ["", "## Paired differences in recall@k (test-selected)", "", "| Target | Frac | Pair | Delta recall | 95% CI | p |", "|---|---|---|---|---|---|"]
        for r in pair_rows:
            if r["frac"] in HEADLINE_FRACS and r["selection"] == "test":
                lines.append(f"| {r['target']} | {int(r['frac']*100)}% | {r['pair']} | {r['delta_recall']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['p_boot']:.4f} |")
    lines += ["", "## Selection sensitivity: test-selected vs validation-selected configuration", "",
              "The validation-selection control (stages/controls/baselines.py) re-chooses each configuration on a validation split. For 15 of",
              "18 (model, target) pairs it returns the same configuration, so every number above is unchanged. Where it",
              "differs, the metric is recomputed at the validation-selected configuration; `test_minus_val` is the paired",
              "difference (positive = the test-selected configuration looks better on the test tail).", ""]
    if sel_rows:
        lines += ["| Target | Model | Frac | Criterion | Test cfg | Val cfg | Test - val | 95% CI |", "|---|---|---|---|---|---|---|---|"]
        for r in sel_rows:
            fr = f"{int(r['frac']*100)}%" if r["frac"] == r["frac"] else "-"     # band-gap window rows have no fraction
            lines.append(f"| {r['target']} | {r['model']} | {fr} | {r['criterion']} | {r['test_cfg']} | {r['val_cfg']} | {r['test_minus_val']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] |")
    else:
        lines.append("_No differing configurations among the targets run, or the control table was not found._")
    if tie_rows:
        lines += ["", "## Tie diagnostics", "", "Share of the test set tied at the true top-k cutoff. A share above k itself makes the",
                  "percentile tail arbitrary; energy above hull is the case (22% exactly on the hull).", "",
                  "| Target | Frac | Cutoff value | Share tied | Ambiguous |", "|---|---|---|---|---|"]
        for r in tie_rows:
            fr = f"{int(r['frac']*100)}%" if r.get("frac") else "-"
            lines.append(f"| {r['target']} | {fr} | {r['cutoff_value']:.4f} | {r['share_tied_at_cutoff']:.3f} | {r['ambiguous']} |")
    if window_rows:
        lines += ["", "## Band-gap window (0.5-3.0 eV)", "", "| Model | Config | n in window | Precision | Recall | F1 |", "|---|---|---|---|---|---|"]
        for r in window_rows:
            lines.append(f"| {r['model']} | {r['config']} | {r['n_true_in_window']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")
    if clf_rows:
        lines += ["", "## Classification targets: class enrichment", "",
                  "Rank by P(class); `precision` = share of the top-k% truly in the class, `lift` = precision / base rate,",
                  "`auroc` = one-vs-rest ranking quality. `selection` marks validation-selected configurations where they differ.", "",
                  "| Target | Model | Sel. | Class | Base rate | Frac | Precision | Recall | Lift | AUROC |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in clf_rows:
            if r["frac"] in HEADLINE_FRACS:
                lines.append(f"| {r['target']} | {r['model']} | {r['selection']} | {r['class']} | {r['base_rate']:.3f} | {int(r['frac']*100)}% | {r['precision']:.3f} | {r['recall']:.3f} | {r['lift']:.2f}x | {r['auroc']:.3f} |")
    (out_dir / "enrichment_table.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
