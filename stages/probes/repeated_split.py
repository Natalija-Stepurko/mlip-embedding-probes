#!/usr/bin/env python3
"""
stages/probes/repeated_split.py — Repeated-split (retraining) significance
tests for claims that a SINGLE fixed split cannot support.

Why this script exists (complements mlipprobes/significance_lib.py)
------------------------------------------------------------------
mlipprobes/significance_lib.py bootstraps the one fixed 80/20 split (seed 42)
used by the 05 pipelines. That is the right (and CHEAP — no retraining) tool
for any paired, same-test-set claim, because both configs being compared are
each fit ONCE and then their predictions are resampled together:
  - "UMA's prediction is better than ORB's."          -> 06 (paired bootstrap)
  - "XGBoost beats linear."                           -> 06 (paired bootstrap)
  - "Concat outperforms node-only."                   -> 06 (paired bootstrap,
    see orb_rep_analysis / uma_rep_analysis and Table C of significance_tables.md)
06 CANNOT support exactly one kind of claim: split-to-split variability, since
a single fixed split has no split variance by construction —
  - "Layer b3 is significantly better than layer b2."
This is the ONLY claim in scope here, and it needs actual RETRAINING over
several different random splits, using a test that accounts for the fact the
splits share overlapping training data (a plain paired t-test across repeats
would be anti-conservative) — the Nadeau & Bengio (2003) corrected resampled
t-test:
    t = mean(d) / sqrt((1/K + n_test/n_train) * var(d))   ~ t distribution, K-1 dof
where d_1..d_K are the K per-repeat metric differences (same seed used for
both layers in a repeat, so the comparison stays paired).

Scope (curated, NOT the full 96-feature-set grid — retraining is expensive):
  A. Best layer vs WORST layer, forcing the "node" representation — a curated
     set of targets with the clearest single-split layer story (crystal_system,
     band_gap, energy_above_hull, bulk_modulus), both sources. Answers "does
     layer depth actually matter, once inside the network".
     - Representation is forced to "node" rather than each target's
       empirically-winning rep (usually concat): one consistent, cheaper
       feature type instead of mixing whatever happened to win.
     - Best-vs-RUNNER-UP-layer was tried first and dropped: adjacent layers on
       a smooth curve are nearly identical, so that comparison was almost
       always a tie/trivial and uninformative by construction.
     - "Worst" excludes the raw pre-GNN encoder/embed layer. Including it would
       mostly re-answer claim B below, and for some UMA targets embed is
       genuinely the best layer, which would make it a confusing "worst" pick.
  B. Best layer vs the raw encoder/embed layer ITSELF (same curated targets,
     same forced "node" rep) — the complementary question to A: does any GNN
     processing help at all over the raw pre-network input embedding, tested
     by actual retraining rather than inferred indirectly from the
     single-split XGB-vs-linear / baseline-improvement results. Rows are
     skipped when the encoder/embed layer already IS the best layer (no gap
     to test).
(There used to be an "XGBoost vs linear" and "concat vs node" repeated-split
family here — both were removed: they duplicated claims 06 already answers
correctly without retraining, and retraining them was wasted compute.)

K=5 repeats, seeds disjoint from the fixed split's seed 42 (so this genuinely
probes split-to-split variability, not the same partition).

Reads  $MLIP_PROBES_ROOT/results/orb/property_prediction/{metrics.csv,features}
       $MLIP_PROBES_ROOT/results/fairchem/property_prediction/{metrics.csv,features}
Writes $MLIP_PROBES_ROOT/results/proppred_repeated_significance/

Example
-------
  uv run python scripts/stages/probes/repeated_split.py --repeats 5
  uv run python scripts/stages/probes/repeated_split.py --repeats 2 --targets band_gap
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import argparse
import csv
import json
import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["legend.frameon"] = False  # no box border around figure legends
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as tdist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
O = importlib.import_module("stages.probes.probes_orb")
U = importlib.import_module("stages.probes.probes_uma")
S = importlib.import_module("mlipprobes.significance_lib")  # classify_effect, _adaptive_xlim

UNRELIABLE_TARGETS: set[str] = set()      # targets to skip; none of the nine
LAYER_CURATED = ["crystal_system", "band_gap", "energy_above_hull", "bulk_modulus"]
SEEDS_DEFAULT = [101, 102, 103, 104, 105]


# ---------------------------------------------------------------------------
# Repeated-split refit primitives (reuse probes_*.py's fit_eval; new seeds each repeat)
# ---------------------------------------------------------------------------

def orb_config_pair(orb_dir, target, task, cfgA, cfgB, seeds, test_size, n_jobs, model="xgb"):
    """cfgA/cfgB = (rep, layer). Same model, same repeat-seed splits, layer or
    rep may differ between A and B."""
    feat_dir = orb_dir / "features"
    canon, ydict = O.load_canonical_targets(feat_dir, [target])
    y = ydict[target]
    repA, layerA = cfgA
    repB, layerB = cfgB
    dA = np.load(feat_dir / f"{layerA}_pooled.npz", allow_pickle=True)
    dB = dA if layerB == layerA else np.load(feat_dir / f"{layerB}_pooled.npz", allow_pickle=True)
    valid = np.where(O.valid_mask(target, y))[0]
    le = LabelEncoder().fit(y[valid]) if task == "clf" else None
    pm = O.primary_metric_name(target)
    out = []
    for seed in seeds:
        strat = y[valid] if task == "clf" else None
        tr, te = train_test_split(valid, test_size=test_size, random_state=seed, stratify=strat)
        rowA, _ = O.fit_eval(target, task, repA, layerA, model, dA["node"], dA["edge"], y, le, tr, te, seed, n_jobs)
        rowB, _ = O.fit_eval(target, task, repB, layerB, model, dB["node"], dB["edge"], y, le, tr, te, seed, n_jobs)
        if rowA is None or rowB is None:
            continue
        out.append((float(rowA[pm]), float(rowB[pm]), rowA["n_train"], rowA["n_test"]))
    return out


def uma_config_pair(uma_dir, target, task, cfgA, cfgB, seeds, test_size, n_jobs, model="xgb"):
    """cfgA/cfgB = (group, family, layer)."""
    feat_dir = uma_dir / "features"
    canon, ydict = U.load_targets(feat_dir, [target])
    canon_pos = {m: i for i, m in enumerate(canon)}
    y = ydict[target]
    gA, fA, lA = cfgA
    gB, fB, lB = cfgB
    XA, idA = U.load_feature_set(gA, fA, lA)
    posA = {m: i for i, m in enumerate(idA)}
    if (gA, fA, lA) == (gB, fB, lB):
        XB, posB = XA, posA
    else:
        XB, idB = U.load_feature_set(gB, fB, lB)
        posB = {m: i for i, m in enumerate(idB)}
    valid = np.where(U.valid_mask(target, y))[0]
    le = LabelEncoder().fit(y[valid]) if task == "clf" else None
    pm = U.primary_metric_name(target)
    out = []
    for seed in seeds:
        strat = y[valid] if task == "clf" else None
        tr_pos, te_pos = train_test_split(valid, test_size=test_size, random_state=seed, stratify=strat)
        train_ids, test_ids = canon[tr_pos], canon[te_pos]
        rowA, _ = U.fit_eval(target, task, gA, fA, lA, XA, posA, canon_pos, y, le,
                             train_ids, test_ids, seed, n_jobs, model)
        rowB, _ = U.fit_eval(target, task, gB, fB, lB, XB, posB, canon_pos, y, le,
                             train_ids, test_ids, seed, n_jobs, model)
        if rowA is None or rowB is None:
            continue
        out.append((float(rowA[pm]), float(rowB[pm]), rowA["n_train"], rowA["n_test"]))
    return out


# ---------------------------------------------------------------------------
# Nadeau-Bengio corrected resampled t-test
# ---------------------------------------------------------------------------

def nadeau_bengio(pairs, alpha=0.05):
    """pairs: list of (metric_a, metric_b, n_train, n_test). Returns dict with
    mean diff, 95% CI, t, dof, p (two-sided)."""
    diffs = np.array([a - b for a, b, _, _ in pairs], dtype=float)
    n_train = np.mean([p[2] for p in pairs])
    n_test = np.mean([p[3] for p in pairs])
    k = len(diffs)
    d_bar = float(diffs.mean())
    if k < 2:
        return dict(k=k, mean=d_bar, lo=float("nan"), hi=float("nan"),
                    t=float("nan"), dof=0, p=float("nan"), diffs=diffs.tolist())
    s2 = float(diffs.var(ddof=1))
    correction = 1.0 / k + n_test / n_train
    dof = k - 1
    if s2 == 0:
        return dict(k=k, mean=d_bar, lo=d_bar, hi=d_bar, t=float("inf"), dof=dof,
                    p=(0.0 if d_bar != 0 else 1.0), diffs=diffs.tolist())
    se = np.sqrt(correction * s2)
    t_stat = d_bar / se
    p = float(2 * tdist.sf(abs(t_stat), dof))
    tcrit = float(tdist.ppf(1 - alpha / 2, dof))
    lo, hi = d_bar - tcrit * se, d_bar + tcrit * se
    return dict(k=k, mean=d_bar, lo=lo, hi=hi, t=float(t_stat), dof=dof, p=p, diffs=diffs.tolist())


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        i = n - rank
        val = p[idx] * n / i
        prev = min(prev, val)
        adj[idx] = prev
    return adj


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _reliable_targets(orb_rows, uma_rows, targets):
    out = []
    for t in targets:
        if t in UNRELIABLE_TARGETS:
            continue
        if O.best_config(orb_rows, t)[0] is None or U.best_config(uma_rows, t)[0] is None:
            continue
        out.append(t)
    return out


def _scored_layers_orb(rows, target, rep):
    pm = O.primary_metric_name(target)
    scored = [(O._metric_lookup(rows, target, rep, l, "xgb", pm), l) for l in O.ALL_LAYERS]
    scored = [(v, l) for v, l in scored if not np.isnan(v)]
    scored.sort(reverse=True)
    return scored  # best first


def _scored_layers_uma(rows, target, group, family):
    pm = U.primary_metric_name(target)
    scored = [(U._lookup(rows, target, group, family, l, "xgb", pm), l) for l in U.GROUP_LAYERS[group]]
    scored = [(v, l) for v, l in scored if not np.isnan(v)]
    scored.sort(reverse=True)
    return scored


def _best_and_worst(scored, exclude_layer):
    """scored: sorted descending [(value, layer), ...]. Best = unrestricted top
    of the list (kept unrestricted since the raw embed/encoder layer is
    occasionally the genuine best, e.g. geometry-heavy UMA targets). Worst =
    bottom of the list AFTER excluding `exclude_layer` (the pre-GNN raw input
    embedding), so 'worst' means the worst layer that has had some real GNN
    processing, not the trivial/uninteresting pre-network baseline."""
    if not scored:
        return None
    best = scored[0]
    filtered = [(v, l) for v, l in scored if l != exclude_layer]
    if not filtered:
        return None
    return best, filtered[-1]


def _best_and_encoder(scored, encoder_layer):
    """Best (unrestricted) layer vs the raw pre-GNN encoder/embed layer itself
    — the complementary question to _best_and_worst: does any GNN processing
    help at all over the raw input embedding, tested by actual retraining
    (rather than inferred from the single-split XGB-vs-linear /
    baseline-improvement results). Returns None if the encoder layer has no
    score, or if the encoder layer already IS the best (no gap to test)."""
    if not scored:
        return None
    best = scored[0]
    enc = next(((v, l) for v, l in scored if l == encoder_layer), None)
    if enc is None or best[1] == encoder_layer:
        return None
    return best, enc


def main():
    ap = argparse.ArgumentParser(description="Repeated-split (retraining) significance tests.")
    ap.add_argument("--orb-dir", default=str(P.RESULTS / "orb/property_prediction"))
    ap.add_argument("--uma-dir", default=str(P.RESULTS / "fairchem/property_prediction"))
    ap.add_argument("--output-dir", default=str(P.RESULTS / "proppred_repeated_significance"))
    ap.add_argument("--targets", default=None, help="comma list; default = all reliable")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    orb_dir, uma_dir = Path(args.orb_dir), Path(args.uma_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS_DEFAULT[:args.repeats]

    orb_rows = O.load_metrics(orb_dir / "metrics.csv")
    uma_rows = U.load_metrics(uma_dir / "metrics.csv")

    if args.targets:
        targets = [O.TARGET_ALIASES.get(t.strip(), t.strip()) for t in args.targets.split(",")]
    else:
        targets = O.order_targets(O.REG_PROPS + O.CAT_PROPS)
    targets = _reliable_targets(orb_rows, uma_rows, targets)
    # An explicit --targets request is honoured in full (e.g. all reliable
    # targets); with no --targets we fall back to the curated layer subset.
    layer_targets = list(targets) if args.targets else [t for t in LAYER_CURATED if t in targets]

    print(f"  reliable targets: {targets}")
    print(f"  layer-comparison curated targets: {layer_targets}")
    print(f"  repeats={len(seeds)} seeds={seeds}")

    # "XGBoost vs linear" and "concat vs node" are NOT retrained here: both are
    # paired-same-test-set claims that mlipprobes/significance_lib.py already
    # answers correctly and cheaply (single-split refit + paired bootstrap;
    # see that script's orb_xgb_lin fields and orb_rep_analysis /
    # uma_rep_analysis). Only "layer X vs layer Y" is a genuine split-
    # robustness claim requiring retraining (a single split has no
    # split-to-split variance by construction).
    # Representation is forced to "node" for this comparison (rather than each
    # target's empirically-winning rep, usually concat) — one consistent,
    # cheaper feature type, so the layer-depth question is asked the same way
    # everywhere instead of mixing whatever representation happened to win.
    layer_recs = []
    encoder_recs = []
    for t in layer_targets:
        task = "clf" if O.is_clf(t) else "reg"

        o_rep = "node"
        scored = _scored_layers_orb(orb_rows, t, o_rep)
        bw = _best_and_worst(scored, exclude_layer="encoder")
        if bw:
            (_, best_l), (_, worst_l) = bw
            print(f"  [layer] {t} ORB/{o_rep}: best={best_l} worst={worst_l} ...")
            pairs = orb_config_pair(orb_dir, t, task, (o_rep, best_l), (o_rep, worst_l),
                                    seeds, args.test_size, args.n_jobs)
            if pairs:
                r = nadeau_bengio(pairs)
                r.update(target=t, source="ORB", cfg=o_rep,
                        a=f"{O.LAYER_LABELS[best_l]}", b=f"{O.LAYER_LABELS[worst_l]}")
                layer_recs.append(r)
        be = _best_and_encoder(scored, encoder_layer="encoder")
        if be:
            (_, best_l), (_, enc_l) = be
            print(f"  [best-v-encoder] {t} ORB/{o_rep}: best={best_l} encoder={enc_l} ...")
            pairs = orb_config_pair(orb_dir, t, task, (o_rep, best_l), (o_rep, enc_l),
                                    seeds, args.test_size, args.n_jobs)
            if pairs:
                r = nadeau_bengio(pairs)
                r.update(target=t, source="ORB", cfg=o_rep,
                        a=f"{O.LAYER_LABELS[best_l]}", b=f"{O.LAYER_LABELS[enc_l]}")
                encoder_recs.append(r)

        u_group = "node"
        ufam_cfg, _ = U.best_within_group(uma_rows, t, u_group)
        if ufam_cfg is not None:
            u_fam, _ = ufam_cfg
            scored = _scored_layers_uma(uma_rows, t, u_group, u_fam)
            bw = _best_and_worst(scored, exclude_layer="embed")
            if bw:
                (_, best_l), (_, worst_l) = bw
                print(f"  [layer] {t} UMA/{u_group}/{u_fam}: best={best_l} worst={worst_l} ...")
                pairs = uma_config_pair(uma_dir, t, task, (u_group, u_fam, best_l), (u_group, u_fam, worst_l),
                                        seeds, args.test_size, args.n_jobs)
                if pairs:
                    r = nadeau_bengio(pairs)
                    r.update(target=t, source="UMA", cfg=f"{u_group}/{u_fam}",
                            a=f"{U.LAYER_LABELS[best_l]}", b=f"{U.LAYER_LABELS[worst_l]}")
                    layer_recs.append(r)
            be = _best_and_encoder(scored, encoder_layer="embed")
            if be:
                (_, best_l), (_, enc_l) = be
                print(f"  [best-v-encoder] {t} UMA/{u_group}/{u_fam}: best={best_l} encoder={enc_l} ...")
                pairs = uma_config_pair(uma_dir, t, task, (u_group, u_fam, best_l), (u_group, u_fam, enc_l),
                                        seeds, args.test_size, args.n_jobs)
                if pairs:
                    r = nadeau_bengio(pairs)
                    r.update(target=t, source="UMA", cfg=f"{u_group}/{u_fam}",
                            a=f"{U.LAYER_LABELS[best_l]}", b=f"{U.LAYER_LABELS[enc_l]}")
                    encoder_recs.append(r)

    if layer_recs:
        q = bh_fdr([r["p"] for r in layer_recs])
        for r, qq in zip(layer_recs, q):
            r["q"] = float(qq)
    if encoder_recs:
        q = bh_fdr([r["p"] for r in encoder_recs])
        for r, qq in zip(encoder_recs, q):
            r["q"] = float(qq)

    _write_outputs(layer_recs, encoder_recs, out_dir, seeds)
    print("\nAll done.")


def _fmt_ci(pt, lo, hi):
    if np.isnan(lo):
        return f"{pt:+.3f} [n/a]"
    return f"{pt:+.3f} [{lo:+.3f},{hi:+.3f}]"


def _write_outputs(layer_recs, encoder_recs, out_dir: Path, seeds):
    # "diffs" persists the K per-repeat metric differences (JSON list) so the
    # paper figures can render the individual per-repeat points, not just mean+CI.
    cols = ["target", "source", "cfg", "a", "b", "k", "mean", "lo", "hi", "t", "dof", "p", "q", "diffs"]

    def _prep(recs):
        out = []
        for r in recs:
            rr = dict(r)
            rr["diffs"] = json.dumps([round(float(x), 6) for x in r.get("diffs", [])])
            out.append(rr)
        return out

    if layer_recs:
        with (out_dir / "repeated_layer_vs_layer.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(_prep(layer_recs))
    if encoder_recs:
        with (out_dir / "repeated_best_vs_encoder.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(_prep(encoder_recs))

    L = [f"Repeated-split (retraining) significance — Nadeau-Bengio corrected t-test, K={len(seeds)} "
         f"repeats, seeds={seeds}",
         "Diffs computed per repeat with the SAME split for both configs (paired); CI/p correct for",
         "the shared-training-data non-independence across repeats (plain t-test would be anti-conservative).",
         "Two layer-depth claims are tested here — the only claims that genuinely need retraining.",
         "'XGBoost vs linear' and 'concat vs node' are answered correctly and cheaply (no retraining) in",
         "mlipprobes/significance_lib.py's paired bootstrap tests. Representation is forced to 'node' (not",
         "each target's empirically-winning rep) for a consistent, cheaper comparison across both claims.",
         "  A. Best layer vs WORST layer, excluding the raw pre-GNN encoder/embed layer from 'worst' —",
         "     does layer depth matter once inside the network.",
         "  B. Best layer vs the raw encoder/embed layer ITSELF — does any GNN processing help at all",
         "     over the raw input embedding (complementary to A; tested by actual retraining rather than",
         "     inferred from the single-split XGB-vs-linear / baseline-improvement results). Rows are",
         "     dropped when the encoder/embed layer already IS the best layer (no gap to test).",
         "BH-FDR q is computed separately within each of the two families (A, B).", ""]

    for recs, heading, csv_name in [(layer_recs, "A. Best layer vs WORST layer, excluding raw encoder/embed "
                                     "(node representation)", None),
                                    (encoder_recs, "B. Best layer vs the raw encoder/embed layer itself "
                                     "(node representation)", None)]:
        if not recs:
            continue
        L += ["=" * 100, heading, "=" * 100,
              f"{'target':22s} {'src':5s} {'rep/config':14s} {'A vs B':13s} "
              f"{'mean Delta A-B (95% CI, NB)':28s} {'k':>3s} {'p':>9s} {'q':>9s} {'effect':22s}"]
        for r in recs:
            eff = S.classify_effect(r["mean"], r["lo"], r["hi"], pos=r["a"], neg=r["b"])
            L.append(f"{r['target']:22s} {r['source']:5s} {r['cfg']:14s} {r['a']+' v '+r['b']:13s} "
                     f"{_fmt_ci(r['mean'], r['lo'], r['hi']):28s} {r['k']:3d} "
                     f"{r['p']:9.2e} {r['q']:9.2e} {eff:22s}")
        L.append("")

    (out_dir / "repeated_significance_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))

    if layer_recs:
        _plot_repeated(layer_recs, out_dir / "repeated_layer_vs_layer.png",
                       "best layer − worst layer", "best", "worst",
                       "Best vs worst layer (node rep, excluding raw encoder/embed) — "
                       "repeated-split robustness (K retrains, Nadeau-Bengio CI)")
        _table_layer_results(layer_recs, out_dir / "repeated_layer_vs_layer_table.png",
                             "best vs worst",
                             "Best vs worst layer, excluding raw encoder/embed (node representation)\n"
                             "Nadeau-Bengio corrected repeated-split test, K retrains per row")
    if encoder_recs:
        _plot_repeated(encoder_recs, out_dir / "repeated_best_vs_encoder.png",
                       "best layer − encoder/embed", "best", "encoder",
                       "Best layer vs raw encoder/embed (node rep) — "
                       "repeated-split robustness (K retrains, Nadeau-Bengio CI)")
        _table_layer_results(encoder_recs, out_dir / "repeated_best_vs_encoder_table.png",
                             "best vs encoder/embed",
                             "Best layer vs raw encoder/embed layer itself (node representation)\n"
                             "Nadeau-Bengio corrected repeated-split test, K retrains per row")
    print(f"\n  outputs -> {out_dir}")


def _table_layer_results(recs, out_path: Path, col_label, title):
    """Rendered table figure (the text summary as an image, for slides/reports)
    for one comparison family (best-vs-worst or best-vs-encoder)."""
    table = []
    for r in recs:
        eff = S.classify_effect(r["mean"], r["lo"], r["hi"], pos=r["a"], neg=r["b"])
        table.append([
            r["target"], r["source"], r["cfg"], f"{r['a']} vs {r['b']}",
            _fmt_ci(r["mean"], r["lo"], r["hi"]), str(r["k"]),
            f"{r['p']:.2e}", f"{r['q']:.2e}", eff,
        ])
    col_w = [0.11, 0.05, 0.11, 0.12, 0.17, 0.04, 0.09, 0.09, 0.22]
    fig, ax = plt.subplots(figsize=(17, 0.5 * len(table) + 1.6)); ax.axis("off")
    tbl = ax.table(cellText=table,
                   colLabels=["target", "src", "rep/config", col_label,
                              "mean Δ (95% CI, NB)", "k", "p", "q", "effect"],
                   colWidths=col_w, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.6)
    ax.set_title(title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [table] -> {out_path}")


def _plot_repeated(recs, out_path: Path, xlabel, pos, neg, title):
    """Strip plot of the K raw per-repeat differences + Nadeau-Bengio mean/CI
    as a diamond marker — visually distinct from the significance forest plots
    (circle marker, single fixed split) to avoid conflating the two designs."""
    rr = list(reversed(recs))
    y = np.arange(len(rr))
    los = [r["lo"] for r in rr if not np.isnan(r["lo"])]
    his = [r["hi"] for r in rr if not np.isnan(r["hi"])]
    xlim = S._adaptive_xlim(los or [0], his or [0], -0.05, 0.1)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(11.5, 0.6 * len(rr) + 1.8))
    for i, r in enumerate(rr):
        diffs = r["diffs"]
        jitter = rng.uniform(-0.15, 0.15, size=len(diffs))
        ax.scatter(diffs, np.full(len(diffs), i) + jitter, color="0.5", s=16,
                  alpha=0.7, zorder=2, label="_nolegend_")
        eff = S.classify_effect(r["mean"], r["lo"], r["hi"], pos=pos, neg=neg)
        c = S._effect_color(eff, pos)
        if not np.isnan(r["lo"]):
            ax.plot([r["lo"], r["hi"]], [i, i], color=c, lw=2.5, zorder=1)
        ax.plot(r["mean"], i, "D", color=c, ms=7, zorder=3, markeredgecolor="k", markeredgewidth=0.5)
        ax.text(xlim[1], i, f"  {r['mean']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]  {eff}  (k={r['k']})",
                va="center", fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(*xlim)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['target']} — {r['source']}" for r in rr], fontsize=8)
    ax.set_xlabel(f"Δ metric = {xlabel}  (grey dots = each repeat; diamond = Nadeau-Bengio mean + 95% CI)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
