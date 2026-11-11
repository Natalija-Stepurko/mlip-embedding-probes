#!/usr/bin/env python3
"""
stages/transfer/label_shift.py -- how much of the external-transfer loss is the labels alone.

For the crystals an external database (JARVIS-DFT 3D, OQMD) shares with MP (cited MP id or StructureMatcher twin)
AND that sit in MP's seed-42 TEST split for the target (so the probe never trained on them), the MP-trained probe of
each target's selected configuration (the one transfer.py transfers) is scored three ways on identical crystals:

  S0  MP structure,        MP label         the in-distribution reference on these crystals
  S1  external structure,  MP label         the external database's own relaxed geometry, MP's labels
  S2  external structure,  external label   = transfer on shared crystals
  S3  MP structure,        external label   MP's geometry, the external database's labels

Two factors change between S0 and S2 (the structure the probe sees, the labels it is scored against), so the
attribution is made order-independent (Shapley over the two factors):

  geometry = 1/2 [(S0 - S1) + (S3 - S2)]        labels = 1/2 [(S0 - S3) + (S1 - S2)]        geometry + labels = S0 - S2

The sequential pair (S0 - S1, S1 - S2) is kept as well; S1 - S2 is the same predictions scored against two label
sets. Macro-F1 uses the fixed class list of the MP probe, so the classes averaged do not change between scores.
For regression every score is also given as MSE (its decomposition has no changing variance denominator) and RMSE.
Paired-bootstrap 95 % intervals resample MP materials, not rows, because several external entries can map to one MP
crystal. For context each row carries the MP label scored as a prediction of the external label on the same
crystals, the MP test score, and transfer.py's disjoint-slice transfer and within-database scores. R^2 and macro-F1
depend on the scored set, so the shared-crystal effects and the disjoint-slice scores are read side by side, not
as an exact sum: the step from shared to disjoint crystals changes structures, labels and target distributions
together.

Writes $MLIP_PROBES_ROOT/results/external_transfer/label_shift_decomposition.csv

    uv run python stages/transfer/label_shift.py --n-jobs 6
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
T = importlib.import_module("stages.transfer.transfer")
O, U, qc = T.O, T.U, T.qc

DBS = ["jarvis_dft_3d", "oqmd"]
OUT = (P.RESULTS / "external_transfer/label_shift_decomposition.csv")


def mp_features(model, mod, mdir, target, cfg):
    """MP pooled features at cfg with ids and labels, as transfer.py's mp_train loads them."""
    task = "clf" if mod.is_clf(target) else "reg"
    feat_dir = mdir / "features"
    if model == "orb":
        rep, layer = cfg
        canon, ydict = O.load_canonical_targets(feat_dir, [target])
        d = np.load(feat_dir / f"{layer}_pooled.npz", allow_pickle=True)
        X = O.build_X(d["node"], d["edge"], rep); ids = np.asarray(canon, dtype=object)
        y = ydict[target]
    else:
        group, fam, layer = cfg
        canon, ydict = U.load_targets(feat_dir, [target])
        X, ids = U.load_feature_set(group, fam, layer)
        pos = {m: i for i, m in enumerate(canon)}
        y = np.array([ydict[target][pos[m]] for m in ids], dtype=object if task == "clf" else float)
    return X, np.asarray(ids, dtype=object), y, task


def scorer(task, n_classes=None):
    if task == "clf":
        labels = np.arange(n_classes) if n_classes else None
        return lambda a, b: f1_score(a, b, average="macro", labels=labels, zero_division=0)
    return r2_score


def mse(a, b):
    return float(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2))


def boot_ci(stat, groups, B, rng):
    """95 % interval of stat(row index) under resampling of MP materials (groups: one id per row)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        ix = np.concatenate([members[g] for g in pick])
        out[b] = stat(ix)
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="orb,uma")
    ap.add_argument("--targets", default=",".join(T.NINE))
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed + 17)
    orb_dir, uma_dir = (P.RESULTS / "orb/property_prediction"), (P.RESULTS / "fairchem/property_prediction")
    mods = {"orb": (O, orb_dir), "uma": (U, uma_dir)}
    metrics = {"orb": O.load_metrics(orb_dir / "metrics.csv"), "uma": U.load_metrics(uma_dir / "metrics.csv")}
    exts = {}
    for ds in DBS:
        structs, ids, rel = T.load_external(P.STRUCTURES_EXTERNAL / ds / "index.jsonl")
        exts[ds] = (T.Ext(ds, structs, ids, rel, (P.RESULTS / "external") / ds), structs)
    tr14 = {ds: pd.read_csv((P.RESULTS / "external_transfer") / ds / "transfer.csv") for ds in DBS}
    rows = []
    for target in args.targets.split(","):
        for model in args.models.split(","):
            mod, mdir = mods[model]
            cfg, mp_score = mod.best_config(metrics[model], target)
            t0 = time.time()
            X, ids, y, task = mp_features(model, mod, mdir, target, cfg)
            sp = np.load(mdir / "splits" / f"{mod.safe(target)}.npz", allow_pickle=True)
            pos = {m: i for i, m in enumerate(ids)}
            test_set = set(map(str, sp["test_mp_ids"]))
            est, le, task, _, _, _ = T.mp_train(model, mod, mdir, target, cfg, args.seed, args.n_jobs)
            print(f"{target} [{model}] {'/'.join(map(str, cfg))}: probe fitted ({time.time() - t0:.0f}s)", flush=True)
            for ds in DBS:
                ext, structs = exts[ds]
                y_ext_all = T.external_y(structs, ext.ids, target)
                if int(T._label_ok(task, y_ext_all).sum()) < 200:
                    continue
                # shared crystals in MP's test split, with their MP id
                pairs = []
                for e in ext.ids:
                    r = structs[e]
                    m = r.get("mpid_reported") if r.get("in_mp_corpus") else r.get("structure_match_mp")
                    if m and m in test_set and m in pos:
                        pairs.append((e, m))
                if model == "orb":
                    rep, layer = cfg
                    d = np.load((P.RESULTS / "external") / ds / "orb" / "features" / f"{layer}_pooled.npz", allow_pickle=True)
                    Xe, ide = O.build_X(d["node"], d["edge"], rep), d["mp_ids"]
                else:
                    group, fam, layer = cfg
                    Xe, ide = T.uma_external_X((P.RESULTS / "external") / ds / "fairchem", group, fam, layer,
                                               np.array([p[0] for p in pairs], dtype=object), args.n_jobs)
                epos = {m: i for i, m in enumerate(ide)}
                pairs = [(e, m) for e, m in pairs if e in epos]
                if len(pairs) < 100:
                    print(f"   {ds}: {len(pairs)} shared test crystals, skipped"); continue
                ei = np.array([epos[e] for e, _ in pairs]); mi = np.array([pos[m] for _, m in pairs])
                y_mp = y[mi]; y_ex = y_ext_all[np.array([ext.pos[e] for e, _ in pairs])]
                ok = np.isfinite(X[mi]).all(1) & np.isfinite(Xe[ei]).all(1) & T._label_ok(task, y_mp) & T._label_ok(task, y_ex)
                if task == "clf":
                    ok &= np.isin(y_mp, le.classes_) & np.isin(y_ex, le.classes_)
                ei, mi, y_mp, y_ex = ei[ok], mi[ok], y_mp[ok], y_ex[ok]
                if task == "clf":
                    y_mp, y_ex = le.transform(y_mp), le.transform(y_ex)
                else:
                    y_mp, y_ex = y_mp.astype(float), y_ex.astype(float)
                p_mp, p_ex = est.predict(X[mi]), est.predict(Xe[ei])
                f = scorer(task, len(le.classes_) if task == "clf" else None)
                groups = np.array([m for _, m in pairs], dtype=object)[ok]
                def four(ix, g=f):
                    return (g(y_mp[ix], p_mp[ix]), g(y_mp[ix], p_ex[ix]), g(y_ex[ix], p_ex[ix]), g(y_ex[ix], p_mp[ix]))
                s0, s1, s2, s3 = four(np.arange(len(y_mp)))
                sym_geo = lambda ix, g=f: 0.5 * ((four(ix, g)[0] - four(ix, g)[1]) + (four(ix, g)[3] - four(ix, g)[2]))
                sym_lab = lambda ix, g=f: 0.5 * ((four(ix, g)[0] - four(ix, g)[3]) + (four(ix, g)[1] - four(ix, g)[2]))
                geo_ci = boot_ci(lambda ix: four(ix)[0] - four(ix)[1], groups, args.boot, rng)
                lab_ci = boot_ci(lambda ix: four(ix)[1] - four(ix)[2], groups, args.boot, rng)
                sgeo_ci = boot_ci(sym_geo, groups, args.boot, rng)
                slab_ci = boot_ci(sym_lab, groups, args.boot, rng)
                extra = {}
                if task == "reg":
                    m0, m1, m2, m3 = four(np.arange(len(y_mp)), mse)
                    from scipy.stats import spearmanr
                    extra = dict(mse0=m0, mse1=m1, mse2=m2, mse3=m3, rmse0=m0 ** 0.5, rmse2=m2 ** 0.5,
                                 # the three errors on one matched subset: probe vs MP label, MP label vs external label, probe vs external label
                                 mae_probe_vs_mp_label=float(np.mean(np.abs(p_mp - y_mp))),
                                 mae_mp_label_vs_ext_label=float(np.mean(np.abs(y_mp - y_ex))),
                                 rmse_mp_label_vs_ext_label=float(np.sqrt(np.mean((y_mp - y_ex) ** 2))),
                                 bias_ext_minus_mp_label=float(np.mean(y_ex - y_mp)),
                                 mae_probe_vs_ext_label=float(np.mean(np.abs(p_ex - y_ex))),
                                 spearman_probe_vs_mp_label=float(spearmanr(p_mp, y_mp)[0]),
                                 spearman_mp_label_vs_ext_label=float(spearmanr(y_mp, y_ex)[0]),
                                 spearman_probe_vs_ext_label=float(spearmanr(p_ex, y_ex)[0]),
                                 mse_geometry_sym=0.5 * ((m1 - m0) + (m2 - m3)), mse_labels_sym=0.5 * ((m3 - m0) + (m2 - m1)),
                                 sd_mp_label=float(np.std(y_mp)), sd_ext_label=float(np.std(y_ex)))
                t = tr14[ds]; t = t[(t.model == model) & (t.target == target) & (t.selection == "test_selected")].set_index("slice")
                dis = t.loc["disjoint"] if "disjoint" in t.index else None
                row = dict(database=ds, model=model, target=target, config="/".join(map(str, cfg)), metric="macro_f1" if task == "clf" else "r2",
                           n_shared_test=int(len(y_mp)), mp_test_score=float(mp_score),
                           n_mp_materials=int(len(np.unique(groups))),
                           s0_mp_structure_mp_label=s0, s1_ext_structure_mp_label=s1, s2_ext_structure_ext_label=s2,
                           s3_mp_structure_ext_label=s3,
                           geometry_effect=s0 - s1, geometry_lo=geo_ci[0], geometry_hi=geo_ci[1],
                           label_effect=s1 - s2, label_lo=lab_ci[0], label_hi=lab_ci[1],
                           geometry_sym=sym_geo(np.arange(len(y_mp))), geometry_sym_lo=sgeo_ci[0], geometry_sym_hi=sgeo_ci[1],
                           labels_sym=sym_lab(np.arange(len(y_mp))), labels_sym_lo=slab_ci[0], labels_sym_hi=slab_ci[1],
                           **extra,
                           mp_label_as_prediction=f(y_ex, y_mp),
                           disjoint_transfer=float(dis.transfer_score) if dis is not None else np.nan,
                           disjoint_within=float(dis.within_ext_score) if dis is not None else np.nan)
                row["drop_mp_test_to_disjoint"] = row["mp_test_score"] - row["disjoint_transfer"]
                row["loss_fig2e"] = row["disjoint_within"] - row["disjoint_transfer"]
                rows.append(row)
                print(f"   {ds}: n={len(y_mp):,} ({row['n_mp_materials']:,} MP materials)  S0 {s0:.3f}  S1 {s1:.3f}  S2 {s2:.3f}  S3 {s3:.3f}  "
                      f"labels(sym) {row['labels_sym']:+.3f} [{slab_ci[0]:+.3f}, {slab_ci[1]:+.3f}]  geometry(sym) {row['geometry_sym']:+.3f}  "
                      f"| MP test {mp_score:.3f} -> disjoint {row['disjoint_transfer']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
