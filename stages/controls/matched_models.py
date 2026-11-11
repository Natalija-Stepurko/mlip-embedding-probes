#!/usr/bin/env python3
"""
stages/controls/matched_models.py — is the ORB/UMA contrast architecture or training data? Two controlled pairs
on one 20,000-material subsample of the paper corpus, probed exactly as in 05 (XGBoost, the
paper's seed-42 split restricted to the subsample, all nine targets):

  matched DATA, different architecture      MACE-OMAT-0 (equivariant, trained on OMat24)
                                            vs ORB-v3-omat (non-equivariant, OMat24)
  matched ARCHITECTURE, different data      ORB-v3-mpa (MPtrj + Alexandria) vs ORB-v3-omat

UMA-S (equivariant, five datasets) and the Magpie composition baseline are probed on the same
rows for reference. Per arm every available layer / read-out is probed and the best is reported;
headline pairs get a paired bootstrap on the shared test rows.

Passes
  --make-subsample   writes <out>/subsample_ids.txt (stratified by crystal system, seed 42)
  (default)          probes every arm whose features exist; resumable per (arm, config, target)

Reads  $MLIP_PROBES_ROOT/results/orb/property_prediction/{features,splits}      ORB-omat pooled features, labels, splits
       $MLIP_PROBES_ROOT/results/orb_mpa/                                        stages/extract/extract_orb.py --orb-variant mpa
       $MLIP_PROBES_ROOT/results/fairchem/plots/...                              UMA pooled caches (via 05)
       $MLIP_PROBES_ROOT/results/mace_omat/pooled/layer_{i}_{family}.npz          stages/extract/extract_mace.py
       $MLIP_PROBES_ROOT/results/exp_magpie/magpie_features.npz
Writes $MLIP_PROBES_ROOT/results/matched_data/{subsample_ids.txt, metrics.csv, best_by_arm.csv, pairs.csv, README.md}
Usage
    uv run python stages/controls/matched_models.py --make-subsample
    uv run python stages/controls/matched_models.py --n-jobs 6
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
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
O = importlib.import_module("stages.probes.probes_orb")
U = importlib.import_module("stages.probes.probes_uma")
S = importlib.import_module("mlipprobes.significance_lib")
T14 = importlib.import_module("stages.transfer.transfer")
qc = importlib.import_module("mlipprobes.common")

TARGETS = ["formation_energy/atom", "energy_above_hull", "band_gap", "density", "bulk_modulus",
           "mag_density", "crystal_system", "metal_insulator", "magnetic_label"]
ARMS = {   # arm -> (architecture, training data, equivariant)
    "orb_omat": ("ORB-v3", "OMat24", False),
    "orb_mpa": ("ORB-v3", "MPtrj+Alexandria", False),
    "mace_omat": ("MACE-OMAT-0", "OMat24", True),
    "uma": ("UMA-S", "OMat24+OC20+ODAC23+OMC25+OMol25", True),
    "magpie": ("Magpie", "-", None),
}
UMA_NODE_FAMILIES = ["full", "l0", "ang_norm", "l1_polar", "l2_polar", "geom_norm"]
PAIRS = [("mace_omat", "orb_omat", "matched data (OMat24): equivariant vs non-equivariant"),
         ("orb_mpa", "orb_omat", "matched architecture (ORB-v3): MPtrj+Alexandria vs OMat24"),
         ("uma", "orb_omat", "the paper's pair: UMA-S vs ORB-v3-omat")]


def make_subsample(orb_dir, out, n, seed):
    canon, ydict = O.load_canonical_targets(orb_dir / "features", ["crystal_system"])
    canon = np.asarray(canon).astype(str)
    excluded = qc.excluded_materials()
    keep = np.array([m not in excluded for m in canon])
    canon, y = canon[keep], np.asarray(ydict["crystal_system"], dtype=object)[keep]
    rng = np.random.default_rng(seed)
    classes = np.array([str(v) for v in y])
    chosen = []
    for c in np.unique(classes):
        idx = np.where(classes == c)[0]
        k = int(round(n * len(idx) / len(classes)))
        chosen += list(rng.choice(idx, min(k, len(idx)), replace=False))
    chosen = sorted(chosen)
    (out / "subsample_ids.txt").write_text("\n".join(canon[chosen]) + "\n")
    print(f"wrote {len(chosen):,} ids to {out / 'subsample_ids.txt'}")


def feature_sets(arm, orb_dir, mpa_dir, mace_dir, magpie_npz):
    """Yield (config_str, X, ids) for every read-out of an arm."""
    if arm in ("orb_omat", "orb_mpa"):
        fdir = (orb_dir / "features") if arm == "orb_omat" else (mpa_dir / "features")
        for layer in O.ALL_LAYERS:
            p = fdir / f"{layer}_pooled.npz"
            if not p.exists():
                continue
            d = np.load(p, allow_pickle=True)
            for rep in O.REPS:
                yield f"{rep}/{layer}", O.build_X(d["node"], d["edge"], rep), np.asarray(d["mp_ids"]).astype(str)
    elif arm == "uma":
        for layer in U.NODE_LAYERS:
            for fam in UMA_NODE_FAMILIES:
                try:
                    X, ids = U.load_feature_set("node", fam, layer)
                except SystemExit:
                    continue
                yield f"node/{fam}/{layer}", X, np.asarray(ids).astype(str)
        for layer in U.EDGE_LAYERS:
            try:
                X, ids = U.load_feature_set("concat", "ang_norm", layer)
            except SystemExit:
                continue
            yield f"concat/ang_norm/{layer}", X, np.asarray(ids).astype(str)
    elif arm == "mace_omat":
        for p in sorted((mace_dir / "pooled").glob("layer_*.npz")):
            z = np.load(p, allow_pickle=True)
            yield p.stem.replace("layer_", "node/"), z["X"], np.asarray(z["mp_ids"]).astype(str)
    elif arm == "magpie":
        z = np.load(magpie_npz, allow_pickle=True)
        yield "magpie", z["X"].astype(np.float32), np.asarray(z["mp_ids"]).astype(str)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-subsample", action="store_true")
    ap.add_argument("--subsample-n", type=int, default=20000)
    ap.add_argument("--orb-dir", type=Path, default=(P.RESULTS / "orb/property_prediction"))
    ap.add_argument("--mpa-dir", type=Path, default=(P.RESULTS / "orb_mpa"))
    ap.add_argument("--mace-dir", type=Path, default=(P.RESULTS / "mace_omat"))
    ap.add_argument("--magpie-npz", type=Path, default=(P.RESULTS / "exp_magpie/magpie_features.npz"))
    ap.add_argument("--out-dir", type=Path, default=(P.RESULTS / "matched_data"))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=6)
    a = ap.parse_args()
    out = a.out_dir; out.mkdir(parents=True, exist_ok=True)
    if a.make_subsample:
        make_subsample(a.orb_dir, out, a.subsample_n, a.seed); return
    sub = [l.strip() for l in open(out / "subsample_ids.txt") if l.strip()]
    sub_set = set(sub)
    if (a.mpa_dir / "encoder").exists() and not (a.mpa_dir / "features" / "backbone_pooled.npz").exists():
        print("[orb_mpa] pooling the .pt files into feature caches")
        O.build_feature_cache(a.mpa_dir, a.mpa_dir / "features", sub, a.n_jobs)

    canon, ydict = O.load_canonical_targets(a.orb_dir / "features", TARGETS)
    canon = np.asarray(canon).astype(str); cpos = {m: i for i, m in enumerate(canon)}
    splits = {t: np.load(a.orb_dir / "splits" / f"{O.safe(t)}.npz", allow_pickle=True) for t in TARGETS}

    path = out / "metrics.csv"
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    done = {(r["arm"], r["config"], r["target"]) for r in rows}
    preds_store = {}      # (arm, config, target) -> (test_ids, y, pred) for the best configs' paired tests
    cols = ["arm", "architecture", "training_data", "equivariant", "config", "target", "task", "metric", "score", "n_train", "n_test"]

    def save():
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    def fit_one(arm, cfg, X, ids, t):
        """XGBoost at one read-out for one target on the subsample's split; None when too few rows.
        Returns (row, (test_ids, y_test, prediction))."""
        arch, data, eq = ARMS[arm]
        pos = {m: i for i, m in enumerate(ids)}
        task = "clf" if O.is_clf(t) else "reg"
        y = ydict[t]
        def take(idl):
            ms = [str(m) for m in idl if str(m) in sub_set and str(m) in pos and str(m) in cpos]
            xi = np.array([pos[m] for m in ms]); yi = np.array([y[cpos[m]] for m in ms], dtype=object)
            ok = (np.array([v is not None and str(v) != "nan" for v in yi]) if task == "clf" else np.isfinite(yi.astype(float)))
            ok &= np.isfinite(X[xi]).all(axis=1)
            return X[xi][ok], yi[ok], np.array(ms)[ok]
        Xtr, ytr, _ = take(splits[t]["train_mp_ids"]); Xte, yte, te_ids = take(splits[t]["test_mp_ids"])
        if len(ytr) < 50 or len(yte) < 20:
            return None, None
        if task == "clf":
            le = LabelEncoder().fit(np.concatenate([ytr, yte]).astype(str))
            ytr, yte = le.transform(ytr.astype(str)), le.transform(yte.astype(str))
            if len(np.unique(ytr)) < 2:
                return None, None
            classes, ytr_fit = np.unique(ytr, return_inverse=True)
        else:
            ytr, yte = ytr.astype(float), yte.astype(float); classes, ytr_fit = None, ytr
        t0 = time.time()
        est = O.make_model(task, "xgb", a.seed, a.n_jobs).fit(Xtr, ytr_fit)
        pred = est.predict(Xte)
        if task == "clf":
            pred = classes[pred]
        sc = T14.score(task, yte, pred)["score"]
        row = dict(arm=arm, architecture=arch, training_data=data, equivariant=eq, config=cfg, target=t, task=task,
                   metric="macro_f1" if task == "clf" else "r2", score=round(sc, 4), n_train=len(ytr), n_test=len(yte))
        print(f"  [{arm}] {cfg:28s} {t:22s} {sc:.4f}  (n={len(ytr)}/{len(yte)}, {time.time()-t0:.0f}s)", flush=True)
        return row, (te_ids, yte, pred)

    for arm in a.arms.split(","):
        n_cfg = 0
        for cfg, X, ids in feature_sets(arm, a.orb_dir, a.mpa_dir, a.mace_dir, a.magpie_npz):
            n_cfg += 1
            for t in TARGETS:
                if (arm, cfg, t) in done:
                    continue
                row, pr = fit_one(arm, cfg, X, ids, t)
                if row is None:
                    continue
                rows.append(row); preds_store[(arm, cfg, t)] = pr
            save()
        if n_cfg == 0:
            print(f"  [{arm}] no features found, skipped")

    # best per arm and target, under the paper's rule that crystal system is read from rotation-invariant
    # channels only (05_property_prediction_fairchem.INVARIANT_ONLY_TARGETS): UMA's full and polar families
    # and MACE's full tensor are excluded for that target
    def allowed(arm, cfg, t):
        if t not in U.INVARIANT_ONLY_TARGETS:
            return True
        if arm == "uma":
            return U._family_allowed(t, cfg.split("/")[1])
        if arm == "mace_omat":
            return not cfg.endswith("_full")
        return True
    best = {}
    for r in rows:
        k = (r["arm"], r["target"])
        if allowed(*k[:1], r["config"], k[1]) and (k not in best or float(r["score"]) > float(best[k]["score"])):
            best[k] = r
    # the paired tests need predictions of the best configurations; refit those scored in an earlier run
    need = {(r["arm"], r["config"], r["target"]) for r in best.values()} - set(preds_store)
    for arm in sorted({k[0] for k in need}):
        for cfg, X, ids in feature_sets(arm, a.orb_dir, a.mpa_dir, a.mace_dir, a.magpie_npz):
            for t in [k[2] for k in need if k[0] == arm and k[1] == cfg]:
                _, pr = fit_one(arm, cfg, X, ids, t)
                if pr is not None:
                    preds_store[(arm, cfg, t)] = pr
    with open(out / "best_by_arm.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows([best[k] for k in sorted(best)])
    # paired comparisons on the shared test rows (recomputed predictions for the best configs)
    rng = np.random.default_rng(a.seed)
    prow = []
    for arm_a, arm_b, label in PAIRS:
        for t in TARGETS:
            ra, rb = best.get((arm_a, t)), best.get((arm_b, t))
            if not ra or not rb:
                continue
            pa, pb = preds_store.get((arm_a, ra["config"], t)), preds_store.get((arm_b, rb["config"], t))
            if pa is None or pb is None:
                prow.append(dict(pair=label, target=t, arm_a=arm_a, cfg_a=ra["config"], score_a=ra["score"], arm_b=arm_b, cfg_b=rb["config"],
                                 score_b=rb["score"], diff=round(float(ra["score"]) - float(rb["score"]), 4), diff_lo="", diff_hi="", p="",
                                 note="best configs scored in an earlier run; rerun with the rows removed for a paired interval"))
                continue
            common = sorted(set(pa[0]) & set(pb[0]))
            ia = {m: i for i, m in enumerate(pa[0])}; ib = {m: i for i, m in enumerate(pb[0])}
            ya = pa[1][[ia[m] for m in common]]; yb = pb[1][[ib[m] for m in common]]
            if not np.array_equal(ya, yb):
                continue
            task = "clf" if O.is_clf(t) else "reg"
            g, lo, hi, p = T14.paired_gain(task, ya, pa[2][[ia[m] for m in common]], pb[2][[ib[m] for m in common]], a.boot, rng)
            prow.append(dict(pair=label, target=t, arm_a=arm_a, cfg_a=ra["config"], score_a=ra["score"], arm_b=arm_b, cfg_b=rb["config"],
                             score_b=rb["score"], diff=round(g, 4), diff_lo=round(lo, 4), diff_hi=round(hi, 4), p=round(p, 6), note=""))
    with open(out / "pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair", "target", "arm_a", "cfg_a", "score_a", "arm_b", "cfg_b", "score_b", "diff", "diff_lo", "diff_hi", "p", "note"])
        w.writeheader(); w.writerows(prow)
    (out / "README.md").write_text(
        f"# Matched-data / matched-architecture comparison\n\n{len(sub):,}-material stratified subsample of the paper corpus; the paper's seed-42 "
        "split restricted to it; XGBoost with the paper's hyperparameters at every available read-out; `best_by_arm.csv` keeps each arm's best "
        "configuration per target (crystal system from rotation-invariant channels only, as in the paper) and `pairs.csv` gives the "
        "paired bootstrap difference on the shared test rows for the headline pairs.\n\n"
        "Arms: " + "; ".join(f"`{k}` = {v[0]} trained on {v[1]}" for k, v in ARMS.items()) + ".\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
