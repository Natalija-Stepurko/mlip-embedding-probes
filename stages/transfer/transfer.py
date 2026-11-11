#!/usr/bin/env python3
"""
stages/transfer/transfer.py — do the Materials-Project-trained probes transfer to another
DFT database? Score each target's TOP XGBoost configuration (selected on MP, refit on
MP's own seed-42 training split) on the structures and labels of an external database
written by stages/data/fetch_external.py and extracted by stages/extract/extract_{orb,uma}.py.

Three slices, never pooled, because they answer different questions:
  disjoint          neither cited from MP nor a StructureMatcher twin of an MP entry --
                    genuinely unseen crystals: this slice carries the TRANSFER claim
  in_mp_corpus      the same crystals we trained on, re-relaxed and re-labelled in the
                    external functional -- a LABEL-PROVENANCE test (input to label_shift.py)
  structure_match   no citation but a twin exists -- reported, treated like in_mp_corpus
  all               the union, for completeness only

Alongside every transfer score, the within-external reference: the same configuration
refit on --within-repeats 80/20 splits of the same slice; the MP-trained probe is scored
on the test rows of each split, so the gap compares the two probes on one population.
Labels are always the external database's own; metal/insulator uses MP's rule
(band_gap == 0 exactly, mlipprobes.common.metal_insulator).

All nine targets are reported. Ones the external source does not carry appear with
status "unavailable" rather than being dropped.

UMA features. probes_uma.py reads UMA's pooled read-outs from organisation_uma.py's UMAP caches, which
do not exist for an external set. This stage pools them itself from the 02 .pt files,
reusing organisation_uma.py's per-atom channel conversions and pooling functions (imported, not
re-implemented), and caches them under <ext-uma>/pooled/.

Reads  $MLIP_PROBES_ROOT/results/{orb,fairchem}/property_prediction/{metrics.csv,features,splits}
       <ext-structures>/index.jsonl      <ext-orb>/<layer>/{nodes,edges}/*.pt
       <ext-uma>/<layer>/{nodes,edges}/*.pt
Writes $MLIP_PROBES_ROOT/results/external_transfer/<ext-name>/{transfer,label_shift}.csv, README.md
       (rows of (model, target) pairs not re-run are kept, so --models uma leaves ORB rows intact)
Usage
    uv run python stages/transfer/transfer.py --ext-name jarvis_dft_3d --within-repeats 5
    uv run python stages/transfer/transfer.py --check-pooling
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
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
O = importlib.import_module("stages.probes.probes_orb")
U = importlib.import_module("stages.probes.probes_uma")
A3 = importlib.import_module("stages.geometry.organisation_uma")   # UMA channel conversions
qc = importlib.import_module("mlipprobes.common")

from scipy.stats import spearmanr
from sklearn.metrics import f1_score, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

NINE = ["formation_energy/atom", "energy_above_hull", "band_gap", "density", "bulk_modulus",
        "mag_density", "crystal_system", "metal_insulator", "magnetic_label"]


# ----------------------------------------------------------------------------- external labels
def load_external(index_path: Path):
    structs, _ = qc.load_manifest(index_path)
    ids = np.array(list(structs.keys()), dtype=object)
    rel = np.array(["in_mp_corpus" if structs[m].get("in_mp_corpus")
                    else ("structure_match" if structs[m].get("structure_match_mp") else "disjoint")
                    for m in ids], dtype=object)
    return structs, ids, rel


def external_y(structs, ids, target):
    """External label array for `target`, using the paper's own derivation rules."""
    if target in qc.CAT_PROPS:
        return qc.cat_array(structs, ids, target)
    return qc.cont_array(structs, ids, target)


# ----------------------------------------------------------------------------- UMA pooling
# family -> per-atom conversion from organisation_uma.py (each maps [N, 9, 128] -> [N, dim]); pooled = mean over atoms
_NODE_MEAN_FAMILIES = {"full": "flat", "ang_norm": "ang_norm", "l0": "l0"}
_NODE_POLAR = {"l1_polar": A3._pool_l1_polar, "l2_polar": A3._pool_l2_polar,
               "geom_norm": A3._pool_geom_norm}
_EDGE_FAMILIES = ["flat", "ang_norm", "l0", "l1_norm", "l2_norm", "geom_norm"]


def _pool_uma_one(pt: Path, group: str, family: str) -> np.ndarray | None:
    """One material's UMA read-out, computed exactly as 03 does for MP: node = per-atom
    conversion then mean over atoms; edge = the file's `edge_pooled` [9, 128] (mean over ALL
    bonds, not the <=48 stored `edge_msg` rows) converted, then L2-normalised."""
    import torch
    d = torch.load(pt, weights_only=False)
    if group == "node":
        if "node" not in d:
            return None
        raw = d["node"].float().numpy()                      # [N, 9, 128]
        if raw.ndim != 3 or raw.shape[0] == 0:
            return None
        if family in _NODE_POLAR:
            return _NODE_POLAR[family](raw)
        return A3._VARIANT_FN[_NODE_MEAN_FAMILIES.get(family, family)](raw).mean(axis=0)
    if "edge_pooled" not in d:
        return None
    v = A3._VARIANT_FN[family](d["edge_pooled"].float().numpy()[None])[0]
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def uma_pooled(ext_uma: Path, group: str, family: str, layer: str, ids, workers: int):
    """(X, ids) for one UMA read-out on the external set, cached under <ext-uma>/pooled/."""
    sub = "nodes" if group == "node" else "edges"
    v = "_v2" if group == "edge" else ""      # v2: edge_pooled read-out (v1 averaged the stored edge_msg rows)
    cache = ext_uma / "pooled" / f"{layer}_{sub}_{family}{v}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["mp_ids"]
    src = ext_uma / layer / sub
    rows, keep = [], []
    t0 = time.time()
    for i, m in enumerate(ids, 1):
        pt = src / f"{m}.pt"
        if not pt.exists():
            continue
        v = _pool_uma_one(pt, group, family)
        if v is not None:
            rows.append(v.astype(np.float32)); keep.append(m)
        if i % 5000 == 0:
            print(f"    pooling uma {layer}/{sub}/{family}: {i:,}/{len(ids):,} ({time.time()-t0:.0f}s)", flush=True)
    X = np.stack(rows) if rows else np.zeros((0, 0), np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=X, mp_ids=np.array(keep, dtype=object))
    return X, np.array(keep, dtype=object)


def uma_external_X(ext_uma, group, family, layer, ids, workers):
    if group in ("node", "edge"):
        return uma_pooled(ext_uma, group, family, layer, ids, workers)
    nfam, efam = U.CONCAT_MAP[family]
    Xn, idn = uma_pooled(ext_uma, "node", nfam, layer, ids, workers)
    Xe, ide = uma_pooled(ext_uma, "edge", efam, layer, ids, workers)
    pose = {m: i for i, m in enumerate(ide)}
    common = [m for m in idn if m in pose]
    posn = {m: i for i, m in enumerate(idn)}
    rn = np.fromiter((posn[m] for m in common), dtype=np.int64, count=len(common))
    re_ = np.fromiter((pose[m] for m in common), dtype=np.int64, count=len(common))
    return np.hstack([Xn[rn], Xe[re_]]).astype(np.float32), np.array(common, dtype=object)


# ----------------------------------------------------------------------------- MP side
def mp_train(model, mod, mdir, target, cfg, seed, n_jobs):
    """Fit the top XGBoost config on MP's seed-42 TRAIN split. Returns (model, encoder, task, n_train, Xtr, ytr)."""
    task = "clf" if mod.is_clf(target) else "reg"
    feat_dir = mdir / "features"
    if model == "orb":
        rep, layer = cfg
        canon, ydict = O.load_canonical_targets(feat_dir, [target])
        d = np.load(feat_dir / f"{layer}_pooled.npz", allow_pickle=True)
        X = O.build_X(d["node"], d["edge"], rep); ids = canon
    else:
        group, fam, layer = cfg
        canon, ydict = U.load_targets(feat_dir, [target])
        X, ids = U.load_feature_set(group, fam, layer)
        pos = {m: i for i, m in enumerate(canon)}
        ydict = {target: np.array([ydict[target][pos[m]] for m in ids], dtype=object if task == "clf" else float)}
    y = ydict[target]
    sp = np.load(mdir / "splits" / f"{mod.safe(target)}.npz", allow_pickle=True)
    pos = {m: i for i, m in enumerate(ids)}
    tr = np.array([pos[m] for m in sp["train_mp_ids"] if m in pos])
    Xtr, ytr = X[tr], y[tr]
    fin = np.isfinite(Xtr).all(axis=1) & (np.array([v is not None for v in ytr]) if task == "clf" else np.isfinite(ytr.astype(float)))
    Xtr, ytr = Xtr[fin], ytr[fin]
    le = None
    if task == "clf":
        le = LabelEncoder().fit(ytr); ytr = le.transform(ytr)
    else:
        ytr = ytr.astype(float)
    est = mod.make_model(task, "xgb", seed, n_jobs).fit(Xtr, ytr)
    return est, le, task, len(Xtr), Xtr, ytr


# ----------------------------------------------------------------------------- pooling check
def check_pooling(n_mats: int = 10, tol: float = 2e-3, seed: int = 0) -> None:
    """Recompute UMA read-outs from MP's .pt files with THIS module's pooling and compare them
    with the stage-03 caches the probes were trained on. Aborts on a mismatch: an earlier
    version averaged the stored edge_msg rows, giving edge features that were wrong by whole
    multiples while every downstream number still looked plausible."""
    rng = np.random.default_rng(seed)
    root = (P.RESULTS / "fairchem")
    worst, bad = 0.0, []
    for layer in ("embed", "block_3"):
        for group in ("node", "edge"):
            for fam in U.GROUP_FAMILIES[group]:
                X, ids = U._load_one(group, fam, layer)
                sub = "nodes" if group == "node" else "edges"
                for i in rng.choice(len(ids), n_mats, replace=False):
                    v = _pool_uma_one(root / layer / sub / f"{ids[i]}.pt", group, fam)
                    err = float(np.max(np.abs(v - X[i])) / (np.max(np.abs(X[i])) + 1e-12))
                    worst = max(worst, err)
                    if err > tol:
                        bad.append((layer, group, fam, round(err, 4)))
    print(f"[pooling check] UMA pooling vs stage-03 cache: worst relative error {worst:.2e} (tol {tol:.0e})")
    if bad:
        raise SystemExit(f"POOLING CHECK FAILED: {bad[:6]}")


# ----------------------------------------------------------------------------- statistics
SLICES = ["disjoint", "in_mp_corpus", "structure_match", "all"]
SEL_SLICES = ["disjoint", "all"]


def metric_name(task):
    return "macro_f1" if task == "clf" else "r2"


def score(task, yt, yp):
    if task == "clf":
        return dict(score=float(f1_score(yt, yp, average="macro")), rmse=None, sigma=None)
    return dict(score=float(r2_score(yt, yp)), rmse=float(root_mean_squared_error(yt, yp)), sigma=float(np.std(yt)))


def _macro_f1_cm(cm):
    tp = np.diag(cm).astype(float)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    ok = den > 0                                    # sklearn's label set: present in y_true or y_pred
    return float(np.mean(2 * tp[ok] / den[ok]))


def _fast_metric(task, y, p, k):
    if task == "clf":
        return _macro_f1_cm(np.bincount(y * k + p, minlength=k * k).reshape(k, k))
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))



def paired_gain(task, y, pa, pb, B, rng, alpha=0.05):
    """Paired bootstrap of metric(a) - metric(b) on shared resampled rows: (gain, lo, hi, p).
    Used by stages/controls/matched_models.py.
    The statistic of significance_lib's paired_boot_diff, computed from confusion counts / sums of squares."""
    y, pa, pb = np.asarray(y), np.asarray(pa), np.asarray(pb)
    k = int(max(y.max(), pa.max(), pb.max())) + 1 if task == "clf" else 0
    point = _fast_metric(task, y, pa, k) - _fast_metric(task, y, pb, k)
    n = len(y); d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        d[b] = _fast_metric(task, y[i], pa[i], k) - _fast_metric(task, y[i], pb[i], k)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi), float(min(1.0, 2.0 * min((d <= 0).mean(), (d >= 0).mean())))

def _label_ok(task, y):
    """Rows with a usable external label (None for categories, NaN for continuous values)."""
    if task == "clf":
        return np.array([v is not None for v in y])
    return np.isfinite(np.asarray(y, dtype=float))


# ----------------------------------------------------------------------------- composition keys
_COMP_CACHE: dict = {}


def comp_keys(formula):
    """(reduced formula, chemical system) of a formula string; (None, None) if unparsable."""
    if formula not in _COMP_CACHE:
        try:
            from pymatgen.core import Composition
            c = Composition(formula)
            _COMP_CACHE[formula] = (c.reduced_formula, "-".join(sorted(str(e) for e in c.elements)))
        except Exception:
            _COMP_CACHE[formula] = (None, None)
    return _COMP_CACHE[formula]


class Ext:
    """What is known about the external set independent of model and target."""
    def __init__(self, name, structs, ids, rel, res_dir):
        self.name, self.structs, self.ids, self.rel, self.res_dir = name, structs, ids, rel, res_dir
        self.pos = {m: i for i, m in enumerate(ids)}
        keys = [comp_keys(structs[m].get("formula")) for m in ids]
        self.red = np.array([k[0] for k in keys], dtype=object)
        self.chem = np.array([k[1] for k in keys], dtype=object)
        self.keep = np.ones(len(ids), bool)          # --subsample restricts every arm to these rows





# ----------------------------------------------------------------------------- arms
class Arm:
    """One feature set's transfer of one target: aligned external rows and labels, the MP-fitted
    probe and its predictions. `_within` memoises within-external reference fits."""
    def __init__(self, name, cfg, task, ids, X, y, le, est, Xtr, ytr, ext, dropped=0):
        self.name, self.cfg, self.task, self.ids, self.X, self.y = name, cfg, task, ids, X, y
        self.le, self.est, self.Xtr, self.ytr, self.dropped, self.ext = le, est, Xtr, ytr, dropped, ext
        self.pred = est.predict(X)
        rows = np.array([ext.pos[m] for m in ids])
        self.rel, self.red, self.chem = ext.rel[rows], ext.red[rows], ext.chem[rows]
        self.n_mp_train = len(ytr)
        self._within = {}


class View:
    """An arm restricted (and re-ordered) to a common id list, with predictions."""
    def __init__(self, arm, common, share_cache):
        pos = {m: i for i, m in enumerate(arm.ids)}
        ix = np.array([pos[m] for m in common])
        self.task, self.cfg = arm.task, arm.cfg
        self.ids, self.X, self.y, self.pred = arm.ids[ix], arm.X[ix], arm.y[ix], arm.pred[ix]
        self.rel, self.red, self.chem = arm.rel[ix], arm.red[ix], arm.chem[ix]
        self._within = arm._within if share_cache else {}


def within_ext(mod, task, X, y, groups, seed, n_jobs):
    """Within-external reference: the same configuration refit on an 80/20 split of the rows;
    (score, rmse, sigma, n_test)."""
    te, p, yte = within_ext_split(mod, task, X, y, groups, seed, n_jobs)
    sc = score(task, yte, p)
    return sc["score"], sc["rmse"], sc["sigma"], len(yte)


def within_ext_split(mod, task, X, y, groups, seed, n_jobs):
    """The within-external refit itself: (test row indices, predictions on them, their labels), so the
    MP-trained probe can be scored on exactly the same test rows (transfer_rows)."""
    n = len(y)
    strat = y if task == "clf" and np.min(np.bincount(y)) >= 2 else None
    tr, te = train_test_split(np.arange(n), test_size=0.2, random_state=seed, stratify=strat)
    ytr, yte = y[tr], y[te]
    if task == "clf":
        classes = np.unique(ytr)
        known = np.isin(yte, classes)
        te, yte = te[known], yte[known]
        ytr_fit = np.searchsorted(classes, ytr)
    else:
        ytr_fit = ytr
    p = mod.make_model(task, "xgb", seed, n_jobs).fit(X[tr], ytr_fit).predict(X[te])
    if task == "clf":
        p = classes[p]
    return te, p, yte


def _too_few(a, n, args, y):
    return n < args.min_rows or (a.task == "clf" and len(np.unique(y)) < 2)


def build_embed_arm(model, mod, mdir, ext, target, cfg, y_all, ext_orb, ext_uma, args):
    """External features at `cfg` (aligned to labelled rows) and the MP-trained probe."""
    if model == "orb":
        rep, layer = cfg
        d = np.load(ext_orb / "features" / f"{layer}_pooled.npz", allow_pickle=True)
        X_ext, ids_ext = O.build_X(d["node"], d["edge"], rep), d["mp_ids"]
    else:
        group, fam, layer = cfg
        X_ext, ids_ext = uma_external_X(ext_uma, group, fam, layer, ext.ids, args.n_jobs)
    task = "clf" if mod.is_clf(target) else "reg"
    y_ext = y_all[np.array([ext.pos[m] for m in ids_ext])]
    ok = np.isfinite(X_ext).all(axis=1) & _label_ok(task, y_ext) & ext.keep[np.array([ext.pos[m] for m in ids_ext])]
    X_ext, y_ext, ids_kept = X_ext[ok], y_ext[ok], np.asarray(ids_ext)[ok]
    est, le, task, _, Xtr, ytr = mp_train(model, mod, mdir, target, cfg, args.seed, args.n_jobs)
    dropped = 0
    if task == "clf":
        known = np.isin(y_ext, le.classes_)
        dropped = int((~known).sum())
        X_ext, y_ext, ids_kept = X_ext[known], y_ext[known], ids_kept[known]
        y_enc = le.transform(y_ext)
    else:
        y_enc = y_ext.astype(float)
    return Arm(model, cfg, task, ids_kept, X_ext, y_enc, le, est, Xtr, ytr, ext, dropped)


# ----------------------------------------------------------------------------- controls
def transfer_rows(mod, target, model, arm, mp_score, args):
    """Headline: per slice, MP-trained score and the within-external reference."""
    rows, cfg_str = [], "/".join(map(str, arm.cfg))
    for sl in SLICES:
        mask = np.ones(len(arm.y), bool) if sl == "all" else (arm.rel == sl)
        n = int(mask.sum())
        if _too_few(arm, n, args, arm.y[mask]):
            rows.append(dict(target=target, model=model, config=cfg_str, slice=sl, status="too_few", n=n,
                             selection="test_selected")); continue
        tr = score(arm.task, arm.y[mask], arm.pred[mask])
        Xm, ym, gm, pm = arm.X[mask], arm.y[mask], arm.red[mask], arm.pred[mask]
        # the within-external reference over R 80/20 splits, and the MP-trained probe scored on the SAME test rows
        # of each split, so the gap compares two probes on one population (paired over splits)
        w_scs, w_rms, same_scs, gaps, w_n = [], [], [], [], 0
        for r in range(max(1, args.within_repeats)):
            te, p, yte = within_ext_split(mod, arm.task, Xm, ym, gm, args.seed + r, args.n_jobs)
            w = score(arm.task, yte, p); same = score(arm.task, yte, pm[te])
            if r == 0:
                arm._within[(sl, n, False)] = (w["score"], w["rmse"], w["sigma"], len(yte)); w_n = len(yte)
            w_scs.append(w["score"]); w_rms.append(w["rmse"] if w["rmse"] is not None else np.nan)
            same_scs.append(same["score"]); gaps.append(same["score"] - w["score"])
        reps = len(w_scs); w_sc = float(np.mean(w_scs)); w_sd = round(float(np.std(w_scs, ddof=1)), 4) if reps > 1 else None
        w_rmse = float(np.nanmean(w_rms)) if np.isfinite(w_rms).any() else None
        rows.append(dict(target=target, model=model, config=cfg_str, slice=sl, status="ok", n=n, selection="test_selected",
                         mp_test_score=None if mp_score is None else round(mp_score, 4), n_mp_train=arm.n_mp_train,
                         transfer_score=round(tr["score"], 4), transfer_rmse=tr["rmse"],
                         within_ext_score=round(w_sc, 4), within_ext_rmse=w_rmse, within_ext_sd=w_sd, within_ext_repeats=reps,
                         within_ext_n_test=w_n, ext_sigma=tr["sigma"],
                         transfer_same_rows_score=round(float(np.mean(same_scs)), 4),
                         transfer_same_rows_sd=round(float(np.std(same_scs, ddof=1)), 4) if reps > 1 else None,
                         gap_same_rows=round(float(np.mean(gaps)), 4),
                         gap_same_rows_sd=round(float(np.std(gaps, ddof=1)), 4) if reps > 1 else None,
                         classes_dropped=arm.dropped))
        print(f"   {sl:16s} n={n:6,}  transfer={tr['score']:.3f}  within-external={w_sc:.3f}"
              + (f" ± {w_sd:.3f} ({reps} splits)" if w_sd is not None else "")
              + f"  same-rows transfer={np.mean(same_scs):.3f} gap={np.mean(gaps):+.3f}"
              + ("" if mp_score is None else f"  (MP {mp_score:.3f})"), flush=True)
    return rows


# ----------------------------------------------------------------------------- IO
STAGE_FILES = {"transfer": "transfer.csv"}


def write_merged(path: Path, new_rows, done):
    """Write `new_rows`, keeping earlier rows of (model, target) pairs NOT re-run now, so a
    `--models uma` re-run leaves the ORB rows and a `--targets` subset leaves the other targets."""
    old = []
    if path.exists():
        with open(path, newline="") as f:
            old = [r for r in csv.DictReader(f) if (r.get("model"), r.get("target")) not in done]
    rows = old + new_rows
    if not rows:
        return
    lead = ["target", "model", "config", "slice", "status", "n"]
    keys = sorted({k for r in rows for k in r}, key=lambda k: (0, lead.index(k), "") if k in lead else (1, 0, k))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def label_shift(structs, emb, target, args, shift_rows):
    """MP vs external labels on the shared crystals (regression targets)."""
    if not hasattr(label_shift, "_mp"):
        label_shift._mp = qc.load_manifest((P.STRUCTURES / "index.jsonl"))[0]
    mp_recs = label_shift._mp
    shared = emb.rel == "in_mp_corpus"
    pairs = [(qc.cont_array(mp_recs, np.array([structs[m]["mpid_reported"]]), target)[0], yv)
             for m, yv in zip(emb.ids[shared], emb.y[shared]) if structs[m].get("mpid_reported") in mp_recs]
    pairs = [(a, b) for a, b in pairs if a is not None and np.isfinite(a)]
    if len(pairs) >= args.min_rows:
        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
        rho, _ = spearmanr(a, b)
        shift_rows.append(dict(target=target, n_shared=len(pairs), mean_ext_minus_mp=float(np.mean(b - a)),
                               rmse_ext_vs_mp=float(np.sqrt(np.mean((b - a) ** 2))), spearman_rho=float(rho)))
        print(f"   label shift (shared crystals, n={len(pairs):,}): ext-mp mean {np.mean(b-a):+.3f}, "
              f"rmse {np.sqrt(np.mean((b-a)**2)):.3f}, rho {rho:.3f}")


def readme(args):
    return (f"# External transfer — {args.ext_name}\n\n"
            "MP-trained top XGBoost config per target, refit on MP's seed-42 train split, scored on the external\n"
            "database's own labels. `transfer.csv`: `transfer_score` vs `within_ext_score` (same config, 80/20 split\n"
            "of the same slice) is the result; `transfer_same_rows_score` / `gap_same_rows` score the MP-trained probe on the\n"
            "test rows of each within-external split, so both probes are compared on one population (mean over splits).\n"
            "Slices: disjoint = unseen crystals (the transfer claim);\n"
            "in_mp_corpus / structure_match = same crystals, external labels (label provenance); all = union.\n"
            "`label_shift.csv`: MP vs external labels on shared crystals (input to stages/transfer/label_shift.py).\n")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ext-name", default="jarvis_dft_3d")
    ap.add_argument("--ext-structures", type=Path, default=None)
    ap.add_argument("--ext-orb", type=Path, default=None)
    ap.add_argument("--ext-uma", type=Path, default=None)
    ap.add_argument("--orb-dir", type=Path, default=(P.RESULTS / "orb/property_prediction"))
    ap.add_argument("--uma-dir", type=Path, default=(P.RESULTS / "fairchem/property_prediction"))
    ap.add_argument("--models", default="orb,uma")
    ap.add_argument("--targets", default=",".join(NINE))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--min-rows", type=int, default=200, help="skip a slice with fewer labelled rows")
    ap.add_argument("--within-repeats", type=int, default=1,
                    help="80/20 splits averaged for the within-external reference of the transfer stage (seeds seed..seed+R-1)")
    ap.add_argument("--subsample", type=int, default=0, help="score a seeded subset of this many external rows (smoke tests)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default $MLIP_PROBES_ROOT/results/external_transfer/<ext-name>")
    ap.add_argument("--check-pooling", action="store_true", help="verify UMA pooling against MP's caches, then exit")
    ap.add_argument("--dry-run", action="store_true", help="report availability and configs only; writes nothing")
    args = ap.parse_args()

    if args.check_pooling:
        check_pooling(); return 0
    stages = ["transfer"]
    models = args.models.split(",")
    if "uma" in models and not args.dry_run:
        check_pooling()

    ext_structures = args.ext_structures or P.STRUCTURES_EXTERNAL / args.ext_name
    ext_orb = args.ext_orb or (P.RESULTS / "external") / args.ext_name / "orb"
    ext_uma = args.ext_uma or (P.RESULTS / "external") / args.ext_name / "fairchem"
    out = args.out_dir or (P.RESULTS / "external_transfer") / args.ext_name; out.mkdir(parents=True, exist_ok=True)

    structs, ids, rel = load_external(ext_structures / "index.jsonl")
    ext = Ext(args.ext_name, structs, ids, rel, (P.RESULTS / "external") / args.ext_name)
    if args.subsample:
        ext.keep[:] = False
        ext.keep[np.random.default_rng(args.seed).choice(len(ids), min(args.subsample, len(ids)), replace=False)] = True
    print(f"{args.ext_name}: {len(ids):,} structures | " +
          " ".join(f"{s}={int((rel == s).sum()):,}" for s in SLICES[:-1]) + f" | stages: {','.join(stages)}")

    if "orb" in models and not args.dry_run:
        feat_dir = ext_orb / "features"
        O.build_feature_cache(ext_orb, feat_dir, list(ids), args.n_jobs)
        O.build_target_table(structs, feat_dir, list(ids))

    metrics = {"orb": O.load_metrics(args.orb_dir / "metrics.csv"), "uma": U.load_metrics(args.uma_dir / "metrics.csv")}
    mods = {"orb": (O, args.orb_dir), "uma": (U, args.uma_dir)}
    targets = args.targets.split(",")
    rng = np.random.default_rng(args.seed + 21)
    res = {s: [] for s in STAGE_FILES}; shift_rows = []; done = set()

    for target in targets:
        y_all = external_y(structs, ids, target)
        n_avail = int(_label_ok("clf" if O.is_clf(target) else "reg", y_all).sum())
        if n_avail < args.min_rows:
            for model in models:
                for s in stages:
                    res[s].append(dict(target=target, model=model, config=None, slice="all", status="unavailable",
                                       n=n_avail, note=f"{n_avail} labelled rows in {args.ext_name}"))
                done.add((model, target))
            print(f"\n{target}: UNAVAILABLE in {args.ext_name} ({n_avail} labelled rows)")
            continue

        for model in models:
            mod, mdir = mods[model]
            cfg, mp_score = mod.best_config(metrics[model], target)
            if cfg is None:
                res["transfer"].append(dict(target=target, model=model, config=None, slice="all", status="no_mp_config", n=0))
                done.add((model, target)); continue
            cfg_str = "/".join(map(str, cfg))
            print(f"\n{target} [{model}] top config {cfg_str}  (MP test score {mp_score:.3f}; external labelled {n_avail:,})")
            if args.dry_run:
                continue
            done.add((model, target))
            t_stage = time.time()
            emb = build_embed_arm(model, mod, mdir, ext, target, cfg, y_all, ext_orb, ext_uma, args)
            print(f"   (embedding arm built: {time.time()-t_stage:.0f}s)", flush=True); t_stage = time.time()

            res["transfer"] += transfer_rows(mod, target, model, emb, mp_score, args)
            if emb.task == "reg" and model == "orb":
                label_shift(structs, emb, target, args, shift_rows)

    if args.dry_run:
        return 0
    for s in stages:
        write_merged(out / STAGE_FILES[s], res[s], done)
    if shift_rows:
        with open(out / "label_shift.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(shift_rows[0])); w.writeheader(); w.writerows(shift_rows)
    (out / "README.md").write_text(readme(args))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
