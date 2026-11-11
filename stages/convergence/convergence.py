"""
stages/convergence/convergence.py — cross-model representational convergence (ORB ↔ UMA).

It measures whether the
two independently-trained foundation models — ORB and UMA (FairChem) — organise atom
space the same way, layer by layer, on the *same atoms*. For every
(ORB layer i, UMA layer j) pair it computes three representational-similarity metrics
and a permutation baseline:

    - linear CKA          (feature-space)
    - SVCCA               (subspace canonical correlation)
    - mutual k-NN         (Platonic-hypothesis neighbour agreement; Huh et al. 2024)

The two models have different architectures and training objectives, so above-baseline
similarity is evidence of a shared, model-independent representation of local atomic
environments. Atoms are aligned by construction: for a given material both models emit
per-atom vectors in the *same atom order* (verified — identical element sequence and
fractional coordinates), so atoms are the exact analog of residues in the bio study.

ORB has 6 distinct node layers (encoder, gnn_stacks_0..4; its "backbone" dump duplicates
gnn_stacks_4 byte-for-byte and is excluded); UMA has 6 (embed, backbone, block_0..3).

UMA's per-atom embedding is 9 irrep channels × 128, read out as:
    - "ang_norm" : cat(L0, ||L1||, ||L2||) = 384 dims — the DEFAULT for all
                   cross-model comparisons. Rotation-invariant, and its content
                   (composition + geometry) matches ORB's node embedding.
    - "full"     : all 9×128 = 1152 dims (rotation-sensitive)
    - "geom_norm": cat(||L1||, ||L2||) = 256 dims (geometry only)
    - "l0"       : invariant scalar channel only (128 dims)
CKA/SVCCA/mutual-kNN need only matched *rows* (atoms), not matched feature dims, so
ORB's 256-d layers compare directly against either UMA variant.

Outputs (under --out-dir, default results/convergence_orb_uma), one set per UMA variant:
    grids.npz                     cka/svcca/mutual_knn/cka_permuted [n_orb, n_uma] per variant + labels
    convergence_uma-<variant>.png the three heatmaps for that variant
    summary.txt                   peak (orb_layer, uma_layer) per metric/variant, baseline

Usage:
    uv run python scripts/stages/convergence/convergence.py \\
        --orb-dir $MLIP_PROBES_ROOT/results/orb --uma-dir $MLIP_PROBES_ROOT/results/fairchem \\
        --out-dir $MLIP_PROBES_ROOT/results/convergence_orb_uma --max-atoms 30000
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlipprobes import common as qc  # linear_cka, svcca

# ORB node layers (input → deep) and short labels
# NOTE: ORB's "backbone" dump is byte-identical to gnn_stacks_4 (same tensor saved
# twice), so it is excluded here; including it produced a duplicated row in every
# grid. UMA's "backbone" is a genuine layer and is kept.
ORB_LAYERS = ["encoder", "gnn_stacks_0", "gnn_stacks_1",
              "gnn_stacks_2", "gnn_stacks_3", "gnn_stacks_4"]
ORB_LBL = ["enc", "g0", "g1", "g2", "g3", "g4"]
# UMA node layers (input → deep) and short labels
# depth order: embed -> block_0..3 -> backbone (backbone is the LAST layer, so it
# belongs at the end of the axis, not second)
UMA_LAYERS = ["embed", "block_0", "block_1", "block_2", "block_3", "backbone"]
UMA_LBL = ["emb", "b0", "b1", "b2", "b3", "bb"]

METRICS = [("cka", "linear CKA"), ("svcca", "SVCCA"), ("mutual_knn", "mutual k-NN")]


# --------------------------------------------------------------------------- helpers
def _center(X: np.ndarray) -> np.ndarray:
    """Column-center (linear_cka requires it)."""
    X = X.astype(np.float64)
    return X - X.mean(0, keepdims=True)


def mutual_knn(X: np.ndarray, Y: np.ndarray, k: int = 10,
               max_rows: int = 4000, seed: int = 42) -> float:
    """Mean fraction of shared neighbours between the two spaces over the same atoms
    (Huh et al. 2024). Local, to keep the shared mlipprobes.common module untouched."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n > max_rows > 0:
        idx = rng.choice(n, max_rows, replace=False)
        X, Y = X[idx], Y[idx]; n = max_rows
    k = min(k, n - 1)
    ix = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(return_distance=False)[:, 1:]
    iy = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ix, iy)]))


def _orb_node(mid: str, layer: str, root: Path) -> np.ndarray:
    return torch.load(root / layer / "nodes" / mid, weights_only=False)["node"].to(torch.float32).numpy()


def _elements(mid: str, layer: str, root: Path) -> list:
    """Per-atom element symbols, in the same row order as the node tensor."""
    return list(torch.load(root / layer / "nodes" / mid, weights_only=False)["elements"])


def _uma_node_raw(mid: str, layer: str, root: Path) -> np.ndarray:
    return torch.load(root / layer / "nodes" / mid, weights_only=False)["node"].to(torch.float32).numpy()


# --------------------------------------------------------------------------- collection
def collect_aligned(orb_root: Path, uma_root: Path, max_atoms: int, cap: int,
                    max_mats: int, seed: int):
    """Build atom-aligned per-layer matrices for both models over an atom budget.

    Returns (orb[layer]->[N,256], uma_raw[layer]->[N,9,128], N).
    Atoms are drawn material-by-material; at most `cap` atoms per material so no large
    unit cell dominates. Both models share atom order, so row i is the same atom.
    """
    rng = np.random.default_rng(seed)
    mids = [p.name for p in (orb_root / "gnn_stacks_4" / "nodes").glob("*.pt")]
    rng.shuffle(mids)
    orb_parts = {L: [] for L in ORB_LAYERS}
    uma_parts = {L: [] for L in UMA_LAYERS}
    el_parts = []
    total = used = mismatched = 0
    for mid in mids[:max_mats]:
        if not (uma_root / "embed" / "nodes" / mid).exists():
            continue
        try:
            A = _orb_node(mid, "gnn_stacks_4", orb_root).shape[0]
            if A == 0:
                continue
            idx = rng.choice(A, min(A, cap), replace=False) if A > cap else np.arange(A)
            ob = {L: _orb_node(mid, L, orb_root)[idx] for L in ORB_LAYERS}
            ub = {L: _uma_node_raw(mid, L, uma_root)[idx] for L in UMA_LAYERS}
            eo = _elements(mid, "gnn_stacks_4", orb_root)
            eu = _elements(mid, "backbone", uma_root)
            if list(eo) != list(eu):      # atom alignment is the premise of every metric here
                mismatched += 1
                continue
            el = [eo[k] for k in idx]
        except Exception:
            continue
        for L in ORB_LAYERS:
            orb_parts[L].append(ob[L])
        for L in UMA_LAYERS:
            uma_parts[L].append(ub[L])
        el_parts.append(np.asarray(el, dtype="<U3"))
        total += len(idx); used += 1
        if total >= max_atoms:
            break
    if used == 0:
        sys.exit("ERROR: no materials with both ORB and UMA embeddings found.")
    orb = {L: np.concatenate(v, 0) for L, v in orb_parts.items()}
    uma_raw = {L: np.concatenate(v, 0) for L, v in uma_parts.items()}
    els = np.concatenate(el_parts, 0)
    print(f"  aligned atoms: N={total:,} from {used:,} materials"
          + (f"  ({mismatched} skipped: element order disagreed)" if mismatched else ""))
    return orb, uma_raw, els, total


def _uma_variant(uma_raw: dict, variant: str) -> dict:
    """UMA read-out channels from the raw [N, 9, 128] tensor.

    full     = flatten 9x128 -> 1152 (rotation-SENSITIVE: carries raw L=1/L=2)
    ang_norm = cat(L0, ||L1||, ||L2||) -> 384  (invariant; composition + geometry)
    geom_norm= cat(||L1||, ||L2||)     -> 256  (invariant; geometry only, no element identity)
    l0       = L=0 scalar channel      -> 128  (invariant; composition only)

    ang_norm is the channel used for the cross-model comparisons: it is the
    invariant read-out whose CONTENT matches ORB's node embedding (both carry
    composition and geometry). geom_norm drops element identity from UMA only,
    which would handicap the UMA->ORB direction for content rather than
    architecture, so it is not used as the matched channel.
    """
    if variant == "full":
        return {L: X.reshape(X.shape[0], -1) for L, X in uma_raw.items()}
    if variant == "l0":
        return {L: X[:, 0, :] for L, X in uma_raw.items()}
    if variant == "ang_norm":
        return {L: np.concatenate([X[:, 0, :],
                                   np.linalg.norm(X[:, 1:4, :], axis=1),
                                   np.linalg.norm(X[:, 4:9, :], axis=1)], axis=1)
                for L, X in uma_raw.items()}
    if variant == "geom_norm":
        return {L: np.concatenate([np.linalg.norm(X[:, 1:4, :], axis=1),
                                   np.linalg.norm(X[:, 4:9, :], axis=1)], axis=1)
                for L, X in uma_raw.items()}
    raise ValueError(variant)


def residualise_elements(X: np.ndarray, els: np.ndarray) -> np.ndarray:
    """Regress out element identity: subtract the mean embedding of each element.

    This is exactly OLS on a one-hot atomic-number design, so the residual is the
    part of the representation NOT explained by which element the atom is. Both
    models are treated identically, which is what makes the comparison symmetric --
    UMA's geom_norm channel would remove composition from one side only.
    """
    R = X.astype(np.float64).copy()
    for e in np.unique(els):
        m = els == e
        R[m] -= R[m].mean(0, keepdims=True)
    return R


# --------------------------------------------------------------------------- grids
def compute_grids(orb: dict, uma: dict, seed: int, k: int = 10):
    """CKA / SVCCA / mutual-kNN + permuted-CKA baseline over the ORB×UMA layer grid."""
    orb_c = {L: _center(orb[L]) for L in ORB_LAYERS}
    uma_c = {L: _center(uma[L]) for L in UMA_LAYERS}
    N = next(iter(orb.values())).shape[0]
    perm = np.random.default_rng(seed).permutation(N)
    nE, nS = len(ORB_LAYERS), len(UMA_LAYERS)
    cka = np.full((nE, nS), np.nan); svc = np.full((nE, nS), np.nan)
    mkn = np.full((nE, nS), np.nan); ckap = np.full((nE, nS), np.nan)
    for i, LO in enumerate(ORB_LAYERS):
        for j, LU in enumerate(UMA_LAYERS):
            cka[i, j] = qc.linear_cka(orb_c[LO], uma_c[LU])
            svc[i, j] = qc.svcca(orb[LO], uma[LU], seed=seed)
            mkn[i, j] = mutual_knn(orb[LO], uma[LU], k=k, seed=seed)
            ckap[i, j] = qc.linear_cka(orb_c[LO], uma_c[LU][perm])
        print(f"  ORB {ORB_LBL[i]:>3s}: CKA {np.round(cka[i], 2)}", flush=True)
    return {"cka": cka, "svcca": svc, "mutual_knn": mkn, "cka_permuted": ckap}


def _plot(grids: dict, variant: str, N: int, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for ax, (key, ttl) in zip(axes, METRICS):
        M = grids[key]
        vmax = max(0.2, float(np.nanmax(M)))
        im = ax.imshow(M, vmin=0, vmax=vmax, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(UMA_LBL))); ax.set_xticklabels(UMA_LBL)
        ax.set_yticks(range(len(ORB_LBL))); ax.set_yticklabels(ORB_LBL)
        ax.set_xlabel("UMA (FairChem) layer"); ax.set_ylabel("ORB layer"); ax.set_title(ttl)
        for a in range(M.shape[0]):
            for b in range(M.shape[1]):
                if np.isfinite(M[a, b]):
                    ax.text(b, a, f"{M[a,b]:.2f}", ha="center", va="center",
                            color="white" if M[a, b] < 0.5 * vmax else "black", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    base = float(np.nanmean(grids["cka_permuted"]))
    fig.suptitle(f"ORB ↔ UMA representational convergence  (UMA node = {variant}; "
                 f"atoms N={N:,}; CKA permutation baseline ≈ {base:.3f})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def _summary_lines(grids: dict, variant: str, N: int) -> list[str]:
    lines = [f"[UMA node variant: {variant}]  (atoms N={N:,})"]
    for key, _ in METRICS:
        M = grids[key]
        i, j = np.unravel_index(np.nanargmax(M), M.shape)
        lines.append(f"  {key:11s} peak {M[i,j]:.3f} at ORB {ORB_LBL[i]} × UMA {UMA_LBL[j]}")
    lines.append(f"  CKA permutation baseline (mean): {np.nanmean(grids['cka_permuted']):.3f}")
    return lines


# --------------------------------------------------------------------------- main
# --------------------------------------------------------------------------- functional convergence
# CKA/SVCCA are *structural* (do the geometries look alike?). The metric literature's
# complaint is that structural similarity need not track function. The standard remedy
# is model stitching: can representation A be linearly mapped into B and still work?
# We add two cheap, principled functional measures on the same atom-aligned embeddings
# (+ pooled material embeddings), no re-extraction:
#   (1) linear cross-predictability: ridge A->B, held-out R² (variance of B recoverable
#       from A). Calibrated by within-model consecutive-layer R² (upper bound) and a
#       shuffled-atom baseline (lower bound).
#   (2) functional stitching: train a linear probe on UMA's pooled embedding, map ORB
#       into UMA-space with a fitted ridge, apply the *same* probe, and ask whether the
#       property accuracy survives.
def _xpred_r2(A, B, seed, alpha=100.0, test=0.3):
    """Held-out R² of a ridge map A -> B (variance-weighted over B's columns)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(seed); n = A.shape[0]
    idx = rng.permutation(n); ntr = int(n * (1 - test)); tr, te = idx[:ntr], idx[ntr:]
    sa = StandardScaler().fit(A[tr]); sb = StandardScaler().fit(B[tr])
    m = Ridge(alpha=alpha).fit(sa.transform(A[tr]), sb.transform(B[tr]))
    return float(r2_score(sb.transform(B[te]), m.predict(sa.transform(A[te])),
                          multioutput="variance_weighted"))


def linear_predictivity(orb, uma, seed):
    """(1) Linear predictivity grid + within-model calibration.

    Ridge map between two models' embeddings, scored by held-out variance-weighted
    R^2 on the target representation. This is the linear-predictivity /
    encoding-model similarity measure; it is unrelated to "cross-prediction-powered
    inference" (a semi-supervised estimation method), hence the explicit name.
    """
    nO, nU = len(ORB_LAYERS), len(UMA_LAYERS)
    R = np.full((nO, nU), np.nan); Rrev = np.full((nO, nU), np.nan)
    for i, LO in enumerate(ORB_LAYERS):
        for j, LU in enumerate(UMA_LAYERS):
            R[i, j] = _xpred_r2(orb[LO], uma[LU], seed)
            Rrev[i, j] = _xpred_r2(uma[LU], orb[LO], seed)
        print(f"  ORB {ORB_LBL[i]:>3s} -> UMA : R2 {np.round(R[i], 2)}", flush=True)
    orb_ord = ["encoder", "gnn_stacks_0", "gnn_stacks_1", "gnn_stacks_2", "gnn_stacks_3", "gnn_stacks_4"]
    uma_ord = ["embed", "block_0", "block_1", "block_2", "block_3", "backbone"]
    orb_consec = np.array([_xpred_r2(orb[a], orb[b], seed) for a, b in zip(orb_ord[:-1], orb_ord[1:])])
    uma_consec = np.array([_xpred_r2(uma[a], uma[b], seed) for a, b in zip(uma_ord[:-1], uma_ord[1:])])
    rng = np.random.default_rng(seed); B = uma["backbone"]
    shuffled = _xpred_r2(orb["gnn_stacks_4"], B[rng.permutation(B.shape[0])], seed)
    return {"orb2uma_r2": R, "uma2orb_r2": Rrev, "orb_consec_r2": orb_consec,
            "uma_consec_r2": uma_consec, "shuffled_r2": shuffled}


def _uma_pooled_map(uma_pool_dir, channel, layer):
    """{mp_id: pooled UMA node vector} for one channel and layer.

    With --uma-pool-dir unset this reads the per-material caches that
    stages/geometry/organisation_uma.py writes (route A). With it set to the downloaded
    gated dataset (route B) it reads embeddings/node/uma-node-{channel}-{layer}.parquet.
    """
    if uma_pool_dir is None:
        p = P.RESULTS / "fairchem/plots/nodes" / f"{layer}_nodes_pooled_{channel}_umap_cache.npz"
        if not p.exists():
            raise SystemExit(f"missing pooled UMA cache: {p}\nRun stages/geometry/organisation_uma.py "
                             "first, or pass --uma-pool-dir pointing at the downloaded gated dataset.")
        d = np.load(p, allow_pickle=True)
        return dict(zip(d["mp_ids"].astype(str), d["X_raw"].astype(np.float32)))
    import pandas as pd
    root = Path(uma_pool_dir)
    for cand in (root / "embeddings/node" / f"uma-node-{channel}-{layer}.parquet",
                 root / f"uma-node-{channel}-{layer}.parquet"):
        if cand.exists():
            u = pd.read_parquet(cand)
            return dict(zip(u["mp_id"].astype(str), u["emb"]))
    raise SystemExit(f"no uma-node-{channel}-{layer}.parquet under {root} or {root}/embeddings/node")


def _pooled_aligned(orb_dir, uma_pool_dir, orb_layer, uma_layer, channel="ang_norm"):
    """Material-level pooled ORB node[256] and UMA node[`channel`] + targets, aligned by mp_id."""
    import pandas as pd
    o = np.load(Path(orb_dir) / "property_prediction/features" / f"{orb_layer}_pooled.npz",
                allow_pickle=True)
    orb_ids = o["mp_ids"].astype(str); orb_emb = o["node"].astype(np.float32)
    umap = _uma_pooled_map(uma_pool_dir, channel, uma_layer)
    tt = np.load(Path(orb_dir) / "property_prediction/features/targets.npz", allow_pickle=True)
    tid = {m: k for k, m in enumerate(tt["mp_ids"].astype(str))}
    keep = [(k, m) for k, m in enumerate(orb_ids)
            if m in umap and m in tid and np.isfinite(orb_emb[k]).all()]
    ok = [k for k, _ in keep]; ids = [m for _, m in keep]
    Xo = orb_emb[ok]
    Xu = np.asarray([np.asarray(umap[m], dtype=np.float32) for m in ids])
    tsel = [tid[m] for m in ids]
    tgt = {key: tt[key][tsel] for key in tt.files if key != "mp_ids"}
    return Xo, Xu, tgt


def functional_stitching(orb_dir, uma_pool_dir, seed, max_mats=40000, channel="ang_norm"):
    """(2) Does a UMA-trained linear probe still work on ORB stitched into UMA-space?

    Subsampled to `max_mats` materials: this is a native-vs-stitched *contrast* on the
    same probe and split, so it does not need the full 155k set to be valid."""
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, f1_score
    REG = ["reg__formation_energy/atom", "reg__energy_above_hull", "reg__band_gap",
           "reg__density", "reg__bulk_modulus", "reg__mag_density"]
    CLF = ["cat__crystal_system", "cat__metal_insulator", "cat__magnetic_label"]
    pairs = [("near-input", "encoder", "embed"), ("deep", "gnn_stacks_4", "backbone")]
    rows = []
    for tag, ol, ul in pairs:
        Xo, Xu, tgt = _pooled_aligned(orb_dir, uma_pool_dir, ol, ul, channel)
        rng = np.random.default_rng(seed); n = len(Xo)
        if n > max_mats:                      # subsample for tractability
            sub = rng.choice(n, max_mats, replace=False)
            Xo, Xu = Xo[sub], Xu[sub]
            tgt = {k: v[sub] for k, v in tgt.items()}
            n = max_mats
        idx = rng.permutation(n); ntr = int(n * 0.8); tr, te = idx[:ntr], idx[ntr:]
        so = StandardScaler().fit(Xo[tr]); su = StandardScaler().fit(Xu[tr])
        Xo_tr, Xo_te = so.transform(Xo[tr]), so.transform(Xo[te])
        Xu_tr, Xu_te = su.transform(Xu[tr]), su.transform(Xu[te])
        # both directions: the probe is always trained on the DESTINATION model's own
        # embedding, then applied to the source model mapped into that space.
        dirs = [("ORB->UMA", Xo_tr, Xo_te, Xu_tr, Xu_te),
                ("UMA->ORB", Xu_tr, Xu_te, Xo_tr, Xo_te)]
        for dname, Xs_tr, Xs_te, Xd_tr, Xd_te in dirs:
            stitch = Ridge(alpha=100.0).fit(Xs_tr, Xd_tr)   # source -> destination space
            Xhat_te = stitch.predict(Xs_te)                 # source stitched into destination space
            for key in REG + CLF:
                y = tgt[key]; clf = key in CLF
                m = np.array([str(v) not in ("", "nan", "None") for v in y]) if clf else np.isfinite(
                    np.asarray(y, dtype=np.float64) if not clf else np.zeros(len(y)))
                if not clf:
                    yv = np.asarray(y, dtype=np.float64); m = np.isfinite(yv)
                ytr = m[tr]; yte = m[te]
                if ytr.sum() < 50 or yte.sum() < 20:
                    continue
                if clf:
                    yt = y[tr][ytr]; ye = y[te][yte]
                    P = LogisticRegression(max_iter=300, C=1.0, n_jobs=-1).fit(Xd_tr[ytr], yt)
                    native = f1_score(ye, P.predict(Xd_te[yte]), average="macro")
                    stit = f1_score(ye, P.predict(Xhat_te[yte]), average="macro")
                    metric = "macro-F1"
                else:
                    yt = yv[tr][ytr]; ye = yv[te][yte]
                    P = Ridge(alpha=10.0).fit(Xd_tr[ytr], yt)
                    native = r2_score(ye, P.predict(Xd_te[yte]))
                    stit = r2_score(ye, P.predict(Xhat_te[yte]))
                    metric = "R2"
                rows.append((tag, dname, key.split("__")[1], metric, round(native, 3),
                             round(stit, 3), round(stit - native, 3)))
                print(f"  [{tag}] {dname} {key.split('__')[1]:20s} "
                      f"native {native:.3f} stitched {stit:.3f}", flush=True)
    return rows


def _pooled_all_layers(orb_dir, uma_pool_dir, channel, max_mats, seed):
    """All ORB and UMA pooled layers on one common material set, plus targets.

    Loaded once so the layer grid does not re-read 12 files per cell.
    """
    import pandas as pd
    orb_raw, uma_raw = {}, {}
    for L in ORB_LAYERS:
        d = np.load(Path(orb_dir) / "property_prediction/features" / f"{L}_pooled.npz",
                    allow_pickle=True)
        orb_raw[L] = (d["mp_ids"].astype(str), d["node"].astype(np.float32))
    for L in UMA_LAYERS:
        uma_raw[L] = _uma_pooled_map(uma_pool_dir, channel, L)
    tt = np.load(Path(orb_dir) / "property_prediction/features/targets.npz", allow_pickle=True)
    tid = {m: k for k, m in enumerate(tt["mp_ids"].astype(str))}

    ids = set(orb_raw[ORB_LAYERS[0]][0]) & set(tid)
    for L in ORB_LAYERS:
        ids &= set(orb_raw[L][0])
    for L in UMA_LAYERS:
        ids &= set(uma_raw[L])
    ids = sorted(ids)
    rng = np.random.default_rng(seed)
    if len(ids) > max_mats:
        ids = [ids[k] for k in sorted(rng.choice(len(ids), max_mats, replace=False))]
    print(f"  [stitch-grid] {len(ids):,} materials common to all layers", flush=True)

    orb = {}
    for L in ORB_LAYERS:
        mp, emb = orb_raw[L]
        pos = {m: k for k, m in enumerate(mp)}
        orb[L] = emb[[pos[m] for m in ids]]
    uma = {L: np.asarray([np.asarray(uma_raw[L][m], dtype=np.float32) for m in ids])
           for L in UMA_LAYERS}
    tsel = [tid[m] for m in ids]
    tgt = {k: tt[k][tsel] for k in tt.files if k != "mp_ids"}
    return orb, uma, tgt


def stitching_grid(orb_dir, uma_pool_dir, seed, max_mats=40000, channel="ang_norm"):
    """Functional stitching over the FULL ORB x UMA layer grid, both directions.

    The native probe depends only on the DESTINATION layer, so it is fitted once per
    destination and reused across every source layer -- 6 probe sets per direction
    instead of 36. Only the ridge map and the prediction change per cell.

    Returns dict of [n_orb, n_uma, n_target] arrays of native / stitched / delta.
    """
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, f1_score
    REG = ["reg__formation_energy/atom", "reg__energy_above_hull", "reg__band_gap",
           "reg__density", "reg__bulk_modulus", "reg__mag_density"]
    CLF = ["cat__crystal_system", "cat__metal_insulator", "cat__magnetic_label"]
    KEYS = CLF + REG

    orb, uma, tgt = _pooled_all_layers(orb_dir, uma_pool_dir, channel, max_mats, seed)
    n = len(next(iter(orb.values())))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n); ntr = int(n * 0.8); tr, te = idx[:ntr], idx[ntr:]

    # standardise every layer once
    Z = {}
    for tagm, D in (("orb", orb), ("uma", uma)):
        for L, X in D.items():
            sc = StandardScaler().fit(X[tr])
            Z[(tagm, L)] = (sc.transform(X[tr]), sc.transform(X[te]))

    # per-target masks
    masks = {}
    for key in KEYS:
        y = tgt[key]
        if key in CLF:
            m = np.array([str(v) not in ("", "nan", "None") for v in y])
            masks[key] = (m, y)
        else:
            yv = np.asarray(y, dtype=np.float64)
            masks[key] = (np.isfinite(yv), yv)

    def fit_native(dst):
        """Probes trained on the destination layer's own embedding + its native scores."""
        Xtr, Xte = Z[dst]
        out = {}
        for key in KEYS:
            m, y = masks[key]
            mtr, mte = m[tr], m[te]
            if mtr.sum() < 50 or mte.sum() < 20:
                out[key] = None; continue
            if key in CLF:
                P = LogisticRegression(max_iter=300, C=1.0, n_jobs=-1).fit(Xtr[mtr], y[tr][mtr])
                nat = f1_score(y[te][mte], P.predict(Xte[mte]), average="macro")
            else:
                P = Ridge(alpha=10.0).fit(Xtr[mtr], y[tr][mtr])
                nat = r2_score(y[te][mte], P.predict(Xte[mte]))
            out[key] = (P, nat)
        return out

    nO, nU, nT = len(ORB_LAYERS), len(UMA_LAYERS), len(KEYS)
    res = {d: {w: np.full((nO, nU, nT), np.nan) for w in ("native", "stitched")}
           for d in ("orb2uma", "uma2orb")}

    for dname, srcs, dsts in (("orb2uma", [("orb", L) for L in ORB_LAYERS],
                                          [("uma", L) for L in UMA_LAYERS]),
                              ("uma2orb", [("uma", L) for L in UMA_LAYERS],
                                          [("orb", L) for L in ORB_LAYERS])):
        for dj, dst in enumerate(dsts):
            probes = fit_native(dst)
            Xd_tr, _ = Z[dst]
            for si, src in enumerate(srcs):
                Xs_tr, Xs_te = Z[src]
                Xhat_te = Ridge(alpha=100.0).fit(Xs_tr, Xd_tr).predict(Xs_te)
                for t, key in enumerate(KEYS):
                    if probes[key] is None:
                        continue
                    P, nat = probes[key]
                    m, y = masks[key]; mte = m[te]
                    pred = P.predict(Xhat_te[mte])
                    st = (f1_score(y[te][mte], pred, average="macro") if key in CLF
                          else r2_score(y[te][mte], pred))
                    # index as [orb, uma] regardless of direction
                    i, j = (si, dj) if dname == "orb2uma" else (dj, si)
                    res[dname]["native"][i, j, t] = nat
                    res[dname]["stitched"][i, j, t] = st
            print(f"  [stitch-grid] {dname} dst={dst[1]} done", flush=True)

    out = {"stitch_targets": np.array([k.split("__")[1] for k in KEYS])}
    for d in res:
        out[f"stitchgrid__{d}_native"] = res[d]["native"]
        out[f"stitchgrid__{d}_stitched"] = res[d]["stitched"]
        out[f"stitchgrid__{d}_delta"] = res[d]["stitched"] - res[d]["native"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="ORB↔UMA cross-model convergence grids")
    ap.add_argument("--orb-dir", default=str(P.RESULTS / "orb"))
    ap.add_argument("--uma-dir", default=str(P.RESULTS / "fairchem"))
    ap.add_argument("--out-dir", default=str(P.RESULTS / "convergence_orb_uma"))
    ap.add_argument("--max-atoms", type=int, default=30000)
    ap.add_argument("--cap", type=int, default=8, help="max atoms sampled per material")
    ap.add_argument("--max-mats", type=int, default=20000, help="material scan ceiling")
    ap.add_argument("--uma-feature", choices=["full", "l0", "ang_norm", "geom_norm", "both"],
                    default="both")
    ap.add_argument("--residualise", action="store_true",
                    help="also run the grids and linear predictivity on element-residualised "
                         "embeddings (per-element mean removed from BOTH models), to test how "
                         "much cross-model similarity is carried by element identity alone")
    ap.add_argument("--stitch-grid", action="store_true",
                    help="functional stitching over the FULL ORB x UMA layer grid "
                         "(both directions, all 9 targets) -> heatmaps")
    ap.add_argument("--xchannel", default="ang_norm",
                    choices=["full", "l0", "ang_norm", "geom_norm"],
                    help="UMA channel for linear predictivity AND stitching; ang_norm is the "
                         "invariant read-out whose content matches ORB's node embedding")
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--functional", action="store_true",
                    help="also run linear cross-prediction R² + functional stitching")
    ap.add_argument("--skip-grids", action="store_true",
                    help="skip the CKA/SVCCA/mutual-kNN grids (functional-only run)")
    ap.add_argument("--pooled-only", action="store_true",
                    help="read no per-atom activations: only the functional stitching (with --functional) and "
                         "the stitching grid (with --stitch-grid), which use pooled vectors; this is what runs "
                         "from the published datasets")
    ap.add_argument("--stitch-max-mats", type=int, default=40000,
                    help="materials subsampled for the stitching test")
    ap.add_argument("--uma-pool-dir", default=None,
                    help="downloaded gated UMA dataset (route B); default: the pooled caches "
                         "written by organisation_uma.py (route A)")
    args = ap.parse_args()

    orb_root, uma_root = Path(args.orb_dir), Path(args.uma_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    variants = ["full", "l0"] if args.uma_feature == "both" else [args.uma_feature]

    t0 = time.time()
    if args.pooled_only:
        if args.residualise:
            raise SystemExit("--residualise works on per-atom activations; drop it with --pooled-only")
        args.skip_grids = True
        orb = uma_raw = els = None
        npz_payload = dict(orb_labels=ORB_LBL, uma_labels=UMA_LBL)
        summary = ["ORB↔UMA cross-model convergence  (pooled vectors only)",
                   f"ORB layers: {ORB_LBL}", f"UMA layers: {UMA_LBL}", ""]
    else:
        print(f"ORB↔UMA convergence  (budget {args.max_atoms:,} atoms, cap {args.cap}/material)")
        orb, uma_raw, els, N = collect_aligned(orb_root, uma_root, args.max_atoms, args.cap,
                                          args.max_mats, args.seed)
        npz_payload = dict(orb_labels=ORB_LBL, uma_labels=UMA_LBL, n_atoms=N)
        summary = [f"ORB↔UMA cross-model convergence  (atoms N={N:,})",
                   f"ORB layers: {ORB_LBL}", f"UMA layers: {UMA_LBL}", ""]
    if not args.skip_grids:
        for variant in variants:
            print(f"\n[grids] UMA node variant = {variant}")
            uma = _uma_variant(uma_raw, variant)
            grids = compute_grids(orb, uma, args.seed, k=args.knn_k)
            _plot(grids, variant, N, out_dir / f"convergence_uma-{variant}.png")
            for key, M in grids.items():
                npz_payload[f"{variant}__{key}"] = M
            summary += _summary_lines(grids, variant, N) + [""]

    if args.residualise:
        print("\n[residualised] removing the per-element mean from BOTH models")
        orb_r = {L: residualise_elements(X, els) for L, X in orb.items()}
        uma_x = _uma_variant(uma_raw, args.xchannel)
        uma_r = {L: residualise_elements(X, els) for L, X in uma_x.items()}
        # how much of each representation was element identity?
        for lab, A, B in (("ORB", orb, orb_r), ("UMA", uma_x, uma_r)):
            for L in list(A)[:1] + list(A)[-1:]:
                kept = float(np.var(B[L]) / np.var(A[L].astype(np.float64)))
                print(f"  [residualised] {lab} {L}: {kept*100:5.1f}% of variance remains", flush=True)
        grids_r = compute_grids(orb_r, uma_r, args.seed, k=args.knn_k)
        for key, M in grids_r.items():
            npz_payload[f"{args.xchannel}_resid__{key}"] = M
        summary += _summary_lines(grids_r, f"{args.xchannel}_resid", N) + [""]
        xp_r = linear_predictivity(orb_r, uma_r, args.seed)
        for key, M in xp_r.items():
            npz_payload[f"linpred_resid__{key}"] = M
        i4, j4 = ORB_LBL.index("g4"), UMA_LBL.index("bb")
        summary += ["Linear predictivity R2, element-residualised (both models):",
                    f"  deep ORB->UMA {xp_r['orb2uma_r2'][i4, j4]:.3f}   "
                    f"UMA->ORB {xp_r['uma2orb_r2'][i4, j4]:.3f}", ""]

    # ---- functional convergence: linear cross-prediction + stitching ----
    if args.functional and args.pooled_only:
        print("\n[functional] --pooled-only: linear predictivity needs per-atom activations, skipped")
    if args.functional and not args.pooled_only:
        uma_x = _uma_variant(uma_raw, args.xchannel)
        print(f"\n[functional] linear predictivity R²  (UMA node = {args.xchannel})")
        xp = linear_predictivity(orb, uma_x, args.seed)
        for key, M in xp.items():
            npz_payload[f"linpred__{key}"] = M
        R = xp["orb2uma_r2"]
        near = R[ORB_LBL.index("enc"), UMA_LBL.index("emb")]
        deep = R[ORB_LBL.index("g4"), UMA_LBL.index("bb")]
        summary += [f"Linear predictivity R² (ORB->UMA, {args.xchannel} channel):",
                    f"  near-input (enc->emb): {near:.3f}",
                    f"  deep       (g4->bb)  : {deep:.3f}",
                    f"  within-ORB consecutive (upper bound): {np.round(xp['orb_consec_r2'],2)}",
                    f"  within-UMA consecutive (upper bound): {np.round(xp['uma_consec_r2'],2)}",
                    f"  shuffled-atom baseline (lower bound): {xp['shuffled_r2']:.3f}", ""]
    if args.functional:
        print("\n[functional] stitching: UMA-trained linear probe applied to ORB->UMA map")
        st = functional_stitching(args.orb_dir, args.uma_pool_dir, args.seed,
                                  max_mats=args.stitch_max_mats, channel=args.xchannel)
        import csv
        with open(out_dir / "functional_stitching.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair", "direction", "target", "metric", "native", "stitched", "delta"])
            w.writerows(st)
        summary += ["Functional stitching (native vs destination-probe on stitched source):"]
        summary += [f"  [{r[0]}] {r[1]:9s} {r[2]:20s} {r[3]:8s} "
                    f"native {r[4]:.3f}  stitched {r[5]:.3f}  Δ {r[6]:+.3f}" for r in st] + [""]

    # Merge into any existing grids.npz rather than replacing it. A partial run
    # (e.g. --skip-grids, or a single --uma-feature) otherwise silently drops the
    # keys it did not recompute, which has destroyed the CKA/SVCCA payload before.
    gpath = out_dir / "grids.npz"
    if gpath.exists():
        prev = np.load(gpath, allow_pickle=True)
        kept = [k for k in prev.files if k not in npz_payload]
        merged = {k: prev[k] for k in kept}
        merged.update(npz_payload)
        if kept:
            print(f"  [grids.npz] kept {len(kept)} existing keys, updated {len(npz_payload)}")
        npz_payload = merged
    if args.stitch_grid:
        print("\n[stitch-grid] functional stitching over the full layer grid")
        sg = stitching_grid(args.orb_dir, args.uma_pool_dir, args.seed,
                            max_mats=args.stitch_max_mats, channel=args.xchannel)
        npz_payload.update(sg)
        D = sg["stitchgrid__orb2uma_delta"]; R = sg["stitchgrid__uma2orb_delta"]
        summary += [f"Stitching grid (mean Delta over 9 targets, {args.xchannel}):",
                    f"  ORB->UMA best cell {np.nanmax(np.nanmean(D, 2)):+.3f}, "
                    f"worst {np.nanmin(np.nanmean(D, 2)):+.3f}",
                    f"  UMA->ORB best cell {np.nanmax(np.nanmean(R, 2)):+.3f}, "
                    f"worst {np.nanmin(np.nanmean(R, 2)):+.3f}", ""]

    np.savez(gpath, **npz_payload)
    (out_dir / "summary.txt").write_text("\n".join(summary).rstrip() + "\n")
    print("\n" + "\n".join(summary).rstrip())
    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
