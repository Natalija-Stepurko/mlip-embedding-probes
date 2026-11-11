"""
mlipprobes/common.py — Shared machinery for ORB and FairChem embedding QC scripts

This module ports the helpers from the QC notebook (orb_embedding_qc.ipynb) into a
re-usable form so that stages/geometry/health_qc_orb.py and stages/geometry/health_qc_uma.py
can drive identical analyses on production-scale cached UMAP outputs.

Sections
--------
1.  Configuration dataclasses (QCConfig, EmbType)
2.  Manifest + property helpers (copied from the organisation scripts)
3.  Sampling helpers (sqrt-weighted stratified slice, uniform query rows)
4.  K-stability sweep math (notebook cell 6, verbatim)
5.  Geometry diagnostics (covariance spectrum, cosine, Jaccard, CKA, SVCCA)
6.  HDBSCAN helpers (adaptive calibration, scatter grid, post-hoc, ARI/NMI)
7.  Summary assembly + progress queue listener
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import gc
import io
import json
import math
import multiprocessing as mp
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

MAG_THR = 1e-4  # |total_magnetization| / volume threshold for "Magnetic" label


@dataclass
class EmbType:
    """One embedding type within a QC run."""
    key: str                        # cache filename token, e.g. "nodes", "nodes_pooled_l1_polar"
    label: str                      # human title, e.g. "Node (atom-level)"
    color: str                      # consistent color across plots
    kind: str                       # "atom" | "pooled" | "edge" | "edge_pooled"
    l2_normalize: bool = False      # True only for ORB edges/edges_pooled
    # Cache path resolver: (emb_dir, plots_dir, layer) -> Path
    cache_path_fn: Callable[[Path, Path, str], Path] | None = None
    # k-NN categorical properties (per emb-type, since per-atom has element_identity)
    knn_cat_props: list[str] = field(default_factory=list)
    sweep_ks_base: list[int] = field(default_factory=lambda: [5, 10, 15, 20, 30, 50])
    # adaptive_k strategy: "sqrt_n_mats", "p75_pts_per_mat"
    adaptive_k: str = "sqrt_n_mats"
    # ARI/NMI extra per-row properties (e.g. element_identity for atoms, bond_type for edges)
    extra_ari_props: list[str] = field(default_factory=list)


@dataclass
class QCConfig:
    """Per-model QC configuration."""
    model_name: str                  # "ORB v3" / "FairChem UMA-S"
    embeddings_dir: Path             # $MLIP_PROBES_ROOT/results/{orb,fairchem}
    plots_dir: Path                  # $MLIP_PROBES_ROOT/results/{orb,fairchem}/plots
    structures_dir: Path             # $MLIP_PROBES_ROOT/structures
    qc_dir: Path                     # plots_dir / "qc"
    layers: list[str]                # ALL_LAYERS in order
    layer_labels: list[str]          # short labels for plots
    emb_types: list[EmbType]
    # Layer cmap (plasma across layer count)
    layer_colors: list = field(default_factory=list)

    def __post_init__(self):
        if not self.layer_colors:
            n = max(len(self.layers) - 1, 1)
            self.layer_colors = [
                plt.get_cmap("plasma")(i / n) for i in range(len(self.layers))
            ]
        self.qc_dir.mkdir(parents=True, exist_ok=True)
        (self.qc_dir / "_cache").mkdir(exist_ok=True)
        (self.qc_dir / "sections").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 2. Manifest + property helpers (copied from the organisation scripts so this module is
#    self-contained; the 03 scripts cannot be imported directly because of the
#    leading digit in their filenames)
# ---------------------------------------------------------------------------

EXCLUDED_CSV = Path(__file__).resolve().parent.parent / "config" / "excluded_materials.csv"


def excluded_materials() -> set:
    """mp_ids the extraction caches hold but the released corpus does not (154,879 -> 154,875)."""
    if not EXCLUDED_CSV.exists():
        return set()
    with open(EXCLUDED_CSV) as fh:
        return {line.split(",")[0].strip() for line in fh.read().splitlines()[1:] if line.strip()}


def load_manifest(jsonl_path: Path) -> tuple[dict, set]:
    """Read $MLIP_PROBES_ROOT/structures/index.jsonl → (mp_id → record dict, set of mp_ids)."""
    structs: dict[str, dict] = {}
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                # cif_file is recorded as written; if the root has moved since, the CIF sits next to the manifest
                cif = rec.get("cif_file")
                if cif and not Path(cif).exists() and (jsonl_path.parent / Path(cif).name).exists():
                    rec["cif_file"] = str(jsonl_path.parent / Path(cif).name)
                structs[rec["mp_id"]] = rec
    return structs, set(structs.keys())


def mag_label(structs: dict, m: str) -> str | None:
    tm  = structs[m].get("total_magnetization")
    vol = structs[m].get("volume")
    if tm is None:
        return None
    if vol and abs(tm / vol) >= MAG_THR:
        return "Magnetic"
    return "Non-magnetic"


def metal_insulator(structs: dict, m: str) -> str | None:
    bg = structs[m].get("band_gap")
    if bg is None:
        return None
    return "Metal" if bg == 0 else "Insulator"


def mag_density_dict(structs: dict) -> dict:
    return {
        m: s["total_magnetization"] / s["volume"]
        if s.get("total_magnetization") is not None and s.get("volume")
        else None
        for m, s in structs.items()
    }


def prop_dict(structs: dict, key: str) -> dict:
    return {m: s.get(key) for m, s in structs.items()}


# Categorical property arrays broadcast to each row (atom/edge/material)
CONT_KEY_MAP = {
    "formation_energy/atom": "formation_energy_per_atom",
    "energy_above_hull":     "energy_above_hull",
    "band_gap":              "band_gap",
    "density":               "density",
    "bulk_modulus":          "bulk_modulus",
}
CONT_PROPS = list(CONT_KEY_MAP.keys()) + ["mag_density"]
CAT_PROPS  = ["crystal_system", "metal_insulator", "magnetic_label"]


def cat_array(structs: dict, mids: np.ndarray, prop: str) -> np.ndarray:
    if prop == "crystal_system":
        return np.array([structs.get(m, {}).get("crystal_system") for m in mids], dtype=object)
    if prop == "metal_insulator":
        return np.array([metal_insulator(structs, m) if m in structs else None
                         for m in mids], dtype=object)
    if prop == "magnetic_label":
        return np.array([mag_label(structs, m) if m in structs else None
                         for m in mids], dtype=object)
    raise ValueError(f"unknown categorical property: {prop}")


def cont_array(structs: dict, mids: np.ndarray, prop: str) -> np.ndarray:
    if prop == "mag_density":
        md = mag_density_dict(structs)
        return np.array([md.get(m) for m in mids], dtype=float)
    key = CONT_KEY_MAP[prop]
    return np.array([structs.get(m, {}).get(key) for m in mids], dtype=float)


def pca_n_for_variance(evr: np.ndarray, threshold: float = 0.90) -> int:
    return int((np.cumsum(evr) < threshold).sum()) + 1


# ---------------------------------------------------------------------------
# 3. Sampling helpers
# ---------------------------------------------------------------------------

def sqrt_weighted_idx(class_arr: np.ndarray, n_target: int,
                       seed: int = 42, floor: int = 20) -> np.ndarray:
    """Stratified sqrt-weighted subsample over class labels.

    Ported from stages/geometry/organisation_orb.py:487-499 / 825-835 — each class gets
    a quota proportional to sqrt(count), with a floor so rare classes survive.
    """
    n_total = len(class_arr)
    if n_target <= 0 or n_target >= n_total:
        idx = np.arange(n_total)
        np.random.default_rng(seed).shuffle(idx)
        return idx

    counts = Counter(class_arr.tolist())
    sqrt_total = sum(np.sqrt(c) for c in counts.values())
    rng = np.random.default_rng(seed)
    picks: list[int] = []
    arr = np.asarray(class_arr)
    for cls, cnt in counts.items():
        n_i = int(n_target * np.sqrt(cnt) / sqrt_total)
        n_i = max(n_i, floor)
        n_i = min(n_i, cnt)
        cls_idx = np.where(arr == cls)[0]
        picks.extend(rng.choice(cls_idx, size=n_i, replace=False).tolist())
    out = np.array(picks)
    rng.shuffle(out)
    return out


def uniform_idx(n_total: int, n_target: int, seed: int = 42) -> np.ndarray:
    """Uniform random subset; returns all indices if n_target<=0 or >= n_total."""
    if n_target <= 0 or n_target >= n_total:
        idx = np.arange(n_total)
        np.random.default_rng(seed).shuffle(idx)
        return idx
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=n_target, replace=False)


# ---------------------------------------------------------------------------
# 4. K-stability sweep math (notebook cell 6, verbatim semantics)
# ---------------------------------------------------------------------------

def sweep_purity(query_labels, full_labels, nn: np.ndarray, Ks: list[int]) -> dict[int, float]:
    """Normalised k-NN purity swept over K.

    query_labels: per-query-row label (length = nn.shape[0])
    full_labels:  per-index-row label (nn values index into this)
    Result: 0 = random, 1 = perfect.
    """
    q_arr = np.asarray(query_labels, dtype=object)
    q_has = np.array([l is not None for l in q_arr])
    if q_has.sum() < 2:
        return {}
    f_arr = np.asarray(full_labels, dtype=object)
    f_has = np.array([l is not None for l in f_arr])
    q_str = np.where(q_has, q_arr.astype(str), "").astype(str)
    f_str = np.where(f_has, f_arr.astype(str), "").astype(str)
    cnt   = Counter(q_arr[q_has].astype(str))
    n_ok  = int(q_has.sum())
    base  = sum((c / n_ok) ** 2 for c in cnt.values())
    rows  = np.where(q_has)[0]
    out: dict[int, float] = {}
    for K in Ks:
        if K >= nn.shape[1]:
            continue
        neigh = nn[rows, :K]
        nb_l = f_str[neigh]
        nb_h = f_has[neigh]
        m    = (nb_l == q_str[rows, None]) & nb_h
        nc   = nb_h.sum(1)
        vp   = nc > 0
        if not vp.any():
            continue
        raw  = float(np.mean((m.sum(1) / np.maximum(nc, 1))[vp]))
        out[K] = (raw - base) / (1 - base) if base < 1 else 0.0
    return out


def sweep_lvr_dict(vdict: dict, query_mp_ids: np.ndarray, full_mp_ids: np.ndarray,
                    nn: np.ndarray, Ks: list[int]) -> dict[int, float | None]:
    """LVR swept over K from a {mp_id: value} dict broadcast to each row."""
    v_q = np.array([vdict.get(m) for m in query_mp_ids], dtype=float)
    v_f = np.array([vdict.get(m) for m in full_mp_ids], dtype=float)
    ok = ~np.isnan(v_q)
    n_ok = int(ok.sum())
    if n_ok < 2:
        return {}
    g = v_q[ok].std()
    if g == 0:
        return {}
    rows = np.where(ok)[0]
    out: dict[int, float | None] = {}
    for K in Ks:
        if K >= nn.shape[1] or n_ok < K + 1:
            continue
        neigh = nn[rows, :K]
        nb = v_f[neigh]
        with np.errstate(all="ignore"):
            s = np.nanstd(nb, 1, ddof=0)
        nv = np.sum(~np.isnan(nb), 1) >= 2
        out[K] = float(s[nv].mean() / g) if nv.any() else None
    return out


def sweep_lvr_arr(query_arr: np.ndarray, full_arr: np.ndarray,
                   nn: np.ndarray, Ks: list[int]) -> dict[int, float | None]:
    """LVR swept over K from a per-row value array (e.g. bond distances)."""
    v_q = np.asarray(query_arr, dtype=float)
    v_f = np.asarray(full_arr, dtype=float)
    ok = ~np.isnan(v_q)
    n_ok = int(ok.sum())
    if n_ok < 2:
        return {}
    g = v_q[ok].std()
    if g == 0:
        return {}
    rows = np.where(ok)[0]
    out: dict[int, float | None] = {}
    for K in Ks:
        if K >= nn.shape[1] or n_ok < K + 1:
            continue
        neigh = nn[rows, :K]
        nb = v_f[neigh]
        with np.errstate(all="ignore"):
            s = np.nanstd(nb, 1, ddof=0)
        nv = np.sum(~np.isnan(nb), 1) >= 2
        out[K] = float(s[nv].mean() / g) if nv.any() else None
    return out


def adaptive_k_value(strategy: str, n_pts: int, mp_ids: np.ndarray | None = None) -> int:
    """Adaptive K choice matching 03 analysis scripts and notebook conventions."""
    if strategy == "sqrt_n_mats":
        n = len(mp_ids) if mp_ids is not None else n_pts
        k = max(10, min(int(round(np.sqrt(n))), n // 10))
    elif strategy == "p75_pts_per_mat":
        assert mp_ids is not None
        counts = np.fromiter(Counter(mp_ids.tolist()).values(), dtype=float)
        k_p75 = int(np.percentile(counts, 75)) + 1 if len(counts) else 10
        k = max(10, min(k_p75, n_pts // 10))
    else:
        raise ValueError(f"unknown adaptive_k strategy: {strategy}")
    return max(2, k)


def adaptive_k_grid(k_ref: int, n_index: int,
                    anchors: tuple[int, ...] = (5, 10, 15, 20, 30, 50)) -> list[int]:
    """K-sweep grid that always brackets the adaptive k_ref with tested points on
    both sides, so the chosen K never sits at the plot's edge.

    Small-K anchors provide low-K context; a geometric spread spanning roughly
    [k_ref/3, 2*k_ref] keeps k_ref interior to the grid (values before *and*
    after it). When k_ref falls inside the anchor range (small k_ref) the grid is
    essentially the anchors; when it is large (e.g. pooled sqrt(n_mats)) the grid
    extends above it. Bounded by the kNN index size (k < n_index)."""
    k_ref = int(max(2, k_ref))
    cap = max(2, int(n_index) - 1)
    grid = {int(a) for a in anchors}
    grid.add(min(k_ref, cap))
    lo = max(2, k_ref // 3)
    hi = min(cap, max(2 * k_ref, k_ref + 4))
    if hi > lo:
        grid.update(int(round(x)) for x in np.geomspace(lo, hi, 6))
    grid = sorted(k for k in grid if 2 <= k <= cap)
    # Hard guarantee: at least one tested K strictly above k_ref when there's room.
    if k_ref < cap and not any(k > k_ref for k in grid):
        grid = sorted(set(grid) | {min(cap, k_ref + max(1, k_ref // 2))})
    return grid


def plot_k_sweep(pur_all: dict, lvr_all: dict,
                  pur_props: list[str], lvr_props: list[str],
                  k_ref: int, title: str, out_path: Path,
                  layer_lbls: list[str], layer_colors: list,
                  xscale: str = "linear") -> None:
    """3-column grid layout for the per-emb-type K-sweep figure (notebook cell 6)."""
    n_pur = len(pur_props)
    n_lvr = len(lvr_props)
    NCOLS  = 3
    HSPACE = 0.45

    n_pur_rows = math.ceil(n_pur / NCOLS)
    n_lvr_rows = math.ceil(n_lvr / NCOLS)

    fig_w = NCOLS * 4.5
    fig_h = 1.2 + 3.8 * (n_pur_rows + n_lvr_rows)
    fig = plt.figure(figsize=(fig_w, fig_h))

    TOP, BOT = 0.88, 0.06
    avail = TOP - BOT
    gap = 0.04

    def _ws(n: int) -> float:
        return n + max(0, n - 1) * HSPACE

    k = (avail - gap) / (_ws(n_pur_rows) + _ws(n_lvr_rows))
    pur_top = TOP
    pur_bot = TOP - k * _ws(n_pur_rows)
    lvr_top = pur_bot - gap
    lvr_bot = BOT

    gs_pur = gridspec.GridSpec(n_pur_rows, NCOLS, left=0.06, right=0.98,
                                top=pur_top, bottom=pur_bot,
                                hspace=HSPACE, wspace=0.30, figure=fig)
    gs_lvr = gridspec.GridSpec(n_lvr_rows, NCOLS, left=0.06, right=0.98,
                                top=lvr_top, bottom=lvr_bot,
                                hspace=HSPACE, wspace=0.30, figure=fig)

    for ci, p in enumerate(pur_props):
        ax = fig.add_subplot(gs_pur[ci // NCOLS, ci % NCOLS])
        for lbl, col in zip(layer_lbls, layer_colors):
            r = pur_all.get(p, {}).get(lbl, {})
            if not r:
                continue
            ks, vs = zip(*sorted(r.items()))
            ax.plot(ks, vs, "o-", color=col, lw=1.8)
        ax.axvline(k_ref, color="tomato", ls="--", lw=1.5)
        ax.set_xscale(xscale)
        ax.set_title(f"Purity — {p}", fontsize=8.5)
        ax.set_xlabel("K", fontsize=8)
        ax.set_ylabel("Norm. purity" if ci % NCOLS == 0 else "", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for ci in range(n_pur, n_pur_rows * NCOLS):
        fig.add_subplot(gs_pur[ci // NCOLS, ci % NCOLS]).axis("off")

    for ci, p in enumerate(lvr_props):
        ax = fig.add_subplot(gs_lvr[ci // NCOLS, ci % NCOLS])
        for lbl, col in zip(layer_lbls, layer_colors):
            r = lvr_all.get(p, {}).get(lbl, {})
            if not r:
                continue
            kvs = [(k_, v) for k_, v in sorted(r.items()) if v is not None]
            if not kvs:
                continue
            ks, vs = zip(*kvs)
            ax.plot(ks, vs, "o-", color=col, lw=1.8)
        ax.axvline(k_ref, color="tomato", ls="--", lw=1.5)
        ax.set_xscale(xscale)
        ax.set_title(f"LVR — {p}", fontsize=8.5)
        ax.set_xlabel("K", fontsize=8)
        ax.set_ylabel("LVR" if ci % NCOLS == 0 else "", fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    for ci in range(n_lvr, n_lvr_rows * NCOLS):
        fig.add_subplot(gs_lvr[ci // NCOLS, ci % NCOLS]).axis("off")

    handles = [plt.Line2D([0], [0], color=c, lw=2, marker="o", ms=6) for c in layer_colors]
    handles.append(plt.Line2D([0], [0], color="tomato", lw=1.5, ls="--"))
    labels = list(layer_lbls) + [f"K={k_ref}"]
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.97), fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=11, y=1.00)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def ksweep_table_text(pur_all: dict, lvr_all: dict,
                       pur_props: list[str], lvr_props: list[str],
                       k_ref: int, layer_lbls: list[str], section: str) -> str:
    buf = io.StringIO()
    buf.write("=" * 60 + "\n")
    buf.write(section + "\n")
    buf.write("=" * 60 + "\n")
    buf.write(f"At K={k_ref}:\n")
    buf.write(f"  {'Property':<28}" + "".join(f"  {l:>6}" for l in layer_lbls) + "\n")
    for p in pur_props:
        row = f"  purity:{p:<21}"
        for lbl in layer_lbls:
            v = pur_all.get(p, {}).get(lbl, {}).get(k_ref)
            row += f"  {v:6.3f}" if v is not None else "     N/A"
        buf.write(row + "\n")
    for p in lvr_props:
        row = f"  LVR:{p:<24}"
        for lbl in layer_lbls:
            r = lvr_all.get(p, {}).get(lbl)
            v = r.get(k_ref) if isinstance(r, dict) else None
            row += f"  {v:6.3f}" if v is not None else "     N/A"
        buf.write(row + "\n")
    buf.write("\n")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 5. Geometry diagnostics
# ---------------------------------------------------------------------------

COND_TOL = 1e-6


def spectrum_eigvals(X: np.ndarray, chunk: int = 10_000) -> np.ndarray:
    """Covariance eigenvalues (descending) of column-centered X.

    Uses a streaming float64 X^T X accumulator + eigh to avoid the ~2.8 GB
    float64 copy that sklearn PCA would make of a 300k × 1152 matrix.
    Result is identical (up to sign) to PCA(n_comp=d).explained_variance_.
    """
    X = X[np.isfinite(X).all(axis=1)]
    if X.shape[0] < 2:
        return np.zeros(0, dtype=np.float64)
    n, d = X.shape
    mean = np.zeros(d, dtype=np.float64)
    for i in range(0, n, chunk):
        mean += X[i:i + chunk].astype(np.float64).sum(axis=0)
    mean /= n
    cov = np.zeros((d, d), dtype=np.float64)
    for i in range(0, n, chunk):
        block = X[i:i + chunk].astype(np.float64) - mean
        cov += block.T @ block
    cov /= max(n - 1, 1)
    lam = np.linalg.eigvalsh(cov)[::-1].copy()
    return np.clip(lam, 0.0, None)


def concentration(lam: np.ndarray) -> tuple[float, float]:
    tot = lam.sum()
    if tot <= 0:
        return float("nan"), float("nan")
    return float(lam[0] / tot), float(lam[:10].sum() / tot)


def condition_number(lam: np.ndarray, tol: float = COND_TOL) -> float:
    if len(lam) == 0 or lam[0] <= 0:
        return float("nan")
    meaningful = lam[lam > tol * lam[0]]
    if len(meaningful) == 0:
        return float("nan")
    return float(lam[0] / meaningful[-1])


def mean_pairwise_cosine(X: np.ndarray, max_n: int = 2000,
                          seed: int = 42) -> tuple[float, float]:
    """Mean ± std of off-diagonal pairwise cosine similarity on a random subset.

    ~0 = healthy, → 1 = collapsing toward a common direction.
    """
    X = X[np.isfinite(X).all(axis=1)]
    if len(X) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if len(X) > max_n:
        X = X[rng.choice(len(X), size=max_n, replace=False)]
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.where(nrm == 0, 1.0, nrm)
    sim = Xn @ Xn.T
    iu = np.triu_indices(len(Xn), k=1)
    vals = sim[iu]
    return float(vals.mean()), float(vals.std())


def jaccard_rows(idx_a: np.ndarray, idx_b: np.ndarray) -> tuple[float, float]:
    """Mean / std of per-row Jaccard between two kNN-index arrays of equal shape."""
    n, k = idx_a.shape
    sa_set = [set(row.tolist()) for row in idx_a]
    sb_set = [set(row.tolist()) for row in idx_b]
    j = np.empty(n, dtype=np.float64)
    for i in range(n):
        sa, sb = sa_set[i], sb_set[i]
        u = len(sa | sb)
        j[i] = (len(sa & sb) / u) if u > 0 else 0.0
    return float(j.mean()), float(j.std())


def load_aligned_raw_layers(emb_type: EmbType, layers: list[str],
                              cache_path_resolver: Callable[[str], Path],
                              row_subsample: int = 0,
                              seed: int = 42) -> tuple[list[np.ndarray | None], np.ndarray | None]:
    """Load X_raw for every layer, intersect finite-row masks, optionally subsample.

    Returns (list of column-centered float64 matrices, joint row mask). Matrices
    are None where the cache is missing. L2-normalizes if emb_type.l2_normalize.
    """
    mats: list[np.ndarray | None] = []
    joint_mask: np.ndarray | None = None
    for layer in layers:
        p = cache_path_resolver(layer)
        if not p.exists():
            mats.append(None)
            continue
        X = np.load(p, allow_pickle=True)["X_raw"]
        if emb_type.l2_normalize:
            nrm = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.where(nrm == 0, 1.0, nrm)
        mats.append(X)
        m = np.isfinite(X).all(axis=1)
        joint_mask = m if joint_mask is None else (joint_mask & m)

    if joint_mask is None:
        return mats, None

    if row_subsample and joint_mask.sum() > row_subsample:
        rng = np.random.default_rng(seed)
        cand = np.where(joint_mask)[0]
        pick = rng.choice(cand, size=row_subsample, replace=False)
        sel = np.zeros_like(joint_mask)
        sel[pick] = True
        joint_mask = sel

    out: list[np.ndarray | None] = []
    for X in mats:
        if X is None:
            out.append(None)
        else:
            Xc = X[joint_mask].astype(np.float64)
            Xc = Xc - Xc.mean(axis=0, keepdims=True)
            out.append(Xc)
    return out, joint_mask


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Feature-space linear CKA (notebook cell 29 verbatim).

    X, Y must be column-centered, same number of rows.
    """
    xy = np.linalg.norm(Y.T @ X, "fro") ** 2
    xx = np.linalg.norm(X.T @ X, "fro")
    yy = np.linalg.norm(Y.T @ Y, "fro")
    return float(xy / (xx * yy)) if xx * yy > 0 else float("nan")


def svcca(Xa: np.ndarray, Xb: np.ndarray, var: float = 0.99,
           max_rows: int = 5000, seed: int = 42) -> float:
    """Notebook cell 30: SVD-denoise each layer, then CCA between subspaces."""
    rng = np.random.default_rng(seed)
    n = Xa.shape[0]
    if n > max_rows and max_rows > 0:
        idx = rng.choice(n, max_rows, replace=False)
        Xa, Xb = Xa[idx], Xb[idx]

    def _reduce(X: np.ndarray) -> np.ndarray:
        Xc = X - X.mean(0, keepdims=True)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        energy = np.cumsum(S ** 2) / np.sum(S ** 2)
        k = int(np.searchsorted(energy, var) + 1)
        return U[:, :k] * S[:k]

    A, B = _reduce(Xa), _reduce(Xb)
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.clip(s, 0, 1).mean())


def plot_cross_layer_heatmaps(mats_by_type: dict[str, np.ndarray],
                                emb_type_titles: dict[str, str],
                                emb_type_colors: dict[str, str],
                                layer_lbls: list[str],
                                title: str, cbar_label: str,
                                cmap: str, out_path: Path,
                                ncols: int = 2) -> None:
    """Render N heatmaps + an adjacent-layer line plot below (notebook cells 29/30)."""
    n_lay = len(layer_lbls)
    keys = list(mats_by_type.keys())
    n_types = len(keys)
    nrows_hm = math.ceil(n_types / ncols)
    height_ratios = [1] * nrows_hm + [0.8]
    fig = plt.figure(figsize=(6.5 * ncols, 3.2 * nrows_hm + 3.2))
    gs = fig.add_gridspec(nrows_hm + 1, ncols,
                          height_ratios=height_ratios,
                          hspace=0.55, wspace=0.30,
                          left=0.08, right=0.88, top=0.93, bottom=0.06)
    im = None
    for i, k in enumerate(keys):
        r, c = i // ncols, i % ncols
        ax = fig.add_subplot(gs[r, c])
        M = mats_by_type[k]
        im = ax.imshow(M, vmin=0, vmax=1, cmap=cmap)
        ax.set_xticks(range(n_lay))
        ax.set_xticklabels(layer_lbls, rotation=45, ha="right")
        ax.set_yticks(range(n_lay))
        ax.set_yticklabels(layer_lbls)
        ax.set_title(emb_type_titles.get(k, k), fontsize=10, pad=6)
        for a in range(n_lay):
            for b in range(n_lay):
                if not np.isnan(M[a, b]):
                    ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center",
                            color="white" if M[a, b] < 0.6 else "black", fontsize=7)
    if im is not None:
        cax = fig.add_axes([0.90, 0.42, 0.015, 0.45])
        fig.colorbar(im, cax=cax, label=cbar_label)

    pair_lbls = [f"{layer_lbls[i]}→{layer_lbls[i + 1]}" for i in range(n_lay - 1)]
    ax2 = fig.add_subplot(gs[nrows_hm, :])
    markers = ["o", "s", "^", "D", "v", "<", ">", "P", "X", "*", "h", "+"]
    for i, k in enumerate(keys):
        M = mats_by_type[k]
        adj = [M[j, j + 1] for j in range(n_lay - 1)]
        ax2.plot(range(n_lay - 1), adj, marker=markers[i % len(markers)],
                  color=emb_type_colors.get(k, f"C{i}"),
                  lw=2, ms=7, label=emb_type_titles.get(k, k))
    ax2.set_xticks(range(n_lay - 1))
    ax2.set_xticklabels(pair_lbls, rotation=20, ha="right")
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel(cbar_label)
    ax2.set_xlabel("Adjacent layer pair")
    ax2.set_title(f"Adjacent-Layer {cbar_label}")
    ax2.grid(True, alpha=0.3)
    # Single-row legend beneath the adjacent-layer line plot (clear of xlabel).
    ax2.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.42),
               ncol=len(keys), frameon=False)

    fig.suptitle(title, y=0.97)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. HDBSCAN helpers (notebook cells 33/35/37)
# ---------------------------------------------------------------------------

def adaptive_hdbscan_params(mp_ids: np.ndarray, n_points: int | None = None,
                              pct: int = 95, mcs_floor: int = 20,
                              ms_floor: int = 10, cap_frac: int = 10) -> tuple[int, int]:
    """min_cluster_size = p95(points/material); min_samples = mcs//2 (floored)."""
    counts = np.fromiter(Counter(np.asarray(mp_ids).tolist()).values(), dtype=float)
    n = int(n_points if n_points is not None else counts.sum())
    mcs = int(np.ceil(np.percentile(counts, pct))) if len(counts) else mcs_floor
    mcs = max(mcs_floor, min(mcs, max(mcs_floor, n // cap_frac)))
    return mcs, max(ms_floor, mcs // 2)


def run_hdbscan(X: np.ndarray, mcs: int, ms: int) -> np.ndarray:
    """sklearn HDBSCAN on X_pca slice. Returns int label array; -1 = noise."""
    from sklearn.cluster import HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=mcs, min_samples=ms, n_jobs=-1)
    return clusterer.fit_predict(X)


def hdbscan_scatter_grid(layer_lbls: list[str],
                           X_umaps: list[np.ndarray | None],
                           labels_per_layer: list[np.ndarray | None],
                           title: str, out_path: Path,
                           ncols: int = 4, legend_top: bool = False) -> None:
    """Per-layer UMAP scatter colored by HDBSCAN cluster (tab20; noise grey).

    Layout: nrows x ncols panels. By default the first empty cell holds the
    legend; with legend_top=True the panels are packed and the legend is placed
    as a multi-row figure legend across the top (better when the layers nearly
    fill the grid, e.g. FairChem's 6 layers).
    """
    n_lay = len(layer_lbls)
    nrows = math.ceil((n_lay + (0 if legend_top else 1)) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                              squeeze=False)
    fig.suptitle(title, fontsize=13, y=0.995)

    # Determine global max cluster count for palette sizing
    max_cl = 0
    for lbl in labels_per_layer:
        if lbl is not None and lbl.size > 0 and (lbl >= 0).any():
            max_cl = max(max_cl, int(lbl.max()) + 1)
    # Cycle a set of distinct qualitative colors (tab20 + tab20b + tab20c = 60),
    # indexing by cluster id so adjacent/large clusters stay differentiable.
    # The previous linspace-over-max_cl sampling of the discrete tab20 map
    # collapsed most clusters (and the whole legend) onto the first blue shades
    # whenever max_cl was large.
    base = np.concatenate([plt.get_cmap("tab20").colors,
                           plt.get_cmap("tab20b").colors,
                           plt.get_cmap("tab20c").colors])
    palette = base[np.arange(max(max_cl, 1)) % len(base)]

    flat = list(axes.flat)
    for i, (lbl_str, X_umap, lbl_arr) in enumerate(zip(layer_lbls, X_umaps, labels_per_layer)):
        ax = flat[i]
        ax.set_xticks([]); ax.set_yticks([])
        if X_umap is None or lbl_arr is None:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                     ha="center", va="center", color="#888")
            continue
        valid = lbl_arr >= 0
        if (~valid).any():
            ax.scatter(X_umap[~valid, 0], X_umap[~valid, 1], s=1, alpha=0.15,
                        c="#dddddd", linewidths=0, rasterized=True)
        if valid.any():
            colors = palette[lbl_arr[valid]]
            ax.scatter(X_umap[valid, 0], X_umap[valid, 1], s=1, alpha=0.45,
                        c=colors, linewidths=0, rasterized=True)
        n_clust = int(lbl_arr.max()) + 1 if (lbl_arr >= 0).any() else 0
        noise_pct = float((lbl_arr == -1).mean()) * 100
        ax.set_title(f"{lbl_str}  k={n_clust}  noise={noise_pct:.1f}%", fontsize=9)

    handles = [mlines.Line2D([], [], marker="o", linestyle="None",
                              color=palette[i], markersize=6, label=f"C{i}")
                for i in range(min(max_cl, 20))]
    handles.append(mlines.Line2D([], [], marker="o", linestyle="None",
                                   color="#dddddd", markersize=6, label="noise"))
    if legend_top:
        # Pack panels; turn off any trailing unused cells; legend across the top.
        for i in range(n_lay, len(flat)):
            flat[i].axis("off")
        ncol_leg = math.ceil(len(handles) / 2)  # ~2 rows
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.5, 0.965), ncol=ncol_leg, fontsize=8,
                   frameon=False, handletextpad=0.3, columnspacing=1.0)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
    else:
        # Legend in first empty panel
        legend_idx = n_lay
        for i in range(legend_idx, len(flat)):
            flat[i].axis("off")
        if legend_idx < len(flat):
            flat[legend_idx].legend(handles=handles, loc="center", fontsize=8,
                                      ncol=2, frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def posthoc_table(labels: np.ndarray, mids: np.ndarray, structs: dict,
                    elements: np.ndarray | None = None,
                    bond_types: np.ndarray | None = None,
                    distances: np.ndarray | None = None) -> str:
    """Per-cluster property breakdown (notebook cells 37-45 style)."""
    buf = io.StringIO()
    extra_cols = []
    if elements is not None:
        extra_cols.append("Element")
    if bond_types is not None:
        extra_cols.append("Bond")
    if distances is not None:
        extra_cols.append("d(Å)")

    head = (f"  {'Cluster':<9}{'N':>8}{'Crystal sys (%)':>22}"
             f"{'%Met':>7}{'%Mag':>7}")
    for c in extra_cols:
        head += f"{c:>14}"
    head += (f"{'Ef':>9}{'EaH':>9}{'Eg':>9}{'MagD':>9}{'Dens':>9}{'Bulk':>9}"
              f"{'eTot':>9}{'eIon':>9}{'eEle':>9}")
    buf.write(head + "\n")
    buf.write("  " + "-" * (len(head) - 2) + "\n")

    md = mag_density_dict(structs)
    for cl in sorted(set(labels.tolist())):
        mask = labels == cl
        n = int(mask.sum())
        c_mids = mids[mask]
        # Dominant crystal system
        cs = [structs.get(m, {}).get("crystal_system") for m in c_mids]
        cs = [c for c in cs if c]
        if cs:
            cc = Counter(cs).most_common(1)[0]
            cs_str = f"{cc[0][:7]:>7}({100*cc[1]/len(cs):>4.1f}%)"
        else:
            cs_str = "       N/A     "
        bg = [structs.get(m, {}).get("band_gap") for m in c_mids]
        bg_valid = [b for b in bg if b is not None]
        pct_met = 100 * sum(1 for b in bg_valid if b == 0) / len(bg_valid) if bg_valid else float("nan")
        md_vals = [md.get(m) for m in c_mids]
        md_valid = [v for v in md_vals if v is not None]
        pct_mag = 100 * sum(1 for v in md_valid if abs(v) >= MAG_THR) / len(md_valid) if md_valid else float("nan")

        def mean_cont(key):
            vs = [structs.get(m, {}).get(key) for m in c_mids]
            vs = [v for v in vs if v is not None]
            return float(np.mean(vs)) if vs else float("nan")

        ef = mean_cont("formation_energy_per_atom")
        eh = mean_cont("energy_above_hull")
        eg = mean_cont("band_gap")
        mag_mean = float(np.mean(md_valid)) if md_valid else float("nan")
        de = mean_cont("density")
        bu = mean_cont("bulk_modulus")

        cl_str = "noise" if cl == -1 else f"C{cl}"
        row = f"  {cl_str:<9}{n:>8,}{cs_str:>22}{pct_met:>7.1f}{pct_mag:>7.1f}"
        if elements is not None:
            elem_vals = elements[mask]
            ec = Counter(elem_vals.tolist()).most_common(1)[0] if len(elem_vals) else ("?", 0)
            row += f"{f'{ec[0]}({100*ec[1]/n:.0f}%)':>14}"
        if bond_types is not None:
            bt_vals = bond_types[mask]
            bc = Counter(bt_vals.tolist()).most_common(1)[0] if len(bt_vals) else ("?", 0)
            row += f"{f'{bc[0]}({100*bc[1]/n:.0f}%)':>14}"
        if distances is not None:
            d_vals = distances[mask]
            d_mean = float(np.mean(d_vals)) if len(d_vals) else float("nan")
            row += f"{d_mean:>14.3f}" if not np.isnan(d_mean) else f"{'N/A':>14}"

        def _fmt(v):
            return f"{v:>9.3f}" if not np.isnan(v) else f"{'N/A':>9}"

        row += _fmt(ef) + _fmt(eh) + _fmt(eg) + _fmt(mag_mean) + _fmt(de) + \
               _fmt(bu)
        buf.write(row + "\n")
    return buf.getvalue()


def ari_nmi_for_row_property(labels: np.ndarray, prop_values: np.ndarray,
                                is_continuous: bool, n_bins: int = 5,
                                min_valid: int = 10) -> tuple[float, float]:
    """ARI + NMI between HDBSCAN labels and a per-row property.

    For continuous properties: bin into quintiles (duplicates='drop').
    Returns (NaN, NaN) if too few valid rows.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    import pandas as pd

    valid = labels != -1
    if is_continuous:
        vals = np.asarray(prop_values, dtype=float)
        mask = valid & np.isfinite(vals)
        if mask.sum() < min_valid:
            return float("nan"), float("nan")
        try:
            bins = pd.qcut(vals[mask], q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            return float("nan"), float("nan")
        if len(np.unique(bins)) < 2:
            return float("nan"), float("nan")
        return (float(adjusted_rand_score(bins, labels[mask])),
                float(normalized_mutual_info_score(bins, labels[mask])))
    else:
        raw = np.asarray(prop_values, dtype=object)
        mask = valid & np.array([v not in (None, "", "None") for v in raw])
        if mask.sum() < min_valid:
            return float("nan"), float("nan")
        if len(np.unique(raw[mask].astype(str))) < 2:
            return float("nan"), float("nan")
        return (float(adjusted_rand_score(raw[mask], labels[mask])),
                float(normalized_mutual_info_score(raw[mask], labels[mask])))


def plot_ari_nmi_heatmap(ari: dict[str, np.ndarray], nmi: dict[str, np.ndarray],
                          props_in_order: list[str],
                          n_cat_split: int,
                          emb_types: list[str], emb_type_titles: dict[str, str],
                          layer_lbls: list[str], title: str,
                          out_path: Path, caveat: str | None = None) -> None:
    """Per-emb-type ARI/NMI heatmaps. Rows: emb_types × {ARI, NMI}; cols: layers.

    Layout: one column per (emb_type), one row per metric (ARI/NMI). Each cell is
    the props × layers heatmap for that (type, metric).
    """
    n_types = len(emb_types)
    n_lay = len(layer_lbls)
    fig, axes = plt.subplots(2, n_types,
                              figsize=(5.5 * n_types, 0.45 * len(props_in_order) + 2.5),
                              squeeze=False)
    fig.suptitle(title + (f"\n{caveat}" if caveat else ""),
                  fontsize=11, y=1.01)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#d8d8d8")
    for col, et in enumerate(emb_types):
        for row, (name, data) in enumerate([("ARI", ari), ("NMI", nmi)]):
            ax = axes[row][col]
            mat = data[et]  # shape (n_props, n_layers)
            mat_masked = np.ma.masked_invalid(mat)
            im = ax.imshow(mat_masked, aspect="auto", cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks(range(n_lay))
            ax.set_xticklabels(layer_lbls, rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(len(props_in_order)))
            ax.set_yticklabels(props_in_order, fontsize=8)
            ax.set_title(f"{emb_type_titles.get(et, et)}\n{name}", fontsize=10)
            # White line between categorical and continuous
            if 0 < n_cat_split < len(props_in_order):
                ax.axhline(n_cat_split - 0.5, color="white", lw=1.6)
            for r in range(len(props_in_order)):
                for c in range(n_lay):
                    v = mat[r, c]
                    if np.isfinite(v):
                        ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                                 fontsize=6, color="black")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Summary assembly + progress queue
# ---------------------------------------------------------------------------

def write_section(qc_dir: Path, order: int, name: str, text: str) -> Path:
    """Write a numbered QC section text file (overwritten on each run)."""
    path = qc_dir / "sections" / f"{order:02d}_{name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def assemble_summary(qc_dir: Path, header: str) -> Path:
    """Concatenate qc/sections/NN_*.txt files into qc/qc_summary.txt in order."""
    paths = sorted((qc_dir / "sections").glob("*.txt"))
    out = qc_dir / "qc_summary.txt"
    with out.open("w") as fh:
        fh.write(header + "\n")
        for p in paths:
            fh.write("\n")
            fh.write(p.read_text())
    return out


def listen(progress_q, n_workers: int) -> None:
    done = 0
    while done < n_workers:
        try:
            msg = progress_q.get(timeout=600)
        except Exception:
            continue
        if isinstance(msg, str) and msg.startswith("__DONE__"):
            done += 1
            print(f"  Worker {msg[8:]} finished  ({done}/{n_workers})", flush=True)
        else:
            print(msg, flush=True)


# ---------------------------------------------------------------------------
# 8. kNN index builder + Jaccard for cached indices
# ---------------------------------------------------------------------------

def build_knn_indices(X_index: np.ndarray, query_rows: np.ndarray,
                       k_max: int, l2_normalize: bool = False) -> np.ndarray:
    """Build kNN indices for query_rows against X_index using sklearn.

    Returns int32 array [n_query, k_max] of indices into X_index.
    The query row itself (if present in the index) is excluded.
    """
    from sklearn.neighbors import NearestNeighbors
    X = X_index
    if l2_normalize:
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.where(nrm == 0, 1.0, nrm)
    k_use = min(k_max + 1, len(X))
    nn = NearestNeighbors(n_neighbors=k_use, algorithm="auto", n_jobs=-1).fit(X)
    Xq = X[query_rows]
    _, idx = nn.kneighbors(Xq)
    # The query row is always at column 0 (self-distance = 0) when query ⊂ index.
    # Drop column 0 and take k_max.  Verify the assumption defensively.
    qrows = query_rows.astype(idx.dtype)
    if (idx[:, 0] == qrows).all():
        return idx[:, 1:k_max + 1].astype(np.int32)
    # Fallback per-row removal (rare; happens only if query rows are duplicated in X).
    out = np.empty((len(query_rows), k_max), dtype=np.int32)
    for i in range(len(query_rows)):
        row = idx[i]
        kept = row[row != qrows[i]][:k_max]
        if len(kept) < k_max:
            pad = np.full(k_max - len(kept), kept[-1] if len(kept) else 0, dtype=row.dtype)
            kept = np.concatenate([kept, pad])
        out[i] = kept
    return out
