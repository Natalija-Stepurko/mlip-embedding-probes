"""
stages/geometry/organisation_uma.py — PCA, UMAP, k-NN purity, LVR for all FairChem UMA layers

Reads embeddings from $MLIP_PROBES_ROOT/results/fairchem/{layer}/nodes/*.pt
Writes analysis outputs to $MLIP_PROBES_ROOT/results/fairchem/plots/{nodes|global|backbone}/

Embedding types analysed (12 total):
  nodes_flat              — per-atom, flatten [N, 9, 128] → [N, 1152]
  nodes_ang_norm          — per-atom, cat(L0, L1_norm, L2_norm) → [N, 384]
  nodes_l0                — per-atom, L=0 channel only → [N, 128]
  nodes_l1_norm           — per-atom, L=1 channel-wise norm → [N, 128]
  nodes_l2_norm           — per-atom, L=2 channel-wise norm → [N, 128]
  nodes_geom_norm         — per-atom, cat(L1_norm, L2_norm), geometry-only → [N, 256]
  nodes_pooled_full       — per-material, mean-pooled + flatten → [1152]
  nodes_pooled_l0         — per-material, L=0 mean → [128]
  nodes_pooled_ang_norm   — per-material, angular-norm mean → [384]
  nodes_pooled_l1_polar   — per-material, ‖mean_atoms(L1)‖ → [128]
  nodes_pooled_l2_polar   — per-material, ‖mean_atoms(L2)‖ → [128]
  nodes_pooled_geom_norm  — per-material, cat(mean(‖L1‖), mean(‖L2‖)) → [256]

Layers (6): embed, block_0 … block_3, backbone

Parallelism: 12 workers (one per embedding type). Pass --num-workers 1 for sequential.

Usage:
    uv run python stages/geometry/organisation_uma.py

    uv run python stages/geometry/organisation_uma.py \\
        --embeddings-dir $MLIP_PROBES_ROOT/results/fairchem \\
        --plots-dir      $MLIP_PROBES_ROOT/results/fairchem/plots \\
        --structures-dir $MLIP_PROBES_ROOT/structures

    # Local test
    uv run python stages/geometry/organisation_uma.py \\
        --embeddings-dir results_test/fairchem \\
        --plots-dir      results_test/fairchem/plots \\
        --structures-dir structures_test \\
        --umap-sample 2000 --num-workers 1
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import argparse
import io
import json
import multiprocessing as mp
import queue as _queue
import re
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["legend.frameon"] = False  # no box border around figure legends
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from umap import UMAP

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_LAYERS = [
    "embed",
    "block_0", "block_1", "block_2", "block_3",
    "backbone",
]

LAYER_LABELS = {
    "embed":    "embed",
    "block_0":  "blk_0",  "block_1": "blk_1",
    "block_2":  "blk_2",  "block_3": "blk_3",
    "backbone": "backbone",
}

# Per-atom embedding types and their dimensionality
PER_ATOM_KEYS = ["nodes_flat", "nodes_ang_norm", "nodes_l0", "nodes_l1_norm", "nodes_l2_norm",
                  "nodes_geom_norm"]
POOLED_KEYS   = ["nodes_pooled_full", "nodes_pooled_ang_norm", "nodes_pooled_l0",
                  "nodes_pooled_l1_polar", "nodes_pooled_l2_polar", "nodes_pooled_geom_norm"]
ALL_EMB_KEYS  = PER_ATOM_KEYS + POOLED_KEYS

EMB_DIM = {
    "nodes_flat":              1152,
    "nodes_ang_norm":          384,
    "nodes_l0":                128,
    "nodes_l1_norm":           128,
    "nodes_l2_norm":           128,
    "nodes_geom_norm":         256,
    "nodes_pooled_full":       1152,
    "nodes_pooled_l0":         128,
    "nodes_pooled_ang_norm":   384,
    "nodes_pooled_l1_polar":   128,
    "nodes_pooled_l2_polar":   128,
    "nodes_pooled_geom_norm":  256,
}

EMB_LABELS = {
    "nodes_flat":              "Nodes flat",
    "nodes_ang_norm":          "Nodes ang-norm",
    "nodes_l0":                "Nodes L=0",
    "nodes_l1_norm":           "Nodes L=1 norm",
    "nodes_l2_norm":           "Nodes L=2 norm",
    "nodes_geom_norm":         "Nodes geom-norm",
    "nodes_pooled_full":       "Pooled full",
    "nodes_pooled_l0":         "Pooled L=0",
    "nodes_pooled_ang_norm":   "Pooled ang-norm",
    "nodes_pooled_l1_polar":   "Pooled L=1 polar",
    "nodes_pooled_l2_polar":   "Pooled L=2 polar",
    "nodes_pooled_geom_norm":  "Pooled geom-norm",
}

# .pt key for each pooled embedding type
POOLED_PT_KEY = {
    "nodes_pooled_full":     "structure_full_mean",
    "nodes_pooled_l0":       "structure_l0_mean",
    "nodes_pooled_ang_norm": "structure_ang_norm",
}

PCA_VARIANCE_THRESHOLD = 0.90
NODE_UMAP_N_DEFAULT    = 300_000   # per-atom subsample (sqrt-weighted by element)
UMAP_NN_NODE           = 15
MAG_THR                = 1e-4

CRYSTAL_SYSTEMS = [
    "Cubic", "Hexagonal", "Trigonal", "Tetragonal",
    "Orthorhombic", "Monoclinic", "Triclinic",
]

CRYSTAL_COLORS = {
    "Cubic":        "#FF7F0E",
    "Hexagonal":    "dimgray",
    "Trigonal":     "#D62728",
    "Tetragonal":   "#17BECF",
    "Orthorhombic": "#2CA02C",
    "Monoclinic":   "#FFD700",
    "Triclinic":    "#9467BD",
}

KNN_PROPS = {
    "nodes_flat":              ["crystal_system", "metal_insulator", "magnetic_label", "element_identity"],
    "nodes_ang_norm":          ["crystal_system", "metal_insulator", "magnetic_label", "element_identity"],
    "nodes_l0":                ["crystal_system", "metal_insulator", "magnetic_label", "element_identity"],
    "nodes_l1_norm":           ["crystal_system", "metal_insulator", "magnetic_label", "element_identity"],
    "nodes_l2_norm":           ["crystal_system", "metal_insulator", "magnetic_label", "element_identity"],
    "nodes_geom_norm":         ["crystal_system", "metal_insulator", "magnetic_label", "element_identity"],
    "nodes_pooled_full":       ["crystal_system", "metal_insulator", "magnetic_label"],
    "nodes_pooled_l0":         ["crystal_system", "metal_insulator", "magnetic_label"],
    "nodes_pooled_ang_norm":   ["crystal_system", "metal_insulator", "magnetic_label"],
    "nodes_pooled_l1_polar":   ["crystal_system", "metal_insulator", "magnetic_label"],
    "nodes_pooled_l2_polar":   ["crystal_system", "metal_insulator", "magnetic_label"],
    "nodes_pooled_geom_norm":  ["crystal_system", "metal_insulator", "magnetic_label"],
}

LVR_MAT_PROPS = [
    ("formation_energy/atom", "formation_energy_per_atom"),
    ("energy_above_hull",     "energy_above_hull"),
    ("band_gap",              "band_gap"),
    ("density (g/cm3)",       "density"),
    ("bulk_modulus (GPa)",    "bulk_modulus"),
]

LVR_CROSS_PROPS = [
    "formation_energy/atom", "energy_above_hull", "band_gap",
    "mag_density (uB/A3)", "density (g/cm3)", "bulk_modulus (GPa)",
]

KNN_COLOR_MAP = {
    "crystal_system":   "#4E9F8A",
    "metal_insulator":  "#E17C05",
    "magnetic_label":   "#A06CD5",
    "element_identity": "steelblue",
}

LVR_COLORS = [
    "#DD8452", "#C44E52", "#CCB974", "#937860", "#DA8BC3",
    "#8C8C8C", "#64B5CD", "#55A868", "#2196F3",
]

EMB_COLORS = {
    "nodes_flat":             "mediumpurple",
    "nodes_ang_norm":         "darkorange",
    "nodes_l0":               "steelblue",
    "nodes_l1_norm":          "tomato",
    "nodes_l2_norm":          "forestgreen",
    "nodes_geom_norm":        "saddlebrown",
    "nodes_pooled_full":      "mediumorchid",
    "nodes_pooled_l0":        "teal",
    "nodes_pooled_ang_norm":  "goldenrod",
    "nodes_pooled_l1_polar":  "#e377c2",
    "nodes_pooled_l2_polar":  "#7f7f7f",
    "nodes_pooled_geom_norm": "#bcbd22",
}

_MACRO_PROPS: set[str] = set()


# ---------------------------------------------------------------------------
# Per-atom conversion functions (numpy, applied after loading "node" key)
# ---------------------------------------------------------------------------

def _convert_flat(node_raw: np.ndarray) -> np.ndarray:
    """[N, 9, 128] float32 → [N, 1152] float32."""
    return node_raw.reshape(node_raw.shape[0], -1)


def _convert_ang_norm(node_raw: np.ndarray) -> np.ndarray:
    """[N, 9, 128] float32 → [N, 384] float32 — cat(L0, L1_norm, L2_norm)."""
    L0 = node_raw[:, 0, :]                           # [N, 128]
    L1_norm = np.linalg.norm(node_raw[:, 1:4, :], axis=1)   # [N, 128]
    L2_norm = np.linalg.norm(node_raw[:, 4:9, :], axis=1)   # [N, 128]
    return np.concatenate([L0, L1_norm, L2_norm], axis=1)


def _convert_l0(node_raw: np.ndarray) -> np.ndarray:
    """[N, 9, 128] float32 → [N, 128] float32 — L=0 channel only."""
    return node_raw[:, 0, :].copy()


def _convert_l1_norm(node_raw: np.ndarray) -> np.ndarray:
    """[N, 9, 128] float32 → [N, 128] float32 — L=1 channel-wise norm over 3 components."""
    return np.linalg.norm(node_raw[:, 1:4, :], axis=1)


def _convert_l2_norm(node_raw: np.ndarray) -> np.ndarray:
    """[N, 9, 128] float32 → [N, 128] float32 — L=2 channel-wise norm over 5 components."""
    return np.linalg.norm(node_raw[:, 4:9, :], axis=1)


def _convert_geom_norm(node_raw: np.ndarray) -> np.ndarray:
    """[N, 9, 128] float32 → [N, 256] float32 — cat(‖L1‖, ‖L2‖): geometry-only, no element identity."""
    L1_norm = np.linalg.norm(node_raw[:, 1:4, :], axis=1)   # [N, 128]
    L2_norm = np.linalg.norm(node_raw[:, 4:9, :], axis=1)   # [N, 128]
    return np.concatenate([L1_norm, L2_norm], axis=1)        # [N, 256]


CONVERT_FN = {
    "nodes_flat":      _convert_flat,
    "nodes_ang_norm":  _convert_ang_norm,
    "nodes_l0":        _convert_l0,
    "nodes_l1_norm":   _convert_l1_norm,
    "nodes_l2_norm":   _convert_l2_norm,
    "nodes_geom_norm": _convert_geom_norm,
}


# ---------------------------------------------------------------------------
# Edge embedding types — per-edge message [m,9,128] (same equivariant layout as
# nodes) at 5 layers (embed=edge-degree + block_0..3; no backbone). Per-edge and
# pooled (per-material mean over ALL edges) share the node variant family.
# ---------------------------------------------------------------------------

EDGE_LAYERS = ["embed", "block_0", "block_1", "block_2", "block_3"]
_EDGE_VARIANTS = ["flat", "ang_norm", "l0", "l1_norm", "l2_norm", "geom_norm"]
EDGE_ATOM_KEYS   = [f"edges_{v}" for v in _EDGE_VARIANTS]
EDGE_POOLED_KEYS = [f"edges_pooled_{v}" for v in _EDGE_VARIANTS]
ALL_EDGE_KEYS    = EDGE_ATOM_KEYS + EDGE_POOLED_KEYS

_VARIANT_FN  = {"flat": _convert_flat, "ang_norm": _convert_ang_norm, "l0": _convert_l0,
                "l1_norm": _convert_l1_norm, "l2_norm": _convert_l2_norm,
                "geom_norm": _convert_geom_norm}
_VARIANT_DIM = {"flat": 1152, "ang_norm": 384, "l0": 128, "l1_norm": 128,
                "l2_norm": 128, "geom_norm": 256}
_EDGE_COLORS = {"flat": "saddlebrown", "ang_norm": "darkorange", "l0": "steelblue",
                "l1_norm": "tomato", "l2_norm": "forestgreen", "geom_norm": "mediumorchid"}


def _edge_variant(emb_key: str) -> str:
    return emb_key.replace("edges_pooled_", "").replace("edges_", "")


def _is_edge(emb_key: str) -> bool:
    return emb_key.startswith("edges_")


def _layers_for(emb_key: str) -> list[str]:
    return EDGE_LAYERS if _is_edge(emb_key) else ALL_LAYERS


for _k in ALL_EDGE_KEYS:
    _v = _edge_variant(_k)
    CONVERT_FN[_k] = _VARIANT_FN[_v]
    EMB_DIM[_k]    = _VARIANT_DIM[_v]
    EMB_LABELS[_k] = ("Edges pooled " if "pooled" in _k else "Edges ") + _v.replace("_", " ")
    EMB_COLORS[_k] = _EDGE_COLORS[_v]
    KNN_PROPS[_k]  = (["crystal_system", "metal_insulator", "magnetic_label"]
                      if "pooled" in _k
                      else ["bond_type", "crystal_system", "metal_insulator", "magnetic_label"])


# ---------------------------------------------------------------------------
# Helpers — PCA / UMAP
# ---------------------------------------------------------------------------

def pca_n_for_variance(evr: np.ndarray,
                        threshold: float = PCA_VARIANCE_THRESHOLD) -> int:
    return int((np.cumsum(evr) < threshold).sum()) + 1


def fit_pca(X: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    finite_mask = np.isfinite(X).all(axis=1)
    if not finite_mask.all():
        X = X[finite_mask]
    n_components = min(256, X.shape[0], X.shape[1])
    pca_full = PCA(n_components=n_components, random_state=42)
    pca_full.fit(X)
    evr   = pca_full.explained_variance_ratio_
    pca_n = pca_n_for_variance(evr)
    X_pca = PCA(n_components=pca_n, random_state=42).fit_transform(X)
    return evr, pca_n, X_pca


def run_umap(X_pca: np.ndarray, n_neighbors: int) -> np.ndarray:
    n_neighbors = min(n_neighbors, len(X_pca) - 1)
    reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
                   metric="euclidean", init="random", random_state=42,
                   low_memory=True, verbose=False)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="n_jobs value 1 overridden")
        return reducer.fit_transform(X_pca)


# ---------------------------------------------------------------------------
# Helpers — figures
# ---------------------------------------------------------------------------

def quality_figure(layer: str, emb_label: str, evr: np.ndarray, pca_n: int,
                   norms: np.ndarray, out_png: Path,
                   color: str, panel_c_label: str, norm_xlabel: str) -> None:
    # pca_n_for_variance() returns len(evr)+1 when the variance threshold is
    # never reached within the computed components (e.g. 1152-D edges_flat at
    # deep layers, where only 256 PCs are fit). Clamp so the scree bars line up
    # with evr and matplotlib does not raise a shape-mismatch on bar().
    pca_n = min(pca_n, len(evr))
    cumvar = np.cumsum(evr) * 100
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"FairChem UMA-S  {emb_label}  [{LAYER_LABELS[layer]}]  — PCA + Norm Quality",
                 fontsize=13, y=1.01)

    axes[0].bar(range(1, pca_n + 1), evr[:pca_n] * 100, color=color, width=0.9)
    axes[0].set_title(f"A: PCA scree  ({pca_n} PCs)")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_xlim(0.5, pca_n + 0.5)

    axes[1].plot(range(1, len(evr) + 1), cumvar, color=color, lw=1.5)
    axes[1].axvline(pca_n, color="tomato", ls="--", lw=1.5,
                    label=f"{pca_n} PCs ({evr[:pca_n].sum():.1%})")
    for thresh, col in [(90, "orange"), (95, "green")]:
        n_pcs = int((np.cumsum(evr) < thresh / 100).sum()) + 1
        axes[1].axhline(thresh, color=col, ls=":", lw=1, label=f"{thresh}% @ PC {n_pcs}")
    axes[1].legend(fontsize=8)
    axes[1].set_title("B: cumulative variance")
    axes[1].set_xlabel("Number of PCs")
    axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_xlim(1, len(evr))
    axes[1].set_ylim(0, 101)

    n_out = int((norms > norms.mean() + 3 * norms.std()).sum())
    axes[2].hist(norms, bins=80, color=color, edgecolor="none", alpha=0.8)
    axes[2].axvline(norms.mean(), color="tomato", ls="--", lw=1.5,
                    label=f"Mean = {norms.mean():.2f}")
    axes[2].axvline(norms.mean() + 3 * norms.std(), color="orange", ls=":", lw=1.5,
                    label=f"Mean+3σ = {norms.mean() + 3 * norms.std():.2f}")
    axes[2].set_title(f"C: norm distribution  ({n_out} > mean+3σ)")
    axes[2].set_xlabel(norm_xlabel)
    axes[2].set_ylabel(panel_c_label)
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()


def pca_quality_grid(nodes_dir: Path, emb_keys: list[str], group_name: str,
                     out_path: Path, layers: list[str] | None = None) -> None:
    """Combined PCA node-quality view: cumulative explained variance for every
    (layer, embedding type) on one figure — columns = layers (model order),
    rows = embedding types. Reads only the cached `evr` arrays (no recompute).
    `layers` defaults to the node layer stack (ALL_LAYERS); pass EDGE_LAYERS for
    edge embeddings (which have no backbone layer)."""
    if out_path.exists():
        return
    layers = ALL_LAYERS if layers is None else layers
    nr, nc = len(emb_keys), len(layers)
    fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.1 * nr),
                             squeeze=False, sharey=True)
    fig.suptitle("FairChem UMA-S — PCA node quality (cumulative explained variance) — "
                 f"{group_name}\ncolumns = layers, rows = embedding types",
                 fontsize=13, y=0.998)
    for r, ek in enumerate(emb_keys):
        color = EMB_COLORS.get(ek, "steelblue")
        for c, layer in enumerate(layers):
            ax = axes[r][c]
            p = nodes_dir / f"{layer}_{ek}_umap_cache.npz"
            if not p.exists():
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="#999")
                ax.set_xticks([]); ax.set_yticks([]); continue
            evr = np.load(p, allow_pickle=True)["evr"]
            cumvar = np.cumsum(evr) * 100
            pca_n = pca_n_for_variance(evr)
            ax.plot(range(1, len(evr) + 1), cumvar, color=color, lw=1.4)
            ax.axvline(pca_n, color="tomato", ls="--", lw=1.0)
            for thr, col in [(90, "orange"), (95, "green")]:
                ax.axhline(thr, color=col, ls=":", lw=0.8)
            ax.set_xlim(1, len(evr)); ax.set_ylim(0, 101)
            ax.text(0.96, 0.06, f"{pca_n} PCs", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6, color="tomato")
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(LAYER_LABELS[layer], fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{EMB_LABELS[ek]}\ncum. var (%)", fontsize=7)
            if r == nr - 1:
                ax.set_xlabel("# PCs", fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def scatter_continuous(ax, X_umap: np.ndarray, values, cmap: str = "plasma",
                        s: int = 1, alpha: float = 0.1, label: str = "") -> None:
    vals = np.asarray([v if v is not None else np.nan for v in values], dtype=float)
    nan_mask = np.isnan(vals)
    if nan_mask.any():
        ax.scatter(X_umap[nan_mask, 0], X_umap[nan_mask, 1],
                   c="#dddddd", s=s, alpha=0.1, linewidths=0, rasterized=True)
    ok = ~nan_mask
    if ok.any():
        sc = ax.scatter(X_umap[ok, 0], X_umap[ok, 1],
                        c=vals[ok], cmap=cmap, s=s, alpha=alpha,
                        linewidths=0, rasterized=True)
        plt.colorbar(sc, ax=ax, label=label)
    ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------------------
# Helpers — k-NN purity + LVR
# ---------------------------------------------------------------------------

def _interp_knn(v: float) -> str:
    return "strong" if v > .5 else "moderate" if v > .25 else "weak" if v > .1 else "none"


def _interp_lvr(v: float) -> str:
    return "tight" if v < .4 else "moderate" if v < .65 else "weak" if v < .85 else "none"


def knn_purity_line(name: str, labels, nn_idx: np.ndarray, X: np.ndarray,
                     build_nn_fn) -> str:
    labels = np.asarray(labels, dtype=object)
    mask   = np.array([l is not None for l in labels])
    valid  = labels[mask].astype(str)
    idx    = nn_idx if mask.all() else build_nn_fn(X[mask])
    counts = Counter(valid)
    n, n_cls = len(valid), len(counts)
    baseline = sum((c / n) ** 2 for c in counts.values())
    raw_mi   = float(np.mean([np.mean(valid[i2] == valid[i]) for i, i2 in enumerate(idx)]))
    norm_mi  = (raw_mi - baseline) / (1 - baseline) if baseline < 1 else 0.0
    class_norms = []
    for cls, cnt in counts.items():
        rows   = np.where(valid == cls)[0]
        base_c = cnt / n
        raw_c  = float(np.mean(valid[idx[rows]] == cls))
        class_norms.append((raw_c - base_c) / (1 - base_c) if base_c < 1 else 0.0)
    norm_ma = float(np.mean(class_norms))
    lift    = raw_mi / baseline if baseline > 0 else float("nan")
    cov     = float(mask[nn_idx].any(axis=1).mean())
    return (
        f"  {name:<25}  n={n:<7} classes={n_cls:<4}"
        f"  raw={raw_mi:.3f}  base={baseline:.3f}"
        f"  norm={norm_mi:.3f} ({_interp_knn(norm_mi)})"
        f"  lift={lift:.2f}x"
        f"  macro_norm={norm_ma:.3f} ({_interp_knn(norm_ma)})"
        f"  cov={cov:.3f}\n"
    )


def lvr_material(name: str, values_by_mp: dict, mp_ids: np.ndarray,
                  nn_idx: np.ndarray, X: np.ndarray, K: int, build_nn_fn) -> str:
    v    = np.array([values_by_mp.get(m, np.nan) for m in mp_ids], dtype=float)
    mask = ~np.isnan(v)
    n_ok = int(mask.sum())
    if n_ok < K + 1:
        return f"  {name:<25}  n={n_ok} -- skipped (too few points)\n"
    idx = nn_idx if mask.all() else build_nn_fn(X[mask])
    vv  = v[mask]
    g   = vv.std()
    if g == 0:
        return f"  {name:<25}  zero variance -- skipped\n"
    cov     = float(mask[nn_idx].any(axis=1).mean())
    lvr_val = float(np.array([vv[i2].std() for i2 in idx]).mean() / g)
    return f"  {name:<25}  n={n_ok:<7} LVR={lvr_val:.3f}  cov={cov:.3f}  -> {_interp_lvr(lvr_val)}\n"


def lvr_array(name: str, values, nn_idx: np.ndarray, X: np.ndarray,
              K: int, build_nn_fn) -> str:
    """LVR for a per-row continuous value array (e.g. bond_distance, per edge)."""
    v    = np.asarray(values, dtype=float)
    mask = np.isfinite(v)
    n_ok = int(mask.sum())
    if n_ok < K + 1:
        return f"  {name:<25}  n={n_ok} -- skipped (too few points)\n"
    idx = nn_idx if mask.all() else build_nn_fn(X[mask])
    vv  = v[mask]
    g   = vv.std()
    if g == 0:
        return f"  {name:<25}  zero variance -- skipped\n"
    cov     = float(mask[nn_idx].any(axis=1).mean())
    lvr_val = float(np.array([vv[i2].std() for i2 in idx]).mean() / g)
    return f"  {name:<25}  n={n_ok:<7} LVR={lvr_val:.3f}  cov={cov:.3f}  -> {_interp_lvr(lvr_val)}\n"


# ---------------------------------------------------------------------------
# Parse k-NN / LVR txt files
# ---------------------------------------------------------------------------

def parse_knn_lvr_txt(path: Path) -> tuple[dict, dict, dict]:
    knn, lvr, cov = {}, {}, {}
    if not path.exists():
        return knn, lvr, cov
    text    = path.read_text()
    section = None
    prop    = None
    for line in text.splitlines():
        if "Categorical" in line:
            section = "knn"
        elif "Continuous" in line:
            section = "lvr"
        elif section == "knn":
            prop = line.strip().split()[0] if line.strip() else None
            if not prop:
                continue
            mi = re.search(r"micro=([-\d.]+)", line)
            ma = re.search(r"macro=([-\d.]+)", line)
            if mi and ma:
                knn[prop] = float(ma.group(1)) if prop in _MACRO_PROPS else float(mi.group(1))
            else:
                m = re.search(r"norm=([-\d.]+)", line)
                if m:
                    knn[prop] = float(m.group(1))
            mc = re.search(r"cov=([\d.]+)", line)
            if mc and prop:
                cov[prop] = float(mc.group(1))
        elif section == "lvr":
            m = re.search(r"LVR=([\d.]+)", line)
            if m:
                prop_m = re.match(r"\s+(\S.*?)\s{2,}", line)
                if prop_m:
                    pname = prop_m.group(1).strip()
                    lvr[pname] = float(m.group(1))
                    mc = re.search(r"cov=([\d.]+)", line)
                    if mc:
                        cov[pname] = float(mc.group(1))
    return knn, lvr, cov


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _load_manifest(jsonl_path: Path) -> tuple[dict, set]:
    structs: dict[str, dict] = {}
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                structs[rec["mp_id"]] = rec
    return structs, set(structs.keys())


def _mag_label(structs: dict, m: str) -> str | None:
    tm  = structs[m].get("total_magnetization")
    vol = structs[m].get("volume")
    if tm is None:
        return None
    if vol and abs(tm / vol) >= MAG_THR:
        return "Magnetic"
    return "Non-magnetic"


def _metal_insulator(structs: dict, m: str) -> str | None:
    bg = structs[m].get("band_gap")
    if bg is None:
        return None
    return "Metal" if bg == 0 else "Insulator"


def _mag_density_dict(structs: dict) -> dict:
    return {
        m: s["total_magnetization"] / s["volume"]
        if s.get("total_magnetization") is not None and s.get("volume")
        else None
        for m, s in structs.items()
    }


def _prop_dict(structs: dict, key: str) -> dict:
    return {m: s.get(key) for m, s in structs.items()}


# ---------------------------------------------------------------------------
# Generic worker — per-atom embeddings
# ---------------------------------------------------------------------------

def _worker_per_atom(worker_id: int, emb_key: str,
                      emb_dir: Path, nodes_dir: Path,
                      jsonl_path: Path, node_umap_n: int,
                      progress_q) -> None:
    matplotlib.use("Agg")
    convert_fn = CONVERT_FN[emb_key]

    def log(msg: str) -> None:
        progress_q.put(f"[{emb_key:<18}] {msg}")

    structs, valid_ids = _load_manifest(jsonl_path)

    K_node: int | None = None
    binding_node: str  = ""

    def build_nn(X: np.ndarray) -> np.ndarray:
        assert K_node is not None
        k = min(K_node, len(X) - 1)
        return (NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=1)
                .fit(X).kneighbors(X, return_distance=False)[:, 1:])

    for layer in ALL_LAYERS:
        t0 = time.time()
        node_dir_l = emb_dir / layer / "nodes"
        log(f"--- {layer} ---")

        cache_path   = nodes_dir / f"{layer}_{emb_key}_umap_cache.npz"
        quality_path = nodes_dir / f"{layer}_{emb_key}_pca_quality.png"

        if not cache_path.exists():
            pt_files = sorted(node_dir_l.glob("*.pt"))

            # Pass 1: collect per-atom metadata (no embedding tensor kept in RAM)
            log(f"  Pass 1: collecting node metadata ...")
            meta_elements:   list[str] = []
            meta_mids:       list[str] = []
            meta_cs:         list[str] = []
            mat_atom_counts: dict[str, int] = {}
            for pt in pt_files:
                if pt.stem not in valid_ids:
                    continue
                d     = torch.load(pt, weights_only=False)
                elems = d["elements"]
                cs    = structs[pt.stem].get("crystal_system", "Unknown")
                n     = len(elems)
                meta_elements.extend(elems)
                meta_mids.extend([pt.stem] * n)
                meta_cs.extend([cs] * n)
                mat_atom_counts[pt.stem] = n

            meta_elements_arr = np.array(meta_elements)
            meta_mids_arr     = np.array(meta_mids)
            meta_cs_arr       = np.array(meta_cs)
            total_atoms       = len(meta_elements_arr)
            log(f"  Total atoms: {total_atoms:,}")

            # Sqrt-weighted subsample by element
            elem_counts = Counter(meta_elements)
            sqrt_total  = sum(np.sqrt(c) for c in elem_counts.values())
            rng_sub     = np.random.default_rng(42)
            umap_idx: list[int] = []
            for elem, count in elem_counts.items():
                n = int(node_umap_n * np.sqrt(count) / sqrt_total)
                n = max(n, 20); n = min(n, count)
                elem_global = np.where(meta_elements_arr == elem)[0]
                umap_idx.extend(rng_sub.choice(elem_global, size=n, replace=False).tolist())
            umap_idx_arr = np.array(umap_idx)
            rng_sub.shuffle(umap_idx_arr)
            log(f"  Subsample: {len(umap_idx_arr):,} atoms "
                f"({len(umap_idx_arr)/total_atoms:.1%} of total, "
                f"{len(elem_counts)} elements)")

            # Build offset map: global atom index → (material, local index)
            offset = 0
            mat_offsets: dict[str, int] = {}
            for pt in pt_files:
                if pt.stem not in valid_ids:
                    continue
                mat_offsets[pt.stem] = offset
                offset += mat_atom_counts[pt.stem]

            idx_by_mat: dict[str, list[int]] = defaultdict(list)
            for gi in umap_idx_arr.tolist():
                mat = meta_mids_arr[gi]
                idx_by_mat[mat].append(gi - mat_offsets[mat])

            # Pass 2: load embeddings for selected atoms and convert
            log(f"  Pass 2: loading + converting subsample ...")
            dim   = EMB_DIM[emb_key]
            gi_to_pos = {int(gi): pos for pos, gi in enumerate(umap_idx_arr)}
            X_n   = np.zeros((len(umap_idx_arr), dim), dtype=np.float32)
            for pt in pt_files:
                if pt.stem not in idx_by_mat:
                    continue
                d       = torch.load(pt, weights_only=False)
                raw     = d["node"].float().numpy()   # [N, 9, 128]
                conv    = convert_fn(raw)              # [N, dim]
                base    = mat_offsets[pt.stem]
                for local_idx in idx_by_mat[pt.stem]:
                    X_n[gi_to_pos[base + local_idx]] = conv[local_idx]

            labels_n   = meta_cs_arr[umap_idx_arr]
            mp_ids_n   = meta_mids_arr[umap_idx_arr]
            elements_n = meta_elements_arr[umap_idx_arr]
            norms_n    = np.linalg.norm(X_n, axis=1)

            log(f"  Fitting PCA ...")
            evr_n, pca_n_n, X_pca_n = fit_pca(X_n)
            log(f"  PCA: {pca_n_n} components ({evr_n[:pca_n_n].sum():.1%})")

            log(f"  Running UMAP ({len(X_pca_n):,} atoms) ...")
            X_umap_n = run_umap(X_pca_n, UMAP_NN_NODE)

            np.savez(cache_path,
                     X_umap=X_umap_n, X_pca=X_pca_n, X_raw=X_n,
                     labels=labels_n, mp_ids=mp_ids_n, elements=elements_n,
                     evr=evr_n, n_total=np.array([total_atoms]))
            log(f"  Saved cache -> {cache_path}")
        else:
            log(f"  Loading cache ...")
            c           = np.load(cache_path, allow_pickle=True)
            X_n         = c["X_raw"]
            X_umap_n    = c["X_umap"]
            labels_n    = c["labels"]
            mp_ids_n    = c["mp_ids"]
            elements_n  = c["elements"]
            evr_n       = c["evr"]
            pca_n_n     = pca_n_for_variance(evr_n)
            norms_n     = np.linalg.norm(X_n, axis=1)
            total_atoms = int(c["n_total"][0]) if "n_total" in c else len(X_n)

        # Calibrate K from backbone subsample
        if layer == "backbone" and K_node is None:
            apm_sub  = np.array(list(Counter(mp_ids_n.tolist()).values()), dtype=float)
            p75_sub  = float(np.percentile(apm_sub, 75))
            k_p75_n  = int(p75_sub) + 1
            k_cap_n  = len(X_n) // 10
            K_node   = max(10, min(k_p75_n, k_cap_n))
            binding_node = "p75+1 (subsample)" if k_p75_n <= k_cap_n else "10% cap (subsample)"
            log(f"  K={K_node}  ({binding_node})")

        if not quality_path.exists():
            quality_figure(layer, EMB_LABELS[emb_key], evr_n, pca_n_n, norms_n,
                           quality_path, EMB_COLORS[emb_key],
                           "Atom count", f"L2 norm of {EMB_DIM[emb_key]}-D {emb_key}")
            log(f"  Saved -> {quality_path}")

        knn_path = nodes_dir / f"{layer}_{emb_key}_knn_lvr.txt"
        if not knn_path.exists() and K_node is not None:
            log(f"  Computing k-NN purity + LVR ...")
            norms_l2 = np.linalg.norm(X_n, axis=1, keepdims=True)
            X_l2     = X_n / np.where(norms_l2 == 0, 1, norms_l2)
            nn_n     = build_nn(X_l2)
            buf = io.StringIO()
            buf.write(f"=== {emb_key} -- {layer} ===\n")
            buf.write(f"Atoms (subsample): {len(X_n):,} of {total_atoms:,} total "
                      f"({len(X_n)/total_atoms:.1%})  "
                      f"Materials: {len(set(mp_ids_n.tolist())):,}  "
                      f"Elements: {len(set(elements_n.tolist()))}\n")
            buf.write(f"K={K_node}  ({binding_node})\n\n")
            buf.write("-- Categorical (k-NN purity: micro / macro) --\n")
            buf.write(knn_purity_line("crystal_system", labels_n, nn_n, X_l2, build_nn))
            buf.write(knn_purity_line("metal_insulator",
                [_metal_insulator(structs, m) for m in mp_ids_n], nn_n, X_l2, build_nn))
            buf.write(knn_purity_line("magnetic_label",
                [_mag_label(structs, m) for m in mp_ids_n], nn_n, X_l2, build_nn))
            buf.write(knn_purity_line("element_identity", elements_n, nn_n, X_l2, build_nn))
            buf.write("\n-- Continuous (LVR) --\n")
            md = _mag_density_dict(structs)
            for name, key in LVR_MAT_PROPS:
                buf.write(lvr_material(name, _prop_dict(structs, key), mp_ids_n,
                                       nn_n, X_l2, K_node, build_nn))
            buf.write(lvr_material("mag_density (uB/A3)", md, mp_ids_n,
                                   nn_n, X_l2, K_node, build_nn))
            knn_path.write_text(buf.getvalue())
            log(f"  Saved -> {knn_path}")

        log(f"  [{layer}] done in {time.time() - t0:.0f}s")

    # Deferred k-NN for layers processed before backbone
    if K_node is None:
        c_bb = np.load(nodes_dir / f"backbone_{emb_key}_umap_cache.npz", allow_pickle=True)
        apm_sub  = np.array(list(Counter(c_bb["mp_ids"].tolist()).values()), dtype=float)
        p75_sub  = float(np.percentile(apm_sub, 75))
        k_p75_n  = int(p75_sub) + 1
        k_cap_n  = len(c_bb["X_raw"]) // 10
        K_node   = max(10, min(k_p75_n, k_cap_n))
        binding_node = "p75+1 (subsample)" if k_p75_n <= k_cap_n else "10% cap"
        log(f"\nK_node (from backbone cache): {K_node}")

    for layer in ALL_LAYERS:
        knn_path = nodes_dir / f"{layer}_{emb_key}_knn_lvr.txt"
        if knn_path.exists():
            continue
        log(f"\n  [{layer}] k-NN + LVR (deferred) ...")
        c_n        = np.load(nodes_dir / f"{layer}_{emb_key}_umap_cache.npz", allow_pickle=True)
        X_n        = c_n["X_raw"]
        labels_n   = c_n["labels"]
        mp_ids_n   = c_n["mp_ids"]
        elements_n = c_n["elements"]
        total_atoms = int(c_n["n_total"][0]) if "n_total" in c_n else len(X_n)
        norms_l2   = np.linalg.norm(X_n, axis=1, keepdims=True)
        X_l2       = X_n / np.where(norms_l2 == 0, 1, norms_l2)
        nn_n       = build_nn(X_l2)
        md         = _mag_density_dict(structs)
        buf = io.StringIO()
        buf.write(f"=== {emb_key} -- {layer} ===\n")
        buf.write(f"Atoms (subsample): {len(X_n):,} of {total_atoms:,} total  "
                  f"Materials: {len(set(mp_ids_n.tolist())):,}\n")
        buf.write(f"K={K_node}  ({binding_node})\n\n")
        buf.write("-- Categorical (k-NN purity: micro / macro) --\n")
        buf.write(knn_purity_line("crystal_system", labels_n, nn_n, X_l2, build_nn))
        buf.write(knn_purity_line("metal_insulator",
            [_metal_insulator(structs, m) for m in mp_ids_n], nn_n, X_l2, build_nn))
        buf.write(knn_purity_line("magnetic_label",
            [_mag_label(structs, m) for m in mp_ids_n], nn_n, X_l2, build_nn))
        buf.write(knn_purity_line("element_identity", elements_n, nn_n, X_l2, build_nn))
        buf.write("\n-- Continuous (LVR) --\n")
        for name, key in LVR_MAT_PROPS:
            buf.write(lvr_material(name, _prop_dict(structs, key), mp_ids_n,
                                   nn_n, X_l2, K_node, build_nn))
        buf.write(lvr_material("mag_density (uB/A3)", md, mp_ids_n,
                               nn_n, X_l2, K_node, build_nn))
        knn_path.write_text(buf.getvalue())
        log(f"  Saved -> {knn_path}")

    progress_q.put(f"__DONE__{worker_id}")


# ---------------------------------------------------------------------------
# Generic worker — pooled embeddings
# ---------------------------------------------------------------------------

def _worker_pooled(worker_id: int, emb_key: str,
                    emb_dir: Path, nodes_dir: Path,
                    jsonl_path: Path, K_pool: int, binding_mat: str,
                    progress_q) -> None:
    matplotlib.use("Agg")
    pt_key = POOLED_PT_KEY[emb_key]

    def log(msg: str) -> None:
        progress_q.put(f"[{emb_key:<18}] {msg}")

    structs, valid_ids = _load_manifest(jsonl_path)

    def build_nn(X: np.ndarray) -> np.ndarray:
        k = min(K_pool, len(X) - 1)
        return (NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=1)
                .fit(X).kneighbors(X, return_distance=False)[:, 1:])

    for layer in ALL_LAYERS:
        t0 = time.time()
        node_dir_l = emb_dir / layer / "nodes"
        log(f"--- {layer} ---")

        cache_path   = nodes_dir / f"{layer}_{emb_key}_umap_cache.npz"
        quality_path = nodes_dir / f"{layer}_{emb_key}_pca_quality.png"

        if not cache_path.exists():
            log(f"  Loading {emb_key} embeddings ...")
            pt_files = sorted(node_dir_l.glob("*.pt"))
            pool_embs, mp_ids_p = [], []
            for pt in pt_files:
                if pt.stem not in valid_ids:
                    continue
                d = torch.load(pt, weights_only=False)
                pool_embs.append(d[pt_key].float().numpy())
                mp_ids_p.append(pt.stem)
            X_p      = np.stack(pool_embs, axis=0)
            mp_ids_p = np.array(mp_ids_p)
            fin_mask = np.isfinite(X_p).all(axis=1)
            if not fin_mask.all():
                log(f"  WARNING: dropping {(~fin_mask).sum()} materials with NaN/inf embeddings")
                X_p      = X_p[fin_mask]
                mp_ids_p = mp_ids_p[fin_mask]
            norms_p = np.linalg.norm(X_p, axis=1)
            log(f"  {len(X_p):,} materials")

            evr_p, pca_n_p, X_pca_p = fit_pca(X_p)
            log(f"  PCA: {pca_n_p} components ({evr_p[:pca_n_p].sum():.1%})")
            log(f"  Running UMAP ({len(X_pca_p):,} materials) ...")
            X_umap_p = run_umap(X_pca_p, UMAP_NN_NODE)

            np.savez(cache_path,
                     X_umap=X_umap_p, X_pca=X_pca_p, X_raw=X_p,
                     mp_ids=mp_ids_p, evr=evr_p)
            log(f"  Saved cache -> {cache_path}")
        else:
            c         = np.load(cache_path, allow_pickle=True)
            X_p       = c["X_raw"]
            X_umap_p  = c["X_umap"]
            mp_ids_p  = c["mp_ids"]
            evr_p     = c["evr"]
            pca_n_p   = pca_n_for_variance(evr_p)
            norms_p   = np.linalg.norm(X_p, axis=1)

        if not quality_path.exists():
            quality_figure(layer, EMB_LABELS[emb_key], evr_p, pca_n_p, norms_p,
                           quality_path, EMB_COLORS[emb_key],
                           "Material count", f"L2 norm of {EMB_DIM[emb_key]}-D {emb_key}")
            log(f"  Saved -> {quality_path}")

        knn_path = nodes_dir / f"{layer}_{emb_key}_knn_lvr.txt"
        if not knn_path.exists():
            log(f"  Computing k-NN purity + LVR ...")
            nn_p = build_nn(X_p)
            buf = io.StringIO()
            buf.write(f"=== {emb_key} -- {layer} ===\n")
            buf.write(f"Materials: {len(X_p):,}\nK={K_pool}  ({binding_mat})\n\n")
            buf.write("-- Categorical (k-NN purity: micro / macro) --\n")
            buf.write(knn_purity_line("crystal_system",
                [structs[m].get("crystal_system") for m in mp_ids_p], nn_p, X_p, build_nn))
            buf.write(knn_purity_line("metal_insulator",
                [_metal_insulator(structs, m) for m in mp_ids_p], nn_p, X_p, build_nn))
            buf.write(knn_purity_line("magnetic_label",
                [_mag_label(structs, m) for m in mp_ids_p], nn_p, X_p, build_nn))
            buf.write("\n-- Continuous (LVR) --\n")
            md = _mag_density_dict(structs)
            for name, key in LVR_MAT_PROPS:
                buf.write(lvr_material(name, _prop_dict(structs, key), mp_ids_p,
                                       nn_p, X_p, K_pool, build_nn))
            buf.write(lvr_material("mag_density (uB/A3)", md, mp_ids_p,
                                   nn_p, X_p, K_pool, build_nn))
            knn_path.write_text(buf.getvalue())
            log(f"  Saved -> {knn_path}")

        log(f"  [{layer}] done in {time.time() - t0:.0f}s")

    progress_q.put(f"__DONE__{worker_id}")


# ---------------------------------------------------------------------------
# Generic worker — pooled embeddings computed at runtime from node tensor
# ---------------------------------------------------------------------------

def _pool_l1_polar(node_raw: np.ndarray) -> np.ndarray:
    """[N, 9, 128] → [128] — ‖mean_atoms(L1)‖: net vector polarization."""
    return np.linalg.norm(node_raw[:, 1:4, :].mean(axis=0), axis=0)


def _pool_l2_polar(node_raw: np.ndarray) -> np.ndarray:
    """[N, 9, 128] → [128] — ‖mean_atoms(L2)‖: net quadrupolar polarization."""
    return np.linalg.norm(node_raw[:, 4:9, :].mean(axis=0), axis=0)


def _pool_geom_norm(node_raw: np.ndarray) -> np.ndarray:
    """[N, 9, 128] → [256] — cat(mean_atoms(‖L1‖), mean_atoms(‖L2‖)): geometry-only pooling."""
    l1_norm_mean = np.linalg.norm(node_raw[:, 1:4, :], axis=1).mean(axis=0)   # [128]
    l2_norm_mean = np.linalg.norm(node_raw[:, 4:9, :], axis=1).mean(axis=0)   # [128]
    return np.concatenate([l1_norm_mean, l2_norm_mean])                         # [256]


_POLAR_COMPUTE_FN = {
    "nodes_pooled_l1_polar":  _pool_l1_polar,
    "nodes_pooled_l2_polar":  _pool_l2_polar,
    "nodes_pooled_geom_norm": _pool_geom_norm,
}


def _worker_pooled_computed(worker_id: int, emb_key: str,
                             emb_dir: Path, nodes_dir: Path,
                             jsonl_path: Path, K_pool: int, binding_mat: str,
                             progress_q) -> None:
    """Pooled worker that computes the material vector from node [N,9,128] at runtime."""
    matplotlib.use("Agg")
    compute_fn = _POLAR_COMPUTE_FN[emb_key]

    def log(msg: str) -> None:
        progress_q.put(f"[{emb_key:<18}] {msg}")

    structs, valid_ids = _load_manifest(jsonl_path)

    def build_nn(X: np.ndarray) -> np.ndarray:
        k = min(K_pool, len(X) - 1)
        return (NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=1)
                .fit(X).kneighbors(X, return_distance=False)[:, 1:])

    for layer in ALL_LAYERS:
        t0 = time.time()
        node_dir_l = emb_dir / layer / "nodes"
        log(f"--- {layer} ---")

        cache_path   = nodes_dir / f"{layer}_{emb_key}_umap_cache.npz"
        quality_path = nodes_dir / f"{layer}_{emb_key}_pca_quality.png"

        if not cache_path.exists():
            log(f"  Computing {emb_key} from node tensors ...")
            pt_files = sorted(node_dir_l.glob("*.pt"))
            pool_embs, mp_ids_p = [], []
            for pt in pt_files:
                if pt.stem not in valid_ids:
                    continue
                d        = torch.load(pt, weights_only=False)
                node_raw = d["node"].float().numpy()   # [N, 9, 128]
                pool_embs.append(compute_fn(node_raw))
                mp_ids_p.append(pt.stem)
            X_p      = np.stack(pool_embs, axis=0)
            mp_ids_p = np.array(mp_ids_p)
            fin_mask = np.isfinite(X_p).all(axis=1)
            if not fin_mask.all():
                log(f"  WARNING: dropping {(~fin_mask).sum()} NaN/inf materials")
                X_p      = X_p[fin_mask]
                mp_ids_p = mp_ids_p[fin_mask]
            norms_p = np.linalg.norm(X_p, axis=1)
            log(f"  {len(X_p):,} materials")

            evr_p, pca_n_p, X_pca_p = fit_pca(X_p)
            log(f"  PCA: {pca_n_p} components ({evr_p[:pca_n_p].sum():.1%})")
            log(f"  Running UMAP ({len(X_pca_p):,} materials) ...")
            X_umap_p = run_umap(X_pca_p, UMAP_NN_NODE)

            np.savez(cache_path,
                     X_umap=X_umap_p, X_pca=X_pca_p, X_raw=X_p,
                     mp_ids=mp_ids_p, evr=evr_p)
            log(f"  Saved cache -> {cache_path}")
        else:
            c         = np.load(cache_path, allow_pickle=True)
            X_p       = c["X_raw"]
            X_umap_p  = c["X_umap"]
            mp_ids_p  = c["mp_ids"]
            evr_p     = c["evr"]
            pca_n_p   = pca_n_for_variance(evr_p)
            norms_p   = np.linalg.norm(X_p, axis=1)

        if not quality_path.exists():
            quality_figure(layer, EMB_LABELS[emb_key], evr_p, pca_n_p, norms_p,
                           quality_path, EMB_COLORS[emb_key],
                           "Material count", f"L2 norm of {EMB_DIM[emb_key]}-D {emb_key}")
            log(f"  Saved -> {quality_path}")

        knn_path = nodes_dir / f"{layer}_{emb_key}_knn_lvr.txt"
        if not knn_path.exists():
            log(f"  Computing k-NN purity + LVR ...")
            nn_p = build_nn(X_p)
            buf = io.StringIO()
            buf.write(f"=== {emb_key} -- {layer} ===\n")
            buf.write(f"Materials: {len(X_p):,}\nK={K_pool}  ({binding_mat})\n\n")
            buf.write("-- Categorical (k-NN purity: micro / macro) --\n")
            buf.write(knn_purity_line("crystal_system",
                [structs[m].get("crystal_system") for m in mp_ids_p], nn_p, X_p, build_nn))
            buf.write(knn_purity_line("metal_insulator",
                [_metal_insulator(structs, m) for m in mp_ids_p], nn_p, X_p, build_nn))
            buf.write(knn_purity_line("magnetic_label",
                [_mag_label(structs, m) for m in mp_ids_p], nn_p, X_p, build_nn))
            buf.write("\n-- Continuous (LVR) --\n")
            md = _mag_density_dict(structs)
            for name, key in LVR_MAT_PROPS:
                buf.write(lvr_material(name, _prop_dict(structs, key), mp_ids_p,
                                       nn_p, X_p, K_pool, build_nn))
            buf.write(lvr_material("mag_density (uB/A3)", md, mp_ids_p,
                                   nn_p, X_p, K_pool, build_nn))
            knn_path.write_text(buf.getvalue())
            log(f"  Saved -> {knn_path}")

        log(f"  [{layer}] done in {time.time() - t0:.0f}s")

    progress_q.put(f"__DONE__{worker_id}")


# ---------------------------------------------------------------------------
# Edge workers (per-edge message [m,9,128] and per-material pooled mean [9,128])
# ---------------------------------------------------------------------------

def _worker_edges(worker_id, emb_key, emb_dir, edges_dir, jsonl_path,
                  edge_umap_n, progress_q):
    """Per-edge analysis: subsample (sqrt-weighted by bond_type), convert, L2-norm,
    PCA/UMAP, k-NN purity (+bond_type) and LVR (+bond_distance)."""
    matplotlib.use("Agg")
    convert_fn = CONVERT_FN[emb_key]
    dim = EMB_DIM[emb_key]

    def log(msg):
        progress_q.put(f"[{emb_key:<22}] {msg}")

    structs, valid_ids = _load_manifest(jsonl_path)
    K_edge = None
    binding = "sqrt(n_edges)"

    def build_nn(X):
        k = min(K_edge, len(X) - 1)
        return (NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=1)
                .fit(X).kneighbors(X, return_distance=False)[:, 1:])

    for layer in EDGE_LAYERS:
        t0 = time.time()
        edir = emb_dir / layer / "edges"
        log(f"--- {layer} ---")
        cache_path = edges_dir / f"{layer}_{emb_key}_umap_cache.npz"
        quality_path = edges_dir / f"{layer}_{emb_key}_pca_quality.png"

        if not cache_path.exists():
            pts = sorted(edir.glob("*.pt"))
            meta_bt, meta_mid, meta_dist, meta_cs = [], [], [], []
            mat_counts = {}
            for pt in pts:
                if pt.stem not in valid_ids:
                    continue
                d = torch.load(pt, weights_only=False)
                bt = d["bond_type"]
                n = len(bt)
                meta_bt.extend(bt)
                meta_mid.extend([pt.stem] * n)
                meta_dist.extend(d["distance"].tolist())
                meta_cs.extend([structs[pt.stem].get("crystal_system", "Unknown")] * n)
                mat_counts[pt.stem] = n
            bt_arr = np.array(meta_bt); mid_arr = np.array(meta_mid)
            dist_arr = np.array(meta_dist, dtype=np.float32); cs_arr = np.array(meta_cs)
            total = len(bt_arr)
            log(f"  total edges: {total:,}")

            cnt = Counter(meta_bt)
            sqrt_total = sum(np.sqrt(c) for c in cnt.values())
            rng = np.random.default_rng(42)
            sidx = []
            for bt, c in cnt.items():
                q = int(edge_umap_n * np.sqrt(c) / sqrt_total)
                q = max(q, 20); q = min(q, c)
                pool = np.where(bt_arr == bt)[0]
                sidx.extend(rng.choice(pool, size=q, replace=False).tolist())
            sidx = np.array(sidx); rng.shuffle(sidx)
            log(f"  subsample: {len(sidx):,} ({len(sidx)/total:.1%}, {len(cnt)} bond types)")

            off = 0; mat_off = {}
            for pt in pts:
                if pt.stem not in valid_ids:
                    continue
                mat_off[pt.stem] = off; off += mat_counts[pt.stem]
            idx_by_mat = defaultdict(list)
            for gi in sidx.tolist():
                idx_by_mat[mid_arr[gi]].append(gi - mat_off[mid_arr[gi]])
            gi_to_pos = {int(gi): pos for pos, gi in enumerate(sidx)}

            X = np.zeros((len(sidx), dim), dtype=np.float32)
            for pt in pts:
                if pt.stem not in idx_by_mat:
                    continue
                d = torch.load(pt, weights_only=False)
                conv = convert_fn(d["edge_msg"].float().numpy())   # [m, dim]
                base = mat_off[pt.stem]
                for li in idx_by_mat[pt.stem]:
                    X[gi_to_pos[base + li]] = conv[li]

            bt_sub = bt_arr[sidx]; mid_sub = mid_arr[sidx]
            dist_sub = dist_arr[sidx]; cs_sub = cs_arr[sidx]
            norms = np.linalg.norm(X, axis=1)                     # raw norms (pre-normalisation)
            X = X / np.where(norms[:, None] == 0, 1, norms[:, None])   # L2-normalise (edges)
            evr, pca_n, X_pca = fit_pca(X)
            log(f"  PCA {pca_n} comps ({evr[:pca_n].sum():.1%}); UMAP {len(X_pca):,} ...")
            X_umap = run_umap(X_pca, UMAP_NN_NODE)
            np.savez(cache_path, X_umap=X_umap, X_pca=X_pca, X_raw=X,
                     labels=cs_sub, mp_ids=mid_sub, bond_types=bt_sub, distances=dist_sub,
                     norms=norms, evr=evr, n_total=np.array([total]))
            log(f"  Saved -> {cache_path}")
        else:
            c = np.load(cache_path, allow_pickle=True)
            X = c["X_raw"]; cs_sub = c["labels"]; mid_sub = c["mp_ids"]
            bt_sub = c["bond_types"]; dist_sub = c["distances"]
            evr = c["evr"]; pca_n = pca_n_for_variance(evr)
            total = int(c["n_total"][0]) if "n_total" in c else len(X)
            norms = c["norms"] if "norms" in c else np.linalg.norm(X, axis=1)

        if K_edge is None:
            K_edge = max(10, min(int(round(np.sqrt(len(X)))), len(X) // 10))
            log(f"  K={K_edge} ({binding})")

        if not quality_path.exists():
            quality_figure(layer, EMB_LABELS[emb_key], evr, pca_n, norms, quality_path,
                           EMB_COLORS[emb_key], "Edge count",
                           f"L2 norm of {dim}-D {emb_key}")
        knn_path = edges_dir / f"{layer}_{emb_key}_knn_lvr.txt"
        if not knn_path.exists():
            nn = build_nn(X)
            buf = io.StringIO()
            buf.write(f"=== {emb_key} -- {layer} ===\n")
            buf.write(f"Edges (subsample): {len(X):,} of {total:,} ({len(X)/total:.1%})  "
                      f"Materials: {len(set(mid_sub.tolist())):,}  "
                      f"Bond types: {len(set(bt_sub.tolist()))}\n")
            buf.write(f"K={K_edge}  ({binding})\n\n-- Categorical (k-NN purity: micro / macro) --\n")
            buf.write(knn_purity_line("bond_type", bt_sub, nn, X, build_nn))
            buf.write(knn_purity_line("crystal_system", cs_sub, nn, X, build_nn))
            buf.write(knn_purity_line("metal_insulator",
                [_metal_insulator(structs, m) for m in mid_sub], nn, X, build_nn))
            buf.write(knn_purity_line("magnetic_label",
                [_mag_label(structs, m) for m in mid_sub], nn, X, build_nn))
            buf.write("\n-- Continuous (LVR) --\n")
            buf.write(lvr_array("bond_distance (A)", dist_sub, nn, X, K_edge, build_nn))
            md = _mag_density_dict(structs)
            for name, key in LVR_MAT_PROPS:
                buf.write(lvr_material(name, _prop_dict(structs, key), mid_sub, nn, X, K_edge, build_nn))
            buf.write(lvr_material("mag_density (uB/A3)", md, mid_sub, nn, X, K_edge, build_nn))
            knn_path.write_text(buf.getvalue())
            log(f"  Saved -> {knn_path}")
        log(f"  [{layer}] done {time.time()-t0:.0f}s")
    progress_q.put(f"__DONE__{worker_id}")


def _worker_edges_pooled(worker_id, emb_key, emb_dir, edges_dir, jsonl_path,
                         _unused, progress_q):
    """Per-material pooled-edge analysis (mean over ALL edges -> [9,128] -> variant)."""
    matplotlib.use("Agg")
    convert_fn = CONVERT_FN[emb_key]
    dim = EMB_DIM[emb_key]

    def log(msg):
        progress_q.put(f"[{emb_key:<22}] {msg}")

    structs, valid_ids = _load_manifest(jsonl_path)
    K = None
    binding = "sqrt(n_mats)"

    def build_nn(X):
        k = min(K, len(X) - 1)
        return (NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=1)
                .fit(X).kneighbors(X, return_distance=False)[:, 1:])

    for layer in EDGE_LAYERS:
        t0 = time.time()
        edir = emb_dir / layer / "edges"
        log(f"--- {layer} ---")
        cache_path = edges_dir / f"{layer}_{emb_key}_umap_cache.npz"
        quality_path = edges_dir / f"{layer}_{emb_key}_pca_quality.png"

        if not cache_path.exists():
            pts = sorted(edir.glob("*.pt"))
            embs, mids = [], []
            for pt in pts:
                if pt.stem not in valid_ids:
                    continue
                d = torch.load(pt, weights_only=False)
                ep = d["edge_pooled"].float().numpy()              # [9, 128]
                embs.append(convert_fn(ep[None])[0])               # [dim]
                mids.append(pt.stem)
            X = np.stack(embs, axis=0); mids = np.array(mids)
            fin = np.isfinite(X).all(axis=1)
            if not fin.all():
                X = X[fin]; mids = mids[fin]
            norms = np.linalg.norm(X, axis=1)                     # raw norms (pre-normalisation)
            X = X / np.where(norms[:, None] == 0, 1, norms[:, None])   # L2-normalise
            log(f"  {len(X):,} materials")
            evr, pca_n, X_pca = fit_pca(X)
            X_umap = run_umap(X_pca, UMAP_NN_NODE)
            np.savez(cache_path, X_umap=X_umap, X_pca=X_pca, X_raw=X, mp_ids=mids,
                     norms=norms, evr=evr)
            log(f"  Saved -> {cache_path}")
        else:
            c = np.load(cache_path, allow_pickle=True)
            X = c["X_raw"]; mids = c["mp_ids"]; evr = c["evr"]
            pca_n = pca_n_for_variance(evr)
            norms = c["norms"] if "norms" in c else np.linalg.norm(X, axis=1)

        if K is None:
            K = max(10, min(int(round(np.sqrt(len(X)))), len(X) // 10))
            log(f"  K={K} ({binding})")
        if not quality_path.exists():
            quality_figure(layer, EMB_LABELS[emb_key], evr, pca_n, norms, quality_path,
                           EMB_COLORS[emb_key], "Material count",
                           f"L2 norm of {dim}-D {emb_key}")
        knn_path = edges_dir / f"{layer}_{emb_key}_knn_lvr.txt"
        if not knn_path.exists():
            nn = build_nn(X)
            buf = io.StringIO()
            buf.write(f"=== {emb_key} -- {layer} ===\n")
            buf.write(f"Materials: {len(X):,}\nK={K}  ({binding})\n\n")
            buf.write("-- Categorical (k-NN purity: micro / macro) --\n")
            buf.write(knn_purity_line("crystal_system",
                [structs[m].get("crystal_system") for m in mids], nn, X, build_nn))
            buf.write(knn_purity_line("metal_insulator",
                [_metal_insulator(structs, m) for m in mids], nn, X, build_nn))
            buf.write(knn_purity_line("magnetic_label",
                [_mag_label(structs, m) for m in mids], nn, X, build_nn))
            buf.write("\n-- Continuous (LVR) --\n")
            md = _mag_density_dict(structs)
            for name, key in LVR_MAT_PROPS:
                buf.write(lvr_material(name, _prop_dict(structs, key), mids, nn, X, K, build_nn))
            buf.write(lvr_material("mag_density (uB/A3)", md, mids, nn, X, K, build_nn))
            knn_path.write_text(buf.getvalue())
            log(f"  Saved -> {knn_path}")
        log(f"  [{layer}] done {time.time()-t0:.0f}s")
    progress_q.put(f"__DONE__{worker_id}")


def _edge_entry(wid, emb_key, emb_dir, edges_dir, jsonl_path, n, q):
    if "pooled" in emb_key:
        _worker_edges_pooled(wid, emb_key, emb_dir, edges_dir, jsonl_path, n, q)
    else:
        _worker_edges(wid, emb_key, emb_dir, edges_dir, jsonl_path, n, q)


# ---------------------------------------------------------------------------
# Worker entry points (one per embedding type)
# ---------------------------------------------------------------------------

def _worker_nodes_flat(wid, emb_dir, nodes_dir, jsonl_path, n, q):
    _worker_per_atom(wid, "nodes_flat", emb_dir, nodes_dir, jsonl_path, n, q)

def _worker_nodes_ang_norm(wid, emb_dir, nodes_dir, jsonl_path, n, q):
    _worker_per_atom(wid, "nodes_ang_norm", emb_dir, nodes_dir, jsonl_path, n, q)

def _worker_nodes_l0(wid, emb_dir, nodes_dir, jsonl_path, n, q):
    _worker_per_atom(wid, "nodes_l0", emb_dir, nodes_dir, jsonl_path, n, q)

def _worker_nodes_l1_norm(wid, emb_dir, nodes_dir, jsonl_path, n, q):
    _worker_per_atom(wid, "nodes_l1_norm", emb_dir, nodes_dir, jsonl_path, n, q)

def _worker_nodes_l2_norm(wid, emb_dir, nodes_dir, jsonl_path, n, q):
    _worker_per_atom(wid, "nodes_l2_norm", emb_dir, nodes_dir, jsonl_path, n, q)

def _worker_nodes_geom_norm(wid, emb_dir, nodes_dir, jsonl_path, n, q):
    _worker_per_atom(wid, "nodes_geom_norm", emb_dir, nodes_dir, jsonl_path, n, q)

def _worker_pooled_full(wid, emb_dir, nodes_dir, jsonl_path, K, b, q):
    _worker_pooled(wid, "nodes_pooled_full", emb_dir, nodes_dir, jsonl_path, K, b, q)

def _worker_pooled_l0(wid, emb_dir, nodes_dir, jsonl_path, K, b, q):
    _worker_pooled(wid, "nodes_pooled_l0", emb_dir, nodes_dir, jsonl_path, K, b, q)

def _worker_pooled_ang_norm(wid, emb_dir, nodes_dir, jsonl_path, K, b, q):
    _worker_pooled(wid, "nodes_pooled_ang_norm", emb_dir, nodes_dir, jsonl_path, K, b, q)

def _worker_pooled_l1_polar(wid, emb_dir, nodes_dir, jsonl_path, K, b, q):
    _worker_pooled_computed(wid, "nodes_pooled_l1_polar", emb_dir, nodes_dir, jsonl_path, K, b, q)

def _worker_pooled_l2_polar(wid, emb_dir, nodes_dir, jsonl_path, K, b, q):
    _worker_pooled_computed(wid, "nodes_pooled_l2_polar", emb_dir, nodes_dir, jsonl_path, K, b, q)

def _worker_pooled_geom_norm(wid, emb_dir, nodes_dir, jsonl_path, K, b, q):
    _worker_pooled_computed(wid, "nodes_pooled_geom_norm", emb_dir, nodes_dir, jsonl_path, K, b, q)


# ---------------------------------------------------------------------------
# Progress listener
# ---------------------------------------------------------------------------

def _listen(progress_q, n_workers: int) -> None:
    done = 0
    while done < n_workers:
        try:
            msg = progress_q.get(timeout=300)
        except Exception:
            continue
        if msg.startswith("__DONE__"):
            done += 1
            print(f"  Worker {msg[8:]} finished  ({done}/{n_workers} done)", flush=True)
        else:
            print(msg, flush=True)


# ---------------------------------------------------------------------------
# Edge global figures
# ---------------------------------------------------------------------------

def emit_edge_global_figures(edges_dir: Path, global_dir: Path, bb_dir: Path,
                             structs: dict) -> None:
    """Global cross-cutting figures for EDGE embeddings, mirroring the node
    global figures but for per-edge (EDGE_ATOM_KEYS) and pooled-edge
    (EDGE_POOLED_KEYS) representations across EDGE_LAYERS. Purely additive: writes
    new ``*_edges*`` files into global_dir alongside the node figures and never
    touches node outputs. Every output is idempotent (skips if it already exists).

    UMA has too many embedding types to share the node grids (as the ORB script
    does), so edges get their own copies. The node-only ``element_identity`` plot
    has no edge analogue; following the ORB ``umap_element_bond_summary`` it is
    replaced by two per-edge figures: bond type and bond distance."""
    print("\n\n========== EDGE GLOBAL SUMMARY ==========")
    LAYERS     = EDGE_LAYERS
    N_LAYERS   = len(LAYERS)
    LAYER_LBLS = [LAYER_LABELS[l] for l in LAYERS]
    SUP_Y      = 0.99

    KNN_BG_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "knn_bg", ["#d3d3d3", "#fff3b0", "#aed6f1", "#a9dfbf"])
    LVR_BG_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "lvr_bg", ["#a9dfbf", "#aed6f1", "#fff3b0", "#d3d3d3"])
    ZONE_ALPHA = 0.45

    def _load_c(layer, ek):
        p = edges_dir / f"{layer}_{ek}_umap_cache.npz"
        return np.load(p, allow_pickle=True) if p.exists() else None

    def _draw_na(ax):
        ax.set_facecolor("#eeeeee")
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="#888888")
        ax.set_xticks([]); ax.set_yticks([])

    def _grid(keys, title):
        n_cols = len(keys)
        fig, axes = plt.subplots(N_LAYERS, n_cols, figsize=(4 * n_cols, 4 * N_LAYERS),
                                 squeeze=False)
        fig.suptitle(title, fontsize=13, y=SUP_Y)
        for ci, k in enumerate(keys):
            axes[0][ci].set_title(EMB_LABELS[k], fontsize=9, pad=6)
        for ri, lbl in enumerate(LAYER_LBLS):
            axes[ri][0].set_ylabel(lbl, fontsize=9, labelpad=5)
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        return fig, axes

    def _get_mp_ids(c):
        return c["mp_ids"].astype(str) if "mp_ids" in c.files else np.array([], dtype=str)

    def _v(s, k):
        v = s.get(k)
        return float(v) if v is not None else np.nan

    def _v_pos(s, k):
        v = s.get(k)
        if v is None:
            return np.nan
        v = float(v)
        return v if v > 0 else np.nan

    def _legend_center(fig, axes, handles, title=None, ncol=4):
        plt.tight_layout(rect=[0, 0, 1.0, 0.94])
        _p0 = axes[0][0].get_position()
        fig.legend(handles=handles, title=title, loc="center", ncol=ncol,
                   fontsize=8, title_fontsize=9, framealpha=0.92,
                   bbox_to_anchor=(0.5, (_p0.y1 + SUP_Y) / 2),
                   bbox_transform=fig.transFigure)

    # ---- categorical: crystal system (per-edge uses cache labels) ----
    def _plot_cs(keys, is_peredge, out, title):
        if out.exists():
            return
        fig, axes = _grid(keys, title)
        for ri, layer in enumerate(LAYERS):
            for ci, ek in enumerate(keys):
                ax = axes[ri][ci]
                c  = _load_c(layer, ek)
                if c is None:
                    _draw_na(ax); continue
                X  = c["X_umap"]
                cs = (c["labels"].astype(str) if (is_peredge and "labels" in c.files)
                      else np.array([structs.get(m, {}).get("crystal_system", "Unknown")
                                     for m in _get_mp_ids(c)]))
                no_data = ~np.isin(cs, CRYSTAL_SYSTEMS)
                if no_data.any():
                    ax.scatter(X[no_data, 0], X[no_data, 1], c="#dddddd",
                               s=1, alpha=0.1, linewidths=0, rasterized=True)
                for _cs in CRYSTAL_SYSTEMS:
                    mask = cs == _cs
                    if mask.any():
                        ax.scatter(X[mask, 0], X[mask, 1], c=[CRYSTAL_COLORS[_cs]],
                                   s=1, alpha=0.1, linewidths=0, rasterized=True)
        handles = [mlines.Line2D([], [], marker="o", linestyle="None",
                                 color=CRYSTAL_COLORS[cs], markersize=6, label=cs)
                   for cs in CRYSTAL_SYSTEMS]
        handles.append(mlines.Line2D([], [], marker="o", linestyle="None",
                                     color="#dddddd", markersize=6, label="no data"))
        _legend_center(fig, axes, handles, title="Crystal system", ncol=4)
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved -> {out}")

    _plot_cs(EDGE_ATOM_KEYS, True,
             global_dir / "umap_crystal_system_edges.png",
             "UMAP — Crystal System  (edges)")
    _plot_cs(EDGE_POOLED_KEYS, False,
             global_dir / "umap_crystal_system_edges_pooled.png",
             "UMAP — Crystal System  (edges pooled)")

    # ---- categorical: binary (metal/insulator, magnetic) via material lookup ----
    def _plot_binary(keys, out, title, get_labels_fn, colors, labels_txt):
        if out.exists():
            return
        fig, axes = _grid(keys, title)
        for ri, layer in enumerate(LAYERS):
            for ci, ek in enumerate(keys):
                ax = axes[ri][ci]
                c  = _load_c(layer, ek)
                if c is None:
                    _draw_na(ax); continue
                X    = c["X_umap"]
                cats = get_labels_fn(_get_mp_ids(c))
                no_data = np.array([v is None for v in cats])
                if no_data.any():
                    ax.scatter(X[no_data, 0], X[no_data, 1], c="#dddddd",
                               s=1, alpha=0.1, linewidths=0, rasterized=True)
                for cat, color in colors.items():
                    mask = np.array([v == cat for v in cats])
                    if mask.any():
                        ax.scatter(X[mask, 0], X[mask, 1], c=color,
                                   s=1, alpha=0.1, linewidths=0, rasterized=True)
        handles = [mlines.Line2D([], [], marker="o", linestyle="None",
                                 color=c, markersize=6, label=l) for l, c in labels_txt]
        _legend_center(fig, axes, handles, ncol=len(labels_txt))
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved -> {out}")

    def _mi_labels(mp_ids):
        bg = [structs.get(m, {}).get("band_gap") for m in mp_ids]
        return [None if v is None else ("Metal" if v == 0 else "Insulator") for v in bg]

    for keys, sfx, lab in [(EDGE_ATOM_KEYS, "edges", "edges"),
                           (EDGE_POOLED_KEYS, "edges_pooled", "edges pooled")]:
        _plot_binary(keys, global_dir / f"umap_metal_insulator_{sfx}.png",
                     f"UMAP — Metal / Insulator  ({lab})", _mi_labels,
                     {"Metal": "teal", "Insulator": "goldenrod"},
                     [("metal", "teal"), ("insulator", "goldenrod"), ("no data", "#dddddd")])

    def _mag_labels(mp_ids):
        out = []
        for m in mp_ids:
            s = structs.get(m, {})
            tm, vol = s.get("total_magnetization"), s.get("volume")
            if tm is None or not vol:
                out.append(None)
            else:
                out.append("magnetic" if abs(tm / vol) >= MAG_THR else "non-magnetic")
        return out

    for keys, sfx, lab in [(EDGE_ATOM_KEYS, "edges", "edges"),
                           (EDGE_POOLED_KEYS, "edges_pooled", "edges pooled")]:
        _plot_binary(keys, global_dir / f"umap_magnetic_label_{sfx}.png",
                     f"UMAP — Magnetic Label  ({lab})", _mag_labels,
                     {"non-magnetic": "turquoise", "magnetic": "crimson"},
                     [("non-magnetic", "turquoise"), ("magnetic", "crimson"),
                      ("no data", "#dddddd")])

    # ---- continuous property grids (material value via mp_ids) ----
    def _cont(keys, out, title, get_val_fn, cmap, norm, cbar_label):
        if out.exists():
            return
        fig, axes = _grid(keys, title)
        sc_last = None
        n_miss = n_tot = 0
        for ri, layer in enumerate(LAYERS):
            for ci, ek in enumerate(keys):
                ax = axes[ri][ci]
                c  = _load_c(layer, ek)
                if c is None:
                    _draw_na(ax); continue
                X    = c["X_umap"]
                vals = np.array([get_val_fn(structs.get(m, {})) for m in _get_mp_ids(c)],
                                dtype=float)
                has  = ~np.isnan(vals)
                n_miss += int((~has).sum()); n_tot += has.size
                if (~has).any():
                    ax.scatter(X[~has, 0], X[~has, 1], c="lightgrey",
                               s=1, alpha=0.1, linewidths=0, rasterized=True)
                if has.any():
                    sc_last = ax.scatter(X[has, 0], X[has, 1], c=vals[has], cmap=cmap,
                                         norm=norm, s=1, alpha=0.1, linewidths=0,
                                         rasterized=True)
        if n_miss > 0:
            miss_pct = 100.0 * n_miss / max(n_tot, 1)
            fig.legend(handles=[mlines.Line2D([], [], marker="o", linestyle="None",
                                              color="lightgrey", markersize=6,
                                              label=f"missing data ({miss_pct:.1f}%)")],
                       loc="upper left", bbox_to_anchor=(0.01, 0.99),
                       fontsize=8, framealpha=0.9, bbox_transform=fig.transFigure)
        plt.tight_layout(rect=[0, 0, 1.0, 0.94])
        _p0 = axes[0][0].get_position()
        _cbar_h, _cbar_w = 0.007, 0.35
        _cy = (_p0.y1 + SUP_Y) / 2
        cax = fig.add_axes([0.5 - _cbar_w / 2, _cy - _cbar_h / 2, _cbar_w, _cbar_h])
        _src = sc_last if sc_last is not None else plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        if sc_last is None:
            _src.set_array([])
        cb = fig.colorbar(_src, cax=cax, orientation="horizontal")
        if sc_last is not None:
            cb.solids.set_alpha(1)
        cax.set_title(cbar_label, fontsize=9, pad=3)
        cb.ax.tick_params(labelsize=7)
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved -> {out}")

    _fe_all  = np.array([v for m in structs for v in [structs[m].get("formation_energy_per_atom")]
                          if v is not None], dtype=float)
    _fe_vmin = float(np.percentile(_fe_all, 1))  if len(_fe_all) else -3.0
    _fe_vmax = float(np.percentile(_fe_all, 99)) if len(_fe_all) else  3.0

    _cont_specs = [
        ("umap_formation_energy", "UMAP — Formation Energy / Atom",
         lambda s: _v(s, "formation_energy_per_atom"), "coolwarm",
         mcolors.Normalize(vmin=_fe_vmin, vmax=_fe_vmax), "Formation energy (eV/atom)"),
        ("umap_energy_above_hull", "UMAP — Energy Above Hull",
         lambda s: _v(s, "energy_above_hull"), "coolwarm",
         mcolors.PowerNorm(gamma=0.4, vmin=0.0, vmax=1.0),
         "Energy above hull (eV/atom)  [sqrt-ish, 0–1 eV]"),
        ("umap_density", "UMAP — Density",
         lambda s: _v(s, "density"), "viridis",
         mcolors.PowerNorm(gamma=0.6, vmin=0, vmax=20.0),
         "Density (g/cm³)  [sqrt-ish, 0–20 g/cm³]"),
        ("umap_bulk_modulus", "UMAP — Bulk Modulus",
         lambda s: _v(s, "bulk_modulus"), "viridis",
         mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=440.0),
         "Bulk modulus (GPa)  [sqrt scale, 0–440 GPa]"),
    ]
    for stem, title, fn, cmap, norm, cbar in _cont_specs:
        _cont(EDGE_ATOM_KEYS, global_dir / f"{stem}_edges.png",
              f"{title}  (edges)", fn, cmap, norm, cbar)
        _cont(EDGE_POOLED_KEYS, global_dir / f"{stem}_edges_pooled.png",
              f"{title}  (edges pooled)", fn, cmap, norm, cbar)

    # ---- band gap (metal split + plasma scale), same recipe as the node path ----
    def _plot_bg(keys, out, lab):
        if out.exists():
            return
        norm_bg = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=5.0)
        fig, axes = _grid(keys, f"UMAP — Band Gap  ({lab})")
        sc_last = None
        for ri, layer in enumerate(LAYERS):
            for ci, ek in enumerate(keys):
                ax = axes[ri][ci]
                c  = _load_c(layer, ek)
                if c is None:
                    _draw_na(ax); continue
                X   = c["X_umap"]
                bg  = np.array([structs.get(m, {}).get("band_gap") for m in _get_mp_ids(c)])
                has = np.array([v is not None for v in bg])
                bgf = np.array([v if v is not None else np.nan for v in bg], dtype=float)
                is_metal    = has & (bgf == 0.0)
                is_nonmetal = has & (bgf > 0.0)
                if (~has).any():
                    ax.scatter(X[~has, 0], X[~has, 1], c="lightgrey",
                               s=1, alpha=0.1, linewidths=0, rasterized=True)
                ax.scatter(X[is_metal, 0], X[is_metal, 1], c="aquamarine",
                           s=1, alpha=0.1, linewidths=0, rasterized=True)
                if is_nonmetal.any():
                    sc_last = ax.scatter(X[is_nonmetal, 0], X[is_nonmetal, 1],
                                         c=bgf[is_nonmetal], cmap="plasma", norm=norm_bg,
                                         s=1, alpha=0.1, linewidths=0, rasterized=True)
        plt.tight_layout(rect=[0, 0, 1.0, 0.94])
        _p0 = axes[0][0].get_position(); _cy = (_p0.y1 + SUP_Y) / 2
        if sc_last is not None:
            cax = fig.add_axes([0.5 - 0.15, _cy - 0.0035, 0.30, 0.007])
            cb  = fig.colorbar(sc_last, cax=cax, orientation="horizontal")
            cb.solids.set_alpha(1)
            cax.set_title("Band gap (eV)  [sqrt scale, 0–5 eV]", fontsize=8, pad=3)
            cb.ax.tick_params(labelsize=7)
            fig.legend(handles=[mpatches.Patch(facecolor="aquamarine", label="Metal (bg=0)")],
                       loc="center left", fontsize=8, framealpha=0,
                       bbox_to_anchor=(0.5 + 0.15 + 0.02, _cy),
                       bbox_transform=fig.transFigure)
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved -> {out}")

    _plot_bg(EDGE_ATOM_KEYS, global_dir / "umap_band_gap_edges.png", "edges")
    _plot_bg(EDGE_POOLED_KEYS, global_dir / "umap_band_gap_edges_pooled.png", "edges pooled")

    # ---- bond type + bond distance (per-edge only; element-identity analogue) ----
    # Reference top-10 bond types from the deepest available per-edge layer.
    _ref_c = None
    for _L in reversed(LAYERS):
        for _k in EDGE_ATOM_KEYS:
            _ref_c = _load_c(_L, _k)
            if _ref_c is not None and "bond_types" in _ref_c.files:
                break
        if _ref_c is not None and "bond_types" in _ref_c.files:
            break
    _top_bt = ([bt for bt, _ in Counter(_ref_c["bond_types"].astype(str).tolist()).most_common(10)]
               if (_ref_c is not None and "bond_types" in _ref_c.files) else [])
    _BT_CMAP = plt.colormaps["gist_rainbow"].resampled(max(len(_top_bt), 1))
    _BT_CM   = {bt: _BT_CMAP(i) for i, bt in enumerate(_top_bt)}

    _bt_out = global_dir / "umap_bond_type_edges.png"
    if not _bt_out.exists():
        fig, axes = _grid(EDGE_ATOM_KEYS, "UMAP — Bond Type  (edges, top-10 bond types)")
        for ri, layer in enumerate(LAYERS):
            for ci, ek in enumerate(EDGE_ATOM_KEYS):
                ax = axes[ri][ci]
                c  = _load_c(layer, ek)
                if c is None or "bond_types" not in c.files:
                    _draw_na(ax); continue
                X  = c["X_umap"]; bt = c["bond_types"].astype(str)
                other = ~np.isin(bt, _top_bt)
                if other.any():
                    ax.scatter(X[other, 0], X[other, 1], c="lightgrey",
                               s=1, alpha=0.1, linewidths=0, rasterized=True)
                for _bt in _top_bt:
                    mask = bt == _bt
                    if mask.any():
                        ax.scatter(X[mask, 0], X[mask, 1], c=[_BT_CM[_bt]],
                                   s=1, alpha=0.1, linewidths=0, rasterized=True)
        handles = ([mlines.Line2D([], [], marker="o", linestyle="None",
                                  color=_BT_CM[bt], markersize=6, label=bt) for bt in _top_bt]
                   + [mlines.Line2D([], [], marker="o", linestyle="None",
                                    color="lightgrey", markersize=6, label="other")])
        _legend_center(fig, axes, handles, title="Bond type (top 10)", ncol=4)
        plt.savefig(_bt_out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved -> {_bt_out}")

    _bd_out = global_dir / "umap_bond_distance_edges.png"
    if not _bd_out.exists():
        _d_ref = (_ref_c["distances"].astype(float)
                  if (_ref_c is not None and "distances" in _ref_c.files) else np.array([1.0, 4.0]))
        _bd_norm = mcolors.Normalize(vmin=float(np.percentile(_d_ref, 1)),
                                     vmax=float(np.percentile(_d_ref, 99)))
        fig, axes = _grid(EDGE_ATOM_KEYS, "UMAP — Bond Distance  (edges)")
        sc_last = None
        for ri, layer in enumerate(LAYERS):
            for ci, ek in enumerate(EDGE_ATOM_KEYS):
                ax = axes[ri][ci]
                c  = _load_c(layer, ek)
                if c is None or "distances" not in c.files:
                    _draw_na(ax); continue
                X = c["X_umap"]; dist = c["distances"].astype(float)
                sc_last = ax.scatter(X[:, 0], X[:, 1], c=dist, cmap="plasma", norm=_bd_norm,
                                     s=1, alpha=0.1, linewidths=0, rasterized=True)
        plt.tight_layout(rect=[0, 0, 1.0, 0.94])
        _p0 = axes[0][0].get_position(); _cy = (_p0.y1 + SUP_Y) / 2
        if sc_last is not None:
            cax = fig.add_axes([0.5 - 0.15, _cy - 0.0035, 0.30, 0.007])
            cb  = fig.colorbar(sc_last, cax=cax, orientation="horizontal")
            cb.solids.set_alpha(1)
            cax.set_title("Bond distance (Å)", fontsize=8, pad=3)
            cb.ax.tick_params(labelsize=7)
        plt.savefig(_bd_out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved -> {_bd_out}")

    # ---- PCA edge-quality grids ----
    pca_quality_grid(edges_dir, EDGE_ATOM_KEYS, "edges",
                     global_dir / "pca_edge_quality_grid_atom.png", layers=LAYERS)
    pca_quality_grid(edges_dir, EDGE_POOLED_KEYS, "edges pooled",
                     global_dir / "pca_edge_quality_grid_pooled.png", layers=LAYERS)

    # =====================================================================
    # Edge cross-layer summaries (parity with node section)
    # =====================================================================
    XLABELS = [LAYER_LABELS[l] for l in LAYERS]
    X_POS   = list(range(N_LAYERS))
    # bond_distance is a per-edge-only property (pooled-edge caches have no
    # per-edge distance), so pooled-edge LVR figures must omit it or they show
    # an empty bond_distance panel. Per-edge figures keep it.
    EDGE_LVR_PROPS_ATOM = LVR_CROSS_PROPS + ["bond_distance (A)"]
    EDGE_LVR_PROPS_POOL = LVR_CROSS_PROPS
    EDGE_LVR_PROPS      = EDGE_LVR_PROPS_ATOM  # superset for tables/metrics (data-filtered)
    _knn_colors = dict(KNN_COLOR_MAP); _knn_colors.setdefault("bond_type", "#B5651D")

    data_all: dict[str, dict[str, tuple[dict, dict, dict]]] = {}
    for ek in ALL_EDGE_KEYS:
        data_all[ek] = {}
        for layer in LAYERS:
            data_all[ek][layer] = parse_knn_lvr_txt(
                edges_dir / f"{layer}_{ek}_knn_lvr.txt")

    _all_knn_vals = [data_all[ek][ly][0].get(pn, np.nan)
                     for ek in ALL_EDGE_KEYS for ly in LAYERS for pn in KNN_PROPS[ek]]
    _knn_min = float(np.nanmin(_all_knn_vals)) if any(not np.isnan(v) for v in _all_knn_vals) else -0.10
    Y_MIN = min(-0.10, np.floor(_knn_min * 10) / 10 - 0.05)
    Y_MAX = 1.05

    def _knn_panel(ax, ek):
        grad = np.linspace(Y_MIN, Y_MAX, 256).reshape(-1, 1)
        ax.imshow(grad, aspect="auto", cmap=KNN_BG_CMAP, vmin=Y_MIN, vmax=1.0,
                  extent=[-0.3, N_LAYERS - 0.7, Y_MIN, Y_MAX], origin="lower",
                  zorder=0, alpha=ZONE_ALPHA)
        ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=2)
        for prop_name in KNN_PROPS[ek]:
            y = np.array([data_all[ek][l][0].get(prop_name, np.nan) for l in LAYERS], dtype=float)
            mask = ~np.isnan(y)
            if mask.any():
                ax.plot(np.array(X_POS)[mask], y[mask], "o-",
                        color=_knn_colors.get(prop_name, "#888888"),
                        lw=2.2, ms=5, zorder=3, label=prop_name)
        ax.set_title(EMB_LABELS[ek], fontsize=10)
        ax.set_xticks(X_POS); ax.set_xticklabels(XLABELS, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("Normalised purity"); ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xlim(-0.3, N_LAYERS - 0.7); ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.25, zorder=1)

    knn_fig = global_dir / "knn_purity_across_layers_edges.png"
    if not knn_fig.exists():
        fig, axes = plt.subplots(2, len(EDGE_ATOM_KEYS),
                                 figsize=(6 * len(EDGE_ATOM_KEYS), 10), squeeze=False)
        fig.suptitle("k-NN Purity (normalised) across UMA layers — edges", fontsize=14, y=1.01)
        for ci, ek in enumerate(EDGE_ATOM_KEYS):
            _knn_panel(axes[0][ci], ek)
        for ci, ek in enumerate(EDGE_POOLED_KEYS):
            _knn_panel(axes[1][ci], ek)
        plt.tight_layout(); plt.savefig(knn_fig, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved -> {knn_fig}")

    def _lvr_panel(ax, ek, props):
        all_vals = [np.array([data_all[ek][l][1].get(p, np.nan) for l in LAYERS], dtype=float)
                    for p in props]
        y_max  = max((np.nanmax(v) for v in all_vals if np.any(~np.isnan(v))), default=1.0)
        y_ceil = max(y_max * 1.08, 0.90)
        grad = np.linspace(0, y_ceil, 256).reshape(-1, 1)
        ax.imshow(grad, aspect="auto", cmap=LVR_BG_CMAP, vmin=0.0, vmax=1.0,
                  extent=[-0.3, N_LAYERS - 0.7, 0, y_ceil], origin="lower",
                  zorder=0, alpha=ZONE_ALPHA)
        for i, prop_name in enumerate(props):
            y = np.array([data_all[ek][l][1].get(prop_name, np.nan) for l in LAYERS], dtype=float)
            mask = ~np.isnan(y)
            if mask.any():
                ax.plot(np.array(X_POS)[mask], y[mask], "o-",
                        color=LVR_COLORS[i % len(LVR_COLORS)], lw=2.2, ms=5,
                        zorder=3, label=prop_name)
        ax.set_title(EMB_LABELS[ek], fontsize=10)
        ax.set_xticks(X_POS); ax.set_xticklabels(XLABELS, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("LVR"); ax.set_ylim(0, y_ceil)
        ax.set_xlim(-0.3, N_LAYERS - 0.7); ax.legend(fontsize=7, loc="upper right", ncol=1)
        ax.grid(axis="y", alpha=0.25, zorder=1)

    lvr_fig = global_dir / "lvr_across_layers_edges.png"
    if not lvr_fig.exists():
        fig, axes = plt.subplots(2, len(EDGE_ATOM_KEYS),
                                 figsize=(6 * len(EDGE_ATOM_KEYS), 12), squeeze=False)
        fig.suptitle("Local Variance Ratio (LVR) across UMA layers — edges\n"
                     "(lower = tighter clustering of the property)", fontsize=14, y=1.02)
        for ci, ek in enumerate(EDGE_ATOM_KEYS):
            _lvr_panel(axes[0][ci], ek, EDGE_LVR_PROPS_ATOM)
        for ci, ek in enumerate(EDGE_POOLED_KEYS):
            _lvr_panel(axes[1][ci], ek, EDGE_LVR_PROPS_POOL)
        plt.tight_layout(); plt.savefig(lvr_fig, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved -> {lvr_fig}")

    # ---- representation comparison (lines = edge types) ----
    _DIM_LBL = {k: f"({EMB_DIM[k]}d)" for k in ALL_EDGE_KEYS}

    def _rep_cmp_knn(out_path, keys, props, title):
        if out_path.exists():
            return
        n = len(props)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True, squeeze=False)
        axes = axes[0]
        _all = [data_all[ek][ly][0].get(p, np.nan) for ek in keys for ly in LAYERS for p in props]
        _ymin = min(-0.10, np.floor(np.nanmin(_all) * 10) / 10 - 0.05
                    if not np.all(np.isnan(_all)) else -0.10)
        for ax, prop_name in zip(axes, props):
            grad = np.linspace(_ymin, 1.05, 256).reshape(-1, 1)
            ax.imshow(grad, aspect="auto", cmap=KNN_BG_CMAP, vmin=_ymin, vmax=1.0,
                      extent=[-0.3, N_LAYERS - 0.7, _ymin, 1.05], origin="lower",
                      zorder=0, alpha=ZONE_ALPHA)
            ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=2)
            for ek in keys:
                y = np.array([data_all[ek][l][0].get(prop_name, np.nan) for l in LAYERS], dtype=float)
                mask = ~np.isnan(y)
                if mask.any():
                    ax.plot(np.array(X_POS)[mask], y[mask], color=EMB_COLORS.get(ek, "#888888"),
                            ls="-", marker="o", ms=5, lw=2.2, zorder=3,
                            label=f"{EMB_LABELS[ek]} {_DIM_LBL[ek]}")
            ax.set_title(prop_name, fontsize=10)
            ax.set_xticks(X_POS); ax.set_xticklabels(XLABELS, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel("Normalised purity"); ax.set_ylim(_ymin, 1.05)
            ax.set_xlim(-0.3, N_LAYERS - 0.7); ax.grid(axis="y", alpha=0.25, zorder=1)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", fontsize=8, framealpha=0.9,
                   ncol=min(len(handles), max(3, len(handles) // 2)),
                   bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure)
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved -> {out_path}")

    def _rep_cmp_lvr(out_path, keys, title, props):
        if out_path.exists():
            return
        ncols = 3
        nrows = (len(props) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                                 sharey=False, squeeze=False)
        fig.suptitle(f"{title}\n(lower LVR = tighter clustering = better)", fontsize=12, y=0.995)
        _ax_flat = list(axes.flat)
        for idx, prop_name in enumerate(props):
            ax = _ax_flat[idx]
            _vals = [data_all[ek][ly][1].get(prop_name, np.nan) for ek in keys for ly in LAYERS]
            _ymax_p = max(np.nanmax(_vals) * 1.08 if not np.all(np.isnan(_vals)) else 1.0, 0.90)
            grad = np.linspace(0, _ymax_p, 256).reshape(-1, 1)
            ax.imshow(grad, aspect="auto", cmap=LVR_BG_CMAP, vmin=0.0, vmax=1.0,
                      extent=[-0.3, N_LAYERS - 0.7, 0, _ymax_p], origin="lower",
                      zorder=0, alpha=ZONE_ALPHA)
            for ek in keys:
                y = np.array([data_all[ek][l][1].get(prop_name, np.nan) for l in LAYERS], dtype=float)
                mask = ~np.isnan(y)
                if mask.any():
                    ax.plot(np.array(X_POS)[mask], y[mask], color=EMB_COLORS.get(ek, "#888888"),
                            ls="-", marker="o", ms=5, lw=2.2, zorder=3,
                            label=f"{EMB_LABELS[ek]} {_DIM_LBL[ek]}")
            ax.set_title(prop_name, fontsize=9)
            ax.set_xticks(X_POS); ax.set_xticklabels(XLABELS, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel("LVR"); ax.set_ylim(0, _ymax_p)
            ax.set_xlim(-0.3, N_LAYERS - 0.7); ax.grid(axis="y", alpha=0.25, zorder=1)
        for idx in range(len(props), len(_ax_flat)):
            _ax_flat[idx].set_visible(False)
        handles, labels = _ax_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", fontsize=8, framealpha=0.9,
                   ncol=min(len(handles), max(3, len(handles) // 2)),
                   bbox_to_anchor=(0.5, 0.945), bbox_transform=fig.transFigure)
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved -> {out_path}")

    _rep_cmp_knn(global_dir / "knn_rep_comparison_edges.png", EDGE_ATOM_KEYS,
                 ["bond_type", "crystal_system", "metal_insulator", "magnetic_label"],
                 "k-NN Purity — edges representations vs layer  (higher = better)")
    _rep_cmp_knn(global_dir / "knn_rep_comparison_edges_pooled.png", EDGE_POOLED_KEYS,
                 ["crystal_system", "metal_insulator", "magnetic_label"],
                 "k-NN Purity — edges pooled representations vs layer  (higher = better)")
    _rep_cmp_lvr(global_dir / "lvr_rep_comparison_edges.png", EDGE_ATOM_KEYS,
                 "LVR — edges representations vs layer", EDGE_LVR_PROPS_ATOM)
    _rep_cmp_lvr(global_dir / "lvr_rep_comparison_edges_pooled.png", EDGE_POOLED_KEYS,
                 "LVR — edges pooled representations vs layer", EDGE_LVR_PROPS_POOL)

    # ---- summary tables (heatmaps) ----
    knn_table = global_dir / "knn_summary_table_edges.png"
    lvr_table = global_dir / "lvr_summary_table_edges.png"
    if not (knn_table.exists() and lvr_table.exists()):
        _KNN_ALL = ["bond_type", "crystal_system", "metal_insulator", "magnetic_label"]
        _EMB_SHORT = {k: EMB_LABELS[k].replace("Edges pooled ", "pool ").replace("Edges ", "")
                      for k in ALL_EDGE_KEYS}

        def _ht_rows(props, mi):
            rows, spans = [], []
            for prop in props:
                embs = [e for e in ALL_EDGE_KEYS
                        if any(not np.isnan(data_all[e][l][mi].get(prop, np.nan)) for l in LAYERS)]
                if not embs:
                    continue
                spans.append((prop, len(rows), len(embs)))
                rows.extend((prop, e) for e in embs)
            return rows, spans

        def _ht_mat(rows, mi):
            mat = np.full((len(rows), N_LAYERS), np.nan)
            for r, (prop, emb) in enumerate(rows):
                for c, layer in enumerate(LAYERS):
                    mat[r, c] = data_all[emb][layer][mi].get(prop, np.nan)
            return mat

        def _ht_draw(ax, mat, rows, spans, cmap, vmin, vmax, title):
            n_rows, n_cols = mat.shape
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
            ax.set_xticks(range(n_cols)); ax.set_xticklabels(XLABELS, rotation=35, ha="right", fontsize=7)
            ax.set_yticks(range(n_rows)); ax.set_yticklabels([_EMB_SHORT[e] for _, e in rows], fontsize=7)
            ax.set_title(title, fontsize=10, pad=14)
            for r in range(n_rows):
                for c in range(n_cols):
                    v = mat[r, c]
                    ax.text(c, r, f"{v:.2f}" if not np.isnan(v) else "–", ha="center", va="center",
                            fontsize=5.5, color="black" if not np.isnan(v) else "#aaaaaa")
            trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            for prop, start, count in spans:
                if start > 0:
                    ax.axhline(start - 0.5, color="#555555", lw=0.9, ls="--", zorder=2)
                ax.text(-0.12, start + (count - 1) / 2, prop, transform=trans, ha="right",
                        va="center", fontsize=7, color="#222222", fontweight="bold")
            ax.set_xlim(-0.5, n_cols - 0.5); ax.set_ylim(n_rows - 0.5, -0.5)

        def _save_ht(rows, spans, mat, cmap, vmin, vmax, title, out_path):
            if not rows:
                return
            fig = plt.figure(figsize=(16, 1.6 + 0.42 * max(len(rows), 1)))
            ax = fig.add_subplot(111)
            _ht_draw(ax, mat, rows, spans, cmap, vmin, vmax, title)
            fig.subplots_adjust(left=0.28, right=0.93, top=0.94, bottom=0.06)
            fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"Saved -> {out_path}")

        kr, ks = _ht_rows(_KNN_ALL, 0)
        lr, ls = _ht_rows(EDGE_LVR_PROPS, 1)
        _save_ht(kr, ks, _ht_mat(kr, 0), KNN_BG_CMAP, -0.1, 1.0,
                 "UMA edges — k-NN Purity (normalised)  [higher = better]", knn_table)
        _save_ht(lr, ls, _ht_mat(lr, 1), LVR_BG_CMAP, 0.0, 1.0,
                 "UMA edges — Local Variance Ratio (LVR)  [lower = better]", lvr_table)

    # ---- metrics summary (text) ----
    summary_txt = global_dir / "metrics_summary_edges.txt"
    if not summary_txt.exists():
        col_w = 10
        buf = io.StringIO()
        buf.write("FairChem UMA-S Edge Embedding Analysis — Metrics Summary\n")
        buf.write("=" * 70 + "\n\n")
        for ek in ALL_EDGE_KEYS:
            buf.write(f"{'=' * 70}\n{EMB_LABELS[ek]}\n{'=' * 70}\n")
            buf.write(f"  {'property':<28}" + "".join(f"{LAYER_LABELS[l]:>{col_w}}" for l in LAYERS) + "\n")
            buf.write("  " + "-" * (28 + col_w * N_LAYERS) + "\n")
            buf.write("  k-NN purity (normalised)\n")
            for prop_name in KNN_PROPS[ek]:
                row = f"  {prop_name:<28}"
                for layer in LAYERS:
                    v = data_all[ek][layer][0].get(prop_name, np.nan)
                    row += f"{v:>{col_w}.3f}" if not np.isnan(v) else f"{'—':>{col_w}}"
                buf.write(row + "\n")
            buf.write("\n  LVR (lower = tighter)\n")
            for prop_name in EDGE_LVR_PROPS:
                if not any(not np.isnan(data_all[ek][l][1].get(prop_name, np.nan)) for l in LAYERS):
                    continue
                row = f"  {prop_name:<28}"
                for layer in LAYERS:
                    v = data_all[ek][layer][1].get(prop_name, np.nan)
                    row += f"{v:>{col_w}.3f}" if not np.isnan(v) else f"{'—':>{col_w}}"
                buf.write(row + "\n")
            buf.write("\n")
        summary_txt.write_text(buf.getvalue())
        print(f"Saved -> {summary_txt}")


# ---------------------------------------------------------------------------
# True edge census (full graph) — parallel one-time pass
# ---------------------------------------------------------------------------

def _census_one(p_str: str):
    """Worker: TRUE bond count + oxygen/carbon/homoatomic families for one material.

    UMA edge .pt files persist only a ≤48/material edge sample but record the real
    graph size in `n_edges_total` (this exactly matches ORB's full edge count, i.e.
    both models share the same graph). Use n_edges_total for bonds/mat so the
    dataset-structure stat reflects the true graph, not the storage cap. Families
    are computed from the stored sample (UMA QC does not display them)."""
    import torch as _t
    from pathlib import Path as _P
    d = _t.load(p_str, weights_only=False)
    bt = d["bond_type"]
    n_true = int(d.get("n_edges_total", len(bt)))
    ox = ca = ho = 0
    for b in bt:
        a1, a2 = b.split("-") if "-" in b else (b, b)
        if "O" in (a1, a2): ox += 1
        if "C" in (a1, a2): ca += 1
        if a1 == a2:        ho += 1
    return (_P(p_str).stem, n_true, ox, ca, ho)


def build_edge_census(census_path: Path, emb_dir: Path, layers: list[str],
                      valid_ids: set, n_workers: int = 16) -> None:
    """Write per-material TRUE bond counts + families to census_path (guarded).

    bonds/mat and family stats must reflect the full graph, not the sqrt-weighted
    UMAP subsample. bond_type is identical across layers, so one layer suffices.
    """
    if census_path.exists():
        return
    edge_dir = next((emb_dir / l / "edges" for l in layers
                     if (emb_dir / l / "edges").exists()), None)
    if edge_dir is None:
        print("  Edge census: no edge directory found — skipping", flush=True)
        return
    pts = [str(p) for p in sorted(edge_dir.glob("*.pt")) if p.stem in valid_ids]
    print(f"--- Edge census: counting true bonds for {len(pts):,} materials "
          f"({n_workers} workers) ---", flush=True)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        rows = pool.map(_census_one, pts, chunksize=64)
    ids = np.array([r[0] for r in rows])
    nb  = np.array([r[1] for r in rows], dtype=np.int64)
    no  = np.array([r[2] for r in rows], dtype=np.int64)
    nc  = np.array([r[3] for r in rows], dtype=np.int64)
    nh  = np.array([r[4] for r in rows], dtype=np.int64)
    np.savez(census_path, mp_ids=ids, n_bonds=nb, n_oxy=no, n_carbon=nc, n_homo=nh)
    print(f"  Edge census saved -> {census_path}  ({len(ids):,} materials, "
          f"{int(nb.sum()):,} total bonds, {time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA, UMAP, k-NN purity and LVR for all FairChem UMA layers"
    )
    parser.add_argument("--embeddings-dir", default=str(P.RESULTS / "fairchem"))
    parser.add_argument("--plots-dir",      default=str(P.RESULTS / "fairchem/plots"))
    parser.add_argument("--structures-dir", default=str(P.STRUCTURES))
    parser.add_argument("--umap-sample",    type=int, default=NODE_UMAP_N_DEFAULT,
                        help="Max atoms for per-atom UMAP subsample, sqrt-weighted by element "
                             "(default %(default)s)")
    parser.add_argument("--num-workers",    type=int, default=12,
                        help="1 = sequential (debug); 12 = one per embedding type (default)")
    args = parser.parse_args()

    emb_dir  = Path(args.embeddings_dir)
    plot_dir = Path(args.plots_dir)
    str_dir  = Path(args.structures_dir)
    node_umap_n = args.umap_sample

    nodes_dir  = plot_dir / "nodes"
    edges_dir  = plot_dir / "edges"
    global_dir = plot_dir / "global"
    bb_dir     = plot_dir / "backbone"
    for d in [nodes_dir, edges_dir, global_dir, bb_dir]:
        d.mkdir(parents=True, exist_ok=True)

    jsonl_path = str_dir / "index.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} not found.")

    # -------------------------------------------------------------------------
    # Calibrate K for pooled workers from backbone node files
    # -------------------------------------------------------------------------
    print("--- Calibrating adaptive K for pooled embeddings ---")
    structs, valid_ids = _load_manifest(jsonl_path)
    print(f"Manifest: {len(structs):,} materials")

    bb_node_files = sorted((emb_dir / "backbone" / "nodes").glob("*.pt"))
    n_mat_bb    = sum(1 for pt in bb_node_files if pt.stem in valid_ids)
    sqrt_k      = int(round(n_mat_bb ** 0.5))
    cap_mat     = n_mat_bb // 10
    K_pool      = max(10, min(sqrt_k, cap_mat))
    binding_mat = "sqrt(n)" if sqrt_k <= cap_mat else "10% cap"
    print(f"  K_pool = {K_pool}  ({n_mat_bb:,} materials, {binding_mat})")

    # -------------------------------------------------------------------------
    # True edge census (full graph) — so QC reports real bonds/mat and edge
    # coverage, not the sqrt-weighted UMAP subsample.
    # -------------------------------------------------------------------------
    build_edge_census(edges_dir / "edge_census.npz", emb_dir, EDGE_LAYERS,
                      valid_ids, n_workers=16)

    # -------------------------------------------------------------------------
    # Spawn / run workers
    # -------------------------------------------------------------------------
    worker_args = [
        (_worker_nodes_flat,        (0,  emb_dir, nodes_dir, jsonl_path, node_umap_n)),
        (_worker_nodes_ang_norm,    (1,  emb_dir, nodes_dir, jsonl_path, node_umap_n)),
        (_worker_nodes_l0,          (2,  emb_dir, nodes_dir, jsonl_path, node_umap_n)),
        (_worker_nodes_l1_norm,     (3,  emb_dir, nodes_dir, jsonl_path, node_umap_n)),
        (_worker_nodes_l2_norm,     (4,  emb_dir, nodes_dir, jsonl_path, node_umap_n)),
        (_worker_nodes_geom_norm,   (5,  emb_dir, nodes_dir, jsonl_path, node_umap_n)),
        (_worker_pooled_full,       (6,  emb_dir, nodes_dir, jsonl_path, K_pool, binding_mat)),
        (_worker_pooled_l0,         (7,  emb_dir, nodes_dir, jsonl_path, K_pool, binding_mat)),
        (_worker_pooled_ang_norm,   (8,  emb_dir, nodes_dir, jsonl_path, K_pool, binding_mat)),
        (_worker_pooled_l1_polar,   (9,  emb_dir, nodes_dir, jsonl_path, K_pool, binding_mat)),
        (_worker_pooled_l2_polar,   (10, emb_dir, nodes_dir, jsonl_path, K_pool, binding_mat)),
        (_worker_pooled_geom_norm,  (11, emb_dir, nodes_dir, jsonl_path, K_pool, binding_mat)),
    ]
    # Edge workers (per-edge + pooled-edge), one per edge embedding type.
    for _i, _ek in enumerate(ALL_EDGE_KEYS):
        worker_args.append(
            (_edge_entry, (12 + _i, _ek, emb_dir, edges_dir, jsonl_path, node_umap_n)))

    if args.num_workers == 1:
        print("\nRunning sequentially (--num-workers 1) ...")
        q: _queue.SimpleQueue = _queue.SimpleQueue()
        import threading

        def _drain(q, n):
            done = 0
            while done < n:
                msg = q.get()
                if msg.startswith("__DONE__"):
                    done += 1
                    print(f"  Worker {msg[8:]} finished  ({done}/{n} done)", flush=True)
                else:
                    print(msg, flush=True)

        drain_t = threading.Thread(target=_drain, args=(q, len(worker_args)), daemon=True)
        drain_t.start()
        for fn, a in worker_args:
            fn(*a, q)
        drain_t.join()
    else:
        print(f"\nSpawning {len(worker_args)} workers (one per embedding type) ...")
        ctx = mp.get_context("spawn")
        mp_q: mp.Queue = ctx.Queue()
        processes = []
        for fn, a in worker_args:
            p = ctx.Process(target=fn, args=(*a, mp_q), daemon=True)
            p.start()
            processes.append(p)
        _listen(mp_q, len(processes))
        for p in processes:
            p.join()

    # =========================================================================
    # Cross-layer summary figures
    # =========================================================================

    print("\n\n========== CROSS-LAYER SUMMARY ==========")

    XLABELS = [LAYER_LABELS[l] for l in ALL_LAYERS]
    X_POS   = list(range(len(ALL_LAYERS)))

    # Load all parsed knn/lvr data
    data_all: dict[str, dict[str, tuple[dict, dict, dict]]] = {}
    for emb_key in ALL_EMB_KEYS:
        data_all[emb_key] = {}
        for layer in ALL_LAYERS:
            txt = nodes_dir / f"{layer}_{emb_key}_knn_lvr.txt"
            data_all[emb_key][layer] = parse_knn_lvr_txt(txt)

    EMB_PANELS = [(k, EMB_LABELS[k]) for k in ALL_EMB_KEYS]

    KNN_BG_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "knn_bg", ["#d3d3d3", "#fff3b0", "#aed6f1", "#a9dfbf"])
    LVR_BG_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "lvr_bg", ["#a9dfbf", "#aed6f1", "#fff3b0", "#d3d3d3"])
    ZONE_ALPHA = 0.45

    def _knn_panel(ax, emb_key, emb_label, Y_MIN, Y_MAX):
        grad = np.linspace(Y_MIN, Y_MAX, 256).reshape(-1, 1)
        ax.imshow(grad, aspect="auto", cmap=KNN_BG_CMAP, vmin=Y_MIN, vmax=1.0,
                  extent=[-0.3, len(ALL_LAYERS) - 0.7, Y_MIN, Y_MAX],
                  origin="lower", zorder=0, alpha=ZONE_ALPHA)
        ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=2)
        for prop_name in KNN_PROPS[emb_key]:
            y = np.array([data_all[emb_key][layer][0].get(prop_name, np.nan)
                          for layer in ALL_LAYERS], dtype=float)
            c_p = [data_all[emb_key][layer][2].get(prop_name, np.nan) for layer in ALL_LAYERS]
            avg_cov = float(np.nanmean(c_p)) if not np.all(np.isnan(c_p)) else np.nan
            label = (f"{prop_name}  ({avg_cov:.0%} cov)" if not np.isnan(avg_cov) else prop_name)
            mask = ~np.isnan(y)
            if mask.any():
                ax.plot(np.array(X_POS)[mask], y[mask], "o-",
                        color=KNN_COLOR_MAP.get(prop_name, "#888888"),
                        lw=2.2, ms=5, zorder=3, label=label)
        ax.set_title(emb_label, fontsize=10)
        ax.set_xticks(X_POS)
        ax.set_xticklabels(XLABELS, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("Normalised purity")
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_xlim(-0.3, len(ALL_LAYERS) - 0.7)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.25, zorder=1)

    _all_knn_vals = [
        data_all[_et][_ly][0].get(_pn, np.nan)
        for _et in ALL_EMB_KEYS
        for _ly in ALL_LAYERS
        for _pn in KNN_PROPS[_et]
    ]
    _knn_data_min = float(np.nanmin(_all_knn_vals)) if any(not np.isnan(v) for v in _all_knn_vals) else -0.10
    Y_MIN = min(-0.10, np.floor(_knn_data_min * 10) / 10 - 0.05)
    Y_MAX = 1.05

    knn_fig = global_dir / "knn_purity_across_layers.png"
    if not knn_fig.exists():
        # Row 0 = per-atom (6 cols), Row 1 = pooled (6 cols)
        _n_cols = max(len(PER_ATOM_KEYS), len(POOLED_KEYS))
        fig1, axes1 = plt.subplots(2, _n_cols, figsize=(6 * _n_cols, 5 * 2), sharey=False)
        fig1.suptitle("k-NN Purity (normalised) across FairChem UMA layers",
                      fontsize=14, y=1.01)
        for ci, emb_key in enumerate(PER_ATOM_KEYS):
            _knn_panel(axes1[0][ci], emb_key, EMB_LABELS[emb_key], Y_MIN, Y_MAX)
        for ci in range(len(PER_ATOM_KEYS), _n_cols):
            axes1[0][ci].set_visible(False)
        for ci, emb_key in enumerate(POOLED_KEYS):
            _knn_panel(axes1[1][ci], emb_key, EMB_LABELS[emb_key], Y_MIN, Y_MAX)
        for ci in range(len(POOLED_KEYS), _n_cols):
            axes1[1][ci].set_visible(False)
        plt.tight_layout()
        plt.savefig(knn_fig, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {knn_fig}")

    def _lvr_panel(ax, emb_key, emb_label):
        all_vals = [np.array([data_all[emb_key][layer][1].get(p, np.nan)
                              for layer in ALL_LAYERS], dtype=float)
                    for p in LVR_CROSS_PROPS]
        y_max  = max((np.nanmax(v) for v in all_vals if np.any(~np.isnan(v))), default=1.0)
        y_ceil = max(y_max * 1.08, 0.90)
        grad   = np.linspace(0, y_ceil, 256).reshape(-1, 1)
        ax.imshow(grad, aspect="auto", cmap=LVR_BG_CMAP, vmin=0.0, vmax=1.0,
                  extent=[-0.3, len(ALL_LAYERS) - 0.7, 0, y_ceil],
                  origin="lower", zorder=0, alpha=ZONE_ALPHA)
        for i, prop_name in enumerate(LVR_CROSS_PROPS):
            y = np.array([data_all[emb_key][layer][1].get(prop_name, np.nan)
                          for layer in ALL_LAYERS], dtype=float)
            c_p = [data_all[emb_key][layer][2].get(prop_name, np.nan) for layer in ALL_LAYERS]
            avg_cov = float(np.nanmean(c_p)) if not np.all(np.isnan(c_p)) else np.nan
            label = (f"{prop_name}  ({avg_cov:.0%} cov)" if not np.isnan(avg_cov) else prop_name)
            mask = ~np.isnan(y)
            if mask.any():
                ax.plot(np.array(X_POS)[mask], y[mask], "o-",
                        color=LVR_COLORS[i % len(LVR_COLORS)],
                        lw=2.2, ms=5, zorder=3, label=label)
        ax.set_title(emb_label, fontsize=10)
        ax.set_xticks(X_POS)
        ax.set_xticklabels(XLABELS, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("LVR")
        ax.set_ylim(0, y_ceil)
        ax.set_xlim(-0.3, len(ALL_LAYERS) - 0.7)
        ax.legend(fontsize=7, loc="upper right", ncol=1)
        ax.grid(axis="y", alpha=0.25, zorder=1)

    lvr_fig = global_dir / "lvr_across_layers.png"
    if not lvr_fig.exists():
        # Row 0 = per-atom (6 cols), Row 1 = pooled (6 cols)
        _n_cols = max(len(PER_ATOM_KEYS), len(POOLED_KEYS))
        fig2, axes2 = plt.subplots(2, _n_cols, figsize=(6 * _n_cols, 6 * 2), sharey=False)
        fig2.suptitle("Local Variance Ratio (LVR) across FairChem UMA layers\n"
                      "(lower = tighter clustering of the property)",
                      fontsize=14, y=1.02)
        for ci, emb_key in enumerate(PER_ATOM_KEYS):
            _lvr_panel(axes2[0][ci], emb_key, EMB_LABELS[emb_key])
        for ci in range(len(PER_ATOM_KEYS), _n_cols):
            axes2[0][ci].set_visible(False)
        for ci, emb_key in enumerate(POOLED_KEYS):
            _lvr_panel(axes2[1][ci], emb_key, EMB_LABELS[emb_key])
        for ci in range(len(POOLED_KEYS), _n_cols):
            axes2[1][ci].set_visible(False)
        plt.tight_layout()
        plt.savefig(lvr_fig, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {lvr_fig}")

    # ── Representation-type comparison: per-property subplots, lines = emb types ──
    # Colors and line styles shared across atom and pooled comparison plots
    _EMB_COLORS = {
        "nodes_flat":              "#1f77b4",
        "nodes_ang_norm":          "#ff7f0e",
        "nodes_l0":                "#2ca02c",
        "nodes_l1_norm":           "#d62728",
        "nodes_l2_norm":           "#9467bd",
        "nodes_geom_norm":         "#8c564b",
        "nodes_pooled_full":       "#1f77b4",
        "nodes_pooled_ang_norm":   "#ff7f0e",
        "nodes_pooled_l0":         "#2ca02c",
        "nodes_pooled_l1_polar":   "#e377c2",
        "nodes_pooled_l2_polar":   "#7f7f7f",
        "nodes_pooled_geom_norm":  "#bcbd22",
    }
    _EMB_LS = {
        "nodes_flat":              "-",
        "nodes_ang_norm":          "--",
        "nodes_l0":                ":",
        "nodes_l1_norm":           "-.",
        "nodes_l2_norm":           (0, (3, 1, 1, 1)),
        "nodes_geom_norm":         (0, (5, 1)),
        "nodes_pooled_full":       "-",
        "nodes_pooled_ang_norm":   "--",
        "nodes_pooled_l0":         ":",
        "nodes_pooled_l1_polar":   "-.",
        "nodes_pooled_l2_polar":   (0, (3, 1, 1, 1)),
        "nodes_pooled_geom_norm":  (0, (5, 1)),
    }
    _DIM_LBL = {k: f"({EMB_DIM[k]}d)" for k in ALL_EMB_KEYS}

    def _rep_cmp_knn(out_path, emb_keys, props, title):
        if out_path.exists():
            return
        n = len(props)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
        if n == 1:
            axes = [axes]
        _all_vals = [data_all[ek][ly][0].get(p, np.nan)
                     for ek in emb_keys for ly in ALL_LAYERS for p in props]
        _ymin = min(-0.10, np.floor(np.nanmin(_all_vals) * 10) / 10 - 0.05
                    if not np.all(np.isnan(_all_vals)) else -0.10)
        _ymax = 1.05
        for ax, prop_name in zip(axes, props):
            grad = np.linspace(_ymin, _ymax, 256).reshape(-1, 1)
            ax.imshow(grad, aspect="auto", cmap=KNN_BG_CMAP, vmin=_ymin, vmax=1.0,
                      extent=[-0.3, len(ALL_LAYERS) - 0.7, _ymin, _ymax],
                      origin="lower", zorder=0, alpha=ZONE_ALPHA)
            ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=2)
            for emb_key in emb_keys:
                y = np.array([data_all[emb_key][layer][0].get(prop_name, np.nan)
                              for layer in ALL_LAYERS], dtype=float)
                mask = ~np.isnan(y)
                if mask.any():
                    ax.plot(np.array(X_POS)[mask], y[mask],
                            color=_EMB_COLORS[emb_key], ls="-",
                            marker="o", ms=5, lw=2.2, zorder=3,
                            label=f"{EMB_LABELS[emb_key]} {_DIM_LBL[emb_key]}")
            ax.set_title(prop_name, fontsize=10)
            ax.set_xticks(X_POS)
            ax.set_xticklabels(XLABELS, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel("Normalised purity")
            ax.set_ylim(_ymin, _ymax)
            ax.set_xlim(-0.3, len(ALL_LAYERS) - 0.7)
            ax.grid(axis="y", alpha=0.25, zorder=1)
        handles, labels = axes[0].get_legend_handles_labels()
        ncol = min(len(handles), max(3, len(handles) // 2))
        fig.legend(handles, labels, loc="upper center", fontsize=8, framealpha=0.9,
                   ncol=ncol, bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure)
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {out_path}")

    def _rep_cmp_lvr(out_path, emb_keys, title):
        if out_path.exists():
            return
        ncols = 3
        nrows = (len(LVR_CROSS_PROPS) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False)
        fig.suptitle(f"{title}\n(lower LVR = tighter clustering = better)", fontsize=12, y=0.995)
        _ax_flat = list(axes.flat)
        for idx, prop_name in enumerate(LVR_CROSS_PROPS):
            ax = _ax_flat[idx]
            _vals = [data_all[ek][ly][1].get(prop_name, np.nan)
                     for ek in emb_keys for ly in ALL_LAYERS]
            _ymax_p = max(np.nanmax(_vals) * 1.08 if not np.all(np.isnan(_vals)) else 1.0, 0.90)
            grad = np.linspace(0, _ymax_p, 256).reshape(-1, 1)
            ax.imshow(grad, aspect="auto", cmap=LVR_BG_CMAP, vmin=0.0, vmax=1.0,
                      extent=[-0.3, len(ALL_LAYERS) - 0.7, 0, _ymax_p],
                      origin="lower", zorder=0, alpha=ZONE_ALPHA)
            for emb_key in emb_keys:
                y = np.array([data_all[emb_key][layer][1].get(prop_name, np.nan)
                              for layer in ALL_LAYERS], dtype=float)
                mask = ~np.isnan(y)
                if mask.any():
                    ax.plot(np.array(X_POS)[mask], y[mask],
                            color=_EMB_COLORS[emb_key], ls="-",
                            marker="o", ms=5, lw=2.2, zorder=3,
                            label=f"{EMB_LABELS[emb_key]} {_DIM_LBL[emb_key]}")
            ax.set_title(prop_name, fontsize=9)
            ax.set_xticks(X_POS)
            ax.set_xticklabels(XLABELS, rotation=35, ha="right", fontsize=7)
            ax.set_ylabel("LVR")
            ax.set_ylim(0, _ymax_p)
            ax.set_xlim(-0.3, len(ALL_LAYERS) - 0.7)
            ax.grid(axis="y", alpha=0.25, zorder=1)
        for idx in range(len(LVR_CROSS_PROPS), len(_ax_flat)):
            _ax_flat[idx].set_visible(False)
        handles, labels = _ax_flat[0].get_legend_handles_labels()
        ncol = min(len(handles), max(3, len(handles) // 2))
        # Legend sits just below the title (not overlapping it); the plot area
        # extends up to 0.90 so there's no large blank strip before the panels.
        fig.legend(handles, labels, loc="upper center", fontsize=8, framealpha=0.9,
                   ncol=ncol, bbox_to_anchor=(0.5, 0.945),
                   bbox_transform=fig.transFigure)
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {out_path}")

    _atom_knn_props = ["crystal_system", "metal_insulator", "magnetic_label", "element_identity"]
    _pool_knn_props = ["crystal_system", "metal_insulator", "magnetic_label"]

    _rep_cmp_knn(global_dir / "knn_rep_comparison_atom.png",
                 PER_ATOM_KEYS, _atom_knn_props,
                 "k-NN Purity — nodes representations vs layer"
                 "  (higher = better)")
    _rep_cmp_knn(global_dir / "knn_rep_comparison_pooled.png",
                 POOLED_KEYS, _pool_knn_props,
                 "k-NN Purity — nodes pooled representations vs layer"
                 "  (higher = better)")
    _rep_cmp_lvr(global_dir / "lvr_rep_comparison_atom.png",
                 PER_ATOM_KEYS,
                 "LVR — nodes representations vs layer")
    _rep_cmp_lvr(global_dir / "lvr_rep_comparison_pooled.png",
                 POOLED_KEYS,
                 "LVR — nodes pooled representations vs layer")

    # PCA node-quality: cumulative explained variance, layers × embedding types.
    pca_quality_grid(nodes_dir, PER_ATOM_KEYS, "nodes",
                     global_dir / "pca_node_quality_grid_atom.png")
    pca_quality_grid(nodes_dir, POOLED_KEYS, "nodes pooled",
                     global_dir / "pca_node_quality_grid_pooled.png")

    summary_txt = global_dir / "metrics_summary.txt"
    if not summary_txt.exists():
        col_w = 10
        buf = io.StringIO()
        buf.write("FairChem UMA-S All-Layer Embedding Analysis — Metrics Summary\n")
        buf.write("=" * 70 + "\n\n")
        for emb_key, emb_label in EMB_PANELS:
            buf.write(f"{'=' * 70}\n{emb_label}\n{'=' * 70}\n")
            header = f"  {'property':<28}" + "".join(
                f"{LAYER_LABELS[l]:>{col_w}}" for l in ALL_LAYERS)
            buf.write(header + "\n")
            buf.write("  " + "-" * (28 + col_w * len(ALL_LAYERS)) + "\n")
            buf.write("  k-NN purity (normalised)\n")
            for prop_name in KNN_PROPS[emb_key]:
                row = f"  {prop_name:<28}"
                for layer in ALL_LAYERS:
                    v = data_all[emb_key][layer][0].get(prop_name, np.nan)
                    row += f"{v:>{col_w}.3f}" if not np.isnan(v) else f"{'—':>{col_w}}"
                buf.write(row + "\n")
            buf.write("\n  LVR (lower = tighter)\n")
            for prop_name in LVR_CROSS_PROPS:
                has_any = any(
                    not np.isnan(data_all[emb_key][layer][1].get(prop_name, np.nan))
                    for layer in ALL_LAYERS
                )
                if not has_any:
                    continue
                row = f"  {prop_name:<28}"
                for layer in ALL_LAYERS:
                    v = data_all[emb_key][layer][1].get(prop_name, np.nan)
                    row += f"{v:>{col_w}.3f}" if not np.isnan(v) else f"{'—':>{col_w}}"
                buf.write(row + "\n")
            buf.write("\n")
        summary_txt.write_text(buf.getvalue())
        print(f"Saved -> {summary_txt}")

    # ── Summary heatmaps (kNN and LVR as separate figures) ──────────────────────
    knn_table = global_dir / "knn_summary_table.png"
    lvr_table = global_dir / "lvr_summary_table.png"
    if not (knn_table.exists() and lvr_table.exists()):
        _KNN_ALL_PROPS = ["crystal_system", "metal_insulator", "magnetic_label", "element_identity"]
        _LVR_ALL_PROPS = LVR_CROSS_PROPS
        _EMB_SHORT = {k: EMB_LABELS[k].replace("Nodes ", "").replace("Pooled ", "pool ")
                      for k in ALL_EMB_KEYS}
        _KNN_HT_CMAP = KNN_BG_CMAP
        _LVR_HT_CMAP = LVR_BG_CMAP

        def _ht_build_rows(props, metric_idx):
            rows, spans = [], []
            for prop in props:
                embs = [e for e in ALL_EMB_KEYS
                        if any(not np.isnan(data_all[e][l][metric_idx].get(prop, np.nan))
                               for l in ALL_LAYERS)]
                if not embs:
                    continue
                spans.append((prop, len(rows), len(embs)))
                for emb in embs:
                    rows.append((prop, emb))
            return rows, spans

        def _ht_build_matrix(rows, metric_idx):
            mat = np.full((len(rows), len(ALL_LAYERS)), np.nan)
            for r, (prop, emb) in enumerate(rows):
                for c, layer in enumerate(ALL_LAYERS):
                    mat[r, c] = data_all[emb][layer][metric_idx].get(prop, np.nan)
            return mat

        def _ht_draw(ax, mat, rows, spans, cmap, vmin, vmax, title):
            n_rows, n_cols = mat.shape
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels([LAYER_LABELS[l] for l in ALL_LAYERS],
                               rotation=35, ha="right", fontsize=7)
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels([_EMB_SHORT[emb] for _, emb in rows], fontsize=7)
            ax.set_title(title, fontsize=10, pad=14)
            for r in range(n_rows):
                for c in range(n_cols):
                    v = mat[r, c]
                    ax.text(c, r, f"{v:.2f}" if not np.isnan(v) else "–",
                            ha="center", va="center", fontsize=5.5,
                            color="black" if not np.isnan(v) else "#aaaaaa")
            trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            for prop, start, count in spans:
                if start > 0:
                    ax.axhline(start - 0.5, color="#555555", lw=0.9, ls="--", zorder=2)
                ax.text(-0.12, start + (count - 1) / 2, prop,
                        transform=trans, ha="right", va="center",
                        fontsize=7, color="#222222", fontweight="bold")
            ax.set_xlim(-0.5, n_cols - 0.5)
            ax.set_ylim(n_rows - 0.5, -0.5)

        knn_rows, knn_spans = _ht_build_rows(_KNN_ALL_PROPS, 0)
        lvr_rows, lvr_spans = _ht_build_rows(_LVR_ALL_PROPS, 1)
        knn_mat = _ht_build_matrix(knn_rows, 0)
        lvr_mat = _ht_build_matrix(lvr_rows, 1)

        def _save_one(rows, spans, mat, cmap, vmin, vmax, title, out_path):
            if not rows:
                return
            n = max(len(rows), 1)
            fig = plt.figure(figsize=(18, 1.6 + 0.42 * n))
            ax = fig.add_subplot(111)
            _ht_draw(ax, mat, rows, spans, cmap, vmin, vmax, title)
            fig.subplots_adjust(left=0.28, right=0.93, top=0.94, bottom=0.06)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved -> {out_path}")

        _save_one(knn_rows, knn_spans, knn_mat, _KNN_HT_CMAP, -0.1, 1.0,
                  "FairChem UMA-S — k-NN Purity (normalised)  [higher = better]", knn_table)
        _save_one(lvr_rows, lvr_spans, lvr_mat, _LVR_HT_CMAP, 0.0, 1.0,
                  "FairChem UMA-S — Local Variance Ratio (LVR)  [lower = better]", lvr_table)

    # =========================================================================
    # UMAP Summary Plots (6 layers × 6 embedding types)
    # Split into per-atom (6×3) and pooled (6×3) figures per property
    # =========================================================================

    _ATOM_KEYS   = PER_ATOM_KEYS
    _POOLED_KEYS = POOLED_KEYS
    _ATOM_LBLS   = [EMB_LABELS[k] for k in _ATOM_KEYS]
    _POOL_LBLS   = [EMB_LABELS[k] for k in _POOLED_KEYS]
    _LAYER_LBLS  = [LAYER_LABELS[l] for l in ALL_LAYERS]
    _SUPTITLE_Y  = 0.99
    N_LAYERS     = len(ALL_LAYERS)

    def _cache_path(layer, emb_key):
        return nodes_dir / f"{layer}_{emb_key}_umap_cache.npz"

    def _load_c(layer, emb_key):
        p = _cache_path(layer, emb_key)
        return np.load(p, allow_pickle=True) if p.exists() else None

    def _draw_na(ax):
        ax.set_facecolor("#eeeeee")
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#888888")
        ax.set_xticks([]); ax.set_yticks([])

    def _grid_atom(title):
        n_cols = len(_ATOM_KEYS)
        fig, axes = plt.subplots(N_LAYERS, n_cols, figsize=(4 * n_cols, 4 * N_LAYERS))
        fig.suptitle(title, fontsize=13, y=_SUPTITLE_Y)
        for ci, lbl in enumerate(_ATOM_LBLS):
            axes[0][ci].set_title(lbl, fontsize=9, pad=6)
        for ri, lbl in enumerate(_LAYER_LBLS):
            axes[ri][0].set_ylabel(lbl, fontsize=9, labelpad=5)
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        return fig, axes

    def _grid_pooled(title):
        n_cols = len(_POOLED_KEYS)
        fig, axes = plt.subplots(N_LAYERS, n_cols, figsize=(4 * n_cols, 4 * N_LAYERS))
        fig.suptitle(title, fontsize=13, y=_SUPTITLE_Y)
        for ci, lbl in enumerate(_POOL_LBLS):
            axes[0][ci].set_title(lbl, fontsize=9, pad=6)
        for ri, lbl in enumerate(_LAYER_LBLS):
            axes[ri][0].set_ylabel(lbl, fontsize=9, labelpad=5)
        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        return fig, axes

    def _get_mp_ids(c):
        return c["mp_ids"].astype(str) if "mp_ids" in c.files else np.array([], dtype=str)

    def _v(s, k):
        v = s.get(k)
        return float(v) if v is not None else np.nan

    def _v_pos(s, k):
        v = s.get(k)
        if v is None: return np.nan
        v = float(v)
        return v if v > 0 else np.nan

    # Helper: render one crystal-system UMAP grid (atom or pooled types)
    def _plot_cs_grid(fig, axes, emb_keys):
        for ri, layer in enumerate(ALL_LAYERS):
            for ci, emb_key in enumerate(emb_keys):
                ax = axes[ri][ci]
                c  = _load_c(layer, emb_key)
                if c is None:
                    _draw_na(ax); continue
                X  = c["X_umap"]
                cs = (c["labels"].astype(str) if emb_key in PER_ATOM_KEYS
                      else np.array([structs.get(m, {}).get("crystal_system", "Unknown")
                                     for m in _get_mp_ids(c)]))
                no_data = ~np.isin(cs, CRYSTAL_SYSTEMS)
                if no_data.any():
                    ax.scatter(X[no_data, 0], X[no_data, 1], c="#dddddd",
                               s=1, alpha=0.1, linewidths=0, rasterized=True)
                for _cs in CRYSTAL_SYSTEMS:
                    mask = cs == _cs
                    if mask.any():
                        ax.scatter(X[mask, 0], X[mask, 1], c=[CRYSTAL_COLORS[_cs]],
                                   s=1, alpha=0.1, linewidths=0, rasterized=True)
        handles = [mlines.Line2D([], [], marker="o", linestyle="None",
                                 color=CRYSTAL_COLORS[cs], markersize=6, label=cs)
                   for cs in CRYSTAL_SYSTEMS]
        handles.append(mlines.Line2D([], [], marker="o", linestyle="None",
                                     color="#dddddd", markersize=6, label="no data"))
        plt.tight_layout(rect=[0, 0, 1.0, 0.94])
        _p0 = axes[0][0].get_position()
        fig.legend(handles=handles, title="Crystal system", loc="center",
                   ncol=4, fontsize=8, title_fontsize=9, framealpha=0.92,
                   bbox_to_anchor=(0.5, (_p0.y1 + _SUPTITLE_Y) / 2),
                   bbox_transform=fig.transFigure)

    # --- Crystal system ---
    _cs_atom = global_dir / "umap_crystal_system_atom.png"
    if not _cs_atom.exists():
        fig, axes = _grid_atom("UMAP — Crystal System  (nodes)")
        _plot_cs_grid(fig, axes, _ATOM_KEYS)
        plt.savefig(_cs_atom, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved -> {_cs_atom}")

    _cs_pool = global_dir / "umap_crystal_system_pooled.png"
    if not _cs_pool.exists():
        fig, axes = _grid_pooled("UMAP — Crystal System  (nodes pooled)")
        _plot_cs_grid(fig, axes, _POOLED_KEYS)
        plt.savefig(_cs_pool, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved -> {_cs_pool}")

    # Helper: binary categorical scatter (metal/insulator, magnetic)
    def _plot_binary_grid(fig, axes, emb_keys, get_labels_fn, colors, labels_txt, no_data_color="#dddddd"):
        for ri, layer in enumerate(ALL_LAYERS):
            for ci, emb_key in enumerate(emb_keys):
                ax = axes[ri][ci]
                c  = _load_c(layer, emb_key)
                if c is None:
                    _draw_na(ax); continue
                X      = c["X_umap"]
                mp_ids = _get_mp_ids(c)
                cats   = get_labels_fn(mp_ids)
                no_data = np.array([v is None for v in cats])
                if no_data.any():
                    ax.scatter(X[no_data, 0], X[no_data, 1], c=no_data_color,
                               s=1, alpha=0.1, linewidths=0, rasterized=True)
                for cat, color in colors.items():
                    mask = np.array([v == cat for v in cats])
                    if mask.any():
                        ax.scatter(X[mask, 0], X[mask, 1], c=color,
                                   s=1, alpha=0.1, linewidths=0, rasterized=True)
        handles = [mlines.Line2D([], [], marker="o", linestyle="None",
                                 color=c, markersize=6, label=l)
                   for l, c in labels_txt]
        plt.tight_layout(rect=[0, 0, 1.0, 0.94])
        _p0 = axes[0][0].get_position()
        fig.legend(handles=handles, loc="center",
                   ncol=len(labels_txt), fontsize=8, framealpha=0.9,
                   bbox_to_anchor=(0.5, (_p0.y1 + _SUPTITLE_Y) / 2),
                   bbox_transform=fig.transFigure)

    # --- Metal / Insulator ---
    def _mi_labels(mp_ids):
        bg = np.array([structs.get(m, {}).get("band_gap") for m in mp_ids])
        return [None if v is None else ("Metal" if v == 0 else "Insulator") for v in bg]

    _mi_colors = {"Metal": "teal", "Insulator": "goldenrod"}
    _mi_txt    = [("metal", "teal"), ("insulator", "goldenrod"), ("no data", "#dddddd")]

    for suffix, keys in [("atom", _ATOM_KEYS), ("pooled", _POOLED_KEYS)]:
        _out = global_dir / f"umap_metal_insulator_{suffix}.png"
        if not _out.exists():
            fig, axes = (_grid_atom if suffix == "atom" else _grid_pooled)(
                f"UMAP — Metal / Insulator  ({'nodes' if suffix == 'atom' else 'nodes pooled'})")
            _plot_binary_grid(fig, axes, keys, _mi_labels, _mi_colors, _mi_txt)
            plt.savefig(_out, dpi=150, bbox_inches="tight"); plt.close()
            print(f"Saved -> {_out}")

    # --- Magnetic label ---
    def _mag_labels(mp_ids):
        return [_mag_label(structs, m) for m in mp_ids]

    _mag_colors = {"Non-magnetic": "turquoise", "Magnetic": "crimson"}
    _mag_txt    = [("non-magnetic", "turquoise"), ("magnetic", "crimson"), ("no data", "#dddddd")]

    for suffix, keys in [("atom", _ATOM_KEYS), ("pooled", _POOLED_KEYS)]:
        _out = global_dir / f"umap_magnetic_label_{suffix}.png"
        if not _out.exists():
            fig, axes = (_grid_atom if suffix == "atom" else _grid_pooled)(
                f"UMAP — Magnetic Label  ({'nodes' if suffix == 'atom' else 'nodes pooled'})")
            _plot_binary_grid(fig, axes, keys, _mag_labels, _mag_colors, _mag_txt)
            plt.savefig(_out, dpi=150, bbox_inches="tight"); plt.close()
            print(f"Saved -> {_out}")

    # --- Element identity (per-atom only) ---
    _elem_out = global_dir / "umap_element_identity.png"
    if not _elem_out.exists():
        fig, axes = _grid_atom("UMAP — Element Identity  (nodes, coloured by atomic number Z)")
        _SYMBOL_TO_Z = {
            "H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,"Ne":10,
            "Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,"Ar":18,
            "K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,"Fe":26,
            "Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,"As":33,"Se":34,
            "Br":35,"Kr":36,"Rb":37,"Sr":38,"Y":39,"Zr":40,"Nb":41,"Mo":42,
            "Tc":43,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"In":49,"Sn":50,
            "Sb":51,"Te":52,"I":53,"Xe":54,"Cs":55,"Ba":56,"La":57,"Ce":58,
            "Pr":59,"Nd":60,"Pm":61,"Sm":62,"Eu":63,"Gd":64,"Tb":65,"Dy":66,
            "Ho":67,"Er":68,"Tm":69,"Yb":70,"Lu":71,"Hf":72,"Ta":73,"W":74,
            "Re":75,"Os":76,"Ir":77,"Pt":78,"Au":79,"Hg":80,"Tl":81,"Pb":82,
            "Bi":83,"Po":84,"At":85,"Rn":86,"Fr":87,"Ra":88,"Ac":89,"Th":90,
            "Pa":91,"U":92,
        }
        sc_last = None
        for ri, layer in enumerate(ALL_LAYERS):
            for ci, emb_key in enumerate(_ATOM_KEYS):
                ax = axes[ri][ci]
                c  = _load_c(layer, emb_key)
                if c is None:
                    _draw_na(ax); continue
                X     = c["X_umap"]
                elems = c["elements"].astype(str)
                Z     = np.array([_SYMBOL_TO_Z.get(e, 0) for e in elems])
                sc_last = ax.scatter(X[:, 0], X[:, 1], c=Z, cmap="gist_ncar",
                                     s=1, alpha=0.1, linewidths=0, rasterized=True)
                ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout(rect=[0, 0, 1.0, 0.94])
        if sc_last is not None:
            _p0 = axes[0][0].get_position()
            _cbar_w, _cbar_h = 0.35, 0.007
            _cx = 0.5 - _cbar_w / 2
            _cy = (_p0.y1 + _SUPTITLE_Y) / 2
            cax = fig.add_axes([_cx, _cy - _cbar_h / 2, _cbar_w, _cbar_h])
            cb  = fig.colorbar(sc_last, cax=cax, orientation="horizontal")
            cb.solids.set_alpha(1)
            cax.set_title("Atomic number Z", fontsize=8, pad=3)
        plt.savefig(_elem_out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved -> {_elem_out}")

    # --- Continuous property summaries ---
    def _cont_grid(out_path_atom, out_path_pool, title, get_val_fn, cmap, norm, cbar_label):
        for out_path, emb_keys, grid_fn, suffix in [
            (out_path_atom, _ATOM_KEYS,   _grid_atom,   "nodes"),
            (out_path_pool, _POOLED_KEYS, _grid_pooled, "nodes pooled"),
        ]:
            if out_path.exists():
                continue
            fig, axes = grid_fn(f"{title}  ({suffix})")
            sc_last = None
            n_miss = n_tot = 0
            for ri, layer in enumerate(ALL_LAYERS):
                for ci, emb_key in enumerate(emb_keys):
                    ax = axes[ri][ci]
                    c  = _load_c(layer, emb_key)
                    if c is None:
                        _draw_na(ax); continue
                    mp_ids = _get_mp_ids(c)
                    vals   = np.array([get_val_fn(structs.get(m, {})) for m in mp_ids], dtype=float)
                    has    = ~np.isnan(vals)
                    n_miss += int((~has).sum()); n_tot += has.size
                    if (~has).any():
                        ax.scatter(c["X_umap"][~has, 0], c["X_umap"][~has, 1], c="lightgrey",
                                   s=1, alpha=0.1, linewidths=0, rasterized=True)
                    if has.any():
                        sc_last = ax.scatter(c["X_umap"][has, 0], c["X_umap"][has, 1],
                                             c=vals[has], cmap=cmap, norm=norm,
                                             s=1, alpha=0.1, linewidths=0, rasterized=True)
            # Legend for the grey "missing data" colour + overall missing fraction.
            if n_miss > 0:
                miss_pct = 100.0 * n_miss / max(n_tot, 1)
                fig.legend(handles=[mlines.Line2D([], [], marker="o", linestyle="None",
                                                  color="lightgrey", markersize=6,
                                                  label=f"missing data ({miss_pct:.1f}%)")],
                           loc="upper left", bbox_to_anchor=(0.01, 0.99),
                           fontsize=8, framealpha=0.9, bbox_transform=fig.transFigure)
            plt.tight_layout(rect=[0, 0, 1.0, 0.94])
            _p0 = axes[0][0].get_position()
            _cbar_h, _cbar_w = 0.007, 0.35
            _cbar_cy = (_p0.y1 + _SUPTITLE_Y) / 2
            cax = fig.add_axes([0.5 - _cbar_w / 2, _cbar_cy - _cbar_h / 2, _cbar_w, _cbar_h])
            _cb_src = sc_last if sc_last is not None else plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            if sc_last is None:
                _cb_src.set_array([])
            cb = fig.colorbar(_cb_src, cax=cax, orientation="horizontal")
            if sc_last is not None:
                cb.solids.set_alpha(1)
            cax.set_title(cbar_label, fontsize=9, pad=3)
            cb.ax.tick_params(labelsize=7)
            plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
            print(f"Saved -> {out_path}")

    # Compute per-property value ranges
    _fe_all  = np.array([v for m in structs for v in [structs[m].get("formation_energy_per_atom")]
                          if v is not None], dtype=float)
    _fe_vmin = float(np.percentile(_fe_all, 1))  if len(_fe_all) else -3.0
    _fe_vmax = float(np.percentile(_fe_all, 99)) if len(_fe_all) else  3.0
    _cont_grid(global_dir / "umap_formation_energy_atom.png",
               global_dir / "umap_formation_energy_pooled.png",
               "UMAP — Formation Energy / Atom",
               lambda s: _v(s, "formation_energy_per_atom"),
               "coolwarm", mcolors.Normalize(vmin=_fe_vmin, vmax=_fe_vmax),
               "Formation energy (eV/atom)")

    _cont_grid(global_dir / "umap_energy_above_hull_atom.png",
               global_dir / "umap_energy_above_hull_pooled.png",
               "UMAP — Energy Above Hull",
               lambda s: _v(s, "energy_above_hull"),
               "coolwarm", mcolors.PowerNorm(gamma=0.4, vmin=0.0, vmax=1.0),
               "Energy above hull (eV/atom)  [sqrt-ish, 0–1 eV]")

    # Band gap
    for suffix, keys, grid_fn in [("atom", _ATOM_KEYS, _grid_atom), ("pooled", _POOLED_KEYS, _grid_pooled)]:
        _bg_out = global_dir / f"umap_band_gap_{suffix}.png"
        if not _bg_out.exists():
            _norm_bg = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=5.0)
            fig, axes = grid_fn(f"UMAP — Band Gap  ({'nodes' if suffix == 'atom' else 'nodes pooled'})")
            sc_last = None
            for ri, layer in enumerate(ALL_LAYERS):
                for ci, emb_key in enumerate(keys):
                    ax = axes[ri][ci]
                    c  = _load_c(layer, emb_key)
                    if c is None:
                        _draw_na(ax); continue
                    mp_ids = _get_mp_ids(c)
                    bg     = np.array([structs.get(m, {}).get("band_gap") for m in mp_ids])
                    has    = np.array([v is not None for v in bg])
                    bg_f   = np.array([v if v is not None else np.nan for v in bg], dtype=float)
                    is_metal    = has & (bg_f == 0.0)
                    is_nonmetal = has & (bg_f  > 0.0)
                    no_data     = ~has
                    if no_data.any():
                        ax.scatter(c["X_umap"][no_data, 0], c["X_umap"][no_data, 1], c="lightgrey",
                                   s=1, alpha=0.1, linewidths=0, rasterized=True)
                    ax.scatter(c["X_umap"][is_metal, 0], c["X_umap"][is_metal, 1],
                               c="aquamarine", s=1, alpha=0.1, linewidths=0, rasterized=True)
                    if is_nonmetal.any():
                        sc_last = ax.scatter(c["X_umap"][is_nonmetal, 0], c["X_umap"][is_nonmetal, 1],
                                             c=bg_f[is_nonmetal], cmap="plasma",
                                             norm=_norm_bg, s=1, alpha=0.1, linewidths=0, rasterized=True)
            plt.tight_layout(rect=[0, 0, 1.0, 0.94])
            _p0 = axes[0][0].get_position(); _cy = (_p0.y1 + _SUPTITLE_Y) / 2
            if sc_last is not None:
                cax = fig.add_axes([0.5 - 0.15, _cy - 0.0035, 0.30, 0.007])
                cb  = fig.colorbar(sc_last, cax=cax, orientation="horizontal")
                cb.solids.set_alpha(1)
                cax.set_title("Band gap (eV)  [sqrt scale, 0–5 eV]", fontsize=8, pad=3)
                cb.ax.tick_params(labelsize=7)
                fig.legend(handles=[mpatches.Patch(facecolor="aquamarine", label="Metal (bg=0)")],
                           loc="center left", fontsize=8, framealpha=0,
                           bbox_to_anchor=(0.5 + 0.15 + 0.02, _cy),
                           bbox_transform=fig.transFigure)
            plt.savefig(_bg_out, dpi=150, bbox_inches="tight"); plt.close()
            print(f"Saved -> {_bg_out}")

    # Remaining continuous properties
    _cont_grid(global_dir / "umap_density_atom.png",
               global_dir / "umap_density_pooled.png",
               "UMAP — Density",
               lambda s: _v(s, "density"),
               "viridis", mcolors.PowerNorm(gamma=0.6, vmin=0, vmax=20.0),
               "Density (g/cm³)  [sqrt-ish, 0–20 g/cm³]")

    _cont_grid(global_dir / "umap_bulk_modulus_atom.png",
               global_dir / "umap_bulk_modulus_pooled.png",
               "UMAP — Bulk Modulus",
               lambda s: _v(s, "bulk_modulus"),
               "viridis", mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=440.0),
               "Bulk modulus (GPa)  [sqrt scale, 0–440 GPa]")

    # Edge global figures: same set of global plots as nodes, for per-edge and
    # pooled-edge representations (new `*_edges*` files; node outputs untouched).
    emit_edge_global_figures(edges_dir, global_dir, bb_dir, structs)

    # ── Backbone (final layer) bar charts ─────────────────────────────────────
    print("\n\n========== BACKBONE SUMMARY ==========")

    _BB_KNN_PROPS = [
        ("crystal_system",   "crystal system",   "#4E9F8A"),
        ("metal_insulator",  "metal/insulator",  "#E17C05"),
        ("magnetic_label",   "magnetic label",   "#A06CD5"),
        ("element_identity", "element identity", "#D65F5F"),
    ]
    _BB_LVR_PROPS = [
        ("formation_energy/atom", "form. energy",   "#DD8452"),
        ("energy_above_hull",     "above hull",     "#C44E52"),
        ("band_gap",              "band gap",       "#CCB974"),
        ("mag_density (uB/A3)",   "mag density",    "#937860"),
        ("density (g/cm3)",       "density",        "#DA8BC3"),
        ("bulk_modulus (GPa)",    "bulk modulus",   "#8C8C8C"),
    ]
    _BB_EMB_LABELS_SHORT = [EMB_LABELS[k].replace("Nodes ", "").replace("Pooled ", "pool\n")
                            for k in ALL_EMB_KEYS]
    _x_bb = np.arange(len(ALL_EMB_KEYS))

    bb_knn_fig = bb_dir / "backbone_knn_purity.png"
    if not bb_knn_fig.exists():
        fig, ax = plt.subplots(figsize=(13, 4))
        w = 0.16
        legend_done = set()
        for pi, (prop, lbl, col) in enumerate(_BB_KNN_PROPS):
            for xi, emb_key in enumerate(ALL_EMB_KEYS):
                val = data_all[emb_key]["backbone"][0].get(prop, np.nan)
                if np.isnan(val):
                    continue
                valid_props = [p for p, _, _ in _BB_KNN_PROPS
                               if not np.isnan(data_all[emb_key]["backbone"][0].get(p, np.nan))]
                n_v = len(valid_props)
                si  = valid_props.index(prop) if prop in valid_props else pi
                off = (si - (n_v - 1) / 2) * w
                ax.bar(_x_bb[xi] + off, val, w,
                       label=lbl if prop not in legend_done else "_nolegend_",
                       color=col, alpha=0.85)
                legend_done.add(prop)
        ax.set_xticks(_x_bb); ax.set_xticklabels(_BB_EMB_LABELS_SHORT, fontsize=9)
        ax.axhline(0,   color="gray",  lw=0.8, ls="--", alpha=0.6)
        ax.axhline(0.5, color="green", lw=0.8, ls=":",  alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("k-NN purity (normalised)")
        ax.set_title("Backbone — k-NN purity (normalised)  [higher = better]", fontsize=12)
        ax.legend(title="Property", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
        plt.tight_layout()
        plt.savefig(bb_knn_fig, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved -> {bb_knn_fig}")

    bb_lvr_fig = bb_dir / "backbone_lvr.png"
    if not bb_lvr_fig.exists():
        fig, ax = plt.subplots(figsize=(14, 4))
        n_lvr = len(_BB_LVR_PROPS)
        w = 0.08
        for pi, (prop, lbl, col) in enumerate(_BB_LVR_PROPS):
            vals = [data_all[emb_key]["backbone"][1].get(prop, np.nan)
                    for emb_key in ALL_EMB_KEYS]
            off  = (pi - (n_lvr - 1) / 2) * w
            ax.bar(_x_bb + off, vals, w, label=lbl, color=col, alpha=0.85)
        ax.set_xticks(_x_bb); ax.set_xticklabels(_BB_EMB_LABELS_SHORT, fontsize=9)
        ax.axhline(0.40, color="green",  lw=0.8, ls=":", alpha=0.5)
        ax.axhline(0.65, color="orange", lw=0.8, ls=":", alpha=0.5)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("LVR")
        ax.set_title("Backbone — LVR  [lower = better]", fontsize=12)
        ax.legend(title="Property", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(bb_lvr_fig, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved -> {bb_lvr_fig}")

    bb_heat_fig = bb_dir / "backbone_summary_table.png"
    if not bb_heat_fig.exists():
        _HT_PUR_PROPS  = ["crystal_system", "metal_insulator", "magnetic_label", "element_identity"]
        _HT_LVR_PROPS  = [p for p, _ in LVR_MAT_PROPS] + ["mag_density (uB/A3)"]
        _HT_PUR_LABELS = _HT_PUR_PROPS
        _HT_LVR_LABELS = [p.replace(" (g/cm3)", "").replace(" (GPa)", "") for p in _HT_LVR_PROPS]
        _pur_mat = np.array([[data_all[et]["backbone"][0].get(p, np.nan)
                              for p in _HT_PUR_PROPS] for et in ALL_EMB_KEYS])
        _lvr_mat = np.array([[data_all[et]["backbone"][1].get(p, np.nan)
                              for p in _HT_LVR_PROPS] for et in ALL_EMB_KEYS])
        _KNN_BG = mcolors.LinearSegmentedColormap.from_list(
            "knn_bg", ["#d3d3d3", "#fff3b0", "#aed6f1", "#a9dfbf"])
        _LVR_BG = mcolors.LinearSegmentedColormap.from_list(
            "lvr_bg", ["#a9dfbf", "#aed6f1", "#fff3b0", "#d3d3d3"])
        _n_pur = len(_HT_PUR_PROPS); _n_lvr_h = len(_HT_LVR_PROPS)
        fig, (_ax_ph, _ax_lh) = plt.subplots(
            1, 2, figsize=(18, max(_n_pur, _n_lvr_h) * 0.6 + 2.2),
            gridspec_kw={"width_ratios": [_n_pur, _n_lvr_h]}
        )
        def _annotate(ax, mat):
            for ri in range(mat.shape[0]):
                for ci in range(mat.shape[1]):
                    v = mat[ri, ci]
                    if not np.isnan(v):
                        ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                                fontsize=7.5, color="black")
        _im_ph = _ax_ph.imshow(_pur_mat.T, aspect="auto", cmap=_KNN_BG,
                                vmin=-0.1, vmax=1.0, interpolation="nearest")
        _ax_ph.set_xticks(range(len(ALL_EMB_KEYS)))
        _ax_ph.set_xticklabels(_BB_EMB_LABELS_SHORT, fontsize=8, rotation=20, ha="right")
        _ax_ph.set_yticks(range(_n_pur))
        _ax_ph.set_yticklabels(_HT_PUR_LABELS, fontsize=8)
        _ax_ph.set_title("k-NN purity (normalised)  [higher = better]", fontsize=10)
        _annotate(_ax_ph, _pur_mat.T)
        fig.colorbar(_im_ph, ax=_ax_ph, shrink=0.6, pad=0.02).ax.tick_params(labelsize=7)
        _im_lh = _ax_lh.imshow(_lvr_mat.T, aspect="auto", cmap=_LVR_BG,
                                vmin=0.0, vmax=1.0, interpolation="nearest")
        _ax_lh.set_xticks(range(len(ALL_EMB_KEYS)))
        _ax_lh.set_xticklabels(_BB_EMB_LABELS_SHORT, fontsize=8, rotation=20, ha="right")
        _ax_lh.set_yticks(range(_n_lvr_h))
        _ax_lh.set_yticklabels(_HT_LVR_LABELS, fontsize=8)
        _ax_lh.set_title("LVR  [lower = better]", fontsize=10)
        _annotate(_ax_lh, _lvr_mat.T)
        fig.colorbar(_im_lh, ax=_ax_lh, shrink=0.6, pad=0.02).ax.tick_params(labelsize=7)
        fig.suptitle("Backbone — metrics summary  (FairChem UMA-S)", fontsize=12)
        plt.tight_layout()
        plt.savefig(bb_heat_fig, dpi=150, bbox_inches="tight"); plt.close()
        print(f"Saved -> {bb_heat_fig}")

    bb_txt = bb_dir / "backbone_metrics_summary.txt"
    if not bb_txt.exists():
        buf = io.StringIO()
        buf.write("FairChem UMA-S Backbone Metrics Summary\n" + "=" * 50 + "\n\n")
        for emb_key in ALL_EMB_KEYS:
            knn_d, lvr_d, _cov_d = data_all[emb_key]["backbone"]
            buf.write(f"\n--- {EMB_LABELS[emb_key]} ---\n")
            for k in KNN_PROPS[emb_key]:
                v = knn_d.get(k, float("nan"))
                buf.write(f"  purity_{k:<30}  {v:.4f}\n" if not np.isnan(v)
                          else f"  purity_{k:<30}  n/a\n")
            for k in LVR_CROSS_PROPS:
                v = lvr_d.get(k, float("nan"))
                buf.write(f"  lvr_{k:<33}  {v:.4f}\n" if not np.isnan(v)
                          else f"  lvr_{k:<33}  n/a\n")
        bb_txt.write_text(buf.getvalue())
        print(f"Saved -> {bb_txt}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
