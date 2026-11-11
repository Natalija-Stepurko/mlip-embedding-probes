"""
stages/geometry/organisation_orb.py — PCA, UMAP, k-NN purity, LVR for all ORB layers

Reads embeddings from $MLIP_PROBES_ROOT/results/orb/{layer}/{nodes|edges}/*.pt
Writes analysis outputs to $MLIP_PROBES_ROOT/results/orb/plots/{nodes|edges|global}/

Output naming: {layer}_{emb_type}_{content}.{ext}
  e.g. gnn_stacks_4_nodes_umap_cache.npz
       encoder_edges_knn_lvr.txt
       gnn_stacks_4_nodes_pooled_pca_quality.png

Embedding types:
  nodes        — per-atom node vectors [N_atoms, 256] per material
  nodes_pooled — mean-pooled node → one [256] per material
  edges        — per-bond vectors [E, 256], L2-normalised, sqrt-weighted subsample for UMAP
  edges_pooled — mean-pooled edge → one [256] per material

Parallelism:
  4 workers run simultaneously, one per embedding type (nodes / nodes_pooled /
  edges / edges_pooled). Each worker processes all 6 layers sequentially.
  Pass --num-workers 1 to run everything sequentially (useful for local testing).

Resume: each output file is checked before writing — already-present files are skipped.
        The UMAP cache (.npz) is the most expensive checkpoint.

Memory notes:
  nodes        — loads all atom embeddings per layer into RAM
                 (~4 GB float32 for 200k materials × ~20 atoms avg)
  nodes_pooled — one 256-D vector per material (~200 MB, trivial)
  edges        — two-pass loading: pass 1 collects bond_type/distance metadata,
                 pass 2 loads only the subsampled edges
  edges_pooled — one 256-D vector per material (~200 MB, trivial)

Usage:
    uv run python stages/geometry/organisation_orb.py

    uv run python stages/geometry/organisation_orb.py \\
        --embeddings-dir $MLIP_PROBES_ROOT/results/orb \\
        --plots-dir      $MLIP_PROBES_ROOT/results/orb/plots \\
        --structures-dir $MLIP_PROBES_ROOT/structures

    # Local test (1 worker, small UMAP sample)
    uv run python stages/geometry/organisation_orb.py \\
        --embeddings-dir results_test/orb \\
        --plots-dir      results_test/orb/plots \\
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
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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

# The ORB "backbone" layer is essentially identical to gnn_stacks_4, so it is
# excluded from all analysis. gnn_stacks_4 is the final layer everywhere that
# used to key off "backbone" (K calibration, reference caches, final-layer
# highlight figures).
ALL_LAYERS = [
    "encoder",
    "gnn_stacks_0",
    "gnn_stacks_1",
    "gnn_stacks_2",
    "gnn_stacks_3",
    "gnn_stacks_4",
]
FINAL_LAYER = ALL_LAYERS[-1]  # "gnn_stacks_4"

LAYER_LABELS = {
    "encoder":      "encoder",
    "gnn_stacks_0": "gnn_0",
    "gnn_stacks_1": "gnn_1",
    "gnn_stacks_2": "gnn_2",
    "gnn_stacks_3": "gnn_3",
    "gnn_stacks_4": "gnn_4",
}

PCA_VARIANCE_THRESHOLD = 0.90
UMAP_N_DEFAULT      = 300_000  # edge subsample target
NODE_UMAP_N_DEFAULT = 500_000  # atom subsample target (sqrt-weighted by element)
UMAP_NN_EDGE        = 50
UMAP_NN_NODE        = 15
MAG_THR        = 1e-4

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
    "nodes":        ["crystal_system", "metal_insulator", "magnetic_label", "element_identity"],
    "nodes_pooled": ["crystal_system", "metal_insulator", "magnetic_label"],
    "edges":        ["bond_type", "crystal_system", "metal_insulator", "magnetic_label"],
    "edges_pooled": ["crystal_system", "metal_insulator", "magnetic_label"],
    "concat_pooled": ["crystal_system", "metal_insulator", "magnetic_label"],
}

LVR_MAT_PROPS = [
    ("formation_energy/atom", "formation_energy_per_atom"),
    ("energy_above_hull",     "energy_above_hull"),
    ("band_gap",              "band_gap"),
    ("density (g/cm3)",       "density"),
    ("bulk_modulus (GPa)",    "bulk_modulus"),
]

KNN_COLOR_MAP = {
    "crystal_system":   "#4E9F8A",
    "metal_insulator":  "#E17C05",
    "magnetic_label":   "#A06CD5",
    "element_identity": "steelblue",
    "bond_type":        "#2E86C1",
}

LVR_COLORS = [
    "#DD8452", "#C44E52", "#CCB974", "#937860", "#DA8BC3",
    "#8C8C8C", "#64B5CD", "#55A868", "#8172B2", "#2196F3",
]

LVR_CROSS_PROPS = [
    "formation_energy/atom", "energy_above_hull", "band_gap",
    "mag_density (uB/A3)", "density (g/cm3)", "bulk_modulus (GPa)",
    "bond_distance (A)",
]

_MACRO_PROPS = {"bond_type"}


# ---------------------------------------------------------------------------
# Helpers — PCA / UMAP
# ---------------------------------------------------------------------------

def pca_n_for_variance(evr: np.ndarray,
                        threshold: float = PCA_VARIANCE_THRESHOLD) -> int:
    return int((np.cumsum(evr) < threshold).sum()) + 1


def fit_pca(X: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    # Drop any rows containing NaN or inf (e.g. materials with zero edges whose
    # mean-pooled embedding is NaN).  Callers must account for the reduced row
    # count if they need to align mp_ids — use drop_nan_rows() for that.
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
    cumvar = np.cumsum(evr) * 100
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"ORB v3  {emb_label}  [{LAYER_LABELS[layer]}]  — PCA + Norm Quality",
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


def scatter_continuous(ax, X_umap: np.ndarray, values, cmap: str = "plasma",
                        s: int = 3, alpha: float = 0.6, label: str = "") -> None:
    vals = np.asarray([v if v is not None else np.nan for v in values], dtype=float)
    nan_mask = np.isnan(vals)
    if nan_mask.any():
        ax.scatter(X_umap[nan_mask, 0], X_umap[nan_mask, 1],
                   c="#dddddd", s=s, alpha=0.6, linewidths=0, rasterized=True)
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


def lvr_per_edge(name: str, values: np.ndarray, nn_idx: np.ndarray,
                  X: np.ndarray, K: int, build_nn_fn) -> str:
    v    = np.asarray(values, dtype=float)
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


# ---------------------------------------------------------------------------
# Parse k-NN / LVR txt files (for cross-layer figures — runs in main)
# ---------------------------------------------------------------------------

def parse_knn_lvr_txt(path: Path) -> tuple[dict, dict, dict]:
    knn, lvr, cov = {}, {}, {}
    if not path.exists():
        return knn, lvr, cov
    text    = path.read_text()
    section = None
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
# Manifest helpers — module-level so workers can call them after loading structs
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
# Worker — nodes
# ---------------------------------------------------------------------------

def _worker_nodes(worker_id: int, emb_dir: Path, nodes_dir: Path,
                   jsonl_path: Path, node_umap_n: int, progress_q) -> None:
    matplotlib.use("Agg")

    def log(msg: str) -> None:
        progress_q.put(f"[nodes   ] {msg}")

    structs, valid_ids = _load_manifest(jsonl_path)

    # K_node is calibrated from the final-layer (gnn_4) node subsample (same
    # approach as K_e for edges). Layers processed before it defer their k-NN txt to a
    # second pass after K_node is known.
    K_node: int | None = None
    binding_node: str  = ""

    def build_nn(X: np.ndarray) -> np.ndarray:
        assert K_node is not None
        return (NearestNeighbors(n_neighbors=K_node + 1, metric="euclidean", n_jobs=1)
                .fit(X).kneighbors(X, return_distance=False)[:, 1:])

    for layer in ALL_LAYERS:
        t0 = time.time()
        node_dir_l = emb_dir / layer / "nodes"
        log(f"--- {layer} ---")

        cache_path   = nodes_dir / f"{layer}_nodes_umap_cache.npz"
        quality_path = nodes_dir / f"{layer}_nodes_pca_quality.png"

        if not cache_path.exists():
            pt_files = sorted(node_dir_l.glob("*.pt"))

            # Pass 1: collect per-atom element + crystal_system metadata.
            # The node tensor is loaded by torch.load but not kept — it goes out
            # of scope at each loop iteration, so peak RAM is one file at a time.
            log(f"  Pass 1: collecting node metadata ...")
            meta_elements:    list[str] = []
            meta_mids:        list[str] = []
            meta_cs:          list[str] = []
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

            # Sqrt-weighted subsample by element (mirrors edge bond-type strategy).
            # Each element gets quota proportional to sqrt(count); floor at 20.
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
            log(f"  Node subsample: {len(umap_idx_arr):,} atoms "
                f"({len(umap_idx_arr)/total_atoms:.1%} of total, "
                f"{len(elem_counts)} elements)")

            # Build cumulative offsets so global idx → (material, local idx)
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

            # Pass 2: load only the selected atom embeddings
            log(f"  Pass 2: loading subsample embeddings ...")
            gi_to_pos = {int(gi): pos for pos, gi in enumerate(umap_idx_arr)}
            X_n = np.zeros((len(umap_idx_arr), 256), dtype=np.float32)
            for pt in pt_files:
                if pt.stem not in idx_by_mat:
                    continue
                d   = torch.load(pt, weights_only=False)
                emb = d["node"].float().numpy()
                base = mat_offsets[pt.stem]
                for local_idx in idx_by_mat[pt.stem]:
                    X_n[gi_to_pos[base + local_idx]] = emb[local_idx]

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
            log(f"  Loading node cache ...")
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

        # Calibrate K_node once from the final-layer (gnn_4) node subsample —
        # mirrors how K_e is calibrated from the final-layer edge subsample.
        if layer == FINAL_LAYER and K_node is None:
            apm_sub  = np.array(list(Counter(mp_ids_n.tolist()).values()), dtype=float)
            p75_sub  = float(np.percentile(apm_sub, 75))
            k_p75_n  = int(p75_sub) + 1
            k_cap_n  = len(X_n) // 10
            K_node   = max(10, min(k_p75_n, k_cap_n))
            binding_node = "p75+1 (subsample)" if k_p75_n <= k_cap_n else "10% cap (subsample)"
            log(f"  K_node = {K_node}  (subsample {len(X_n):,} atoms, "
                f"p75={p75_sub:.1f} atoms/material, {binding_node})")

        if not quality_path.exists():
            quality_figure(layer, "nodes", evr_n, pca_n_n, norms_n, quality_path,
                           "mediumpurple", "Atom count",
                           "L2 norm of 256-D node embedding")
            log(f"  Saved -> {quality_path}")

        knn_path = nodes_dir / f"{layer}_nodes_knn_lvr.txt"
        if not knn_path.exists() and K_node is not None:
            log(f"  Computing nodes k-NN purity + LVR ...")
            norms_l2 = np.linalg.norm(X_n, axis=1, keepdims=True)
            X_l2     = X_n / np.where(norms_l2 == 0, 1, norms_l2)
            nn_n     = build_nn(X_l2)
            buf = io.StringIO()
            buf.write(f"=== Node Embeddings -- {layer} ===\n")
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

    # Deferred k-NN for layers that ran before the final layer (K_node was None
    # then). gnn_4 is last in ALL_LAYERS so this only triggers if its cache
    # already existed on a prior run and K_node was never set during the loop.
    if K_node is None:
        c_n      = np.load(nodes_dir / f"{FINAL_LAYER}_nodes_umap_cache.npz", allow_pickle=True)
        apm_sub  = np.array(list(Counter(c_n["mp_ids"].tolist()).values()), dtype=float)
        p75_sub  = float(np.percentile(apm_sub, 75))
        k_p75_n  = int(p75_sub) + 1
        k_cap_n  = len(c_n["X_raw"]) // 10
        K_node   = max(10, min(k_p75_n, k_cap_n))
        binding_node = "p75+1 (subsample)" if k_p75_n <= k_cap_n else "10% cap (subsample)"
        log(f"\nK_node (from {FINAL_LAYER} cache): {K_node}  ({binding_node})")

    for layer in ALL_LAYERS:
        knn_path = nodes_dir / f"{layer}_nodes_knn_lvr.txt"
        if knn_path.exists():
            continue
        log(f"\n  [{layer}] nodes k-NN + LVR (deferred) ...")
        c_n        = np.load(nodes_dir / f"{layer}_nodes_umap_cache.npz", allow_pickle=True)
        X_n        = c_n["X_raw"]
        labels_n   = c_n["labels"]
        mp_ids_n   = c_n["mp_ids"]
        elements_n = c_n["elements"]
        total_atoms = int(c_n["n_total"][0]) if "n_total" in c_n else len(X_n)
        norms_l2   = np.linalg.norm(X_n, axis=1, keepdims=True)
        X_l2       = X_n / np.where(norms_l2 == 0, 1, norms_l2)
        nn_n       = build_nn(X_l2)
        buf = io.StringIO()
        buf.write(f"=== Node Embeddings -- {layer} ===\n")
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

    progress_q.put(f"__DONE__{worker_id}")


# ---------------------------------------------------------------------------
# Worker — nodes_pooled
# ---------------------------------------------------------------------------

def _worker_nodes_pooled(worker_id: int, emb_dir: Path, nodes_dir: Path,
                           jsonl_path: Path, K_np: int, binding_mat: str,
                           progress_q) -> None:
    matplotlib.use("Agg")

    def log(msg: str) -> None:
        progress_q.put(f"[nodes_p ] {msg}")

    structs, valid_ids = _load_manifest(jsonl_path)

    def build_nn(X: np.ndarray) -> np.ndarray:
        return (NearestNeighbors(n_neighbors=K_np + 1, metric="euclidean", n_jobs=1)
                .fit(X).kneighbors(X, return_distance=False)[:, 1:])

    for layer in ALL_LAYERS:
        t0 = time.time()
        node_dir_l = emb_dir / layer / "nodes"
        log(f"--- {layer} ---")

        cache_path   = nodes_dir / f"{layer}_nodes_pooled_umap_cache.npz"
        quality_path = nodes_dir / f"{layer}_nodes_pooled_pca_quality.png"

        if not cache_path.exists():
            log(f"  Loading nodes_pooled embeddings ...")
            pt_files = sorted(node_dir_l.glob("*.pt"))
            struct_embs, mp_ids_np = [], []
            for pt in pt_files:
                if pt.stem not in valid_ids:
                    continue
                d = torch.load(pt, weights_only=False)
                struct_embs.append(d["structure"].float().numpy())
                mp_ids_np.append(pt.stem)
            X_np      = np.stack(struct_embs, axis=0)
            mp_ids_np = np.array(mp_ids_np)
            # Drop materials whose structure embedding is NaN/inf
            _fin_np = np.isfinite(X_np).all(axis=1)
            if not _fin_np.all():
                log(f"  WARNING: dropping {(~_fin_np).sum()} materials with NaN/inf embeddings")
                X_np      = X_np[_fin_np]
                mp_ids_np = mp_ids_np[_fin_np]
            norms_np  = np.linalg.norm(X_np, axis=1)
            log(f"  {len(X_np):,} materials")

            evr_np, pca_n_np, X_pca_np = fit_pca(X_np)
            log(f"  PCA: {pca_n_np} components ({evr_np[:pca_n_np].sum():.1%})")

            log(f"  Running UMAP ({len(X_pca_np):,} materials) ...")
            X_umap_np = run_umap(X_pca_np, UMAP_NN_NODE)

            np.savez(cache_path,
                     X_umap=X_umap_np, X_pca=X_pca_np, X_raw=X_np,
                     mp_ids=mp_ids_np, evr=evr_np)
            log(f"  Saved cache -> {cache_path}")
        else:
            c         = np.load(cache_path, allow_pickle=True)
            X_np      = c["X_raw"]
            X_umap_np = c["X_umap"]
            mp_ids_np = c["mp_ids"]
            evr_np    = c["evr"]
            pca_n_np  = pca_n_for_variance(evr_np)
            norms_np  = np.linalg.norm(X_np, axis=1)

        if not quality_path.exists():
            quality_figure(layer, "nodes_pooled", evr_np, pca_n_np, norms_np, quality_path,
                           "steelblue", "Material count",
                           "L2 norm of 256-D nodes_pooled embedding")
            log(f"  Saved -> {quality_path}")

        knn_path = nodes_dir / f"{layer}_nodes_pooled_knn_lvr.txt"
        if not knn_path.exists():
            log(f"  Computing nodes_pooled k-NN purity + LVR ...")
            nn_np = build_nn(X_np)
            buf = io.StringIO()
            buf.write(f"=== nodes_pooled Embeddings -- {layer} ===\n")
            buf.write(f"Materials: {len(X_np):,}\n")
            buf.write(f"K={K_np}  ({binding_mat})\n\n")
            buf.write("-- Categorical (k-NN purity: micro / macro) --\n")
            buf.write(knn_purity_line("crystal_system",
                [structs[m].get("crystal_system") for m in mp_ids_np], nn_np, X_np, build_nn))
            buf.write(knn_purity_line("metal_insulator",
                [_metal_insulator(structs, m) for m in mp_ids_np], nn_np, X_np, build_nn))
            buf.write(knn_purity_line("magnetic_label",
                [_mag_label(structs, m) for m in mp_ids_np], nn_np, X_np, build_nn))
            buf.write("\n-- Continuous (LVR) --\n")
            md = _mag_density_dict(structs)
            for name, key in LVR_MAT_PROPS:
                buf.write(lvr_material(name, _prop_dict(structs, key), mp_ids_np,
                                       nn_np, X_np, K_np, build_nn))
            buf.write(lvr_material("mag_density (uB/A3)", md, mp_ids_np,
                                   nn_np, X_np, K_np, build_nn))
            knn_path.write_text(buf.getvalue())
            log(f"  Saved -> {knn_path}")

        log(f"  [{layer}] done in {time.time() - t0:.0f}s")

    progress_q.put(f"__DONE__{worker_id}")


# ---------------------------------------------------------------------------
# Worker — edges  (K_e calibrated internally from the final-layer cache)
# ---------------------------------------------------------------------------

def _worker_edges(worker_id: int, emb_dir: Path, edges_dir: Path,
                   jsonl_path: Path, umap_n: int, progress_q) -> None:
    matplotlib.use("Agg")

    def log(msg: str) -> None:
        progress_q.put(f"[edges   ] {msg}")

    structs, valid_ids = _load_manifest(jsonl_path)

    K_e: int | None = None
    binding_e: str  = ""

    def build_nn_e(X: np.ndarray) -> np.ndarray:
        assert K_e is not None
        return (NearestNeighbors(n_neighbors=K_e + 1, metric="euclidean", n_jobs=1)
                .fit(X).kneighbors(X, return_distance=False)[:, 1:])

    # ── Pass 1 (once for all layers) ─────────────────────────────────────────
    # Bond types and distances come from the crystal structure — they are
    # identical across all 6 NN layers. Collecting them once and reusing the
    # same subsample indices for every layer avoids 6 redundant full-directory
    # reads (each ≈ 154k torch.load calls on large edge .pt files).
    _needs_emb: list[str] = [l for l in ALL_LAYERS
                              if not (edges_dir / f"{l}_edges_umap_cache.npz").exists()]
    _umap_idx_arr:   np.ndarray              = np.empty(0, dtype=np.intp)
    _idx_by_mat:     dict[str, list[int]]    = defaultdict(list)
    _gi_to_pos:      dict[int, int]          = {}
    _mat_offsets:    dict[str, int]          = {}
    _sub_bond_types: np.ndarray | None       = None
    _sub_dists:      np.ndarray | None       = None
    _sub_mp_ids:     np.ndarray | None       = None

    if _needs_emb:
        _ref_layer = _needs_emb[0]
        _ref_edir  = emb_dir / _ref_layer / "edges"
        _ref_pts   = [p for p in sorted(_ref_edir.glob("*.pt")) if p.stem in valid_ids]
        log(f"  Pass 1 (once for all layers): collecting edge metadata from {_ref_layer} ...")
        _mt_l: list[str]   = []
        _md_l: list[float] = []
        _mm_l: list[str]   = []
        _mec:  dict[str, int] = {}
        for _p1 in _ref_pts:
            _d1  = torch.load(_p1, weights_only=False)
            _bt1 = _d1["bond_type"]
            _n1  = len(_bt1)
            _mt_l.extend(_bt1)
            _md_l.extend(_d1["distance"].numpy().tolist())
            _mm_l.extend([_p1.stem] * _n1)
            _mec[_p1.stem] = _n1
        _mta = np.array(_mt_l)
        _mda = np.array(_md_l, dtype=np.float32)
        _mma = np.array(_mm_l)
        log(f"  Total edges: {len(_mta):,}")

        _tc  = Counter(_mta.tolist())
        _st  = sum(np.sqrt(c) for c in _tc.values())
        _rng = np.random.default_rng(42)
        _ui: list[int] = []
        for _btk, _cnt in _tc.items():
            _ni = int(umap_n * np.sqrt(_cnt) / _st)
            _ni = max(_ni, 20); _ni = min(_ni, _cnt)
            _btg = np.where(_mta == _btk)[0]
            _ui.extend(_rng.choice(_btg, size=_ni, replace=False).tolist())
        _umap_idx_arr = np.array(_ui)
        _rng.shuffle(_umap_idx_arr)
        log(f"  UMAP subsample: {len(_umap_idx_arr):,} edges")

        _off = 0
        for _p1 in _ref_pts:
            _mat_offsets[_p1.stem] = _off
            _off += _mec[_p1.stem]
        for _gi in _umap_idx_arr.tolist():
            _idx_by_mat[_mma[_gi]].append(int(_gi) - _mat_offsets[_mma[_gi]])
        _gi_to_pos      = {int(_gi): _pos for _pos, _gi in enumerate(_umap_idx_arr)}
        _sub_bond_types = _mta[_umap_idx_arr]
        _sub_dists      = _mda[_umap_idx_arr]
        _sub_mp_ids     = _mma[_umap_idx_arr]

    for layer in ALL_LAYERS:
        t0 = time.time()
        edge_dir_l = emb_dir / layer / "edges"
        log(f"--- {layer} ---")

        cache_path   = edges_dir / f"{layer}_edges_umap_cache.npz"
        quality_path = edges_dir / f"{layer}_edges_pca_quality.png"

        if not cache_path.exists():
            # Metadata (bond types, distances, subsample indices) were collected
            # once before the loop — only load the embeddings for this layer.
            log(f"  Pass 2: loading subsample embeddings ...")
            X_sub = np.zeros((len(_umap_idx_arr), 256), dtype=np.float32)
            for _mat_id, _local_idxs in _idx_by_mat.items():
                _pt2 = edge_dir_l / f"{_mat_id}.pt"
                if not _pt2.exists():
                    continue
                _d2   = torch.load(_pt2, weights_only=False)
                _emb2 = _d2["edge_emb"].float().numpy()
                _base2 = _mat_offsets[_mat_id]
                for _li in _local_idxs:
                    X_sub[_gi_to_pos[_base2 + _li]] = _emb2[_li]

            norms_raw = np.linalg.norm(X_sub, axis=1)
            X_norm    = X_sub / np.where(norms_raw[:, None] == 0, 1, norms_raw[:, None])

            log(f"  Fitting PCA ...")
            evr_e, pca_n_e, X_pca_e = fit_pca(X_norm)
            log(f"  PCA: {pca_n_e} components ({evr_e[:pca_n_e].sum():.1%})")

            log(f"  Running UMAP ({len(X_pca_e):,} edges) ...")
            X_umap_e = run_umap(X_pca_e, UMAP_NN_EDGE)

            np.savez(cache_path,
                     X_umap=X_umap_e, X_pca=X_pca_e, X_raw=X_sub,
                     bond_types=_sub_bond_types,
                     distances=_sub_dists,
                     mp_ids=_sub_mp_ids,
                     umap_idx=_umap_idx_arr, evr=evr_e)
            log(f"  Saved cache -> {cache_path}")

            norms_for_q = norms_raw
            evr_for_q   = evr_e
            pca_n_for_q = pca_n_e
        else:
            c_e         = np.load(cache_path, allow_pickle=True)
            X_sub       = c_e["X_raw"]
            evr_e       = c_e["evr"]
            pca_n_e     = pca_n_for_variance(evr_e)
            norms_for_q = np.linalg.norm(X_sub, axis=1)
            evr_for_q   = evr_e
            pca_n_for_q = pca_n_e

        if not quality_path.exists():
            quality_figure(layer, "edges", evr_for_q, pca_n_for_q,
                           norms_for_q, quality_path,
                           "darkorange", "Edge count",
                           "L2 norm (raw, before normalisation)")
            log(f"  Saved -> {quality_path}")

        # Always load from cache for scatter plots + k-NN
        c_e          = np.load(cache_path, allow_pickle=True)
        X_sub_e      = c_e["X_raw"]
        X_umap_e     = c_e["X_umap"]
        bt_sub       = c_e["bond_types"]
        dists_sub    = c_e["distances"]
        mp_ids_sub_e = c_e["mp_ids"]

        # Calibrate K_e once, from the final-layer (gnn_4) edge subsample.
        # Anchor to bonds-per-bond-type (not bonds-per-material): bond type is
        # the primary clustering axis in edge space, and per-type counts are
        # large enough (~hundreds–thousands) to give a meaningful adaptive K.
        if layer == FINAL_LAYER and K_e is None:
            bpt_e   = np.array(list(Counter(bt_sub.tolist()).values()), dtype=float)
            n_bb_e  = len(mp_ids_sub_e)
            p75_e   = float(np.percentile(bpt_e, 75))
            k_p75_e = int(p75_e) + 1
            k_cap_e = n_bb_e // 10
            K_e       = max(10, min(k_p75_e, k_cap_e))
            binding_e = "p75+1 (by bond-type)" if k_p75_e <= k_cap_e else "10% cap"
            log(f"  K_e = {K_e}  (subsample {n_bb_e:,} bonds, {len(bpt_e):.0f} bond types, p75={p75_e:.1f}, {binding_e})")

        knn_path = edges_dir / f"{layer}_edges_knn_lvr.txt"
        if not knn_path.exists() and K_e is not None:
            log(f"  Computing edges k-NN purity + LVR ...")
            norms_l2 = np.linalg.norm(X_sub_e, axis=1, keepdims=True)
            X_l2     = X_sub_e / np.where(norms_l2 == 0, 1, norms_l2)
            nn_e     = build_nn_e(X_l2)
            buf = io.StringIO()
            buf.write(f"=== Edge Embeddings -- {layer} ===\n")
            buf.write(f"Edges (subsample): {len(X_l2):,}  "
                      f"Materials: {len(set(mp_ids_sub_e.tolist())):,}  "
                      f"Bond types: {len(set(bt_sub.tolist()))}\n")
            buf.write(f"K={K_e}  ({binding_e})\n\n")
            buf.write("-- Categorical (k-NN purity: micro / macro) --\n")
            buf.write(knn_purity_line("bond_type", bt_sub, nn_e, X_l2, build_nn_e))
            buf.write(knn_purity_line("crystal_system",
                [structs[m].get("crystal_system") for m in mp_ids_sub_e],
                nn_e, X_l2, build_nn_e))
            buf.write(knn_purity_line("metal_insulator",
                [_metal_insulator(structs, m) for m in mp_ids_sub_e], nn_e, X_l2, build_nn_e))
            buf.write(knn_purity_line("magnetic_label",
                [_mag_label(structs, m) for m in mp_ids_sub_e], nn_e, X_l2, build_nn_e))
            buf.write("\n-- Continuous (LVR) --\n")
            buf.write(lvr_per_edge("bond_distance (A)", dists_sub, nn_e, X_l2, K_e, build_nn_e))
            md = _mag_density_dict(structs)
            for name, key in LVR_MAT_PROPS:
                buf.write(lvr_material(name, _prop_dict(structs, key), mp_ids_sub_e,
                                       nn_e, X_l2, K_e, build_nn_e))
            buf.write(lvr_material("mag_density (uB/A3)", md, mp_ids_sub_e,
                                   nn_e, X_l2, K_e, build_nn_e))
            knn_path.write_text(buf.getvalue())
            log(f"  Saved -> {knn_path}")

        log(f"  [{layer}] done in {time.time() - t0:.0f}s")

    # Deferred k-NN for layers processed before the final layer (K_e was None then)
    if K_e is None:
        c_e = np.load(edges_dir / f"{FINAL_LAYER}_edges_umap_cache.npz", allow_pickle=True)
        bpt_e   = np.array(list(Counter(c_e["bond_types"].tolist()).values()), dtype=float)
        n_bb_e  = len(c_e["mp_ids"])
        p75_e   = float(np.percentile(bpt_e, 75))
        k_p75_e = int(p75_e) + 1
        k_cap_e = n_bb_e // 10
        K_e       = max(10, min(k_p75_e, k_cap_e))
        binding_e = "p75+1 (by bond-type)" if k_p75_e <= k_cap_e else "10% cap"

    for layer in ALL_LAYERS:
        knn_path = edges_dir / f"{layer}_edges_knn_lvr.txt"
        if knn_path.exists():
            continue
        log(f"  [{layer}] edges k-NN + LVR (deferred) ...")
        c_e          = np.load(edges_dir / f"{layer}_edges_umap_cache.npz", allow_pickle=True)
        X_sub_e      = c_e["X_raw"]
        bt_sub       = c_e["bond_types"]
        dists_sub    = c_e["distances"]
        mp_ids_sub_e = c_e["mp_ids"]
        norms_l2     = np.linalg.norm(X_sub_e, axis=1, keepdims=True)
        X_l2         = X_sub_e / np.where(norms_l2 == 0, 1, norms_l2)
        nn_e         = build_nn_e(X_l2)
        buf = io.StringIO()
        buf.write(f"=== Edge Embeddings -- {layer} ===\n")
        buf.write(f"Edges (subsample): {len(X_l2):,}  "
                  f"Materials: {len(set(mp_ids_sub_e.tolist())):,}  "
                  f"Bond types: {len(set(bt_sub.tolist()))}\n")
        buf.write(f"K={K_e}  ({binding_e})\n\n")
        buf.write("-- Categorical (k-NN purity: micro / macro) --\n")
        buf.write(knn_purity_line("bond_type", bt_sub, nn_e, X_l2, build_nn_e))
        buf.write(knn_purity_line("crystal_system",
            [structs[m].get("crystal_system") for m in mp_ids_sub_e],
            nn_e, X_l2, build_nn_e))
        buf.write(knn_purity_line("metal_insulator",
            [_metal_insulator(structs, m) for m in mp_ids_sub_e], nn_e, X_l2, build_nn_e))
        buf.write(knn_purity_line("magnetic_label",
            [_mag_label(structs, m) for m in mp_ids_sub_e], nn_e, X_l2, build_nn_e))
        buf.write("\n-- Continuous (LVR) --\n")
        buf.write(lvr_per_edge("bond_distance (A)", dists_sub, nn_e, X_l2, K_e, build_nn_e))
        md = _mag_density_dict(structs)
        for name, key in LVR_MAT_PROPS:
            buf.write(lvr_material(name, _prop_dict(structs, key), mp_ids_sub_e,
                                   nn_e, X_l2, K_e, build_nn_e))
        buf.write(lvr_material("mag_density (uB/A3)", md, mp_ids_sub_e,
                               nn_e, X_l2, K_e, build_nn_e))
        knn_path.write_text(buf.getvalue())
        log(f"  Saved -> {knn_path}")

    progress_q.put(f"__DONE__{worker_id}")


# ---------------------------------------------------------------------------
# Worker — edges_pooled
# ---------------------------------------------------------------------------

def _worker_edges_pooled(worker_id: int, emb_dir: Path, edges_dir: Path,
                          jsonl_path: Path, K_ep: int, binding_mat: str,
                          progress_q) -> None:
    matplotlib.use("Agg")

    def log(msg: str) -> None:
        progress_q.put(f"[edges_p ] {msg}")

    structs, valid_ids = _load_manifest(jsonl_path)

    def build_nn(X: np.ndarray) -> np.ndarray:
        return (NearestNeighbors(n_neighbors=K_ep + 1, metric="euclidean", n_jobs=1)
                .fit(X).kneighbors(X, return_distance=False)[:, 1:])

    for layer in ALL_LAYERS:
        t0 = time.time()
        edge_dir_l = emb_dir / layer / "edges"
        log(f"--- {layer} ---")

        cache_path   = edges_dir / f"{layer}_edges_pooled_umap_cache.npz"
        quality_path = edges_dir / f"{layer}_edges_pooled_pca_quality.png"

        if not cache_path.exists():
            log(f"  Loading edges_pooled embeddings ...")
            pt_files = sorted(edge_dir_l.glob("*.pt"))
            ep_embs, ep_mids = [], []
            for pt in pt_files:
                if pt.stem not in valid_ids:
                    continue
                d = torch.load(pt, weights_only=False)
                ep_embs.append(d["edge_structure_emb"].float().numpy())
                ep_mids.append(pt.stem)
            X_ep      = np.stack(ep_embs, axis=0)
            mp_ids_ep = np.array(ep_mids)
            # Drop materials whose edge_structure_emb is NaN/inf (zero-edge materials)
            _fin_ep = np.isfinite(X_ep).all(axis=1)
            if not _fin_ep.all():
                log(f"  WARNING: dropping {(~_fin_ep).sum()} materials with NaN/inf embeddings")
                X_ep      = X_ep[_fin_ep]
                mp_ids_ep = mp_ids_ep[_fin_ep]
            norms_ep  = np.linalg.norm(X_ep, axis=1)
            log(f"  {len(X_ep):,} materials")

            evr_ep, pca_n_ep, X_pca_ep = fit_pca(X_ep)
            log(f"  PCA: {pca_n_ep} components ({evr_ep[:pca_n_ep].sum():.1%})")

            log(f"  Running UMAP ({len(X_pca_ep):,} materials) ...")
            X_umap_ep = run_umap(X_pca_ep, UMAP_NN_NODE)

            np.savez(cache_path,
                     X_umap=X_umap_ep, X_pca=X_pca_ep, X_raw=X_ep,
                     mp_ids=mp_ids_ep, evr=evr_ep)
            log(f"  Saved cache -> {cache_path}")
        else:
            c_ep      = np.load(cache_path, allow_pickle=True)
            X_ep      = c_ep["X_raw"]
            X_umap_ep = c_ep["X_umap"]
            mp_ids_ep = c_ep["mp_ids"]
            evr_ep    = c_ep["evr"]
            pca_n_ep  = pca_n_for_variance(evr_ep)
            norms_ep  = np.linalg.norm(X_ep, axis=1)

        if not quality_path.exists():
            quality_figure(layer, "edges_pooled", evr_ep, pca_n_ep, norms_ep, quality_path,
                           "teal", "Material count",
                           "L2 norm of 256-D edges_pooled embedding")
            log(f"  Saved -> {quality_path}")

        knn_path = edges_dir / f"{layer}_edges_pooled_knn_lvr.txt"
        if not knn_path.exists():
            log(f"  Computing edges_pooled k-NN purity + LVR ...")
            nn_ep = build_nn(X_ep)
            buf = io.StringIO()
            buf.write(f"=== edges_pooled Embeddings -- {layer} ===\n")
            buf.write(f"Materials: {len(X_ep):,}\n")
            buf.write(f"K={K_ep}  ({binding_mat})\n\n")
            buf.write("-- Categorical (k-NN purity: micro / macro) --\n")
            buf.write(knn_purity_line("crystal_system",
                [structs[m].get("crystal_system") for m in mp_ids_ep],
                nn_ep, X_ep, build_nn))
            buf.write(knn_purity_line("metal_insulator",
                [_metal_insulator(structs, m) for m in mp_ids_ep], nn_ep, X_ep, build_nn))
            buf.write(knn_purity_line("magnetic_label",
                [_mag_label(structs, m) for m in mp_ids_ep], nn_ep, X_ep, build_nn))
            buf.write("\n-- Continuous (LVR) --\n")
            md = _mag_density_dict(structs)
            for name, key in LVR_MAT_PROPS:
                buf.write(lvr_material(name, _prop_dict(structs, key), mp_ids_ep,
                                       nn_ep, X_ep, K_ep, build_nn))
            buf.write(lvr_material("mag_density (uB/A3)", md, mp_ids_ep,
                                   nn_ep, X_ep, K_ep, build_nn))
            knn_path.write_text(buf.getvalue())
            log(f"  Saved -> {knn_path}")

        log(f"  [{layer}] done in {time.time() - t0:.0f}s")

    progress_q.put(f"__DONE__{worker_id}")


# ---------------------------------------------------------------------------
# concat_pooled  (pooled node ++ pooled edge, [512], material-level)
# ---------------------------------------------------------------------------

def emit_concat_pooled(nodes_dir: Path, edges_dir: Path, jsonl_path: Path,
                        K_cp: int, binding_mat: str) -> None:
    """Concatenated material representation, mirroring the `concat` rep in
    stages/probes/probes_orb.py: per material, the pooled node `structure`
    [256] concatenated with the pooled `edge_structure_emb` [256] -> [512].

    Built by aligning the existing nodes_pooled / edges_pooled caches by mp_id
    (no .pt re-reads), then the same PCA/UMAP/kNN/LVR pipeline as the other
    pooled reps. Runs in main() after the pooled workers finish, so both source
    caches exist. Caches live in nodes_dir alongside nodes_pooled."""
    structs, valid_ids = _load_manifest(jsonl_path)

    def build_nn(X: np.ndarray) -> np.ndarray:
        return (NearestNeighbors(n_neighbors=K_cp + 1, metric="euclidean", n_jobs=1)
                .fit(X).kneighbors(X, return_distance=False)[:, 1:])

    print("\n========== CONCAT_POOLED (pooled node ++ pooled edge) ==========", flush=True)
    for layer in ALL_LAYERS:
        t0 = time.time()
        cache_path   = nodes_dir / f"{layer}_concat_pooled_umap_cache.npz"
        quality_path = nodes_dir / f"{layer}_concat_pooled_pca_quality.png"
        knn_path     = nodes_dir / f"{layer}_concat_pooled_knn_lvr.txt"
        node_cache   = nodes_dir / f"{layer}_nodes_pooled_umap_cache.npz"
        edge_cache   = edges_dir / f"{layer}_edges_pooled_umap_cache.npz"

        if not cache_path.exists():
            if not (node_cache.exists() and edge_cache.exists()):
                print(f"  [{layer}] SKIP concat_pooled — missing pooled cache "
                      f"(node={node_cache.exists()}, edge={edge_cache.exists()})", flush=True)
                continue
            cn = np.load(node_cache, allow_pickle=True)
            ce = np.load(edge_cache, allow_pickle=True)
            Xn = cn["X_raw"]; mn = cn["mp_ids"].astype(str)
            Xe = ce["X_raw"]; me = ce["mp_ids"].astype(str)
            e_row = {m: i for i, m in enumerate(me)}
            keep_n = [i for i, m in enumerate(mn) if m in e_row]  # node order preserved
            X_cp = np.concatenate(
                [Xn[keep_n], Xe[[e_row[mn[i]] for i in keep_n]]], axis=1)
            mp_ids_cp = mn[keep_n]
            n_drop = len(mn) - len(keep_n)
            if n_drop:
                print(f"  [{layer}] {n_drop} node-materials had no pooled edge vector "
                      f"(dropped from concat)", flush=True)
            norms_cp = np.linalg.norm(X_cp, axis=1)
            print(f"  [{layer}] {len(X_cp):,} materials ({X_cp.shape[1]}-D) — fitting PCA ...",
                  flush=True)

            evr_cp, pca_n_cp, X_pca_cp = fit_pca(X_cp)
            print(f"  [{layer}] PCA: {pca_n_cp} components "
                  f"({evr_cp[:pca_n_cp].sum():.1%}) — running UMAP ...", flush=True)
            X_umap_cp = run_umap(X_pca_cp, UMAP_NN_NODE)

            np.savez(cache_path,
                     X_umap=X_umap_cp, X_pca=X_pca_cp, X_raw=X_cp,
                     mp_ids=mp_ids_cp, evr=evr_cp)
            print(f"  [{layer}] Saved cache -> {cache_path}", flush=True)
        else:
            c         = np.load(cache_path, allow_pickle=True)
            X_cp      = c["X_raw"]
            mp_ids_cp = c["mp_ids"].astype(str)
            evr_cp    = c["evr"]
            pca_n_cp  = pca_n_for_variance(evr_cp)
            norms_cp  = np.linalg.norm(X_cp, axis=1)

        if not quality_path.exists():
            quality_figure(layer, "concat_pooled", evr_cp, pca_n_cp, norms_cp, quality_path,
                           "#2ca02c", "Material count",
                           "L2 norm of 512-D concat_pooled embedding")
            print(f"  [{layer}] Saved -> {quality_path}", flush=True)

        if not knn_path.exists():
            print(f"  [{layer}] concat_pooled k-NN purity + LVR ...", flush=True)
            nn_cp = build_nn(X_cp)
            buf = io.StringIO()
            buf.write(f"=== concat_pooled Embeddings -- {layer} ===\n")
            buf.write(f"Materials: {len(X_cp):,}\n")
            buf.write(f"K={K_cp}  ({binding_mat})\n\n")
            buf.write("-- Categorical (k-NN purity: micro / macro) --\n")
            buf.write(knn_purity_line("crystal_system",
                [structs[m].get("crystal_system") for m in mp_ids_cp], nn_cp, X_cp, build_nn))
            buf.write(knn_purity_line("metal_insulator",
                [_metal_insulator(structs, m) for m in mp_ids_cp], nn_cp, X_cp, build_nn))
            buf.write(knn_purity_line("magnetic_label",
                [_mag_label(structs, m) for m in mp_ids_cp], nn_cp, X_cp, build_nn))
            buf.write("\n-- Continuous (LVR) --\n")
            md = _mag_density_dict(structs)
            for name, key in LVR_MAT_PROPS:
                buf.write(lvr_material(name, _prop_dict(structs, key), mp_ids_cp,
                                       nn_cp, X_cp, K_cp, build_nn))
            buf.write(lvr_material("mag_density (uB/A3)", md, mp_ids_cp,
                                   nn_cp, X_cp, K_cp, build_nn))
            knn_path.write_text(buf.getvalue())
            print(f"  [{layer}] Saved -> {knn_path}", flush=True)

        print(f"  [{layer}] done in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Progress listener
# ---------------------------------------------------------------------------

def _listen(progress_q, n_workers: int) -> None:
    """Print messages from workers until all n_workers send __DONE__."""
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
# Main
# ---------------------------------------------------------------------------
# True edge census (full graph) — parallel one-time pass
# ---------------------------------------------------------------------------

def _census_one(p_str: str):
    """Worker: count bonds + oxygen/carbon/homoatomic families for one material."""
    import torch as _t
    from pathlib import Path as _P
    bt = _t.load(p_str, weights_only=False)["bond_type"]
    ox = ca = ho = 0
    for b in bt:
        a1, a2 = b.split("-") if "-" in b else (b, b)
        if "O" in (a1, a2): ox += 1
        if "C" in (a1, a2): ca += 1
        if a1 == a2:        ho += 1
    return (_P(p_str).stem, len(bt), ox, ca, ho)


def build_edge_census(census_path: Path, emb_dir: Path, layers: list[str],
                      valid_ids: set, n_workers: int) -> None:
    """Write per-material TRUE bond counts + families to census_path (guarded).

    bonds/mat and family stats must reflect the full graph, not the sqrt-weighted
    UMAP subsample. bond_type is identical across layers, so one layer suffices.
    Reads ~150k edge .pt files in parallel (one-time; cached by exists-guard).
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA, UMAP, k-NN purity and LVR for all ORB layers"
    )
    parser.add_argument("--embeddings-dir", default=str(P.RESULTS / "orb"))
    parser.add_argument("--plots-dir",      default=str(P.RESULTS / "orb/plots"))
    parser.add_argument("--structures-dir", default=str(P.STRUCTURES))
    parser.add_argument("--umap-sample",      type=int, default=UMAP_N_DEFAULT,
                        help="Max edges for edge UMAP subsample (default %(default)s)")
    parser.add_argument("--node-umap-sample", type=int, default=NODE_UMAP_N_DEFAULT,
                        help="Max atoms for node UMAP subsample, sqrt-weighted by element "
                             "(default %(default)s)")
    parser.add_argument("--num-workers",      type=int, default=4,
                        help="1 = sequential (debug); 4 = one worker per embedding type "
                             "(default %(default)s)")
    args = parser.parse_args()

    emb_dir  = Path(args.embeddings_dir)
    plot_dir = Path(args.plots_dir)
    str_dir  = Path(args.structures_dir)
    umap_n        = args.umap_sample
    node_umap_n   = args.node_umap_sample

    nodes_dir  = plot_dir / "nodes"
    edges_dir  = plot_dir / "edges"
    global_dir = plot_dir / "global"
    for d in [nodes_dir, edges_dir, global_dir]:
        d.mkdir(parents=True, exist_ok=True)

    jsonl_path = str_dir / "index.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} not found — run stages/data/fetch_mp.py first.")

    # -------------------------------------------------------------------------
    # Calibrate K_node, K_np, K_ep from final-layer (gnn_4) nodes (fast scan)
    # -------------------------------------------------------------------------
    print(f"--- Calibrating adaptive K values from {FINAL_LAYER} ---")
    structs, valid_ids = _load_manifest(jsonl_path)
    print(f"Manifest: {len(structs):,} materials")

    # Count materials from final-layer node files without loading tensors.
    # K_node is calibrated inside _worker_nodes from the node subsample
    # (same approach as K_e in _worker_edges), so no tensor loading needed here.
    bb_node_files = sorted((emb_dir / FINAL_LAYER / "nodes").glob("*.pt"))
    n_mat_bb    = sum(1 for pt in bb_node_files if pt.stem in valid_ids)
    sqrt_k      = int(round(n_mat_bb ** 0.5))
    cap_mat     = n_mat_bb // 10
    K_np        = max(10, min(sqrt_k, cap_mat))
    K_ep        = K_np
    binding_mat = "sqrt(n)" if sqrt_k <= cap_mat else "10% cap"
    print(f"  K_np = K_ep = {K_np}  ({n_mat_bb:,} materials, {binding_mat})")
    print(f"  K_node: calibrated from {FINAL_LAYER} node subsample inside nodes worker")

    # -------------------------------------------------------------------------
    # True edge census (full graph) — for QC bonds/mat + coverage tables
    # -------------------------------------------------------------------------
    build_edge_census(edges_dir / "edge_census.npz", emb_dir, ALL_LAYERS,
                      valid_ids, n_workers=16)

    # -------------------------------------------------------------------------
    # Spawn / run workers
    # -------------------------------------------------------------------------
    worker_args = [
        (_worker_nodes,        (0, emb_dir, nodes_dir, jsonl_path, node_umap_n)),
        (_worker_nodes_pooled, (1, emb_dir, nodes_dir, jsonl_path, K_np,   binding_mat)),
        (_worker_edges,        (2, emb_dir, edges_dir, jsonl_path, umap_n)),
        (_worker_edges_pooled, (3, emb_dir, edges_dir, jsonl_path, K_ep,   binding_mat)),
    ]

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
    # concat_pooled (built from the pooled caches — after workers complete)
    # =========================================================================
    emit_concat_pooled(nodes_dir, edges_dir, jsonl_path, K_np, binding_mat)

    # =========================================================================
    # Cross-layer summary figures (run in main after all workers complete)
    # =========================================================================

    print("\n\n========== CROSS-LAYER SUMMARY ==========")

    EMB_PANELS = [
        ("nodes",         "Node",          nodes_dir),
        ("nodes_pooled",  "Nodes pooled",  nodes_dir),
        ("edges",         "Edge",          edges_dir),
        ("edges_pooled",  "Edges pooled",  edges_dir),
        ("concat_pooled", "Concat pooled", nodes_dir),
    ]

    XLABELS = [LAYER_LABELS[l] for l in ALL_LAYERS]
    X_POS   = list(range(len(ALL_LAYERS)))

    data_all: dict[str, dict[str, tuple[dict, dict]]] = {}
    for emb_type, _, d in EMB_PANELS:
        data_all[emb_type] = {}
        for layer in ALL_LAYERS:
            txt = d / f"{layer}_{emb_type}_knn_lvr.txt"
            data_all[emb_type][layer] = parse_knn_lvr_txt(txt)

    KNN_BG_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "knn_bg", ["#d3d3d3", "#fff3b0", "#aed6f1", "#a9dfbf"])
    LVR_BG_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "lvr_bg", ["#a9dfbf", "#aed6f1", "#fff3b0", "#d3d3d3"])
    ZONE_ALPHA = 0.45

    knn_fig = global_dir / "knn_purity_across_layers.png"
    if not knn_fig.exists():
        fig1, axes1 = plt.subplots(2, 3, figsize=(22, 10), sharey=False)
        for _ax in axes1.flat[len(EMB_PANELS):]:
            _ax.set_visible(False)
        fig1.suptitle("k-NN Purity (normalised) across ORB layers",
                      fontsize=14, y=1.01)
        # Dynamic Y_MIN: expand below -0.10 if any norm values fall further negative
        _all_knn_vals = [
            data_all[_et][_ly][0].get(_pn, np.nan)
            for _et, _, _ in EMB_PANELS
            for _ly in ALL_LAYERS
            for _pn in KNN_PROPS[_et]
        ]
        _knn_data_min = float(np.nanmin(_all_knn_vals)) if any(not np.isnan(v) for v in _all_knn_vals) else -0.10
        Y_MIN = min(-0.10, np.floor(_knn_data_min * 10) / 10 - 0.05)
        Y_MAX = 1.05
        for ax, (emb_type, emb_label, _) in zip(axes1.flat, EMB_PANELS):
            grad = np.linspace(Y_MIN, Y_MAX, 256).reshape(-1, 1)
            ax.imshow(grad, aspect="auto", cmap=KNN_BG_CMAP, vmin=Y_MIN, vmax=1.0,
                      extent=[-0.3, len(ALL_LAYERS) - 0.7, Y_MIN, Y_MAX],
                      origin="lower", zorder=0, alpha=ZONE_ALPHA)
            ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=2)
            for prop_name in KNN_PROPS[emb_type]:
                y = np.array([data_all[emb_type][layer][0].get(prop_name, np.nan)
                              for layer in ALL_LAYERS], dtype=float)
                c_vals = np.array([data_all[emb_type][layer][2].get(prop_name, np.nan)
                                   for layer in ALL_LAYERS], dtype=float)
                mask = ~np.isnan(y)
                if mask.any():
                    x_p = np.array(X_POS)[mask]
                    y_p = y[mask]
                    c_p = c_vals[mask]
                    color = KNN_COLOR_MAP.get(prop_name, "#888888")
                    avg_cov = float(np.nanmean(c_p)) if not np.all(np.isnan(c_p)) else np.nan
                    label = (f"{prop_name}  ({avg_cov:.0%} cov)"
                             if not np.isnan(avg_cov) else prop_name)
                    ax.plot(x_p, y_p, "o-", color=color, lw=2.2, ms=6, zorder=3, label=label)
            ax.set_title(emb_label, fontsize=11)
            ax.set_xticks(X_POS)
            ax.set_xticklabels(XLABELS, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Normalised purity")
            ax.set_ylim(Y_MIN, Y_MAX)
            ax.set_xlim(-0.3, len(ALL_LAYERS) - 0.7)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(axis="y", alpha=0.25, zorder=1)
        plt.tight_layout()
        plt.savefig(knn_fig, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {knn_fig}")

    lvr_fig = global_dir / "lvr_across_layers.png"
    if not lvr_fig.exists():
        fig2, axes2 = plt.subplots(2, 3, figsize=(24, 11), sharey=False)
        for _ax in axes2.flat[len(EMB_PANELS):]:
            _ax.set_visible(False)
        fig2.suptitle("Local Variance Ratio (LVR) across ORB layers\n"
                      "(lower = tighter clustering of the property)",
                      fontsize=14, y=1.02)
        for ax, (emb_type, emb_label, _) in zip(axes2.flat, EMB_PANELS):
            all_vals = [np.array([data_all[emb_type][layer][1].get(p, np.nan)
                                  for layer in ALL_LAYERS], dtype=float)
                        for p in LVR_CROSS_PROPS]
            y_max  = max((np.nanmax(v) for v in all_vals if np.any(~np.isnan(v))), default=1.0)
            y_ceil = max(y_max * 1.08, 0.90)
            grad   = np.linspace(0, y_ceil, 256).reshape(-1, 1)
            ax.imshow(grad, aspect="auto", cmap=LVR_BG_CMAP, vmin=0.0, vmax=1.0,
                      extent=[-0.3, len(ALL_LAYERS) - 0.7, 0, y_ceil],
                      origin="lower", zorder=0, alpha=ZONE_ALPHA)
            for i, prop_name in enumerate(LVR_CROSS_PROPS):
                y = np.array([data_all[emb_type][layer][1].get(prop_name, np.nan)
                              for layer in ALL_LAYERS], dtype=float)
                c_vals = np.array([data_all[emb_type][layer][2].get(prop_name, np.nan)
                                   for layer in ALL_LAYERS], dtype=float)
                mask = ~np.isnan(y)
                if mask.any():
                    x_p = np.array(X_POS)[mask]
                    y_p = y[mask]
                    c_p = c_vals[mask]
                    color = LVR_COLORS[i % len(LVR_COLORS)]
                    avg_cov = float(np.nanmean(c_p)) if not np.all(np.isnan(c_p)) else np.nan
                    label = (f"{prop_name}  ({avg_cov:.0%} cov)"
                             if not np.isnan(avg_cov) else prop_name)
                    ax.plot(x_p, y_p, "o-", color=color, lw=2.2, ms=6, zorder=3, label=label)
            ax.set_title(emb_label, fontsize=11)
            ax.set_xticks(X_POS)
            ax.set_xticklabels(XLABELS, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("LVR")
            ax.set_ylim(0, y_ceil)
            ax.set_xlim(-0.3, len(ALL_LAYERS) - 0.7)
            ax.legend(fontsize=7.5, loc="upper right", ncol=1)
            ax.grid(axis="y", alpha=0.25, zorder=1)
        plt.tight_layout()
        plt.savefig(lvr_fig, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {lvr_fig}")

    # PCA quality grid — cumulative explained variance per (emb type, layer).
    # Mirrors the FairChem UMA `pca_quality_grid`; reads only cached `evr`.
    pca_fig = global_dir / "pca_quality_grid.png"
    if not pca_fig.exists():
        _PCA_COLORS = {"nodes": "steelblue", "nodes_pooled": "#2E86C1",
                       "edges": "#E67E22", "edges_pooled": "#C0392B",
                       "concat_pooled": "#d4a017"}  # mustard
        nr, nc = len(EMB_PANELS), len(ALL_LAYERS)
        figp, axesp = plt.subplots(nr, nc, figsize=(2.6 * nc, 2.1 * nr),
                                   squeeze=False, sharey=True)
        figp.suptitle("ORB v3 — PCA quality (cumulative explained variance)\n"
                      "columns = layers, rows = representations",
                      fontsize=13, y=0.998)
        for r, (emb_type, emb_label, d) in enumerate(EMB_PANELS):
            color = _PCA_COLORS.get(emb_type, "steelblue")
            for c, layer in enumerate(ALL_LAYERS):
                ax = axesp[r][c]
                p = d / f"{layer}_{emb_type}_umap_cache.npz"
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
                    ax.set_ylabel(f"{emb_label}\ncum. var (%)", fontsize=7)
                if r == nr - 1:
                    ax.set_xlabel("# PCs", fontsize=7)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        figp.savefig(pca_fig, dpi=150, bbox_inches="tight")
        plt.close(figp)
        print(f"Saved -> {pca_fig}")

    summary_txt = global_dir / "metrics_summary.txt"
    if not summary_txt.exists():
        col_w = 10
        buf = io.StringIO()
        buf.write("ORB v3 All-Layer Embedding Analysis — Metrics Summary\n")
        buf.write("=" * 70 + "\n\n")
        for emb_type, emb_label, _ in EMB_PANELS:
            buf.write(f"{'=' * 70}\n{emb_label}\n{'=' * 70}\n")
            header = f"  {'property':<28}" + "".join(
                f"{LAYER_LABELS[l]:>{col_w}}" for l in ALL_LAYERS)
            buf.write(header + "\n")
            buf.write("  " + "-" * (28 + col_w * len(ALL_LAYERS)) + "\n")
            buf.write("  k-NN purity (normalised)\n")
            for prop_name in KNN_PROPS[emb_type]:
                row = f"  {prop_name:<28}"
                for layer in ALL_LAYERS:
                    v = data_all[emb_type][layer][0].get(prop_name, np.nan)
                    row += f"{v:>{col_w}.3f}" if not np.isnan(v) else f"{'—':>{col_w}}"
                buf.write(row + "\n")
            buf.write("\n  LVR (lower = tighter)\n")
            for prop_name in LVR_CROSS_PROPS:
                has_any = any(
                    not np.isnan(data_all[emb_type][layer][1].get(prop_name, np.nan))
                    for layer in ALL_LAYERS
                )
                if not has_any:
                    continue
                row = f"  {prop_name:<28}"
                for layer in ALL_LAYERS:
                    v = data_all[emb_type][layer][1].get(prop_name, np.nan)
                    row += f"{v:>{col_w}.3f}" if not np.isnan(v) else f"{'—':>{col_w}}"
                buf.write(row + "\n")
            buf.write("\n")
        summary_txt.write_text(buf.getvalue())
        print(f"Saved -> {summary_txt}")

    # ── Cross-layer summary heatmap (all properties × embedding types × layers) ─
    knn_table = global_dir / "knn_summary_table.png"
    lvr_table = global_dir / "lvr_summary_table.png"
    if not (knn_table.exists() and lvr_table.exists()):
        import matplotlib.gridspec as gridspec
        import matplotlib.transforms as mtransforms

        _KNN_ALL_PROPS = ["crystal_system", "metal_insulator", "magnetic_label",
                          "bond_type", "element_identity"]
        _LVR_ALL_PROPS = ["formation_energy/atom", "energy_above_hull", "band_gap",
                          "mag_density (uB/A3)", "density (g/cm3)", "bulk_modulus (GPa)",
                          "bond_distance (A)"]
        _EMB_SHORT = {"nodes": "nodes", "nodes_pooled": "nodes_p.",
                      "edges": "edges", "edges_pooled": "edges_p.",
                      "concat_pooled": "concat_p."}

        _EMB_KEYS   = ["nodes", "nodes_pooled", "edges", "edges_pooled", "concat_pooled"]

        _KNN_HT_CMAP = mcolors.LinearSegmentedColormap.from_list(
            "knn_bg", ["#d3d3d3", "#fff3b0", "#aed6f1", "#a9dfbf"])
        _LVR_HT_CMAP = mcolors.LinearSegmentedColormap.from_list(
            "lvr_bg", ["#a9dfbf", "#aed6f1", "#fff3b0", "#d3d3d3"])

        def _ht_build_rows(props, metric_idx):
            rows, spans = [], []
            for prop in props:
                embs = [e for e in _EMB_KEYS
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
            _XLBLS = [LAYER_LABELS[l] for l in ALL_LAYERS]
            im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
            ax.set_xticks(range(n_cols))
            ax.set_xticklabels(_XLBLS, rotation=35, ha="right", fontsize=8.5)
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels([_EMB_SHORT[emb] for _, emb in rows], fontsize=8)
            ax.set_title(title, fontsize=11, pad=14)
            for r in range(n_rows):
                for c in range(n_cols):
                    v = mat[r, c]
                    ax.text(c, r, f"{v:.3f}" if not np.isnan(v) else "–",
                            ha="center", va="center", fontsize=6.5,
                            color="black" if not np.isnan(v) else "#aaaaaa")
            trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
            for prop, start, count in spans:
                if start > 0:
                    ax.axhline(start - 0.5, color="#555555", lw=0.9, ls="--", zorder=2)
                ax.text(-0.12, start + (count - 1) / 2, prop,
                        transform=trans, ha="right", va="center",
                        fontsize=8, color="#222222", fontweight="bold")
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
            fig = plt.figure(figsize=(14, 1.6 + 0.42 * n))
            ax = fig.add_subplot(111)
            _ht_draw(ax, mat, rows, spans, cmap, vmin, vmax, title)
            fig.subplots_adjust(left=0.28, right=0.93, top=0.94, bottom=0.06)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved -> {out_path}")

        _save_one(knn_rows, knn_spans, knn_mat, _KNN_HT_CMAP, -0.1, 1.0,
                  "ORB — k-NN Purity (normalised)  [higher = better]", knn_table)
        _save_one(lvr_rows, lvr_spans, lvr_mat, _LVR_HT_CMAP, 0.0, 1.0,
                  "ORB — Local Variance Ratio (LVR)  [lower = better]", lvr_table)

    # ── UMAP Summary Plots (layers × embedding types) ───────────────────────
    _CACHE_PATHS = {
        "nodes":        lambda lyr: nodes_dir / f"{lyr}_nodes_umap_cache.npz",
        "nodes_pooled": lambda lyr: nodes_dir / f"{lyr}_nodes_pooled_umap_cache.npz",
        "edges":        lambda lyr: edges_dir / f"{lyr}_edges_umap_cache.npz",
        "edges_pooled": lambda lyr: edges_dir / f"{lyr}_edges_pooled_umap_cache.npz",
        "concat_pooled": lambda lyr: nodes_dir / f"{lyr}_concat_pooled_umap_cache.npz",
    }
    _EMB_KEYS   = ["nodes", "nodes_pooled", "edges", "edges_pooled", "concat_pooled"]
    _EMB_LABELS = ["Nodes", "Nodes pooled", "Edges", "Edges pooled", "Concat pooled"]
    _ALPHA_FOR  = {"nodes": 0.1, "nodes_pooled": 0.1,
                   "edges": 0.1, "edges_pooled": 0.1, "concat_pooled": 0.1}
    _LAYER_LBLS = [LAYER_LABELS[l] for l in ALL_LAYERS]
    _SUPTITLE_Y = 0.99

    def _load_c(layer, emb_key):
        p = _CACHE_PATHS[emb_key](layer)
        return np.load(p, allow_pickle=True) if p.exists() else None

    def _draw_na(ax):
        ax.set_facecolor("#eeeeee")
        ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#888888")
        ax.set_xticks([]); ax.set_yticks([])

    def _grid(title):
        _ncols = len(_EMB_KEYS)
        _nrows = len(ALL_LAYERS)
        fig, axes = plt.subplots(_nrows, _ncols,
                                 figsize=(4 * _ncols, 26 * _nrows / 7))
        fig.suptitle(title, fontsize=14, y=_SUPTITLE_Y)
        for ci, lbl in enumerate(_EMB_LABELS):
            axes[0][ci].set_title(lbl, fontsize=11, pad=6)
        for ri, lbl in enumerate(_LAYER_LBLS):
            axes[ri][0].set_ylabel(lbl, fontsize=10, labelpad=6)
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

    # --- crystal system ---
    _cs_summary = global_dir / "umap_crystal_system_summary.png"
    if not _cs_summary.exists():
        fig, axes = _grid("UMAP — Crystal System")
        for ri, layer in enumerate(ALL_LAYERS):
            for ci, emb_key in enumerate(_EMB_KEYS):
                ax = axes[ri][ci]
                c = _load_c(layer, emb_key)
                if c is None:
                    _draw_na(ax); continue
                X = c["X_umap"]
                cs = (c["labels"].astype(str) if emb_key == "nodes"
                      else np.array([structs.get(m, {}).get("crystal_system", "Unknown")
                                     for m in _get_mp_ids(c)]))
                a = _ALPHA_FOR[emb_key]
                for _cs in CRYSTAL_SYSTEMS:
                    mask = cs == _cs
                    if mask.any():
                        ax.scatter(X[mask, 0], X[mask, 1], c=[CRYSTAL_COLORS[_cs]],
                                   s=1, alpha=a, linewidths=0, rasterized=True)
        handles = [mlines.Line2D([], [], marker="o", linestyle="None",
                                 color=CRYSTAL_COLORS[cs], markersize=7, label=cs)
                   for cs in CRYSTAL_SYSTEMS]
        plt.tight_layout(rect=[0, 0, 1.0, 0.94])
        _p0 = axes[0][0].get_position()
        fig.legend(handles=handles, title="Crystal system", loc="center",
                   ncol=len(CRYSTAL_SYSTEMS), fontsize=9, title_fontsize=10,
                   framealpha=0.92,
                   bbox_to_anchor=(0.5, (_p0.y1 + _SUPTITLE_Y) / 2),
                   bbox_transform=fig.transFigure)
        plt.savefig(_cs_summary, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {_cs_summary}")

    # --- metal / insulator ---
    _mi_summary = global_dir / "umap_metal_insulator_summary.png"
    if not _mi_summary.exists():
        fig, axes = _grid("UMAP — Metal / Insulator")
        for ri, layer in enumerate(ALL_LAYERS):
            for ci, emb_key in enumerate(_EMB_KEYS):
                ax = axes[ri][ci]
                c = _load_c(layer, emb_key)
                if c is None:
                    _draw_na(ax); continue
                X      = c["X_umap"]
                mp_ids = _get_mp_ids(c)
                a      = _ALPHA_FOR[emb_key]
                bg     = np.array([structs.get(m, {}).get("band_gap") for m in mp_ids])
                has    = np.array([v is not None for v in bg])
                bg_f   = np.array([v if v is not None else np.nan for v in bg], dtype=float)
                is_metal     = has & (bg_f == 0.0)
                is_insulator = has & (bg_f  > 0.0)
                no_data      = ~has
                if no_data.any():
                    ax.scatter(X[no_data, 0], X[no_data, 1], c="#dddddd",
                               s=1, alpha=0.1, linewidths=0, rasterized=True)
                ax.scatter(X[is_metal, 0], X[is_metal, 1], c="teal",
                           s=1, alpha=a, linewidths=0, rasterized=True)
                ax.scatter(X[is_insulator, 0], X[is_insulator, 1], c="goldenrod",
                           s=1, alpha=a, linewidths=0, rasterized=True)
        handles = [mlines.Line2D([], [], marker="o", linestyle="None",
                                 color=c, markersize=6, label=l)
                   for l, c in [("metal", "teal"), ("insulator", "goldenrod"),
                                 ("no data", "#dddddd")]]
        plt.tight_layout(rect=[0, 0, 1.0, 0.94])
        _p0 = axes[0][0].get_position()
        fig.legend(handles=handles, title="Metal / Insulator", loc="center",
                   ncol=3, fontsize=9, title_fontsize=10, framealpha=0.9,
                   bbox_to_anchor=(0.5, (_p0.y1 + _SUPTITLE_Y) / 2),
                   bbox_transform=fig.transFigure)
        plt.savefig(_mi_summary, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {_mi_summary}")

    # --- magnetic label ---
    _mag_summary = global_dir / "umap_magnetic_label_summary.png"
    if not _mag_summary.exists():
        fig, axes = _grid("UMAP — Magnetic Label")
        for ri, layer in enumerate(ALL_LAYERS):
            for ci, emb_key in enumerate(_EMB_KEYS):
                ax = axes[ri][ci]
                c = _load_c(layer, emb_key)
                if c is None:
                    _draw_na(ax); continue
                X      = c["X_umap"]
                mp_ids = _get_mp_ids(c)
                a      = _ALPHA_FOR[emb_key]
                mag_d  = np.array([
                    structs.get(m, {}).get("total_magnetization") / structs.get(m, {}).get("volume")
                    if (structs.get(m, {}).get("total_magnetization") is not None
                        and structs.get(m, {}).get("volume"))
                    else np.nan
                    for m in mp_ids], dtype=float)
                has_m     = ~np.isnan(mag_d)
                is_mag    = has_m & (np.abs(mag_d) >= MAG_THR)
                is_nonmag = has_m & (np.abs(mag_d) <  MAG_THR)
                no_data   = ~has_m
                if no_data.any():
                    ax.scatter(X[no_data, 0], X[no_data, 1], c="#dddddd",
                               s=1, alpha=0.1, linewidths=0, rasterized=True)
                ax.scatter(X[is_nonmag, 0], X[is_nonmag, 1], c="turquoise",
                           s=1, alpha=a, linewidths=0, rasterized=True)
                ax.scatter(X[is_mag, 0], X[is_mag, 1], c="crimson",
                           s=1, alpha=a, linewidths=0, rasterized=True)
        handles = [mlines.Line2D([], [], marker="o", linestyle="None",
                                 color=c, markersize=6, label=l)
                   for l, c in [("non-magnetic", "turquoise"), ("magnetic", "crimson"),
                                 ("no data", "#dddddd")]]
        plt.tight_layout(rect=[0, 0, 1.0, 0.94])
        _p0 = axes[0][0].get_position()
        fig.legend(handles=handles, title="Magnetic label", loc="center",
                   ncol=3, fontsize=9, title_fontsize=10, framealpha=0.9,
                   bbox_to_anchor=(0.5, (_p0.y1 + _SUPTITLE_Y) / 2),
                   bbox_transform=fig.transFigure)
        plt.savefig(_mag_summary, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {_mag_summary}")

    # --- element identity (node) | bond type (edge) | bond distance (edge) ---
    _eb_summary = global_dir / "umap_element_bond_summary.png"
    if not _eb_summary.exists():
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
        _bb_edge_c = _load_c(FINAL_LAYER, "edges")
        _top_bt = ([bt for bt, _ in Counter(_bb_edge_c["bond_types"].astype(str).tolist()).most_common(10)]
                   if _bb_edge_c is not None else [])
        _BT_CMAP2 = plt.colormaps["gist_rainbow"].resampled(max(len(_top_bt), 1))
        _BT_CM2   = {bt: _BT_CMAP2(i) for i, bt in enumerate(_top_bt)}

        fig, axes = plt.subplots(len(ALL_LAYERS), 3,
                                 figsize=(13, 26 * len(ALL_LAYERS) / 7))
        fig.suptitle(
            "UMAP — Element Identity (node)  |  Bond Type (edge)  |  Bond Distance (edge)",
            fontsize=13, y=_SUPTITLE_Y)

        sc_elem = None
        sc_dist = None
        for ri, layer in enumerate(ALL_LAYERS):
            # left: element identity (node)
            ax = axes[ri][0]
            c_n = _load_c(layer, "nodes")
            if c_n is not None:
                X_n   = c_n["X_umap"]
                elems = c_n["elements"].astype(str)
                Z     = np.array([_SYMBOL_TO_Z.get(e, 0) for e in elems])
                sc_elem = ax.scatter(X_n[:, 0], X_n[:, 1], c=Z, cmap="gist_ncar",
                                     s=1, alpha=0.1, linewidths=0, rasterized=True)
            else:
                _draw_na(ax)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylabel(_LAYER_LBLS[ri], fontsize=10, labelpad=6)

            # middle: bond type (edge)
            ax  = axes[ri][1]
            ax2 = axes[ri][2]
            c_e = _load_c(layer, "edges")
            if c_e is not None:
                X_e  = c_e["X_umap"]
                bt   = c_e["bond_types"].astype(str)
                dist = c_e["distances"].astype(float)
                other = ~np.isin(bt, _top_bt)
                if other.any():
                    ax.scatter(X_e[other, 0], X_e[other, 1], c="lightgrey",
                               s=1, alpha=0.1, linewidths=0, rasterized=True)
                for _bt in _top_bt:
                    mask = bt == _bt
                    if mask.any():
                        ax.scatter(X_e[mask, 0], X_e[mask, 1], c=[_BT_CM2[_bt]],
                                   s=1, alpha=0.1, linewidths=0, rasterized=True)
                # right: bond distance
                sc_dist = ax2.scatter(X_e[:, 0], X_e[:, 1], c=dist, cmap="plasma",
                                      s=1, alpha=0.1, linewidths=0, rasterized=True)
            else:
                _draw_na(ax); _draw_na(ax2)
            ax.set_xticks([]); ax.set_yticks([])
            ax2.set_xticks([]); ax2.set_yticks([])

        axes[0][0].set_title("Node — Element Identity", fontsize=11, pad=6)
        axes[0][1].set_title("Edge — Bond Type", fontsize=11, pad=6)
        axes[0][2].set_title("Edge — Bond Distance (Å)", fontsize=11, pad=6)

        _bt_handles = ([mlines.Line2D([], [], marker="o", linestyle="None",
                                      color=_BT_CM2[bt], markersize=6, label=bt)
                        for bt in _top_bt]
                       + [mlines.Line2D([], [], marker="o", linestyle="None",
                                        color="lightgrey", markersize=6, label="other")])

        plt.tight_layout(rect=[0, 0, 1.0, 0.94])
        _p0     = axes[0][0].get_position()
        _p1     = axes[0][1].get_position()
        _p2     = axes[0][2].get_position()
        _leg_cy = (_p0.y1 + _SUPTITLE_Y) / 2
        _cbar_h = 0.007
        _cbar_w = _p0.width * 0.72

        fig.legend(handles=_bt_handles, title="Bond type (top 10)", loc="center",
                   ncol=3, fontsize=7, title_fontsize=8, framealpha=0.9,
                   bbox_to_anchor=(_p1.x0 + _p1.width / 2, _leg_cy),
                   bbox_transform=fig.transFigure)

        if sc_elem is not None:
            _cx0  = _p0.x0 + (_p0.width - _cbar_w) / 2
            cax_z = fig.add_axes([_cx0, _leg_cy - _cbar_h / 2, _cbar_w, _cbar_h])
            cb_z  = fig.colorbar(sc_elem, cax=cax_z, orientation="horizontal")
            cb_z.solids.set_alpha(1)
            cax_z.set_title("Atomic number Z", fontsize=8, pad=3)
            cb_z.ax.tick_params(labelsize=7)

        if sc_dist is not None:
            _cx2  = _p2.x0 + (_p2.width - _cbar_w) / 2
            cax_d = fig.add_axes([_cx2, _leg_cy - _cbar_h / 2, _cbar_w, _cbar_h])
            cb_d  = fig.colorbar(sc_dist, cax=cax_d, orientation="horizontal")
            cb_d.solids.set_alpha(1)
            cax_d.set_title("Bond distance (Å)", fontsize=8, pad=3)
            cb_d.ax.tick_params(labelsize=7)

        plt.savefig(_eb_summary, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {_eb_summary}")

    # --- continuous property helper ---
    def _cont_summary(out_path, title, get_val_fn, cmap, norm, cbar_label,
                      extra_legend=None):
        if out_path.exists():
            return
        fig, axes = _grid(title)
        sc_last = None
        n_miss = n_tot = 0
        for ri, layer in enumerate(ALL_LAYERS):
            for ci, emb_key in enumerate(_EMB_KEYS):
                ax = axes[ri][ci]
                c = _load_c(layer, emb_key)
                if c is None:
                    _draw_na(ax); continue
                X      = c["X_umap"]
                mp_ids = _get_mp_ids(c)
                vals   = np.array([get_val_fn(structs.get(m, {})) for m in mp_ids],
                                  dtype=float)
                has    = ~np.isnan(vals)
                n_miss += int((~has).sum()); n_tot += has.size
                a      = _ALPHA_FOR[emb_key]
                if (~has).any():
                    ax.scatter(X[~has, 0], X[~has, 1], c="lightgrey",
                               s=1, alpha=0.1, linewidths=0, rasterized=True)
                if has.any():
                    sc_last = ax.scatter(X[has, 0], X[has, 1],
                                         c=vals[has], cmap=cmap, norm=norm,
                                         s=1, alpha=a, linewidths=0, rasterized=True)
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
        # Always draw the colorbar — use sc_last if data exists, otherwise a
        # ScalarMappable so the scale is visible even when no points are colored.
        cax = fig.add_axes([0.5 - _cbar_w / 2, _cbar_cy - _cbar_h / 2, _cbar_w, _cbar_h])
        _cb_src = sc_last if sc_last is not None else plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        if sc_last is None:
            _cb_src.set_array([])
        cb  = fig.colorbar(_cb_src, cax=cax, orientation="horizontal")
        if sc_last is not None:
            cb.solids.set_alpha(1)
        cax.set_title(cbar_label, fontsize=9, pad=3)
        cb.ax.tick_params(labelsize=7)
        if extra_legend:
            fig.legend(**extra_legend)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {out_path}")

    # formation energy — p1/p99 normalised
    _fe_all  = np.array([v for m in structs
                          for v in [structs[m].get("formation_energy_per_atom")]
                          if v is not None], dtype=float)
    _fe_vmin = float(np.percentile(_fe_all, 1))  if len(_fe_all) else -3.0
    _fe_vmax = float(np.percentile(_fe_all, 99)) if len(_fe_all) else  3.0
    _cont_summary(global_dir / "umap_formation_energy_summary.png",
                  "UMAP — Formation Energy / Atom",
                  lambda s: _v(s, "formation_energy_per_atom"),
                  "coolwarm", mcolors.Normalize(vmin=_fe_vmin, vmax=_fe_vmax),
                  "Formation energy (eV/atom)")

    # energy above hull — PowerNorm sqrt-ish
    _cont_summary(global_dir / "umap_energy_above_hull_summary.png",
                  "UMAP — Energy Above Hull",
                  lambda s: _v(s, "energy_above_hull"),
                  "coolwarm", mcolors.PowerNorm(gamma=0.4, vmin=0.0, vmax=1.0),
                  "Energy above hull (eV/atom)  [sqrt-ish, 0–1 eV]")

    # band gap — metals (aquamarine) + non-metals (plasma PowerNorm)
    _bg_summary = global_dir / "umap_band_gap_summary.png"
    if not _bg_summary.exists():
        _norm_bg = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=5.0)
        fig, axes = _grid("UMAP — Band Gap")
        sc_last = None
        for ri, layer in enumerate(ALL_LAYERS):
            for ci, emb_key in enumerate(_EMB_KEYS):
                ax = axes[ri][ci]
                c = _load_c(layer, emb_key)
                if c is None:
                    _draw_na(ax); continue
                X      = c["X_umap"]
                mp_ids = _get_mp_ids(c)
                bg     = np.array([structs.get(m, {}).get("band_gap") for m in mp_ids])
                has    = np.array([v is not None for v in bg])
                bg_f   = np.array([v if v is not None else np.nan for v in bg], dtype=float)
                is_metal    = has & (bg_f == 0.0)
                is_nonmetal = has & (bg_f  > 0.0)
                no_data     = ~has
                a = _ALPHA_FOR[emb_key]
                if no_data.any():
                    ax.scatter(X[no_data, 0], X[no_data, 1], c="lightgrey",
                               s=1, alpha=0.1, linewidths=0, rasterized=True)
                ax.scatter(X[is_metal, 0], X[is_metal, 1], c="aquamarine",
                           s=1, alpha=a, linewidths=0, rasterized=True)
                if is_nonmetal.any():
                    sc_last = ax.scatter(X[is_nonmetal, 0], X[is_nonmetal, 1],
                                         c=bg_f[is_nonmetal], cmap="plasma",
                                         norm=_norm_bg, s=1, alpha=a,
                                         linewidths=0, rasterized=True)
        plt.tight_layout(rect=[0, 0, 1.0, 0.94])
        _p0 = axes[0][0].get_position()
        _cbar_h, _cbar_w = 0.007, 0.30
        _cbar_cx = 0.5 - _cbar_w / 2 - 0.06
        _cbar_cy = (_p0.y1 + _SUPTITLE_Y) / 2
        if sc_last is not None:
            cax = fig.add_axes([_cbar_cx, _cbar_cy - _cbar_h / 2, _cbar_w, _cbar_h])
            cb  = fig.colorbar(sc_last, cax=cax, orientation="horizontal")
            cb.solids.set_alpha(1)
            cax.set_title("Band gap (eV)  [non-metals, sqrt scale, 0–5 eV]", fontsize=9, pad=3)
            cb.ax.tick_params(labelsize=7)
            fig.legend(handles=[mpatches.Patch(facecolor="aquamarine", label="Metal (bg = 0)")],
                       loc="center left", fontsize=8, framealpha=0,
                       bbox_to_anchor=(_cbar_cx + _cbar_w + 0.025, _cbar_cy),
                       bbox_transform=fig.transFigure)
        plt.savefig(_bg_summary, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {_bg_summary}")

    # magnetic density — non-magnetic (aquamarine) + magnetic (hot_r, log)
    _magd_summary = global_dir / "umap_mag_density_summary.png"
    if not _magd_summary.exists():
        _mag_d_all = np.array([
            structs[m]["total_magnetization"] / structs[m]["volume"]
            for m in structs
            if structs[m].get("total_magnetization") is not None
            and structs[m].get("volume")
        ], dtype=float)
        _mag_d_pos = np.abs(_mag_d_all[np.abs(_mag_d_all) >= MAG_THR])
        _vmax_mag  = float(np.percentile(_mag_d_pos, 99)) if len(_mag_d_pos) else 1.0
        fig, axes = _grid("UMAP — Magnetic Density (μB/Å³)")
        sc_last = None
        for ri, layer in enumerate(ALL_LAYERS):
            for ci, emb_key in enumerate(_EMB_KEYS):
                ax = axes[ri][ci]
                c = _load_c(layer, emb_key)
                if c is None:
                    _draw_na(ax); continue
                X      = c["X_umap"]
                mp_ids = _get_mp_ids(c)
                a      = _ALPHA_FOR[emb_key]
                mag_d  = np.array([
                    structs.get(m, {}).get("total_magnetization") / structs.get(m, {}).get("volume")
                    if (structs.get(m, {}).get("total_magnetization") is not None
                        and structs.get(m, {}).get("volume"))
                    else np.nan
                    for m in mp_ids], dtype=float)
                has_m     = ~np.isnan(mag_d)
                is_mag    = has_m & (np.abs(mag_d) >= MAG_THR)
                is_nonmag = has_m & (np.abs(mag_d) <  MAG_THR)
                no_data   = ~has_m
                if no_data.any():
                    ax.scatter(X[no_data, 0], X[no_data, 1], c="#dddddd",
                               s=1, alpha=0.1, linewidths=0, rasterized=True)
                ax.scatter(X[is_nonmag, 0], X[is_nonmag, 1], c="aquamarine",
                           s=1, alpha=a, linewidths=0, rasterized=True)
                if is_mag.any():
                    sc_last = ax.scatter(X[is_mag, 0], X[is_mag, 1],
                                         c=np.abs(mag_d[is_mag]), cmap="hot_r",
                                         norm=mcolors.LogNorm(vmin=MAG_THR, vmax=_vmax_mag),
                                         s=1, alpha=a, linewidths=0, rasterized=True)
        plt.tight_layout(rect=[0, 0, 1.0, 0.94])
        _p0 = axes[0][0].get_position()
        _cbar_h, _cbar_w = 0.007, 0.30
        _cbar_cx = 0.5 - _cbar_w / 2 - 0.06
        _cbar_cy = (_p0.y1 + _SUPTITLE_Y) / 2
        if sc_last is not None:
            cax = fig.add_axes([_cbar_cx, _cbar_cy - _cbar_h / 2, _cbar_w, _cbar_h])
            cb  = fig.colorbar(sc_last, cax=cax, orientation="horizontal")
            cb.solids.set_alpha(1)
            cax.set_title("|Mag. density| (μB/Å³)  [log scale, 99th pct max]", fontsize=9, pad=3)
            cb.ax.tick_params(labelsize=7)
            fig.legend(handles=[mpatches.Patch(facecolor="aquamarine", label="Non-magnetic")],
                       loc="center left", fontsize=8, framealpha=0,
                       bbox_to_anchor=(_cbar_cx + _cbar_w + 0.025, _cbar_cy),
                       bbox_transform=fig.transFigure)
        plt.savefig(_magd_summary, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {_magd_summary}")

    # density — PowerNorm sqrt-ish, viridis
    _cont_summary(global_dir / "umap_density_summary.png",
                  "UMAP — Density",
                  lambda s: _v(s, "density"),
                  "viridis", mcolors.PowerNorm(gamma=0.6, vmin=0, vmax=20.0),
                  "Density (g/cm³)  [sqrt-ish, 0–20 g/cm³]")

    # bulk modulus — PowerNorm sqrt, viridis
    _cont_summary(global_dir / "umap_bulk_modulus_summary.png",
                  "UMAP — Bulk Modulus",
                  lambda s: _v(s, "bulk_modulus"),
                  "viridis", mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=440.0),
                  "Bulk modulus (GPa)  [sqrt scale, 0–440 GPa]")

    # ── Final-layer (gnn_4) k-NN purity + LVR bar charts ─────────────────────
    bb_dir = plot_dir / "gnn_4"
    bb_dir.mkdir(parents=True, exist_ok=True)

    _BB_KNN_PROPS = [
        ("crystal_system",   "crystal system",          "#4E9F8A"),
        ("metal_insulator",  "metal / insulator",       "#E17C05"),
        ("magnetic_label",   "magnetic label",          "#A06CD5"),
        ("element_identity", "element identity (node)", "#D65F5F"),
        ("bond_type",        "bond type (edge, macro)", "#2E86C1"),
    ]
    _BB_LVR_PROPS = [
        ("formation_energy/atom", "formation energy",  "#DD8452"),
        ("energy_above_hull",     "energy above hull", "#C44E52"),
        ("band_gap",              "band gap",          "#CCB974"),
        ("mag_density (uB/A3)",   "mag density",       "#937860"),
        ("density (g/cm3)",       "density",           "#DA8BC3"),
        ("bulk_modulus (GPa)",    "bulk modulus",      "#8C8C8C"),
        ("bond_distance (A)",     "bond distance (edge)", "#2196F3"),
    ]
    _BB_EMB_KEYS   = ["nodes", "nodes_pooled", "edges", "edges_pooled", "concat_pooled"]
    _BB_EMB_LABELS = ["Nodes", "Nodes\npooled", "Edges", "Edges\npooled", "Concat\npooled"]
    _x_bb = np.arange(len(_BB_EMB_KEYS))

    bb_knn_fig = bb_dir / "gnn_4_knn_purity.png"
    if not bb_knn_fig.exists():
        fig, ax = plt.subplots(figsize=(11, 4))
        w = 0.15
        legend_done = set()
        for pi, (prop, lbl, col) in enumerate(_BB_KNN_PROPS):
            for xi, emb_key in enumerate(_BB_EMB_KEYS):
                val = data_all[emb_key][FINAL_LAYER][0].get(prop, np.nan)
                if np.isnan(val):
                    continue
                valid_props = [p for p, _, _ in _BB_KNN_PROPS
                               if not np.isnan(data_all[emb_key][FINAL_LAYER][0].get(p, np.nan))]
                n_v = len(valid_props)
                si  = valid_props.index(prop) if prop in valid_props else pi
                off = (si - (n_v - 1) / 2) * w
                ax.bar(_x_bb[xi] + off, val, w,
                       label=lbl if prop not in legend_done else "_nolegend_",
                       color=col, alpha=0.85)
                legend_done.add(prop)
        ax.set_xticks(_x_bb); ax.set_xticklabels(_BB_EMB_LABELS, fontsize=10)
        ax.axhline(0,   color="gray",  lw=0.8, ls="--", alpha=0.6)
        ax.axhline(0.5, color="green", lw=0.8, ls=":",  alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("k-NN purity (normalised)")
        ax.set_title("gnn_4 — k-NN purity (normalised)  [higher = better]", fontsize=12)
        ax.legend(title="Property", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
        plt.tight_layout()
        plt.savefig(bb_knn_fig, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {bb_knn_fig}")

    bb_lvr_fig = bb_dir / "gnn_4_lvr.png"
    if not bb_lvr_fig.exists():
        fig, ax = plt.subplots(figsize=(12, 4))
        n_lvr = len(_BB_LVR_PROPS)
        w = 0.10
        for pi, (prop, lbl, col) in enumerate(_BB_LVR_PROPS):
            vals = [data_all[emb_key][FINAL_LAYER][1].get(prop, np.nan)
                    for emb_key in _BB_EMB_KEYS]
            off  = (pi - (n_lvr - 1) / 2) * w
            ax.bar(_x_bb + off, vals, w, label=lbl, color=col, alpha=0.85)
        ax.set_xticks(_x_bb); ax.set_xticklabels(_BB_EMB_LABELS, fontsize=10)
        ax.axhline(0.40, color="green",  lw=0.8, ls=":", alpha=0.5)
        ax.axhline(0.65, color="orange", lw=0.8, ls=":", alpha=0.5)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("LVR")
        ax.set_title("gnn_4 — LVR  [lower = better]", fontsize=12)
        ax.legend(title="Property", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
        plt.tight_layout()
        plt.savefig(bb_lvr_fig, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {bb_lvr_fig}")

    # ── Final-layer (gnn_4) summary heatmap ──────────────────────────────────
    bb_heat_fig = bb_dir / "gnn_4_summary_table.png"
    if not bb_heat_fig.exists():
        _HT_PUR_PROPS  = ["crystal_system", "metal_insulator", "magnetic_label",
                           "element_identity", "bond_type"]
        _HT_PUR_LABELS = ["crystal_system", "metal_insulator", "magnetic_label",
                           "element_identity\n(node only)", "bond_type\n(edge, macro)"]
        _HT_LVR_PROPS  = ["formation_energy/atom", "energy_above_hull", "band_gap",
                           "mag_density (uB/A3)", "density (g/cm3)", "bulk_modulus (GPa)",
                           "bond_distance (A)"]
        _HT_LVR_LABELS = ["formation_energy/atom", "energy_above_hull", "band_gap",
                           "mag_density", "density", "bulk_modulus",
                           "bond_distance\n(edge only)"]
        _pur_mat = np.array([[data_all[et][FINAL_LAYER][0].get(p, np.nan)
                              for p in _HT_PUR_PROPS] for et in _BB_EMB_KEYS])
        _lvr_mat = np.array([[data_all[et][FINAL_LAYER][1].get(p, np.nan)
                              for p in _HT_LVR_PROPS] for et in _BB_EMB_KEYS])
        _n_pur = len(_HT_PUR_PROPS); _n_lvr_h = len(_HT_LVR_PROPS)
        fig, (_ax_ph, _ax_lh) = plt.subplots(
            1, 2,
            figsize=(16, max(_n_pur, _n_lvr_h) * 0.6 + 2.2),
            gridspec_kw={"width_ratios": [_n_pur, _n_lvr_h]}
        )
        _KNN_BG = mcolors.LinearSegmentedColormap.from_list(
            "knn_bg", ["#d3d3d3", "#fff3b0", "#aed6f1", "#a9dfbf"])
        _LVR_BG = mcolors.LinearSegmentedColormap.from_list(
            "lvr_bg", ["#a9dfbf", "#aed6f1", "#fff3b0", "#d3d3d3"])
        def _annotate(ax, mat):
            for ri in range(mat.shape[0]):
                for ci in range(mat.shape[1]):
                    v = mat[ri, ci]
                    if not np.isnan(v):
                        ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                                fontsize=8, color="black")
        _im_ph = _ax_ph.imshow(_pur_mat.T, aspect="auto", cmap=_KNN_BG,
                                vmin=-0.1, vmax=1.0, interpolation="nearest")
        _ax_ph.set_xticks(range(len(_BB_EMB_LABELS)))
        _ax_ph.set_xticklabels(_BB_EMB_LABELS, fontsize=9, rotation=15, ha="right")
        _ax_ph.set_yticks(range(_n_pur))
        _ax_ph.set_yticklabels(_HT_PUR_LABELS, fontsize=9)
        _ax_ph.set_title("k-NN purity (normalised)  [higher = better]", fontsize=11)
        _annotate(_ax_ph, _pur_mat.T)
        fig.colorbar(_im_ph, ax=_ax_ph, shrink=0.6, pad=0.02).ax.tick_params(labelsize=7)
        _im_lh = _ax_lh.imshow(_lvr_mat.T, aspect="auto", cmap=_LVR_BG,
                                vmin=0.0, vmax=1.0, interpolation="nearest")
        _ax_lh.set_xticks(range(len(_BB_EMB_LABELS)))
        _ax_lh.set_xticklabels(_BB_EMB_LABELS, fontsize=9, rotation=15, ha="right")
        _ax_lh.set_yticks(range(_n_lvr_h))
        _ax_lh.set_yticklabels(_HT_LVR_LABELS, fontsize=9)
        _ax_lh.set_title("LVR  [lower = better]", fontsize=11)
        _annotate(_ax_lh, _lvr_mat.T)
        fig.colorbar(_im_lh, ax=_ax_lh, shrink=0.6, pad=0.02).ax.tick_params(labelsize=7)
        fig.suptitle("gnn_4 — metrics summary", fontsize=13)
        plt.tight_layout()
        plt.savefig(bb_heat_fig, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved -> {bb_heat_fig}")

    # ── Final-layer (gnn_4) metrics summary txt ──────────────────────────────
    bb_txt = bb_dir / "gnn_4_metrics_summary.txt"
    if not bb_txt.exists():
        buf = io.StringIO()
        buf.write("gnn_4 Metrics Summary\n" + "=" * 50 + "\n\n")
        for emb_key, emb_lbl, _ in EMB_PANELS:
            knn_d, lvr_d, _cov_d = data_all[emb_key][FINAL_LAYER]
            buf.write(f"\n--- {emb_lbl} ---\n")
            for k in KNN_PROPS[emb_key]:
                v = knn_d.get(k, float("nan"))
                buf.write(f"  purity_{k:<30}  {v:.4f}\n" if not np.isnan(v)
                          else f"  purity_{k:<30}  n/a\n")
            for k in LVR_CROSS_PROPS:
                v = lvr_d.get(k, float("nan"))
                if not np.isnan(v):
                    buf.write(f"  lvr_{k:<33}  {v:.4f}\n")
        bb_txt.write_text(buf.getvalue())
        print(f"Saved -> {bb_txt}")

    # =========================================================================
    # Edge-specific analysis
    # =========================================================================
    # Imports used only in this section
    from sklearn.linear_model import LinearRegression as _LinReg
    from sklearn.metrics import r2_score as _r2_score
    from sklearn.manifold import trustworthiness as _trustworthiness

    # Helper: load an edge UMAP cache (returns None if not yet written)
    def _load_ec(layer: str):
        p = edges_dir / f"{layer}_edges_umap_cache.npz"
        if not p.exists():
            return None
        return np.load(p, allow_pickle=True)

    # Helper: boolean family mask for bond-type array
    def _fam_mask_ec(bond_types, name: str):
        parts = np.array([bt.split("-") for bt in bond_types])
        el, er = parts[:, 0], parts[:, 1]
        if name == "oxygen":     return (el == "O") | (er == "O")
        if name == "carbon":     return (el == "C") | (er == "C")
        if name == "homoatomic": return el == er
        raise ValueError(name)

    _FAM_NAMES = ["oxygen", "carbon", "homoatomic"]
    _MIN_R2_COUNT = 50  # minimum edges per bond type for R² estimate

    # ── Edge R² analysis (bond distance vs embedding) — all layers ──────────
    print("\n--- Edge R² analysis (bond distance → embedding) ---")
    for _layer in ALL_LAYERS:
        _r2_ov   = edges_dir / f"{_layer}_edge_dist_r2_overview.png"
        _r2_reps = edges_dir / f"{_layer}_edge_dist_r2_reps.png"
        if _r2_ov.exists() and _r2_reps.exists():
            continue

        _ec = _load_ec(_layer)
        if _ec is None:
            print(f"  [{_layer}] cache missing — skipping R² analysis")
            continue

        _X_raw_r = _ec["X_raw"].astype(np.float32)
        _norms_r = np.linalg.norm(_X_raw_r, axis=1, keepdims=True)
        _X_nr    = _X_raw_r / np.where(_norms_r == 0, 1, _norms_r)
        _bt_r    = _ec["bond_types"]
        _dists_r = _ec["distances"].astype(float)

        _r2_rows = []
        for _btype in np.unique(_bt_r):
            _mask_b = _bt_r == _btype
            if _mask_b.sum() < _MIN_R2_COUNT:
                continue
            _Xb = _X_nr[_mask_b]
            _d  = _dists_r[_mask_b]

            # R²(mean) — vectorised Pearson r² across all 256 dims
            _dc    = _d - _d.mean()
            _std_d = _dc.std()
            if _std_d == 0:
                continue
            _Xbc     = _Xb - _Xb.mean(axis=0)
            _r_vec   = (_dc @ _Xbc) / (len(_dc) * _std_d * (_Xb.std(axis=0) + 1e-12))
            _r2_mean = float((_r_vec ** 2).mean())

            # R²(PC1) — fit distance → first PC of that bond type
            _pc1 = PCA(n_components=1, random_state=42).fit_transform(_Xb).ravel()
            _lr  = _LinReg().fit(_d.reshape(-1, 1), _pc1)
            _r2p = float(_r2_score(_pc1, _lr.predict(_d.reshape(-1, 1))))

            _r2_rows.append((_btype, int(_mask_b.sum()), _r2_mean, _r2p))

        if not _r2_rows:
            print(f"  [{_layer}] no bond types with >= {_MIN_R2_COUNT} edges — skipping")
            continue

        _r2_rows.sort(key=lambda r: r[3], reverse=True)
        print(f"\n  [{_layer}] R² top-10 bond types:")
        print(f"    {'bond_type':<14}  {'n':>6}  {'R²(mean)':>9}  {'R²(PC1)':>8}")
        for _bt2, _cn2, _rm2, _rp2 in _r2_rows[:10]:
            print(f"    {_bt2:<14}  {_cn2:>6}  {_rm2:>9.4f}  {_rp2:>8.4f}")

        _r2pc1_arr  = np.array([r[3] for r in _r2_rows])
        _r2mn_arr   = np.array([r[2] for r in _r2_rows])
        _n_arr2     = np.array([r[1] for r in _r2_rows])
        _blabels    = [r[0] for r in _r2_rows]
        _nc_r2      = mcolors.Normalize(vmin=_r2pc1_arr.min(), vmax=_r2pc1_arr.max())

        if not _r2_ov.exists():
            _fig1, (_a1, _a2) = plt.subplots(1, 2, figsize=(14, 5))
            _a1.hist(_r2pc1_arr, bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
            _a1.axvline(np.median(_r2pc1_arr), color="crimson", linestyle="--",
                        linewidth=1.5, label=f"median = {np.median(_r2pc1_arr):.3f}")
            _a1.set_xlabel("R²(PC1)")
            _a1.set_ylabel("# bond types")
            _a1.set_title(f"{_layer}: R²(distance→PC1) distribution\n"
                          f"({len(_r2_rows)} bond types, ≥{_MIN_R2_COUNT} edges each)")
            _a1.legend()

            _sc_r2 = _a2.scatter(_n_arr2, _r2pc1_arr, c=_r2pc1_arr, cmap="RdYlGn_r",
                                 norm=_nc_r2, s=20, alpha=0.85, edgecolors="none", zorder=3)
            _a2.set_xscale("log")
            _a2.set_xlabel("Number of edges (log scale)")
            _a2.set_ylabel("R²(PC1)")
            _a2.set_title(f"{_layer}: R²(PC1) vs bond-type frequency")
            _fig1.colorbar(_sc_r2, ax=_a2, label="R²(PC1)")
            for _idx2 in list(range(3)) + list(range(len(_r2_rows) - 3, len(_r2_rows))):
                _a2.annotate(_blabels[_idx2],
                             xy=(_n_arr2[_idx2], _r2pc1_arr[_idx2]),
                             xytext=(4, 2), textcoords="offset points", fontsize=7)
            _fig1.tight_layout()
            _fig1.savefig(_r2_ov, dpi=150, bbox_inches="tight")
            plt.close(_fig1)
            print(f"  Saved -> {_r2_ov}")

        if not _r2_reps.exists():
            _nr2 = len(_r2_rows)
            _tiers = [
                ("high",   _r2_rows[:3]),
                ("medium", _r2_rows[_nr2 // 2 - 1: _nr2 // 2 + 2]),
                ("low",    _r2_rows[-3:]),
            ]
            _fig2, _axes2 = plt.subplots(3, 3, figsize=(12, 10))
            _fig2.suptitle(
                f"{_layer}: distance vs PC1 per bond type\n"
                "(slope = distance explains PC1; flat = something else drives variation)",
                fontsize=12,
            )
            for _ri2, (_tier_name, _tier) in enumerate(_tiers):
                for _ci2, (_btype2, _cn2, _rm2, _rp2) in enumerate(_tier):
                    _mask2 = _bt_r == _btype2
                    _Xb2   = _X_nr[_mask2]
                    _d2    = _dists_r[_mask2]
                    _pc1v  = PCA(n_components=1, random_state=42).fit_transform(_Xb2).ravel()
                    _axc2  = _axes2[_ri2][_ci2]
                    _d2n   = mcolors.Normalize(vmin=_d2.min(), vmax=_d2.max())
                    _sc3   = _axc2.scatter(_d2, _pc1v, c=_d2, cmap="viridis",
                                           norm=_d2n, s=8, alpha=0.6, edgecolors="none")
                    _dr    = np.linspace(_d2.min(), _d2.max(), 200)
                    _reg2  = _LinReg().fit(_d2.reshape(-1, 1), _pc1v)
                    _axc2.plot(_dr, _reg2.predict(_dr.reshape(-1, 1)),
                               color="crimson", linewidth=1.5)
                    _fig2.colorbar(_sc3, ax=_axc2, label="Distance (Å)")
                    _axc2.set_title(f"{_btype2}\nR²(PC1)={_rp2:.3f}  n={_cn2}", fontsize=8)
                    _axc2.set_xlabel("distance (Å)")
                    _axc2.set_ylabel("PC1")
            _fig2.tight_layout()
            _fig2.savefig(_r2_reps, dpi=150, bbox_inches="tight")
            plt.close(_fig2)
            print(f"  Saved -> {_r2_reps}")

    # ── Per-family final-layer (gnn_4) PCA + UMAP ───────────────────────────
    print("\n--- Per-family edge PCA + UMAP (gnn_4) ---")
    _bb_cache_ec = _load_ec(FINAL_LAYER)
    if _bb_cache_ec is not None:
        _bb_X_raw_f = _bb_cache_ec["X_raw"].astype(np.float32)
        _bb_bt_f    = _bb_cache_ec["bond_types"]
        _bb_mids_f  = _bb_cache_ec["mp_ids"]
        _bb_norms_f = np.linalg.norm(_bb_X_raw_f, axis=1)
        _bb_dists_f = _bb_cache_ec["distances"].astype(float)
        _bb_Xn_f    = _bb_X_raw_f / np.where(_bb_norms_f[:, None] == 0, 1, _bb_norms_f[:, None])

        for _fname in _FAM_NAMES:
            _fam_cache = edges_dir / f"family_{_fname}_umap_cache.npz"
            if _fam_cache.exists():
                print(f"  [{_fname}] cache exists — skipping")
                continue

            _fm_f  = _fam_mask_ec(_bb_bt_f, _fname)
            _Xf_f  = _bb_Xn_f[_fm_f]
            if len(_Xf_f) < 20:
                print(f"  [{_fname}] only {len(_Xf_f)} edges — skipping")
                continue
            _btf_f = _bb_bt_f[_fm_f]
            _nf_f  = _bb_norms_f[_fm_f]
            _df_f  = _bb_dists_f[_fm_f]
            _mf_f  = _bb_mids_f[_fm_f]

            _n_full_f = min(_Xf_f.shape[0], _Xf_f.shape[1])
            _pca_fv   = PCA(n_components=_n_full_f, random_state=42)
            _pca_fv.fit(_Xf_f)
            _evr_fv  = _pca_fv.explained_variance_ratio_
            _cumv_f  = np.cumsum(_evr_fv)
            _npcs_f  = int((_cumv_f < 0.90).sum()) + 1
            print(f"  [{_fname}]  {len(_Xf_f):,} edges  |  90 % var → {_npcs_f} PCs")

            _pca_fr = PCA(n_components=_npcs_f, random_state=42)
            _Xpca_f = _pca_fr.fit_transform(_Xf_f)

            _UMAP_NN_FAM = 15
            print(f"    UMAP n_neighbors={_UMAP_NN_FAM}  {_npcs_f} PCs → 2D ...")
            _reducer_f = UMAP(n_components=2, n_neighbors=_UMAP_NN_FAM, min_dist=0.1,
                              metric="euclidean", init="random", random_state=42,
                              low_memory=True, verbose=False)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="n_jobs value 1 overridden")
                _Xumap_f = _reducer_f.fit_transform(_Xpca_f)

            _evr_save_f = np.zeros(256, dtype=float)
            _evr_save_f[:len(_evr_fv)] = _evr_fv

            np.savez(_fam_cache,
                     X_umap=_Xumap_f, X_pca_umap=_Xpca_f,
                     evr=_evr_save_f, n_pcs=np.array(_npcs_f),
                     umap_n_neighbors=np.array(_UMAP_NN_FAM),
                     bond_types=_btf_f, norms=_nf_f,
                     distances=_df_f, mp_ids=_mf_f)
            print(f"    Saved -> {_fam_cache}")
    else:
        print("  gnn_4 edge cache not found — skipping family UMAP")

    # ── Per-family quality checks (4-panel) ──────────────────────────────────
    print("\n--- Per-family edge quality checks ---")
    _TOP_NORM_N_F = 12
    _SUBSAMPLE_N_F = 2000
    _RNG_QC_F = np.random.default_rng(42)

    for _fname_qc in _FAM_NAMES:
        _qc_out = edges_dir / f"edge_quality_family_{_fname_qc}.png"
        if _qc_out.exists():
            continue
        _fc_path = edges_dir / f"family_{_fname_qc}_umap_cache.npz"
        if not _fc_path.exists():
            print(f"  [{_fname_qc}] cache not found — run family UMAP first")
            continue

        _cqc      = np.load(_fc_path, allow_pickle=True)
        _Xumap_qc = _cqc["X_umap"]
        _Xpca_qc  = _cqc["X_pca_umap"]
        _evr_qc   = _cqc["evr"]
        _npcs_qc  = int(_cqc["n_pcs"])
        _unn_qc   = int(_cqc["umap_n_neighbors"])
        _bt_qc    = _cqc["bond_types"]
        _norms_qc = _cqc["norms"]
        _cumvar_qc = np.cumsum(_evr_qc)
        _K_fam_qc  = max(10, min(len(_Xumap_qc) // 10, 500))

        print(f"\n  [{_fname_qc}]  {len(_Xumap_qc):,} edges  |  {_npcs_qc} PCs (90 % var)")
        for _thr in [0.80, 0.90, 0.95]:
            _n_at = int((_cumvar_qc < _thr).sum()) + 1
            print(f"    {_thr:.0%} variance @ PC {_n_at}")

        # KNN preservation — subsample
        _nsub_qc  = min(_SUBSAMPLE_N_F, len(_Xpca_qc))
        _sidx_qc  = _RNG_QC_F.choice(len(_Xpca_qc), size=_nsub_qc, replace=False)
        _Xpca_sub = _Xpca_qc[_sidx_qc]
        _Xump_sub = _Xumap_qc[_sidx_qc]

        _K_vals_qc = sorted(set([5, 10, 15, 20, 30, _K_fam_qc]))
        _kmax_qc   = max(_K_vals_qc)
        _nbrs_pca_qc  = NearestNeighbors(n_neighbors=_kmax_qc + 1).fit(_Xpca_sub)
        _nbrs_umap_qc = NearestNeighbors(n_neighbors=_kmax_qc + 1).fit(_Xump_sub)
        _, _pca_knn_qc  = _nbrs_pca_qc.kneighbors(_Xpca_sub)
        _, _umap_knn_qc = _nbrs_umap_qc.kneighbors(_Xump_sub)

        _pres_qc = []
        for _k_qc in _K_vals_qc:
            _hits_qc = sum(
                len(set(_pca_knn_qc[_i, 1:_k_qc + 1]) & set(_umap_knn_qc[_i, 1:_k_qc + 1]))
                for _i in range(_nsub_qc)
            ) / (_nsub_qc * _k_qc)
            _pres_qc.append(_hits_qc)
            _base_qc = _k_qc / (_nsub_qc - 1)
            _mkr     = " <- UMAP n_neighbors" if _k_qc == _unn_qc else ""
            print(f"    KNN preservation k={_k_qc:>2}: {_hits_qc:.1%}  "
                  f"(random {_base_qc:.1%}){_mkr}")

        _trust_qc = _trustworthiness(_Xpca_sub, _Xump_sub, n_neighbors=15)
        print(f"    Trustworthiness (k=15, n={_nsub_qc:,}): {_trust_qc:.4f}  "
              f"({'good' if _trust_qc > 0.9 else 'moderate' if _trust_qc > 0.8 else 'low'})")

        _fig_qc, _axes_qc = plt.subplots(2, 2, figsize=(13, 10))
        _fig_qc.suptitle(
            f"ORB v3 Edge Embeddings — {_fname_qc.capitalize()} family\n"
            f"PCA ({_npcs_qc} PCs, 90 % var) + UMAP (n_neighbors={_unn_qc}) Quality Checks",
            fontsize=13, y=1.01,
        )

        # A: scree
        _ax_qc = _axes_qc[0, 0]
        _ax_qc.bar(range(1, _npcs_qc + 1), _evr_qc[:_npcs_qc] * 100,
                   color="steelblue", width=0.9)
        _ax_qc.set_xlabel("Principal component")
        _ax_qc.set_ylabel("Explained variance (%)")
        _ax_qc.set_title(f"A: PCA scree plot ({_npcs_qc} retained PCs, 90 % var)")
        _ax_qc.set_xlim(0.5, _npcs_qc + 0.5)

        # B: cumulative variance
        _nfp = int((_evr_qc > 0).sum())
        _ax_qc = _axes_qc[0, 1]
        _ax_qc.plot(range(1, _nfp + 1), _cumvar_qc[:_nfp] * 100, color="steelblue", lw=1.5)
        _ax_qc.axvline(_npcs_qc, color="tomato", ls="--", lw=1.5,
                       label=f"{_npcs_qc} PCs ({_evr_qc[:_npcs_qc].sum():.1%})")
        for _tb, _cb in [(80, "purple"), (90, "orange"), (95, "green")]:
            _nat = int((_cumvar_qc < _tb / 100).sum()) + 1
            _ax_qc.axhline(_tb, color=_cb, ls=":", lw=1, label=f"{_tb} % @ PC {_nat}")
        _ax_qc.set_xlabel("Number of principal components")
        _ax_qc.set_ylabel("Cumulative explained variance (%)")
        _ax_qc.set_title("B: Cumulative explained variance")
        _ax_qc.legend(fontsize=8)
        _ax_qc.set_xlim(1, _nfp)
        _ax_qc.set_ylim(0, 101)

        # C: KNN preservation
        _ax_qc = _axes_qc[1, 0]
        _bsls  = [_k / (_nsub_qc - 1) * 100 for _k in _K_vals_qc]
        _ax_qc.plot(_K_vals_qc, [_p * 100 for _p in _pres_qc],
                    "o-", color="steelblue", lw=2, label="Observed")
        _ax_qc.plot(_K_vals_qc, _bsls, "s--", color="tomato", lw=1.5, label="Random baseline")
        _ax_qc.fill_between(_K_vals_qc, _bsls,
                            [_p * 100 for _p in _pres_qc], alpha=0.15, color="steelblue")
        _ax_qc.axvline(_unn_qc, color="darkorange", ls=":", lw=1.5,
                       label=f"UMAP n_neighbors={_unn_qc}")
        _ax_qc.set_xlabel("k (neighbours)")
        _ax_qc.set_ylabel("Fraction of neighbours preserved (%)")
        _ax_qc.set_title(
            f"C: KNN preservation: PCA vs UMAP-2\n"
            f"(subsample n={_nsub_qc:,}, UMAP n_neighbors={_unn_qc})"
        )
        _ax_qc.legend(fontsize=9)
        _ax_qc.set_ylim(0, 100)

        # D: norm distribution per bond type
        _ax_qc = _axes_qc[1, 1]
        from collections import Counter as _CtrQC
        _tc_qc  = _CtrQC(_bt_qc.tolist())
        _tops   = [_bt for _bt, _ in _tc_qc.most_common(_TOP_NORM_N_F)]
        _nbt    = [_norms_qc[_bt_qc == _bt] for _bt in _tops]
        _order_qc = np.argsort([np.median(_n3) for _n3 in _nbt])
        _ax_qc.boxplot(
            [_nbt[_i] for _i in _order_qc], vert=False, patch_artist=True,
            boxprops=dict(facecolor="steelblue", alpha=0.6),
            medianprops=dict(color="tomato", lw=2),
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
            tick_labels=[_tops[_i] for _i in _order_qc],
        )
        _ax_qc.set_xlabel("L2 norm of raw edge embedding")
        _ax_qc.set_title(
            f"D: Norm per bond type (top {_TOP_NORM_N_F} by count)\n"
            f"mean={_norms_qc.mean():.2f},  std={_norms_qc.std():.2f}"
        )
        _ax_qc.tick_params(axis="y", labelsize=8)

        plt.tight_layout()
        _fig_qc.savefig(_qc_out, dpi=150, bbox_inches="tight")
        plt.close(_fig_qc)
        print(f"  Saved -> {_qc_out}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
