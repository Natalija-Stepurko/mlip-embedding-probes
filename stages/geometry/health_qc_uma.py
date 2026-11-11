"""
stages/geometry/health_qc_uma.py — QC analyses for the FairChem UMA embedding pipeline

Production-scale port of orb_embedding_qc.ipynb. Operates on cached UMAP outputs
written by scripts/stages/geometry/organisation_uma.py:
  $MLIP_PROBES_ROOT/results/fairchem/plots/nodes/{layer}_{emb_key}_umap_cache.npz

Layers (6): embed, block_0..3, backbone
Embedding types (12): 6 per-atom (flat, ang_norm, l0, l1_norm, l2_norm, geom_norm)
                       + 6 pooled  (full, ang_norm, l0, l1_polar, l2_polar, geom_norm)

Analyses: same as the ORB QC, minus edges & bond-family sweep.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import argparse
import gc
import io
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["legend.frameon"] = False  # no box border around figure legends
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlipprobes import common as qc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_LAYERS = ["embed", "block_0", "block_1", "block_2", "block_3", "backbone"]
LAYER_LBLS = ["embed", "blk_0", "blk_1", "blk_2", "blk_3", "BB"]

PER_ATOM_KEYS = ["nodes_flat", "nodes_ang_norm", "nodes_l0",
                  "nodes_l1_norm", "nodes_l2_norm", "nodes_geom_norm"]
POOLED_KEYS   = ["nodes_pooled_full", "nodes_pooled_ang_norm", "nodes_pooled_l0",
                  "nodes_pooled_l1_polar", "nodes_pooled_l2_polar", "nodes_pooled_geom_norm"]
ALL_EMB_KEYS  = PER_ATOM_KEYS + POOLED_KEYS

EMB_LABELS = {
    "nodes_flat":              "Nodes flat",
    "nodes_ang_norm":          "Nodes ang-norm",
    "nodes_l0":                "Nodes L=0",
    "nodes_l1_norm":           "Nodes L=1 norm",
    "nodes_l2_norm":           "Nodes L=2 norm",
    "nodes_geom_norm":         "Nodes geom-norm",
    "nodes_pooled_full":       "Pooled full",
    "nodes_pooled_ang_norm":   "Pooled ang-norm",
    "nodes_pooled_l0":         "Pooled L=0",
    "nodes_pooled_l1_polar":   "Pooled L=1 polar",
    "nodes_pooled_l2_polar":   "Pooled L=2 polar",
    "nodes_pooled_geom_norm":  "Pooled geom-norm",
}
EMB_COLORS = {
    "nodes_flat":             "mediumpurple",
    "nodes_ang_norm":         "darkorange",
    "nodes_l0":               "steelblue",
    "nodes_l1_norm":          "tomato",
    "nodes_l2_norm":          "forestgreen",
    "nodes_geom_norm":        "saddlebrown",
    "nodes_pooled_full":      "mediumorchid",
    "nodes_pooled_ang_norm":  "goldenrod",
    "nodes_pooled_l0":        "teal",
    "nodes_pooled_l1_polar":  "#e377c2",
    "nodes_pooled_l2_polar":  "#7f7f7f",
    "nodes_pooled_geom_norm": "#bcbd22",
}

PUR_PROPS_ATOM   = ["crystal_system", "metal_insulator", "magnetic_label", "element_identity"]
PUR_PROPS_POOL   = ["crystal_system", "metal_insulator", "magnetic_label"]
LVR_PROPS        = list(qc.CONT_PROPS)  # 9 continuous

LAYER_COLORS = [plt.get_cmap("plasma")(i / (len(ALL_LAYERS) - 1)) for i in range(len(ALL_LAYERS))]

# ---- Edge embedding types (5 layers: embed=edge-degree + block_0..3; no backbone) ----
EDGE_LAYERS = ["embed", "block_0", "block_1", "block_2", "block_3"]
EDGE_LAYER_LBLS = ["embed", "blk_0", "blk_1", "blk_2", "blk_3"]
EDGE_LAYER_COLORS = [plt.get_cmap("plasma")(i / (len(EDGE_LAYERS) - 1)) for i in range(len(EDGE_LAYERS))]
_EDGE_VARIANTS = ["flat", "ang_norm", "l0", "l1_norm", "l2_norm", "geom_norm"]
EDGE_ATOM_KEYS   = [f"edges_{v}" for v in _EDGE_VARIANTS]
EDGE_POOLED_KEYS = [f"edges_pooled_{v}" for v in _EDGE_VARIANTS]
ALL_EDGE_KEYS    = EDGE_ATOM_KEYS + EDGE_POOLED_KEYS
_EDGE_COLORS = {"flat": "saddlebrown", "ang_norm": "darkorange", "l0": "steelblue",
                "l1_norm": "tomato", "l2_norm": "forestgreen", "geom_norm": "mediumorchid"}
for _k in ALL_EDGE_KEYS:
    _v = _k.replace("edges_pooled_", "").replace("edges_", "")
    EMB_LABELS[_k] = ("Edges pooled " if "pooled" in _k else "Edges ") + _v.replace("_", " ")
    EMB_COLORS[_k] = _EDGE_COLORS[_v]

# Per-edge purity adds bond_type; per-edge LVR adds bond_distance.
PUR_PROPS_EDGE = ["bond_type", "crystal_system", "metal_insulator", "magnetic_label"]


def _is_edge(et: str) -> bool:
    return et.startswith("edges_")


def _is_pooled(et: str) -> bool:
    return et in POOLED_KEYS or et in EDGE_POOLED_KEYS


def _layers_for(et: str) -> list[str]:
    return EDGE_LAYERS if _is_edge(et) else ALL_LAYERS


def _layer_lbls_for(et: str) -> list[str]:
    return EDGE_LAYER_LBLS if _is_edge(et) else LAYER_LBLS


def _ref_layer(et: str) -> str:
    """Cross-layer reference cache (defines query/hdbscan slices). Edges have no
    backbone; their caches are row-aligned across layers, so use the deepest block."""
    return "block_3" if _is_edge(et) else "backbone"


def cache_path(plots_dir: Path, layer: str, et: str) -> Path:
    sub = "edges" if _is_edge(et) else "nodes"
    return plots_dir / sub / f"{layer}_{et}_umap_cache.npz"


# ---------------------------------------------------------------------------
# Dataset counts pass (atoms only for FairChem)
# ---------------------------------------------------------------------------

def collect_dataset_counts(emb_dir: Path, valid_ids: set, qc_dir: Path,
                              structs: dict | None = None,
                              plots_dir: Path | None = None) -> dict:
    """Atom counts (manifest nsites) + true bond counts (full-graph edge census)."""
    out = qc_dir / "dataset_counts.json"
    if out.exists():
        return json.loads(out.read_text())
    if structs is None:
        raise ValueError("structs required")
    # True bonds/families from the census written by the organisation script (if present).
    bonds_by_mat: dict[str, int] = {}
    fam_by_mat: dict[str, dict[str, int]] = {}
    bonds_is_proxy = True
    census = (plots_dir / "edges" / "edge_census.npz") if plots_dir is not None else None
    if census is not None and census.exists():
        cen = np.load(census, allow_pickle=True)
        cm = cen["mp_ids"].astype(str)
        cb = cen["n_bonds"]; co = cen["n_oxy"]; cc = cen["n_carbon"]; ch = cen["n_homo"]
        for i, m in enumerate(cm):
            bonds_by_mat[m] = int(cb[i])
            fam_by_mat[m] = {"oxygen": int(co[i]), "carbon": int(cc[i]),
                             "homoatomic": int(ch[i])}
        bonds_is_proxy = False
    counts: dict[str, dict] = {}
    bond_families = {"oxygen": [], "carbon": [], "homoatomic": []}
    have_bonds = len(bonds_by_mat) > 0
    for m in valid_ids:
        if m not in structs:
            continue
        n_at = structs[m].get("nsites")
        if n_at is None:
            continue
        rec = {"n_atoms": int(n_at)}
        if have_bonds:
            fam = fam_by_mat.get(m, {"oxygen": 0, "carbon": 0, "homoatomic": 0})
            rec["n_bonds"] = bonds_by_mat.get(m, 0)
            bond_families["oxygen"].append(fam["oxygen"])
            bond_families["carbon"].append(fam["carbon"])
            bond_families["homoatomic"].append(fam["homoatomic"])
        counts[m] = rec
    payload = {"per_material": counts, "has_bonds": have_bonds,
               "bonds_is_proxy": bonds_is_proxy, "bond_families": bond_families,
               "_note": "atoms/mat from manifest nsites"
                        + ("; bonds/mat and families from full-graph edge census"
                           if have_bonds else "")}
    out.write_text(json.dumps(payload))
    print(f"    Saved {out}", flush=True)
    return payload


def section_property_coverage(counts: dict, structs: dict) -> str:
    """% of datapoints (materials / nodes=atoms / edges=bonds) carrying each property."""
    per = counts["per_material"]
    has_bonds = counts.get("has_bonds", False)
    mids = list(per.keys())
    n_at = np.array([per[m]["n_atoms"] for m in mids], dtype=float)
    n_bo = (np.array([per[m].get("n_bonds", 0) for m in mids], dtype=float)
            if has_bonds else None)
    tot_mat = float(len(mids)); tot_at = float(n_at.sum())
    tot_bo = float(n_bo.sum()) if has_bonds else 0.0

    getters = [
        ("formation_energy/atom", lambda s: s.get("formation_energy_per_atom")),
        ("energy_above_hull",     lambda s: s.get("energy_above_hull")),
        ("band_gap",              lambda s: s.get("band_gap")),
        ("density",               lambda s: s.get("density")),
        ("bulk_modulus",          lambda s: s.get("bulk_modulus")),
        ("crystal_system",        lambda s: s.get("crystal_system")),
        ("metal_insulator",       lambda s: s.get("band_gap")),
        ("magnetic_label",        lambda s: s.get("total_magnetization")),
    ]
    buf = io.StringIO()
    buf.write("\n  Property coverage  (% of datapoints whose material has the property):\n")
    hdr = f"  {'property':<24}{'%_materials':>12}{'%_nodes':>10}"
    hdr += f"{'%_edges':>10}\n" if has_bonds else "\n"
    buf.write(hdr)
    for label, getter in getters:
        have = np.array([1.0 if (m in structs and getter(structs[m]) is not None) else 0.0
                         for m in mids])
        mat_c  = 100 * have.sum() / tot_mat if tot_mat else 0.0
        node_c = 100 * (have * n_at).sum() / tot_at if tot_at else 0.0
        row = f"  {label:<24}{mat_c:>11.1f}%{node_c:>9.1f}%"
        if has_bonds:
            edge_c = 100 * (have * n_bo).sum() / tot_bo if tot_bo else 0.0
            row += f"{edge_c:>9.1f}%"
        buf.write(row + "\n")
    note = "  %_nodes weights materials by atom count"
    note += "; %_edges by bond count.\n" if has_bonds else ".\n"
    buf.write("\n" + note)
    return buf.getvalue()


def section_dataset_stats(counts: dict, structs: dict) -> str:
    per = counts["per_material"]
    atoms = np.array([v["n_atoms"] for v in per.values()], dtype=float)
    buf = io.StringIO()
    buf.write("=" * 60 + "\nDataset Structure  (backbone .pt files)\n" + "=" * 60 + "\n")
    buf.write(f"  Materials: {len(atoms):,}\n\n")
    buf.write(f"  atoms/mat   mean={atoms.mean():.1f}  med={np.median(atoms):.0f}  "
              f"min={atoms.min():.0f}  max={atoms.max():.0f}  "
              f"p25={np.percentile(atoms,25):.0f}  p75={np.percentile(atoms,75):.0f}\n")
    if counts.get("has_bonds"):
        bonds = np.array([v.get("n_bonds", 0) for v in per.values()], dtype=float)
        buf.write(f"  bonds/mat   mean={bonds.mean():.1f}  med={np.median(bonds):.0f}  "
                  f"min={bonds.min():.0f}  max={bonds.max():.0f}  "
                  f"p25={np.percentile(bonds,25):.0f}  p75={np.percentile(bonds,75):.0f}"
                  f"   [full-graph edge census, from n_edges_total]\n")
        buf.write("    Note: UMA persisted only a <=48-edge/material sample for the "
                  "embedding analysis\n    (then a 300k sqrt-weighted UMAP subsample); "
                  "the true graph above matches ORB's.\n")
    buf.write(section_property_coverage(counts, structs))
    buf.write("\n")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_emb_type(et: str, emb_dir: Path, plots_dir: Path, qc_dir: Path,
                       jsonl_path: Path,
                       hdbscan_sample: int, knn_query_sample: int,
                       jaccard_k: int, geom_sample: int, seed: int,
                       progress_q) -> None:
    matplotlib.use("Agg")

    def log(msg: str) -> None:
        progress_q.put(f"[{et:<25}] {msg}")

    is_pooled = _is_pooled(et)
    structs, _ = qc.load_manifest(jsonl_path)
    cache_dir = qc_dir / "_cache"

    bb_path = cache_path(plots_dir, _ref_layer(et), et)
    if not bb_path.exists():
        log(f"reference cache missing → {bb_path}")
        progress_q.put(f"__DONE__{et}")
        return
    bb = np.load(bb_path, allow_pickle=True)
    n_total = int(bb["X_raw"].shape[0])
    log(f"reference cache ({_ref_layer(et)}): {n_total:,} rows × {bb['X_raw'].shape[1]}-d")

    q_path = cache_dir / f"{et}_query_idx.npy"
    if q_path.exists():
        query_idx = np.load(q_path)
    else:
        if is_pooled:
            q_n = knn_query_sample if (knn_query_sample and knn_query_sample < n_total) else n_total
        else:
            q_n = knn_query_sample if knn_query_sample else n_total
        query_idx = qc.uniform_idx(n_total, q_n, seed=seed)
        np.save(q_path, query_idx)
    log(f"query rows: {len(query_idx):,}")

    s_path = cache_dir / f"{et}_hdbscan_idx.npy"
    if s_path.exists():
        hdb_idx = np.load(s_path)
    else:
        if is_pooled:
            hdb_idx = np.arange(n_total)
        else:
            if "elements" in bb.files:
                cls_arr = bb["elements"]
            elif "bond_types" in bb.files:
                cls_arr = bb["bond_types"]
            else:
                cls_arr = bb["mp_ids"]
            n_target = hdbscan_sample if hdbscan_sample else n_total
            hdb_idx = qc.sqrt_weighted_idx(cls_arr, n_target, seed=seed)
        np.save(s_path, hdb_idx)
    log(f"hdbscan slice: {len(hdb_idx):,}")

    mp_ids_q = bb["mp_ids"][query_idx]
    if is_pooled:
        k_ref = qc.adaptive_k_value("sqrt_n_mats", len(query_idx), mp_ids_q)
    else:
        k_ref = qc.adaptive_k_value("p75_pts_per_mat", len(query_idx), mp_ids_q)
    # Adaptive K-sweep grid: always brackets k_ref with values before and after
    # it (k < index size), so the chosen K never sits at the plot's edge.
    Ks = qc.adaptive_k_grid(k_ref, n_total)
    K_MAX = max(Ks + [jaccard_k])
    log(f"K-sweep set: {Ks}   (adaptive K_ref={k_ref})")

    hdb_mids = bb["mp_ids"][hdb_idx]
    mcs, ms = qc.adaptive_hdbscan_params(hdb_mids, n_points=len(hdb_idx))
    log(f"HDBSCAN params (on slice): MCS={mcs}, MS={ms}")

    md = qc.mag_density_dict(structs)
    cont_dicts = {
        "formation_energy/atom": qc.prop_dict(structs, "formation_energy_per_atom"),
        "energy_above_hull":     qc.prop_dict(structs, "energy_above_hull"),
        "band_gap":              qc.prop_dict(structs, "band_gap"),
        "density":               qc.prop_dict(structs, "density"),
        "bulk_modulus":          qc.prop_dict(structs, "bulk_modulus"),
        "mag_density":           md,
    }

    del bb
    gc.collect()

    for layer in _layers_for(et):
        t0 = time.time()
        log(f"--- {layer} ---")
        path = cache_path(plots_dir, layer, et)
        if not path.exists():
            log("  cache missing, skipping"); continue
        sweep_p = cache_dir / f"{layer}_{et}_sweep.json"
        geom_p  = cache_dir / f"{layer}_{et}_geom.npz"
        knn_p   = cache_dir / f"{layer}_{et}_knn_idx.npz"
        hdb_p   = cache_dir / f"{layer}_{et}_hdbscan.npz"

        need_sweep = not sweep_p.exists()
        need_geom  = not geom_p.exists()
        need_knn   = not knn_p.exists()
        need_hdb   = not hdb_p.exists()

        if not any([need_sweep, need_geom, need_knn, need_hdb]):
            log(f"  all cached ({time.time()-t0:.0f}s)"); continue

        cache = np.load(path, allow_pickle=True)
        X_raw = cache["X_raw"]
        X_pca = cache["X_pca"]
        mp_ids = cache["mp_ids"]
        elements = cache["elements"] if "elements" in cache.files else None
        labels_arr = cache["labels"] if "labels" in cache.files else None
        bond_types = cache["bond_types"] if "bond_types" in cache.files else None
        distances = cache["distances"] if "distances" in cache.files else None

        if need_geom:
            lam = qc.spectrum_eigvals(X_raw)
            cos_m, cos_s = qc.mean_pairwise_cosine(X_raw, max_n=geom_sample, seed=seed)
            np.savez(geom_p, lam=lam, cos_mean=cos_m, cos_std=cos_s)
            log(f"  geometry: spectrum({len(lam)}), cos={cos_m:.4f}±{cos_s:.4f}")

        if need_knn:
            log(f"  building kNN: index={len(X_raw):,} × {X_raw.shape[1]}d, queries={len(query_idx):,}, K_max={K_MAX}")
            t_knn = time.time()
            nn_idx = qc.build_knn_indices(X_raw, query_idx, K_MAX, l2_normalize=False)
            np.savez(knn_p, nn_idx=nn_idx, query_idx=query_idx)
            log(f"  kNN built ({time.time()-t_knn:.0f}s, shape={nn_idx.shape})")
        else:
            nn_idx = np.load(knn_p)["nn_idx"]

        if need_sweep:
            t_sw = time.time()
            mp_q = mp_ids[query_idx]
            mp_full = mp_ids
            cs_full = labels_arr if labels_arr is not None else qc.cat_array(structs, mp_full, "crystal_system")
            mi_full = qc.cat_array(structs, mp_full, "metal_insulator")
            mg_full = qc.cat_array(structs, mp_full, "magnetic_label")
            cs_q = cs_full[query_idx]; mi_q = mi_full[query_idx]; mg_q = mg_full[query_idx]

            pur: dict = {}
            pur["crystal_system"]  = qc.sweep_purity(cs_q, cs_full, nn_idx, Ks)
            pur["metal_insulator"] = qc.sweep_purity(mi_q, mi_full, nn_idx, Ks)
            pur["magnetic_label"]  = qc.sweep_purity(mg_q, mg_full, nn_idx, Ks)
            if not is_pooled and elements is not None:
                pur["element_identity"] = qc.sweep_purity(elements[query_idx], elements, nn_idx, Ks)
            if not is_pooled and bond_types is not None:
                pur["bond_type"] = qc.sweep_purity(bond_types[query_idx], bond_types, nn_idx, Ks)

            lvr: dict = {}
            for prop_name, vd in cont_dicts.items():
                lvr[prop_name] = qc.sweep_lvr_dict(vd, mp_q, mp_full, nn_idx, Ks)
            if not is_pooled and distances is not None:
                lvr["bond_distance"] = qc.sweep_lvr_arr(distances[query_idx], distances, nn_idx, Ks)

            payload = {
                "Ks": Ks, "k_ref": int(k_ref),
                "pur": {k: {str(kk): v for kk, v in m.items()} for k, m in pur.items()},
                "lvr": {k: {str(kk): v for kk, v in m.items()} for k, m in lvr.items()},
            }
            sweep_p.write_text(json.dumps(payload))
            log(f"  sweep done ({time.time()-t_sw:.0f}s)")

        if need_hdb:
            t_h = time.time()
            X_slice = X_pca[hdb_idx]
            labels = qc.run_hdbscan(X_slice, mcs, ms)
            valid = labels >= 0
            n_clust = int(labels.max()) + 1 if valid.any() else 0
            noise = float((~valid).mean())
            X_umap_slice = cache["X_umap"][hdb_idx]
            np.savez(hdb_p, labels=labels, slice_idx=hdb_idx, X_umap=X_umap_slice,
                      mcs=mcs, ms=ms, n_clusters=n_clust, noise_frac=noise)
            log(f"  HDBSCAN: clusters={n_clust}, noise={noise:.1%} ({time.time()-t_h:.0f}s)")

        del cache, X_raw, X_pca
        gc.collect()
        log(f"  [{layer}] done ({time.time()-t0:.0f}s)")

    progress_q.put(f"__DONE__{et}")


# ---------------------------------------------------------------------------
# Renderers (mostly mirror ORB; cross-type plots use per-atom row + pooled row)
# ---------------------------------------------------------------------------

def load_sweep(qc_dir: Path, et: str, layer: str) -> dict | None:
    p = qc_dir / "_cache" / f"{layer}_{et}_sweep.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    d["pur"] = {k: {int(kk): v for kk, v in m.items()} for k, m in d["pur"].items()}
    d["lvr"] = {k: {int(kk): v for kk, v in m.items()} for k, m in d["lvr"].items()}
    return d


def render_sweep_figure(qc_dir: Path, et: str) -> tuple[int, str]:
    out_p = qc_dir / f"knn_stability_{et}.png"
    is_p = _is_pooled(et)
    if is_p:
        pur_props = PUR_PROPS_POOL
    elif _is_edge(et):
        pur_props = PUR_PROPS_EDGE
    else:
        pur_props = PUR_PROPS_ATOM
    lvr_props = LVR_PROPS + (["bond_distance"] if (_is_edge(et) and not is_p) else [])
    layers = _layers_for(et)
    lbls = _layer_lbls_for(et)
    colors = EDGE_LAYER_COLORS if _is_edge(et) else LAYER_COLORS
    pur_all: dict = {p: {} for p in pur_props}
    lvr_all: dict = {p: {} for p in lvr_props}
    k_ref = 15
    for layer, lbl in zip(layers, lbls):
        d = load_sweep(qc_dir, et, layer)
        if not d:
            continue
        k_ref = d["k_ref"]
        for p in pur_props:
            pur_all[p][lbl] = d["pur"].get(p, {})
        for p in lvr_props:
            lvr_all[p][lbl] = d["lvr"].get(p, {})
    if not out_p.exists():
        qc.plot_k_sweep(pur_all, lvr_all, pur_props, lvr_props, k_ref,
                          f"FairChem UMA — K-Stability — {EMB_LABELS[et]}",
                          out_p, lbls, colors, xscale="linear")
    return k_ref, qc.ksweep_table_text(pur_all, lvr_all, pur_props, lvr_props,
                                          k_ref, lbls, f"K-sweep: {et}")


def _two_row_cross_type(values_per_type: dict[str, np.ndarray],
                          ylabel: str, title: str, out_path: Path,
                          stds: dict[str, np.ndarray] | None = None) -> None:
    """Two stacked panels: per-atom row over pooled row."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.arange(len(ALL_LAYERS))
    for ax, group, label in [(axes[0], PER_ATOM_KEYS, "per-atom"),
                              (axes[1], POOLED_KEYS, "pooled")]:
        for et in group:
            v = values_per_type.get(et)
            if v is None:
                continue
            ax.plot(x, v, "o-", color=EMB_COLORS[et], lw=2, ms=6, label=EMB_LABELS[et])
            if stds is not None and stds.get(et) is not None:
                s = stds[et]
                ax.fill_between(x, v - s, v + s, color=EMB_COLORS[et], alpha=0.10)
        ax.set_xticks(x); ax.set_xticklabels(LAYER_LBLS)
        ax.set_ylabel(ylabel)
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=3, loc="best")
    axes[1].set_xlabel("Layer")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_pca_dim(qc_dir: Path, plots_dir: Path, out_path: Path) -> str:
    if out_path.exists():
        return ""
    vals: dict[str, np.ndarray] = {}
    for et in ALL_EMB_KEYS:
        v = []
        for layer in ALL_LAYERS:
            p = cache_path(plots_dir, layer, et)
            if p.exists():
                evr = np.load(p, allow_pickle=True)["evr"]
                v.append(qc.pca_n_for_variance(evr))
            else:
                v.append(np.nan)
        vals[et] = np.array(v, dtype=float)
    _two_row_cross_type(vals, "# PCs (90% var)",
                          "FairChem UMA — PCA Dimensionality Across Layers", out_path)
    buf = io.StringIO()
    buf.write("=" * 60 + "\nPCA Dimensionality (90% variance) — All Layers\n" + "=" * 60 + "\n")
    buf.write(f"  {'layer':<8}" + "".join(f"{EMB_LABELS[et][:10]:>12}" for et in ALL_EMB_KEYS) + "\n")
    for j, lbl in enumerate(LAYER_LBLS):
        line = f"  {lbl:<8}"
        for et in ALL_EMB_KEYS:
            v = vals[et][j]
            line += f"{v:>12.0f}" if not np.isnan(v) else f"{'N/A':>12}"
        buf.write(line + "\n")
    return buf.getvalue() + "\n"


def render_cosine(qc_dir: Path, out_path: Path) -> str:
    if out_path.exists():
        return ""
    means: dict[str, np.ndarray] = {}
    stds:  dict[str, np.ndarray] = {}
    for et in ALL_EMB_KEYS:
        m = []; s = []
        for layer in ALL_LAYERS:
            p = qc_dir / "_cache" / f"{layer}_{et}_geom.npz"
            if p.exists():
                d = np.load(p)
                m.append(float(d["cos_mean"])); s.append(float(d["cos_std"]))
            else:
                m.append(np.nan); s.append(np.nan)
        means[et] = np.array(m); stds[et] = np.array(s)
    _two_row_cross_type(means, "Mean cosine sim",
                          "FairChem UMA — Q1 Collapse: Mean Pairwise Cosine ±std (X_raw)",
                          out_path, stds=stds)
    return ""


def render_spectrum(qc_dir: Path, out_path: Path) -> str:
    if out_path.exists():
        return ""
    # Build 2x2 grid: top row concentration (per-atom | pooled); bottom row κ (log)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    x = np.arange(len(ALL_LAYERS))
    buf = io.StringIO()
    buf.write("=" * 60 + "\nSpectrum Concentration & Condition Number\n" + "=" * 60 + "\n")
    for col, (group, gname) in enumerate([(PER_ATOM_KEYS, "per-atom"), (POOLED_KEYS, "pooled")]):
        for et in group:
            top1 = []; top10 = []; kappa = []
            for layer in ALL_LAYERS:
                p = qc_dir / "_cache" / f"{layer}_{et}_geom.npz"
                if p.exists():
                    lam = np.load(p)["lam"]
                    a, b = qc.concentration(lam)
                    k = qc.condition_number(lam)
                else:
                    a = b = k = np.nan
                top1.append(a); top10.append(b); kappa.append(k)
            axes[0, col].plot(x, top1, "o-", color=EMB_COLORS[et], lw=1.8, ms=5,
                                label=f"{EMB_LABELS[et]} (λ₁/Σλ)")
            axes[0, col].plot(x, top10, "o--", color=EMB_COLORS[et], lw=1.2, ms=4, alpha=0.6)
            axes[1, col].plot(x, kappa, "o-", color=EMB_COLORS[et], lw=1.8, ms=5,
                                label=EMB_LABELS[et])
        axes[0, col].set_xticks(x); axes[0, col].set_xticklabels(LAYER_LBLS)
        axes[0, col].set_ylim(0, 1.02); axes[0, col].grid(alpha=0.3)
        axes[0, col].set_title(f"Spectrum Concentration ({gname})\nsolid=λ₁/Σλ, dashed=top10/Σλ")
        axes[0, col].set_ylabel("Variance share")
        axes[0, col].legend(fontsize=6, ncol=2, loc="best")
        axes[1, col].set_xticks(x); axes[1, col].set_xticklabels(LAYER_LBLS)
        axes[1, col].set_yscale("log"); axes[1, col].grid(alpha=0.3, which="both")
        axes[1, col].set_title(f"Condition Number κ ({gname})")
        axes[1, col].set_ylabel("κ = λ₁/λ_r")
        axes[1, col].legend(fontsize=7, ncol=2, loc="best")
    fig.suptitle("FairChem UMA — Q2 Anisotropy: Spectrum Concentration & Condition Number")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue() + "  (see spectrum_concentration.png)\n\n"


def render_jaccard(qc_dir: Path, out_path: Path, jaccard_k: int) -> str:
    if out_path.exists():
        return ""
    mu_d: dict[str, np.ndarray] = {}
    sd_d: dict[str, np.ndarray] = {}
    pair_lbls = [f"{LAYER_LBLS[i]}→{LAYER_LBLS[i+1]}" for i in range(len(ALL_LAYERS) - 1)]
    for et in ALL_EMB_KEYS:
        mus = []; sds = []
        for i in range(len(ALL_LAYERS) - 1):
            la, lb = ALL_LAYERS[i], ALL_LAYERS[i + 1]
            pa = qc_dir / "_cache" / f"{la}_{et}_knn_idx.npz"
            pb = qc_dir / "_cache" / f"{lb}_{et}_knn_idx.npz"
            if pa.exists() and pb.exists():
                ia = np.load(pa)["nn_idx"][:, :jaccard_k]
                ib = np.load(pb)["nn_idx"][:, :jaccard_k]
                mu, sd = qc.jaccard_rows(ia, ib)
                mus.append(mu); sds.append(sd)
            else:
                mus.append(np.nan); sds.append(np.nan)
        mu_d[et] = np.array(mus); sd_d[et] = np.array(sds)
    # 2-row layout: per-atom row + pooled row
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.arange(len(pair_lbls))
    for ax, group, label in [(axes[0], PER_ATOM_KEYS, "per-atom"),
                              (axes[1], POOLED_KEYS, "pooled")]:
        for et in group:
            m = mu_d.get(et); s = sd_d.get(et)
            if m is None:
                continue
            ax.plot(x, m, "o-", color=EMB_COLORS[et], lw=2, ms=6, label=EMB_LABELS[et])
            ax.fill_between(x, m - s, m + s, color=EMB_COLORS[et], alpha=0.10)
        ax.set_xticks(x); ax.set_xticklabels(pair_lbls, rotation=15, ha="right")
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
        ax.set_title(label); ax.set_ylabel(f"Jaccard (K={jaccard_k})")
        ax.legend(fontsize=7, ncol=3, loc="best")
    axes[1].set_xlabel("Adjacent layer pair")
    fig.suptitle("FairChem UMA — Q3 Stability: kNN Jaccard Overlap Across Adjacent Layers")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ""


def render_cka_svcca(plots_dir: Path, qc_dir: Path,
                       out_cka: Path, out_svcca: Path,
                       cka_sample: int, svcca_sample: int, svcca_var: float,
                       seed: int) -> tuple[str, str]:
    """Render two separate figures per metric: per-atom and pooled."""
    n_lay = len(ALL_LAYERS)
    text_cka = text_svcca = ""

    for group, gname in [(PER_ATOM_KEYS, "atom"), (POOLED_KEYS, "pooled")]:
        cka_path = qc_dir / f"cka_heatmaps_{gname}.png"
        svcca_path = qc_dir / f"svcca_{gname}.png"

        if cka_path.exists() and svcca_path.exists():
            continue

        cka_mats: dict[str, np.ndarray] = {}
        svcca_mats: dict[str, np.ndarray] = {}
        for et in group:
            is_pooled_t = et in POOLED_KEYS
            def resolver(layer: str, _et=et) -> Path:
                return cache_path(plots_dir, layer, _et)
            et_emb = qc.EmbType(key=et, label=EMB_LABELS[et], color=EMB_COLORS[et],
                                  kind="pooled" if is_pooled_t else "atom",
                                  l2_normalize=False)
            if not cka_path.exists():
                cka_n = 0 if is_pooled_t else cka_sample
                mats, _ = qc.load_aligned_raw_layers(et_emb, ALL_LAYERS, resolver,
                                                      row_subsample=cka_n, seed=seed)
                M = np.full((n_lay, n_lay), np.nan)
                for a in range(n_lay):
                    for b in range(a, n_lay):
                        if mats[a] is None or mats[b] is None:
                            continue
                        v = qc.linear_cka(mats[a], mats[b])
                        M[a, b] = M[b, a] = v
                cka_mats[et] = M
                del mats; gc.collect()
            if not svcca_path.exists():
                mats, _ = qc.load_aligned_raw_layers(et_emb, ALL_LAYERS, resolver,
                                                      row_subsample=svcca_sample, seed=seed)
                M = np.full((n_lay, n_lay), np.nan)
                for a in range(n_lay):
                    for b in range(a, n_lay):
                        if mats[a] is None or mats[b] is None:
                            continue
                        v = qc.svcca(mats[a], mats[b], var=svcca_var, max_rows=0, seed=seed)
                        M[a, b] = M[b, a] = v
                svcca_mats[et] = M
                del mats; gc.collect()
        if cka_mats:
            qc.plot_cross_layer_heatmaps(cka_mats, EMB_LABELS, EMB_COLORS, LAYER_LBLS,
                                           f"FairChem UMA — Linear CKA Between Layers ({gname})",
                                           "Linear CKA", "viridis", cka_path, ncols=3)
            buf = io.StringIO()
            buf.write(f"\nLinear CKA — Adjacent Pairs ({gname})\n")
            for et, M in cka_mats.items():
                buf.write(f"  {EMB_LABELS[et]}: " +
                           "  ".join(f"{M[i,i+1]:.3f}" for i in range(n_lay - 1)) + "\n")
            text_cka += buf.getvalue()
        if svcca_mats:
            qc.plot_cross_layer_heatmaps(svcca_mats, EMB_LABELS, EMB_COLORS, LAYER_LBLS,
                                           f"FairChem UMA — SVCCA Between Layers ({gname})",
                                           "SVCCA", "magma", svcca_path, ncols=3)
            buf = io.StringIO()
            buf.write(f"\nSVCCA — Adjacent Pairs ({gname})\n")
            for et, M in svcca_mats.items():
                buf.write(f"  {EMB_LABELS[et]}: " +
                           "  ".join(f"{M[i,i+1]:.3f}" for i in range(n_lay - 1)) + "\n")
            text_svcca += buf.getvalue()
    if text_cka:
        text_cka = "=" * 60 + "\nLinear CKA Between Layers\n" + "=" * 60 + text_cka + "\n"
    if text_svcca:
        text_svcca = "=" * 60 + "\nSVCCA Between Layers\n" + "=" * 60 + text_svcca + "\n"
    return text_cka, text_svcca


def render_hdbscan_per_type(qc_dir: Path, plots_dir: Path, jsonl_path: Path) -> str:
    structs, _ = qc.load_manifest(jsonl_path)
    buf = io.StringIO()
    buf.write("=" * 60 + "\nHDBSCAN — All Layers Summary\n" + "=" * 60 + "\n")
    for et in ALL_EMB_KEYS + ALL_EDGE_KEYS:
        layers = _layers_for(et)
        lbls = _layer_lbls_for(et)
        out_scatter = qc_dir / f"hdbscan_{et}_all_layers.png"
        out_posthoc = qc_dir / f"hdbscan_{et}_all_layers_posthoc.txt"
        X_umaps: list[np.ndarray | None] = []
        labels_list: list[np.ndarray | None] = []
        mcs = ms = 0
        for layer in layers:
            p = qc_dir / "_cache" / f"{layer}_{et}_hdbscan.npz"
            if p.exists():
                d = np.load(p, allow_pickle=True)
                X_umaps.append(d["X_umap"]); labels_list.append(d["labels"])
                mcs = int(d["mcs"]); ms = int(d["ms"])
            else:
                X_umaps.append(None); labels_list.append(None)
        if not out_scatter.exists():
            qc.hdbscan_scatter_grid(lbls, X_umaps, labels_list,
                                       f"FairChem UMA — HDBSCAN ({EMB_LABELS[et]}) — All Layers",
                                       out_scatter, ncols=3, legend_top=True)
        if not out_posthoc.exists():
            tx = io.StringIO()
            tx.write(f"HDBSCAN post-hoc tables — {EMB_LABELS[et]}\n")
            tx.write(f"MCS={mcs}  MS={ms}\n")
            slice_p = qc_dir / "_cache" / f"{et}_hdbscan_idx.npy"
            slice_idx = np.load(slice_p)
            for layer, labels in zip(layers, labels_list):
                if labels is None:
                    continue
                cache = np.load(cache_path(plots_dir, layer, et), allow_pickle=True)
                mids = cache["mp_ids"][slice_idx]
                elements = cache["elements"][slice_idx] if "elements" in cache.files else None
                bond_types = cache["bond_types"][slice_idx] if "bond_types" in cache.files else None
                distances = cache["distances"][slice_idx] if "distances" in cache.files else None
                tx.write(f"\n--- Layer: {layer} ---\n")
                tx.write(qc.posthoc_table(labels, mids, structs, elements=elements,
                                          bond_types=bond_types, distances=distances))
            out_posthoc.write_text(tx.getvalue())
        row = f"  {EMB_LABELS[et]:<22} MCS={mcs:>4} MS={ms:>4}  "
        for labels in labels_list:
            if labels is None:
                row += f"{'N/A':>10}"; continue
            n_cl = int(labels.max()) + 1 if (labels >= 0).any() else 0
            nz = float((labels == -1).mean()) * 100
            row += f"k={n_cl:<2} nz={nz:>5.1f}%  "
        buf.write(row + "\n")
    return buf.getvalue() + "\n"


def render_ari_nmi(qc_dir: Path, plots_dir: Path, jsonl_path: Path) -> str:
    structs, _ = qc.load_manifest(jsonl_path)
    caveat = ("Caveat: per-atom ARI/NMI vs material properties broadcast each material's "
              "value to all its atoms. The metric is partly confounded with material identity. "
              "element_identity is a pure per-atom signal and not affected.")
    # (group name, emb keys, layers, layer labels, extra categorical, extra continuous)
    GROUPS = [
        ("atom",        PER_ATOM_KEYS,    ALL_LAYERS,  LAYER_LBLS,      "element_identity", None),
        ("pooled",      POOLED_KEYS,      ALL_LAYERS,  LAYER_LBLS,      None,               None),
        ("edge",        EDGE_ATOM_KEYS,   EDGE_LAYERS, EDGE_LAYER_LBLS, "bond_type",        "bond_distance"),
        ("edge_pooled", EDGE_POOLED_KEYS, EDGE_LAYERS, EDGE_LAYER_LBLS, None,               None),
    ]
    for gname, ets, layers, lbls, extra_cat, extra_cont in GROUPS:
        out_path = qc_dir / f"hdbscan_ari_nmi_{gname}.png"
        if out_path.exists():
            continue
        prop_list = list(qc.CAT_PROPS) + ([extra_cat] if extra_cat else []) \
            + qc.CONT_PROPS + ([extra_cont] if extra_cont else [])
        n_cat_split = len(qc.CAT_PROPS) + (1 if extra_cat else 0)
        n_lay = len(layers)
        ari_d: dict[str, np.ndarray] = {et: np.full((len(prop_list), n_lay), np.nan) for et in ets}
        nmi_d: dict[str, np.ndarray] = {et: np.full((len(prop_list), n_lay), np.nan) for et in ets}
        for et in ets:
            slice_p = qc_dir / "_cache" / f"{et}_hdbscan_idx.npy"
            if not slice_p.exists():
                continue
            slice_idx = np.load(slice_p)
            for li, layer in enumerate(layers):
                h_p = qc_dir / "_cache" / f"{layer}_{et}_hdbscan.npz"
                if not h_p.exists():
                    continue
                labels = np.load(h_p, allow_pickle=True)["labels"]
                cache = np.load(cache_path(plots_dir, layer, et), allow_pickle=True)
                mids = cache["mp_ids"][slice_idx]
                for pi, prop in enumerate(prop_list):
                    if prop in qc.CAT_PROPS:
                        vals = qc.cat_array(structs, mids, prop); is_cont = False
                    elif prop == "element_identity":
                        vals = cache["elements"][slice_idx] if "elements" in cache.files else None
                        is_cont = False
                    elif prop == "bond_type":
                        vals = cache["bond_types"][slice_idx] if "bond_types" in cache.files else None
                        is_cont = False
                    elif prop == "bond_distance":
                        vals = cache["distances"][slice_idx] if "distances" in cache.files else None
                        is_cont = True
                    else:
                        vals = qc.cont_array(structs, mids, prop); is_cont = True
                    if vals is None:
                        continue
                    a, n = qc.ari_nmi_for_row_property(labels, vals, is_cont)
                    ari_d[et][pi, li] = a
                    nmi_d[et][pi, li] = n
        qc.plot_ari_nmi_heatmap(ari_d, nmi_d, prop_list, n_cat_split,
                                  ets, EMB_LABELS, lbls,
                                  f"FairChem UMA — HDBSCAN ARI/NMI ({gname})",
                                  out_path, caveat=caveat)
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", default=str(P.RESULTS / "fairchem"))
    parser.add_argument("--plots-dir",      default=str(P.RESULTS / "fairchem/plots"))
    parser.add_argument("--structures-dir", default=str(P.STRUCTURES))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hdbscan-sample",   type=int, default=100_000)
    parser.add_argument("--knn-query-sample", type=int, default=50_000)
    parser.add_argument("--cka-sample",       type=int, default=10_000)
    parser.add_argument("--svcca-sample",     type=int, default=5_000)
    parser.add_argument("--svcca-var",        type=float, default=0.99)
    parser.add_argument("--geom-sample",      type=int, default=2_000)
    parser.add_argument("--jaccard-k",        type=int, default=15)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--sections", default="all")
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)
    plots_dir = Path(args.plots_dir)
    qc_dir = plots_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    (qc_dir / "_cache").mkdir(exist_ok=True)
    (qc_dir / "sections").mkdir(exist_ok=True)
    jsonl_path = Path(args.structures_dir) / "index.jsonl"

    sections = (set("dataset pca ksweep geometry cka svcca hdbscan".split())
                if args.sections == "all" else set(args.sections.split(",")))

    structs, valid = qc.load_manifest(jsonl_path)
    print(f"--- FairChem QC --- manifest: {len(structs):,} materials ---", flush=True)

    if "dataset" in sections:
        counts = collect_dataset_counts(emb_dir, valid, qc_dir, structs=structs,
                                         plots_dir=plots_dir)
        text = section_dataset_stats(counts, structs)
        qc.write_section(qc_dir, 1, "dataset_structure", text)
        print(text, flush=True)

    if "pca" in sections:
        text = render_pca_dim(qc_dir, plots_dir, qc_dir / "pca_dimensionality.png")
        if text:
            qc.write_section(qc_dir, 2, "pca_dimensionality", text)
            print(text, flush=True)

    if any(s in sections for s in ("ksweep", "geometry", "hdbscan")):
        print("\nSpawning workers (node + edge embedding types) ...", flush=True)
        worker_targets = list(ALL_EMB_KEYS) + list(ALL_EDGE_KEYS)
        if args.num_workers == 1:
            import queue as _q
            q: _q.SimpleQueue = _q.SimpleQueue()
            import threading
            def _drain():
                done = 0
                while done < len(worker_targets):
                    msg = q.get()
                    if isinstance(msg, str) and msg.startswith("__DONE__"):
                        done += 1
                        print(f"  Worker {msg[8:]} finished  ({done}/{len(worker_targets)})", flush=True)
                    else:
                        print(msg, flush=True)
            t = threading.Thread(target=_drain, daemon=True); t.start()
            for et in worker_targets:
                _worker_emb_type(et, emb_dir, plots_dir, qc_dir, jsonl_path,
                                  args.hdbscan_sample, args.knn_query_sample,
                                  args.jaccard_k, args.geom_sample, args.seed, q)
            t.join()
        else:
            ctx = mp.get_context("spawn")
            mpq: mp.Queue = ctx.Queue()
            running: list = []
            pending = list(worker_targets)
            results_done = 0
            print(f"  workers={args.num_workers}, types={len(pending)}", flush=True)
            # Roll our own pool over a fixed worker count
            while pending or running:
                while pending and len(running) < args.num_workers:
                    et = pending.pop(0)
                    p = ctx.Process(target=_worker_emb_type,
                                    args=(et, emb_dir, plots_dir, qc_dir, jsonl_path,
                                          args.hdbscan_sample, args.knn_query_sample,
                                          args.jaccard_k, args.geom_sample, args.seed, mpq),
                                    daemon=True)
                    p.start()
                    running.append((et, p))
                # Drain messages briefly, then check for finished processes
                drained = 0
                while drained < 50:
                    try:
                        msg = mpq.get(timeout=1.0)
                    except Exception:
                        break
                    drained += 1
                    if isinstance(msg, str) and msg.startswith("__DONE__"):
                        results_done += 1
                        print(f"  Worker {msg[8:]} finished  ({results_done}/{len(worker_targets)})", flush=True)
                    else:
                        print(msg, flush=True)
                # Reap finished
                still = []
                for (et, p) in running:
                    if p.is_alive():
                        still.append((et, p))
                    else:
                        p.join()
                running = still

    print("\n--- Rendering figures ---", flush=True)

    if "ksweep" in sections:
        for et in ALL_EMB_KEYS + ALL_EDGE_KEYS:
            k_ref, txt = render_sweep_figure(qc_dir, et)
            qc.write_section(qc_dir, 3, f"ksweep_{et}", txt)
            print(f"  knn_stability_{et}.png done (k_ref={k_ref})", flush=True)

    if "geometry" in sections:
        render_cosine(qc_dir, qc_dir / "pairwise_cosine_similarity.png")
        render_spectrum(qc_dir, qc_dir / "spectrum_concentration.png")
        render_jaccard(qc_dir, qc_dir / "jaccard_knn_overlap.png", args.jaccard_k)

    if "cka" in sections or "svcca" in sections:
        text_c, text_s = render_cka_svcca(plots_dir, qc_dir,
                                              qc_dir / "cka_atom_placeholder.png",
                                              qc_dir / "svcca_atom_placeholder.png",
                                              args.cka_sample, args.svcca_sample,
                                              args.svcca_var, args.seed)
        if text_c:
            qc.write_section(qc_dir, 8, "cka", text_c)
        if text_s:
            qc.write_section(qc_dir, 9, "svcca", text_s)

    if "hdbscan" in sections:
        txt = render_hdbscan_per_type(qc_dir, plots_dir, jsonl_path)
        qc.write_section(qc_dir, 10, "hdbscan", txt); print(txt[:2000], flush=True)
        render_ari_nmi(qc_dir, plots_dir, jsonl_path)

    summary = qc.assemble_summary(qc_dir, header=(
        "FairChem UMA-S — Embedding QC Summary\n" + "=" * 60))
    print(f"\nSaved summary -> {summary}", flush=True)


if __name__ == "__main__":
    main()
