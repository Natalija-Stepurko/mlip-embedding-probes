"""
stages/geometry/health_qc_orb.py — Quality-control analyses for the ORB v3 embedding pipeline

Production-scale port of orb_embedding_qc.ipynb. Operates on cached UMAP outputs
written by scripts/stages/geometry/organisation_orb.py:
  $MLIP_PROBES_ROOT/results/orb/plots/nodes/{layer}_nodes[_pooled]_umap_cache.npz
  $MLIP_PROBES_ROOT/results/orb/plots/edges/{layer}_edges[_pooled]_umap_cache.npz

Analyses (faithful to notebook):
  1. Dataset structure (atoms/mat, bonds/mat, bond families) — single .pt pass
  2. PCA dimensionality across layers × 4 emb types
  3. K-stability sweeps (purity + LVR), per emb type
  4. Bond-family K sweeps (oxygen / carbon / homoatomic)
  5. Q1 collapse — mean pairwise cosine similarity
  6. Q2 anisotropy — spectrum concentration + condition number
  7. Q3a stability — kNN Jaccard overlap between adjacent layers
  7b. Q3b — Linear CKA between all layer pairs (heatmap + line)
  7c. Q3c — SVCCA between all layer pairs (heatmap + line)
  8. HDBSCAN — adaptive calibration, per-type all-layer UMAP grids,
              post-hoc property tables, ARI/NMI heatmap (atom / pooled / edge)

Outputs land in `$MLIP_PROBES_ROOT/results/orb/plots/qc/` (+ qc/_cache/, qc/sections/).

Usage:
    uv run python stages/geometry/health_qc_orb.py
    uv run python stages/geometry/health_qc_orb.py --num-workers 1     # sequential
    uv run python stages/geometry/health_qc_orb.py --hdbscan-sample 0  # full data
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
import os
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["legend.frameon"] = False  # no box border around figure legends
import matplotlib.pyplot as plt
import numpy as np

# Allow running as a script from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlipprobes import common as qc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The ORB "backbone" layer is essentially identical to gnn_stacks_4 and is
# excluded from QC. gnn_stacks_4 is the final/reference layer everywhere that
# used to key off "backbone" (row-alignment reference cache, dataset census).
ALL_LAYERS = ["encoder", "gnn_stacks_0", "gnn_stacks_1", "gnn_stacks_2",
              "gnn_stacks_3", "gnn_stacks_4"]
LAYER_LBLS = ["Enc", "GNN-0", "GNN-1", "GNN-2", "GNN-3", "GNN-4"]
FINAL_LAYER = ALL_LAYERS[-1]  # "gnn_stacks_4"

ET_COLORS = {
    "nodes":         "#e07b39",
    "nodes_pooled":  "#5b8dd9",
    "edges":         "#59a14f",
    "edges_pooled":  "#9467bd",
    "concat_pooled": "#d4a017",  # mustard — kept distinct from the green edges colour
}
ET_TITLES = {
    "nodes":         "Node (atom-level)",
    "nodes_pooled":  "Nodes Pooled (material)",
    "edges":         "Edge (bond-level)",
    "edges_pooled":  "Edges Pooled (material)",
    "concat_pooled": "Concat Pooled (material)",
}

PUR_PROPS_NODE  = ["crystal_system", "metal_insulator", "magnetic_label", "element_identity"]
PUR_PROPS_POOL  = ["crystal_system", "metal_insulator", "magnetic_label"]
PUR_PROPS_EDGE  = ["crystal_system", "metal_insulator", "magnetic_label", "bond_type"]
LVR_PROPS_BASE  = list(qc.CONT_PROPS)  # 9 continuous
LVR_PROPS_EDGE  = LVR_PROPS_BASE + ["bond_distance"]


def cache_path_nodes(emb_dir: Path, plots_dir: Path, layer: str, et: str) -> Path:
    return plots_dir / "nodes" / f"{layer}_{et}_umap_cache.npz"


def cache_path_edges(emb_dir: Path, plots_dir: Path, layer: str, et: str) -> Path:
    return plots_dir / "edges" / f"{layer}_{et}_umap_cache.npz"


# ---------------------------------------------------------------------------
# Scalar-cache helpers — persist Q3 (Jaccard / CKA / SVCCA) and ARI/NMI values
# as machine-readable JSON alongside the figures, so the numbers are directly
# accessible without re-parsing the text report or re-reading the plots.
# ---------------------------------------------------------------------------

def _jsonable(o):
    """Recursively convert numpy / NaN into JSON-native types (NaN -> null)."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, float):
        return None if np.isnan(o) else o
    if isinstance(o, np.integer):
        return int(o)
    return o


def dump_scalars(qc_dir: Path, name: str, obj) -> Path:
    """Write `obj` to qc/_cache/scalars/{name}.json (strict, NaN->null JSON)."""
    sc_dir = qc_dir / "_cache" / "scalars"
    sc_dir.mkdir(parents=True, exist_ok=True)
    p = sc_dir / f"{name}.json"
    p.write_text(json.dumps(_jsonable(obj), indent=2, allow_nan=False))
    return p


def scalar_path(qc_dir: Path, name: str) -> Path:
    return qc_dir / "_cache" / "scalars" / f"{name}.json"


# ---------------------------------------------------------------------------
# Dataset counts pass — manifest atoms + edge census (cached to json)
# ---------------------------------------------------------------------------

def collect_dataset_counts(emb_dir: Path, valid_ids: set, qc_dir: Path,
                              plots_dir: Path | None = None,
                              structs: dict | None = None) -> dict:
    """Dataset stats from the manifest (atoms/mat) and the edge census (bonds).

    Avoids iterating 150k .pt files. Prefers the full-graph edge census; the
    fallback final-layer (gnn_4) edge cache holds the sqrt-weighted subsample
    with per-edge mp_ids and bond_types, giving bond_family proxies.
    """
    out = qc_dir / "dataset_counts.json"
    if out.exists():
        return json.loads(out.read_text())
    if plots_dir is None or structs is None:
        raise ValueError("plots_dir and structs are required when cache absent")
    # Atoms: from manifest nsites
    atoms_by_mat = {m: s.get("nsites") for m, s in structs.items() if s.get("nsites") is not None}
    bonds_by_mat: dict[str, int] = {}
    fam_by_mat: dict[str, dict[str, int]] = {}
    # Bonds: prefer the TRUE full-graph census written by the organisation script; only fall
    # back to the sqrt-weighted UMAP subsample (a misleading ~2 bonds/mat proxy)
    # if the census is absent.
    census = plots_dir / "edges" / "edge_census.npz"
    bonds_is_proxy = True
    if census.exists():
        cen = np.load(census, allow_pickle=True)
        cm = cen["mp_ids"].astype(str)
        cb = cen["n_bonds"]; co = cen["n_oxy"]; cc = cen["n_carbon"]; ch = cen["n_homo"]
        for i, m in enumerate(cm):
            bonds_by_mat[m] = int(cb[i])
            fam_by_mat[m] = {"oxygen": int(co[i]), "carbon": int(cc[i]),
                             "homoatomic": int(ch[i])}
        bonds_is_proxy = False
    else:
        edge_bb = plots_dir / "edges" / f"{FINAL_LAYER}_edges_umap_cache.npz"
        if edge_bb.exists():
            ec = np.load(edge_bb, allow_pickle=True)
            bt = ec["bond_types"].astype(str)
            mids = ec["mp_ids"].astype(str)
            # bonds per material (subsample)
            bond_counter = Counter(mids.tolist())
            for m, c in bond_counter.items():
                bonds_by_mat[m] = c
            # family per material (subsample)
            for i, m in enumerate(mids):
                b = bt[i]
                a, b2 = b.split("-") if "-" in b else (b, b)
                d = fam_by_mat.setdefault(m, {"oxygen": 0, "carbon": 0, "homoatomic": 0})
                if "O" in (a, b2):
                    d["oxygen"] += 1
                if "C" in (a, b2):
                    d["carbon"] += 1
                if a == b2:
                    d["homoatomic"] += 1
    counts: dict[str, dict] = {}
    bond_families = {"oxygen": [], "carbon": [], "homoatomic": []}
    for m in valid_ids:
        if m not in structs:
            continue
        n_at = atoms_by_mat.get(m, 0)
        n_b = bonds_by_mat.get(m, 0)
        fam = fam_by_mat.get(m, {"oxygen": 0, "carbon": 0, "homoatomic": 0})
        counts[m] = {"n_atoms": int(n_at), "n_bonds": int(n_b),
                     "n_oxy": fam["oxygen"], "n_c": fam["carbon"], "n_homo": fam["homoatomic"]}
        bond_families["oxygen"].append(fam["oxygen"])
        bond_families["carbon"].append(fam["carbon"])
        bond_families["homoatomic"].append(fam["homoatomic"])
    payload = {"per_material": counts, "bond_families": bond_families,
               "bonds_is_proxy": bonds_is_proxy,
               "_note": "atoms/mat from manifest nsites; bonds/mat and families derived "
                        + ("from the gnn_4 edges sqrt-weighted subsample (PROXY)"
                           if bonds_is_proxy else
                           "from the full-graph edge census (organisation script)")}
    out.write_text(json.dumps(payload))
    print(f"    Saved {out}", flush=True)
    return payload


def section_property_coverage(counts: dict, structs: dict) -> str:
    """% of datapoints (materials / nodes=atoms / edges=bonds) carrying each property.

    Nodes and edges inherit their material's property, but the coverage differs
    because materials are weighted by atom-count (nodes) vs bond-count (edges).
    """
    per = counts["per_material"]
    mids = list(per.keys())
    n_at = np.array([per[m]["n_atoms"] for m in mids], dtype=float)
    n_bo = np.array([per[m]["n_bonds"] for m in mids], dtype=float)
    tot_mat = float(len(mids)); tot_at = float(n_at.sum()); tot_bo = float(n_bo.sum())

    getters = [
        ("formation_energy/atom", lambda s: s.get("formation_energy_per_atom")),
        ("energy_above_hull",     lambda s: s.get("energy_above_hull")),
        ("band_gap",              lambda s: s.get("band_gap")),
        ("density",               lambda s: s.get("density")),
        ("bulk_modulus",          lambda s: s.get("bulk_modulus")),
        ("crystal_system",        lambda s: s.get("crystal_system")),
        ("metal_insulator",       lambda s: s.get("band_gap")),  # derived from band_gap
        ("magnetic_label",        lambda s: s.get("total_magnetization")),
    ]
    buf = io.StringIO()
    buf.write("\n  Property coverage  (% of datapoints whose material has the property):\n")
    buf.write(f"  {'property':<24}{'%_materials':>12}{'%_nodes':>10}{'%_edges':>10}\n")
    for label, getter in getters:
        have = np.array([1.0 if (m in structs and getter(structs[m]) is not None) else 0.0
                         for m in mids])
        mat_c  = 100 * have.sum() / tot_mat if tot_mat else 0.0
        node_c = 100 * (have * n_at).sum() / tot_at if tot_at else 0.0
        edge_c = 100 * (have * n_bo).sum() / tot_bo if tot_bo else 0.0
        buf.write(f"  {label:<24}{mat_c:>11.1f}%{node_c:>9.1f}%{edge_c:>9.1f}%\n")
    buf.write("\n  %_nodes weights materials by atom count; %_edges by bond count.\n")
    return buf.getvalue()


def section_dataset_stats(counts: dict, edges_bb_cache: Path, structs: dict) -> str:
    per = counts["per_material"]
    bfam = counts["bond_families"]
    atoms = np.array([v["n_atoms"] for v in per.values()], dtype=float)
    bonds = np.array([v["n_bonds"] for v in per.values()], dtype=float)
    is_proxy = counts.get("bonds_is_proxy", True)
    bonds_src = ("[proxy from gnn_4-edge sqrt-weighted subsample]" if is_proxy
                 else "[full-graph edge census]")
    fam_scope = "subsample proxy" if is_proxy else "full dataset"
    buf = io.StringIO()
    buf.write("=" * 60 + "\n")
    buf.write("Dataset Structure  (gnn_4 .pt files)\n")
    buf.write("=" * 60 + "\n")
    buf.write(f"  Materials: {len(atoms):,}\n\n")
    buf.write(f"  atoms/mat   mean={atoms.mean():.1f}  med={np.median(atoms):.0f}  "
              f"min={atoms.min():.0f}  max={atoms.max():.0f}  "
              f"p25={np.percentile(atoms,25):.0f}  p75={np.percentile(atoms,75):.0f}\n")
    buf.write(f"  bonds/mat   mean={bonds.mean():.1f}  med={np.median(bonds):.0f}  "
              f"min={bonds.min():.0f}  max={bonds.max():.0f}  "
              f"p25={np.percentile(bonds,25):.0f}  p75={np.percentile(bonds,75):.0f}"
              f"   {bonds_src}\n")
    if edges_bb_cache.exists():
        ec = np.load(edges_bb_cache, allow_pickle=True)
        bt_sub = ec["bond_types"]
        n_sub = len(bt_sub)
        buf.write(f"\n  gnn_4 edge subsample (for embedding analysis): {n_sub:,} bonds "
                  f"(sqrt-weighted by bond_type)\n")
    buf.write(f"\n  Bond families  (per-material totals, {fam_scope}):\n")
    buf.write(f"  {'family':<14}{'sum_bonds':>14}{'%_total':>10}"
              f"{'p25/mat':>10}{'med/mat':>10}{'p75/mat':>10}\n")
    total = float(np.sum(bonds))
    for fam, arr_l in bfam.items():
        arr = np.array(arr_l, dtype=float)
        tot = arr.sum()
        pct = 100 * tot / total if total > 0 else 0
        if len(arr) > 0:
            buf.write(f"  {fam:<14}{tot:>14,.0f}{pct:>10.2f}"
                      f"{np.percentile(arr,25):>10.0f}"
                      f"{np.percentile(arr,50):>10.0f}"
                      f"{np.percentile(arr,75):>10.0f}\n")
    buf.write(section_property_coverage(counts, structs))
    buf.write("\n")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Worker — one per embedding type. Phase 1: per-layer per-type intermediates.
# ---------------------------------------------------------------------------

def _path_for(emb_dir: Path, plots_dir: Path, layer: str, et: str) -> Path:
    # concat_pooled caches are written into the nodes/ dir by script 03.
    if et in ("nodes", "nodes_pooled", "concat_pooled"):
        return cache_path_nodes(emb_dir, plots_dir, layer, et)
    return cache_path_edges(emb_dir, plots_dir, layer, et)


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(nrm == 0, 1.0, nrm)


def _worker_emb_type(et: str, emb_dir: Path, plots_dir: Path, qc_dir: Path,
                       jsonl_path: Path,
                       hdbscan_sample: int, knn_query_sample: int,
                       jaccard_k: int, geom_sample: int, seed: int,
                       progress_q) -> None:
    matplotlib.use("Agg")

    def log(msg: str) -> None:
        progress_q.put(f"[{et:<14}] {msg}")

    is_atom_lvl = et == "nodes"
    is_pooled = et in ("nodes_pooled", "edges_pooled", "concat_pooled")
    is_edge_lvl = et == "edges"
    l2n = et in ("edges", "edges_pooled")

    structs, valid = qc.load_manifest(jsonl_path)
    cache_dir = qc_dir / "_cache"

    # Determine query/HDBSCAN slices once from the final-layer (gnn_4) cache (rows aligned across layers)
    bb_path = _path_for(emb_dir, plots_dir, FINAL_LAYER, et)
    if not bb_path.exists():
        log(f"reference ({FINAL_LAYER}) cache missing → {bb_path}")
        progress_q.put(f"__DONE__{et}")
        return
    bb = np.load(bb_path, allow_pickle=True)
    n_total = int(bb["X_raw"].shape[0])
    log(f"reference ({FINAL_LAYER}) cache: {n_total:,} rows × {bb['X_raw'].shape[1]}-d")

    # query rows for kNN sweeps/Jaccard
    q_path = cache_dir / f"{et}_query_idx.npy"
    if q_path.exists():
        query_idx = np.load(q_path)
    else:
        if is_pooled:
            # pooled: keep all rows by default (cheap)
            q_n = knn_query_sample if (knn_query_sample and knn_query_sample < n_total) else n_total
        else:
            q_n = knn_query_sample if knn_query_sample else n_total
        query_idx = qc.uniform_idx(n_total, q_n, seed=seed)
        np.save(q_path, query_idx)
    log(f"query rows: {len(query_idx):,}")

    # HDBSCAN slice (sqrt-weighted)
    s_path = cache_dir / f"{et}_hdbscan_idx.npy"
    if s_path.exists():
        hdb_idx = np.load(s_path)
    else:
        if is_pooled:
            hdb_idx = np.arange(n_total)  # always full for pooled
        else:
            cls_arr = bb["elements"] if "elements" in bb.files else bb["bond_types"]
            n_target = hdbscan_sample if hdbscan_sample else n_total
            hdb_idx = qc.sqrt_weighted_idx(cls_arr, n_target, seed=seed)
    np.save(s_path, hdb_idx)
    log(f"hdbscan slice: {len(hdb_idx):,}")

    # Adaptive K — pooled uses sqrt(n_mats); atoms/edges use p75(rows/mat)+1
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

    # HDBSCAN params from slice
    hdb_mids = bb["mp_ids"][hdb_idx]
    mcs, ms = qc.adaptive_hdbscan_params(hdb_mids, n_points=len(hdb_idx))
    log(f"HDBSCAN params (on slice): MCS={mcs}, MS={ms}")

    # Property dicts / row arrays needed for sweeps
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

    # ── Phase 1: per-layer ──────────────────────────────────────────────────
    for layer in ALL_LAYERS:
        t0 = time.time()
        log(f"--- {layer} ---")
        path = _path_for(emb_dir, plots_dir, layer, et)
        if not path.exists():
            log(f"  cache missing, skipping")
            continue

        # Files produced this layer
        sweep_p = cache_dir / f"{layer}_{et}_sweep.json"
        geom_p  = cache_dir / f"{layer}_{et}_geom.npz"
        knn_p   = cache_dir / f"{layer}_{et}_knn_idx.npz"
        hdb_p   = cache_dir / f"{layer}_{et}_hdbscan.npz"

        need_sweep = not sweep_p.exists()
        need_geom  = not geom_p.exists()
        need_knn   = not knn_p.exists()
        need_hdb   = not hdb_p.exists()

        if not any([need_sweep, need_geom, need_knn, need_hdb]):
            log(f"  all cached, skipping ({time.time()-t0:.0f}s)")
            continue

        cache = np.load(path, allow_pickle=True)
        X_raw = cache["X_raw"]
        if l2n:
            X_raw = _l2_normalize_rows(X_raw)
        X_pca = cache["X_pca"]
        mp_ids = cache["mp_ids"]
        elements = cache["elements"] if "elements" in cache.files else None
        labels_arr = cache["labels"] if "labels" in cache.files else None  # crystal_system per row
        bond_types = cache["bond_types"] if "bond_types" in cache.files else None
        distances = cache["distances"] if "distances" in cache.files else None

        # Geometry (spectrum + cosine)
        if need_geom:
            lam = qc.spectrum_eigvals(X_raw)
            cos_m, cos_s = qc.mean_pairwise_cosine(X_raw, max_n=geom_sample, seed=seed)
            np.savez(geom_p, lam=lam, cos_mean=cos_m, cos_std=cos_s)
            log(f"  geometry: spectrum({len(lam)}), cos={cos_m:.4f}±{cos_s:.4f}")

        # kNN indices (for sweeps + Jaccard)
        if need_knn:
            log(f"  building kNN: index={len(X_raw):,} rows × {X_raw.shape[1]}d, queries={len(query_idx):,}, K_max={K_MAX}")
            t_knn = time.time()
            nn_idx = qc.build_knn_indices(X_raw, query_idx, K_MAX, l2_normalize=False)
            np.savez(knn_p, nn_idx=nn_idx, query_idx=query_idx)
            log(f"  kNN built ({time.time()-t_knn:.0f}s, shape={nn_idx.shape})")
        else:
            nn_idx = np.load(knn_p)["nn_idx"]

        # K-sweep purity + LVR (nn_idx points into the full cache, so pass full + query arrays)
        if need_sweep:
            t_sw = time.time()
            mp_q = mp_ids[query_idx]
            mp_full = mp_ids
            cs_full = labels_arr if labels_arr is not None else qc.cat_array(structs, mp_full, "crystal_system")
            mi_full = qc.cat_array(structs, mp_full, "metal_insulator")
            mg_full = qc.cat_array(structs, mp_full, "magnetic_label")
            cs_q = cs_full[query_idx]; mi_q = mi_full[query_idx]; mg_q = mg_full[query_idx]

            pur: dict[str, dict[int, float]] = {}
            pur["crystal_system"]  = qc.sweep_purity(cs_q, cs_full, nn_idx, Ks)
            pur["metal_insulator"] = qc.sweep_purity(mi_q, mi_full, nn_idx, Ks)
            pur["magnetic_label"]  = qc.sweep_purity(mg_q, mg_full, nn_idx, Ks)
            if et == "nodes" and elements is not None:
                pur["element_identity"] = qc.sweep_purity(elements[query_idx], elements, nn_idx, Ks)
            if et == "edges" and bond_types is not None:
                pur["bond_type"] = qc.sweep_purity(bond_types[query_idx], bond_types, nn_idx, Ks)

            lvr: dict[str, dict[int, float | None]] = {}
            for prop_name, vd in cont_dicts.items():
                lvr[prop_name] = qc.sweep_lvr_dict(vd, mp_q, mp_full, nn_idx, Ks)
            if et == "edges" and distances is not None:
                lvr["bond_distance"] = qc.sweep_lvr_arr(distances[query_idx], distances, nn_idx, Ks)

            sweep_payload = {
                "Ks": Ks, "k_ref": int(k_ref),
                "pur": {k: {str(kk): v for kk, v in m.items()} for k, m in pur.items()},
                "lvr": {k: {str(kk): v for kk, v in m.items()} for k, m in lvr.items()},
            }
            sweep_p.write_text(json.dumps(sweep_payload))
            log(f"  sweep done ({time.time()-t_sw:.0f}s)")

        # HDBSCAN on slice
        if need_hdb:
            t_h = time.time()
            X_slice = X_pca[hdb_idx]
            labels = qc.run_hdbscan(X_slice, mcs, ms)
            valid_mask = labels >= 0
            n_clust = int(labels.max()) + 1 if valid_mask.any() else 0
            noise = float((~valid_mask).mean())
            X_umap_slice = cache["X_umap"][hdb_idx]
            np.savez(hdb_p, labels=labels, slice_idx=hdb_idx,
                      X_umap=X_umap_slice, mcs=mcs, ms=ms,
                      n_clusters=n_clust, noise_frac=noise)
            log(f"  HDBSCAN: clusters={n_clust}, noise={noise:.1%} ({time.time()-t_h:.0f}s)")

        del cache, X_raw, X_pca
        gc.collect()
        log(f"  [{layer}] done ({time.time()-t0:.0f}s)")

    progress_q.put(f"__DONE__{et}")


# ---------------------------------------------------------------------------
# Phase 2 renderers
# ---------------------------------------------------------------------------

def load_sweep(qc_dir: Path, et: str, layer: str) -> dict | None:
    p = qc_dir / "_cache" / f"{layer}_{et}_sweep.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    d["pur"] = {k: {int(kk): v for kk, v in m.items()} for k, m in d["pur"].items()}
    d["lvr"] = {k: {int(kk): v for kk, v in m.items()} for k, m in d["lvr"].items()}
    return d


def render_sweep_figure(qc_dir: Path, et: str, layer_lbls: list[str],
                         layer_colors: list, pur_props: list[str],
                         lvr_props: list[str], title: str,
                         out_path: Path, xscale: str = "linear") -> tuple[int, str]:
    if out_path.exists():
        # Re-derive k_ref + table text from any layer's sweep (they share k_ref)
        for layer in ALL_LAYERS:
            d = load_sweep(qc_dir, et, layer)
            if d:
                k_ref = d["k_ref"]
                break
        else:
            return 15, ""
        # rebuild table only
        pur_all, lvr_all = {p: {} for p in pur_props}, {p: {} for p in lvr_props}
        for layer, lbl in zip(ALL_LAYERS, layer_lbls):
            d = load_sweep(qc_dir, et, layer)
            if not d:
                continue
            for p in pur_props:
                pur_all[p][lbl] = d["pur"].get(p, {})
            for p in lvr_props:
                lvr_all[p][lbl] = d["lvr"].get(p, {})
        return k_ref, qc.ksweep_table_text(pur_all, lvr_all, pur_props, lvr_props,
                                             k_ref, layer_lbls, f"K-sweep: {et}")
    pur_all: dict = {p: {} for p in pur_props}
    lvr_all: dict = {p: {} for p in lvr_props}
    k_ref = 15
    for layer, lbl in zip(ALL_LAYERS, layer_lbls):
        d = load_sweep(qc_dir, et, layer)
        if not d:
            continue
        k_ref = d["k_ref"]
        for p in pur_props:
            pur_all[p][lbl] = d["pur"].get(p, {})
        for p in lvr_props:
            lvr_all[p][lbl] = d["lvr"].get(p, {})
    qc.plot_k_sweep(pur_all, lvr_all, pur_props, lvr_props, k_ref, title,
                     out_path, layer_lbls, layer_colors, xscale=xscale)
    return k_ref, qc.ksweep_table_text(pur_all, lvr_all, pur_props, lvr_props,
                                         k_ref, layer_lbls, f"K-sweep: {et}")


def render_pca_dim(qc_dir: Path, plots_dir: Path, out_path: Path) -> str:
    if out_path.exists():
        return ""  # text section regenerated only when missing
    fig, ax = plt.subplots(figsize=(10, 5))
    markers = ["o", "s", "^", "D", "P"]
    table = io.StringIO()
    table.write("=" * 60 + "\nPCA Dimensionality (90% variance) — All Layers\n" + "=" * 60 + "\n")
    table.write(f"  {'layer':<10}" + "".join(f"{et:>16}" for et in ET_TITLES) + "\n")
    rows: dict[str, list] = {et: [] for et in ET_TITLES}
    for layer in ALL_LAYERS:
        for et in ET_TITLES:
            path = _path_for(Path("/dummy"), plots_dir, layer, et)
            if path.exists():
                evr = np.load(path, allow_pickle=True)["evr"]
                rows[et].append(qc.pca_n_for_variance(evr))
            else:
                rows[et].append(np.nan)
    for j, lbl in enumerate(LAYER_LBLS):
        line = f"  {lbl:<10}"
        for et in ET_TITLES:
            v = rows[et][j]
            line += f"{v:>16.0f}" if not np.isnan(v) else f"{'N/A':>16}"
        table.write(line + "\n")
    table.write("\n")

    x = np.arange(len(ALL_LAYERS))
    for i, et in enumerate(ET_TITLES):
        ax.plot(x, rows[et], marker=markers[i], color=ET_COLORS[et],
                 lw=2, ms=7, label=ET_TITLES[et])
    ax.set_xticks(x); ax.set_xticklabels(LAYER_LBLS)
    ax.set_xlabel("Layer"); ax.set_ylabel("# principal components (90% variance)")
    ax.set_title("ORB v3 — PCA Dimensionality Across Layers")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return table.getvalue()


def render_cosine(qc_dir: Path, out_path: Path) -> str:
    if out_path.exists():
        return ""
    means: dict[str, list] = {et: [] for et in ET_TITLES}
    stds:  dict[str, list] = {et: [] for et in ET_TITLES}
    for et in ET_TITLES:
        for layer in ALL_LAYERS:
            p = qc_dir / "_cache" / f"{layer}_{et}_geom.npz"
            if p.exists():
                d = np.load(p)
                means[et].append(float(d["cos_mean"]))
                stds[et].append(float(d["cos_std"]))
            else:
                means[et].append(np.nan); stds[et].append(np.nan)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ALL_LAYERS))
    markers = ["o", "s", "^", "D", "P"]
    for i, et in enumerate(ET_TITLES):
        m = np.array(means[et]); s = np.array(stds[et])
        ax.plot(x, m, marker=markers[i], color=ET_COLORS[et], lw=2, ms=7, label=ET_TITLES[et])
        ax.fill_between(x, m - s, m + s, color=ET_COLORS[et], alpha=0.12)
    ax.set_xticks(x); ax.set_xticklabels(LAYER_LBLS)
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean pairwise cosine similarity")
    ax.set_title("ORB v3 — Q1 Collapse: Mean Pairwise Cosine ±std (X_raw)")
    ax.axhline(0, color="grey", ls=":", lw=0.8)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf = io.StringIO()
    buf.write("=" * 60 + "\nMean Pairwise Cosine Similarity (Q1 collapse)\n" + "=" * 60 + "\n")
    buf.write(f"  {'layer':<10}" + "".join(f"{et:>18}" for et in ET_TITLES) + "\n")
    for j, lbl in enumerate(LAYER_LBLS):
        row = f"  {lbl:<10}"
        for et in ET_TITLES:
            m, s = means[et][j], stds[et][j]
            row += f"{m:>10.3f}±{s:<6.3f}" if not np.isnan(m) else f"{'N/A':>18}"
        buf.write(row + "\n")
    buf.write("\n")
    return buf.getvalue()


def render_spectrum(qc_dir: Path, out_path: Path) -> str:
    if out_path.exists():
        return ""
    top1: dict[str, list] = {et: [] for et in ET_TITLES}
    top10: dict[str, list] = {et: [] for et in ET_TITLES}
    kappa: dict[str, list] = {et: [] for et in ET_TITLES}
    for et in ET_TITLES:
        for layer in ALL_LAYERS:
            p = qc_dir / "_cache" / f"{layer}_{et}_geom.npz"
            if p.exists():
                lam = np.load(p)["lam"]
                a, b = qc.concentration(lam)
                k = qc.condition_number(lam)
                top1[et].append(a); top10[et].append(b); kappa[et].append(k)
            else:
                top1[et].append(np.nan); top10[et].append(np.nan); kappa[et].append(np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(ALL_LAYERS))
    markers = ["o", "s", "^", "D", "P"]
    for i, et in enumerate(ET_TITLES):
        axes[0].plot(x, top1[et], marker=markers[i], color=ET_COLORS[et], lw=2, ms=6,
                      label=f"{ET_TITLES[et]} (λ₁/Σλ)")
        axes[0].plot(x, top10[et], marker=markers[i], color=ET_COLORS[et], lw=1.5,
                      ls="--", ms=5, alpha=0.7)
    axes[0].set_xticks(x); axes[0].set_xticklabels(LAYER_LBLS)
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Variance share")
    axes[0].set_title("Spectrum Concentration\n(solid = top-1 λ₁/Σλ, dashed = top-10)")
    axes[0].set_ylim(0, 1.02); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=7)
    for i, et in enumerate(ET_TITLES):
        axes[1].plot(x, kappa[et], marker=markers[i], color=ET_COLORS[et], lw=2, ms=6,
                      label=ET_TITLES[et])
    axes[1].set_xticks(x); axes[1].set_xticklabels(LAYER_LBLS)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel(f"Condition number κ = λ₁/λ_r  (tol={qc.COND_TOL:g})")
    axes[1].set_yscale("log"); axes[1].set_title("Covariance Condition Number")
    axes[1].grid(alpha=0.3, which="both"); axes[1].legend(fontsize=7)
    fig.suptitle("ORB v3 — Q2 Anisotropy: Spectrum Concentration & Condition Number")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf = io.StringIO()
    buf.write("=" * 60 + "\nSpectrum Concentration & Condition Number (Q2)\n" + "=" * 60 + "\n")
    buf.write(f"  {'layer':<10}")
    for et in ET_TITLES:
        buf.write(f"{et + ' λ1/Σλ':>16}{'top10':>10}{'κ':>10}")
    buf.write("\n")
    for j, lbl in enumerate(LAYER_LBLS):
        row = f"  {lbl:<10}"
        for et in ET_TITLES:
            a, b, k = top1[et][j], top10[et][j], kappa[et][j]
            row += (f"{a:>16.3f}{b:>10.3f}{k:>10.1f}" if not np.isnan(a) else
                    f"{'N/A':>16}{'N/A':>10}{'N/A':>10}")
        buf.write(row + "\n")
    buf.write("\n")
    return buf.getvalue()


def render_jaccard(qc_dir: Path, out_path: Path, jaccard_k: int) -> str:
    sc_p = scalar_path(qc_dir, "jaccard")
    if out_path.exists() and sc_p.exists():
        return ""
    mu_d: dict[str, list] = {et: [] for et in ET_TITLES}
    sd_d: dict[str, list] = {et: [] for et in ET_TITLES}
    pair_lbls = [f"{LAYER_LBLS[i]}→{LAYER_LBLS[i+1]}" for i in range(len(ALL_LAYERS) - 1)]
    for et in ET_TITLES:
        # Need same query rows across pairs — they're cached, so they match.
        for i in range(len(ALL_LAYERS) - 1):
            la, lb = ALL_LAYERS[i], ALL_LAYERS[i + 1]
            pa = qc_dir / "_cache" / f"{la}_{et}_knn_idx.npz"
            pb = qc_dir / "_cache" / f"{lb}_{et}_knn_idx.npz"
            if not (pa.exists() and pb.exists()):
                mu_d[et].append(np.nan); sd_d[et].append(np.nan); continue
            ia = np.load(pa)["nn_idx"][:, :jaccard_k]
            ib = np.load(pb)["nn_idx"][:, :jaccard_k]
            mu, sd = qc.jaccard_rows(ia, ib)
            mu_d[et].append(mu); sd_d[et].append(sd)
    # Persist scalars (always refreshed, independent of the figure guard).
    dump_scalars(qc_dir, "jaccard", {
        "metric": "knn_jaccard_overlap_adjacent_layers",
        "jaccard_k": jaccard_k,
        "pairs": pair_lbls,
        "data": {et: {pair_lbls[j]: {"mu": mu_d[et][j], "sd": sd_d[et][j]}
                       for j in range(len(pair_lbls))}
                 for et in ET_TITLES},
    })
    if out_path.exists():
        return ""  # scalars refreshed; figure + report text already exist
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pair_lbls))
    markers = ["o", "s", "^", "D", "P"]
    for i, et in enumerate(ET_TITLES):
        m = np.array(mu_d[et]); s = np.array(sd_d[et])
        ax.plot(x, m, marker=markers[i], color=ET_COLORS[et], lw=2, ms=7, label=ET_TITLES[et])
        ax.fill_between(x, m - s, m + s, color=ET_COLORS[et], alpha=0.12)
    ax.set_xticks(x); ax.set_xticklabels(pair_lbls, rotation=20, ha="right")
    ax.set_xlabel("Adjacent layer pair")
    ax.set_ylabel(f"Mean kNN Jaccard overlap (K={jaccard_k})")
    ax.set_ylim(0, 1.02)
    ax.set_title("ORB v3 — Q3 Stability: kNN Neighbourhood Jaccard Overlap")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf = io.StringIO()
    buf.write("=" * 60 + f"\nkNN Jaccard Overlap (K={jaccard_k}) Between Adjacent Layers\n" + "=" * 60 + "\n")
    buf.write(f"  {'pair':<14}" + "".join(f"{et:>20}" for et in ET_TITLES) + "\n")
    for j, lbl in enumerate(pair_lbls):
        row = f"  {lbl:<14}"
        for et in ET_TITLES:
            m, s = mu_d[et][j], sd_d[et][j]
            row += f"{m:>10.3f}±{s:<8.3f}" if not np.isnan(m) else f"{'N/A':>20}"
        buf.write(row + "\n")
    buf.write("\n")
    return buf.getvalue()


def render_cka_svcca(emb_dir: Path, plots_dir: Path, qc_dir: Path,
                      out_cka: Path, out_svcca: Path,
                      cka_sample: int, svcca_sample: int, svcca_var: float,
                      seed: int) -> tuple[str, str]:
    text_cka = text_svcca = ""
    n_lay = len(ALL_LAYERS)
    cka_mats: dict[str, np.ndarray] = {}
    svcca_mats: dict[str, np.ndarray] = {}
    # Compute if either the figure OR the scalar cache is missing.
    need_cka = (not out_cka.exists()) or (not scalar_path(qc_dir, "cka").exists())
    need_svcca = (not out_svcca.exists()) or (not scalar_path(qc_dir, "svcca").exists())
    for et in ET_TITLES:
        is_pooled = et in ("nodes_pooled", "edges_pooled", "concat_pooled")
        # Build column-centered, row-aligned X_raw per layer
        def resolver(layer: str, _et=et) -> Path:
            return _path_for(emb_dir, plots_dir, layer, _et)
        et_emb = qc.EmbType(key=et, label=ET_TITLES[et], color=ET_COLORS[et],
                              kind="pooled" if is_pooled else ("edge" if et == "edges" else "atom"),
                              l2_normalize=(et in ("edges", "edges_pooled")))
        # CKA: subsample (full for pooled)
        if need_cka:
            cka_n = 0 if is_pooled else cka_sample
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
        if need_svcca:
            mats, _ = qc.load_aligned_raw_layers(et_emb, ALL_LAYERS, resolver,
                                                   row_subsample=svcca_sample, seed=seed)
            M = np.full((n_lay, n_lay), np.nan)
            for a in range(n_lay):
                for b in range(a, n_lay):
                    if mats[a] is None or mats[b] is None:
                        continue
                    v = qc.svcca(mats[a], mats[b], var=svcca_var,
                                  max_rows=0, seed=seed)  # already subsampled
                    M[a, b] = M[b, a] = v
            svcca_mats[et] = M
            del mats; gc.collect()
    # Persist scalars (full L×L matrices + adjacent-pair convenience view).
    def _adjacent(M: np.ndarray) -> dict:
        return {f"{LAYER_LBLS[i]}→{LAYER_LBLS[i+1]}": M[i, i + 1] for i in range(n_lay - 1)}
    if need_cka and cka_mats:
        dump_scalars(qc_dir, "cka", {
            "metric": "linear_cka_between_layers",
            "row_subsample": cka_sample,
            "layers": LAYER_LBLS,
            "matrices": {et: cka_mats[et] for et in cka_mats},
            "adjacent": {et: _adjacent(cka_mats[et]) for et in cka_mats},
        })
    if need_svcca and svcca_mats:
        dump_scalars(qc_dir, "svcca", {
            "metric": "svcca_between_layers",
            "row_subsample": svcca_sample,
            "svcca_var": svcca_var,
            "layers": LAYER_LBLS,
            "matrices": {et: svcca_mats[et] for et in svcca_mats},
            "adjacent": {et: _adjacent(svcca_mats[et]) for et in svcca_mats},
        })
    if not out_cka.exists() and cka_mats:
        qc.plot_cross_layer_heatmaps(cka_mats, ET_TITLES, ET_COLORS, LAYER_LBLS,
                                       "ORB v3 — Linear CKA Between Layers (X_raw, aligned rows)",
                                       "Linear CKA", "viridis", out_cka, ncols=2)
        buf = io.StringIO()
        buf.write("=" * 60 + "\nLinear CKA Between Layers — Adjacent Pairs\n" + "=" * 60 + "\n")
        for et, M in cka_mats.items():
            buf.write(f"\n  {ET_TITLES[et]}\n")
            buf.write("  " + "  ".join(f"{LAYER_LBLS[i]}→{LAYER_LBLS[i+1]}: {M[i,i+1]:.3f}"
                                          for i in range(n_lay-1)) + "\n")
        text_cka = buf.getvalue()
    if not out_svcca.exists() and svcca_mats:
        qc.plot_cross_layer_heatmaps(svcca_mats, ET_TITLES, ET_COLORS, LAYER_LBLS,
                                       "ORB v3 — SVCCA Between Layers (X_raw, aligned rows)",
                                       "SVCCA", "magma", out_svcca, ncols=2)
        buf = io.StringIO()
        buf.write("=" * 60 + "\nSVCCA Between Layers — Adjacent Pairs\n" + "=" * 60 + "\n")
        for et, M in svcca_mats.items():
            buf.write(f"\n  {ET_TITLES[et]}\n")
            buf.write("  " + "  ".join(f"{LAYER_LBLS[i]}→{LAYER_LBLS[i+1]}: {M[i,i+1]:.3f}"
                                          for i in range(n_lay-1)) + "\n")
        text_svcca = buf.getvalue()
    return text_cka, text_svcca


def render_hdbscan_per_type(qc_dir: Path, emb_dir: Path, plots_dir: Path,
                              jsonl_path: Path) -> str:
    structs, _ = qc.load_manifest(jsonl_path)
    buf = io.StringIO()
    buf.write("=" * 60 + "\nHDBSCAN — All Layers Summary\n" + "=" * 60 + "\n")
    buf.write(f"  {'space':<14}{'MCS':>6}{'MS':>5}" +
              "".join(f"{LAYER_LBLS[i] + ' k':>10}{LAYER_LBLS[i] + ' nz':>10}" for i in range(len(ALL_LAYERS))) + "\n")
    for et in ET_TITLES:
        out_scatter = qc_dir / f"hdbscan_{et}_all_layers.png"
        out_posthoc = qc_dir / f"hdbscan_{et}_all_layers_posthoc.txt"
        # Collect labels + UMAPs
        X_umaps: list[np.ndarray | None] = []
        labels_list: list[np.ndarray | None] = []
        mcs = ms = 0
        for layer in ALL_LAYERS:
            p = qc_dir / "_cache" / f"{layer}_{et}_hdbscan.npz"
            if p.exists():
                d = np.load(p, allow_pickle=True)
                X_umaps.append(d["X_umap"])
                labels_list.append(d["labels"])
                mcs = int(d["mcs"]); ms = int(d["ms"])
            else:
                X_umaps.append(None); labels_list.append(None)
        # Scatter grid
        if not out_scatter.exists():
            qc.hdbscan_scatter_grid(LAYER_LBLS, X_umaps, labels_list,
                                      f"ORB v3 — HDBSCAN ({ET_TITLES[et]}) — All Layers",
                                      out_scatter, ncols=4)
        # Per-cluster posthoc tables (one per layer for per-layer detail)
        if not out_posthoc.exists():
            tx = io.StringIO()
            tx.write(f"HDBSCAN post-hoc tables — {ET_TITLES[et]}\n")
            tx.write(f"MCS={mcs}  MS={ms}\n")
            for layer, labels in zip(ALL_LAYERS, labels_list):
                if labels is None:
                    continue
                slice_p = qc_dir / "_cache" / f"{et}_hdbscan_idx.npy"
                slice_idx = np.load(slice_p)
                cache = np.load(_path_for(emb_dir, plots_dir, layer, et), allow_pickle=True)
                mids = cache["mp_ids"][slice_idx]
                elements = cache["elements"][slice_idx] if "elements" in cache.files else None
                bond_types = cache["bond_types"][slice_idx] if "bond_types" in cache.files else None
                distances = cache["distances"][slice_idx] if "distances" in cache.files else None
                tx.write(f"\n--- Layer: {layer} ---\n")
                tx.write(qc.posthoc_table(labels, mids, structs,
                                           elements=elements,
                                           bond_types=bond_types,
                                           distances=distances))
            out_posthoc.write_text(tx.getvalue())
        # Summary row
        row = f"  {et:<14}{mcs:>6}{ms:>5}"
        for labels in labels_list:
            if labels is None:
                row += f"{'N/A':>10}{'N/A':>10}"; continue
            n_cl = int(labels.max()) + 1 if (labels >= 0).any() else 0
            nz = float((labels == -1).mean()) * 100
            row += f"{n_cl:>10}{nz:>9.1f}%"
        buf.write(row + "\n")
    buf.write("\n")
    return buf.getvalue()


def render_ari_nmi(qc_dir: Path, emb_dir: Path, plots_dir: Path,
                     jsonl_path: Path) -> str:
    structs, _ = qc.load_manifest(jsonl_path)
    buf = io.StringIO()
    buf.write("=" * 60 + "\nHDBSCAN ARI / NMI vs Material Properties\n" + "=" * 60 + "\n")
    caveat = ("Caveat: per-atom and per-edge ARI/NMI vs material properties broadcast each "
              "material's value to all its rows. The metric is then partly confounded with "
              "material identity. Pure per-row signals (element_identity, bond_type) are not.")
    KINDS = {
        "individual": (["nodes", "edges"], None),
        "pooled":     (["nodes_pooled", "edges_pooled", "concat_pooled"], PUR_PROPS_POOL),
    }
    n_lay = len(ALL_LAYERS)

    for kind, (ets, _) in KINDS.items():
        out_path = qc_dir / f"hdbscan_ari_nmi_{kind}.png"
        sc_p = scalar_path(qc_dir, f"ari_nmi_{kind}")
        if out_path.exists() and sc_p.exists():
            continue

        # Build property row list (categorical first). The per-row "individual"
        # figure unions node- and edge-specific properties onto one set of rows;
        # each property is only computed where it applies (the other column shows
        # grey/NaN), so nodes and edges share a single figure like pooled does.
        if kind == "individual":
            prop_list = (list(qc.CAT_PROPS) + ["element_identity", "bond_type"]
                         + qc.CONT_PROPS + ["bond_distance"])
            n_cat_split = len(qc.CAT_PROPS) + 2
        else:
            prop_list = list(qc.CAT_PROPS) + qc.CONT_PROPS
            n_cat_split = len(qc.CAT_PROPS)

        ari_d: dict[str, np.ndarray] = {et: np.full((len(prop_list), n_lay), np.nan) for et in ets}
        nmi_d: dict[str, np.ndarray] = {et: np.full((len(prop_list), n_lay), np.nan) for et in ets}

        for et in ets:
            slice_p = qc_dir / "_cache" / f"{et}_hdbscan_idx.npy"
            slice_idx = np.load(slice_p)
            for li, layer in enumerate(ALL_LAYERS):
                h_p = qc_dir / "_cache" / f"{layer}_{et}_hdbscan.npz"
                if not h_p.exists():
                    continue
                labels = np.load(h_p, allow_pickle=True)["labels"]
                cache = np.load(_path_for(emb_dir, plots_dir, layer, et), allow_pickle=True)
                mids = cache["mp_ids"][slice_idx]
                for pi, prop in enumerate(prop_list):
                    if prop in qc.CAT_PROPS:
                        vals = qc.cat_array(structs, mids, prop)
                        is_cont = False
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
                        vals = qc.cont_array(structs, mids, prop)
                        is_cont = True
                    if vals is None:
                        continue
                    a, n = qc.ari_nmi_for_row_property(labels, vals, is_cont)
                    ari_d[et][pi, li] = a
                    nmi_d[et][pi, li] = n
        # Persist scalars: {ari|nmi}[emb_type][property][layer_label] = value
        dump_scalars(qc_dir, f"ari_nmi_{kind}", {
            "metric": "hdbscan_cluster_vs_property_agreement",
            "kind": kind,
            "emb_types": list(ets),
            "layers": LAYER_LBLS,
            "properties": list(prop_list),
            "n_cat_split": n_cat_split,
            "ari": {et: {prop_list[pi]: {LAYER_LBLS[li]: ari_d[et][pi, li]
                                          for li in range(n_lay)}
                          for pi in range(len(prop_list))} for et in ets},
            "nmi": {et: {prop_list[pi]: {LAYER_LBLS[li]: nmi_d[et][pi, li]
                                          for li in range(n_lay)}
                          for pi in range(len(prop_list))} for et in ets},
        })
        if not out_path.exists():
            qc.plot_ari_nmi_heatmap(ari_d, nmi_d, prop_list, n_cat_split,
                                     ets, ET_TITLES, LAYER_LBLS,
                                     f"ORB v3 — HDBSCAN ARI/NMI ({kind})",
                                     out_path,
                                     caveat=(caveat if kind == "individual" else None))
        # Append to summary
        buf.write(f"\n  {kind.upper()}  emb_types={ets}\n")
        for et in ets:
            buf.write(f"\n    {ET_TITLES[et]}\n")
            buf.write(f"      {'property':<22}")
            for lbl in LAYER_LBLS:
                buf.write(f"{lbl + ' ARI':>10}{lbl + ' NMI':>10}")
            buf.write("\n")
            for pi, prop in enumerate(prop_list):
                row = f"      {prop:<22}"
                for li in range(n_lay):
                    a = ari_d[et][pi, li]
                    n = nmi_d[et][pi, li]
                    row += (f"{a:>10.3f}{n:>10.3f}" if np.isfinite(a) and np.isfinite(n)
                            else f"{'N/A':>10}{'N/A':>10}")
                buf.write(row + "\n")
    return buf.getvalue() if buf.tell() > len(caveat) + 100 else (caveat + "\n")


# ---------------------------------------------------------------------------
# Family sweep (ORB-specific)
# ---------------------------------------------------------------------------

def render_family_sweep(qc_dir: Path, plots_dir: Path, out_path: Path) -> str:
    """Per-bond-family K stability sweep (oxygen / carbon / homoatomic).

    Computes inline on edge caches with family masks and per-family weighted purity
    (crystal_system) + LVR (bond_distance). Log-x K range 5..200.
    """
    if out_path.exists():
        return ""
    FAMILIES = ["oxygen", "carbon", "homoatomic"]
    Ks = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200]

    def family_masks(bt: np.ndarray) -> dict[str, np.ndarray]:
        masks = {}
        masks["oxygen"]     = np.array(["O" in (b.split("-") if "-" in b else [b, b]) for b in bt])
        masks["carbon"]     = np.array(["C" in (b.split("-") if "-" in b else [b, b]) for b in bt])
        masks["homoatomic"] = np.array([b.split("-")[0] == b.split("-")[-1] if "-" in b else True for b in bt])
        return masks

    structs, _ = qc.load_manifest((P.STRUCTURES / "index.jsonl"))

    # 3 rows (families) × 2 cols (purity, LVR)
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.suptitle("ORB v3 — Per-Bond-Family K-Stability Sweep", fontsize=12, y=0.995)
    layer_colors = [plt.get_cmap("plasma")(i / (len(ALL_LAYERS) - 1)) for i in range(len(ALL_LAYERS))]

    buf = io.StringIO()
    buf.write("=" * 60 + "\nBond-Family K Stability Sweep\n" + "=" * 60 + "\n")
    from sklearn.neighbors import NearestNeighbors
    for fi, fam in enumerate(FAMILIES):
        for li, layer in enumerate(ALL_LAYERS):
            cache_p = cache_path_edges(Path("/dummy"), plots_dir, layer, "edges")
            if not cache_p.exists():
                continue
            cache = np.load(cache_p, allow_pickle=True)
            X = _l2_normalize_rows(cache["X_raw"])
            bt = cache["bond_types"]
            mids = cache["mp_ids"]
            distances = cache["distances"]
            masks = family_masks(bt)
            mask = masks[fam]
            if mask.sum() < 30:
                continue
            Xf = X[mask]; bt_f = bt[mask]; mids_f = mids[mask]; d_f = distances[mask]
            cs_f = np.array([structs.get(m, {}).get("crystal_system") for m in mids_f], dtype=object)
            k_use = min(max(Ks) + 1, len(Xf))
            nn = NearestNeighbors(n_neighbors=k_use, algorithm="auto", n_jobs=-1).fit(Xf)
            _, idx = nn.kneighbors(Xf)
            idx = idx[:, 1:]
            # Family sweep queries == full family set, so query_labels == full_labels
            pur = qc.sweep_purity(cs_f, cs_f, idx, Ks)
            lvr = qc.sweep_lvr_arr(d_f, d_f, idx, Ks)
            ks_p = sorted(pur.keys())
            axes[fi, 0].plot(ks_p, [pur[k] for k in ks_p], "o-",
                              color=layer_colors[li], lw=1.6, label=LAYER_LBLS[li])
            ks_l = sorted(lvr.keys())
            axes[fi, 1].plot(ks_l, [lvr[k] for k in ks_l], "o-",
                              color=layer_colors[li], lw=1.6, label=LAYER_LBLS[li])
        axes[fi, 0].axvline(15, color="tomato", ls="--", lw=1.3)
        axes[fi, 1].axvline(15, color="tomato", ls="--", lw=1.3)
        axes[fi, 0].set_xscale("log"); axes[fi, 1].set_xscale("log")
        axes[fi, 0].set_title(f"{fam} — Purity (crystal_system)", fontsize=10)
        axes[fi, 1].set_title(f"{fam} — LVR (bond_distance)", fontsize=10)
        axes[fi, 0].set_xlabel("K"); axes[fi, 1].set_xlabel("K")
        axes[fi, 0].set_ylabel("Norm. purity"); axes[fi, 1].set_ylabel("LVR")
        axes[fi, 0].set_ylim(0, 1); axes[fi, 1].set_ylim(0, 1)
        axes[fi, 0].grid(alpha=0.3); axes[fi, 1].grid(alpha=0.3)
        if fi == 0:
            axes[fi, 0].legend(fontsize=8, ncol=4, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.write("  (See family_k_stability_sweep.png; per-K values omitted for brevity)\n\n")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", default=str(P.RESULTS / "orb"))
    parser.add_argument("--plots-dir",      default=str(P.RESULTS / "orb/plots"))
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
    parser.add_argument("--sections",
                        default="all",
                        help="comma list: dataset,pca,ksweep,family,geometry,cka,svcca,hdbscan or 'all'")
    args = parser.parse_args()

    emb_dir = Path(args.embeddings_dir)
    plots_dir = Path(args.plots_dir)
    qc_dir = plots_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    (qc_dir / "_cache").mkdir(exist_ok=True)
    (qc_dir / "sections").mkdir(exist_ok=True)
    jsonl_path = Path(args.structures_dir) / "index.jsonl"

    sections = (set("dataset pca ksweep family geometry cka svcca hdbscan".split())
                if args.sections == "all" else set(args.sections.split(",")))

    structs, valid = qc.load_manifest(jsonl_path)
    print(f"--- ORB QC --- manifest: {len(structs):,} materials ---", flush=True)

    # ── Section 1: Dataset counts ────────────────────────────────────────────
    if "dataset" in sections:
        counts = collect_dataset_counts(emb_dir, valid, qc_dir, plots_dir=plots_dir, structs=structs)
        text = section_dataset_stats(counts, cache_path_edges(emb_dir, plots_dir, FINAL_LAYER, "edges"), structs)
        qc.write_section(qc_dir, 1, "dataset_structure", text)
        print(text, flush=True)

    # ── Section 2: PCA dimensionality ────────────────────────────────────────
    if "pca" in sections:
        text = render_pca_dim(qc_dir, plots_dir, qc_dir / "pca_dimensionality.png")
        if text:
            qc.write_section(qc_dir, 2, "pca_dimensionality", text)
            print(text, flush=True)

    # ── Sections 3,5,6,7,8 are produced in workers (per-emb-type intermediates) ──
    if any(s in sections for s in ("ksweep", "geometry", "hdbscan")):
        print("\nSpawning workers (one per embedding type) ...", flush=True)
        worker_targets = []
        for et in ["nodes", "nodes_pooled", "edges", "edges_pooled", "concat_pooled"]:
            worker_targets.append(et)
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
            procs = []
            for et in worker_targets:
                p = ctx.Process(target=_worker_emb_type,
                                args=(et, emb_dir, plots_dir, qc_dir, jsonl_path,
                                      args.hdbscan_sample, args.knn_query_sample,
                                      args.jaccard_k, args.geom_sample, args.seed, mpq),
                                daemon=True)
                p.start(); procs.append(p)
            qc.listen(mpq, len(procs))
            for p in procs:
                p.join()

    # ── Phase 2: per-section rendering ──────────────────────────────────────
    print("\n--- Rendering figures ---", flush=True)

    if "ksweep" in sections:
        for et in ["nodes", "nodes_pooled", "edges", "edges_pooled", "concat_pooled"]:
            out_p = qc_dir / f"knn_stability_{et}.png"
            pur_props = (PUR_PROPS_NODE if et == "nodes" else
                          PUR_PROPS_EDGE if et == "edges" else
                          PUR_PROPS_POOL)
            lvr_props = LVR_PROPS_EDGE if et == "edges" else LVR_PROPS_BASE
            xscale = "log" if et == "edges" else "linear"
            layer_colors = [plt.get_cmap("plasma")(i / (len(ALL_LAYERS) - 1)) for i in range(len(ALL_LAYERS))]
            k_ref, txt = render_sweep_figure(qc_dir, et, LAYER_LBLS, layer_colors,
                                                pur_props, lvr_props,
                                                f"ORB v3 K-Stability — {ET_TITLES[et]}",
                                                out_p, xscale=xscale)
            qc.write_section(qc_dir, 3, f"ksweep_{et}", txt)
            print(f"  Saved {out_p} (k_ref={k_ref})", flush=True)

    if "family" in sections:
        out_p = qc_dir / "family_k_stability_sweep.png"
        txt = render_family_sweep(qc_dir, plots_dir, out_p)
        if txt:
            qc.write_section(qc_dir, 4, "family_sweep", txt)
            print(f"  Saved {out_p}", flush=True)

    if "geometry" in sections:
        txt1 = render_cosine(qc_dir, qc_dir / "pairwise_cosine_similarity.png")
        if txt1:
            qc.write_section(qc_dir, 5, "geometry_cosine", txt1); print(txt1, flush=True)
        txt2 = render_spectrum(qc_dir, qc_dir / "spectrum_concentration.png")
        if txt2:
            qc.write_section(qc_dir, 6, "geometry_spectrum", txt2); print(txt2, flush=True)
        txt3 = render_jaccard(qc_dir, qc_dir / "jaccard_knn_overlap.png", args.jaccard_k)
        if txt3:
            qc.write_section(qc_dir, 7, "geometry_jaccard", txt3); print(txt3, flush=True)

    if "cka" in sections or "svcca" in sections:
        out_cka = qc_dir / "cka_heatmaps.png"
        out_svcca = qc_dir / "svcca.png"
        txt_c, txt_s = render_cka_svcca(emb_dir, plots_dir, qc_dir, out_cka, out_svcca,
                                          args.cka_sample, args.svcca_sample,
                                          args.svcca_var, args.seed)
        if txt_c:
            qc.write_section(qc_dir, 8, "cka", txt_c); print(f"  Saved {out_cka}", flush=True)
        if txt_s:
            qc.write_section(qc_dir, 9, "svcca", txt_s); print(f"  Saved {out_svcca}", flush=True)

    if "hdbscan" in sections:
        txt = render_hdbscan_per_type(qc_dir, emb_dir, plots_dir, jsonl_path)
        qc.write_section(qc_dir, 10, "hdbscan", txt); print(txt[:2000], flush=True)
        txt2 = render_ari_nmi(qc_dir, emb_dir, plots_dir, jsonl_path)
        qc.write_section(qc_dir, 11, "hdbscan_ari_nmi", txt2)

    # ── Assemble summary ────────────────────────────────────────────────────
    summary = qc.assemble_summary(qc_dir, header=(
        "ORB v3 — Embedding QC Summary\n" + "=" * 60))
    print(f"\nSaved summary -> {summary}", flush=True)


if __name__ == "__main__":
    main()
