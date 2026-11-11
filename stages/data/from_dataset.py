"""
stages/data/from_dataset.py — route B: rebuild the stage inputs from the published datasets

Writes, under $MLIP_PROBES_ROOT, the files that the probe stages and everything downstream of them
read, so they run without fetching structures or extracting embeddings:

    structures/index.jsonl          one record per material: formula, volume, the MP properties
    structures/<mp_id>.cif          only with --cifs
    results/orb/property_prediction/features/{layer}_pooled.npz
                                    ORB pooled node and edge vectors, six layers
    results/fairchem/plots/{nodes,edges}/{layer}_{nodes|edges}_pooled_{family}_umap_cache.npz
                                    UMA pooled vectors (X_raw), every family and layer
    results/{orb,fairchem}/property_prediction/splits/<target>.npz
                                    the published seed-42 train/test split of each target
    results/exp_magpie/magpie_features.npz
                                    Magpie composition features, rebuilt from the formulas

The corpus is the released 154,875 materials, in the order the paper used. The paper's probes saw
four further materials that the release excludes (config/excluded_materials.csv), so a probe score
rerun from these inputs can differ from the published one in the fourth decimal place.

Each dataset is given as a local copy or as its Hub id, which is then downloaded (the UMA repository is
gated: accept its licence on the Hub and `hf auth login` first). Without --uma only the ORB side is written.

    uv run python stages/data/from_dataset.py --orb NatalijaStepurko/orb-mp-embeddings \\
        --uma NatalijaStepurko/uma-mp-embeddings-gated
    uv run python stages/data/from_dataset.py --orb /path/to/orb-mp-embeddings --uma /path/to/uma-mp-embeddings-gated

Then run route A from the probe stages onwards (stages/probes/probes_orb.py, probes_uma.py, significance.py,
...). Stages that need per-atom activations or re-embed structures still need route A: the geometry and
health stages, the rotation, random-weight and matched-model controls, and the external transfer.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from mlipprobes import common as qc

ORB_LAYERS = ["encoder", "gnn_stacks_0", "gnn_stacks_1", "gnn_stacks_2", "gnn_stacks_3", "gnn_stacks_4"]
UMA_NODE = (["full", "ang_norm", "l0", "l1_polar", "l2_polar", "geom_norm"],
            ["embed", "block_0", "block_1", "block_2", "block_3", "backbone"])
UMA_EDGE = (["flat", "ang_norm", "l0", "l1_norm", "l2_norm", "geom_norm"],
            ["embed", "block_0", "block_1", "block_2", "block_3"])
TARGETS = qc.CONT_PROPS + qc.CAT_PROPS
INDEX_FIELDS = ["formula", "nsites", "crystal_system", "volume", "energy_above_hull",
                "formation_energy_per_atom", "band_gap", "bulk_modulus", "total_magnetization", "density"]


def column(target):
    """materials.parquet column of a target (and of its split__ flag)."""
    return qc.CONT_KEY_MAP.get(target, target)


def safe(target):
    return target.replace("/", "_per_")      # the probe stages' file token


def locate(src, patterns):
    """A local dataset folder as given, or a Hub dataset id downloaded to the HF cache."""
    if Path(src).exists():
        return Path(src)
    from huggingface_hub import snapshot_download
    print(f"  downloading {src} ({', '.join(patterns)})", flush=True)
    return Path(snapshot_download(src, repo_type="dataset", allow_patterns=patterns))


def matrix(table):
    """The `emb` list column of a shard as a dense float32 matrix."""
    col = table.column("emb").combine_chunks()
    dim = len(col[0])
    flat = col.flatten().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    if flat.size != dim * len(col):
        raise SystemExit("ragged embedding column")
    return flat.reshape(len(col), dim)


def aligned(path, ids, pos):
    """Rows of one shard placed in the canonical order; NaN rows for materials the shard lacks."""
    t = pq.read_table(path, columns=["mp_id", "emb"])
    X = matrix(t)
    out = np.full((len(ids), X.shape[1]), np.nan, dtype=np.float32)
    rows = np.array([pos[m] for m in t.column("mp_id").to_pylist()])
    out[rows] = X
    return out


def write_index(mat, str_dir, cifs, orb_dir):
    str_dir.mkdir(parents=True, exist_ok=True)
    recs = mat[["mp_id", "space_group_symbol"] + INDEX_FIELDS].to_dict("records")
    with open(str_dir / "index.jsonl", "w") as f:
        for r in recs:
            rec = {"mp_id": r["mp_id"]}
            if cifs:
                rec["cif_file"] = str(str_dir / f"{r['mp_id']}.cif")
            for k in INDEX_FIELDS:
                v = r[k]
                rec[k] = None if v is None or (isinstance(v, float) and v != v) else (int(v) if k == "nsites" else v)
            rec["space_group"] = r["space_group_symbol"]
            f.write(json.dumps(rec) + "\n")
    (str_dir / "meta.json").write_text(json.dumps(
        {"source": "published dataset (stages/data/from_dataset.py)", "total_written": len(recs)}, indent=2) + "\n")
    print(f"  index: {len(recs):,} records -> {str_dir / 'index.jsonl'}", flush=True)
    if cifs:
        from pymatgen.core import Lattice, Structure
        s = pq.read_table(orb_dir / "data/structures.parquet").to_pydict()
        t0 = time.time()
        for i, m in enumerate(s["mp_id"]):
            st = Structure(Lattice(np.reshape(s["lattice"][i], (3, 3))), s["species"][i],
                           np.reshape(s["frac_coords"][i], (-1, 3)))
            st.to(filename=str(str_dir / f"{m}.cif"))
            if (i + 1) % 20000 == 0:
                print(f"    cif {i + 1:,}/{len(s['mp_id']):,} ({time.time() - t0:.0f}s)", flush=True)
        print(f"  cifs: {len(s['mp_id']):,} written", flush=True)


def check_labels(mat, structs, ids):
    """The labels the stages derive from the index must equal the published ones."""
    for t in TARGETS:
        pub = mat[column(t)].to_numpy()
        if t in qc.CAT_PROPS:
            mine = qc.cat_array(structs, ids, t)
            ok = all((a is None and (b is None or b != b)) or a == b for a, b in zip(mine, pub))
        else:
            mine = qc.cont_array(structs, ids, t)
            pub = pub.astype(float)
            ok = np.allclose(mine, pub, rtol=1e-6, atol=1e-9, equal_nan=True)
        if not ok:
            raise SystemExit(f"label mismatch for {t}: index-derived labels differ from materials.parquet")
    print(f"  labels: all {len(TARGETS)} targets derived from the index equal the published columns", flush=True)


def write_splits(mat, structs, ids, out_dirs):
    for t in TARGETS:
        flag = mat[f"split__{column(t)}"].to_numpy()
        y = qc.cat_array(structs, ids, t) if t in qc.CAT_PROPS else qc.cont_array(structs, ids, t)
        valid = np.array([v is not None for v in y]) if t in qc.CAT_PROPS else np.isfinite(y.astype(float))
        tr, te = np.where(flag == "train")[0], np.where(flag == "test")[0]
        if not np.array_equal(np.sort(np.concatenate([tr, te])), np.where(valid)[0]):
            raise SystemExit(f"split of {t} does not cover exactly the labelled materials")
        for d in out_dirs:
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"{safe(t)}.npz", train_idx=tr, test_idx=te,
                     train_mp_ids=ids[tr], test_mp_ids=ids[te])
    print(f"  splits: {len(TARGETS)} targets -> {', '.join(str(d) for d in out_dirs)}", flush=True)


def write_orb(orb_dir, ids, pos, feat_dir):
    feat_dir.mkdir(parents=True, exist_ok=True)
    for layer in ORB_LAYERS:
        node = aligned(orb_dir / f"embeddings/node/orb-node-{layer}.parquet", ids, pos)
        edge = aligned(orb_dir / f"embeddings/edge/orb-edge-{layer}.parquet", ids, pos)
        np.savez(feat_dir / f"{layer}_pooled.npz", mp_ids=ids, node=node, edge=edge)
        print(f"  orb {layer}: node {node.shape}, edge rows missing {int(np.isnan(edge).any(1).sum())}", flush=True)


def write_uma(uma_dir, ids, pos, plots_dir):
    n = 0
    for group, sub, (families, layers) in (("node", "nodes", UMA_NODE), ("edge", "edges", UMA_EDGE)):
        (plots_dir / sub).mkdir(parents=True, exist_ok=True)
        for fam in families:
            for layer in layers:
                X = aligned(uma_dir / f"embeddings/{group}/uma-{group}-{fam}-{layer}.parquet", ids, pos)
                keep = ~np.isnan(X).any(1)
                np.savez(plots_dir / sub / f"{layer}_{sub}_pooled_{fam}_umap_cache.npz",
                         X_raw=X[keep], mp_ids=ids[keep].astype(str))
                n += 1
    print(f"  uma: {n} pooled caches -> {plots_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orb", required=True, help="the open dataset: local folder or Hub id")
    ap.add_argument("--uma", help="the gated UMA dataset: local folder or Hub id")
    ap.add_argument("--cifs", action="store_true", help="also write one CIF per material (for stages that re-embed)")
    ap.add_argument("--no-magpie", action="store_true", help="skip the Magpie features (needs matminer, ~3 min)")
    a = ap.parse_args()
    t0 = time.time()

    pats = ["data/materials.parquet", "embeddings/*"] + (["data/structures.parquet"] if a.cifs else [])
    orb_dir = locate(a.orb, pats)
    import pandas as pd
    mat = pd.read_parquet(orb_dir / "data/materials.parquet")
    mat = mat.astype({c: object for c in mat.columns if mat[c].dtype == "string" or str(mat[c].dtype) == "str"})
    mat = mat.where(mat.notna(), None)
    ids = mat["mp_id"].astype(str).to_numpy(dtype=object)
    pos = {m: i for i, m in enumerate(ids)}
    print(f"{len(ids):,} materials from {orb_dir}", flush=True)

    write_index(mat, P.STRUCTURES, a.cifs, orb_dir)
    structs, _ = qc.load_manifest(P.STRUCTURES / "index.jsonl")
    check_labels(mat, structs, ids)
    write_splits(mat, structs, ids, [P.RESULTS / "orb/property_prediction/splits",
                                     P.RESULTS / "fairchem/property_prediction/splits"])
    write_orb(orb_dir, ids, pos, P.RESULTS / "orb/property_prediction/features")
    if a.uma:
        write_uma(locate(a.uma, ["embeddings/*"]), ids, pos, P.RESULTS / "fairchem/plots")
    if not a.no_magpie:
        from mlipprobes import significance_lib as L
        L._build_magpie_cache(list(ids), {m: s["formula"] for m, s in structs.items()},
                              cache=str(P.RESULTS / "exp_magpie/magpie_features.npz"))
    print(f"done in {(time.time() - t0) / 60:.1f} min; now run the probe stages", flush=True)


if __name__ == "__main__":
    main()
