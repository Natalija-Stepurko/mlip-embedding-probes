"""
stages/extract/extract_uma.py — extract FairChem UMA all-layer node embeddings

Reads CIFs from --structures-dir, writes .pt files to --results-dir.

Directory layout created under --results-dir:
    embed/nodes/
    block_0/nodes/ ... block_3/nodes/
    backbone/nodes/

6 layers are extracted in a single forward pass per material:
    embed      — initial node embedding (input to first block, after sphere/charge embeddings)
    block_0 … block_3 — output of each eSCNMD_Block (equivariant message-passing block)
    backbone   — final backbone output (post block_3 layer-norm), from output["node_embedding"]

Note: uma-s-1p2 uses eSCNMDMoeBackbone with 4 blocks (not 12 as in the base eSCNMDBackbone).

Each node .pt contains node embeddings for one material + 3 pre-computed pooled variants + metadata.
    Keys:  node [N,9,128] float16  (raw per-atom equivariant embedding)
           structure_full_mean [1152] float16  (mean over atoms, then flatten)
           structure_l0_mean   [128]  float16  (L=0 channel mean over atoms)
           structure_ang_norm  [384]  float16  (cat of L0_mean, L1_norm_mean, L2_norm_mean)
           model, mp_id, elements, frac_coords

EDGE embeddings are also captured in the same forward pass, at 5 layers (embed = edge-degree
embedding + block_0..3; no backbone edge layer). Each edge .pt under {layer}/edges/ contains:
    edge_msg [m,9,128] float16  (l-ordered per-edge message, sqrt-weighted subsample, <= MAX_EDGES_PER_MAT)
    edge_pooled [9,128] float16 (mean over ALL edges — faithful, pre-subsample)
    bond_type/element_i/element_j (len m), distance [m] float32, n_edges_total, model, mp_id
The per-edge tensor is captured (in the global l-ordering, same basis as nodes) by shadowing the
backend's edge_degree_scatter / permute_wigner_inv_edge_to_node just before the scatter-to-nodes.

Resume: all 6 node + 5 edge .pt files must exist for a material to be considered done.

Parallelism: --num-workers processes (default 2 — UMA weights ~1 GB each).
             --torch-threads controls torch.set_num_threads() per worker (default 4).

Usage:
    uv run python stages/extract/extract_uma.py

    uv run python stages/extract/extract_uma.py \\
        --structures-dir $MLIP_PROBES_ROOT/structures \\
        --results-dir $MLIP_PROBES_ROOT/results/fairchem

    # Local test
    uv run python stages/extract/extract_uma.py \\
        --structures-dir $MLIP_PROBES_ROOT/structures \\
        --results-dir $MLIP_PROBES_ROOT/results/fairchem \\
        --limit 5 --num-workers 1
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Layer configuration — 14 layers: initial embed + 12 blocks + backbone
# ---------------------------------------------------------------------------
LAYER_KEYS = [
    "embed",
    "block_0", "block_1", "block_2", "block_3",
    "backbone",
]

# Edge embeddings exist at the encoder (edge-degree embedding) and at each of the
# 4 message-passing blocks — NOT at `backbone` (post-aggregation node-only LN).
EDGE_LAYERS = [
    "embed",
    "block_0", "block_1", "block_2", "block_3",
]

# Per-material cap on stored raw edges (storage lever). Edges are ~20-50x more
# numerous than atoms; we keep the pooled mean over ALL edges (faithful) and a
# sqrt-weighted (by bond_type) subsample of raw edges to keep storage node-scale.
MAX_EDGES_PER_MAT = 48


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _make_dirs(results_dir: Path) -> None:
    for layer in LAYER_KEYS:
        (results_dir / layer / "nodes").mkdir(parents=True, exist_ok=True)
    for layer in EDGE_LAYERS:
        (results_dir / layer / "edges").mkdir(parents=True, exist_ok=True)
    print(f"  Output dirs ready under {results_dir}")


def _node_path(results_dir: Path, layer: str, mp_id: str) -> Path:
    return results_dir / layer / "nodes" / f"{mp_id}.pt"


def _edge_path(results_dir: Path, layer: str, mp_id: str) -> Path:
    return results_dir / layer / "edges" / f"{mp_id}.pt"


def _nodes_done(results_dir: Path, mp_id: str) -> bool:
    return all(_node_path(results_dir, k, mp_id).exists() for k in LAYER_KEYS)


def _edges_done(results_dir: Path, mp_id: str) -> bool:
    return all(_edge_path(results_dir, k, mp_id).exists() for k in EDGE_LAYERS)


def _all_done(results_dir: Path, mp_id: str) -> bool:
    return _nodes_done(results_dir, mp_id) and _edges_done(results_dir, mp_id)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _reset_all_params(module) -> tuple[int, int, list]:
    """Randomise every parameter of an nn.Module in place — same helper as
    stages/extract/extract_orb.py, duplicated rather than imported since these two
    scripts are otherwise independent extraction pipelines. See that file's copy
    for the full rationale; verified on UMA's calc.predictor.model (an
    AveragedModel wrapper — named_parameters() still recurses through it
    correctly): 46 submodules reset directly, 72 tensors via fallback, 151/151
    parameters confirmed changed, forward pass still runs cleanly afterward."""
    import torch
    n_ok, n_fallback, failures = 0, 0, []
    reset_via_module = set()
    for m in module.modules():
        if hasattr(m, "reset_parameters"):
            try:
                m.reset_parameters()
                n_ok += 1
                reset_via_module.update(id(p) for p in m.parameters(recurse=False))
            except Exception as ex:
                failures.append((type(m).__name__, repr(ex)))
    for _name, p in module.named_parameters():
        if id(p) in reset_via_module:
            continue
        with torch.no_grad():
            if p.dim() >= 2:
                torch.nn.init.kaiming_uniform_(p, a=5 ** 0.5)
            elif p.dim() == 1 and p.numel() > 1:
                torch.nn.init.normal_(p, std=0.02)
            else:
                torch.nn.init.uniform_(p, -0.02, 0.02)
        n_fallback += 1
    return n_ok, n_fallback, failures


def _load_fairchem(random_init: bool = False, random_init_seed: int = 1234):
    """random_init: pretraining-vs-architecture control, see
    stages/extract/extract_orb.py's _load_orb() docstring for the full rationale
    (same fixed-seed-per-worker requirement applies here)."""
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    import torch
    from fairchem.core import FAIRChemCalculator
    calc = FAIRChemCalculator.from_model_checkpoint("uma-s-1p2", task_name="omat")
    if random_init:
        torch.manual_seed(random_init_seed)
        n_ok, n_fallback, failures = _reset_all_params(calc.predictor.model)
        print(f"  [random-init] reset {n_ok} submodules directly, {n_fallback} tensors "
             f"via shape-based fallback ({len(failures)} module types needed the fallback)")
    return calc


# ---------------------------------------------------------------------------
# Pooled embedding helpers (numpy, no torch needed)
# ---------------------------------------------------------------------------

def _pool_full_mean(node: "np.ndarray") -> "np.ndarray":
    """[N, 9, 128] → [1152] float16 — mean over atoms then flatten."""
    import numpy as np
    return node.mean(axis=0).flatten().astype(np.float16)


def _pool_l0_mean(node: "np.ndarray") -> "np.ndarray":
    """[N, 9, 128] → [128] float16 — L=0 channel mean over atoms."""
    import numpy as np
    return node[:, 0, :].mean(axis=0).astype(np.float16)


def _pool_ang_norm(node: "np.ndarray") -> "np.ndarray":
    """[N, 9, 128] → [384] float16 — cat(L0_mean, L1_norm_mean, L2_norm_mean)."""
    import numpy as np
    L0 = node[:, 0, :]        # [N, 128]
    L1 = node[:, 1:4, :]      # [N, 3, 128]
    L2 = node[:, 4:9, :]      # [N, 5, 128]
    return np.concatenate([
        L0.mean(axis=0),
        np.linalg.norm(L1, axis=1).mean(axis=0),
        np.linalg.norm(L2, axis=1).mean(axis=0),
    ]).astype(np.float16)


# ---------------------------------------------------------------------------
# Edge helpers
# ---------------------------------------------------------------------------

def _bond_type(ei: str, ej: str) -> str:
    """Order-independent bond label, e.g. ('O','C') -> 'C-O'."""
    return "-".join(sorted([ei, ej]))


def _sqrt_subsample_idx(bond_types: list[str], cap: int, seed: int = 42) -> "np.ndarray":
    """Indices keeping <= cap edges, sqrt-weighted by bond_type (>=1 per type)."""
    import numpy as np
    from collections import Counter
    n = len(bond_types)
    if n <= cap:
        return np.arange(n)
    arr = np.asarray(bond_types)
    cnt = Counter(bond_types)
    sqrt_tot = sum(np.sqrt(c) for c in cnt.values())
    rng = np.random.default_rng(seed)
    idx: list[int] = []
    for bt, c in cnt.items():
        q = max(1, int(round(cap * np.sqrt(c) / sqrt_tot)))
        q = min(q, c)
        pool = np.where(arr == bt)[0]
        idx.extend(rng.choice(pool, size=q, replace=False).tolist())
    out = np.array(sorted(set(idx)))
    if len(out) > cap:
        out = np.sort(rng.choice(out, size=cap, replace=False))
    return out


# ---------------------------------------------------------------------------
# Tensor extraction helper (robust to tensor / dict / tuple / SO3_Embedding)
# ---------------------------------------------------------------------------

def _extract_3d(obj):
    """Return a 3-D cpu float32 tensor from a block output, or None."""
    import torch
    if isinstance(obj, torch.Tensor) and obj.dim() == 3:
        return obj.detach().cpu().float()
    if isinstance(obj, dict):
        for k in ("node_embedding", "x", "node_features"):
            if k in obj:
                r = _extract_3d(obj[k])
                if r is not None:
                    return r
    if isinstance(obj, (tuple, list)) and len(obj) > 0:
        return _extract_3d(obj[0])
    if hasattr(obj, "embedding"):
        return _extract_3d(obj.embedding)
    return None


# ---------------------------------------------------------------------------
# All-layer embedding extraction
# ---------------------------------------------------------------------------

def _extract(calc, atoms, elements: list[str],
             capture_edges: bool = True, edge_cap: int = MAX_EDGES_PER_MAT) -> tuple[dict, dict]:
    """
    Single forward pass over the UMA backbone capturing both node and edge
    embeddings. Returns (node_result, edge_result).

    node_result[layer] = {node [N,9,128] f16, structure_full_mean, structure_l0_mean,
                          structure_ang_norm}  (as before)
    edge_result[layer] = {edge_msg [m,9,128] f16 (subsampled), edge_pooled [9,128] f16
                          (mean over ALL edges), bond_type/element_i/element_j (len m),
                          distance [m] f32, n_edges_total int}

    Edges are captured as the l-ordered per-edge tensor BEFORE scatter-to-nodes, by
    shadowing the backend's `edge_degree_scatter` (encoder/`embed`) and
    `permute_wigner_inv_edge_to_node` (blocks). The backend is the single shared
    instance; methods are @staticmethod so instance attrs cleanly shadow them.
    """
    import torch
    import numpy as np

    layer_embs: dict[str, "torch.Tensor"] = {}
    handles = []

    backbone = calc.predictor.model.module.backbone
    backend = backbone.backend
    n_blocks = len(backbone.blocks)

    # ---- node hooks (unchanged) ----
    def embed_pre_hook(module, inp):
        x_list = inp if isinstance(inp, (tuple, list)) else (inp,)
        for x in x_list:
            t = _extract_3d(x)
            if t is not None:
                layer_embs["embed"] = t.half()
                return
    handles.append(backbone.blocks[0].register_forward_pre_hook(embed_pre_hook))

    for i, block in enumerate(backbone.blocks):
        def _make_hook(key):
            def hook(module, inp, out):
                t = _extract_3d(out)
                if t is not None:
                    layer_embs[key] = t.half()
            return hook
        handles.append(block.register_forward_hook(_make_hook(f"block_{i}")))

    def backbone_hook(module, inp, out):
        layer_embs["backbone"] = out["node_embedding"].detach().cpu().half()
    handles.append(backbone.register_forward_hook(backbone_hook))

    # ---- edge capture ----
    # The model runs the backbone forward more than once (energy + a second pass),
    # so we attribute each backend edge call to the module currently executing
    # (edge_degree_embedding -> "embed"; block[i].edge_wise -> "block_i") via
    # per-module pre-hooks, and keep only the FIRST occurrence of each layer.
    edge_caps: dict[str, tuple] = {}      # layer -> (msg [E,9,128] f16 cpu, edge_index [2,E] cpu)
    dist_holder: dict[str, "torch.Tensor"] = {}
    ctx: dict[str, "str | None"] = {"layer": None}
    orig_pwi = orig_eds = None
    if capture_edges:
        def dist_pre_hook(module, inp):
            if "dist" not in dist_holder and len(inp) > 0 and inp[0] is not None:
                dist_holder["dist"] = inp[0].detach().cpu().float()
        handles.append(backbone.distance_expansion.register_forward_pre_hook(dist_pre_hook))

        def _mk_ctx(layer):
            def pre(module, inp):
                ctx["layer"] = layer
            return pre
        handles.append(backbone.edge_degree_embedding.register_forward_pre_hook(_mk_ctx("embed")))
        for i, block in enumerate(backbone.blocks):
            handles.append(block.edge_wise.register_forward_pre_hook(_mk_ctx(f"block_{i}")))

        orig_pwi = backend.permute_wigner_inv_edge_to_node
        orig_eds = backend.edge_degree_scatter

        def wrap_eds(*args, **kwargs):
            # (x, radial_output, wigner_inv, edge_index, m_0_num_coefficients, sphere_channels, ...)
            L = ctx["layer"]
            if L is not None and L not in edge_caps:
                radial_output, wigner_inv, edge_index = args[1], args[2], args[3]
                m0, ch = args[4], args[5]
                # no_grad + detach so this side-capture never alters the autograd
                # graph (the edge_wise/edge_degree forward is activation-checkpointed;
                # adding graph tensors here breaks the recomputation invariant).
                with torch.no_grad():
                    radial = radial_output.detach().reshape(-1, m0, ch)
                    xee = torch.bmm(wigner_inv.detach()[:, :, :m0], radial)
                edge_caps[L] = (xee.cpu().half(), edge_index.detach().cpu())
            return orig_eds(*args, **kwargs)

        def wrap_pwi(*args, **kwargs):
            # (x_message, wigner_inv, edge_index, num_nodes, node_offset)
            L = ctx["layer"]
            if L is not None and L not in edge_caps:
                x_message, wigner_inv, edge_index = args[0], args[1], args[2]
                with torch.no_grad():
                    xr = torch.bmm(wigner_inv.detach(), x_message.detach())
                edge_caps[L] = (xr.cpu().half(), edge_index.detach().cpu())
            return orig_pwi(*args, **kwargs)

        backend.edge_degree_scatter = wrap_eds
        backend.permute_wigner_inv_edge_to_node = wrap_pwi

    try:
        atoms.calc = calc
        calc.reset()
        atoms.get_potential_energy()
    finally:
        for h in handles:
            h.remove()
        if capture_edges:
            for name, orig in (("edge_degree_scatter", orig_eds),
                               ("permute_wigner_inv_edge_to_node", orig_pwi)):
                try:
                    delattr(backend, name)
                except AttributeError:
                    setattr(backend, name, orig)

    # ---- assemble node result (unchanged) ----
    node_result: dict = {}
    for key in LAYER_KEYS:
        if key not in layer_embs:
            continue
        node_f32 = layer_embs[key].float().numpy()
        node_result[key] = {
            "node":                layer_embs[key],
            "structure_full_mean": _pool_full_mean(node_f32),
            "structure_l0_mean":   _pool_l0_mean(node_f32),
            "structure_ang_norm":  _pool_ang_norm(node_f32),
        }

    # ---- assemble edge result ----
    edge_result: dict = {}
    if capture_edges:
        missing = [l for l in EDGE_LAYERS if l not in edge_caps]
        if missing:
            raise RuntimeError(f"edge capture missing layers: {missing} (got {list(edge_caps)})")

        ei = edge_caps["embed"][1].numpy()
        E = ei.shape[1]
        src, tgt = ei[0].tolist(), ei[1].tolist()
        elem_i = [elements[s] for s in src]
        elem_j = [elements[t] for t in tgt]
        bond_types = [_bond_type(a, b) for a, b in zip(elem_i, elem_j)]
        dist = dist_holder.get("dist")
        dist = dist.numpy() if dist is not None else np.full(E, np.nan, dtype=np.float32)
        if dist.shape[0] != E:
            dist = np.full(E, np.nan, dtype=np.float32)

        sub = _sqrt_subsample_idx(bond_types, edge_cap)
        sub_list = sub.tolist()

        for key in EDGE_LAYERS:
            msg = edge_caps[key][0]                     # [E, 9, 128] f16
            if msg.shape[0] != E:
                raise RuntimeError(f"edge count mismatch at {key}: {msg.shape[0]} vs {E}")
            edge_result[key] = {
                "edge_msg":      msg[sub].contiguous(),                 # [m,9,128] f16
                "edge_pooled":   msg.float().mean(dim=0).half(),        # [9,128] f16 (all edges)
                "bond_type":     [bond_types[k] for k in sub_list],
                "element_i":     [elem_i[k] for k in sub_list],
                "element_j":     [elem_j[k] for k in sub_list],
                "distance":      torch.from_numpy(dist[sub].astype(np.float32)),
                "n_edges_total": int(E),
            }

    return node_result, edge_result


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(worker_id: int, mp_ids: list[str], structures_dir: Path,
            results_dir: Path, torch_threads: int, progress_q,
            random_init: bool = False, random_init_seed: int = 1234) -> None:
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    import torch
    import numpy as np
    torch.set_num_threads(torch_threads)

    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    adaptor = AseAtomsAdaptor()

    progress_q.put((worker_id, None, "loading_model"))
    calc = _load_fairchem(random_init=random_init, random_init_seed=random_init_seed)
    progress_q.put((worker_id, None, "model_ready"))

    errors = []
    for mp_id in mp_ids:
        if _all_done(results_dir, mp_id):
            progress_q.put((worker_id, mp_id, "skip"))
            continue

        cif_path = structures_dir / f"{mp_id}.cif"
        if not cif_path.exists():
            progress_q.put((worker_id, mp_id, "missing_cif"))
            errors.append((mp_id, "CIF not found"))
            continue

        try:
            pmg         = Structure.from_file(str(cif_path))
            elements    = [str(sp) for sp in pmg.species]
            frac_coords = pmg.frac_coords.tolist()
            atoms       = adaptor.get_atoms(pmg)

            node_embs, edge_embs = _extract(calc, atoms, elements)

            missing_layers = [k for k in LAYER_KEYS if k not in node_embs]
            if missing_layers:
                raise RuntimeError(f"Hooks did not capture node layers: {missing_layers}. "
                                   f"Check backbone architecture access path.")
            missing_edges = [k for k in EDGE_LAYERS if k not in edge_embs]
            if missing_edges:
                raise RuntimeError(f"Did not capture edge layers: {missing_edges}. "
                                   f"Check backend edge-method shadowing.")

            for k, embs in node_embs.items():
                node_p = _node_path(results_dir, k, mp_id)
                if not node_p.exists():
                    torch.save({
                        "node":                embs["node"],
                        "structure_full_mean": torch.from_numpy(embs["structure_full_mean"]),
                        "structure_l0_mean":   torch.from_numpy(embs["structure_l0_mean"]),
                        "structure_ang_norm":  torch.from_numpy(embs["structure_ang_norm"]),
                        "model":               "uma-s-1p2",
                        "mp_id":               mp_id,
                        "elements":            elements,
                        "frac_coords":         frac_coords,
                    }, node_p)

            for k, e in edge_embs.items():
                edge_p = _edge_path(results_dir, k, mp_id)
                if not edge_p.exists():
                    torch.save({
                        "edge_msg":      e["edge_msg"],
                        "edge_pooled":   e["edge_pooled"],
                        "bond_type":     e["bond_type"],
                        "element_i":     e["element_i"],
                        "element_j":     e["element_j"],
                        "distance":      e["distance"],
                        "n_edges_total": e["n_edges_total"],
                        "model":         "uma-s-1p2",
                        "mp_id":         mp_id,
                    }, edge_p)

            progress_q.put((worker_id, mp_id, "done"))

        except Exception as exc:
            progress_q.put((worker_id, mp_id, f"error: {exc}"))
            errors.append((mp_id, str(exc)))

    progress_q.put((worker_id, None, f"finished:{len(errors)}"))


# ---------------------------------------------------------------------------
# Progress listener
# ---------------------------------------------------------------------------

def _listen(progress_q, total: int, num_workers: int) -> dict:
    done = 0
    skipped = 0
    errors: list[tuple[str, str]] = []
    workers_finished = 0
    t0 = time.time()

    while workers_finished < num_workers:
        worker_id, mp_id, status = progress_q.get()

        if status == "loading_model":
            print(f"  worker {worker_id}: loading UMA model ...")
        elif status == "model_ready":
            print(f"  worker {worker_id}: model ready")
        elif status == "skip":
            skipped += 1
        elif status == "done":
            done += 1
            elapsed = time.time() - t0
            rate = (done + skipped) / elapsed if elapsed > 0 else 0
            remaining = total - done - skipped
            eta = remaining / rate if rate > 0 else float("inf")
            eta_str = f"{eta/3600:.1f}h" if eta < 86400 else f"{eta/86400:.1f}d"
            print(
                f"\r  {done + skipped:>7,}/{total:,}  "
                f"done={done:,}  skipped={skipped:,}  "
                f"errors={len(errors)}  "
                f"rate={rate:.2f}/s  ETA~{eta_str}",
                end="", flush=True,
            )
        elif status.startswith("error:"):
            errors.append((mp_id, status[6:]))
            print(f"\n  WARNING {mp_id}: {status[6:]}")
        elif status.startswith("finished:"):
            workers_finished += 1

    print()
    return {"done": done, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract FairChem UMA all-layer node embeddings for all MP structures"
    )
    parser.add_argument("--structures-dir", default=str(P.STRUCTURES),
                        help="Directory containing CIFs and index.jsonl (default: $MLIP_PROBES_ROOT/structures)")
    parser.add_argument("--results-dir", default=str(P.RESULTS / "fairchem"),
                        help="Root output directory for .pt files (default: $MLIP_PROBES_ROOT/results/fairchem)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Parallel worker processes (default: 2 — UMA weights ~1 GB each)")
    parser.add_argument("--torch-threads", type=int, default=4,
                        help="torch.set_num_threads() per worker (default: 4)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of materials — for local testing only")
    parser.add_argument("--random-init", action="store_true",
                        help="pretraining-vs-architecture control: extract from an "
                             "architecturally identical but randomly-initialised model "
                             "instead of the pretrained checkpoint. Use a separate "
                             "--results-dir (e.g. $MLIP_PROBES_ROOT/results/fairchem_random_init) — "
                             "never write random-init embeddings into the real results tree.")
    parser.add_argument("--random-init-seed", type=int, default=1234,
                        help="seed for --random-init, applied identically in every worker "
                             "so all materials are probed by the SAME random network "
                             "(default: 1234)")
    args = parser.parse_args()

    structures_dir = Path(args.structures_dir)
    results_dir    = Path(args.results_dir)

    jsonl_path = structures_dir / "index.jsonl"
    if not jsonl_path.exists():
        sys.exit(
            f"ERROR: {jsonl_path} not found.\n"
            f"  Run stages/data/fetch_mp.py --structures-dir {structures_dir} first."
        )

    print("Reading index.jsonl ...")
    mp_ids: list[str] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                mp_ids.append(json.loads(line)["mp_id"])

    if args.limit:
        mp_ids = mp_ids[: args.limit]

    total = len(mp_ids)
    already_done = sum(1 for mid in mp_ids if _all_done(results_dir, mid))
    print(f"  {total:,} materials in manifest")
    print(f"  {already_done:,} already complete — will skip")
    print(f"  {total - already_done:,} to process")

    if already_done == total:
        print("Nothing to do — all materials already extracted.")
        return

    _make_dirs(results_dir)

    print("\nPre-warming FairChem UMA model cache ...")
    print("  Tip: export HF_HOME=~/.cache/huggingface to keep weights off the root disk")
    if args.random_init:
        print(f"  --random-init: seed={args.random_init_seed} (every worker reinitialises "
             f"to the same random weights)")
    _load_fairchem(random_init=args.random_init, random_init_seed=args.random_init_seed)
    print("  Model weights cached — workers will load from local cache.\n")

    num_workers = min(args.num_workers, total - already_done)
    chunks: list[list[str]] = [[] for _ in range(num_workers)]
    for i, mid in enumerate(mp_ids):
        chunks[i % num_workers].append(mid)

    print(f"Spawning {num_workers} workers ({args.torch_threads} torch threads each) ...")
    for i, chunk in enumerate(chunks):
        print(f"  worker {i}: {len(chunk):,} materials assigned")

    ctx = mp.get_context("spawn")
    progress_q: mp.Queue = ctx.Queue()

    processes = []
    t0 = time.time()
    for worker_id, chunk in enumerate(chunks):
        p = ctx.Process(
            target=_worker,
            args=(worker_id, chunk, structures_dir, results_dir,
                  args.torch_threads, progress_q,
                  args.random_init, args.random_init_seed),
            daemon=True,
        )
        p.start()
        processes.append(p)

    summary = _listen(progress_q, total, num_workers)
    for p in processes:
        p.join()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} min  ({elapsed / 3600:.2f} h)")
    print(f"  Processed:  {summary['done']:,}")
    print(f"  Skipped:    {summary['skipped']:,}")
    print(f"  Errors:     {len(summary['errors']):,}")

    if summary["errors"]:
        error_log = results_dir / "extract_errors.json"
        error_log.write_text(json.dumps(summary["errors"], indent=2))
        print(f"  Error log:  {error_log}")

    node_gb = sum(
        f.stat().st_size
        for layer in LAYER_KEYS
        for f in (results_dir / layer / "nodes").glob("*.pt")
    ) / 1024 ** 3
    edge_gb = sum(
        f.stat().st_size
        for layer in EDGE_LAYERS
        for f in (results_dir / layer / "edges").glob("*.pt")
    ) / 1024 ** 3
    print(f"  Disk used:  nodes {node_gb:.1f} GB + edges {edge_gb:.1f} GB under {results_dir}")


if __name__ == "__main__":
    main()
