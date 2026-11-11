"""
stages/extract/extract_orb.py — extract ORB v3 all-layer embeddings for all downloaded structures

Reads CIFs from --structures-dir, writes .pt files to --results-dir.

Directory layout created under --results-dir:
    backbone/nodes/      backbone/edges/
    encoder/nodes/       encoder/edges/
    gnn_stacks_0/nodes/  gnn_stacks_0/edges/
    gnn_stacks_1/nodes/  gnn_stacks_1/edges/
    gnn_stacks_2/nodes/  gnn_stacks_2/edges/
    gnn_stacks_3/nodes/  gnn_stacks_3/edges/
    gnn_stacks_4/nodes/  gnn_stacks_4/edges/

All 7 layers are extracted in a single forward pass per material:
    backbone   — final model output (node_features / edge_features)
    encoder    — output of the encoder module
    gnn_stacks_0 … gnn_stacks_4 — output of each GNN message-passing stack

Each .pt contains node OR edge embeddings for one material + metadata.
    Node .pt keys:  node [N,256] float16, structure [256] float16,
                    model, mp_id, elements, frac_coords
    Edge .pt keys:  edge_emb [E,256] float16, edge_structure_emb [256] float16,
                    model, mp_id, element_i, element_j, bond_type,
                    distance [E] float32, bond_vector [E,3] float32

Resume: all 14 .pt files (7 layers × node+edge) must exist for a material to be
        considered done. Partial writes are detected and re-run cleanly.

Parallelism: --num-workers processes (default 4), each with its own ORB model.
             --torch-threads controls torch.set_num_threads() per worker (default 2).
             4 workers × 2 threads = 8 logical CPUs = full utilisation on the
             4-core/8-thread Xeon instance.

HuggingFace cache: model weights (~1 GB) are downloaded to HF_HOME on first run.
             export HF_HOME=~/.cache/huggingface  ← keeps weights off the root disk

Usage:
    # Full run (remote server) — uses correct defaults automatically
    uv run python stages/extract/extract_orb.py

    # Override dirs if needed
    uv run python stages/extract/extract_orb.py \\
        --structures-dir $MLIP_PROBES_ROOT/structures \\
        --results-dir $MLIP_PROBES_ROOT/results/orb

    # Local test — 5 materials, 1 worker, throwaway dirs
    uv run python stages/extract/extract_orb.py \\
        --structures-dir structures_test \\
        --results-dir results_test/orb \\
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
# Layer configuration — all 7 layers, backbone first
# ---------------------------------------------------------------------------
LAYER_KEYS = [
    "backbone",
    "encoder",
    "gnn_stacks_0",
    "gnn_stacks_1",
    "gnn_stacks_2",
    "gnn_stacks_3",
    "gnn_stacks_4",
]


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _make_dirs(results_dir: Path) -> None:
    """Create all output directories under results_dir."""
    for layer in LAYER_KEYS:
        (results_dir / layer / "nodes").mkdir(parents=True, exist_ok=True)
        (results_dir / layer / "edges").mkdir(parents=True, exist_ok=True)
    print(f"  Output dirs ready under {results_dir}")


def _node_path(results_dir: Path, layer: str, mp_id: str) -> Path:
    return results_dir / layer / "nodes" / f"{mp_id}.pt"


def _edge_path(results_dir: Path, layer: str, mp_id: str) -> Path:
    return results_dir / layer / "edges" / f"{mp_id}.pt"


def _all_done(results_dir: Path, mp_id: str) -> bool:
    """Return True only if all 14 .pt files (7 layers × node+edge) exist."""
    return all(
        _node_path(results_dir, k, mp_id).exists() and
        _edge_path(results_dir, k, mp_id).exists()
        for k in LAYER_KEYS
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _reset_all_params(module) -> tuple[int, int, list]:
    """Randomise every parameter of an nn.Module in place. Tries each submodule's
    own reset_parameters() first (respects any custom init scheme); on failure
    (e.g. a <2D weight tensor that breaks torch's default kaiming/fan-based init —
    observed on one ORB Linear submodule) falls back to reinitialising that
    module's own (non-recursive) parameters directly by shape. Verified on both
    ORB and UMA: every parameter tensor changes, and a real forward pass still
    runs cleanly afterward (see the random-init control's verification note)."""
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


ORB_MODELS = {"omat": "orb_v3_conservative_inf_omat", "mpa": "orb_v3_conservative_inf_mpa"}


def _load_orb(random_init: bool = False, random_init_seed: int = 1234, variant: str = "omat"):
    """
    Load ORB v3 model + calculator.
    TORCHDYNAMO_DISABLE must be set before torch is imported.

    random_init: for the pretraining-vs-architecture control (10_random_init_control /
    the label-efficiency work) — after loading the normal pretrained weights, overwrite
    every parameter with a fresh random draw, same architecture, same forward pass.
    torch.manual_seed(random_init_seed) is set immediately before the reset so every
    worker process (each of which loads its own copy of the model) reinitialises to
    the SAME random weights rather than an independent draw per worker — otherwise
    different materials would be probed by different random networks depending on
    which worker happened to process them, a confound the analysis can't see.
    """
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    import torch  # noqa: F401 — imported for side effects (dtype setting)
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.inference.calculator import ORBCalculator

    orbff, atoms_adapter = getattr(pretrained, ORB_MODELS[variant])(
        device="cpu",
        precision="float32-high",
    )
    if random_init:
        torch.manual_seed(random_init_seed)
        n_ok, n_fallback, failures = _reset_all_params(orbff)
        print(f"  [random-init] reset {n_ok} submodules directly, {n_fallback} tensors "
             f"via shape-based fallback ({len(failures)} module types needed the fallback)")
    orb_calc = ORBCalculator(orbff, atoms_adapter=atoms_adapter, device="cpu")
    return orbff, orb_calc


# ---------------------------------------------------------------------------
# All-layer embedding extraction
# ---------------------------------------------------------------------------

def _extract(orbff, orb_calc, atoms, elements: list[str]) -> tuple[dict, dict]:
    """
    Register forward hooks on backbone (model-level), encoder, and each
    gnn_stacks layer. Run one forward pass, capture all 7 layers, return:

        layer_embs : dict[layer_key -> {
                         "node":     [N,256] float16,
                         "edge_emb": [E,256] float16,
                     }]
        edge_meta  : dict with element_i/j, bond_type, distance, bond_vector

    Mean-pooled "structure" and "edge_structure_emb" keys are added after the
    forward pass (no second pass needed).
    """
    import torch  # noqa: F401

    layer_embs: dict = {}
    batch_cache: dict = {}
    handles = []

    saved_compiled = orbff.model._compiled_call_impl
    orbff.model._compiled_call_impl = None          # expose normal hook path

    # Model-level hook: captures backbone output + AtomGraphs for edge geometry
    def backbone_hook(module, inp, out):
        batch = inp[0]
        batch_cache["senders"]   = batch.senders.detach().cpu()
        batch_cache["receivers"] = batch.receivers.detach().cpu()
        batch_cache["vectors"]   = batch.edge_features["vectors"].detach().cpu().float()
        layer_embs["backbone"] = {
            "node":     out["node_features"].detach().cpu().half(),  # [N, 256] float16
            "edge_emb": out["edge_features"].detach().cpu().half(),  # [E, 256] float16
        }
    handles.append(orbff.model.register_forward_hook(backbone_hook))

    # Encoder hook: output is (encoded_nodes, encoded_edges)
    def encoder_hook(module, inp, out):
        nodes, edges = out
        layer_embs["encoder"] = {
            "node":     nodes.detach().cpu().half(),
            "edge_emb": edges.detach().cpu().half(),
        }
    handles.append(orbff.model._encoder.register_forward_hook(encoder_hook))

    # Per-layer hooks on gnn_stacks[i]
    for i, gnn in enumerate(orbff.model.gnn_stacks):
        def _make_hook(key):
            def hook_fn(module, inp, out):
                nodes, edges = out
                layer_embs[key] = {
                    "node":     nodes.detach().cpu().half(),
                    "edge_emb": edges.detach().cpu().half(),
                }
            return hook_fn
        handles.append(gnn.register_forward_hook(_make_hook(f"gnn_stacks_{i}")))

    try:
        atoms.calc = orb_calc
        orb_calc.reset()                            # bypass ASE result cache
        atoms.get_potential_energy()
    finally:
        for h in handles:
            h.remove()
        orbff.model._compiled_call_impl = saved_compiled

    # Add mean-pooled structure embeddings (no extra forward pass)
    for key in layer_embs:
        layer_embs[key]["structure"]          = layer_embs[key]["node"].mean(dim=0)
        layer_embs[key]["edge_structure_emb"] = layer_embs[key]["edge_emb"].mean(dim=0)

    # Edge metadata (topology/geometry — identical across all layers)
    senders   = batch_cache["senders"].tolist()
    receivers = batch_cache["receivers"].tolist()
    elem_i    = [elements[s] for s in senders]
    elem_j    = [elements[r] for r in receivers]
    edge_meta = {
        "element_i":   elem_i,
        "element_j":   elem_j,
        "bond_type":   ["-".join(sorted([a, b])) for a, b in zip(elem_i, elem_j)],
        "distance":    batch_cache["vectors"].norm(dim=-1),   # [E] float32, Angstrom
        "bond_vector": batch_cache["vectors"],                # [E, 3] float32, Angstrom
    }
    return layer_embs, edge_meta


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(worker_id: int, mp_ids: list[str], structures_dir: Path,
            results_dir: Path, torch_threads: int, progress_q,
            random_init: bool = False, random_init_seed: int = 1234, variant: str = "omat") -> None:
    """
    Process a slice of mp_ids. Loads its own ORB model, skips already-done
    materials, writes node+edge .pt files for all 7 layers.
    """
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    import torch
    torch.set_num_threads(torch_threads)

    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    adaptor = AseAtomsAdaptor()

    progress_q.put((worker_id, None, "loading_model"))
    orbff, orb_calc = _load_orb(random_init=random_init, random_init_seed=random_init_seed, variant=variant)
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

            layer_embs, edge_meta = _extract(orbff, orb_calc, atoms, elements)

            for k, embs in layer_embs.items():
                node_p = _node_path(results_dir, k, mp_id)
                edge_p = _edge_path(results_dir, k, mp_id)

                if not node_p.exists():
                    torch.save({
                        "node":        embs["node"],        # [N, 256] float16
                        "structure":   embs["structure"],   # [256]    float16
                        "model":       "orb-v3",
                        "mp_id":       mp_id,
                        "elements":    elements,
                        "frac_coords": frac_coords,
                    }, node_p)

                if not edge_p.exists():
                    torch.save({
                        "edge_emb":           embs["edge_emb"],             # [E, 256] float16
                        "edge_structure_emb": embs["edge_structure_emb"],   # [256]    float16
                        "model":              "orb-v3",
                        "mp_id":              mp_id,
                        "element_i":          edge_meta["element_i"],
                        "element_j":          edge_meta["element_j"],
                        "bond_type":          edge_meta["bond_type"],
                        "distance":           edge_meta["distance"],        # [E]   float32
                        "bond_vector":        edge_meta["bond_vector"],     # [E,3] float32
                    }, edge_p)

            progress_q.put((worker_id, mp_id, "done"))

        except Exception as exc:
            progress_q.put((worker_id, mp_id, f"error: {exc}"))
            errors.append((mp_id, str(exc)))

    progress_q.put((worker_id, None, f"finished:{len(errors)}"))


# ---------------------------------------------------------------------------
# Progress listener (runs in main process)
# ---------------------------------------------------------------------------

def _listen(progress_q, total: int, num_workers: int) -> dict:
    """
    Drain the progress queue, print a running status line, return summary.
    Exits when all workers report 'finished'.
    """
    done = 0
    skipped = 0
    errors: list[tuple[str, str]] = []
    workers_finished = 0
    t0 = time.time()

    while workers_finished < num_workers:
        worker_id, mp_id, status = progress_q.get()

        if status == "loading_model":
            print(f"  worker {worker_id}: loading ORB model ...")
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

    print()  # newline after running status line
    return {"done": done, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ORB v3 all-layer embeddings for all MP structures"
    )
    parser.add_argument("--structures-dir", default=str(P.STRUCTURES),
                        help="Directory containing CIFs and index.jsonl "
                             "(default: $MLIP_PROBES_ROOT/structures)")
    parser.add_argument("--results-dir", default=str(P.RESULTS / "orb"),
                        help="Root output directory for .pt files "
                             "(default: $MLIP_PROBES_ROOT/results/orb)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Parallel worker processes (default: 4)")
    parser.add_argument("--torch-threads", type=int, default=2,
                        help="torch.set_num_threads() per worker (default: 2)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of materials — for local testing only")
    parser.add_argument("--random-init", action="store_true",
                        help="pretraining-vs-architecture control: extract from an "
                             "architecturally identical but randomly-initialised model "
                             "instead of the pretrained checkpoint. Use a separate "
                             "--results-dir (e.g. $MLIP_PROBES_ROOT/results/orb_random_init) — never "
                             "write random-init embeddings into the real results tree.")
    parser.add_argument("--orb-variant", choices=list(ORB_MODELS), default="omat",
                        help="which ORB-v3 checkpoint: omat (OMat24; the paper) or mpa (MPtrj + Alexandria)")
    parser.add_argument("--ids-file", default=None,
                        help="text file with one mp_id per line: extract only these materials")
    parser.add_argument("--random-init-seed", type=int, default=1234,
                        help="seed for --random-init, applied identically in every worker "
                             "so all materials are probed by the SAME random network "
                             "(default: 1234)")
    args = parser.parse_args()

    structures_dir = Path(args.structures_dir)
    results_dir    = Path(args.results_dir)

    # --- Validate inputs ------------------------------------------------------
    jsonl_path = structures_dir / "index.jsonl"
    if not jsonl_path.exists():
        sys.exit(
            f"ERROR: {jsonl_path} not found.\n"
            f"  Run stages/data/fetch_mp.py --structures-dir {structures_dir} first."
        )

    # --- Read manifest --------------------------------------------------------
    print("Reading index.jsonl ...")
    mp_ids: list[str] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                mp_ids.append(json.loads(line)["mp_id"])

    if args.ids_file:
        wanted = {l.strip() for l in open(args.ids_file) if l.strip()}
        mp_ids = [m for m in mp_ids if m in wanted]
        print(f"  --ids-file: {len(mp_ids):,} of {len(wanted):,} requested ids are in the manifest")
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

    # --- Create output directories --------------------------------------------
    _make_dirs(results_dir)

    # --- Pre-warm HuggingFace model cache -------------------------------------
    # Downloads weights once in the main process so all 4 workers load from
    # local cache rather than racing to download simultaneously.
    print("\nPre-warming ORB model cache ...")
    print("  Tip: export HF_HOME=~/.cache/huggingface to keep weights off the root disk")
    if args.random_init:
        print(f"  --random-init: seed={args.random_init_seed} (every worker reinitialises "
             f"to the same random weights)")
    _load_orb(random_init=args.random_init, random_init_seed=args.random_init_seed, variant=args.orb_variant)
    print("  Model weights cached — workers will load from local cache.\n")

    # --- Split work across workers --------------------------------------------
    num_workers = min(args.num_workers, total - already_done)
    chunks: list[list[str]] = [[] for _ in range(num_workers)]
    for i, mid in enumerate(mp_ids):
        chunks[i % num_workers].append(mid)

    print(f"Spawning {num_workers} workers "
          f"({args.torch_threads} torch threads each) ...")
    for i, chunk in enumerate(chunks):
        print(f"  worker {i}: {len(chunk):,} materials assigned")

    # --- Spawn workers --------------------------------------------------------
    ctx = mp.get_context("spawn")       # safer than fork with torch
    progress_q: mp.Queue = ctx.Queue()

    processes = []
    t0 = time.time()
    for worker_id, chunk in enumerate(chunks):
        p = ctx.Process(
            target=_worker,
            args=(worker_id, chunk, structures_dir, results_dir,
                  args.torch_threads, progress_q,
                  args.random_init, args.random_init_seed, args.orb_variant),
            daemon=True,
        )
        p.start()
        processes.append(p)

    # --- Collect progress -----------------------------------------------------
    summary = _listen(progress_q, total, num_workers)
    for p in processes:
        p.join()

    # --- Final summary --------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} min  ({elapsed / 3600:.2f} h)")
    print(f"  Processed:  {summary['done']:,}")
    print(f"  Skipped:    {summary['skipped']:,}")
    print(f"  Errors:     {len(summary['errors']):,}")

    if summary["errors"]:
        error_log = results_dir / "extract_errors.json"
        error_log.write_text(json.dumps(summary["errors"], indent=2))
        print(f"  Error log:  {error_log}")

    total_gb = sum(
        f.stat().st_size
        for layer in LAYER_KEYS
        for subdir in ("nodes", "edges")
        for f in (results_dir / layer / subdir).glob("*.pt")
    ) / 1024 ** 3
    print(f"  Disk used:  {total_gb:.1f} GB under {results_dir}")


if __name__ == "__main__":
    main()
