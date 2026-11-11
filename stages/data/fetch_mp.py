"""
stages/data/fetch_mp.py — download all non-deprecated MP structures + write CIFs + JSONL manifest

API key: set MP_API_KEY in the environment (MPRester reads it automatically).
    export MP_API_KEY=your_key_here

Outputs (under --structures-dir, default $MLIP_PROBES_ROOT/structures):
    meta.json        run-level metadata (total count, seed, timestamps)
    index.jsonl      one JSON record per material, appended as each CIF is written
    <mp_id>.cif      one CIF per material

Resume: already-written mp_ids are read from index.jsonl at startup and skipped.

Validation: runs automatically after download — nulls out physically implausible
numeric values in-place in index.jsonl (rewrites the file once at the end).
Pass --no-validate to skip.

Usage:
    uv run python stages/data/fetch_mp.py
    uv run python stages/data/fetch_mp.py --structures-dir $MLIP_PROBES_ROOT/structures --validate
    uv run python stages/data/fetch_mp.py --structures-dir structures_test --limit 20  # local test, safe
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from mlipprobes import paths as P  # noqa: E402  (one root for every input and output)
del _sys, _Path
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Fields requested from the MP API
# ---------------------------------------------------------------------------
SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "nsites",
    "symmetry",
    "volume",
    "energy_above_hull",
    "formation_energy_per_atom",
    "band_gap",
    "bulk_modulus",
    "total_magnetization",
    "density",
    "structure",
]

# Physical plausibility bounds: property -> (min, max), None = no bound
PLAUSIBILITY_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "bulk_modulus":              (0,    500),
    "band_gap":                  (0,     25),
    "formation_energy_per_atom": (-15,   10),
    "energy_above_hull":         (0,   None),
    "density":                   (0,     30),
    "volume":                    (0,   None),
    "total_magnetization":       (None, None),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_api_key() -> None:
    """
    Ensure MP_API_KEY is available as an env var.
    Falls back to a git-ignored api_keys.py (for local dev). Looked up in the
    project root and in notebooks/ (where api_keys.example.py lives).
    """
    if os.environ.get("MP_API_KEY"):
        return
    # Local dev fallback: api_keys.py in the project root or notebooks/
    script_dir = Path(__file__).parent
    for cand in (script_dir.parent, script_dir.parent / "notebooks"):
        sys.path.insert(0, str(cand))
    try:
        from api_keys import MP_API_KEY  # type: ignore[import]
        os.environ["MP_API_KEY"] = MP_API_KEY
        print("  (API key loaded from api_keys.py)")
    except ImportError:
        sys.exit(
            "ERROR: MP_API_KEY not set and api_keys.py not found.\n"
            "  export MP_API_KEY=your_key_here\n"
            "  or: cp notebooks/api_keys.example.py notebooks/api_keys.py  (then add your key)\n"
            "  Get a key at https://materialsproject.org/dashboard"
        )


def _doc_to_record(doc, struct_dir: Path) -> dict:
    """Convert an MPRester SummaryDoc to a manifest record dict."""
    mp_id = doc.material_id
    bm = doc.bulk_modulus
    return {
        "mp_id":                     mp_id,
        "formula":                   doc.formula_pretty,
        "cif_file":                  str(struct_dir / f"{mp_id}.cif"),
        "nsites":                    doc.nsites,
        "crystal_system":            doc.symmetry.crystal_system.value if doc.symmetry else None,
        "space_group":               doc.symmetry.symbol               if doc.symmetry else None,
        "volume":                    round(doc.volume, 3)               if doc.volume   else None,
        "energy_above_hull":         doc.energy_above_hull,
        "formation_energy_per_atom": doc.formation_energy_per_atom,
        "band_gap":                  doc.band_gap,
        "bulk_modulus":              round(bm.get("vrh"), 4) if bm and bm.get("vrh") is not None else None,
        "total_magnetization":       doc.total_magnetization,
        "density":                   doc.density,
    }


def _load_done_ids(jsonl_path: Path) -> set[str]:
    """Return the set of mp_ids already present in index.jsonl."""
    if not jsonl_path.exists():
        return set()
    done = set()
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["mp_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def _validate_manifest(jsonl_path: Path) -> None:
    """
    Read index.jsonl, null out values outside PLAUSIBILITY_BOUNDS, rewrite in place.
    Prints a summary of what was changed.
    """
    print("\n-- Validation pass --")
    records = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    nulled: dict[str, list[tuple[str, float]]] = {}
    for rec in records:
        for prop, (lo, hi) in PLAUSIBILITY_BOUNDS.items():
            val = rec.get(prop)
            if val is None:
                continue
            out_of_range = (lo is not None and val < lo) or (hi is not None and val > hi)
            if out_of_range:
                nulled.setdefault(prop, []).append((rec["mp_id"], val))
                rec[prop] = None

    if not nulled:
        print("  All values within plausibility bounds — nothing changed.")
    else:
        total = sum(len(v) for v in nulled.values())
        for prop, cases in sorted(nulled.items()):
            lo, hi = PLAUSIBILITY_BOUNDS[prop]
            bound_str = f"[{lo}, {hi}]"
            print(f"  {prop:<35} {len(cases):>5} nulled  (bounds {bound_str})")
            for mp_id, val in cases[:3]:
                print(f"      {mp_id}: {val}")
            if len(cases) > 3:
                print(f"      ... and {len(cases) - 3} more")
        print(f"  Total: {total} value(s) set to null")

        tmp = jsonl_path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        tmp.replace(jsonl_path)
        print(f"  index.jsonl rewritten ({jsonl_path.stat().st_size / 1024 / 1024:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download MP structures + write CIFs + JSONL manifest")
    parser.add_argument("--structures-dir", default=str(P.STRUCTURES),
                        help="Directory for CIFs and manifest (default: $MLIP_PROBES_ROOT/structures)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap total materials (for local testing; omit for full MP)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed used when --limit subsamples (default: 42)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                        help="MPRester chunk size per API page (default: 1000)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip plausibility validation pass after download")
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip download, only run validation on existing index.jsonl")
    args = parser.parse_args()

    struct_dir = Path(args.structures_dir)
    jsonl_path = struct_dir / "index.jsonl"
    meta_path  = struct_dir / "meta.json"

    # --validate-only: skip straight to validation
    if args.validate_only:
        if not jsonl_path.exists():
            sys.exit(f"ERROR: {jsonl_path} not found — nothing to validate.")
        _validate_manifest(jsonl_path)
        return

    _check_api_key()

    # Import here so the --validate-only path doesn't need it
    try:
        from mp_api.client import MPRester
    except ImportError:
        sys.exit("ERROR: mp_api not installed.  uv add mp-api")

    try:
        from tqdm import tqdm
    except ImportError:
        # Graceful fallback: wrap iterable with identity
        def tqdm(it, **kwargs):  # type: ignore[misc]
            return it

    struct_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume: find already-written mp_ids -----------------------------------
    done_ids = _load_done_ids(jsonl_path)
    if done_ids:
        print(f"Resuming: {len(done_ids):,} mp_ids already in index.jsonl — will skip them.")

    # --- Fetch docs from MP API -----------------------------------------------
    print(f"Connecting to Materials Project API ...")
    t0 = time.time()

    with MPRester() as mpr:  # reads MP_API_KEY from env automatically

        # First, get the total count (fast, no structure data)
        count_docs = mpr.materials.summary.search(
            deprecated=False,
            fields=["material_id"],
            chunk_size=args.chunk_size,
        )
        all_ids = [doc.material_id for doc in count_docs]

    total_available = len(all_ids)
    print(f"  MP total (non-deprecated): {total_available:,}")

    # Apply --limit subsampling if requested
    if args.limit is not None and args.limit < total_available:
        import random
        rng = random.Random(args.seed)
        # Don't re-fetch already-done ones; preserve them
        remaining_pool = [mid for mid in all_ids if mid not in done_ids]
        need = max(0, args.limit - len(done_ids))
        selected_ids = set(rng.sample(remaining_pool, min(need, len(remaining_pool))))
        selected_ids |= done_ids  # keep already-done ones
        print(f"  Subsampled to {args.limit:,} (seed={args.seed})")
    else:
        selected_ids = set(all_ids)

    to_fetch = [mid for mid in selected_ids if mid not in done_ids]
    print(f"  To fetch this run: {len(to_fetch):,}")

    if not to_fetch:
        print("Nothing left to fetch — all done.")
        if not args.no_validate:
            _validate_manifest(jsonl_path)
        return

    # --- Write meta.json (update on each run) ---------------------------------
    meta = {
        "started_at":        datetime.now(timezone.utc).isoformat(),
        "total_available":   total_available,
        "target_count":      args.limit or total_available,
        "seed":              args.seed,
        "structures_dir":    str(struct_dir),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    # --- Stream docs in chunks, write CIF + append to JSONL ------------------
    # Fetch full docs (with structure) only for the ids we still need
    chunk_size = args.chunk_size
    total_written = len(done_ids)
    errors: list[tuple[str, str]] = []

    print(f"\nFetching {len(to_fetch):,} structures in chunks of {chunk_size} ...")

    with MPRester() as mpr, jsonl_path.open("a") as manifest_f:
        # Process to_fetch in batches to avoid sending 150k ids in one request
        for batch_start in range(0, len(to_fetch), chunk_size):
            batch_ids = to_fetch[batch_start : batch_start + chunk_size]

            try:
                docs = mpr.materials.summary.search(
                    material_ids=batch_ids,
                    fields=SUMMARY_FIELDS,
                    chunk_size=chunk_size,
                )
            except Exception as exc:
                print(f"\n  WARNING: batch {batch_start}–{batch_start + len(batch_ids)} failed: {exc}")
                errors.extend((mid, str(exc)) for mid in batch_ids)
                continue

            batch_iter = tqdm(
                docs,
                total=len(batch_ids),
                desc=f"batch {batch_start // chunk_size + 1}/{(len(to_fetch) + chunk_size - 1) // chunk_size}",
                unit="mat",
                leave=False,
            )

            for doc in batch_iter:
                mp_id = doc.material_id
                try:
                    cif_path = struct_dir / f"{mp_id}.cif"
                    doc.structure.to(fmt="cif", filename=str(cif_path))
                    record = _doc_to_record(doc, struct_dir)
                    manifest_f.write(json.dumps(record) + "\n")
                    manifest_f.flush()
                    total_written += 1
                except Exception as exc:
                    errors.append((mp_id, str(exc)))
                    print(f"\n  WARNING: {mp_id} failed: {exc}")

            elapsed = time.time() - t0
            rate = total_written / elapsed if elapsed > 0 else 0
            remaining = len(selected_ids) - total_written
            eta_s = remaining / rate if rate > 0 else float("inf")
            eta_str = f"{eta_s / 3600:.1f} h" if eta_s < 86400 else f"{eta_s / 86400:.1f} d"
            print(
                f"  Progress: {total_written:,}/{len(selected_ids):,}  "
                f"({rate:.1f} mat/s  ETA ~{eta_str})"
            )

    # --- Final summary --------------------------------------------------------
    elapsed_total = time.time() - t0
    print(f"\nDone in {elapsed_total / 60:.1f} min")
    print(f"  CIFs written:  {total_written:,}")
    print(f"  Errors:        {len(errors):,}")
    print(f"  index.jsonl:   {jsonl_path.stat().st_size / 1024 / 1024:.1f} MB")

    if errors:
        error_log = struct_dir / "fetch_errors.json"
        error_log.write_text(json.dumps(errors, indent=2))
        print(f"  Error list:    {error_log}")

    # Update meta with completion info
    meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    meta["total_written"] = total_written
    meta["total_errors"]  = len(errors)
    meta_path.write_text(json.dumps(meta, indent=2))

    if not args.no_validate:
        _validate_manifest(jsonl_path)


if __name__ == "__main__":
    main()
