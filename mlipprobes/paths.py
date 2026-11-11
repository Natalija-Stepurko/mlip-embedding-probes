"""One root for everything the pipeline reads and writes.

Set MLIP_PROBES_ROOT (default: ./mlip-probes, relative to where you run the command). Every stage writes under
it and finds the previous stages' outputs there, so a single setting moves the whole study:

    structures/            Materials Project CIFs + index.jsonl         (stages/data/fetch_mp.py)
    structures_external/   JARVIS-DFT 3D and OQMD                        (stages/data/fetch_external.py)
    results/orb/           ORB-v3 caches, geometry, probes               (stages/extract .. probes)
    results/fairchem/      UMA-S caches, geometry, probes
    results/...            every other stage's outputs, one folder each

Scripts also accept explicit --*-dir arguments; those override the root for that one input or output.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("MLIP_PROBES_ROOT", "mlip-probes")).expanduser().resolve()
STRUCTURES = ROOT / "structures"
STRUCTURES_EXTERNAL = ROOT / "structures_external"
RESULTS = ROOT / "results"
