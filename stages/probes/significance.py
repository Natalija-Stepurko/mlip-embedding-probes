#!/usr/bin/env python3
"""Paired-bootstrap significance of the probe results (the paper's Table S3 and the UMA-vs-ORB, XGBoost-vs-linear
and node-vs-edge comparisons): B resamples of the fixed seed-42 test set, Benjamini-Hochberg across the twelve
targets of the sweep. Reads both models' metrics.csv (stages/probes/probes_{orb,uma}.py) and refits the selected
configurations once. Writes results/proppred_significance/.

    uv run python stages/probes/significance.py            # all targets
    uv run python stages/probes/significance.py --targets band_gap,crystal_system --n-jobs 8
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mlipprobes import significance_lib as L


def main():
    # the core run is the library's own main() without any control flag
    return L.main()


if __name__ == "__main__":
    raise SystemExit(main())
