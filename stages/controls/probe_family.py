#!/usr/bin/env python3
"""Probe family (paper §A.12, §B.5, Tables S2, S12, Fig. S3): at each target's selected configuration, the linear
probe, XGBoost, the k-NN predictor and a multilayer perceptron on the same rows, with RMSE, fit time and paired
bootstrap intervals; and the check that the linear probe selects the same layer as XGBoost.

  --probe-family   the four probes at the selected configurations  -> results/proppred_controls/probe_family/
  --mlp-probe      the MLP alone                                   -> results/proppred_controls/mlp_probe/
  --probe-peaks    linear-probe peak layer against XGBoost's, from the sweep (no new fits) -> probe_peaks/

    uv run python stages/controls/probe_family.py --probe-family --probe-peaks
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stages.controls._common import parser, load, L


def main():
    ap = parser(__doc__)
    for flag in ("probe-family", "mlp-probe", "probe-peaks"):
        ap.add_argument("--" + flag, action="store_true")
    a = ap.parse_args()
    orb_dir, uma_dir, orb_rows, uma_rows, targets, ctl = load(a)
    if a.probe_family:
        L.run_probe_family(orb_rows, uma_rows, orb_dir, uma_dir, ctl / "probe_family", targets, a.seed, a.n_jobs, a.bootstrap)
    if a.mlp_probe:
        L.run_mlp_probe(orb_rows, uma_rows, orb_dir, uma_dir, ctl / "mlp_probe", targets, a.seed, a.n_jobs)
    if a.probe_peaks:
        L.run_probe_peaks(orb_rows, uma_rows, ctl / "probe_peaks", targets)
    if not (a.probe_family or a.mlp_probe or a.probe_peaks):
        ap.error("choose at least one of --probe-family, --mlp-probe, --probe-peaks")


if __name__ == "__main__":
    main()
