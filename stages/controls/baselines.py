#!/usr/bin/env python3
"""Baselines, selection and leakage controls on the trained probes (paper §A.4, Tables S5-S9, S11, S18, S19).
Each flag runs one control and writes results/proppred_controls/<name>/; several flags can be combined.

  --composition-baseline   Magpie composition-only XGBoost on the same split; paired gain of each embedding
  --val-selection          re-select every configuration on a validation split carved from the training data
  --grouped-split          composition-grouped and chemical-system-grouped splits against the random split
  --probe-selectivity      real-label minus permuted-label score (control task)
  --knn-predictor          the parameter-free k-NN predictor at the selected configurations (Euclidean, cosine)
  --same-material          per-atom k-NN purity with same-material neighbours excluded
  --whitening              material-level k-NN purity under PCA-whitened distance
  --log-bulk-modulus       bulk modulus re-probed on log10(K_VRH)
  --band-gap-subset        band gap re-scored on the non-metal test subset
  --invariant-crystal      crystal-system CI on the rotation-invariant footing

    uv run python stages/controls/baselines.py --composition-baseline --knn-predictor
    uv run python stages/controls/baselines.py --val-selection --grouped-split --probe-selectivity
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stages.controls._common import parser, load, L


def main():
    ap = parser(__doc__)
    for flag in ("composition-baseline", "val-selection", "grouped-split", "probe-selectivity", "knn-predictor",
                 "same-material", "whitening", "log-bulk-modulus", "band-gap-subset", "invariant-crystal"):
        ap.add_argument("--" + flag, action="store_true")
    ap.add_argument("--comp-features", choices=["magpie", "element_fractions"], default="magpie")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--val-sweep-cap", type=int, default=50000)
    a = ap.parse_args()
    orb_dir, uma_dir, orb_rows, uma_rows, targets, ctl = load(a)
    ran = False
    if a.composition_baseline:
        L.run_composition_baseline(orb_rows, uma_rows, orb_dir, uma_dir, ctl / "composition_baseline", targets, a.seed, a.n_jobs, a.bootstrap, a.comp_features); ran = True
    if a.val_selection:
        L.run_val_selection(orb_rows, uma_rows, orb_dir, uma_dir, ctl / "val_selection", targets, a.val_frac, a.val_sweep_cap, a.seed, a.n_jobs, a.bootstrap); ran = True
    if a.log_bulk_modulus:
        L.run_log_bulk_modulus(orb_rows, uma_rows, orb_dir, uma_dir, ctl / "log_bulk_modulus", a.seed, a.n_jobs, a.bootstrap); ran = True
    if a.probe_selectivity:
        L.run_probe_selectivity(orb_rows, uma_rows, orb_dir, uma_dir, ctl / "probe_selectivity", targets, a.seed, a.n_jobs); ran = True
    if a.band_gap_subset:
        L.run_band_gap_subset(orb_dir, uma_dir, orb_rows, uma_rows, ctl / "band_gap_subset", a.seed, a.n_jobs); ran = True
    if a.knn_predictor:
        L.run_knn_predictor(orb_dir, uma_dir, orb_rows, uma_rows, ctl / "knn_predictor", a.seed); ran = True
    if a.same_material:
        L.run_same_material(orb_dir, uma_dir, ctl / "same_material", a.seed); ran = True
    if a.grouped_split:
        L.run_grouped_split(orb_rows, uma_rows, orb_dir, uma_dir, ctl / "grouped_split", targets, a.seed, a.n_jobs); ran = True
    if a.whitening:
        L.run_whitening(orb_rows, uma_rows, orb_dir, uma_dir, ctl / "whitening", a.seed); ran = True
    if a.invariant_crystal:
        L.run_invariant_crystal(orb_rows, uma_rows, orb_dir, uma_dir, ctl / "invariant_crystal", a.seed, a.n_jobs, a.bootstrap); ran = True
    if not ran:
        ap.error("choose at least one control flag (see --help)")


if __name__ == "__main__":
    main()
