#!/usr/bin/env python3
"""Rotation controls (paper §3.4, §A.4, §A.12, Tables S14-S15, Fig. 4b). Both re-embed structures through the
frozen models, so they need the model weights and the extraction modules.

  --rotation              crystal-system probes re-scored on a test subsample after a random rigid rotation of
                          each structure (every UMA-S view, every ORB-v3 read-out)   -> proppred_controls/rotation/
  --rotation-invariance   feature-level test: relative change of every pooled read-out under rotation, the identity
                          pass against the cache, edge counts, crystal-system F1 per rotation -> rotation_invariance/

  --rotation-random-uma   the --rotation test for the random-weight UMA-S network (extract_uma.py --random-init tree
                          at results/fairchem_random_init, probes from its own property_prediction run; ORB skipped)
                                                                                    -> proppred_controls/rotation_random_uma/

    uv run python stages/controls/rotation.py --rotation --rotation-sub 1000
    uv run python stages/controls/rotation.py --rotation-random-uma
    uv run python stages/controls/rotation.py --rotation-invariance --rotinv-sub 300 --rotinv-n 5
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stages.controls._common import parser, load, L


def main():
    ap = parser(__doc__)
    ap.add_argument("--rotation", action="store_true")
    ap.add_argument("--rotation-sub", type=int, default=1000, help="test materials rotated for --rotation")
    ap.add_argument("--rotation-random-uma", action="store_true")
    ap.add_argument("--rotation-invariance", action="store_true")
    ap.add_argument("--rotinv-sub", type=int, default=300, help="materials for --rotation-invariance")
    ap.add_argument("--rotinv-n", type=int, default=5, help="rotations per material for --rotation-invariance")
    a = ap.parse_args()
    orb_dir, uma_dir, orb_rows, uma_rows, targets, ctl = load(a)
    if a.rotation:
        L.run_rotation(orb_dir, uma_dir, ctl / "rotation", a.rotation_sub, a.seed, a.n_jobs)
    if a.rotation_random_uma:
        L.run_rotation(orb_dir, L.P.RESULTS / "fairchem_random_init" / "property_prediction", ctl / "rotation_random_uma",
                       a.rotation_sub, a.seed, a.n_jobs, uma_random_init=True)
    if a.rotation_invariance:
        L.run_rotation_invariance(orb_dir, uma_dir, ctl / "rotation_invariance", a.rotinv_sub, a.rotinv_n, a.seed, a.n_jobs)
    if not (a.rotation or a.rotation_random_uma or a.rotation_invariance):
        ap.error("choose --rotation, --rotation-random-uma and/or --rotation-invariance")


if __name__ == "__main__":
    main()
