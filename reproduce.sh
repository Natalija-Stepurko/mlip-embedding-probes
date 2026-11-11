#!/usr/bin/env bash
# Route A, end to end, in the order the README documents. Every stage is resume-safe, so the script can be
# re-run after an interruption. Needs MP_API_KEY, a Hugging Face login with the UMA terms accepted, and
# MLIP_PROBES_ROOT (default ./mlip-probes). Stops at the first failing stage.
set -euo pipefail
export MLIP_PROBES_ROOT="${MLIP_PROBES_ROOT:-./mlip-probes}"
J="${N_JOBS:-8}"
R="$MLIP_PROBES_ROOT/results"
run() { echo "== $(date -u +%FT%TZ) $*"; "$@"; }

run uv run python stages/data/fetch_mp.py

for m in orb uma; do
  run uv run python stages/extract/extract_$m.py --num-workers $J --torch-threads 1
  run uv run python stages/geometry/organisation_$m.py --num-workers $J
  run uv run python stages/geometry/health_qc_$m.py --num-workers $J
done
run uv run python stages/probes/probes_orb.py --num-workers $J
run uv run python stages/probes/probes_uma.py --n-jobs $J
run uv run python stages/probes/significance.py --n-jobs $J
run uv run python stages/probes/repeated_split.py --n-jobs $J

run uv run python stages/controls/baselines.py --composition-baseline --val-selection --grouped-split \
    --probe-selectivity --knn-predictor --same-material --whitening --log-bulk-modulus --band-gap-subset --invariant-crystal --n-jobs $J
run uv run python stages/controls/probe_family.py --probe-family --mlp-probe --probe-peaks --n-jobs $J
run uv run python stages/controls/rotation.py --rotation --rotation-invariance --rotinv-sub 300 --rotinv-n 5 --n-jobs $J

run uv run python stages/extract/extract_orb.py --results-dir $R/orb_random_init --random-init --num-workers $J --torch-threads 1
run uv run python stages/extract/extract_uma.py --results-dir $R/fairchem_random_init --random-init --num-workers $J --torch-threads 1
run uv run python stages/probes/probes_orb.py --embeddings-dir $R/orb_random_init --output-dir $R/orb_random_init/property_prediction --num-workers $J
run uv run python stages/geometry/organisation_uma.py --embeddings-dir $R/fairchem_random_init --plots-dir $R/fairchem_random_init/plots --num-workers $J
run uv run python stages/probes/probes_uma.py --cache-root $R/fairchem_random_init/plots --output-dir $R/fairchem_random_init/property_prediction --n-jobs $J
run uv run python stages/controls/random_init.py
run uv run python stages/controls/rotation.py --rotation-random-uma --n-jobs $J

run uv run python stages/controls/matched_models.py --make-subsample
run uv run python stages/extract/extract_orb.py --orb-variant mpa --ids-file $R/matched_data/subsample_ids.txt --results-dir $R/orb_mpa --num-workers $J --torch-threads 1
run uv run --project envs/mace python stages/extract/extract_mace.py --ids-file $R/matched_data/subsample_ids.txt --results-dir $R/mace_omat --num-workers 4 --torch-threads 2
run uv run python stages/controls/matched_models.py --n-jobs $J

run uv run python stages/downstream/screening_enrichment.py --n-jobs $J
for arm in orb uma magpie; do for axis in train test; do
  run uv run python stages/downstream/label_efficiency.py curves --arm $arm --axis $axis --n-jobs $J
done; done
for pass in summarise gain split-fraction; do run uv run python stages/downstream/label_efficiency.py $pass; done
run uv run python stages/downstream/label_efficiency.py gnn --n-jobs $J

run uv run python stages/convergence/convergence.py
run uv run python stages/convergence/convergence.py --uma-feature ang_norm --functional --residualise
run uv run python stages/convergence/convergence.py --stitch-grid --skip-grids
run uv run python stages/convergence/convergence_significance.py

run uv run python stages/data/fetch_external.py --source jarvis-tools
run uv run python stages/data/fetch_external.py --source oqmd --oqmd-n 40000
for db in jarvis_dft_3d oqmd; do
  run uv run python stages/extract/extract_orb.py --structures-dir $MLIP_PROBES_ROOT/structures_external/$db --results-dir $R/external/$db/orb --num-workers $J --torch-threads 1
  run uv run python stages/extract/extract_uma.py --structures-dir $MLIP_PROBES_ROOT/structures_external/$db --results-dir $R/external/$db/fairchem --num-workers $J --torch-threads 1
  run uv run python stages/transfer/transfer.py --ext-name $db --within-repeats 5 --n-jobs $J
done
run uv run python stages/transfer/label_shift.py --n-jobs $J
run uv run python stages/transfer/matching_audit.py --n 1000
echo "== done. Compare with the paper: uv run python stages/check/check_numbers.py --hf-repo NatalijaStepurko/orb-mp-embeddings"
