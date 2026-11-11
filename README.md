# mlip-embedding-probes

Code for **"Out-of-training properties in frozen interatomic potential embeddings are decodable and
ordered by physical scale"**. This repository and the two Hugging Face datasets it reads are shared by
both papers from this study, at the AI4Mat workshop (AI for Accelerated Materials Design) and at the
AI for Science workshop, both at NeurIPS 2026; code and data are identical for the two.

Two foundation machine-learned interatomic potentials, **ORB-v3** (`orb_v3_conservative_inf_omat`,
non-equivariant) and **UMA-S** (`uma-s-1p2`, SO(3)-equivariant), are run frozen over 154,875 Materials
Project crystals. Every layer's node and edge activations are cached and probed for nine material
properties. The code here recomputes every number in the paper: the probe scores, their statistics, the
controls, the cross-model comparison and the cross-database test. It contains no manuscript, figure or
table-formatting code; the published numbers themselves live in the dataset (below), and the last stage
compares your run against them.

The paper's data: two Hugging Face dataset repositories, one permissive
([`NatalijaStepurko/orb-mp-embeddings`](https://huggingface.co/datasets/NatalijaStepurko/orb-mp-embeddings):
Materials Project metadata, targets and structures, ORB-v3 embeddings, the two check tables) and one gated
([`NatalijaStepurko/uma-mp-embeddings-gated`](https://huggingface.co/datasets/NatalijaStepurko/uma-mp-embeddings-gated):
UMA-S embeddings, FAIR Chemistry License).

If you use this code, please cite: Stepurko, N. and Laulainen, J. (2026). *Out-of-training properties in frozen interatomic potential embeddings are decodable and ordered by physical scale.* Presented at the AI4Mat workshop (AI for Accelerated Materials Design) and the AI for Science workshop, both at NeurIPS 2026; the same code and datasets accompany both papers.

---

## What you need before you start

| | Needed for | Notes |
|---|---|---|
| **Python 3.12 and [`uv`](https://docs.astral.sh/uv/)** | everything | `uv sync --locked` builds the exact environment from `uv.lock`; without uv, `pip install -r requirements.txt` installs the same versions (see *Setup*) |
| **Materials Project API key** | `fetch_mp.py` | free, from [next-gen.materialsproject.org](https://next-gen.materialsproject.org); `export MP_API_KEY=…` |
| **Hugging Face account with the UMA terms accepted** | `extract_uma.py` | accept at [huggingface.co/facebook/UMA](https://huggingface.co/facebook/UMA), then `hf auth login` |
| **A CPU machine with 8 cores and ≥ 64 GB RAM** | all | nothing uses a GPU; the paper's numbers come from one 8-core box |
| **About 1.2 TB of disk** | route A | the per-layer activation caches for both models dominate; route B needs about 30 GB |
| **Time** | route A | about two days for the main study plus about three for the controls and the cross-database test |

Optional: the MACE-OMAT-0 checkpoint (matched-model control) is downloaded by `extract_mace.py` under its
own licence (Academic Software License), which that script asks you to accept.

### One root for everything

Every stage reads and writes under one folder, set once:

```bash
export MLIP_PROBES_ROOT=/wherever/you/have/space      # default: ./mlip-probes
```

Inside it the stages create `structures/`, `structures_external/` and `results/<stage>/`. Nothing else is
written anywhere. Every script also takes explicit `--*-dir` arguments if you want one input or output
elsewhere; `--help` lists them.

---

## Setup

```bash
git clone https://github.com/Natalija-Stepurko/mlip-embedding-probes && cd mlip-embedding-probes
uv sync --locked                               # main environment, Python 3.12, exactly as in uv.lock
export MP_API_KEY=…                            # Materials Project
hf auth login                                  # Hugging Face, for the UMA-S weights
export MLIP_PROBES_ROOT=/wherever/you/have/space
```

Run every command below through `uv run python …`. The MACE control alone uses a second environment,
`envs/mace/` (its own `pyproject.toml` and `uv.lock`), because MACE pins a different torch / e3nn stack:
`uv run --project envs/mace python stages/extract/extract_mace.py …`.

`--locked` makes `uv sync` stop with an error if `uv.lock` no longer matches `pyproject.toml`, so the
environment is never re-resolved to newer versions. The main versions in the lock file:

| package | version | package | version |
|---|---|---|---|
| Python | 3.12 | torch | 2.8.0 |
| orb-models | 0.6.2 | fairchem-core | 2.19.0 |
| xgboost | 3.3.0 | scikit-learn | 1.8.0 |
| numpy | 2.4.4 | pandas | 2.3.3 |
| pymatgen | 2026.3.23 | matminer | 0.10.1 |
| umap-learn | 0.5.12 | hdbscan | 0.8.44 |
| mace-torch (`envs/mace`) | 0.3.16 | e3nn (`envs/mace`) | 0.4.4 |

**Without uv.** `requirements.txt` and `envs/mace/requirements.txt` are exported from the two lock files
(`uv export --frozen --no-hashes --no-emit-project`) and pin every package. Make one Python 3.12 virtual
environment for each, then:

```bash
pip install -r requirements.txt                 # main environment
pip install -r envs/mace/requirements.txt       # second environment, MACE control only (CPU torch index)
```

and run the commands below with `python …` in the matching environment instead of `uv run python …`.

---

## Two routes

**Route A, from scratch.** Fetch the structures, extract the embeddings, run everything. This is how the
paper's numbers were produced and it reproduces all of them.

**Route B, from the published embeddings.** Skip fetching and extraction. One command turns the two dataset
repositories into the inputs the later stages read: the structure index with every target, the pooled ORB and
UMA vectors at every layer, the published train/test splits and the Magpie features. It takes about five minutes
once the datasets are downloaded (it downloads them itself when given their Hub ids):

```bash
uv run python stages/data/from_dataset.py --orb NatalijaStepurko/orb-mp-embeddings \
    --uma NatalijaStepurko/uma-mp-embeddings-gated
```

Then run the stages from section 4 on. Everything that works on pooled embeddings runs unchanged: the probes,
significance and repeated splits, the controls in `baselines.py` except `--same-material`, the probe-family
control, screening, label efficiency and the stitching (`convergence.py --pooled-only --functional
--stitch-grid`). The released corpus omits four materials that the paper's probes saw, so a rerun probe score
differs from the published one by up to a few thousandths, inside `check_numbers.py`'s tolerance. Stages that
need per-atom activations or re-embed structures still need route A: geometry and health, the same-material,
rotation, random-weight and matched-model controls, the per-atom convergence grids, the external transfer and
the from-scratch network. `label_efficiency.py --from-dataset DIR` also reads the open dataset directly.

A **smoke test** of route A on a small corpus is `fetch_mp.py --limit 2000`, then the same commands as
below with smaller sample and bootstrap settings. It takes about 20 hours on an 8-core CPU machine: about
5 of them are the two health checks, and about 3 are the OQMD download. The scores will not match the paper
(different data), so `check_numbers.py` reports FAIL, but every stage runs.

---

## Stages

Folders are in run order; inside a folder the scripts are listed in the order they run. Every script is
idempotent and resume-safe: outputs are guarded by existence checks, so a stopped run continues where it
left off, and re-running a finished stage does nothing.

### 1 `stages/data/` — structures and labels

| script | what it does | writes |
|---|---|---|
| `fetch_mp.py` | Materials Project structures and the nine targets, through the API; writes CIFs and `index.jsonl`, excludes the four ids in `config/excluded_materials.csv`, validates the download. `--limit N` for a smoke test. | `structures/` |
| `from_dataset.py` | Route B: rebuilds `structures/index.jsonl`, the pooled ORB and UMA caches, the published splits and the Magpie features from the two dataset repositories (local copies or Hub ids); `--cifs` also writes the structures. | `structures/`, `results/` |
| `fetch_external.py` | JARVIS-DFT 3D (`--source jarvis-tools`) and OQMD (`--source oqmd`) in the same manifest format, each entry matched to MP (cited id or structure match); JARVIS's bulk modulus is the Voigt–Reuss–Hill average recomputed from the elastic tensor. | `structures_external/<db>/` |

### 2 `stages/extract/` — one frozen forward pass per model

| script | what it does | writes |
|---|---|---|
| `extract_orb.py` | ORB-v3: node and edge activations at every layer (encoder, gnn_0–4, backbone), per structure. `--random-init` for the random-weight control, `--orb-variant mpa` for the matched-architecture control. | `results/orb/` |
| `extract_uma.py` | UMA-S: the [9, 128] equivariant node and edge tensors at every layer (embed, block_0–3, backbone). `--random-init` as above. | `results/fairchem/` |
| `extract_mace.py` | MACE-OMAT-0 pooled read-outs on the matched-data subsample (second environment). | `results/mace_omat/` |

### 3 `stages/geometry/` — unsupervised organisation and embedding health

| script | what it does | writes |
|---|---|---|
| `organisation_orb.py`, `organisation_uma.py` | PCA, UMAP, chance-corrected k-NN purity and 1−LVR per layer and read-out (UMA: per channel view); also builds the pooled per-material caches every later stage reads. **Run before `health_qc_*`.** | `results/<model>/plots/` |
| `health_qc_orb.py`, `health_qc_uma.py` | PCA-90, cosine, condition number, adjacent-layer CKA / SVCCA / k-NN Jaccard, HDBSCAN with ARI / NMI against every target. | `results/<model>/plots/qc/` |

### 4 `stages/probes/` — the supervised probes and their statistics

| script | what it does | writes |
|---|---|---|
| `probes_orb.py`, `probes_uma.py` | XGBoost and linear probes over every layer × read-out (× channel view) on the fixed seed-42 split; `metrics.csv` is the table every later stage reads. | `results/<model>/property_prediction/` |
| `significance.py` | paired bootstrap (B = 2000) for UMA vs ORB, XGBoost vs linear, node vs edge, BH-FDR across the twelve targets of the sweep; the full metric table with CIs. | `results/proppred_significance/` |
| `repeated_split.py` | Nadeau–Bengio repeated-split test for the layer-depth claims (five retrains). | `results/proppred_repeated_significance/` |

### 5 `stages/controls/` — does the result survive?

| script | what it does | writes |
|---|---|---|
| `baselines.py` | composition-only Magpie baseline, validation-split selection, grouped splits, permuted-label selectivity, k-NN predictor, same-material and whitening controls on purity, log bulk modulus, band-gap subset, invariant crystal-system CI; one flag each. | `results/proppred_controls/<name>/` |
| `probe_family.py` | linear, XGBoost, k-NN and MLP on the same rows at the selected configurations; the linear probe's peak layer against XGBoost's. | `results/proppred_controls/{probe_family,mlp_probe,probe_peaks}/` |
| `rotation.py` | crystal-system probes under random rigid rotations, for the pretrained models and for the random-weight UMA-S network; the feature-level rotation test of every read-out. | `results/proppred_controls/{rotation,rotation_random_uma,rotation_invariance}/` |
| `random_init.py` | pretrained against the same architecture with random weights, after extracting and probing the random networks (commands below). | `results/random_init_control/` |
| `matched_models.py` | ORB-v3 omat, ORB-v3 mpa, MACE-OMAT-0 and UMA-S on one 20,000-material subsample: matched data and matched architecture. | `results/matched_data/` |

### 6 `stages/downstream/` — screening and label efficiency

| script | what it does | writes |
|---|---|---|
| `screening_enrichment.py` | recall of the true top tails, hull precision, band-gap window and per-class AUROC of the selected probes against the composition baseline. | `results/screening_enrichment/` |
| `label_efficiency.py` | learning curves of the embedding and Magpie probes on frozen test sets, the paired gains, the stable / saturated sizes, and the from-scratch message-passing network (`mlipprobes/gnn.py`). Subcommands: `curves`, `summarise`, `gain`, `split-fraction`, `gnn`. | `results/label_efficiency/` |

### 7 `stages/convergence/` — do the two models agree?

| script | what it does | writes |
|---|---|---|
| `convergence.py` | CKA, SVCCA, mutual k-NN and linear predictivity between every ORB × UMA layer pair, on atom-aligned embeddings, and functional stitching on the pooled ones. `--pooled-only` runs the stitching alone, without per-atom activations (route B). | `results/convergence_orb_uma/` |
| `convergence_significance.py` | the peak values under atom resampling and a permutation baseline. | `results/convergence_orb_uma/significance/` |

### 8 `stages/transfer/` — the MP-trained probes on other databases

| script | what it does | writes |
|---|---|---|
| `transfer.py` | the selected probes applied without refitting to JARVIS-DFT 3D and OQMD, against the same configuration refit inside each database on the same test rows (five 80/20 splits). | `results/external_transfer/<db>/transfer.csv` |
| `label_shift.py` | on shared crystals, the four-way scoring that separates the label part from the geometry part. | `results/external_transfer/label_shift_decomposition.csv` |
| `matching_audit.py` | leakage and coordinate checks on the external sets. | `results/external_transfer/matching_audit.json` |

### 9 `stages/check/` — compare with the paper

`check_numbers.py` reads the published result tables from the dataset (`--dataset DIR` or `--hf-repo ID`)
and prints, for the probe scores and the bootstrap point estimates, how many rows matched and the largest
difference from your run.

---

## Run order (route A)

`reproduce.sh` runs everything below in this order (`N_JOBS` sets the core count); the commands are listed so that any stage can be run on its own.

```bash
# 1  data
uv run python stages/data/fetch_mp.py

# 2-4  ORB-v3, then UMA-S: extraction, geometry (organisation before health), probes
uv run python stages/extract/extract_orb.py --num-workers 8 --torch-threads 1
uv run python stages/geometry/organisation_orb.py --num-workers 8
uv run python stages/geometry/health_qc_orb.py --num-workers 8
uv run python stages/probes/probes_orb.py --num-workers 8
uv run python stages/extract/extract_uma.py --num-workers 8 --torch-threads 1
uv run python stages/geometry/organisation_uma.py --num-workers 8
uv run python stages/geometry/health_qc_uma.py --num-workers 8
uv run python stages/probes/probes_uma.py --n-jobs 8

# 4  statistics
uv run python stages/probes/significance.py
uv run python stages/probes/repeated_split.py

# 5  controls
uv run python stages/controls/baselines.py --composition-baseline --val-selection --grouped-split \
    --probe-selectivity --knn-predictor --same-material --whitening --log-bulk-modulus --band-gap-subset --invariant-crystal
uv run python stages/controls/probe_family.py --probe-family --mlp-probe --probe-peaks
uv run python stages/controls/rotation.py --rotation --rotation-invariance --rotinv-sub 300 --rotinv-n 5
#    random weights: extract and probe both architectures again, then compare
uv run python stages/extract/extract_orb.py --results-dir $MLIP_PROBES_ROOT/results/orb_random_init --random-init --num-workers 8 --torch-threads 1
uv run python stages/extract/extract_uma.py --results-dir $MLIP_PROBES_ROOT/results/fairchem_random_init --random-init --num-workers 8 --torch-threads 1
uv run python stages/probes/probes_orb.py --embeddings-dir $MLIP_PROBES_ROOT/results/orb_random_init --output-dir $MLIP_PROBES_ROOT/results/orb_random_init/property_prediction --num-workers 8
uv run python stages/geometry/organisation_uma.py --embeddings-dir $MLIP_PROBES_ROOT/results/fairchem_random_init --plots-dir $MLIP_PROBES_ROOT/results/fairchem_random_init/plots --num-workers 8
uv run python stages/probes/probes_uma.py --cache-root $MLIP_PROBES_ROOT/results/fairchem_random_init/plots --output-dir $MLIP_PROBES_ROOT/results/fairchem_random_init/property_prediction --n-jobs 8
uv run python stages/controls/random_init.py
uv run python stages/controls/rotation.py --rotation-random-uma        # needs the random-weight UMA probes above
#    matched models: the subsample, the two extra extractions, the comparison
uv run python stages/controls/matched_models.py --make-subsample
uv run python stages/extract/extract_orb.py --orb-variant mpa --ids-file $MLIP_PROBES_ROOT/results/matched_data/subsample_ids.txt --results-dir $MLIP_PROBES_ROOT/results/orb_mpa --num-workers 8 --torch-threads 1
uv run --project envs/mace python stages/extract/extract_mace.py --ids-file $MLIP_PROBES_ROOT/results/matched_data/subsample_ids.txt --results-dir $MLIP_PROBES_ROOT/results/mace_omat --num-workers 4 --torch-threads 2
uv run python stages/controls/matched_models.py

# 6  screening and label efficiency
uv run python stages/downstream/screening_enrichment.py --n-jobs 8
for arm in orb uma magpie; do for axis in train test; do
  uv run python stages/downstream/label_efficiency.py curves --arm $arm --axis $axis --n-jobs 8
done; done
uv run python stages/downstream/label_efficiency.py summarise
uv run python stages/downstream/label_efficiency.py gain
uv run python stages/downstream/label_efficiency.py split-fraction
uv run python stages/downstream/label_efficiency.py gnn --n-jobs 8

# 7  cross-model convergence
uv run python stages/convergence/convergence.py
uv run python stages/convergence/convergence.py --uma-feature ang_norm --functional --residualise
uv run python stages/convergence/convergence.py --stitch-grid --skip-grids
uv run python stages/convergence/convergence_significance.py

# 8  cross-database transfer: the external sets, their embeddings, the transfer test and the label/geometry parts
uv run python stages/data/fetch_external.py --source jarvis-tools
uv run python stages/data/fetch_external.py --source oqmd --oqmd-n 40000
for db in jarvis_dft_3d oqmd; do
  uv run python stages/extract/extract_orb.py --structures-dir $MLIP_PROBES_ROOT/structures_external/$db --results-dir $MLIP_PROBES_ROOT/results/external/$db/orb --num-workers 8 --torch-threads 1
  uv run python stages/extract/extract_uma.py --structures-dir $MLIP_PROBES_ROOT/structures_external/$db --results-dir $MLIP_PROBES_ROOT/results/external/$db/fairchem --num-workers 8 --torch-threads 1
  uv run python stages/transfer/transfer.py --ext-name $db --within-repeats 5 --n-jobs 8
done
uv run python stages/transfer/label_shift.py --n-jobs 8
uv run python stages/transfer/matching_audit.py --n 1000

# 9  compare with the paper
uv run python stages/check/check_numbers.py --hf-repo NatalijaStepurko/orb-mp-embeddings
```

### Cost

Wall-clock on the 8-core CPU machine the paper used.

| stage | time |
|---|---|
| extraction, ORB-v3 / UMA-S | ≈ 7 h / 13 h (0.16 / 0.30 s per structure) |
| geometry and health, both models | ≈ 4–6 h |
| probes, all configurations | ≈ 3 h |
| significance and repeated splits | ≈ 2 h |
| controls (`baselines.py`, all flags) | ≈ 5 h, dominated by validation-split selection |
| probe family and rotation | ≈ 2.5 h |
| random-weight control (two more extractions and probe sweeps) | ≈ 33 h |
| matched models | ≈ 3 h |
| screening, label efficiency, from-scratch network | ≈ 10 h |
| cross-model convergence | ≈ 15 min |
| external sets: extraction and transfer | ≈ 25 h |

---

## Layout

```
mlipprobes/        shared code: paths.py (the root), common.py (targets, labels, purity, LVR, CKA, SVCCA),
                   significance_lib.py (the significance test and every control), gnn.py (the from-scratch network)
stages/            the entry points, grouped as above; each has a docstring with its inputs, outputs and flags
config/            excluded_materials.csv, label_efficiency_thresholds.csv (written by label_efficiency.py summarise)
envs/mace/         the second environment
reproduce.sh       the whole run order as one script
pyproject.toml, uv.lock, .python-version
```

## Questions and problems

Open an issue on this repository; questions are answered on a best-effort basis.

## Licences

Code: MIT (`LICENSE`). The models are third-party and downloaded at run time under their own terms:
ORB-v3 (Apache-2.0), UMA-S (FAIR Chemistry License, gated), MACE-OMAT-0 (Academic Software License).
Materials Project data is CC-BY 4.0; JARVIS-DFT is a NIST work not subject to copyright in the United
States; OQMD is CC-BY 4.0.
