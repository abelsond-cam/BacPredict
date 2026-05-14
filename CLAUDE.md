# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Fine-tune [Bacformer](https://github.com/amina-BS/bacformer) on *Klebsiella pneumoniae* subsp. *pneumoniae* (KPSC) to predict two phenotypes from per-genome ESM embeddings:
1. **AMR** — binary resistance per antibiotic (EBI AST records)
2. **Isolation source** — binary pair classification (e.g. blood vs respiratory) as a proxy for infection niche

Embeddings are generated from protein sequences derived from genome assemblies. Both tasks share the same `{sample_accession}_esm_embeddings.pt` files as input.

## HPC connection

```
Host:      login.hpc.cam.ac.uk
User:      dca36
SSH:       ControlMaster auto, ControlPersist 8h (~/.ssh/sockets/)
Workspace: /home/dca36/workspace/predict_kleb_by_bacformer
Python:    always uv run python (never python or python3 directly)
```

Run commands on HPC:
```bash
ssh dca36@login.hpc.cam.ac.uk "<command>"
```

## Key data paths (all on RDS)

All data lives under:
```
/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/
```

| What | Path |
|---|---|
| Metadata (full) | `final/metadata_final_curated_all_samples_and_columns.tsv` (233 MB) |
| Metadata (slimmed) | `final/metadata_final_curated_slimmed.tsv` (50 MB) |
| ESM embeddings | `processed/klebsiella_esm_embeddings/` (84 139 `.pt` files) |
| Protein sequences | `processed/klebsiella_protein_sequences/` |
| Completed SR MAG experiments | `processed/train_on_sr_mags/` |
| Completed genome experiments | `processed/train_on_complete_genomes/` (currently empty — next task) |
| Complete vs SR analysis | `processed/complete_vs_sr_genomes/` |
| AMR preprocessed CSVs | `processed/binary_ast.csv`, `processed/binary_ast_with_split.csv` |

Paths are hardcoded in each script (no central data_paths module). Check the top of any script before running.

## Package layout

| Module | Purpose |
|---|---|
| `src/predict_kleb_by_bacformer/pp/preprocess_ebi_amr_records.py` | Parse EBI AST CSV → binary resistance labels per sample |
| `src/predict_kleb_by_bacformer/pp/prepare_esmc_embeddings_and_labels_to_finetune_amr.py` | Merge AST labels + embeddings → split CSV (default); `--write-pt-files` for legacy per-sample `.pt` |
| `src/predict_kleb_by_bacformer/pp/prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py` | Merge isolation-source labels + embeddings → split CSV (default); `--write-pt-files` for legacy `.pt` |
| `src/predict_kleb_by_bacformer/pp/split_utils.py` | Shared `add_splits()` (single 70/10/20) and `generate_kfold_splits()` (k-fold + fixed evaluate holdout) |
| `src/predict_kleb_by_bacformer/tl/datasets.py` | `LabelInjectingFileDataset` — lazy load of `{sample}_esm_embeddings.pt`, label injected at runtime from in-memory dict |
| `src/predict_kleb_by_bacformer/tl/train_amr.py` | Fine-tune Bacformer for one antibiotic; supports `--n-folds`/`--fold`/`--seed` for k-fold CV |
| `src/predict_kleb_by_bacformer/tl/train_isolation_source.py` | Fine-tune Bacformer for an isolation-source pair; same k-fold args |
| `src/predict_kleb_by_bacformer/pp/stratified_isolation_source_sampling.py` | Select balanced cohort for an isolation-source pair |
| `src/predict_kleb_by_bacformer/pp/generate_bacformer_embeddings.py` | Generate ESM embeddings from protein sequences |
| `src/predict_kleb_by_bacformer/pp/preprocess_assemblies_to_protein_sequences.py` | GFF/assembly → `.faa` protein sequences |
| `src/predict_kleb_by_bacformer/pp/genome_assemblies_from_bacformer_embeddings.py` | Identify which assemblies have/lack embeddings |
| `src/predict_kleb_by_bacformer/pp/filter_esmc_embeddings_by_klebsiella.py` | Filter embedding set to KPSC-only samples |
| `src/predict_kleb_by_bacformer/pp/extract_anndata_with_bacformer_protein_embeddings.py` | Extract AnnData from Bacformer embeddings for EDA |

## Commands

```bash
# Install (editable) — on HPC use uv run python, locally use uv run
uv pip install -e .

# Run a script
uv run python src/predict_kleb_by_bacformer/pp/stratified_isolation_source_sampling.py --help

# Tests
pytest tests/

# Lint
ruff check src/
```

## Experiment structure

Completed experiments live under `processed/train_on_sr_mags/`:
- `training_ast/`, `training_blood_faeces/`, `training_blood_respiratory/`, `training_blood_urine/`, `training_blood_wound/`, `training_faeces_respiratory/`, `training_urine_catheter/`, `training_urine_respiratory/`

Next experiments go under `processed/train_on_complete_genomes/` (currently empty).

## Training data architecture

Embedding files are large (~1 TB total across 84 k samples) and are **never duplicated per
experiment**. The pipeline keeps a single canonical embedding store and uses small CSVs to
record split assignments and labels.

**1. The embedding store (read-only, shared across experiments)**

```
processed/klebsiella_esm_embeddings/{sample_accession}_esm_embeddings.pt
```

Each `.pt` holds `prot_embeddings` (shape `[n_proteins, dim]`), `attention_mask`, and contig
indices. These files contain **no labels** — they are pure inputs.

**2. The split CSV (canonical record of who-went-where)**

Each prepare script writes one CSV per experiment to RDS — these are **permanent**, not
temporary. They are the system of record for both labels and split assignments.

| Experiment | CSV path |
|---|---|
| AMR | `processed/binary_ast_with_split.csv` |
| Isolation-source pair | `processed/<experiment_dir>/binary_<pair_slug>_with_split.csv` |

Columns:

| Column | Meaning |
|---|---|
| `Sample` | Sample accession — joins to `{Sample}_esm_embeddings.pt` |
| `<label_column>` | Binary label (e.g. `amikacin`, `blood_faeces_label`); may be NaN if missing |
| `train_val_eval` | One of `train` / `validate` / `evaluate` (single 70/10/20 split, `seed=1` by default) |

By default the prepare script writes **only** this CSV. The legacy per-sample labeled `.pt`
files are gated behind `--write-pt-files` and are no longer needed.

**3. Training (lazy, runtime label injection)**

`LabelInjectingFileDataset` (`tl/datasets.py`) takes:
- A list of sample IDs (filtered to one split)
- A `label_map: dict[sample_id → int]` (built from the CSV in memory — small, picklable, safe for DataLoader workers)
- The path to the shared embedding store

`__getitem__` opens one `.pt` file at a time and attaches the label from the dict. **No
labeled `.pt` copies are created.**

**4. Reproducing a result**

A run is fully described by `(input split CSV) + (training script CLI args)`. The CSV pins
the labels and the single-split assignment; the CLI args pin the model checkpoint, learning
rate, and (for k-fold) `n_folds`/`fold`/`seed`/`evaluate_seed`. Keep the CSV alongside the
model checkpoints under the experiment directory.

## K-fold CV and split semantics

Single-split mode (default) reads `train_val_eval` from the CSV directly.

K-fold mode (passing `--n-folds N`) ignores the CSV's `train_val_eval` column and instead
calls `generate_kfold_splits(df, n_folds=N, seed=SEED, evaluate_seed=EVALUATE_SEED)` at
training time. This:

1. Selects a **fixed evaluate holdout** (default 20 % of unique Sample IDs) controlled
   solely by `evaluate_seed` — identical for every `(fold, seed)` combination of an
   experiment.
2. Shuffles the remaining 80 % using `seed` and splits it into `N` folds with
   `numpy.array_split`. Fold *i* uses fold *i* as validation and the union of the others
   as training.

K-fold splits are **not written to disk** — they are derived deterministically from
`(unique sample IDs, n_folds, seed, evaluate_seed)`. To reproduce, replay those four
inputs against the same input CSV. To distinguish runs, each output checkpoint dir is
auto-suffixed with `_fold{NN}_seed{S}`.

The Slurm array `--array=0-14` runs 5 folds × 3 seeds = 15 jobs:
`FOLD = SLURM_ARRAY_TASK_ID % 5`, `SEED = SLURM_ARRAY_TASK_ID / 5 + 1`.

## Data-leakage guarantees

The split logic is designed so that **no Sample ID can appear in more than one split
within a single training run**, and the evaluate holdout is preserved across the entire
k-fold sweep.

- **Single-split mode.** `add_splits()` shuffles the unique values of the `Sample` column
  and partitions them into train (70 %) / validate (10 %) / evaluate (20 %). A sample is
  in exactly one split. Tested in `tests/test_split_utils.py::test_add_splits_no_overlap`.
- **K-fold mode.** Evaluate is selected first and removed from the pool. K-fold then
  partitions the remaining samples into mutually disjoint validation folds. For any
  given `(fold, seed)`:
  - `evaluate ∩ train = ∅` and `evaluate ∩ validate = ∅`
  - `train ∩ validate = ∅`
  - Across folds, **train sets share samples** (this is intrinsic to k-fold and is
    correct — only the validate set rotates). Different folds within the same run never
    share validation samples.
  - The evaluate set is identical across every `(fold, seed)` pair when `evaluate_seed`
    is held constant.
  Tested in `tests/test_split_utils.py::test_kfold_*` and
  `tests/test_pt_training_pipeline.py::test_kfold_*`.

**Caveats — what these guarantees do NOT cover:**

- **Duplicate isolates under different accessions.** The split is over unique values of
  the `Sample` column. If a single biological isolate appears under multiple sample
  accessions in the metadata, those copies will be split independently and could end up
  in both train and evaluate. Deduplication is an upstream metadata-curation problem.
- **Bacformer pre-training overlap.** Bacformer was pre-trained on MAGs / complete
  genomes. If samples in our evaluate set were in that pre-training corpus, the
  encoder has already seen them — this is representation-level leakage that cannot
  be addressed by sample splitting in this repo.
- **Changing `--evaluate-seed` mid-experiment.** The evaluate holdout is only stable as
  long as `evaluate_seed` is held constant. Pin it once per experiment; don't sweep it.
- **Pre-existing `train_val_eval` column when `--n-folds` is set.** The column is
  ignored in k-fold mode, so a sample previously labelled `evaluate` in the CSV may end
  up in `train` for some fold of a k-fold run. This is by design — k-fold owns its own
  splitting — but means the CSV's column does not constrain k-fold behaviour.

## Slurm scripts

| Script | Purpose |
|---|---|
| `train_on_slurm_amr.sh` | GPU array job (`--array=0-14`, 5 folds × 3 seeds): fine-tune for one antibiotic |
| `train_isolation_source.sh` | GPU array job (`--array=0-14`): fine-tune for isolation-source pair |
| `prepare_iso_source_data_for_training.sh` | CPU job: write the split CSV for an isolation-source pair |
| `run_bacformer_embeddings.sh` / `_array.sh` | Generate ESM embeddings |
| `preprocess_protein_sequences.sh` | GFF → protein sequences |
| `cpu_slurm.sh` | Generic CPU job template |

**Notes:**
- Isolation-source pair tokens are edited directly in the `.sh` file, not passed as CLI args.
- `FOLD` and `SEED` are computed from `SLURM_ARRAY_TASK_ID` inside the script. Comment out the `#SBATCH --array=...` line and remove the `--n-folds` arg to fall back to single-split mode using the CSV's `train_val_eval` column.

## Next steps (active work)

1. **Test with HPC datasets** — rsync the `dev` branch, run a short k-fold dry-run (`--n-samples 50 --max-steps 20`) on one experiment to verify paths and CSV-driven loading work end-to-end
2. **Merge `dev` → `main`** once the dry-run passes
3. **Complete genome predictions** — assess how many complete genomes have embeddings; run the 15-job array under `train_on_complete_genomes/`
4. **Klebsiella-specific retraining** — retrain Bacformer base model on Klebsiella before fine-tuning for predictions

## Code style

- Line length: 120 characters
- Docstrings: NumPy convention
- Ruff rules: B, BLE, C4, D, E, F, I, RUF100, TID, UP, W
- Python 3.10–3.14
