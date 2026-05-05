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
| `src/predict_kleb_by_bacformer/pp/prepare_esmc_embeddings_and_labels_to_finetune_amr.py` | Merge AST labels + embeddings → per-sample `.pt` with 70/10/20 split |
| `src/predict_kleb_by_bacformer/tl/train_amr.py` | Fine-tune Bacformer for one antibiotic (Slurm: `train_on_slurm_amr.sh`) |
| `src/predict_kleb_by_bacformer/pp/stratified_isolation_source_sampling.py` | Select balanced cohort for an isolation-source pair |
| `src/predict_kleb_by_bacformer/pp/prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py` | Merge isolation-source labels + embeddings → per-sample `.pt` |
| `src/predict_kleb_by_bacformer/tl/train_isolation_source.py` | Fine-tune Bacformer for an isolation-source pair (Slurm: `train_isolation_source.sh`) |
| `src/predict_kleb_by_bacformer/pp/generate_bacformer_embeddings.py` | Generate ESM embeddings from protein sequences |
| `src/predict_kleb_by_bacformer/pp/preprocess_assemblies_to_protein_sequences.py` | GFF/assembly → `.faa` protein sequences |

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

## Slurm scripts

| Script | Purpose |
|---|---|
| `train_on_slurm_amr.sh` | GPU job: fine-tune for one antibiotic |
| `train_isolation_source.sh` | GPU job: fine-tune for isolation-source pair |
| `prepare_iso_source_data_for_training.sh` | CPU job: prepare `.pt` files for isolation-source |
| `run_bacformer_embeddings.sh` / `_array.sh` | Generate ESM embeddings |
| `preprocess_protein_sequences.sh` | GFF → protein sequences |
| `cpu_slurm.sh` | Generic CPU job template |

**Note:** isolation-source pair tokens are edited directly in the `.sh` file, not passed as CLI args.

## Next steps (active work)

1. **Test with HPC datasets** — rsync code, run tests against real data paths, verify all scripts resolve correct locations
2. **Complete genome predictions** — assess how many complete genomes have embeddings; run train/eval pipeline under `train_on_complete_genomes/`
3. **Klebsiella-specific retraining** — retrain Bacformer base model on Klebsiella before fine-tuning for predictions

## Code style

- Line length: 120 characters
- Docstrings: NumPy convention
- Ruff rules: B, BLE, C4, D, E, F, I, RUF100, TID, UP, W
- Python 3.10–3.14
