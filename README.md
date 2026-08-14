# BacPredict

Use Bacformer to predict Klebsiella behaviour in AST testing and niche (using isolation source as a proxy for infection vs colonisation at a body site).

> ## ⚠ The pipeline tables below describe a layout retired on 2026-07-11
>
> **Start at [`PROJECT_STATE.md`](PROJECT_STATE.md)** — the single authority on where this project
> actually is — and [`CLAUDE.md`](CLAUDE.md) for conventions and the current package map.
>
> Every `src/kleb_ast/…` and `src/tb_ast/…` path below is dead. Those packages were consolidated into
> one organism-agnostic engine plus thin per-organism apps: `src/bacpredict/engine/` and
> `src/bacpredict/apps/{tb,kleb}/`. The single AMR trainer is now
> `python -m bacpredict.engine.finetune.finetune_amr --task tb_ast|kleb_ast`.
>
> **Two specifics below are not merely renamed but wrong:**
>
> - **Labelled per-sample `.pt` files are not created.** `ast_training/{train,validate,evaluate}/{Sample}_with_ast.pt`
>   was replaced by lazy loading with runtime label injection — the embedding store is read-only and
>   never duplicated per experiment.
> - **`binary_ast_with_split.csv` is not the holdout authority.** Its `train_val_eval` column is a
>   single-split partition that k-fold training ignores. The holdout comes from the materialised
>   per-drug `<drug>_split.csv` via `engine.splits.load_splits`, and reading anything else is how the
>   2026-07 read-out leak happened.
>
> This file is kept as an orientation to the *shape* of the two tasks, which has not changed.

---

## Project flow (two prediction tasks)

The repo supports two fine-tuning tracks that share per-genome ESM embedding files (`{sample_accession}_esm_embeddings.pt`) but differ in label construction and training setup.

### Task 1 - Antibiotic resistance (AMR, multi-label per genome)

Goal: binary resistance labels per antibiotic (plus optional regression log MIC outputs during preprocessing).


| What runs                                                                                                                                   | Role                                                                                | Input                                                                                             | Output                                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Step 1:** `bacpredict.engine.ast_labels.parse_ebi_ast_to_binary` | Preprocess EBI AST records and pivot to sample-level tables                         | Raw EBI AMR CSV; min-antibiotic threshold                                                         | `binary_ast.csv`, metadata CSV, `regression_log_mic.csv`, antibiogram                                              |
| **Step 2:** `bacpredict.engine.splits` (writes `<drug>_split.csv`)                                        | Add train/validate/evaluate splits and merge AST labels into per-sample `.pt` files | `binary_ast.csv`, embedding files `{Sample}_esm_embeddings.pt`                                    | `<drug>_split.csv` (the holdout authority) + `binary_ast_with_split.csv` (labels only) |
| **Step 3:** `python -m bacpredict.engine.finetune.finetune_amr --task kleb_ast`                           | Fine-tune Bacformer for one selected antibiotic label                               | the read-only embedding store, `<drug>_split.csv`, `--drug <antibiotic>` | Fine-tuned checkpoints and training logs                                                                           |


---

### Task 2 - Isolation source (binary pair task, e.g. blood vs respiratory)

Goal: create a pair-specific binary isolation-source task and train on pair-specific split outputs.


| What runs                                                                                                         | Role                                                                                                                                    | Input                                                                                                                                                  | Output                                                                                                                                                                        |
| ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Step 1:** `src/kleb_iso_source/stratified_isolation_source_sampling.py`                            | Select and balance two isolation-source categories with country + study-thread stratification                                           | Metadata TSV, `--isolation-sources <token1> <token2>`, optional `--ratio`, optional `--filter-by-study-setting`                                        | `/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/training_<token1>_<token2>/stratified_selected_isolation_source_metadata.tsv`                               |
| **Step 2:** `src/kleb_iso_source/prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py` (cluster: `src/kleb_iso_source/scripts/prepare_iso_source_data_for_training.sh`) | Resolve the same token pair, create pair-specific binary labels, split 70/10/20, prune missing embeddings, and write merged `.pt` files | `--isolation-sources <token1> <token2>`, optional `--input-metadata-file`, optional `--embeddings-dir`                                                   | `training_<token1>_<token2>/binary_<pair_slug>.csv`, `training_<token1>_<token2>/binary_<pair_slug>_with_split.csv`, `training_<token1>_<token2>/{train,validate,evaluate}/...` |
| **Step 3:** `src/kleb_iso_source/train_isolation_source.py` (cluster: `src/kleb_iso_source/scripts/train_isolation_source.sh`)                                        | Train Bacformer from pair-specific split folders                                                                                        | `--isolation-sources <token1> <token2>` (same tokens as prepare); optional path overrides                                                              | Model checkpoints under the pair-specific `training_<slug1>_<slug2>/` directory (default output path)                                                                        |


---

## Placeholder - genomes to protein files to ESM embeddings

To be filled in next.

- Download or collect GFF files.
- Derive `.faa` protein FASTA files.
- Run ESM embedding generation so each retained sample has `{sample_accession}_esm_embeddings.pt`.

Until this section is completed, treat the embeddings directory as a prerequisite input to both prepare scripts.

---

## End-to-end example: blood vs respiratory

### Default workflow (auto-resolved directories)

**1. Stratified sample selection** (usually fine to run with `uv` from an interactive shell):

```bash
uv run python src/kleb_iso_source/stratified_isolation_source_sampling.py \
  --isolation-sources blood respiratory
```

**2. Prepare merged `.pt` files** (`prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py`). This step reads many embedding files and is **too slow for a typical login node** at scale (roughly five minutes per thousand genomes). On the cluster, use the CPU Slurm wrapper: edit `isolation_sources="blood respiratory"` in `src/kleb_iso_source/scripts/prepare_iso_source_data_for_training.sh` (same idea as `src/kleb_iso_source/scripts/cpu_slurm.sh`), then submit:

```bash
sbatch src/kleb_iso_source/scripts/prepare_iso_source_data_for_training.sh
```
NOTE:  You must edit this script to include the isolation sources you want to train on (not CLI)

**3. Fine-tune Bacformer** on a GPU node. Edit `isolation_sources` in `src/kleb_iso_source/scripts/train_isolation_source.sh` so it matches the pair you prepared, then:

```bash
sbatch src/kleb_iso_source/scripts/train_isolation_source.sh
```
NOTE:  You must edit this script to include the isolation sources you want to train on (not CLI)

---

Both the scripts infer and share:

- `/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/training_blood_respiratory/`
- `stratified_selected_isolation_source_metadata.tsv`
- `stratify_isolation_source_sampling.log`
- default embeddings dir: `/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_esm_embeddings`
- default ratio = 2:1

### Optional overrides

The data sources and directories that are used in both scripts can be updated. 
For example:

```bash
uv run python src/kleb_iso_source/stratified_isolation_source_sampling.py \
  --isolation-sources blood respiratory \
  --metadata-file /path/to/metadata.tsv \
  --output-file /path/to/training_blood_respiratory/stratified_selected_isolation_source_metadata.tsv \
  --ratio 4  # This makes ratio 4:1
```

```bash
uv run python src/kleb_iso_source/prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py \
  --isolation-sources blood respiratory \
  --input-metadata-file /path/to/training_blood_respiratory/stratified_selected_isolation_source_metadata.tsv \
  --embeddings-dir /path/to/klebsiella_esm_embeddings
```

For Slurm-based prepare, add the same extra flags to the `uv run python ...` line inside `src/kleb_iso_source/scripts/prepare_iso_source_data_for_training.sh` (or run that command inside an interactive CPU allocation instead of on the login node). For training overrides (paths, learning rate, and so on), edit `src/kleb_iso_source/scripts/train_isolation_source.sh` or call `train_isolation_source.py` with the same arguments from a GPU job.

---

## Related docs

More detail lives under `docs/`:

- `docs/amr_prediction.md`
- `docs/isolation_source_prediction.md`

