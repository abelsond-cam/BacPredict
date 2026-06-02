# CLAUDE.md

Guidance for Claude Code working in this repository. Per-task detail lives in each task folder's own `CLAUDE.md`.

> **Active program plan (week of 2026-05-30):** see [`~/.claude/PROGRAM_PLAN_2026-05-30.md`](../../.claude/PROGRAM_PLAN_2026-05-30.md). Workstreams **A** (iso-source expansion + stratification + explainability), **B** (complete-genome eval-set surgery, touches `src/tl/`), and **E** (Bacformer SNP-representation probe) live in this repo. Per-task detail is appended to each sub-CLAUDE.md.

## Project purpose

Fine-tune [Bacformer](https://github.com/amina-BS/bacformer) on bacterial genome embeddings to predict downstream phenotypes. The repo hosts **six parallel experiments**, each in its own task folder with its own `CLAUDE.md` and SLURM scripts:

| Task | Folder | Status |
|---|---|---|
| 1. AST in *M. tuberculosis* | [src/tb_ast/](src/tb_ast/) | Active |
| 2. AST in *Klebsiella pneumoniae* | [src/kleb_ast/](src/kleb_ast/) | Active |
| 3. Isolation source in *Klebsiella* | [src/kleb_iso_source/](src/kleb_iso_source/) | Active |
| 6. `predictHGT` embedding diagnostic | [src/predict_hgt/](src/predict_hgt/) | Diagnostic, can run in parallel |

Task 4 (mixed-assembly detection) and Task 5 (DefensePredictor on short reads) are deferred — condensed plans live in [ToDo.md](ToDo.md). Recreate as `src/admixture/` and `src/dp_short_read/` when work actually starts.

This root file holds only global conventions and shared-infra docs. **Per-task plans, status, and running notes live in each task folder's `CLAUDE.md`.** The cross-task live tracker (current state + remaining milestones per task) is [ToDo.md](ToDo.md). Three to four agents run concurrently, one per active task (see §0.5).

## §0 — Global conventions (apply to every task)

### §0.1 Base model

- All experiments start from the **Bacformer complete-genomes model** (not the MAG-trained model).
- **Refresh the Bacformer weights from Hugging Face first.** Previous local weights had defects since fixed by the authors. Until that refresh happens no benchmark we publish is comparable to current SOTA.
- Earlier runs used the older MAG-trained model — every one of those benchmarks needs re-running once the refreshed complete-genomes model is in place.

### §0.2 Three-stage testing protocol

Every experiment goes through these stages in order. **Do not skip ahead.**

| Stage | Purpose | Scale | Folds × seeds | Where |
| :-- | :-- | :-- | :-- | :-- |
| **A. Smoke** | Pipeline runs end-to-end | n=10 | 1 × 1 | MacBook M1 CPU (or HPC login). Code MUST run with CUDA disabled. |
| **B. Overfit** | Loss → ~0 on a tiny set | n=10, train=test | 1 × 1 | Local or HPC interactive |
| **C. Full** | Headline result | full data, one canonical drug/task first | 1 × 1 | GPU HPC SLURM, ~36 h budget, early-stopping ≈ 15 epochs |

Folds × seeds (≥5 each) are an **advanced final step**, only for external publication. Do not burn compute on them during exploration.

### §0.3 Paths

- HPC root: `/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/`
- Raw data: `project_k/david/raw/<task>/`
- Processed data: `project_k/david/processed/<task>/`
- Local dev: `/Users/davidabelson/developer/BacPredict/`

Per-task data paths live in each task's `CLAUDE.md`.

**Metadata source.** Klebsiella cohort labels (host / isolation source / AMR / study type /
sequencing provenance) come from `metadata_v2_all_samples_and_columns.tsv` produced by the
sibling BacHGT repo. Authoritative description — cohort definition, row keying, every flag, every
column's source —
[`~/developer/BacHGT/src/bac_metadata/METADATA_v2_README.md`](../BacHGT/src/bac_metadata/METADATA_v2_README.md).
The Bacformer-derived `predicted_{antibiotic}_AST` + `EBI_{antibiotic}_AST` columns produced by
this repo are intended to be merged back into v2 (see the README's §12).

### §0.4 What we report at each milestone

For every full run: **AUROC, AUPRC, sensitivity, specificity, balanced accuracy, confusion matrix, calibration curve, per-class breakdown.** Save model checkpoint + a versioned results JSON for diffing.

**For AMR tasks (Tasks 1 and 2), every report MUST additionally be stratified by resistance mechanism — HGT/acquired vs chromosomal point mutation.** Central hypothesis of the programme. Mechanism labels: WHO V2 catalogue (TB), AMRFinderPlus + Kleborate (Kp).

### §0.5 Concurrent agents — one per task

Three to four Claude Code agents typically run in parallel on this repo, one per active task (Task 1 = [src/tb_ast/](src/tb_ast/), Task 2 = [src/kleb_ast/](src/kleb_ast/), Task 3 = [src/kleb_iso_source/](src/kleb_iso_source/), optionally Task 6 = [src/predict_hgt/](src/predict_hgt/)). To avoid stepping on each other:

- **Stay in your task folder.** Edits should be confined to your own `src/<task>/` package and its tests under `tests/<task>/`. Touch shared code ([src/tl/](src/tl/), top-level configs, root docs) only when truly necessary, and call it out explicitly so the user can coordinate.
- **Pull before you start, push as soon as a unit of work is done.** Don't sit on uncommitted changes — another agent on a parallel branch may be about to push something that touches adjacent files. Frequent small commits + pushes keep merge conflicts narrow and local.
- **Scope `git add` to your task folder.** Prefer `git add src/<your_task>/ tests/<your_task>/` over `git add -A`/`git add .`. Never stage another task's in-progress edits; if you see unrelated modified files from a sibling agent, leave them untouched.
- **Pull is also scoped — only when you need it.** Run `git pull --rebase` on your own branch before pushing; do not rebase or merge other agents' branches into yours without being asked.
- **One branch per agent.** Each task agent should work on its own branch (e.g. `task1/...`, `task2/...`); do not commit to `main` or to another task's branch.

## HPC connection

```
Host:      login.hpc.cam.ac.uk
User:      dca36
SSH:       ControlMaster auto, ControlPersist 8h (~/.ssh/sockets/)
Workspace: /home/dca36/workspace/BacPredict
Python:    always uv run python (never python or python3 directly)
```

Run commands on HPC: `ssh dca36@login.hpc.cam.ac.uk "<command>"`.

**Login node usage.** Jobs that complete in under **15 minutes** and stay under **128 MB RAM** (purely CPU — no GPU) can run directly on the login node without SLURM. Anything heavier (model inference, embedding generation, training, large-data parses) belongs in a SLURM job. Worked example: regenerating combined figures / summary CSVs from already-saved per-drug `eval_scores.npz` (see [src/kleb_ast/scripts/regen_panel_summary.sh](src/kleb_ast/scripts/regen_panel_summary.sh)) — pure matplotlib + small npz reads, finishes in seconds, not worth a queue wait.

## Package layout

`src/` is flat — nine top-level packages, no `bacpredict/` wrapper. Distribution name stays `bacpredict`.

**Shared toolbox** ([src/tl/](src/tl/), umbrella package — drop any generic helper here):

| Package | What it does |
|---|---|
| [src/tl/embed/](src/tl/embed/) | Protein sequences → ESM-C → Bacformer embeddings |
| [src/tl/genome_download/](src/tl/genome_download/) | Acquire and catalogue raw genomic data (BakRep, NCBI Datasets) |
| [src/tl/train/](src/tl/train/) | `split_utils.py` (70/10/20 + k-fold + fixed evaluate holdout) and `datasets.py` (`LabelInjectingFileDataset` — lazy load + runtime label injection) |

**Task packages** (one per experiment; each owns its own scripts, prep code, train entrypoint, and `CLAUDE.md`): [tb_ast](src/tb_ast/), [kleb_ast](src/kleb_ast/), [kleb_iso_source](src/kleb_iso_source/), [predict_hgt](src/predict_hgt/).

Task packages import shared helpers as `from tl.train.split_utils import ...`, `from tl.train.datasets import ...`, `from tl.embed.* import ...`. They do not import each other.

## Commands

```bash
# Install (editable) — on HPC use uv run python, locally use uv run
uv pip install -e .

# Run a script (path is now src/<package>/<module>.py — no bacpredict middle layer)
uv run python src/kleb_iso_source/stratified_isolation_source_sampling.py --help

# Tests
pytest tests/

# Lint
ruff check src/
```

## Training data architecture

Embedding files are large (~1 TB total across 84k samples) and are **never duplicated per experiment**. One canonical embedding store + small CSVs recording split assignments and labels.

**1. Embedding store (read-only, shared across experiments):**
```
processed/klebsiella_esm_embeddings/{sample_accession}_esm_embeddings.pt
```
Each `.pt` holds `prot_embeddings` (shape `[n_proteins, dim]`), `attention_mask`, contig indices. **No labels.**

**2. Split CSV (canonical record of who-went-where):** each prepare script writes one CSV per experiment to RDS — these are **permanent**.

| Experiment | CSV path |
|---|---|
| AMR | `processed/binary_ast_with_split.csv` |
| Isolation-source pair | `processed/<experiment_dir>/binary_<pair_slug>_with_split.csv` |

Columns: `Sample` (joins to `{Sample}_esm_embeddings.pt`), `<label_column>` (binary; NaN allowed), `train_val_eval` (one of `train` / `validate` / `evaluate`).

By default the prepare script writes **only** this CSV. Legacy per-sample labeled `.pt` files are gated behind `--write-pt-files`.

**3. Training (lazy, runtime label injection):** `LabelInjectingFileDataset` (`tl/train/datasets.py`) takes the sample-ID list for one split, a `label_map: dict[sample_id → int]` built from the CSV in memory, and the path to the shared embedding store. `__getitem__` opens one `.pt` at a time and attaches the label. **No labeled `.pt` copies are created.**

**4. Reproducing a result:** `(input split CSV) + (training script CLI args)`. CSV pins labels + single-split assignment; CLI args pin checkpoint, LR, and (for k-fold) `n_folds`/`fold`/`seed`/`evaluate_seed`. Keep the CSV alongside the checkpoints under the experiment directory.

## K-fold CV and split semantics

Single-split mode (default) reads `train_val_eval` from the CSV directly.

K-fold mode (`--n-folds N`) ignores the CSV's `train_val_eval` column and calls `generate_kfold_splits(df, n_folds=N, seed=SEED, evaluate_seed=EVALUATE_SEED)` at training time:

1. **Fixed evaluate holdout** (default 20% of unique Sample IDs) controlled solely by `evaluate_seed` — identical for every `(fold, seed)` combination of an experiment.
2. Shuffle the remaining 80% with `seed`, split into `N` folds (`numpy.array_split`). Fold *i* uses fold *i* as validation, union of others as training.

K-fold splits are **not written to disk** — derived deterministically from `(unique sample IDs, n_folds, seed, evaluate_seed)`. To reproduce, replay those four inputs against the same input CSV. Output checkpoint dirs auto-suffixed `_fold{NN}_seed{S}`.

The Slurm array `--array=0-14` runs 5 folds × 3 seeds = 15 jobs: `FOLD = SLURM_ARRAY_TASK_ID % 5`, `SEED = SLURM_ARRAY_TASK_ID / 5 + 1`.

## Data-leakage guarantees

The split logic is designed so that **no Sample ID can appear in more than one split within a single training run**, and the evaluate holdout is preserved across the entire k-fold sweep.

- **Single-split.** `add_splits()` shuffles unique `Sample` values, partitions into train 70 / validate 10 / evaluate 20. Tested in [tests/tl/train/test_split_utils.py::test_add_splits_no_overlap](tests/tl/train/test_split_utils.py).
- **K-fold.** Evaluate selected first and removed from the pool; remaining samples partitioned into mutually disjoint validation folds. For any `(fold, seed)`: `evaluate ∩ train = ∅`, `evaluate ∩ validate = ∅`, `train ∩ validate = ∅`. Across folds, **train sets share samples** (intrinsic to k-fold; only validate rotates). Evaluate is identical across every `(fold, seed)` when `evaluate_seed` is held constant. Tested in [tests/tl/train/test_split_utils.py](tests/tl/train/test_split_utils.py) and [tests/kleb_ast/test_pt_training_pipeline.py](tests/kleb_ast/test_pt_training_pipeline.py).

**Caveats — what these guarantees do NOT cover:**

- **Duplicate isolates under different accessions.** Split is over unique `Sample` values. If one biological isolate appears under multiple accessions, those copies are split independently and could land in both train and evaluate. Deduplication is an upstream metadata problem.
- **Bacformer pre-training overlap.** Bacformer was pre-trained on MAGs / complete genomes. Samples in our evaluate set may have been in that corpus — representation-level leakage not addressable by sample splitting in this repo.
- **Changing `--evaluate-seed` mid-experiment.** Evaluate holdout is only stable while `evaluate_seed` is held constant. Pin once per experiment.
- **Pre-existing `train_val_eval` column when `--n-folds` is set.** Ignored in k-fold mode — a sample previously labelled `evaluate` may end up in `train` for some fold. By design; k-fold owns its own splitting.

## Code style

- Line length: 120 characters
- Docstrings: NumPy convention
- Ruff rules: B, BLE, C4, D, E, F, I, RUF100, TID, UP, W
- Python 3.10–3.14
