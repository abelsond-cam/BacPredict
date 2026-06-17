# CLAUDE.md

Guidance for Claude Code working in this repository. Per-task detail lives in each task folder's own `CLAUDE.md`.

> **Plans.** The living plan + tracker for this repo is [`ToDo.md`](ToDo.md) — current forward priorities and per-task state. (The earlier `~/.claude/PROGRAM_PLAN_2026-05-30.md` is superseded.)

## Project purpose

Fine-tune [Bacformer](https://github.com/amina-BS/bacformer) on bacterial genome embeddings to predict downstream phenotypes. The repo hosts **seven parallel experiments**, each in its own task folder with its own `CLAUDE.md` and SLURM scripts:

| Task | Folder | Status |
|---|---|---|
| 1. AST in *M. tuberculosis* | [src/tb_ast/](src/tb_ast/) | Active |
| 2. AST in *Klebsiella pneumoniae* | [src/kleb_ast/](src/kleb_ast/) | Active |
| 3. Isolation source in *Klebsiella* | [src/kleb_iso_source/](src/kleb_iso_source/) | Active |
| 6. `predictHGT` embedding diagnostic | [src/predict_hgt/](src/predict_hgt/) | Diagnostic, can run in parallel |
| 7. SNP-embedding signal-loss diagnostic | [src/snp_embeddings/](src/snp_embeddings/) | Active (diagnostic — why TB AST is poor) |

Task 4 (mixed-assembly detection) and Task 5 (DefensePredictor on short reads) are deferred — condensed plans live in [ToDo.md](ToDo.md). Recreate as `src/admixture/` and `src/dp_short_read/` when work actually starts.

This root file holds only global conventions and shared-infra docs. **Per-task plans, status, and running notes live in each task folder's `CLAUDE.md`.** The cross-task live tracker (current state + remaining milestones per task) is [ToDo.md](ToDo.md). Three to four agents run concurrently, one per active task (see §0.5).

## §0 — Global conventions (apply to every task)

### §0.1 Base model

- All experiments start from the **Bacformer complete-genomes model** (not the MAG-trained model).
- **Refresh the Bacformer weights from Hugging Face first.** Previous local weights had defects since fixed by the authors. Until that refresh happens no benchmark we publish is comparable to current SOTA.
- Earlier runs used the older MAG-trained model — every one of those benchmarks needs re-running once the refreshed complete-genomes model is in place.
- **Bacformer internals — reference of record:** the v2 manuscript *Bacformer_main_text_14062026* (Google Drive id `1yGnKCgJgY56rbDzqtFR8YLZ9bObZLfVa`; D. Abelson 2nd author; the version being submitted) is the source of truth for Bacformer's architecture, its **kNN ESM protein-family construction** (the template for clustering Bacformer protein embeddings into gene families), and its AMR gene-prioritisation. The internal-dev fork of the Bacformer code lives on HPC at `~/workspace/Bacformer-internal` (also a personal GitHub fork).

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

### §0.5a HPC resource requests — be generous, **never under-call**

CSD3 charges wall time used, not wall time requested; CPU cores and memory on
ampere/icelake nodes are abundant. **Dying mid-job because we under-requested
is the expensive failure mode.** Always lean toward more — *especially* time.

- **Time: triple any estimate.** If reasoning says 2 h, request 6 h (or 8 h if
  the estimate has any uncertainty at all — and it usually does). Over-requesting
  time costs *nothing*; running out of time kills hours of GPU compute. Never
  trim the time budget to "look efficient" — there's no prize for it.
- **CPUs / `--num-workers`: use what the node has.** Ampere allocates ~32 cores
  per GPU; request `--cpus-per-task=8` (the per-GPU partition default) and set
  the DataLoader `--num-workers` to match (8). CPU cores are idle if you don't
  use them; there is no penalty for "asking for more workers."
- **Memory: be generous.** Single-GPU ampere jobs effectively get ~250 GB
  regardless of what you request; defaulting to `--mem=128G` or higher is
  free and harmless. Don't trim memory requests below your *actual* peak.

Sources of truth (read these before tuning SLURM directives):
- A100 / ampere partition: <https://docs.hpc.cam.ac.uk/hpc/user-guide/a100.html>
- icelake CPU partition: <https://docs.hpc.cam.ac.uk/hpc/user-guide/icelake.html>

When in doubt: **more**. The cost of asking for more than you need is zero;
the cost of asking for less is the entire job.

### §0.5 Concurrent agents — one per task

> ⚠️ **READ THIS BEFORE ANY GIT COMMAND.** Three to four Claude Code agents edit this
> repository **at the same time**, sharing the local checkout *and* the single HPC checkout at
> `/home/dca36/workspace/BacPredict`. **Git state (branch, index, working tree, history) is
> shared and easily corrupted.** Treat every git operation as something that can destroy or
> entangle another agent's live, uncommitted work. When in doubt, **stop and ask the user** —
> a question is always cheaper than a recovery.

Active task → branch map: Task 1 = [src/tb_ast/](src/tb_ast/) (`task1/...`),
Task 2 = [src/kleb_ast/](src/kleb_ast/) (`task2/...`),
Task 3 = [src/kleb_iso_source/](src/kleb_iso_source/) (`task3/...`),
Task 5 = [src/dp_short_read/](src/dp_short_read/) (`task5/...`),
Task 6 = [src/predict_hgt/](src/predict_hgt/) (`task6/...`),
Task 7 = [src/snp_embeddings/](src/snp_embeddings/) (`snp-embeddings`).

**Stay in your task folder.** Confine edits to your own `src/<task>/` package and `tests/<task>/`.
Touch shared code ([src/tl/](src/tl/)), top-level configs, or root docs only when truly necessary,
and call it out explicitly so the user can coordinate.

**Commits & pushes — ALL of your work, and ONLY your work:**

- **Include everything *you* did.** A commit/push for a unit of work must capture *every* change
  this agent made for it — never leave your own edits stranded uncommitted across a hand-off.
- **Never include another agent's work without asking first — very clearly.** A commit/push must
  contain *only* changes this agent authored. If `git status` shows modified/staged files you did
  not create, **STOP**: do not `git add -A`, `git add .`, or `git commit -a`. Stage explicit paths
  inside your own task folder only (`git add src/<your_task>/ tests/<your_task>/`). If your changes
  are entangled with a sibling's uncommitted edits, **ask the user before committing anything** —
  describe exactly which files are whose and wait for an explicit go-ahead.
- **Never amend, squash, or rewrite a commit** that may contain another agent's work.

**Branches & history — never without careful planning AND explicit approval:**

- **Do not switch branches.** No `git checkout <branch>`, `git switch`, or `git checkout -b` on
  either the local or the HPC checkout without first planning it and **asking the user**. Switching
  the working tree out from under a sibling mid-edit is how work gets lost. Stay on the branch you
  were started on; if you think you need a new one, ask first.
- **Do not rebase, merge, reset, cherry-pick, revert, force-push, or `git restore`/`checkout -- <file>`**
  across branches or over shared history without very careful planning and explicit user approval.
  `git pull --rebase` is allowed **only** on your own branch, only when you actually need it.
- **One branch per agent.** Never commit to `main` or to another task's branch.

**On the shared HPC checkout especially:**

- Before running a job, check `git branch --show-current`. If it is **not** your branch, another
  agent owns the working tree right now — **do not `git checkout` to switch it.** Ask the user how
  to proceed (e.g. a separate worktree) rather than yanking the shared tree onto your branch.
- **Never delete `.git/index.lock` or other git internals** unless you have confirmed no other
  process or agent is mid-operation — and prefer to flag it to the user instead.
- Push frequently and in small units so your work is durable, but apply every rule above first.

## HPC connection

```
Host:      login.hpc.cam.ac.uk
User:      dca36
SSH:       ControlMaster auto, ControlPersist 8h (~/.ssh/sockets/)
Workspace: /home/dca36/workspace/BacPredict
Python:    always uv run python (never python or python3 directly)
```

Run commands on HPC: `ssh dca36@login.hpc.cam.ac.uk "<command>"`.

**Login node usage.** Jobs that complete in under **15 minutes** and stay under **128 GB RAM** (purely CPU — no GPU) can run directly on the login node without SLURM. (The real enforcement is CSD3's *watchdog*, which kills large or long-lived login-node processes; there is no published hard memory figure, so treat 128 GB / 15 min as a safe working ceiling.) Anything heavier (model inference, embedding generation, training, large-data parses) belongs in a SLURM job. Worked example: regenerating combined figures / summary CSVs from already-saved per-drug `eval_scores.npz` (see [src/kleb_ast/scripts/regen_panel_summary.sh](src/kleb_ast/scripts/regen_panel_summary.sh)) — pure matplotlib + small npz reads, finishes in seconds, not worth a queue wait.

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
