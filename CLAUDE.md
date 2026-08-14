# CLAUDE.md

Guidance for Claude Code working in this repository. Per-task detail lives in each task folder's own `CLAUDE.md`.

> ## ⛔ Read [`PROJECT_STATE.md`](PROJECT_STATE.md) first
>
> **It is the single authority on current state** — status, numbers of record, what is in flight, and
> what is next, per layer. **This file holds conventions only and deliberately contains no results.**
>
> If a number here, in a task `CLAUDE.md`, in a memory, or anywhere else disagrees with
> `PROJECT_STATE.md`, **`PROJECT_STATE.md` wins**. A number is quotable only if it names the artifact
> it was read from. `ToDo.md` is **retired** — see [`docs/_retired/`](docs/_retired/).

## Project purpose

Fine-tune [Bacformer](https://github.com/amina-BS/bacformer) on bacterial genome embeddings to predict downstream phenotypes.

> **⚠️ Consolidated layout (2026-07).** The former parallel task folders (`tb_ast`, `kleb_ast`,
> `pangena_predict`, `tl`) were merged into **one engine + thin per-organism apps** under
> `src/bacpredict/`. See [`src/bacpredict/docs/`](src/bacpredict/docs/) for the pipeline overview. The
> retired [`~/.claude/plans/the-cambridge-hpc-and-dreamy-thacker.md`] plan drove the move.

| Area | Folder | Notes |
|---|---|---|
| **Engine** (organism-agnostic pipeline) | [src/bacpredict/engine/](src/bacpredict/engine/) | stages `ast_labels` · `download` · `embedding` · **`splits`** · `finetune` · `gene_lr` · `segment_amr_lr` · `concat` · `ref_catalogues` · `plots` |
| **App: TB AST** | [src/bacpredict/apps/tb/](src/bacpredict/apps/tb/) | WHO/TB-Profiler adapter, tbprofiler pixi, download helpers |
| **App: Kp AST** | [src/bacpredict/apps/kleb/](src/bacpredict/apps/kleb/) | CARD + Kleborate adapters, AMR sidecar pipeline, metadata curation |
| **Archived** (concluded TB SNP diagnostic) | [src/bacpredict/_archive/](src/bacpredict/_archive/) | excluded from wheel/ruff/pytest |
| AST-over-time (predict across ~80k → resistance trends) | [src/amr_over_time/](src/amr_over_time/) | separate package (uses the engine); **the primary `metadata_v2` consumer** — the core now only touches the Kleborate-comparator + CARD mut/WT slice. Ran; results mixed. |
| Isolation source in *Klebsiella* | [src/kleb_iso_source/](src/kleb_iso_source/) | separate package (uses the engine) |
| Pyseer GWAS | [src/bac_pyseer/](src/bac_pyseer/) | separate package |
| gene_array_lasso | [src/gene_array_lasso/](src/gene_array_lasso/) | separate package |

Mixed-assembly detection and the `predictHGT` embedding diagnostic are **parked** — their milestones
are in [docs/_parked/](docs/_parked/). DefensePredictor on short reads is deferred; `dp_short_read`
exists as a stub and keeps its own milestones.

`splits` is the newest stage and the one that must not be bypassed: it materialises a per-drug
`<drug>_split.csv`, and **`engine.splits.load_splits` is the one reader every downstream arm uses**.
Resolving a holdout any other way is what caused the 2026-07 read-out leak (§ *Data-leakage guarantees*).

The single AMR trainer is `bacpredict.engine.finetune.finetune_amr` (run `python -m …`, `--task tb_ast|kleb_ast`; both organisms train in **bf16**).

This root file holds only global conventions and shared-infra docs. **Per-task mechanics and running
notes live in each task folder's `CLAUDE.md`; state and numbers live only in
[`PROJECT_STATE.md`](PROJECT_STATE.md).** Three to four agents run concurrently, one per active task
(see §0.5).

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
| **A. Smoke** | Pipeline runs end-to-end | n=10 | 1 × 1 | **A short GPU sbatch — not the login node.** See below |
| **B. Overfit** | Loss → ~0 on a tiny set | n=10, train=test | 1 × 1 | Local or HPC interactive |
| **C. Full** | Headline result | full data, one canonical drug/task first | 1 × 1 | GPU HPC SLURM, ~36 h budget, early-stopping ≈ 15 epochs |

Folds × seeds (≥5 each) are an **advanced final step**, only for external publication. Do not burn compute on them during exploration.

> **⚠ Stage A must be a short GPU sbatch.** The old "MacBook CPU or HPC login, CUDA disabled" rule
> was wrong for this codebase: Bacformer-large plus per-sample RDS embedding I/O exceeds login-node
> limits, and a CPU login-node Stage A **silently produces empty tensorboard events** — it looks like
> it passed. A ~90 s single-GPU job is the correct smoke test.
>
> This is also why `dtype="auto"` must never be used to load Bacformer: it was introduced to make a
> CPU smoke test work, it resolves to **fp32** for Bacformer-large, and fp32 is measurably worse
> (~5 pp on TB rifampin). Cast explicitly to bf16. The smoke test it was serving is unsupported anyway.

**Known gap — no shared 36 h GPU template.** Per-task Stage C scripts exist but were never unified.
The gotcha they each encode: `--max-steps` must fit the 36 h wall at ~3 s/step, or the run must rely
on early stopping.

### §0.3 Paths

Cluster-dependent working-data root (layout `raw/<task>/`, `processed/<task>/`, `final/` is the same
on both — only the root differs). Code resolves the root via one env var,
`bacpredict.engine.config.resolve_data_root()` (`--data-root` arg → `$BACPREDICT_DATA_ROOT` →
`$SCRATCHDIR` → CSD3 path → error); individual input/output paths stay CLI-overridable.

- **Isambard:** root **`$SCRATCHDIR`** = `/scratch/u6fp/dca36.u6fp` (single root; model
  weights are the exception — `HF_HOME`/`TORCH_HOME` sit on `$PROJECTDIR/david/cache/`).
- **CSD3/UoHPC (operational again 29 Jul 2026; only tape cold-storage down):** root `/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/`
- Raw data: `<root>/raw/<task>/` · Processed: `<root>/processed/<task>/` · Curated: `<root>/final/`
- Local dev: `/Users/davidabelson/developer/BacPredict/`

**Per-cluster data maps** (actual store locations, counts, env, SBATCH — the lookup for filling any
`--input`/`--output`): [`src/bacpredict/docs/ISAMBARD_DATA.md`](src/bacpredict/docs/ISAMBARD_DATA.md)
(live-verified) · [`src/bacpredict/docs/HPC_DATA.md`](src/bacpredict/docs/HPC_DATA.md) (seeded from
docs; verify on CSD3's return). Storage-tier detail per cluster:
[`~/.claude/cluster_isambard.md`](~/.claude/cluster_isambard.md) ·
[`~/.claude/cluster_uohpc.md`](~/.claude/cluster_uohpc.md).

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

### §0.5a HPC resource requests + storage discipline — see the global cluster docs

**Cluster guidance is now cluster-agnostic and lives at the `~/.claude` level.** Read those before
running or tuning any `#SBATCH` directive — do not re-derive them here:

- [`~/.claude/CLAUDE.md`](~/.claude/CLAUDE.md) → **"Working on HPC clusters"** — the agnostic
  rules that hold on every cluster (storage discipline, no-`du`, logs/caches/envs off `$HOME`,
  code-via-git-not-scp, no login-node compute, be generous with `--time`, keep GPUs saturated).
- **The user says which cluster each session.** Then read the per-cluster doc for exact
  accounts, partitions, node sizes, storage tiers + quotas, and gotchas:
  [`~/.claude/cluster_isambard.md`](~/.claude/cluster_isambard.md) (aarch64 Grace+GH200; `workq`/
  `brics.u6fp`; `$PROJECTDIR` 200 TiB / `$SCRATCHDIR` 5 TiB / `$HOME` 100 GiB) ·
  [`~/.claude/cluster_uohpc.md`](~/.claude/cluster_uohpc.md) (CSD3 x86 `icelake`/`ampere`;
  `FLOTO-*` accounts; RDS `project_k`).

> **As of 29 Jul 2026 CSD3 is fully operational again** (SSH, RDS, `ampere` GPU + CPU jobs; only
> cold-storage *tape* files still inaccessible). Both CSD3 and Isambard are available — confirm
> which cluster each session.

When in doubt: **more**. The cost of asking for more than you need is zero;
the cost of asking for less is the entire job.

### §0.5 Concurrent agents — one per task

> ⚠️ **READ THIS BEFORE ANY GIT COMMAND.** Three to four Claude Code agents edit this
> repository **at the same time**, sharing the local checkout *and* the single HPC checkout at
> `/home/dca36/workspace/BacPredict`. **Git state (branch, index, working tree, history) is
> shared and easily corrupted.** Treat every git operation as something that can destroy or
> entangle another agent's live, uncommitted work. When in doubt, **stop and ask the user** —
> a question is always cheaper than a recovery.

Active area → folder map: TB AST = [src/bacpredict/apps/tb/](src/bacpredict/apps/tb/),
Kp AST = [src/bacpredict/apps/kleb/](src/bacpredict/apps/kleb/),
shared engine = [src/bacpredict/engine/](src/bacpredict/engine/),
isolation source = [src/kleb_iso_source/](src/kleb_iso_source/),
GWAS = [src/bac_pyseer/](src/bac_pyseer/). (Task 6 `predict_hgt` retired; the TB SNP diagnostic is
archived under `src/bacpredict/_archive/`.)

**Editing the shared engine affects every organism** — call it out when you touch `src/bacpredict/engine/`.
App-only changes stay within `src/bacpredict/apps/<organism>/` and its tests.
Touch the shared engine ([src/bacpredict/engine/](src/bacpredict/engine/)), top-level configs, or root
docs only when truly necessary, and call it out explicitly so the user can coordinate.

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

**Which cluster / login / SSH / storage tiers — see the global cluster docs** (the user says which
cluster each session): [`~/.claude/cluster_isambard.md`](~/.claude/cluster_isambard.md) and
[`~/.claude/cluster_uohpc.md`](~/.claude/cluster_uohpc.md) (CSD3, operational again 29 Jul 2026), plus the
agnostic [`~/.claude/CLAUDE.md`](~/.claude/CLAUDE.md) → "Working on HPC clusters".

- **Python:** always `uv run python` (never `python`/`python3` directly). On Isambard use an
  **aarch64** uv/pixi env (fresh solve) — CSD3/laptop solves don't transfer.
- **Run commands** over ssh (Isambard: `ssh u6fp.aip2.isambard "<command>"`; CSD3: `ssh dca36@login.hpc.cam.ac.uk "<command>"`).
- **Login-node usage** (both clusters): short (<~5–15 min), single-process, CPU-only orchestration
  may run inline; anything heavier (model inference, embedding, training, large-data parses) → a
  SLURM job. On Isambard, login-node processes are additionally *killed* on SSH disconnect — see its
  doc. Worked example of a fine login-node task: regenerating combined figures / summary CSVs from
  saved per-drug `eval_scores.npz` (see [src/bacpredict/engine/scripts/](src/bacpredict/engine/scripts/)).

## Package layout

`src/` holds **one wrapper package plus six standalone packages**. Distribution name is `bacpredict`.

**[src/bacpredict/](src/bacpredict/) — the AMR pipeline.** The engine is organism-agnostic; the apps
are thin adapters that supply only what differs between organisms.

| Engine subpackage | What it does |
|---|---|
| [engine/ast_labels/](src/bacpredict/engine/ast_labels/) | EBI/WHO AST tables → a binary label frame |
| [engine/download/](src/bacpredict/engine/download/) | Acquire and catalogue raw genomic data (BakRep, NCBI Datasets) |
| [engine/embedding/](src/bacpredict/engine/embedding/) | Protein sequences → ESM-C → Bacformer; baclm coding + non-coding |
| [engine/splits/](src/bacpredict/engine/splits/) | **The one holdout authority.** Materialises `<drug>_split.csv`; `load_splits` is the only reader |
| [engine/finetune/](src/bacpredict/engine/finetune/) | `finetune_amr` (the single AMR trainer), `evaluate`, `holdout.py`, `metrics`, `datasets` |
| [engine/gene_lr/](src/bacpredict/engine/gene_lr/) | Per-gene and per-region frozen-embedding logistic regressions |
| [engine/segment_amr_lr/](src/bacpredict/engine/segment_amr_lr/) | The clean-core segment LR (coding / intergenic / rRNA / upstream) |
| [engine/concat/](src/bacpredict/engine/concat/) | The AMR concat ladder |
| [engine/ref_catalogues/](src/bacpredict/engine/ref_catalogues/) | Catalogue determinant handling behind the ceilings |
| [engine/plots/](src/bacpredict/engine/plots/) | Every figure |

Apps: [apps/tb/](src/bacpredict/apps/tb/) (WHO / TB-Profiler) and
[apps/kleb/](src/bacpredict/apps/kleb/) (CARD + Kleborate, AMR sidecar, metadata curation).

**Standalone packages** (each with its own `CLAUDE.md`): [bac_pyseer](src/bac_pyseer/) (GWAS —
`kleb_iso_source` for invasion, `ast_gwas` for AMR), [kleb_iso_source](src/kleb_iso_source/)
(invasion fine-tuning), [amr_over_time](src/amr_over_time/),
[gene_array_lasso](src/gene_array_lasso/), [genome_prep](src/genome_prep/),
[dp_short_read](src/dp_short_read/) (stub).

They import engine helpers as `from bacpredict.engine.splits.load_splits import ...`,
`from bacpredict.engine.finetune.metrics import ...`, and so on. They do not import each other —
except that `bac_pyseer.ast_gwas` deliberately reuses `bac_pyseer.kleb_iso_source`'s GGCAT build and
sharded LMM rather than forking them.

> **⚠ Dead paths in older docs and every pre-2026-07-11 memory.** `src/tl/` → `engine/`;
> `src/tb_ast/`, `src/kleb_ast/` → `apps/{tb,kleb}/`; `src/pangena_predict/` → split across
> `engine/gene_lr/` and `_archive/`; `src/predict_hgt/`, `src/admixture/` → never existed / retired,
> see [docs/_parked/](docs/_parked/). Full table in [`PROJECT_STATE.md`](PROJECT_STATE.md) §2.

## Commands

```bash
# Install (editable) — on HPC use uv run python, locally use uv run
uv pip install -e .

# Run an engine stage as a module
uv run python -m bacpredict.engine.finetune.finetune_amr --help

# Run a standalone-package script by path
uv run python src/kleb_iso_source/stratified_isolation_source_sampling.py --help

# Tests
pytest tests/

# Lint
ruff check src/
```

## Training data architecture

Embedding files are large and are **never duplicated per experiment**: an embedding store + small CSVs
recording split assignments and labels.

**1. Embedding stores (read-only, one `.pt` per sample).** There are **two distinct stores** — do not
conflate them (this was the pre-2026-07 confusion):

- **Per-task AST stores (Isambard, current).** The only ESM/baclm/Bacformer embeddings on Isambard.
  One store per organism per stage, holding just that task's AST cohort:
  ```
  $BACPREDICT_DATA_ROOT/processed/train_{tb,kleb}_ast/{esm,baclm,bacformer,protein_sequences,intergenic}/{Sample}_<suffix>.pt
  ```
  Live counts: TB 38,257 · Kp 9,724 (see `src/bacpredict/docs/ISAMBARD_DATA.md`). This is what the
  trainer + all AST engine/apps modules resolve to (`OrganismConfig.data_root()` = `<root>/processed/train_<task>_ast`).
- **Flat whole-Klebsiella store (Cambridge/CSD3 only).** `processed/klebsiella_esm_embeddings/{Sample}_esm_embeddings.pt`
  — the ~84k-genome *whole Klebsiella genomics* set (a superset that includes most Kp-AST samples).
  **Consumed by `kleb_iso_source`, not the AST pipeline**, and it lives only on CSD3 — it is **not**
  present on Isambard. Don't point AST work at it.

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

> ### ⚠ The leak that actually happened was NOT in split generation
>
> Everything below was true and tested throughout, and it did not prevent a real leak — because the
> leak was in **downstream read-out scoring**, one layer away from where the guarantees live.
>
> **What happened.** Models are trained k-fold (fold 0 / seed 1). The ladder, the concat modules and
> the FT genome-mean cache each resolved "evaluate" independently, by reading the CSV's *single-split*
> `train_val_eval` column — a different partition entirely. For the worst-affected Kp drug, **81% of
> the genomes scored as held-out were in the model's own train/validate**, inflating its reported
> AUROC by more than a tenth (the figures are in `PROJECT_STATE.md` §7 and
> `visualisations/_superseded/README.md`).
>
> **Why the guarantees did not catch it.** They assert that a *split* is disjoint. They say nothing
> about whether a *consumer* asked for the right split. Every module that resolved a holdout its own
> way was a separate opportunity to ask wrongly, and one of them did.
>
> **The fix is structural, not a patch.** `engine/splits` materialises a per-drug `<drug>_split.csv`
> and **`load_splits` is the single reader**; the ladder fits on the FT train set and tests on the FT
> holdout, and guards its cache coverage (`9060617` → `eb39ce5` → `25e48cc`). Split-table ↔ deployed
> holdout equivalence is verified for all 32 drugs. **Never resolve a holdout any other way** — if you
> find yourself reading `train_val_eval` directly, you are reproducing this bug.
>
> **The fine-tunes were never affected**, which is why the July checkpoints remain authoritative.

The split logic is designed so that **no Sample ID can appear in more than one split within a single training run**, and the evaluate holdout is preserved across the entire k-fold sweep.

- **Single-split.** `add_splits()` shuffles unique `Sample` values, partitions into train 70 / validate 10 / evaluate 20. Tested in [tests/engine/finetune/test_split_utils.py::test_add_splits_no_overlap](tests/engine/finetune/test_split_utils.py).
- **K-fold.** Evaluate selected first and removed from the pool; remaining samples partitioned into mutually disjoint validation folds. For any `(fold, seed)`: `evaluate ∩ train = ∅`, `evaluate ∩ validate = ∅`, `train ∩ validate = ∅`. Across folds, **train sets share samples** (intrinsic to k-fold; only validate rotates). Evaluate is identical across every `(fold, seed)` when `evaluate_seed` is held constant. Tested in [tests/engine/finetune/test_split_utils.py](tests/engine/finetune/test_split_utils.py) and [tests/apps/kleb/test_pt_training_pipeline.py](tests/apps/kleb/test_pt_training_pipeline.py).

**Caveats — what these guarantees do NOT cover:**

- **Duplicate isolates under different accessions.** Split is over unique `Sample` values. If one biological isolate appears under multiple accessions, those copies are split independently and could land in both train and evaluate. Deduplication is an upstream metadata problem.
- **Bacformer pre-training overlap.** Bacformer was pre-trained on MAGs / complete genomes. Samples in our evaluate set may have been in that corpus — representation-level leakage not addressable by sample splitting in this repo.
- **Changing `--evaluate-seed` mid-experiment.** Evaluate holdout is only stable while `evaluate_seed` is held constant. Pin once per experiment.
- **Pre-existing `train_val_eval` column when `--n-folds` is set.** Ignored in k-fold mode — a sample previously labelled `evaluate` may end up in `train` for some fold. By design; k-fold owns its own splitting. **This is the exact gap the read-out leak fell through** — the column still exists and still looks authoritative to a reader who does not know it is ignored.
- **Anything that resolves a holdout without `engine.splits.load_splits`.** The guarantees above cover split *generation* only. A consumer reading the wrong split is not a split bug and will not be caught by any test of the splitter.

## Code style

- Line length: 120 characters
- Docstrings: NumPy convention
- Ruff rules: B, BLE, C4, D, E, F, I, RUF100, TID, UP, W
- Python 3.10–3.14
