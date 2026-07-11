# Task 1 — AST in *M. tuberculosis*

This is one of the task folders under `src/`. See the root [CLAUDE.md](../../CLAUDE.md) for §0 global conventions (base model, three-stage protocol, paths, reporting requirements). Cross-task status lives in [ToDo.md](../../ToDo.md).

## Aim

Predict antibiotic susceptibility in TB from genome embeddings, starting with ESM-C protein embeddings then fine-tuning from the refreshed Bacformer complete-genomes model. Establish whether Bacformer-based prediction can improve on the WHO 2nd-edition catalogue for drugs in the "goldilocks zone."

## Status

- AST labels downloaded from EBI for the full TB set.
- Assemblies + GFFs already in `project_k/david/raw/tb/`.
- Protein lists extracted into `project_k/david/processed/train_tb_ast/`.
- Per-antibiotic resistance stats: `project_k/david/processed/train_tb_ast/antibiotic_testing_stats.csv`.
- **Not yet done:** ESM-C embeddings; Bacformer fine-tuning; refreshed-model run.

## The "goldilocks zone" — which drugs to train on

- **Tier 1 — saturated by catalogue** (positive-control sanity checks): RIF, INH, AMK/KAN/CAP, STR. Catalogue already strong, so little ML headroom — RIF is the canonical first test (catalogue ~93–97% sens / 98.5–99% spec, AUROC ≥ 0.97; INH ~0.96). Resistance is chromosomal (*rpoB*/*katG*), the *harder* case for an HGT-leaning transformer.
- **Tier 2 — goldilocks (target for headline runs):** the middle tier where representation learning should beat lookup tables.
- **Tier 3 — data-limited, parked:** bedaquiline, linezolid, clofazimine, delamanid/pretomanid (each < ~1,000 resistant n; below the threshold ML needs).

Tier 2 detail (catalogue/ML numbers from the earlier TB deep-research review — kept here as the standing reference):

| Drug | Catalogue sens (/spec) | ML AUROC reported | n resistant (~) | Why it's a good target |
| :-- | :-- | :-- | :-- | :-- |
| **PZA (flagship)** | 26–66% | ~0.90–0.93 | ~2,500 | ~600 distinct *pncA* LOF alleles, mostly singletons — exactly where representation learning beats lookup |
| EMB | 80–94% / 91–94% | 0.88–0.92 | ~3,000 | *embB* M306V/I MIC-shifters straddling the ECOFF; structure adds ~3–5 AUROC pts |
| MXF | ~70% / ~92% | ~0.90 | ~3,200 | *gyrA* 90/91/94 with very different MIC effects vs the breakpoint |
| LFX | ~73% / ~94% | ~0.91 | ~3,000 | same story as MXF |
| Ethionamide | moderate | – | – | heterogeneous *ethA*/*ethR*/*inhA* — possible ML target |

## Central hypothesis being tested here

Bacformer should excel where resistance is driven by **HGT / gene acquisition** and add much less where resistance is driven by **chromosomal point mutations in conserved core genes**. TB is a near-worst case (almost all TB resistance is chromosomal point mutation) — a *conservative* test. Modest gain on the rare HGT-borne cases (e.g. acquired *eis* for KAN, some *rrs* alleles for STR) is still informative; the larger gain is expected in Kleb (Task 2). Every AMR result MUST be **stratified by resistance mechanism**.

## Plan / workflow milestones

1. **Refresh Bacformer weights from Hugging Face.** Shared infrastructure — do once, used by every task. Blocks everything below.
2. **Kick off ESM-C embeddings for the full TB protein set.** Slow — start immediately and let it run.
3. **Stage A smoke test on rifampicin** (n=10, CPU local). Pipeline check.
4. **Stage B overfit check on rifampicin** (n=10, train=test).
5. **Stage C full run on rifampicin** (full data, 1 fold × 1 seed, 36 h GPU SLURM). RIF is the canonical first test — failure here means the pipeline is wrong. Should match or beat the catalogue.
6. **Fan out to all goldilocks-zone drugs with > 1,000 resistant cases.** Single fold/seed each. **PZA is the flagship**; EMB / MXF / LFX next.
7. **Compare against WHO 2nd-edition catalogue and CRyPTIC ML benchmarks** for each drug.
8. **HGT-vs-vertical stratified performance — central hypothesis test.** Classify each resistant isolate via the **WHO V2 catalogue** (point mutation in a core gene vs acquired allele / gene gain). Re-compute AUROC, sensitivity, specificity **separately** for the two strata; report the delta. Mixed-mechanism isolates → own bucket. Where possible, hold out HGT-borne resistance from training and test transfer.
9. **Decide whether to invest in folds × seeds** for any drug for publication.

## Week of 2026-05-30 — assigned workstream items (B4, E3 control)

Anchor: program plan `~/.claude/PROGRAM_PLAN_2026-05-30.md`.

- **B4 — re-queue rifampin Stage C with eval-bias-toward-complete.** Once
  `tl/train/split_utils.py` learns `bias_eval_toward` (B2) and the prepare
  script propagates `is_complete` (B1), re-queue the rifampin run currently
  in flight (job 29776879 — let it finish for the unbiased baseline; the
  eval-bias re-run is the comparison). Same drill for the other 9
  goldilocks-zone drugs once they finish their first Stage C.
- **E3 — control task for Klebsiella-specific continued pretraining.**
  If E1 warrants E3, the new Kp-specialised Bacformer checkpoint should be
  used to fine-tune TB drugs as a **control**. Hypothesis: TB AUROCs should
  NOT improve (Kp-specific pretraining shouldn't help a non-Kp organism).
  If they do improve, the gain is general (better encoding of bacterial
  genome structure) rather than Kp-specific — which is interesting but
  means E3's framing in the paper changes.

No new code in this folder unless E3 fires.

## Three-stage testing protocol (recap of root §0.2)

| Stage | Scale | Folds × seeds | Where |
| :-- | :-- | :-- | :-- |
| **A. Smoke** | n=10 | 1 × 1 | MacBook M1 CPU (or HPC login) — code must run with CUDA disabled |
| **B. Overfit** | n=10, train=test | 1 × 1 | Local or HPC interactive |
| **C. Full** | full data, one canonical drug first | 1 × 1 | GPU HPC SLURM, ~36 h, early-stopping ≈ 15 epochs |

Folds × seeds (≥5 each) are an **advanced final step**, only for external publication. Do not burn compute on them during exploration.

## Reporting

Per root §0.4: AUROC, AUPRC, sensitivity, specificity, balanced accuracy, confusion matrix, calibration curve, per-drug / per-class breakdown. Save checkpoint + versioned results JSON for diffing.

**AMR-specific (mandatory):** every result must additionally be **stratified by resistance mechanism — HGT/acquired vs chromosomal point mutation**. Mechanism labels from the WHO V2 catalogue.

## Files in this folder

- `build_tb_input_csv.py` — TB (`Sample`, `sr_assembly_file`, `sr_gff_file`) input CSV from disk.
- `scripts/tb_collect_bakrep_samples.py` — TB-specific BakRep collection helper.
- `scripts/run_download_assemblies.sh` — CPU SLURM: download TB assemblies (ATB primary + NCBI fallback). Auto-retries until convergence (`--max-passes`, default 10).
- `scripts/run_download_bakrep.sh` — CPU SLURM: download TB Bakta GFF3s from BakRep. Same auto-retry loop.
- `scripts/run_embeddings_array_tb.sh` — GPU array SLURM: run ESM-C embeddings over the TB protein set.

Generic shared infrastructure lives in [`../tl/embed/`](../tl/embed/), [`../tl/genome_download/`](../tl/genome_download/), [`../tl/train/`](../tl/train/) (k-fold + lazy-dataset helpers). The actual training loop will reuse `train_amr.py` style code from [`../kleb_ast/`](../kleb_ast/) — when first needed, copy and adapt rather than abstract prematurely.

## Open questions / parked

- The EBI AST labels have known inconsistencies (different DSTs, different breakpoints). Refining against curated datasets (CRyPTIC, WHO V2) is **downstream** — only after the approach is shown worth pursuing in TB.

## Running notes

<!-- Agent appends here as work proceeds. Date entries (YYYY-MM-DD) and link to checkpoints / results JSON. -->

### 2026-05-25 — Stage A scaffolding on rifampin

Goal: refresh Bacformer to the complete-genomes weights and prove the pipeline
runs end-to-end on CPU. Confirmed HPC has more done than this doc previously
suggested.

Bacformer model ID refreshed (root §0.1):
- [src/tl/embed/generate_embeddings.py](../tl/embed/generate_embeddings.py) default
  now `macwiatrak/bacformer-large-masked-complete-genomes`.
- New [`train_amr.py`](train_amr.py) defaults to the same.

State on HPC at `project_k/david/processed/train_tb_ast/`:
- `binary_ast.csv` (40,021 rows × 20 drug columns) and `ebi_parsed_ast_metadata.csv`
  are already in place — EBI parsing is done.
- `tb_esm_embeddings/`: **35,156 / 38,248 protein-input rows have embeddings (~92%)**.
  ESM-C array job not yet fully converged; reruns recommended before Stage C.
- Drug column name in the CSV is **`rifampin`** (US spelling), not `rifampicin`.
  TB scripts and SLURM default to `rifampin` accordingly.

`prepare_esmc_embeddings_and_labels_to_finetune_amr.py` first run (2026-05-25):
- 33,687 unique samples retained after dropping 6,334 missing-embedding rows.
- Splits 70/10/20: train 23,611 / validate 3,390 / evaluate 6,686.
- Rifampin labels (per split, non-NaN): train 22,970 / validate 3,291 / evaluate 6,482.
- Output CSV: `processed/train_tb_ast/binary_ast_with_split.csv` (2.5 MB).
- Sidecar: `processed/train_tb_ast/ast_training/ast_samples_not_in_dataset.csv` lists the 6,334 dropped IDs.

Stage A smoke test (drug=rifampin, n_samples=10):
- First attempt on HPC login (CPU) was killed at the start of training (login-node
  CPU policy; the run also surfaced a `Detected kernel version 4.18.0 … can cause
  the process to hang` warning). Re-ran as a 10-min GPU SLURM job
  ([scripts/smoke_test_rifampin_gpu.sh](scripts/smoke_test_rifampin_gpu.sh)) per
  user direction.
- **PASS** (SLURM job 29712625, exit 0:0, wall 5:51 / 10:00 budget):
  - Train loss 1.059 → 0.0001 across 32 epochs.
  - Eval AUROC hit 1.0 by epoch 2 and stayed there (n=10 train=val ⇒ perfect
    memorisation expected; this confirms loss + backward + eval all wired up).
  - Checkpoints save and auto-rotate (`save_total_limit=1`).
  - `load_best_model_at_end` succeeded at the end of training.
  - Best checkpoint: `processed/train_tb_ast/checkpoints/smoke_rifampin_29712625/checkpoint-*`.

Note for Stage B/C: with `save_total_limit=1` the early "best" checkpoint
gets deleted as later ones save, then `load_best_model_at_end` tries to
reload the deleted path. The smoke run finished cleanly anyway (last
checkpoint coincidentally matched), but for real runs we should bump
`save_total_limit` to ≥ 3 or change `metric_for_best_model` so we don't
race the cleanup.

Open follow-ups (not blocking Stage A but should be addressed before Stage C):
- Backport `dtype="auto"` HF loading idiom to [src/kleb_ast/train_amr.py](../kleb_ast/train_amr.py)
  (the `.to(torch.bfloat16)` cast there pegs the kleb Stage A on CPU).
  [Resolved for kleb_iso_source already; kleb_ast still pending.]
- Adopt the richer kleb_ast/metrics module (now provides `build_results_payload`,
  `compute_full_metrics`, `write_results_json` per root §0.4 reporting requirements).
  Either move it to `tl/train/metrics.py` and share, or copy + adapt into `tb_ast/`.
  The current `tb_ast/train_amr.py` carries an older inline metrics function.

### 2026-05-28 — status before Stage B/C

- **TB ESM-C embeddings now 100% complete: 38,248 / 38,248.** The remaining ~8%
  filled by embedding array jobs 29712701_* (one `_4` TIMEOUT, re-run) and
  29765627_*, all COMPLETED. `binary_ast_with_split.csv` should be regenerated so
  the prune step picks up the newly-embedded samples before any full run.
- No TB Stage B or Stage C job has been launched yet (only the Stage A smoke
  29712625 exists). Nothing crashed.
- **Before launching TB Stage C, fix the step budget.**
  [scripts/train_on_slurm_amr_tb.sh](scripts/train_on_slurm_amr_tb.sh) currently
  passes `--max-steps 100000`, which at the observed ~3 s/step does NOT fit the
  36h wall — it would be killed around ~37% (~step 36-40k). Confirmed live by the
  sibling iso_source Stage C run hitting exactly this. Either lower `--max-steps`
  to what 36h allows (early stopping should bound it anyway) or raise the wall /
  rely on early-stopping firing first. Also apply the `save_total_limit` bump
  noted above for the same run.

### 2026-05-28 — Stage C launched (rifampin, single split)

Per user direction: skip an explicit Stage B (the Stage A n=10 train=val run
already serves as the overfit check) and go straight to Stage C. The
`--max-steps 100000` step-budget concern above is **deliberately left as-is** —
early stopping bounds the run.

Shared §0.4 metrics adopted:
- `metrics.py` now lives at [../tl/train/metrics.py](../tl/train/metrics.py)
  (moved out of kleb_ast into the shared toolbox). NB: the move had been
  committed upstream but left two `from kleb_ast.metrics import` statements
  dangling (kleb_ast/train_amr.py + the moved test) — repointed both to
  `tl.train.metrics` in the same commit, which un-breaks those imports.
- [train_amr.py](train_amr.py) re-synced to the §0.4 reporting path: sets
  `split_source` + `evaluate_ids` per branch (smoke/kfold/csv) and writes a
  versioned `results.json` (full metric set: AUROC, AUPRC, sens, spec, bal-acc,
  F1, confusion matrix, calibration) on the evaluate holdout after training.
  Keeps `dtype="auto"` (no `.to(torch.bfloat16)` regression) and TB defaults.

Split regenerated at 100% embedding coverage
(`prepare_esmc_embeddings_and_labels_to_finetune_amr.py`, 2026-05-28 14:29):
- 40,021 AST rows → **36,684 kept** (3,337 still missing embeddings), up from
  33,687 at the earlier ~92% coverage.
- Rifampin non-NaN per split: train 24,977 / validate 3,574 / evaluate 7,075,
  ~31% resistant across all three (healthy balance).
- Note: the prepare script crawls stat-ing ~38k embedding files; on a loaded
  login node it took ~10 min. Next time run it as a CPU sbatch job (~1 h wall).

Stage C single-split run:
- New [scripts/train_on_slurm_amr_tb_stage_c.sh](scripts/train_on_slurm_amr_tb_stage_c.sh)
  — no `--array`/`--n-folds`/`--fold`/`--seed`, so `run()` reads
  `train_val_eval` straight from the CSV (`split_source="csv"`). The k-fold
  array sweep stays in `train_on_slurm_amr_tb.sh` (publication only).
- **Submitted: SLURM job 29776879** (ampere, FLOTO-SL2-GPU, 36 h),
  drug=rifampin, output
  `processed/train_tb_ast/checkpoints/mycobacterium_tuberculosis_rifampin_stage_c_29776879/`.
  Queued (PENDING/Priority) at submit; `results.json` expected in the checkpoint
  dir on completion.

### 2026-05-29 — RDS dir rename + fan-out to the drug panel

- **ESM-C embeddings now complete: 38,248 (= all protein-input rows).** The split
  CSV was regenerated 2026-05-28 14:29 to **36,684 rows** — the full intersection of
  40,021 labelled samples ∩ 38,248 embeddings (the remaining 3,337 labels have no
  embedding; 36,684 is the achievable ceiling). The 2026-05-25 figures above
  (~92% coverage, 33,687 retained) are superseded. Rifampin job 29776879 started
  14:56, after the regen, so it and all fan-out jobs train on the full 36,684 split.
- Renamed the TB data dir `processed/tb` → **`processed/train_tb_ast`** to match
  the `train_kleb_ast` / `train_on_sr_mags` convention. A compat symlink
  `processed/tb → train_tb_ast` was left in place so the in-flight rifampin job
  29776879 (which had hard-coded `processed/tb` paths) keeps resolving; it can be
  removed once that job finishes. All scripts now point at `train_tb_ast` directly.
- Rifampin Stage C at ~epoch 6: validation AUROC plateauing ~0.88–0.89 (balanced
  acc ~0.81). Consistent with the conservative-test expectation for a chromosomal
  point-mutation drug.
- **Fan-out:** launched single-split Stage C for the 9 next-highest-resistance
  drugs (rifampin already running). Selection = top-10 by resistant-label count,
  all > 1,000 R: isoniazid (12,838 R), ethambutol (5,266), rifabutin (4,384),
  levofloxacin (2,614), streptomycin (2,537), moxifloxacin (2,475),
  pyrazinamide (2,415), ethionamide (2,210), kanamycin (2,179). Launcher
  parametrized by drug (`$1`).
- Job IDs (all single-split Stage C, ampere/FLOTO-SL2-GPU, 36 h): isoniazid
  29824376, ethambutol 29824377, rifabutin 29824378, levofloxacin 29824379,
  streptomycin 29824380, moxifloxacin 29824381, pyrazinamide 29824382,
  ethionamide 29824383, kanamycin 29824384. Each writes its `results.json` to
  `processed/train_tb_ast/checkpoints/mycobacterium_tuberculosis_<drug>_stage_c_<jobid>/`.
