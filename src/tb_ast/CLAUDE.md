# Task 1 — AST in *M. tuberculosis*

This is one of six task folders under `src/`. See the root [CLAUDE.md](../../CLAUDE.md) for §0 global conventions (base model, three-stage protocol, paths, reporting requirements), and [BacPredict_Training_Plan.md](../../BacPredict_Training_Plan.md) §1 for the full long-form plan that this file condenses.

## Aim

Predict antibiotic susceptibility in TB from genome embeddings, starting with ESM-C protein embeddings then fine-tuning from the refreshed Bacformer complete-genomes model. Establish whether Bacformer-based prediction can improve on the WHO 2nd-edition catalogue for drugs in the "goldilocks zone."

## Status

- AST labels downloaded from EBI for the full TB set.
- Assemblies + GFFs already in `project_k/david/raw/tb/`.
- Protein lists extracted into `project_k/david/processed/tb/`.
- Per-antibiotic resistance stats: `project_k/david/processed/tb/antibiotic_testing_stats.csv`.
- **Not yet done:** ESM-C embeddings; Bacformer fine-tuning; refreshed-model run.

## The "goldilocks zone" — which drugs to train on

- **Tier 1 — saturated by catalogue** (positive-control sanity checks): RIF, INH, AMK/KAN/CAP, STR.
- **Tier 2 — goldilocks (target for headline runs):** **PZA (flagship)**, EMB, MXF, LFX, ethionamide.
- **Tier 3 — data-limited, parked:** bedaquiline, linezolid, clofazimine, delamanid/pretomanid (each < 1,000 resistant n).

See training plan §1 for the underlying catalogue / ML AUROC numbers and resistance n.

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

- `build_tb_input_csv.py` — TB (`Sample`, `assembly_file`, `gff_file`) input CSV from disk.
- `scripts/tb_collect_bakrep_samples.py` — TB-specific BakRep collection helper.
- `scripts/run_download_assemblies.sh` — CPU SLURM: download TB assemblies (ATB primary + NCBI fallback). Auto-retries until convergence (`--max-passes`, default 10).
- `scripts/run_download_bakrep.sh` — CPU SLURM: download TB Bakta GFF3s from BakRep. Same auto-retry loop.
- `scripts/run_embeddings_array_tb.sh` — GPU array SLURM: run ESM-C embeddings over the TB protein set.

Generic shared infrastructure lives in [`../tl/embed/`](../tl/embed/), [`../tl/genome_download/`](../tl/genome_download/), [`../tl/train/`](../tl/train/) (k-fold + lazy-dataset helpers). The actual training loop will reuse `train_amr.py` style code from [`../kleb_ast/`](../kleb_ast/) — when first needed, copy and adapt rather than abstract prematurely.

## Open questions / parked

- The EBI AST labels have known inconsistencies (different DSTs, different breakpoints). Refining against curated datasets (CRyPTIC, WHO V2) is **downstream** — only after the approach is shown worth pursuing in TB.

## Running notes

<!-- Agent appends here as work proceeds. Date entries (YYYY-MM-DD) and link to checkpoints / results JSON. -->
