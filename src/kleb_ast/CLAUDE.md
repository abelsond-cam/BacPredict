# Task 2 — AST in *Klebsiella pneumoniae*

This is one of six task folders under `src/`. See the root [CLAUDE.md](../../CLAUDE.md) for §0 global conventions (base model, three-stage protocol, paths, reporting requirements), and [BacPredict_Training_Plan.md](../../BacPredict_Training_Plan.md) §2 for the long-form plan that this file condenses.

## Aim

Predict susceptibility for clinically relevant antibiotics in *Klebsiella pneumoniae*, where we expect Bacformer to do best — resistance is heavily HGT-driven (carbapenemases, ESBLs, *mcr*) on plasmids and ICEs. This is the natural home for our AUROC 0.99 result, **and the strong test of the central HGT-vs-vertical hypothesis** (see Task 1 / training plan §1): unlike TB, Kp has both classes of mechanism well-represented in the same dataset, so a clean stratified comparison is possible.

## Status

- We already have trained Kp prediction models but have **not formally evaluated** them.
- All previous training used the **older Bacformer weights** (now superseded) and the MAG-trained model.
- Source notebook: `/Users/davidabelson/developer/BacHGT/docs/notebooks/amr_ebi_records.ipynb`.

## Central hypothesis being tested here

Bacformer should excel where resistance is driven by **HGT / gene acquisition** (carbapenemases like KPC/NDM/OXA-48, ESBLs, aminoglycoside-modifying enzymes, *mcr*) and add much less where resistance is driven by **chromosomal point mutations** (e.g. FQ via *gyrA*/*parC* QRDR, *ramR/ramA*-driven efflux, *ompK35*/*K36* porin loss, *pmrAB*/*phoPQ* colistin). Kp is the **strong** test — both classes are well-represented. Every AMR result MUST be **stratified by resistance mechanism**.

## Three sub-steps (in order)

1. **Evaluate current Kp models** and save the results as the benchmark we are trying to beat. No retraining yet — just produce the report.
2. **Retrain from the refreshed Bacformer complete-genomes weights.** The main retraining. Compare against (1).
3. **Retrain a small subset of drugs from the MAG-trained model.** Hypothesis: minimal difference. One-paragraph confirmation result for the paper, not a full benchmark.

## Plan / workflow milestones

1. Stage A smoke test on the existing Kp pipeline against the refreshed model (one canonical drug — meropenem or ceftriaxone).
2. Stage B overfit check.
3. Stage C full run on canonical drug.
4. Fan out across the Kp drug panel.
5. Side-by-side: old-Bacformer benchmark vs new-Bacformer vs MAG-model.
6. **HGT-vs-vertical stratified performance — central hypothesis test.** For each isolate, run **AMRFinderPlus** and **Kleborate** to label every resistance determinant by origin (acquired gene / HGT vs chromosomal point mutation). Per drug, stratify and report AUROC / sens / spec **separately** for HGT-resistant vs vertically-resistant isolates. Mixed-mechanism → own bucket. Headline figure for the paper: the **delta** in Bacformer's gain over baseline between the two strata. Strongest test = held-out-by-mechanism (train on one stratum, test transfer to the other).

## Three-stage testing protocol (recap of root §0.2)

| Stage | Scale | Folds × seeds | Where |
| :-- | :-- | :-- | :-- |
| **A. Smoke** | n=10 | 1 × 1 | MacBook M1 CPU (or HPC login) — code must run with CUDA disabled |
| **B. Overfit** | n=10, train=test | 1 × 1 | Local or HPC interactive |
| **C. Full** | full data | 1 × 1 | GPU HPC SLURM, ~36 h |

Folds × seeds (≥5 each) only for external publication.

## Reporting

Per root §0.4: AUROC, AUPRC, sens, spec, balanced acc, confusion matrix, calibration curve, per-drug / per-class breakdown. Save checkpoint + versioned results JSON.

**AMR-specific (mandatory):** every result must additionally be **stratified by resistance mechanism — HGT/acquired vs chromosomal**. Mechanism labels from **AMRFinderPlus + Kleborate**.

## Files in this folder

Training entrypoints
- `train_amr.py` — fine-tune Bacformer for one antibiotic; `--n-folds`/`--fold`/`--seed` for k-fold CV.
- `scripts/train_on_slurm_amr.sh` — GPU array SLURM (5 folds × 3 seeds = 15 jobs).

Label / data prep
- `prepare_esmc_embeddings_and_labels_to_finetune_amr.py` — merge AST labels + embeddings → split CSV.
- `preprocess_ebi_amr_records.py` — parse EBI AST CSV → binary resistance labels.
- `convert_ast_data.py` — `process_klebsiella_ast_data()` helper.

Kleb-specific metadata / embedding curation
- `add_paths_gff_fna_to_metadata.py` — populate `assembly_file` + `gff_file` in the Kleb metadata TSV.
- `add_bakta_gbff_downloaded_flag.py` — scan `klebsiella_gbff/` and update metadata.
- `find_missing_embeddings.py` — list `kpsc_final_list` samples missing embeddings.
- `filter_esmc_embeddings_by_klebsiella.py` — filter embedding parquets to KPSC-only.
- `extract_anndata_with_bacformer_protein_embeddings.py` — AnnData from Bacformer embeddings (Clonal group / K_locus / K_type).
- `scripts/flatten_klebsiella_gff3.py` — Kleb-side GFF flattening helper.
- `scripts/add_paths_gff_fna_to_metadata.sh` — wrapper.

Imports from [`../tl/train/`](../tl/train/) (split_utils, datasets) and [`../tl/embed/`](../tl/embed/) and [`../tl/genome_download/`](../tl/genome_download/) for shared infrastructure.

## Downstream / parked experiments (all on hold)

- **(i) Held-out lineages / held-out subspecies** — test generalisation across ST/CG boundaries. "Does the model learn AMR biology or lineage shortcuts?"
- **(ii) Drug-class embedding** — single model across all carbapenems (or aminoglycosides) with drug as input embedding.
- **(iii) Explainability** — Captum integrated gradients + feature ablation, gene-level attribution table per drug.
- **(iv) Cross-trained model attribution** for explainability robustness.
- **(v) Pre-train on Kp complete-genome masked-gene prediction**, then fine-tune to AMR (plays into Task 3).
- **(vi) Read-depth-aware gene copy correction** — duplicate AMR proteins in Bacformer input when depth is ~2× genome average. Advanced (changes input construction).

## Running notes

<!-- Agent appends here as work proceeds. -->
