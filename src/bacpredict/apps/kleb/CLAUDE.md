# Task 2 — AST in *Klebsiella pneumoniae* (`apps/kleb`)

> **⚠️ Post-consolidation note (2026-07).** This package moved from `src/kleb_ast/` to
> `src/bacpredict/apps/kleb/` in the engine consolidation. The organism-agnostic pipeline now lives in
> `src/bacpredict/engine/` (stages: `labels`, `download`, `embedding`, `finetune`, `gene_lr`, `concat`,
> `catalogue`, `plots`); this folder holds only Kp specifics (CARD/Kleborate adapters, the AMR sidecar
> pipeline, metadata curation, the epidemiology plotter). **Fine-tuning is now the single shared trainer**
> `bacpredict.engine.finetune.finetune_amr` (invoke `python -m …`, `--task kleb_ast`); the old
> `kleb_ast/train_amr.py` + `prepare_esmc_…` are gone (merged into the engine). Both organisms train in
> **bf16**. Many file/import references in the sections below predate the move — trust the engine layout.
>
> **Catalogue policy (REVISED — reverses the "Kleborate ceiling" section below).** **CARD is the DEFAULT
> Kp determinant ceiling** (`card_determinant_lr`): it resolves to *specific mutations*, which Kleborate's
> per-isolate calls cannot, and CARD is *also* the acquired-gene **locator** (`card_gene_locator` supplies
> the flat protein index of blaKPC/armA/… that Bakta under-annotates — you cannot embed a gene you cannot
> locate). **Kleborate (`kleborate_determinant_lr`) is retained as a comparator** for readers who treat it
> as the gold standard. Both ceiling runners share `engine.catalogue.base.score_onehot_frame`. (Memory
> `kleborate-ceiling-vs-amr-tools` updated to match.)

See the root [CLAUDE.md](../../../CLAUDE.md) for §0 global conventions (base model, three-stage protocol, paths, reporting requirements). Cross-task status lives in [ToDo.md](../../../ToDo.md).

## Aim

Predict susceptibility for clinically relevant antibiotics in *Klebsiella pneumoniae*, where we expect Bacformer to do best — resistance is heavily HGT-driven (carbapenemases, ESBLs, *mcr*) on plasmids and ICEs. This is the natural home for our AUROC 0.99 result, **and the strong test of the central HGT-vs-vertical hypothesis** (see Task 1 [tb_ast/CLAUDE.md](../tb_ast/CLAUDE.md)): unlike TB, Kp has both classes of mechanism well-represented in the same dataset, so a clean stratified comparison is possible.

## Status

- We already have trained Kp prediction models but have **not formally evaluated** them.
- All previous training used the **older Bacformer weights** (now superseded) and the MAG-trained model.
- Source notebook: `/Users/davidabelson/developer/BacHGT/docs/notebooks/amr_ebi_records.ipynb`.

## Central hypothesis being tested here

Bacformer should excel where resistance is driven by **HGT / gene acquisition** (carbapenemases like KPC/NDM/OXA-48, ESBLs, aminoglycoside-modifying enzymes, *mcr*) and add much less where resistance is driven by **chromosomal point mutations** (e.g. FQ via *gyrA*/*parC* QRDR, *ramR/ramA*-driven efflux, *ompK35*/*K36* porin loss, *pmrAB*/*phoPQ* colistin). Kp is the **strong** test — both classes are well-represented. Every AMR result MUST be **stratified by resistance mechanism**.

## Kleborate determinant "ceiling" — the catalogue baseline (pangena_predict port)

The TB diagnostic (Task 7, `src/pangena_predict/`) measures every drug's Bacformer read-out against a
**catalogue ceiling** — the AUROC a one-hot of all known resistance determinants reaches through the
same k-fold LR. For TB that catalogue is the WHO/TB-Profiler variant set. **For Kp the catalogue is
Kleborate**, whose per-isolate determinant calls are already in `metadata_v2` (no re-run needed).

- **Module:** [`kleborate_determinant_lr.py`](kleborate_determinant_lr.py) — the Kp analogue of
  `pangena_predict/tbprofiler_gene_lr.py`. Per drug it builds a determinant one-hot from the relevant
  Kleborate columns and scores it through `pangena_predict.kfold_probe.run_kfold_probe`, emitting one
  **bar per Kleborate column** (tagged `acquired_hgt` / `chromosomal_coding` / `chromosomal_mutation` /
  `porin_truncation` / `truncation_lof` — the HGT-vs-chromosomal axis) plus the combined
  `__ALL_Kleborate__` **ceiling** row, to `docs/visualisations/kp_<drug>/kleborate_determinant_lr_<drug>.csv`.
  Light CPU (login node / small sbatch, no GPU). Inputs: `final/metadata_v2_all_samples_and_columns.tsv`
  + `processed/train_kleb_ast/binary_ast_with_split.csv`, joined on `Sample`. The `bac_kleborate.parsing`
  cell semantics are **vendored** (that package is in the sibling BacHGT repo, not a dependency here).
- **Why Kleborate alone — resolved 2026-06-19 (memory `kleborate-ceiling-vs-amr-tools`).** For the KpSC
  module Kleborate v3 (v3.2.4 = what built metadata_v2) is **CARD-derived** (CARD v3.2.9), *not*
  AMRFinderPlus (that engine drives Kleborate's *Escherichia* module). A 2025 benchmark of 8 tools
  (Sci Rep s41598-025-24333-9) found Kleborate detects the most ARGs and that **integrating other tools
  does not improve Kp determinant-based prediction** ("fewer, well-curated features outperform quantity").
  The weak β-lactam/tetracycline drugs are **literature-wide catalogue knowledge gaps** — *no* tool
  predicts them well — so a low Kleborate ceiling there is faithful, not an artefact. **The asymmetry that
  matters:** true catalogue ceiling ≥ Kleborate ceiling, so incompleteness only changes a conclusion where
  concat *beats* the ceiling (the Kp analogue of TB pyrazinamide) — there, double-check; where concat is
  *below* it, the conclusion is safe. Decision: Kleborate alone, coverage-annotated per drug; no
  AMRFinder/RGI/ResFinder re-run.
- **REVIEW POINT — the drug→Kleborate-column map** (`DRUG_COLUMNS` in the module) is a clinical-pharmacology
  judgment. β-lactams map *inclusively* to all `Bla_*` + `Omp_mutations` (a ceiling should use every
  plausible determinant and let the LR weight them). Residual deep-research item, off the critical path:
  ResFinder/PointFinder vs Kleborate point-mutation coverage for **colistin / azithromycin** (neither was
  in the 2025 study).

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

## Week of 2026-05-30 — assigned workstream items (B4, E1, E2/E3)

Anchor: program plan `~/.claude/PROGRAM_PLAN_2026-05-30.md`.
Shared-infra agent runs B first (branch `feat/complete-genome-aware-splits`)
then E1/E2 (branch `feat/bacformer-snp-probe`); the actual re-runs land in
this folder.

**Central hypothesis for this week:** chromosomal-SNP-driven resistance
(cipro is the canonical exemplar) is systematically under-predicted because
(i) Bacformer collapses near-identical alleles into the same protein family,
and (ii) short-read assemblies further dilute the per-protein signal.

- **B4 — re-run Stage C with eval-bias-toward-complete.** Once
  `tl/train/split_utils.py` learns `bias_eval_toward` (B2) and the prepare
  script propagates `is_complete` (B1), re-run Stage C for meropenem,
  ceftriaxone, gentamicin **plus cipro** (cipro is added as the chromosomal-
  leaning bellwether). Compare AUROC to the 2026-05-29 unbiased baselines
  (0.969 / 0.983 / 0.978 / TBD). Prediction: a small but real bump because
  the holdout is easier; cipro's delta is the headline.
- **E1 — gyrA/gyrB/parC allele probe (RUN FIRST in workstream E).** New
  script `allele_representation_probe.py` (here or in `predict_hgt/` — same
  methodology as predict_hgt's HGT-preserving vs context-attractor
  diagnostic; coordinate with the predict_hgt agent on location before
  duplicating). For ~50–100 cipro-R and matched cipro-S isolates, identify
  gyrA/gyrB/parC by locus_tag (Kleborate AMR call carries the WHO-style
  mutation). Pull per-protein ESM-C embedding (mean-pool per protein) and
  Bacformer contextualised per-protein embedding (before genome pooling).
  Report pairwise cosine distance between WT and mutant within each model +
  per-residue ESM-C deltas as sanity check. **This gates E2 + E3:** if
  ESM-C separates them but Bacformer doesn't → Bacformer is the bottleneck
  and E2/E3 are worth running. If neither separates → the fix is upstream
  of Bacformer.
- **E2 — attention-weighted genome pooling.** Only if E1 warrants. Replace
  Bacformer's mean-pool with a learned attention head
  (`src/tl/train/attention_pool.py`, owned by shared-infra agent). Smoke +
  Stage C on cipro and blood/faeces. Headline test: cipro AUROC delta vs
  mean-pool baseline. Don't expect movement on meropenem (HGT-bound, 0.97+).
- **E3 — Klebsiella-specific continued masked pretraining.** Only if E1
  warrants. Continued masked-LM on Kp genomes using the existing 50k
  centroid vocabulary (cluster_centers.npy on the Bacformer RDS). Output
  is a new checkpoint that re-fine-tunes all four BacPredict tasks; this
  folder is the primary downstream consumer.

Mechanism stratification (HGT vs chromosomal — central programme
hypothesis) is still required for every result. Produce the stratified
report block alongside the cipro / ceftriaxone / meropenem / gentamicin
Stage C results, even on the first eval-bias re-run.

Open follow-up (still applies): backport `dtype="auto"` HF loading idiom
to [train_amr.py](train_amr.py) — `.to(torch.bfloat16)` regression hits
the CPU Stage A path (already fixed in tb_ast `4956f91` and
kleb_iso_source `2d5866e`).

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
- `preprocess_ebi_amr_records.py` — thin Kp wrapper; delegates to the canonical parser.
- **Canonical EBI AST → binary parser now lives in `pangena_predict/parse_ebi_ast_to_binary.py`**
  (organism-agnostic; was the misnamed `kleb_ast/convert_ast_data.py`). Use it for TB and Kp alike.

Kleb-specific metadata / embedding curation
- `add_paths_gff_fna_to_metadata.py` — populate `sr_assembly_file` + `sr_gff_file` in the Kleb metadata TSV.
- `add_bakta_gbff_downloaded_flag.py` — scan `klebsiella_gbff/` and update metadata.
- `find_missing_embeddings.py` — list `kpsc_final_list` samples missing embeddings.
- `filter_esmc_embeddings_by_klebsiella.py` — filter embedding parquets to KPSC-only.
- `extract_anndata_with_bacformer_protein_embeddings.py` — AnnData from Bacformer embeddings (Clonal group / K_locus / K_type).
- `scripts/flatten_klebsiella_gff3.py` — Kleb-side GFF flattening helper.
- `scripts/add_paths_gff_fna_to_metadata.sh` — wrapper.

Determinant ceiling / mechanism stratification
- `kleborate_determinant_lr.py` — per-drug Kleborate determinant one-hot LR → the catalogue **ceiling** + per-mechanism bars (HGT vs chromosomal). The Kp analogue of `pangena_predict/tbprofiler_gene_lr.py`. See the section above.

Imports from [`../tl/train/`](../tl/train/) (split_utils, datasets) and [`../tl/embed/`](../tl/embed/) and [`../tl/genome_download/`](../tl/genome_download/) for shared infrastructure, and from [`../pangena_predict/`](../pangena_predict/) (`kfold_probe`, and `locate_gene` for the Kp gene→embedding-index port).

## Downstream / parked experiments (all on hold)

- **(i) Held-out lineages / held-out subspecies** — test generalisation across ST/CG boundaries. "Does the model learn AMR biology or lineage shortcuts?"
- **(ii) Drug-class embedding** — single model across all carbapenems (or aminoglycosides) with drug as input embedding.
- **(iii) Explainability** — Captum integrated gradients + feature ablation, gene-level attribution table per drug.
- **(iv) Cross-trained model attribution** for explainability robustness.
- **(v) Pre-train on Kp complete-genome masked-gene prediction**, then fine-tune to AMR (plays into Task 3).
- **(vi) Read-depth-aware gene copy correction** — duplicate AMR proteins in Bacformer input when depth is ~2× genome average. Advanced (changes input construction).

## Running notes

<!-- Agent appends here as work proceeds. -->

### 2026-05-29 — Sub-step 2 (CG-weights retrain) done for first 3 drugs

**Infra (Phase 0).** Added `metrics.py`: `compute_full_metrics` (§0.4 block — AUROC,
AUPRC, sens, spec, balanced acc, F1, confusion matrix, calibration), HF-Trainer
wrapper, and `write_results_json` (schema in `docs/results_schema.md`).
`train_amr.py` auto-writes `results.json` (evaluate-holdout metrics) to the
checkpoint dir after `trainer.train()`. Defaults flipped to the refreshed CG
weights `macwiatrak/bacformer-large-masked-complete-genomes`. SLURM: `sbatch
--array=0` = single fold 0 / seed 1 Stage C; `--export=ALL,DRUG=<drug>` drives
per-drug fan-out (keep `--array=0-14` for the publication 5×3 sweep).

**Stage A/B.** Smoke + overfit passed (n=10, loss→0, AUROC→1). The all-class-0
threshold collapse seen on n=10 is a tiny-set/early-checkpoint artifact — absent
at full scale (confirmed below).

**Stage C benchmarks** (CG weights, kfold fold 0 / seed 1, fixed evaluate holdout):

| Drug | n_eval | AUROC | AUPRC | Sens | Spec | Bal acc |
|---|---|---|---|---|---|---|
| ceftriaxone | 641 | 0.983 | 0.993 | 0.983 | 0.889 | 0.936 |
| gentamicin | 943 | 0.978 | 0.970 | 0.955 | 0.973 | 0.964 |
| meropenem | 880 | 0.969 | 0.945 | 0.865 | 0.964 | 0.915 |

Checkpoints + `results.json` under
`processed/train_kleb_ast/models/finetune/klebsiella_pneumoniae_<drug>_lr_0.00015_finetuned_fold00_seed1/`.
meropenem sens (0.865) is the softest — 43/318 R below the 0.5 threshold despite
strong ranking; threshold tuning could recover these.

**Data layout.** All AST CSVs co-located under `processed/train_kleb_ast/`
(`binary_ast.csv`, `binary_ast_with_split.csv`, `regression_log_mic.csv`,
`klebsiella_ebi_metadata.csv`, `ast_samples_not_in_dataset.csv`). Producer/consumer
path defaults updated accordingly.

**Deferred (by user).** Sub-step 1 (evaluate legacy MAG checkpoints).
AMRFinderPlus/Kleborate HGT-vs-chromosomal mechanism stratification.

**Next.** (1) Fan out to the full panel (~top 23 drugs by sampling — see binary_ast
counts; exclude intrinsic ampicillin, verify `pentizidone`, consider colistin for
the chromosomal arm). (2) Formal test-set evaluation of the 3 done drugs with ROC +
PR curves.
