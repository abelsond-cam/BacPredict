# Task 3 — Isolation source in *Klebsiella*

This is one of six task folders under `src/`. See the root [CLAUDE.md](../../CLAUDE.md) for §0 global conventions, and [BacPredict_Training_Plan.md](../../BacPredict_Training_Plan.md) §3 for the long-form plan.

## Aim

Test whether Bacformer fine-tuned via BacPredict can pick up genomic markers that distinguish invasive from non-invasive infection. The canonical comparison is **blood vs stool** (the field has converged on this — cleanest two-source comparison: largest n, least ambiguous microbiology, strongest biological interpretation). Liver abscess vs stool is an even stronger hvKp signal where it can be text-mined from metadata, even at modest n.

## Status

- Models already trained on isolation source, achieving **AUROC 0.55–0.62 (poor)**.
- Those models used the **old Bacformer weights** and the **MAG-trained model** — both to be replaced.
- Labels are noisy: ~50k human samples with isolation source, of which ~13k urine, ~13k blood, ~10k stool, ~7k respiratory, ~3k wound/abscess (a messy mixed category — liver abscess, ascites, surgical drains, etc.).
- Urine is likely a bad target (many superficial UTIs; minority invasive) — leave for downstream stratified analysis.

## Design decisions (deliberate simplifications for the first pass)

- **Use all samples from all countries.** No stratified sampling, no max 2:1 country ratio. Accept the bias; see how high we can push the model. Country-stratified sampling is downstream.
- **Start with blood vs stool only.** Get one clean number first.
- Regenerate train/val/eval splits from scratch and re-label embeddings against the refreshed Bacformer model.

## Plan / workflow milestones

1. Regenerate train/val/eval splits for blood vs stool.
2. Stage A smoke test (blood-vs-stool, n=10).
3. Stage B overfit check.
4. Stage C full run — single fold/seed, full data, 36 h GPU.
5. Evaluate against the prior 0.55–0.62 AUROC benchmark. If meaningfully better, we have a result worth pursuing.

## Three-stage testing protocol (recap of root §0.2)

| Stage | Scale | Folds × seeds | Where |
| :-- | :-- | :-- | :-- |
| **A. Smoke** | n=10 | 1 × 1 | MacBook M1 CPU (or HPC login) |
| **B. Overfit** | n=10, train=test | 1 × 1 | Local or HPC interactive |
| **C. Full** | full data | 1 × 1 | GPU HPC SLURM, ~36 h |

Folds × seeds (≥5 each) only for external publication.

## Reporting

Per root §0.4: AUROC, AUPRC, sens, spec, balanced acc, confusion matrix, calibration curve, per-class breakdown. Save checkpoint + versioned results JSON for diffing.

## Files in this folder

Training entrypoint
- `train_isolation_source.py` — fine-tune Bacformer for one isolation-source pair; `--n-folds`/`--fold`/`--seed` for k-fold CV.
- `scripts/train_isolation_source.sh` — GPU array SLURM. Pair tokens edited inline in the .sh file.

Label / cohort prep
- `prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py` — merge labels + embeddings → split CSV.
- `stratified_isolation_source_sampling.py` — select balanced cohort for one isolation-source pair.
- `isolation_source_cli_parsing.py` — pair-token CLI parsing (shared between prepare + train scripts).
- `scripts/prepare_iso_source_data_for_training.sh` — CPU SLURM wrapper.
- `scripts/cpu_slurm.sh` — generic CPU job template used by the prep step.

Imports from [`../tl/train/`](../tl/train/) (split_utils, datasets) and [`../tl/embed/`](../tl/embed/) for shared infrastructure.

## Downstream / parked experiments

- **(i) Klebsiella-specific pre-training before isolation source.** Same idea as Task 2 downstream (v): masked next-gene prediction on Kp complete genomes, then fine-tune to isolation source. Hypothesis: makes the model meaningfully more aware of Kp accessory architecture and phylogeny.
- **(ii) Complete-genomes-only fine-tuning subset.** Train/val/eval drawn exclusively from complete genomes. Smaller dataset (worse, ceteris paribus) but possibly better because inputs include plasmid- and ICE-borne virulence factors that are exactly what drives invasion.
- **(iii) Matched-pair short-read vs complete-genome comparison.** Isolates with both a complete genome and a short-read assembly from the same sample? Only fully-controlled way to isolate the "assembly quality" effect.
- **(iv) Gene-level explainability** (Captum IG + ablation). Expected hits: *rmpA/rmpA2*, *iuc*, *iro*, K1/K2 capsule loci, *ybt*/*clb* on ICE*Kp*.
- **(v) Stepwise AUROC across modelling layers.** raw ESM-C → frozen Bacformer → fine-tuned Bacformer. Quantifies what contextualisation buys and what fine-tuning adds. Low priority.

## Caveat for framing

What we predict here is **not "virulence" in the LD50 sense** — it is "probability of isolation from a sterile site given gut carriage" = invasion probability. Defensible and clinically relevant, but reviewers will (correctly) push back on calling source-of-isolation a virulence phenotype without acknowledging the indirection. Frame accordingly.

## Running notes

<!-- Agent appends here as work proceeds. -->
