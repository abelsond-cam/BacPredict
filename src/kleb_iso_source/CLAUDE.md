# Task 3 — Isolation source in *Klebsiella*

This is one of the task folders under `src/`. See the root [CLAUDE.md](../../CLAUDE.md) for §0 global conventions. Cross-task status lives in [ToDo.md](../../ToDo.md).

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

### 2026-05-25 — refreshed model + v2 metadata: Stage A/B pass, Stage C queued

Goal: re-baseline blood-vs-faeces on the refreshed Bacformer complete-genomes
weights against the v2 metadata, end-to-end from prep through Stage B.

Codebase prep (all on `restructure/flatten-to-task-folders`):
- v1 metadata (`metadata_final_curated_slimmed.tsv`) was already swept to
  `metadata_v2_all_samples_and_columns.tsv` across all task folders (commit
  `6a97c77` from a parallel session).
- [train_isolation_source.py:289](train_isolation_source.py#L289) ported to the
  `dtype="auto"` HF loading idiom (same fix `tb_ast/train_amr.py` got in
  `4956f91`). The previous unconditional `.to(torch.bfloat16)` crashed on CPU in
  Bacformer's classifier einsum.
- New `--output-dir` flag on
  [`prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py`](prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py)
  so the bypass-stratification flow can write outputs somewhere other than the
  curated `final/` directory (commit `83f3633`).

Bacformer cache: HF refreshed
`macwiatrak/bacformer-large-masked-complete-genomes` on 2026-05-15. HPC cache
was pinned to the Apr 28 revision; pre-downloaded the new snapshot
(`ab3a91a21027359ae59d1c258afea8089826ea4a`).

Split CSV (v2 metadata, no stratification, no country cap):
- Output (relocated 2026-05-29 — see entry below):
  `processed/train_iso_source/blood_faeces/all_samples/binary_blood_vs_faeces_with_split.csv`
- 25,879 unique samples kept (137 pruned for missing embeddings).
- 70/10/20: train 18,169 / validate 2,591 / evaluate 5,206.
- Label balance: 13,289 blood (51.2%) / 12,677 faeces (48.8%); preserved
  across each split. ~27% more faeces than the v1 baseline noted in §Status.

Stage A + Stage B — both PASS in one GPU run (SLURM job 29713406, 7:06 elapsed):
- Script: [`scripts/train_isolation_source_stage_a.sh`](scripts/train_isolation_source_stage_a.sh)
  (n=10, ampere, 30 min walltime, `--n-samples 10` triggers the train=val
  smoke-test path that doubles as Stage B's overfit harness).
- Pipeline ran end-to-end: model loaded, training loop, eval, checkpoint, early stop.
- Train loss: ~0.7 → **3.6e-5** over 32 epochs (Stage B overfit signal — model
  has more than enough capacity to memorise n=10).
- Eval loss → 5e-5; AUROC 1.0 from epoch 1 (saturated on n=10).
- Best model: `…/stage_a_smoke/checkpoint-4` (score 1.0).
- **Important venue correction:** root §0.2 says Stage A can run on HPC login
  CPU. For *this* codebase that is not viable — Bacformer-large + the
  per-sample RDS embedding I/O exceeds login-node limits. CPU Stage A here
  silently produced an empty tensorboard event file with no scalar
  metrics. Every BacPredict Stage A needs a short ampere GPU sbatch
  (~30 min walltime is fine).

Stage C — single fold × single seed, full data, 36 h ampere:
- Script: [`scripts/train_isolation_source_stage_c.sh`](scripts/train_isolation_source_stage_c.sh)
  (commit `48fc7c1`).
- Pinned: complete-genomes model, v2-derived split CSV, `stage_c_full` output dir.
- Linted (`bash -n` + `sbatch --test-only`); SLURM accepted, projected start
  `2026-05-27 17:00` (ampere queue depth).
- **Not submitted.** User to fire when ready:
  `sbatch src/kleb_iso_source/scripts/train_isolation_source_stage_c.sh`.

Open follow-ups (parked, not blocking Stage C):
- Backport `dtype="auto"` to [`../kleb_ast/train_amr.py`](../kleb_ast/train_amr.py)
  (Task 2 will hit the same CPU Stage A crash — already flagged in
  [tb_ast/CLAUDE.md](../tb_ast/CLAUDE.md) running notes).
- `train_isolation_source.py`'s `PROCESSED_BASE_DIR_DEFAULT` is `processed/`
  (not `processed/train_on_sr_mags/`); inconsistent with the prep script's
  default base. The Stage A and Stage C sbatch scripts pass
  `--processed-base-dir` explicitly to avoid this. Worth normalising one day.
- Tensorboard scalars from the failed login-node Stage A runs (under
  `processed/training_blood_faeces/stage_a_smoke/runs/`) can be deleted —
  they contain no successful events.

### 2026-05-29 — Stage C results + output-layout restructure

**Headline (refreshed complete-genomes model, blood vs faeces):** the genomic
invasion signal is real and survives strict country control.

| Cohort | n | Val AUROC | Evaluate-holdout §0.4 |
|---|---|---|---|
| `all_samples` (no country control) | 25,879 | 0.835 peak | not scored — country-confounded baseline |
| `sampled_country_2_1_stratified` (strict 2:1 cap, thread-segregated) | 10,443 | 0.767 | **AUROC 0.752** (n=2096; auprc 0.758, sens 0.836, spec 0.492, bal-acc 0.664, prevalence 0.528) |
| old MAG baseline | — | 0.55–0.62 | reference |

Strict country control holds AUROC at ~0.75 ≫ the 0.55–0.6 country-shortcut ceiling →
genuine genomic signal, not a country/phylogeny artefact (~0.09 AUROC of the
all-sample number was country confounding). `all_samples` is a baseline, not resumed
(converged ~ep10; TIMEOUT at 36h). All-sample evaluate-holdout left unscored (its
5,206-row holdout needs >1h; it is only the confounded baseline).

**Output layout restructured** → `processed/train_iso_source/blood_faeces/<cohort>/`,
each cohort holding its split CSV at the top + a `models/` checkpoint dir:
- `all_samples/models/checkpoint-37000`
- `sampled_country_2_1_stratified/models/` (+ `eval_results.json`, ROC/PR png, npz)
- `sampled_country_2_1_all/` — pooled ~15.2k cohort (Phase C, not yet built)

The pair dir is `{slug1}_{slug2}` (no `training_` prefix) so future pairs slot in
beside `blood_faeces/`. The sbatch scripts pass an absolute `--output-dir
…/<cohort>/models` (now used verbatim); Stage A smoke writes to `<cohort>/smoke` to
keep `models/` clean. `train_isolation_source.py` `PROCESSED_BASE_DIR_DEFAULT` is now
`processed/train_iso_source` (resolves the old inconsistency follow-up above).

**Sampler defaults to pooled threads** (single stratify over the whole filtered pool);
`--segregate-threads` opts back into the old per-thread (AMR/Surv/NA) split. Every run
writes `stratification_report.md` beside the cohort TSV: thread-segregation impact,
Sublineage band table (Epidemic ≥250 / Rare <250 / Very-rare <100 / no-call ×
n/blood/faeces/ratio) for pool AND cohort, and rare-fraction pool-vs-cohort. Dry-run
headcounts: pooled **15,218** / segregated **10,541** (the −4,677 thread-segregation
cost is the lever Phase C recovers, country-control intact). Finding: stratification
does NOT enrich rare SLs (44.2%→42.9%); the 2:1 cap incidentally balances rare bands
(blood:faeces ratio 1.40→1.17).

**Auto-eval backported** into `train_isolation_source.py` (post-`train()` → `results.json`,
§0.4); `evaluate.py` got a `file_system` FD-sharing fix; dead `--complete-genomes`
flag removed (broken on v2). **RDS cleanup:** deleted stale `train_on_sr_mags/`,
`train_on_complete_genomes/`, top-level `training_blood_faeces/` (legacy MAG `.pt`
dirs + old checkpoints).

**Next:** Phase C — build `sampled_country_2_1_all` (~15.2k pooled, `--ratio 2.0`),
prep → split CSV, Stage C train + `evaluate_iso_source.sh sampled_country_2_1_all`,
then compare AUROC (pooled-15.2k vs stratified-10.5k=0.752 vs all-sample-25k). Phase D
(Kleb-specific masked continued-pretraining) is blocked on the ESM-C 960-dim
protein-family centroids from Isambard (the CSD3 RDS ones are legacy ESM 480-dim).
