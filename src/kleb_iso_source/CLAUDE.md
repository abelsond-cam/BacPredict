# Task 3 — Isolation source in *Klebsiella*

This is one of the task folders under `src/`. See the root [CLAUDE.md](../../CLAUDE.md) for §0 global conventions. Cross-task status lives in [PROJECT_STATE.md](../../PROJECT_STATE.md).

## Aim

Test whether Bacformer fine-tuned via BacPredict can pick up genomic markers that distinguish invasive from non-invasive infection. The canonical comparison is **blood vs stool** (the field has converged on this — cleanest two-source comparison: largest n, least ambiguous microbiology, strongest biological interpretation). Liver abscess vs stool is an even stronger hvKp signal where it can be text-mined from metadata, even at modest n.

## ⚠ Precision — every published number here is fp32 (2026-08-11)

`train_isolation_source.py` loaded the model with `dtype="auto"`, which resolves to
**fp32** master weights for Bacformer-large. It was changed to an unconditional
`.to(torch.bfloat16)` cast in **`a817ac2` (2026-07-26)** — *after* every result in
this file (all dated 2026-05-29/31). So:

| | |
|---|---|
| Deployed checkpoints (0.786 / 0.762 / 0.827) | **fp32** |
| `train_isolation_source.py` at HEAD | **bf16** |

bf16 beat fp32 by ~7 pp AUROC on TB rifampin in a controlled A/B, so a bf16 re-run
is expected to move these numbers — direction unproven for this phenotype. Re-runs
write to a `models_bf16/` sibling so the fp32 checkpoints survive for the A/B.
`results.json` now records `run_config.precision`, so this can never be ambiguous again.

**Which number to quote.** The defensible headline is **0.786** on
`sampled_country_2_1_all` (country-controlled, n=14,211, eval n=2,822) — and it is the
cohort the pyseer GWAS ran on, so it is the correct comparator throughout. `all_samples`
0.827 is country-**confounded** and sits *below* its own linear metadata baseline (0.857);
do not quote it as a headline.

## Comparator results (2026-08-11) — what the 0.786 is worth

Three independent comparators, all scored on the **same** `sampled_country_2_1_all` holdout
as the deployed model. Cohort n=14,119 (train 9,885 / validate 1,412 / evaluate 2,822).

**1. Unitig GWAS model** — `bac_pyseer.kleb_iso_source.unitig_presence_model`. The 33,039
significant hit unitigs as a genome × unitig presence matrix (13,602 × 33,039, 108,796,553
non-zeros — matches the GWAS placement count exactly), L2 LR with `C` swept on validate.

| | AUROC on the same 2,715 genomes |
|---|---:|
| unitig L2 (C=0.01) | 0.781 |
| **Bacformer** | **0.787** |
| delta (paired bootstrap) | **+0.0055, 95% CI [−0.0110, +0.0230]** |

**The CI spans zero — a statistical tie**, not a Bacformer win. And the unitig model held a
real advantage, since its feature set was selected by an LMM fitted over the whole cohort
*including this holdout*. Bacformer matching a leakage-advantaged accessory-sequence model
is the claim; "beating it" is not. A leakage-free re-run (LMM selection on train+validate
only, cohort `sampled_country_2_1_all_trainval`, n=10,887) is the publication number.
`C` mattered: validate fell 0.775 → 0.728 from C=0.01 to C=10, so the repo's pinned C=1.0
would have understated the comparator by ~2.5 pp. L1 peaked at 0.770 with 817 non-zero
unitigs (the interpretable shortlist — still gene-unannotated).

**2. Per-Sublineage** — `bacpredict.engine.finetune.stratified_metrics` (95% bootstrap CIs):

| Sublineage | n | AUROC | 95% CI |
|---|--:|--:|---|
| pooled | 2,822 | 0.786 | — |
| SL258 | 472 | **0.858** | 0.823–0.890 |
| SL15 | 107 | 0.841 | 0.760–0.913 |
| SL307 | 155 | 0.815 | 0.745–0.878 |
| SL17 | 189 | 0.806 | 0.744–0.866 |
| SL147 | 207 | 0.738 | 0.666–0.809 |
| other (556 rare SLs) | 1,692 | 0.759 | 0.736–0.781 |

Discrimination **holds within every major clone**; four of five sit at or above pooled, SL258
significantly so. Lineage identity is therefore not what the model is reading.
*Hypotheses only, pending David's read:* the signal may be within-clone accessory content, and
pooling across 561 sublineages may cost AUROC (the rare-SL bucket is the weakest stratum). Not
resolved here — do not plan downstream work off this without discussion.

**2b. Whole-cohort scoring → more clones, three scopes** (`score_cohort.py` job `33494112`;
tables via `scripts/stratified_tables_iso_source.sh`). Scoring all 14,119 genomes reproduced the
deployed holdout number exactly (evaluate 0.7858 vs 0.786) and gave train 0.9590 / validate 0.7943.

**That train-vs-evaluate gap is why the whole-set numbers cannot answer the within-clone question
on their own.** The model memorises hard and train is 70% of the cohort, so an all-splits per-clone
AUROC is mostly recall of fitted rows. Hence a third scope, `heldout` = validate + evaluate: nothing
fitted on, n 2,822 → 4,234, and more clones clear n≥100. Quote `evaluate`; read `heldout` for the
within-clone claim; treat `all` as a pattern check only.

| Sublineage | n (heldout) | AUROC | 95% CI | | n (all) | AUROC (all — fitted-on) |
|---|--:|--:|---|---|--:|--:|
| pooled | 4,234 | 0.788 | 0.775–0.801 | | 14,119 | 0.914 |
| SL15 | 167 | **0.890** | 0.840–0.932 | | 536 | 0.949 |
| SL258 | 703 | 0.844 | 0.813–0.872 | | 2,416 | 0.933 |
| SL307 | 247 | 0.828 | 0.778–0.878 | | 796 | 0.922 |
| SL17 | 275 | 0.814 | 0.761–0.863 | | 960 | 0.925 |
| SL147 | 315 | 0.739 | 0.676–0.793 | | 1,022 | 0.866 |
| SL37 | 100 | 0.662 | 0.555–0.768 | | 326 | 0.868 |
| other | 2,427 | 0.762 | 0.743–0.780 | | 5,776 | 0.906 |

At `all` scope **20 sublineages** clear n≥100 and **every one** scores 0.849–0.966 against pooled
0.914 — no clone fails, though these are inflated. By clonal group: 15 CGs at `all` scope (CG258
0.935 … CG17 0.967, CG147 lowest at 0.864), only 4 at `heldout` (CG258 0.847, CG307 0.827,
CG15 0.890, CG147 0.737) and 4 at `evaluate` — clonal groups are too fine-grained for the holdout
alone, which is exactly what the extra scopes were for.

**Held-out discrimination survives inside every clone tested, with real spread: 0.662 (SL37) to
0.890 (SL15).** Four of six sit at or above pooled. SL37 is the weakest and its CI is wide
(n=100) — it is the one group whose interval comes close to the rare-SL bucket. Whether the spread
is biological (clone-specific invasion routes) or an artefact of per-clone prevalence/provenance is
**an open question for David**, not something to build on.

**3. Kleborate annotation** — `linear_baselines`, no country/Sublineage terms:

| Model | n_feat | AUROC | Bacformer margin |
|---|--:|--:|--:|
| virulence_score (total) | 1 | **0.489** | +0.297 |
| virulence factors one-hot | 6 | 0.552 | +0.234 |
| virulence + AMR one-hot | 23 | 0.638 | +0.148 |
| all Kleborate | 27 | 0.640 | +0.146 |
| *AMR classes alone* | 17 | 0.617 | +0.169 |
| country + Sublineage | 1,192 | 0.694 | +0.092 |
| richest linear stack | 1,360 | 0.731 | +0.055 |

**Kleborate's total virulence score is at chance (0.489)** for blood-vs-faeces, and AMR
annotation predicts invasion *better* than virulence annotation does (0.617 vs 0.552) — likely
a healthcare-association confound, not a virulence finding. Reproduces the Jun-2026 ladder
(country+SL 0.694, full stack 0.731) to 3 dp.

Artifacts: `<cohort>/kpsc_human/models/cohort_scores.npz` +
`per_{sublineage,clonal_group}_metrics_{evaluate,heldout,all}.{csv,json,png}`;
`<cohort>/kpsc_human/linear_baselines_v2.json`;
`…/pyseer_iso_source/blood_faeces/sampled_country_2_1_all/gwas_unitig_lmm/presence_model/`.

## Status

> ⚠ **SUPERSEDED — status lives in `PROJECT_STATE.md` §3.2.** Stage C is **done** on all three
> KPSC-clean cohorts; the headline is the country-controlled pooled cohort at **0.786**, not the
> 0.55–0.62 below (that was the old MAG-weights benchmark this work was built to beat, and it did,
> decisively). The design decision below to "use all samples from all countries" was also
> **reversed**: `all_samples` reaches 0.827 but sits *below* its own linear metadata baseline, so it
> is country-confounded and **must not be quoted**. What follows is kept for the labelling notes and
> the reasoning, not the state.

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

## Week of 2026-05-30 — assigned workstream (A)

Anchor: program plan `~/.claude/PROGRAM_PLAN_2026-05-30.md` — Workstream A.
Branch: `task3/iso-source-expansion`.

The 2026-05-29 blood-vs-faeces result (AUROC 0.835 pooled / 0.752 country-
balanced) is the trigger. This workstream widens the proof and characterises
cohorts well enough that country/SL are demonstrably not doing the work.

- **A1 — respiratory vs faeces.** Run sampler + prepare for the
  `respiratory_vs_faeces` token pair in BOTH cohorts (pooled all-samples + 2:1
  country-balanced stratified). Outputs:
  `processed/train_iso_source/respiratory_faeces/<cohort>/`. Sampler + prepare
  are already label-agnostic — only the token pair changes. Reuse existing
  Stage C sbatch scripts (parameterised on cohort subdir, commit 61748cd).
  Gut-check AUROC ≥ 0.65 on the country-balanced cohort.
- **A2 — wound vs faeces. DEFERRED.** Wound is a heterogeneous sum (swabs,
  abscesses, surgical drains) and needs a label-cleanup pass over metadata_v2
  first. Not in scope this week; tracked here so it isn't forgotten.
- **A3 — cohort stratification visuals.** New `stratification_plots.py`:
  - Country side-by-side bar (blood vs faeces per country, top-N + "other",
    sorted desc on blood). One PNG per cohort.
  - Per-country blood:faeces ratio histogram × n_samples (most countries
    should sit near parity).
  - SL stratification — three views: top-15 epidemic SLs (n≥250) bar chart;
    SLs 100–250 bar chart; SLs <100 bubble scatter (n_samples on x,
    blood:faeces ratio on y).
  PNGs committed alongside the cohort dir (`stratification_plots/`); the
  existing `stratification_report.md` embeds them. Reuse
  `_log_final_country_table()` for counts.
- **A4 — baseline AUROCs from metadata alone.** New harness in
  `bacpredict/engine/finetune/linear_baselines.py` (generic; was `src/tl/train/`
  before the engine consolidation) + thin wrapper here. sklearn
  LogisticRegression on (i) one-hot `country_parsed`, (ii) one-hot
  `Sublineage`, (iii) the combination. Fit on the same train/eval split as
  the Bacformer fine-tune. Numbers added to `stratification_report.md` and to
  Stage C `results.json` under a new `baselines` key.
- **A5 — explainability (occlusion + integrated gradients).** New code in
  `bacpredict/engine/explain/` (generic, Captum-based; the old plan said
  `engine/explain/ (does not exist — `) + `explain_iso_source.py` wrapper.
  Run on both the country-confounded (`all_samples`) and country-controlled
  (`stratified`) checkpoints; rank proteins by aggregated importance.
  **Rank-shift between cohorts is the phylogeny-vs-signal filter:** genes
  that lose the most rank-importance under country control are phylogeny-
  confounded; the ones that hold up are the iso-source signal. Cross-
  reference with Kleborate virulence loci (rmpA/rmpA2, iuc, iro, K1/K2,
  ybt/clb). Smoke on n=50 from the evaluate holdout before scaling.
  Once A1 is built, repeat A5 across the 2×2 of {blood-vs-faeces,
  respiratory-vs-faeces} × {non-select, country-balanced} — that's the
  sharpest phylogeny-vs-signal separation.

**Dependency on Workstream B** (complete-genome eval-set surgery): once B2
lands `bias_eval_toward` in `engine/finetune/split_utils.py`, re-run the stratified
Stage C with `--bias-eval-toward is_complete` and compare AUROC delta.

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

Imports from [`../bacpredict/engine/finetune/`](../bacpredict/engine/finetune/) (split_utils, datasets) and [`../bacpredict/engine/embedding/`](../bacpredict/engine/embedding/) for shared infrastructure.

Documentation
- [`docs/iso_source_summary.ipynb`](docs/iso_source_summary.ipynb) — summary notebook for the blood-vs-faeces workstream. Three sampling methods, stratification stats + plots (pooled headline), Bacformer §0.4 + ROC, linear-model baseline (`linear_baselines.json`) comparison, and a Bacformer-vs-linear-baseline overlay ROC. Built by `docs/_build_summary_notebook.py`; pre-rendered figures sit in `docs/figures/`.

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
  `dtype="auto"` HF loading idiom (same fix `engine/finetune/finetune_amr.py` got in
  `4956f91`). The previous unconditional `.to(torch.bfloat16)` crashed on CPU in
  Bacformer's classifier einsum.
  > **⚠ SUPERSEDED — see the 2026-08-11 precision note below.** `dtype="auto"`
  > resolves to **fp32** master weights for this model, which is the *underperforming*
  > setting. It was reverted to an unconditional bf16 cast in `a817ac2` (2026-07-26).
  > Every result dated 2026-05-29/31 in this file was therefore trained in **fp32**.
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
- Script: [`scripts/train_isolation_source_cohort.sh`](scripts/train_isolation_source_cohort.sh),
  parameterised by cohort. It replaced the three `train_isolation_source_stage_c{,_pooled,_stratified}.sh`
  copies, **deleted 2026-08-13**: each hardcoded `output_dir=<cohort>/models`, so re-running one
  would have overwritten the deployed fp32 checkpoints in place.
- Pinned: complete-genomes model, v2-derived split CSV; writes to `models_bf16/` (never `models/`).
- **The checkpoint inventory — which trained models exist, fp32 vs bf16, and which checkpoint to
  evaluate for each cohort — is documented in the header of
  [`scripts/evaluate_iso_source.sh`](scripts/evaluate_iso_source.sh).** Read it before evaluating
  anything; picking the wrong directory yields a plausible-looking number from the wrong model.
- Early stopping: `metric_for_best_model=eval_auroc`, patience **12**. It was 30, which could not
  fire inside a 36 h wall — the 2026-08-11 bf16 runs peaked at steps 31,500 / 30,500 / 15,000 and
  all three were killed by the wall having accrued only 9 / 15 / 23 non-improving evals.
- ⚠ `max_steps=100000` is **not** a mere cap — `warmup_steps = max_steps × warmup_proportion` and the
  LR decays to zero at `max_steps`, so it defines the learning-rate schedule. Lowering it to shorten
  runs makes a different model, not the same one stopped earlier, and breaks comparability with
  every fp32 result. Shorten runs with patience, never with `max_steps`.

Open follow-ups (parked, not blocking Stage C):
  - ⛔ **DELETED — `dtype="auto"` must NOT be backported anywhere.** For Bacformer-large it resolves to
  **fp32**, which is measurably *worse* (~5 pp AUROC on TB rifampin, like-for-like). Every trainer casts
  unconditionally to bf16 and that is correct. The motivation — making a CPU Stage A work — is moot:
  Stage A must be a short **GPU** sbatch (root `CLAUDE.md` §0.2), because a CPU Stage A silently writes
  empty tensorboard events and only looks like it passed.
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

### 2026-05-29 (later) — KPSC-contamination bug + filter fix

Spotted while reviewing the fresh pooled cohort's stratification report (~1,248
"No Sublineage call" samples in the cohort). Root cause: the sampler never
restricted to `kpsc_final_list==True`, and the prepare script's filter funnel
**never applied `host_category=="human"` either** (verified via `git log -S` —
the host filter was never in prepare's history, not removed by a recent edit).

| Cohort | n trained | Non-KPSC in trained split | Null-Sublineage in trained split |
|---|---|---|---|
| `sampled_country_2_1_stratified` (the 0.752 result) | 10,446 | 676 | 709 |
| `all_samples` (25,879) | — | not separately measured; also missing host filter | — |

Of the 1,519 no-call pool samples, 1,433 are non-KPSC and **86 are
`kpsc_final_list==True` but lack a Sublineage** (124 such samples globally;
v2 metadata data-quality, not our bug). User decision: require BOTH
`kpsc_final_list==True` AND `Sublineage.notna()` so the cohort has a zero
no-call band.

**Fix:** added the KPSC + Sublineage filter as Filter 3 in
`stratified_isolation_source_sampling.py` (between host and study_setting), and
added host + KPSC + Sublineage filters inside `filter_and_create_pair_label` in
`prepare_…isolation_source.py` (so both flows produce KPSC-clean, human-only
cohorts). Filter funnel is now logged.

**Rebuild scope (user-approved: all three).** Existing trained artifacts moved
to `<cohort>/mixed_species/` under each cohort dir (kept as a reference; the
0.752 number stands as the mixed-species result for diffing). Then rebuild all
three cohort definitions with the fix, retrain Stage C on each, compare.

### 2026-05-31 — KPSC-clean results + stratification plots

**KPSC-clean retrain results (eval-holdout §0.4 from auto-eval `results.json`):**

| Cohort | n | Eval-holdout AUROC | vs mixed-species reference |
|---|---|---|---|
| `sampled_country_2_1_stratified` | 9,866 | **0.762** (n_eval=1951) | +0.010 vs mixed-species 0.752 — contamination did NOT drive the headline |
| `sampled_country_2_1_all` (pooled) | 14,211 | **0.786** (n_eval=2822) | +0.024 vs stratified — pooling the AMR/Surv/NA threads pulls weight |
| `all_samples` | 21,533 | TIMEOUT @ 36h; best ckpt `checkpoint-31000` (val 0.829 @ ep 16.5) being scored separately | — |

The all_samples resume was **skipped** as overkill (val was already plateauing 0.75–0.82 around ep 20, and all_samples is the country-confounded baseline, not a headline). Scoring the existing `checkpoint-31000` directly via `evaluate_iso_source.sh all_samples kpsc_human checkpoint-31000` instead — sbatch added a 3rd `CKPT_SUBDIR` arg for pinning a specific checkpoint when models/ holds multiple.

**Stratification plots (new artifact).** `stratification_plots.py` writes two paired-bar PNGs per cohort to `<cohort>/<flavor>/stratification_plots/{country.png, sublineage.png}`. Per group: left bar = initial pool n, right bar = accepted cohort n; colored by blood:faeces ratio (`RdBu_r` diverging colormap, log scale, clipped 0.25–4.0, neutral at 1.0). Groups with pool n < cutoff aggregate into a single "other" bar. Cutoffs tunable independently via `--country-min-samples` (default 100) and `--sl-min-samples` (default 100; the rare-SL knob). Standalone CLI replays the sampler's filter pipeline so plots can be backfilled for existing cohorts without re-running the sampler:

```
uv run python -m kleb_iso_source.stratification_plots \
  --cohort-tsv .../stratified_selected_isolation_source_metadata.tsv \
  --isolation-sources blood faeces \
  --out-dir .../stratification_plots
```
