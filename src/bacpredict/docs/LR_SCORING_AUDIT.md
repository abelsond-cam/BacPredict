# LR scoring audit — what every AMR figure actually computes

> ## ⚠ The reasoning is sound; the numbers in §"The redundancy proof" are stale — 2026-08-14
>
> **Correction to an earlier banner on this file.** A previous version of this note claimed the
> document had been "re-checked in full against the artifacts; nothing needed changing." **That was
> wrong.** The *code paths* were re-checked; the *numbers* were not, and an independent review then
> found they do not match the current ladder artifacts.
>
> **What this document gets right, and it is the important part:** the *explanations*. Rung-1
> `ft_mean` is a fresh L2 logistic regression re-probing the frozen FT genome-mean — not the deployed
> head's logit — so a small head-vs-mean gap is expected rather than a regression. A near-zero
> gene/IGR lift is **genuine redundancy** with the FT genome-mean. The equal-width-block and
> zero-imputation arguments hold. That is what this file is for, and it stands.
>
> **What is stale:** every figure in the redundancy-proof table below, and two of its three block
> identities. Against the current cluster artifacts
> (`…/train_tb_ast/pangena_predict/amr_ladder/<drug>/<drug>_amr_ladder_table.csv`):
>
> | drug | table says rung-1 | artifact | table says +gene | artifact | table says +non-coding | artifact |
> |---|--:|--:|---|---|---|---|
> | isoniazid | 0.940 | **0.8944** | +0.028 (katG) | **+0.0536** (katG) | −0.004 | −0.0041 |
> | ethionamide | 0.863 | **0.8151** | −0.011 (rpoB) | **+0.0194** (rpoB) | +0.057 (fabG1) | **+0.0933** |
> | kanamycin | 0.894 | **0.8587** | +0.002 (rpoB) | **−0.0054 (embB)** | +0.022 (rrs rRNA) | **+0.0426 (`mura→ogt`)** |
>
> **The kanamycin row matters most and is the most wrong.** It is cited as the rRNA positive control,
> and the current artifact did not select an rRNA block at all. Do not quote it as evidence that the
> rRNA channel works.
>
> **The direction of the argument survives, and on the current numbers it survives more strongly** —
> the non-redundant lifts are *larger* than stated (katG +0.054, fabG1 +0.093), which is what the
> redundancy claim needs. But do not quote a figure from the table below. Regenerate it from the
> artifacts; the cluster tables are current (post-leak-fix), it is the checked-in mirror that is not.
>
> Numbers of record: [`PROJECT_STATE.md`](../../../PROJECT_STATE.md) §3.1.

This is the reference for the sample scope, imputation, feature, split, and metric behind **each number** in
the Kp/TB AMR figures (ladder, catalogue comparison panel, causal comparison, per-protein LR, IGR). It exists
so we never re-litigate "is this comparison fair?" — it was written after a full code audit prompted by two
surprising results (a flat ladder and a "terrible" catalogue-panel baclm bar), both of which the audit
explained.

## The one-line summary

- **The ladder is correct.** Rung-1 `ft_mean` is a *fresh L2 logistic regression re-probing the frozen FT
  genome-mean* (not the deployed head's logit), so `ft_mean ≈ head − 0.01…0.03` is the expected head-vs-mean
  gap, **not** a regression. The near-zero gene/IGR lift is **genuine redundancy** — the fine-tuned
  genome-mean already encodes the determinant — proven by the *same* pipeline delivering large lift on
  non-redundant blocks (TB positive controls, below).
- **The catalogue comparison panel mis-compared** (now relabelled). Its one-hot bar is presence over **all**
  samples; its baclm bar is the embedding over **carriers only**. For a penetrant HGT gene the carriers are
  ~all-resistant → single-class → the fit collapses → the baclm bar goes blank/low. That is an artifact of the
  scope mismatch, **not** the model failing to read the gene. The bars answer *different questions* and are
  labelled as such.
- **Imputed LR AUROC tracks penetrance.** Zero-imputing non-carriers makes the AUROC depend on
  `P(resistant | carrier)`: high-penetrance acquired gene → high (recovers the one-hot); low-penetrance → ~0.5.

## The scope/imputation/feature/split/metric table

| Number (figure) | sample scope | imputation | feature | fit → score | metric shown |
|---|---|---|---|---|---|
| **CARD/WHO one-hot & ceiling** (`card_determinant_lr` → `ref_catalogues.base.score_onehot_frame`) | ALL cohort samples | non-carriers are genuine 0-rows | presence indicator 0/1 | deployment **train → holdout** | `eval_auroc` (holdout) |
| **Ladder rung-1 `ft_mean`** (`concat/build_amr_ladder._score`) | ALL cohort samples | n/a | FT genome-mean, 960-d, **frozen** (re-probed by a fresh L2 LR — not the head) | deployment **train → holdout** | `eval_auroc` (holdout) |
| **Ladder rungs 2–4** (`+gene / +IGR / +both`) | ALL cohort samples | gene/IGR block **zero-imputed** for non-carriers (`concat_ingredients.impute_block`) | `hstack(mean_960, block_960)`, one `StandardScaler` over the whole concat | deployment **train → holdout** | `eval_auroc` (holdout); `lift_vs_ft` = rung − rung1 on the same holdout |
| **Per-gene / IGR ranking — imputed** (`per_segment_lr --impute-absent-zero`) | ALL read genomes | non-carriers = full zero-vector (presence implicit via the shared scaler) | 960-d embedding | OOF-train / deployment holdout | `lr_auroc_<drug>` (OOF **train**) **and** `eval_auroc_<drug>` (**holdout**) |
| **Per-gene / IGR ranking — carrier-only** (`per_segment_lr`, default) | CARRIERS only (drop-absent) | none | 960-d embedding | OOF-train / deployment holdout | same two columns |
| **Catalogue comparison panel — baclm bar** (`plot_catalogue_vs_embeddings._score_gene` → `run_kfold_probe`) | **CARRIERS only** (absent **and** multi-copy dropped) | none | 960-d embedding, **no presence channel** | internal 20% *carrier* holdout, 5×3 = 15 runs | k-fold mean AUROC |
| **Catalogue comparison panel — one-hot bar** | ALL cohort samples | non-carriers 0-rows | presence 0/1 | deployment train → holdout | `eval_auroc` (holdout) |

Key: `lr_auroc_<drug>` is the **out-of-fold TRAIN** AUROC (used for *selection*, leakage-free);
`eval_auroc_<drug>` is the **deployment HOLDOUT** AUROC (the honest deployable number).

## What the figures display (post-correction)

- **Ladder** — holdout (`eval_auroc`), unchanged.
- **CARD determinant histogram** — the one-hot ceiling (holdout), unchanged.
- **causal_comparison** — two panels (top: LR on all genomes / imputed; bottom: LR on carriers only). Bars now
  show the **deployment holdout** (`eval_auroc_`); regions are still **selected** by `lr_auroc_` (train-OOF,
  leakage-free). The ◆ (rung-2 coding gene) and ★ (rung-3 non-coding region) mark the blocks the ladder
  actually routed.
- **Per-protein LR** and **IGR (per-IGR / upstream / per-unit)** — bars show the **holdout** (`eval_auroc_`);
  ordering/selection by `lr_auroc_` (train-OOF).
- **Catalogue comparison panel** — one-hot (presence, all samples) vs baclm (embedding, **carriers only**).
  These answer **different questions** and are **not** a like-for-like comparison; a blank baclm bar means the
  determinant's carriers are ~all-resistant (single-class → unscoreable), not that the model can't read it.

## The redundancy proof (why zero ladder lift is real, not a bug)

The identical ladder pipeline yields large positive lift whenever the appended block is non-redundant with the
FT genome-mean — so a flat lift is a statement about redundancy, not a broken concat:

| drug | rung-1 ft_mean | +gene lift | +non-coding lift |
|---|--:|--:|--:|
| isoniazid | 0.940 | **+0.028** (katG) | −0.004 |
| ethionamide | 0.863 | −0.011 (rpoB) | **+0.057** (fabG1 promoter) |
| kanamycin | 0.894 | +0.002 (rpoB) | **+0.022** (rrs rRNA) |
| Kp cephalosporins | 0.95–0.99 | ≈0 | ≈0 |

Equal-width blocks (960 + 960) under a single `StandardScaler` rule out dimensional swamping and magnitude
domination; zero-imputation preserves presence/absence through scaling. So the Kp ≈0 lifts are the FT
genome-mean already carrying the determinant signal.

## Known secondary caveats (documented, low impact)

1. Ladder `n_carriers` is the **whole-cohort** carrier count, not the holdout carrier count; a block carried
   only in the train split would give exactly 0 holdout lift by construction while still reporting a large
   `n_carriers`. Moot for the core genes selected here, real for accessory blocks.
2. The appended ladder block shares the FT-mean's global `C=1.0` L2 shrinkage, so a weakly-informative block's
   coefficients are shrunk (can push a marginal block to slightly-negative lift).
3. The ladder ceiling "same holdout" holds by construction **iff** the catalogue CSV was built from the same
   `<drug>_split.csv` — convention, not an enforced cross-check.
4. `_best_from_ranking` has a latent `eval_auroc_`-selection fallback that only triggers if a ranking has **no**
   non-NaN `lr_auroc_` column (never in normal operation).
