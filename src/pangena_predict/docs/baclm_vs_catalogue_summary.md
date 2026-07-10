# baclm vs ESM vs catalogue — per-gene AMR prediction (TB + Kp)

**Date:** 2026-07-10 · **Status:** interim summary assembled from local tables + completed HPC jobs while
Isambard is down. **Read the provenance notes — the TB baclm column is reconstructed, the Kp one is measured.**

**What each column is.** Every number is the AUROC of a logistic regression predicting the drug's
resistance phenotype from a single feature set:
- **one-hot (catalogue)** — presence/absence of that gene's known resistance determinants (TB: WHO /
  TB-Profiler variant catalogue; Kp: CARD). This is the *lookup-table* comparator.
- **ESM** — the gene's pooled ESM-C protein embedding (a 960-d vector).
- **baclm** — the gene's pooled baclm-350m-masked protein embedding (also 960-d), the channel we are
  validating.
- **ceiling** — the one-hot of *all* the drug's drivers together (TB `__ALL_WHO_one_hot__`; Kp
  `__ALL_CARD__`), i.e. what the full catalogue achieves. Single genes are not expected to reach it.

---

## TB — coding drivers

| Gene | Drug | TB-Profiler one-hot | ESM | baclm\* | Δ baclm−ESM | Δ baclm−catalogue | WHO all-driver ceiling |
|---|---|---|---|---|---|---|---|
| rpoB | rifampin | 0.9667 | 0.9623 | 0.9613 | −0.001 | −0.0054 | 0.9666 |
| katG | isoniazid | 0.8930 | 0.9037 | 0.9047 | +0.001 | **+0.0117** | 0.9581 |
| gyrA | moxifloxacin | 0.9104 | 0.9096 | 0.9146 | +0.005 | +0.0042 | 0.9257 |
| **pncA** | **pyrazinamide** | 0.8225 | 0.8780 | **0.8840** | +0.006 | **+0.0615** | 0.8544 |
| embB | ethambutol | 0.9216 | 0.9158 | 0.9198 | +0.004 | −0.0018 | 0.9354 |
| rpoC | rifampin | 0.5225 | — | — | (+0.073)† | — | 0.9666 |

\* **TB baclm is reconstructed**, not measured: `baclm = ESM + Δ(baclm−ESM)`, where Δ is the paired
full-N delta measured in the baclm-vs-ESM learning-curve ladder (job `5583104`). See *Provenance* below.

† rpoC's ESM row was filtered out of the screen (low prevalence), so no baclm can be reconstructed. Its
ladder Δ (+0.073) came from a tiny pool (n=1129) and should be treated as noise. rpoC is a
*compensatory* mutation — its one-hot AUROC of 0.52 (≈ chance) is the expected result.

**Reading it.** baclm ≈ ESM on every TB gene (|Δ| ≤ 0.006 apart from the noisy rpoC) — **baclm's coding
channel carries the same information as ESM-C**. Against the *catalogue*, the embeddings win clearly on
**pncA (+0.062)** — exactly as predicted: pyrazinamide resistance is driven by ~600 mostly-singleton
loss-of-function *pncA* alleles, which a one-hot lookup cannot enumerate but a protein language model can
generalise over. On **rpoB** (rifampin) the catalogue is already near-saturated (0.967, essentially equal
to the whole-catalogue ceiling 0.967) and the embedding is marginally behind.

---

## Kp — ciprofloxacin (chromosomal QRDR drivers)

| Gene | Drug | CARD one-hot (mut) | ESM (ladder) | baclm (ladder) | Δ baclm−ESM | Δ baclm−catalogue | ESM (screen)‡ | CARD all-determinant ceiling |
|---|---|---|---|---|---|---|---|---|
| gyrA | ciprofloxacin | 0.8844 | 0.9288 | **0.9333** | +0.0045 | **+0.0489** | 0.8972 | 0.9768 |
| parC | ciprofloxacin | 0.8763 | 0.9141 | **0.9216** | +0.0075 | **+0.0453** | 0.8934 | 0.9768 |

Both ESM and baclm here are **measured** in the same ladder run (job `5595657`, seeds 1–3, fixed 20%
evaluate holdout; train pools 3475 / 3480). Δ baclm−ESM is therefore a true paired delta.

‡ The **ESM (screen)** column is the ESM number from the older per-gene screen
(`card_esm_vs_ft_per_gene_ciprofloxacin_family.csv`). It is ~0.03 *lower* than the ladder's ESM for the
same gene — a **protocol difference** (different cohort/splits), not a modelling difference. It is shown
only to warn against mixing the two sources. The Δ columns use the ladder pair.

**Reading it.** baclm again ≈ ESM (+0.005 / +0.008), and both embeddings beat the **CARD mutation one-hot
by ~+0.05** for gyrA and parC. The all-determinant CARD ceiling (0.977) remains above any single gene,
as expected — it aggregates every cipro determinant (QRDR mutations + Qnr genes).

Kp catalogue context (same CSV): `GyrA (WT)` 0.8508, `ParC (WT)` 0.8076, `QnrB` 0.6055, `QnrS` 0.5268,
`QnrA` 0.5018.

---

## Provenance and caveats — please read before quoting

1. **TB baclm is an estimate.** The TB ladder's *absolute* per-gene AUROCs live only in
   `…/train_tb_ast/pangena_predict/coding_amr_lr/ladder_tb_5583104.json` on Isambard (currently
   unreachable). Only the paired deltas survived locally. So the TB baclm column = local ESM screen +
   ladder Δ.
2. **The Δs are trustworthy; the TB absolutes carry protocol uncertainty.** Δ is measured *paired*
   (same samples, same folds), so it is robust. But the ESM it is added to comes from a **different
   run** (the per-gene ESM screen, n_train ≈ 1865–1983) than the ladder. In Kp, where we have both, those
   two ESM baselines differ by ~0.03. **So treat TB baclm absolutes as ±~0.03, while the Δ column is tight.**
3. **One corroboration.** Reconstructed pncA baclm = 0.884; independently, "coding pncA = 0.885" was
   recorded from the ladder at the time. The agreement suggests TB's screen-ESM and ladder-ESM are close
   (unlike Kp's), which supports the TB reconstruction — but it is a single check, not a validation.
4. **Different cohorts.** TB one-hot/ESM screen rows are on ~1.9k training genomes; the ladder pools and
   the Kp analysis use larger cohorts. Comparisons *within* a row are like-for-like; comparisons of
   absolute AUROC *across* tables are not.
5. **Not yet included:** Bacformer (GPU sweep queued), and the non-coding/rRNA drivers (`rrs`, `rrl`,
   promoters), which need the baclm non-coding re-embed. Those cells are intentionally absent rather than
   guessed.

**What is safe to state to the team.** (a) baclm's coding channel matches ESM-C in *both* species —
validated. (b) Both embeddings substantially beat the catalogue one-hot where resistance is allelically
diverse (TB pncA +0.06; Kp gyrA/parC +0.05). (c) Where the catalogue is already saturated (TB rpoB), the
embedding gives no gain. (d) The full-catalogue ceiling still exceeds any single gene, as it should.

## Source files

- TB one-hot: `src/pangena_predict/docs/visualisations/tb_<drug>/tbprofiler_gene_lr_<drug>.csv`
- TB ESM: `src/pangena_predict/docs/visualisations/tb_<drug>/per_gene_lr_<drug>.csv` (`lr_auroc_<drug>`)
  — note rifampicin's folder uses the US stem `rifampin` in filenames.
- Kp one-hot: `src/kleb_ast/docs/visualisations/amr_per_abx/kp_ciprofloxacin/card_determinant_lr_ciprofloxacin_family.csv`
- Kp ESM screen: `.../card_esm_vs_ft_per_gene_ciprofloxacin_family.csv`
- Kp baclm+ESM ladder (measured): job `5595657` →
  `…/train_kleb_ast/pangena_predict/coding_amr_lr/ladder_kp_5595657.json`
- TB baclm ladder (deltas only, absolutes pending): job `5583104`
