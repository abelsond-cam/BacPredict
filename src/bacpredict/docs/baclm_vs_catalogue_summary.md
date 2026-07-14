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
| rpoC | rifampin | 0.5225 | — | — | — † | — | 0.9666 |

\* **TB baclm is reconstructed**, not measured: `baclm = ESM + Δ(baclm−ESM)`, where Δ is the paired
full-N delta measured in the baclm-vs-ESM learning-curve ladder (job `5583104`). See *Provenance* below.

† **rpoC is a coding gene** (the catalogue CSV records `rpoC,coding,…`, with `embeddable=False`), but it
is **absent from the 1,946-gene ESM screen** — the reason is not determinable from the local files (it is
*not* low prevalence; rpoC is core/essential). So no ESM, hence no reconstructed baclm.
The ladder did emit a Δ of +0.073 for rpoC, **but on a train pool of only n=1129** — implausibly small for
a core gene (rifampin has ~25k labelled). That points to heavy gene-location skipping, so the value is an
**artifact and must not be quoted**. rpoC is a *compensatory* mutation, and its one-hot AUROC of 0.52
(≈ chance) is the expected result.

**Reading it.** baclm ≈ ESM on every TB gene (|Δ| ≤ 0.006 apart from the noisy rpoC) — **baclm's coding
channel carries the same information as ESM-C**. Against the *catalogue*, the embeddings win clearly on
**pncA (+0.062)** — exactly as predicted: pyrazinamide resistance is driven by ~600 mostly-singleton
loss-of-function *pncA* alleles, which a one-hot lookup cannot enumerate but a protein language model can
generalise over. On **rpoB** (rifampin) the catalogue is already near-saturated (0.967, essentially equal
to the whole-catalogue ceiling 0.967) and the embedding is marginally behind.

---

## TB — non-coding (promoter) drivers

baclm's **intergenic** embedding vector (the DNA region immediately 5′ of the flanking gene) vs the
catalogue's promoter one-hot. Measured in the promoter-IGR probe (job `5583276`), same ladder harness.

**Naming:** our probe anchors on the *flanking gene* whose 5′ intergenic region it pulls. `fabG1` is the
gene abutting the **mabA-inhA operon promoter** — the catalogue calls that same locus `inhA (promoter)`.
They are the same region.

| Promoter (locus) | Drug | Catalogue one-hot | baclm IGR | Δ baclm−catalogue | catalogue AUPRC | n genomes w/ variant |
|---|---|---|---|---|---|---|
| fabG1 → *inhA* promoter | ethionamide | 0.8257 | **0.823** | −0.003 | 0.6264 | 1893 |
| fabG1 → *inhA* promoter | isoniazid | 0.6458 | **0.642** | −0.004 | 0.5462 | 3916 |
| eis promoter | kanamycin | 0.6204 | **0.629** | +0.009 | 0.2771 | 816 |
| pncA promoter | pyrazinamide | 0.5316 | **0.560** | +0.028 | 0.2019 | 154 |

**Reading it.** On all four loci **baclm's non-coding vector matches the WHO promoter one-hot to within
±0.03 — without being told which variants exist.** The ethionamide result (0.823) is the headline: a
genuine non-coding AMR signal (inhA over-expression via the operon promoter), recovered from raw
intergenic DNA. Note the *ranking* is also reproduced: ethionamide ≫ isoniazid > eis/kanamycin > pncA/PZA.
The weak pncA-promoter number is *correct biology* — pyrazinamide resistance is driven by *pncA*
**gene-body** loss-of-function (coding pncA ≈ 0.88), not its promoter (only 154 carriers).

Build audit for these loci was clean: 100% / 99% CDS-flanked, 0% RNA-abutting, 0% truncated — so these
IGR vectors are trustworthy in the *current* baclm build (they do not depend on the pending re-embed).

### Not yet available — rRNA drivers (blocked on the non-coding re-embed)

The rRNA bodies were never embedded (the old build took only gaps *between* features, so `rrs`/`rrl`
fell into no store). These rows are pending the 2d re-embed:

| Locus | Drug | Catalogue one-hot | baclm |
|---|---|---|---|
| rrs (16S) | kanamycin | 0.7782 | pending re-embed |
| rrs (16S) | streptomycin | 0.5863 | pending re-embed |

Other non-coding rows in the catalogue we have not probed: `embA (promoter)` / ethambutol 0.5617,
`ethA (promoter)` / ethionamide 0.5181, `ahpC (promoter)` / isoniazid 0.5096 — all ≈ chance.

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
5. **The promoter table mixes protocols too.** baclm IGR AUROCs come from the ladder harness; the
   catalogue promoter one-hot from the TB-Profiler k-fold probe. The agreement (±0.03 on all four) is
   therefore *indicative* rather than a controlled head-to-head — but it is close enough, and consistent
   in rank order, to be a real signal.
6. **Not yet included:** Bacformer (GPU sweep queued), and the **rRNA** drivers (`rrs`, `rrl`), which need
   the baclm non-coding re-embed. Those cells are intentionally absent rather than guessed.
7. **rpoC must not be quoted** (see † above) — coding, but no ESM row and an artifactual ladder Δ.

**What is safe to state to the team.**
(a) baclm's **coding** channel matches ESM-C in *both* species — validated (|Δ| ≤ 0.008).
(b) baclm's **non-coding** channel reproduces the WHO promoter catalogue to within ±0.03 on all four
probed promoters, from raw intergenic DNA — including a genuine non-coding hit, **fabG1/inhA promoter →
ethionamide 0.823**.
(c) Both embeddings substantially **beat the catalogue where resistance is allelically diverse**
(TB pncA +0.062; Kp gyrA/parC +0.045–0.049).
(d) Where the catalogue is already saturated (TB rpoB, one-hot 0.967 ≈ the whole-catalogue ceiling), the
embedding gives no gain.
(e) The full-catalogue ceiling still exceeds any single gene, as it should.

## Source files

- TB one-hot: `src/bacpredict/visualisations/tb/<drug>/tbprofiler_gene_lr_<drug>.csv`
- TB ESM: `src/bacpredict/visualisations/tb/<drug>/per_gene_lr_<drug>.csv` (`lr_auroc_<drug>`)
  — note rifampicin's folder uses the US stem `rifampin` in filenames.
- Kp one-hot: `src/bacpredict/visualisations/kp/ciprofloxacin/card_determinant_lr_ciprofloxacin_family.csv`
- Kp ESM screen: `.../card_esm_vs_ft_per_gene_ciprofloxacin_family.csv`
- Kp baclm+ESM ladder (measured): job `5595657` →
  `…/train_kleb_ast/pangena_predict/coding_amr_lr/ladder_kp_5595657.json`
- TB baclm ladder (deltas only, absolutes pending): job `5583104`
