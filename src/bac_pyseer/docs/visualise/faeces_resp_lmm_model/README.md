# faeces vs respiratory — LMM GWAS (replication contrast)

Second invasive-niche contrast, run to test whether the blood-vs-faeces invasion signal
recurs. Same pipeline as blood/faeces (parameterised: `PAIR=faeces_respiratory`,
`LABEL_COL=respiratory_vs_faeces_label`), same LMM method, same VE-ranked postprocess.

- Cohort `sampled_country_2_1_all` — **n = 9,169** (faeces 4,737 / respiratory 4,432, ratio
  1.07), country-balanced 2:1, pooled threads. Tokens: faeces = *"faeces & rectal swabs"*,
  respiratory = *"lower respiratory, endotracheal"*. **faeces = 0, respiratory = 1**, so
  β>0 = the respiratory (invasive) side, mirroring blood=1.
- LMM (kinship from `similarity_pyseer`), big-SL `--lineage` (≥100-sample SLs), af 1–99%.
  Job 30710175 (chain 30710172–75).

## Calibration

- **Genomic inflation λ = 0.498** — same conservative regime as blood/faeces (λ=0.562).
  The method behaves identically on this clonal cohort: controlled bulk, clean tail.
- **88 significant hits** (Bonferroni 1.76e-7 over 284,368 patterns; 326,146 variants);
  63 in-gene, 2 virulence-flagged. Ranked by variance explained (see `../lmm_model/` for
  the methodology).

## The replication test — capsule + fimbrial adhesion recur; iron does not

`cross_contrast_overlap_blood_vs_resp.tsv` lists the **33 genes significant in *both*
contrasts**. Crucially, **both contrasts share faeces as the control**, so a gene that is
*faeces-associated in both* recurs **trivially** — it marks a gut-resident sub-clade
(the SL258/SL307 clonal blocks) over-represented in faeces vs any non-gut niche. **31 of
the 33 are this** (β<0 in both) and are *not* informative replication.

**The informative test is invasion-direction recurrence (β>0 in both). Exactly two genes
pass:**

| gene | blood β | resp β | blood VE | resp VE | lineage |
|---|---|---|---|---|---|
| **capsule assembly Wzi** (`KPN_RS13515`) | +0.27 | +0.25 | 1.0% | 1.0% | cross-lineage |
| **fimbria/pilus usher** (`KPN_RS24485`) | +0.20 | +0.24 | 3.2% | **5.0%** | cross-lineage |

Both are the **same SNP**, same direction, comparable-or-larger effect, surfacing
independently against **two different invasive niches** (blood, respiratory) with gut as
baseline. A reproducible, cross-lineage **adhesion + capsule invasion signature**.

**The blood-vs-faeces iron theme does NOT replicate:** iron-cofactor redox enzyme, nadB,
phoA, dnaK, and the TonB-dependent siderophore receptor are **absent** from the faeces-resp
hits; btuB appears but with the *opposite* (faeces) sign. So that core-genome iron signal
was largely **blood-specific or weak**, not a general invasion determinant.

(Reproducibility across niches outweighs single-contrast effect size: *wzi* is modest in VE
but robust, which is the stronger evidence of a real signal.)

## Caveats (unchanged from the core-genome design)

- This is the **chromosomal / core-allele axis** — accessory determinants (capsule *type*,
  aerobactin, yersiniabactin) are invisible. The unitig GWAS is the accessory/HGT test.
- Effect sizes are small; these are leads, not classifiers.

## Files

- `manhattan_resp_faeces_lmm.png` — all significant hits sized by variance explained,
  top hits labelled by gene/product name (MAF>5%); `↑`/`↓` = allele direction.
- `qq_resp_faeces_lmm.png` — QQ (λ=0.498).
- `respiratory_vs_faeces_hits_annotated.tsv` — the 88 hits, VE-ranked, gene-mapped.
- `respiratory_vs_faeces_gwas_summary.json` — λ, thresholds, counts.
- `cross_contrast_overlap_blood_vs_resp.tsv` — the 33 genes significant in both contrasts,
  with both betas/VEs and a `concordant_invasion` flag (the 2 that replicate up).

## Reproduce

```
PAIR=faeces_respiratory COHORT=sampled_country_2_1_all LABEL_COL=respiratory_vs_faeces_label \
  OUT_STEM=respiratory_vs_faeces POS_LABEL='respiratory (invasion)' PAIR_TITLE='faeces vs respiratory' \
  COHORT_CSV=.../faeces_respiratory/sampled_country_2_1_all/kpsc_human/binary_respiratory_vs_faeces_labels.csv \
  USE_LMM=1 sbatch --time=24:00:00 src/bac_pyseer/kleb_iso_source/scripts/run_pyseer.sh 10 gwas_lmm 100
```
