# LMM model — blood-vs-faeces GWAS (the trustworthy result)

Diagnostics from the **linear mixed model** structure correction
(`pyseer --lmm --similarity <kinship>`, FaST-LMM) on the country-balanced
`sampled_country_2_1_all` cohort (n=13,602; 372,238 variants at af 1–99%;
big-SL `--lineage`, 19 Sublineages ≥100 samples kept + `other`; job 30673611,
run `gwas_lmm`). The kinship was built once by `similarity_pyseer` from the
variant Rtab (`similarity.tsv`, 1.4 GB, shared/reused across LMM runs).

**Verdict: LMM is the correct method for this cohort. This is the result we trust.**
It replaces the MDS fixed-effects attempt (`../mds_model/`, λ=4.34, abandoned).

## Why (the numbers)

- **Genomic inflation λ = 0.562.** Below 1 — the bulk is mildly *over*-corrected
  (conservative), the opposite of MDS's λ=4.34 genome-wide inflation. This is the
  expected, benign behaviour of an LMM on highly clonal data: the kinship random
  effect absorbs the polygenic background relatedness, deflating the median test
  statistic. **It does not manufacture false positives** — a deflated null means the
  110 hits that clear Bonferroni are high-confidence; the only risk is *missing* some
  true signal, not inventing it.
- **The QQ shape is textbook-good** (`qq_lmm_bigsl_af1.png`): the bulk tracks at/below
  the null (controlled, no shelf) and then a **clean, well-separated upper tail** of
  genuinely-associated variants shoots up to −log10(p) ≈ 10–28. Controlled bulk +
  distinct signal tail = a healthy bacterial GWAS. Contrast MDS, where the *entire*
  distribution sat above the diagonal.
- **The Manhattan** (`manhattan_lmm_bigsl_af1.png`): a flat genome-wide baseline
  (−log10 p ≈ 1) with discrete peaks above the Bonferroni line scattered across
  NC_009648 — sparse, separated signals, not a carpet of inflation.
- **110 significant hits** (vs 6,657 under MDS) over 353,051 unique patterns,
  Bonferroni 0.05/patterns = 1.42e-7. 88 fall inside a gene; 2 are virulence-flagged.

## Ranking — effect size, not direction or raw significance

**Direction (β sign) is not a meaningful ranking axis.** A presence/absence variant is
symmetric: a "faeces" hit at af 20% means the *absence* allele (80%) carries the blood
signal — same information, coded by which allele is the reference. The table is therefore
ranked by **variance explained ≈ f(1−f)·β²** (`var_explained_pct`), the direction-agnostic
effect size; `direction` is kept only as an attribute. (pyseer's built-in `variant_h2` is
*not* used for ranking — it weights the LMM genetic-variance contribution and puts the rare
lineage-private markers on top, the opposite of what we want.)

**Effect sizes are uniformly small.** Median variance explained per pattern ≈ 1.3%, max
≈ 8%. No single core variant predicts invasion; most invasive isolates carry the reference
allele at any given hit. Expected: invasiveness is polygenic *and* driven heavily by
**accessory** determinants — capsule **type**, aerobactin (*iuc*), yersiniabactin (*ybt*),
all mobile/accessory and **invisible to this core-genome scan**. These hits are individual
leads on the chromosomal axis, not a classifier.

**Clonal blocks.** The 110 significant variants are only **88 unique presence/absence
patterns**. Two big blocks — **9 genes on one SL258 pattern** and **11 on one 'other'
pattern** (both rare, af≈1.4–1.7%, faeces-direction) — are each *one* co-inherited
sub-clade signal reported once per gene. pyseer tests every variant **univariately**, so
these are neither independent hits nor a combined-stronger signal; they mark a sub-branch
whose phenotype skew survives the LMM kinship correction. `pattern_group` / `n_in_pattern`
flag them; within a block you cannot resolve which gene (if any) is causal (perfect LD).

## Strongest signatures (by variance explained; patterns collapsed)

| VE% | gene | dir | lineage | af | β |
|---|---|---|---|---|---|
| 8.0 | KPN_RS10755 (riboflavin synthase) | faeces | other | 0.069 | −0.56 |
| 7.8 | dnaK (chaperone) | invasion | **SL307** | 0.065 | +0.57 |
| 7.7 | phoA (alkaline phosphatase) | invasion | other | 0.109 | +0.45 |
| 7.5 | KPN_RS02245 (RcnA Ni/Co efflux) | faeces | **SL307** | 0.074 | −0.52 |
| 6.6 | KPN_RS10715 (HTH regulator) | invasion | **SL147** | 0.109 | +0.41 |
| 4.7 | focA (formate transporter) | faeces | other | 0.107 | −0.35 |
| 3.6 | nfuA (Fe-S biogenesis) | invasion | other | 0.586 | +0.19 |
| 3.5 | KPN_RS09430 (iron-cofactor redox) | invasion | **cross-lineage** | 0.115 | +0.29 |
| 3.4 | KPN_RS11350 (siderophore receptor) | invasion | **SL17** | 0.059 | +0.39 |
| 3.2 | KPN_RS24485 (fimbrial usher) | invasion | **cross-lineage** | 0.274 | +0.20 |

**Reading:** several of the strongest are **lineage-restricted** (dnaK→SL307, RcnA→SL307,
HTH→SL147, siderophore→SL17) — confounded even by effect size, so not generalisable
invasion signals. The strongest **cross-lineage invasion** hits (the generalisable ones)
are the **iron-cofactor redox enzyme**, the **fimbrial usher**, and **nadB**; capsule
***wzi*** is real but modest (~1% VE), *not* the headline I first reported. The coherent
**capsule + fimbrial + core-iron** theme holds — but as a spread of small cross-lineage
effects, not a few dominant genes, with the accessory iron/capsule machinery untested here.

## Files

- `qq_lmm_bigsl_af1.png` — QQ of structure-adjusted p (λ=0.562; controlled bulk + tail).
- `manhattan_lmm_bigsl_af1.png` — Manhattan on NC_009648, Bonferroni line.
- `manhattan_lmm_bigsl_af1_annotated.png` — Manhattan with the **top hits labelled by
  variance explained** (point size ∝ VE; `↑`/`↓` = which allele, a minor glyph — *not* the
  ranking axis).
- `blood_vs_faeces_hits_annotated.tsv` — the **110 significant hits**, gene-mapped +
  virulence cross-ref, now **ranked by `var_explained_pct`** with `pattern_group` /
  `n_in_pattern` flagging the clonal blocks (`direction` is an attribute, not the sort key).
  This is the result we keep.
- `hits_by_direction_then_lineage.tsv` — *superseded* by the variance-explained ranking
  above; kept only as the earlier direction-first view.
- `blood_vs_faeces_gwas_summary.json` — λ, thresholds, counts.

## Provenance / reproduce

```
USE_LMM=1 sbatch src/bac_pyseer/kleb_iso_source/scripts/run_pyseer.sh 10 gwas_lmm 100
```
(`K=10` is ignored under `--lmm`; `gwas_lmm` = output subdir; `100` = min Sublineage
size for `--lineage` attribution. Reuses `similarity.tsv` if present.)

## Status of the fixed-effects K-sweep

**Not needed.** MDS cannot capture this cohort's relatedness at any practical K — the
scree (`../mds_model/scree_mds_bigsl.png`) shows K=200 captures only 12.8%. The LMM
result above is the method of record; the held K-sweep is dropped.
