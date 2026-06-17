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

## Direction & lineage — the cross-lineage filter

Re-sorting the 110 hits by direction → lineage (`hits_by_direction_then_lineage.tsv`)
sharpens the story. **Only 18 hits are blood/invasion (β>0); 92 are faeces (β<0)** —
and much of the faeces mass is residual clonal-block leakage (see caveat below). The
blood side is the small, clean, interesting set, and it splits by lineage attribution:

| blood class | n | reading |
|---|---|---|
| **cross-lineage (blank)** | **8** | no single SL dominates → genuine cross-lineage signal |
| other-bucket | 5 | ambiguous (collapsed small-SLs *or* background) |
| single-SL | 5 | likely lineage-restricted markers |

**No blood gene is significant under two *different* named SLs** — but that's partly a
pyseer artifact (`--lineage` assigns each variant one "most-associated" lineage). The
real cross-lineage tell is **blank attribution + allele frequency**: a blood variant at
af 0.27 (fim-usher), 0.61–0.86 (iolB), or 0.74 (`KPN_RS10380`, *labelled* SL147 but far
too common to sit in one SL) must span many lineages regardless of the label.
`manhattan_lmm_bigsl_af1_annotated.png` encodes this: **bold bright-red = cross-lineage,
muted red (with the SL tag) = single-SL/other.**

## Biology — what survived a conservative test

**Blood / invasion (β > 0) — the encouraging, cross-lineage signal.** After proper
structure correction the canonical *Klebsiella* invasion machinery surfaces among the
positive-β hits:

- **Capsule assembly Wzi** (`KPN_RS13515`, pos 2,744,372; β=+0.27; **no SL
  attribution → cross-lineage**) — capsule is the textbook hypervirulence/invasion
  determinant; a cross-lineage capsule hit up in blood is exactly the real signal we
  hoped for. *(virulence-flagged)*
- **TonB-dependent siderophore receptor** (`KPN_RS11350`, pos 2,307,956; β=+0.39;
  SL17) — iron piracy is central to invasiveness. *(virulence-flagged)*
- **btuB TonB-dependent receptor** (`KPN_RS22930`) — appears on both strands of the
  signal (a blood + variant and a faeces − variant), iron/cobalamin uptake.
- **fimbria/pilus outer-membrane usher** (`KPN_RS24485`, β=+0.20) — adhesion/invasion.
- plus dnaK, nadB, phoA, an iron-containing redox enzyme, and others.

**Faeces (β < 0) — read with caution: residual clonal blocks.** Many of the *strongest*
faeces hits are rare (af ≈ 1.4%) and share **identical lrt-pvalue and identical β across
genes scattered all over the chromosome**, every one attributed to a single Sublineage
(blocks for SL258, SL35, SL307). That signature — one clonal frame, many unrelated
genes, one p-value — is a **lineage-marker artifact**, not gene-level biology. Even the
LMM leaks a little clonal signal here. So interpret the **cross-lineage** faeces hits
(no/`other` SL attribution, varied p-values — e.g. the glycoside-hydrolase-19, FocA,
nitrate-reductase, mutY hits) and treat the identical-stat single-SL blocks as
lineage tags.

## Files

- `qq_lmm_bigsl_af1.png` — QQ of structure-adjusted p (λ=0.562; controlled bulk + tail).
- `manhattan_lmm_bigsl_af1.png` — Manhattan on NC_009648, Bonferroni line.
- `manhattan_lmm_bigsl_af1_annotated.png` — same Manhattan with the **18 blood/invasion
  genes labelled** (bold red = cross-lineage, muted red + SL tag = single-SL/other).
- `hits_by_direction_then_lineage.tsv` — the 110 hits re-sorted blood-first, then by
  lineage class (cross-lineage → other → single-SL).
- `blood_vs_faeces_hits_annotated.tsv` — the **110 significant hits**, gene-mapped +
  virulence cross-ref. The result (unlike the MDS TSV, this one we keep and use).
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
