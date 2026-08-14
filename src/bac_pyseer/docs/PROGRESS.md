# *Klebsiella* invasion GWAS — overview

**What primes a *Klebsiella* isolate to cause invasive disease (blood / respiratory) rather than gut carriage
(faeces)?** A `pyseer` GWAS across complementary axes — all on the country-balanced blood/faeces (+
resp/faeces) cohort, one reference (`NC_009648`, MGH 78578), faeces as the shared control, every hit oriented
to its **invasion allele**. The detailed write-ups live in per-axis documents:

| Axis | Doc | State |
|---|---|---|
| **Variant (core-SNP)** — chromosomal alleles | [`PROGRESS_VARIANTS.md`](PROGRESS_VARIANTS.md) | done; permutation-validated. Open: revisit MDS, hotspot mapping |
| **Unitig (accessory / HGT)** — whole accessory sequence | [`PROGRESS_UNITIGS.md`](PROGRESS_UNITIGS.md) | run + calibration resolved (no af ceiling; signal at common af). Next: geNomad plasmid/chromosome mapping |
| **Gene presence/absence** (Panaroo) | [`PROGRESS_PANAROO.md`](PROGRESS_PANAROO.md) | placeholder; may not run (scale + unitig redundancy) |
| **Independent-origin hotspot** — recurrent-mutation sub-analysis of the variants | within [`PROGRESS_VARIANTS.md`](PROGRESS_VARIANTS.md) §5 | done |

## Cross-axis headlines

- **Variant axis** replicates near-completely across blood↔respiratory (**85/86 independent patterns
  concordant**, p≈1.1×10⁻²⁴, r²=0.78); most invasion variance sits in **common, population-wide alleles**;
  coherent themes = **transcriptional-regulator de-repression** + **iron / Fe-S acquisition**. Conservative
  (λ<1) and permutation-validated.
- **Unitig axis**: the common-af inflation is **real accessory signal**, not structure (within-lineage
  permutation λ_perm≈1 at *all* af, at sublineage **and** clonal-group resolution) — so no af ceiling, and
  the signal is concentrated at common af but **LD-redundant** (clonal co-inheritance).
  ⚠ **Do not label it "HGT".** The geNomad mapping is **done**
  ([`PROGRESS_UNITIGS.md`](PROGRESS_UNITIGS.md)) and it splits by direction: the **blood/invasion** side is
  ~82% **chromosomal**, while the faeces side is MGE-borne. An accessory-genome signal is not automatically
  an acquired one.

## Method common to every axis

Calibration & reliability are assessed identically on each axis: **(1)** locus universe filtered to **≥1%
penetrance** (~2.04M → ~372k variant loci); **(2)** **allele-frequency-stratified** genomic-inflation λ
(a single genome-wide λ hides *where* mis-calibration sits); **(3)** a **within-lineage permutation null**
(shuffle phenotype within each sublineage → structure preserved, signal destroyed → re-fit fresh) to separate
real signal from residual structure. See each axis doc for the numbers.

Pipeline / run detail: [`../kleb_iso_source/CLAUDE.md`](../kleb_iso_source/CLAUDE.md). Cross-task tracker:
[`PROJECT_STATE.md`](../../../PROJECT_STATE.md).
