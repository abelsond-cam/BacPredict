# Per-source variant-hotspot Chi-sq — secondary, orthogonal to the phenotype GWAS

**This is a secondary analysis.** The primary result is the niche-specific phenotype GWAS (the variant
hits + their consequence spectrum) — see the hub [`../../PROGRESS.md`](../../PROGRESS.md) §4. This page
asks a *different, orthogonal* question: **does an invasion niche accrue more distinct functional
variation in a gene than the background — i.e. is there diversifying selection / an "arms race" behind
the phenotypic hotspots?** Evidence first; conclusions kept tentative.

## Method

For each gene *g* and source group *G*, count the **distinct** `(POS,REF,ALT)` loci of a consequence
class seen in ≥1 sample of *G* (*distinct-locus richness* — a SNP that swept a sub-clade counts once, so
sample-level clonal expansion does not inflate it). Per gene, a **share-based 2×2 Fisher** (one-sided):
is gene *g* a larger fraction of *G1*'s variant pool than of *G2*'s? Share-based, so unequal group sizes
don't bias it and gene length cancels. Distinct loci come from the per-sample caches joined to the SnpEff
effect map. BH-FDR per contrast × subset; significant = `padj<0.05 & share_ratio>1`. Cached group sizes:
blood 7,176 / faeces 6,978 / respiratory 4,327.

**Two controls** carry the interpretation:
- **population-vs-niche-specific** — each niche is tested both against the gut pair (`blood_vs_faeces`,
  `respiratory_vs_faeces`) and against the rest of the population (`blood_vs_rest`, `resp_vs_rest`);
- **synonymous** — a neutral-expectation negative control. Non-syn enrichment *above* the synonymous
  baseline is the codon-level (functional) signal; non-syn ≈ synonymous is broad sequence diversity.

**Ancestry.** The hotspot author's pipeline is ancestry-corrected, and distinct-locus richness already
removes sample-level clonal expansion — so no separate ancestral-reconstruction step is required here.

## Evidence — significant genes (padj<0.05) per contrast × subset

| contrast | synonymous | non_syn | LoF | all_coding |
|---|--:|--:|--:|--:|
| `blood_vs_faeces` | 0 | 1 | 0 | 0 |
| `blood_vs_rest` | 0 | 0 | 0 | 0 |
| `respiratory_vs_faeces` | 1 | 5 | 1 | 2 |
| `resp_vs_rest` | 7 | 9 | 4 | 12 |

The signal is **respiratory-weighted**; blood is essentially flat at the gene level (one non-syn hit,
`sufB`). The synonymous control then splits the significant genes into two classes
([`functional_vs_density_table.tsv`](functional_vs_density_table.tsv);
[`../../progress_figures/hotspot_codon_vs_clade.png`](../../progress_figures/hotspot_codon_vs_clade.png)):

### Class 1 — codon-level functional (non-syn ≫ synonymous)

Enriched for non-synonymous variation with the synonymous baseline flat — diversification at the protein
level. Modest share-ratios (1.3–2.8×), synonymous-clean:

- **Regulators / two-component** (see [`../../PROGRESS.md`](../../PROGRESS.md) §5): `phoQ` (both
  respiratory contrasts), `mgrB`, `ramA`, `qseC`.
- **Iron / Fe-S**: `sufB` — the only functional hit shared by blood (the lone `blood_vs_faeces` non-syn
  hit) and respiratory.
- **Glucose / PQQ metabolism**: `bglF` + a membrane PQQ-dependent glucose dehydrogenase / `pqqF` /
  Gfo-Idh-MocA / SDR cluster.
- **LoF breakout**: `ompK35` (porin), `carA`/`carB`, plus `mgrB` (LoF) — candidate de-repression (§5).

### Class 2 — clade-linked sequence diversity (non-syn ≈ synonymous)

The largest raw "hotspots" are enriched **equally** in synonymous and non-syn → **not** codon-level
diversifying selection, but genuine **between-clade sequence diversity** in chromosomally-integrated
mobile elements. Variants are anchored to the chromosome `NC_009648`, so these are its integrated copies
(not the reference plasmids, which are outside the call set):

- the **cps K-locus** capsule region (~2.72–2.75 Mb: `wzi`, `gndA`, `ugd`, polysaccharide export) — the
  most hypervariable, clade-defining locus in *Klebsiella*;
- a chromosomally-integrated **pathogenicity/defence island** (~5.14 Mb, `KPN_RS25300`–`25395`): a
  *"STY4528 family pathogenicity-island replication protein"* + ParB partition protein + the **DISARM**
  anti-phage system (`drmABCD`) + restriction-modification + an Abi toxin;
- a **prophage** tail gene (`KPN_RS07680`, ~1.59 Mb).

These are **not discarded as confounds**: they are real clade-associated diversity in integrated mobile
elements, consistent with the clade-adaptation reading in [`../../PROGRESS.md`](../../PROGRESS.md) §2/§4.
The synonymous control simply tells us the mechanism is broad sequence diversity, not per-codon selection.
Note **`wzi`** is also a cross-lineage *variant*-GWAS invasion hit — there as a specific synonymous
allele; here its hotspot status is K-locus hypervariability. Two different, both-real observations.

## Caveats (tentative, not strong)

- Residual population structure cannot be *fully* excluded — but the variant-kinship LMM that defines the
  phenotype hits already over-corrects for it (λ<1; [`../../PROGRESS.md`](../../PROGRESS.md) §2), so the
  burden of proof runs against "it's just structure," not for it.
- The functional / clade split (non-syn vs synonymous share-ratio) is a triage, not a formal between-group
  dN/dS interaction test. Effects are modest. This is a screen that *qualifies* the phenotype hits — it
  does not establish association.

## Files

| file | contents |
|---|---|
| `significant_hits.tsv` | the 42 `padj<0.05 & SR>1` rows (all contrasts × subsets) |
| `functional_vs_density_table.tsv` | per gene×contrast: share-ratio per subset, `NS/syn`, and the functional / clade-diversity / LoF-specific / mixed verdict |
| `source_hotspot_manifest.json` | group sizes, per-subset `R` totals, genes-tested + sig counts |
| *(RDS)* `…/source_hotspot/chisq/per_gene_enrichment.tsv.gz` | the full 76,736-row long table |
