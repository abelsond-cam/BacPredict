# Per-source variant-hotspot enrichment (Phase 3)

**Question.** Does an invasion niche (blood / respiratory) accrue *functional* variation in a gene
faster than the gut / the other niches? This is the niche test that the whole-population dN/dS
annotation (Phase 3a, a documented NULL — see [`../hotspot_annotation/`](../hotspot_annotation/))
could not answer.

**Method** (`kleb_iso_source/source_hotspot_chisq.py`, run by `scripts/run_source_hotspot_chisq.sh`,
job `30955221`, 19 min). For each gene *g* and source group *G* we count the **distinct**
`(POS, REF, ALT)` loci of a consequence class seen in ≥1 sample of *G* — *distinct-locus richness*,
so a SNP that swept a sub-clade counts **once** (clonal expansion does not inflate it). Per gene we
run a **share-based 2×2 Fisher** (one-sided, *greater*):

```
            gene g            all other coding genes
focal G1 :  r(g,G1)           R(G1) − r(g,G1)
compar G2:  r(g,G2)           R(G2) − r(g,G2)
```

i.e. *is gene g a larger fraction of G1's variant pool than of G2's?* Share-based, so the unequal
group sizes don't bias it; gene length cancels (same gene across groups). Distinct loci come from the
per-sample caches (`extract_sample_loci.py`) joined to the SnpEff effect map
(`annotate_locus_consequence.py`, which carries `locus_tag` → no GFF needed). BH-FDR per
contrast×subset; "significant" = `padj<0.05 & share_ratio>1`.

**Four consequence subsets**, and the design hinges on one of them:

| subset | classes | role |
|---|---|---|
| `synonymous` | LOW | **negative control** — neutral expectation |
| `non_syn` | MODERATE+HIGH | **primary** — functional variation |
| `LoF` | HIGH | breakout — the de-repression hypothesis |
| `all_coding` | LOW+MOD+HIGH | sanity — raw coding mutational density |

**The synonymous control is the whole game.** A gene enriched in `non_syn` but **not** in
`synonymous` (`NS/syn` share-ratio ≫ 1, syn not significant) is a genuine **functional/selective**
signal. A gene enriched **equally** in both (`NS/syn ≈ 1`, syn *also* significant) is just elevated
**mutational density / clade structure**, not selection — and we reject it.

**Four contrasts.** Niche pairs `blood_vs_faeces`, `respiratory_vs_faeces` (invasion vs gut, matching
the variant GWAS) and one-vs-rest `blood_vs_rest`, `resp_vs_rest` (*rest* = the other labelled
niches, focal excluded). Cohort sizes (cached): blood 7,176 / faeces 6,978 / respiratory 4,327.

---

## Headline

**The signal is almost entirely respiratory; blood is null; and the largest raw "hotspots" are
clade/MGE-density artifacts the synonymous control rejects.** Significant genes (`padj<0.05`):

| contrast | synonymous | non_syn | LoF | all_coding |
|---|--:|--:|--:|--:|
| `blood_vs_faeces` | 0 | **1** | 0 | 0 |
| `blood_vs_rest` | 0 | **0** | 0 | 0 |
| `respiratory_vs_faeces` | 1 | **5** | 1 | 2 |
| `resp_vs_rest` | 7 | **9** | 4 | 12 |

After the synonymous-control verdict (`functional_vs_density_table.tsv`): **12 functional(non-syn),
6 density/clade, 5 LoF-specific, 9 mixed/weak.**

### 1. A modest but clean functional signal in respiratory: regulators + iron + glucose metabolism

These are enriched for **non-synonymous** variation with `synonymous` flat (`NS/syn > 1.3`, syn
`padj` ns) — selection above the clade baseline, not density. Effects are modest (share-ratio
1.3–2.1) but the synonymous control is clean:

- **Two-component / transcriptional regulators — the de-repression theme.**
  **`phoQ`** (sensor kinase of PhoPQ; Mg²⁺ / antimicrobial-peptide sensing, polymyxin resistance,
  virulence) is the anchor — significant in **both** respiratory contrasts (`NS/syn` 1.56 & 1.77,
  syn `padj` 0.91–0.97). **`mgrB`** (the PhoPQ feedback repressor; its LoF is the canonical
  colistin-resistance / PhoPQ-de-repression route — `NS/syn 2.79`), **`ramA`** (AcrAB-TolC efflux /
  MDR master regulator — `NS/syn 1.68`), and **`qseC`** (quorum-sensing sensor kinase — `NS/syn
  1.53`) round it out. A coherent picture of respiratory-niche selection acting on the *regulators*
  of stress / resistance / virulence two-component systems.
- **Iron / Fe-S — `sufB`** (Fe-S cluster assembly): the **only** functional hit in `blood_vs_faeces`
  (`NS/syn 1.6`) **and** a functional hit in respiratory (`NS/syn 1.5`) — a cross-niche invasion
  signal in iron metabolism, echoing the blood iron-restriction theme from the variant GWAS.
- **Glucose / PQQ metabolism — `bglF`** (PTS sugar transporter) plus a cluster of cofactor/redox
  genes (membrane-bound PQQ-dependent glucose dehydrogenase, `pqqF`, a Gfo/Idh/MocA oxidoreductase,
  an SDR oxidoreductase) — metabolic-adaptation candidates.

### 2. LoF breakout (de-repression hypothesis)

LoF-specific hotspots (significant in `LoF`, not driven by the broader non-syn pool): **`ompK35`**
(porin loss — a known carbapenem/cephalosporin-resistance mechanism; `LoF` SR 1.62, respiratory,
**a GWAS hit in the respiratory-invasion direction**), **`carA`/`carB`** (carbamoyl-phosphate
synthase operon), **alpha-galactosidase**, and **`mgrB`** (LoF SR 2.11 — consistent with §1). Porin
and PhoPQ-repressor LoF in the invasion direction fit the de-repression hypothesis.

### 3. Density / clade artifacts — rejected by the synonymous control

The biggest raw `all_coding` "hotspots" are **equally enriched in synonymous** (`NS/syn ≈ 0.9–1.0`,
syn `padj` significant) → **mutational density / clade structure, not selection**:

- the **cps K-locus** capsule-biosynthesis block — `wzi`, `gndA`, `ugd`, polysaccharide-export
  (contiguous `KPN_RS134xx–135xx`; the most hypervariable, clade-defining region in *Klebsiella*);
- **prophage / restriction-modification islands** — type I restriction endonuclease, Y-family DNA
  polymerase, replication endonuclease, phage portal/tail-sheath, DNA cytosine methyltransferase.

These are exactly the contiguous-locus-tag blocks (genomic islands over-represented in
respiratory-enriched clades) the screen is designed to flag and discount. **Note `wzi`:** it is a
cross-lineage *variant*-GWAS invasion hit, but here it is enriched **equally in synonymous and
non-syn** — so its hotspot status is K-locus hypervariability, *not* per-gene diversifying selection.
The GWAS *wzi* signal is a specific-allele association, a different (and still real) thing.

### 4. Blood is null at the gene level

`blood_vs_faeces` / `blood_vs_rest` are essentially empty (one hit, `sufB`). The rich blood-invasion
variant-GWAS signature (iron / capsule / fimbrial) is therefore **allele-specific association, not
gene-level hypervariability** — blood isolates do not accrue a broader functional repertoire per gene
than gut isolates. Coherent with the two tests asking different questions.

---

## Caveats — a screen, not proof

- **Clade structure is only partly controlled.** Distinct-locus richness removes *sample-level*
  clonal expansion, and the synonymous control removes genes whose enrichment is pure mutational
  density. Neither removes the case where a respiratory-enriched **clade** carries many *distinct*
  adaptive regulator mutations — that would still inflate `non_syn` with `synonymous` flat and read
  as "functional." The rigorous follow-up is **homoplasy / convergence counting on the phylogeny**
  (do these `phoQ`/`mgrB`/`ramA` mutations arise *independently* across lineages?) — ancestral
  reconstruction, not in scope here.
- **The functional / synonymous split is qualitative** (`NS/syn` ratio + whether syn is significant),
  not a formal between-group dN/dS interaction test. It is a triage, deliberately so.
- **Effects are modest** (share-ratios 1.3–2.1). This is a hypothesis-generating screen over ~5,000
  genes, not a confirmatory result. The honest summary: the big raw hotspots say almost nothing
  (clade/MGE density); the only non-trivial signal is a modest, synonymous-clean, **regulator-focused
  non-syn excess in the respiratory niche**, plus a cross-niche iron (`sufB`) hit.

## Files

| file | contents |
|---|---|
| `significant_hits.tsv` | the 42 `padj<0.05 & SR>1` rows (all contrasts × subsets) |
| `functional_vs_density_table.tsv` | per gene×contrast: share-ratio in every subset, `NS/syn`, and the **functional / density-clade / LoF-specific / mixed** verdict |
| `source_hotspot_manifest.json` | group sizes, per-subset `R` totals, genes-tested + sig counts |
| *(RDS)* `…/source_hotspot/chisq/per_gene_enrichment.tsv.gz` | the full 76,736-row long table (all genes, all contrasts × subsets) |
