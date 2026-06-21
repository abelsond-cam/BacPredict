# Hotspot / dN-dS annotation of the invasion-GWAS hits — a documented NULL

**Bottom line: annotating the invasion-GWAS hits against the _whole-population_ per-gene hotspot +
dN/dS table tells us almost nothing about invasion.** It is retained here only as a record of what
was run (so it isn't redone expecting signal). The question actually worth answering —
*does the invasion niche accrue variants in a gene faster than the population background?* — is
**niche-specific** and needs the per-isolation-source hotspot-rate **Chi-sq** (planned, §last),
computed directly from the per-sample variant→loci collation rather than read out of a
whole-population ratio.

## What was run

[`annotate_hits_with_hotspots.py`](../../../kleb_iso_source/annotate_hits_with_hotspots.py) merges
the combined per-gene table ([`combined_poisson_test_variant_hotspots.txt`](../../../data/combined_poisson_test_variant_hotspots.txt);
5,422 genes, 768 flagged `is_sig`, on the MGH 78578 / `KPN_RS` annotation) onto the two LMM hit
tables by `locus_tag`, and tests enrichment — Fisher on hotspot `is_sig`, Mann–Whitney on `dn_ds` —
for all hit genes and invasion-direction (β>0) hit genes vs the tested-gene background. Outputs:
`*_hits_with_hotspots.tsv` + `hotspot_enrichment_summary.{json,md}`.

## Why it is uninformative (why we discount it)

1. **A "hotspot" is just a gene with an excess *variant count* — a whole-population "this gene is
   variable" property.** GWAS-hit genes are by construction variable genes, so a hit↔hotspot
   association is **partly mechanical** (the background is itself the variant-bearing gene set). The
   one nominally significant result (blood/faeces hit genes 27% `is_sig` vs 14% background, Fisher
   p≈4e-4) is largely that artifact: it does **not** localise to the invasion direction (invasion-only
   subset p=0.15) and is **absent** in faeces/respiratory (p=0.21).
2. **`dn_ds` here is a raw non-synonymous:synonymous _count_ ratio, not a site-normalized dN/dS.**
   ~¾ of random point mutations are non-synonymous, so the neutral expectation is not 1 — empirically
   the genome-wide median of this ratio is ~1.7. The invasion hits sit at 2.2–3.2 (only marginally
   above; Mann–Whitney p=0.07) and are **missense-dominated, 1–8% loss-of-function** — i.e. neither
   clearly selected nor degraded. The genuinely high ratios (regulators, 12–38) are whole-population
   hotspots, **not** invasion hits. So there is **no positive-selection / "arms-race" signal** here
   (an earlier draft of this README over-claimed one; that is retracted).
3. **No control for population structure or homoplasy.** The hotspot test is whole-population and
   cannot tell recurrent independent mutation (a true hotspot) from a single ancestral change spread
   by clonal expansion. **Ancestral reconstruction + phylogeny control are prerequisites** for any
   "this gene mutates fast" statement and are not done here.
4. **The table's metrics are unconfirmed.** It is a dataframe merge (`n.x`/`n.y` suffixes); `n.x`=
   synonymous / `n.y`=non-synonymous and `dn_ds = n.y/n.x` held for the genes spot-checked but
   **not for all 5,422 rows**, so `n.x`/`n.y`/`r`/`d.x_mod` and the Poisson test need confirming with
   the data's author before anything is built on them.

## The question worth pursuing: per-isolation-source hotspot-rate Chi-sq

Does a gene accrue variants **faster within the invasion niche (blood / respiratory) than in the
whole-population background?** That is niche-specific and answerable from our per-sample
`<Sample>.loci.tsv.gz` cache (the same data feeding the variant GWAS): per gene, the variant rate in
the source group vs the background rate → Chi-sq, with the structure/homoplasy caveats above
addressed as far as the data allow. **To be planned (plan mode).**

## Files (kept as a documented null — not a finding)

| File | What |
|---|---|
| `blood_faeces_hits_with_hotspots.tsv` | 110 blood/faeces hits + `hot_*` columns |
| `faeces_respiratory_hits_with_hotspots.tsv` | 88 faeces/resp hits + `hot_*` columns |
| `hotspot_enrichment_summary.json` / `.md` | Fisher + Mann–Whitney, all-genes & invasion-direction |
