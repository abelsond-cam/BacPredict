# Whole-population dN/dS hotspot annotation — a documented NULL (superseded)

Annotating the invasion-GWAS hits against a **whole-population** per-gene dN/dS + variant-hotspot
table answers the wrong question and is **uninformative**: GWAS hits are by construction in variable
genes, so a hit↔hotspot association is largely mechanical; there is no directional or selection signal.

The question worth asking is **niche-specific** — *does an invasion niche accrue variation in a gene
faster than the background?* — and is answered by the per-source hotspot Chi-sq:
[`../source_hotspot_chisq/`](../source_hotspot_chisq/) (with its synonymous + population-vs-niche
controls). See the hub [`../../PROGRESS.md`](../../PROGRESS.md).

Kept here: this README only. The result tables + enrichment summaries were removed and the code
(`annotate_hits_with_hotspots.py`) was deleted (both recoverable from git history). The per-gene table
it used, `data/combined_poisson_test_variant_hotspots.txt`, is **retained** — it is the live
gene-annotation input to `source_hotspot_chisq.py`.
