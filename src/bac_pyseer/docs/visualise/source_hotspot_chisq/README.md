# per-source distinct-locus-richness Chi-sq (figures & tables)

Artifacts for the **§4a / §5 cross-checks** in [`../../PROGRESS.md`](../../PROGRESS.md): does an invasion
niche carry *more distinct variation* in a gene than the background — a **hypervariability vs codon-level
selection** split? Secondary to the genome-wide Poisson recurrent-mutation test that drives §5.

- `functional_vs_density_table.tsv` — per gene: `SR_syn` / `SR_nonsyn` / `NS_over_syn` + `verdict`
  (`density/clade` = hypervariable, set aside in §4a; `functional(non-syn)` = codon-level selection).
- `significant_hits.tsv` (padj<0.05 & share-ratio>1), `source_hotspot_manifest.json`.

Reproduce: `kleb_iso_source/scripts/run_source_hotspot_chisq.sh`.
