# bac_pyseer / kleb_iso_source — invasive-disease GWAS

Pyseer + hotspot GWAS for the invasive-disease signal, starting with **blood vs
faeces** isolation source. Package overview: [CLAUDE.md](../CLAUDE.md); global
conventions: root [CLAUDE.md](../../../CLAUDE.md) §0. Milestones are tracked in
[ToDo.md](../../../ToDo.md) under "Pyseer GWAS → kleb_iso_source".

Three strands:

- **(a) Hotspot-rate Chi-sq** — per-source hotspot rate vs the whole-population
  background mutation rate at each locus (control). **Blocked on Aaron uploading
  hotspots to HPC.**
- **(b) Pyseer unitig GWAS (KPSC-wide)** — variant calls → mutation loci vs the
  reference per sample; filter low-frequency loci; Jaccard pairwise distances; unitig
  GWAS.
- **(c) Pyseer presence/absence GWAS** — same variant calls + the per-SL Panaroo GPA.

Not started — an agent will plan the detail and begin (strand (a) remains blocked).
