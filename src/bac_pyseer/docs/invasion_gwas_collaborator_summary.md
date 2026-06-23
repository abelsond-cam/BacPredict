# *Klebsiella* invasion GWAS — collaborator summary

*Short, collaborator-facing abstract. The canonical, fully-detailed write-up (all axes, figures,
caveats, cross-axis synthesis) is the hub [`PROGRESS.md`](PROGRESS.md).*

**Question & design.** Does genomic variation distinguish **invasive** *Klebsiella* (blood,
respiratory) from **gut carriage** (faeces)? Two contrasts share faeces as baseline — blood vs faeces
(n=13,602) and respiratory vs faeces (n=9,169) — on core-genome variant calls vs the MGH 78578
chromosome (`NC_009648`), corrected by a linear mixed model (genome-wide kinship; λ = 0.562 / 0.498,
both < 1 = conservative). This is the **chromosomal / core-allele axis**; accessory determinants
(capsule *type*, aerobactin *iuc*, yersiniabactin *ybt*) are mobile/accessory and invisible here —
that's the in-progress unitig GWAS.

**Headline.** A reproducible, cross-lineage **adhesion + capsule** invasion signature — the
**fimbrial/pilus usher** (`KPN_RS24485`) and **capsule-assembly *wzi*** (`KPN_RS13515`), the same SNP
β>0 in *both* invasive niches — plus **BtuB**; and a **blood-specific iron-acquisition** signal
(siderophore receptor, iron-redox, NfuA) that is coherently null in respiratory (consistent with
blood iron-restriction). Effect sizes are small (≤8% variance explained): leads, not a classifier.

**New since the prior version (see the hub):**
- **Per-hit SNP consequence.** The **blood-invasion-direction hits carry no protein-coding change** (10
  synonymous + 8 noncoding, 0 missense/LoF), whereas respiratory-invasion *does* include missense/LoF.
  An observation, not a verdict — synonymous/noncoding can be regulatory/expression-level drivers or can
  tag a background; to be resolved on the accessory (unitig/GPA) axes.
- **Clade-specific hits are treated as real, not confounding** (the LMM over-corrects population, λ<1) —
  plausibly clade-specific adaptation to the acquired accessory genome.
- **Per-source hotspot Chi-sq (secondary/orthogonal):** a modest respiratory-weighted signal of
  functional diversification in **regulators** (`phoQ`, `mgrB`, `ramA`, `qseC`) + iron (`sufB`); the
  largest raw hotspots are clade-linked sequence diversity in chromosomally-integrated mobile elements
  (cps K-locus, a DISARM pathogenicity island, a prophage), not codon-level selection.

**Caveats.** Between-strain (association, not tracked gut→blood); chromosomal axis only; conservative
test (λ<1 = a floor on signal); consequence ≠ causality.

**Next.** Unitig GWAS (accessory/HGT axis); Panaroo gene-presence/absence; faeces vs liver/abscess
(blocked on recurating the mixed liver/abscess category in BacHGT). Supporting detail + figures:
[`PROGRESS.md`](PROGRESS.md) and [`visualise/`](visualise/).
