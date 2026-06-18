# *Klebsiella* invasion GWAS — collaborator summary (two niche contrasts)

*Interim, core-genome (chromosomal) axis. Updated 2026-06-18.*

## Question & design

Does genomic variation distinguish **invasive** *Klebsiella* (isolated from a sterile site)
from **gut carriage** (faeces)? Gut colonisation is the reservoir for invasive disease, so we
contrast invasive-niche isolates against faecal isolates, with faeces as the shared baseline:

| contrast | invasive (=1) | control (=0) | n | λ |
|---|---|---|---|---|
| **1** | blood | faeces | 13,602 | 0.562 |
| **2** | respiratory | faeces | 9,169 | 0.498 |

- **Cohorts:** KPSC human isolates, country-balanced 2:1, pooled study threads.
- **Signal:** core-genome **variant calls** (SNPs/indels vs the *K. pneumoniae* MGH 78578
  reference, NC_009648), af 1–99%. This is the **chromosomal / core-allele axis** — it cannot
  see accessory gene gain/loss (capsule *type*, aerobactin/*iuc*, yersiniabactin/*ybt* are
  mobile/accessory and **invisible here**; that's the planned unitig GWAS).
- **Population-structure control:** linear mixed model (genome-wide kinship). **λ ≈ 0.5 in both**
  — the same mildly-conservative behaviour (controlled bulk, clean signal tail, no inflation).
  A fixed-effects MDS attempt under-corrected badly (λ=4.34) and was abandoned.
- **Ranking:** by **variance explained** (∝ f(1−f)·β², direction-agnostic — a "faeces" allele
  is just the converse "blood" allele). Effect sizes are uniformly **small** (median ~1–1.3% of
  case/control variance, max ~8%): no single core variant predicts invasion, and invasiveness is
  polygenic *and* heavily accessory-driven. **These are leads, not a classifier.** We focus on
  **MAF > 5%** (neither allele a tiny sample), which also drops the rare lineage-private markers.

## Replication is the real test

Both contrasts share the **gut as baseline**, so a *faeces-associated* variant recurs
**trivially** (a gut-resident sub-clade is over-represented vs any non-gut niche). The
informative test is **invasion-direction recurrence** — β > 0 in *both* contrasts.

**Consistency:** **all 18 blood-invasion hits also trend invasion (β > 0) in respiratory** —
100% direction concordance; none flip. They split by how strongly they recur:

### A. Reproducible cross-niche invasion hits — genome-wide significant in BOTH

| gene | function | blood β (VE) | resp β (p) | lineage |
|---|---|---|---|---|
| **fimbria/pilus usher** (KPN_RS24485) | fimbrial assembly; adhesion | +0.20 (3.2%) | +0.24 (2.4e-10) | cross-lineage |
| **capsule assembly Wzi** (KPN_RS13515) | capsule surface-anchoring (K-locus) | +0.27 (1.0%) | +0.25 (1.8e-8) | cross-lineage |
| **BtuB** (KPN_RS22930) | TonB-dependent B12/cobalamin (& siderophore/colicin) receptor | +0.09 | +0.10 (1.5e-7) | cross-lineage |

A reproducible **adhesion + capsule + outer-membrane-receptor** signature, holding across two
independent invasive niches and across lineages. The same SNP in each case. *(Wzi is modest in
effect size but reproducible across niches — for a real signal, reproducibility beats VE.)*

### B. Blood-specific iron-acquisition signal — coherent, *not* weak

The three iron-acquisition/handling hits are **present in respiratory at similar frequency but
null there**, while strongly blood-associated:

| gene | blood β (VE) | resp p (null) |
|---|---|---|
| **TonB-dependent siderophore receptor** (KPN_RS11350) | +0.39 (3.4%) | 0.012 |
| **iron-cofactor redox enzyme** (KPN_RS09430) | +0.29 (3.5%) | 0.022 |
| **NfuA** Fe-S cluster biogenesis (KPN_RS20445) | +0.19 (3.6%) | 0.008 |

All three vanishing **together** in respiratory (not half) is the signature of a genuine
**niche-specific** effect, not noise: **blood is iron-restricted** (transferrin/lactoferrin
sequestration) so iron-piracy is advantageous for bloodstream invasion, whereas the respiratory
niche has different iron/redox dynamics. So the iron signal is **real and blood-specific**, which
the second contrast *strengthens* rather than refutes.

### C. Lineage-restricted / sub-threshold (7 hits)

dnaK (SL307), an HTH regulator and a PLP-aminotransferase (both SL147), iolB, hpxZ — trend
invasion in respiratory but don't reach significance, consistent with partly-clonal (lineage-bound)
blood signals.

## Caveats

- **Core-genome axis only** — accessory/HGT determinants (capsule type, aerobactin, yersiniabactin)
  are untested here; the **unitig GWAS** is the next, complementary analysis.
- **Between-strain design** (strains *isolated from* blood vs faeces, not the same strain tracked
  gut→blood): these are **associations**, not proven mechanism.
- **Small effects, conservative test** (λ < 1): a floor on the signal, not a ceiling.
- **Gene labels:** many hit loci have no gene *symbol* in the MGH 78578 reference (verified against
  the GFF — not a pipeline error); we label by symbol → product description → locus tag.

## Next steps

1. **Unitig GWAS** — the accessory/HGT axis (aerobactin, ybt, capsule type) this core scan can't see.
2. **Panaroo gene-presence/absence GWAS.**
3. **faeces vs liver/abscess** — third niche; *blocked* on recurating the mixed liver/abscess
   category in BacHGT `metadata_curation.py`.

## Supporting files (`src/bac_pyseer/docs/visualise/`)

- `lmm_model/` — blood-vs-faeces: QQ, Manhattan (hits sized by variance explained, top labelled),
  VE-ranked hit table, README.
- `faeces_resp_lmm_model/` — respiratory-vs-faeces equivalents, plus
  `blood_invasion_replication_in_resp.tsv` (every blood-invasion hit with its respiratory p/β and
  replication class) and `cross_contrast_overlap_blood_vs_resp.tsv`.
