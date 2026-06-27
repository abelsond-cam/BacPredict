# *Klebsiella* invasion GWAS — variant (core-SNP) axis

> **Axis docs.** This is the **variant (core-SNP)** axis. The accessory/HGT axes have their own documents —
> **[`PROGRESS_UNITIGS.md`](PROGRESS_UNITIGS.md)** (unitig LMM) and **[`PROGRESS_PANAROO.md`](PROGRESS_PANAROO.md)**
> (gene presence/absence; placeholder). Cross-axis overview: **[`PROGRESS.md`](PROGRESS.md)**.

**What primes a *Klebsiella* isolate to cause invasive disease (blood / respiratory) rather than gut
carriage (faeces)?** This is the write-up of the **variant (core-SNP) axis**: two linear-mixed-model GWAS
sharing faeces as the control, every hit re-oriented to its **invasion allele**, read for *where the
invasion signal sits, how reproducibly, in which lineages, and in which genes*. Per-axis figures + tables
live under [`visualise/`](visualise/) and [`progress_figures/`](progress_figures/).

**Posture.** Evidence first, conclusions tentative. We do **not** equate synonymous/noncoding with
"non-functional". The LMM over-corrects population structure (λ≪1), so clade-restricted hits are treated
as **real**, not confounding. Effect sizes are small (≤8% variance each): these are *leads*, not a
classifier.

**Headline.** The blood GWAS **replicates near-completely in an independent respiratory cohort** —
**85 of 86 independent patterns share the invasion direction** (binomial p≈1.1×10⁻²⁴; β correlated in
magnitude, r²=0.78). Most invasion variance sits in **common alleles of genes present in the reference**
(79% of blood's invasion variance at invasive_af ≥ 0.5) — invasion is largely a **population-wide adapted
state**, not a rare-variant phenomenon. The coherent gene themes are **transcriptional regulators
(de-repression)** and **iron / Fe-S acquisition**; a minority of hits sit in **independently recurrent
(convergent) mutation hotspots**, the rest are single-origin alleles spread clonally.

---

## Data, cohorts, and how we read a hit

One reference for every sample: **`NC_009648`** (*K. pneumoniae* MGH 78578 chromosome, 5,315,120 bp).
Calls are anchored to this chromosome **only** — its five plasmids (`NC_009649`–`53`) are **not** in the
call set, so plasmid-borne accessory content is invisible to this axis (that is what the unitig/GPA axes,
§6, are for). Per-contrast orientation ([`orientation_table.tsv`](orientation_table.tsv)):

| contrast | n (control / invasive) | λ | variants | hits | rare / common invasion allele |
|---|---|--:|--:|--:|--:|
| blood vs faeces | 13,602 (6,426 / 7,176) | **0.562** | 372,238 | 110 | 14 / 96 |
| resp vs faeces | 9,169 (4,737 / 4,432) | **0.498** | 326,146 | 88 | 43 / 45 |

Method of record: **LMM** (`pyseer --lmm`, FaST-LMM kinship from the core SNPs); the fixed-effects MDS
attempt was abandoned (λ=4.34, severe under-correction). **λ<1 in both** ⇒ the kinship random effect is a
*conservative* (arguably over-) correction, so surviving signals — including clade-attributed ones — are
unlikely to be pure structure artefacts.

**The variant set (locus universe).** Each sample is called against `NC_009648` with snippy (acceptance
filter `GT="1/1" && QUAL≥100 && DP≥3`), giving **2,038,383** candidate SNP + short-indel loci across the
cohort. We then drop every locus carried by **<1% of isolates** (the collaborators' design — a locus must be
present in ≥137 of the 13,602 samples), leaving **372,543** loci; the GWAS itself uses the **af 1–99%**
window — **372,238** variants analysed in blood/faeces, 326,146 in resp/faeces. The analysis universe is
therefore **common-to-intermediate chromosomal variation**: ultra-rare per-sample variants are filtered out,
not modelled — which is also why the allele-frequency-stratified calibration below is the right lens.

**Calibration across the frequency spectrum (completion check).** A single genome-wide λ can hide *where*
any mis-calibration sits, so we recompute it within allele-frequency bins
([`genomic_inflation_by_af.tsv`](genomic_inflation_by_af.tsv), via
[`genomic_inflation_by_af.py`](../kleb_iso_source/genomic_inflation_by_af.py)). The variant axis is
**conservative at every frequency**, never exceeding 1:

| af bin | blood λ | resp λ |
|---|--:|--:|
| 0.01–0.05 | 0.55 | 0.47 |
| 0.05–0.20 | 0.55 | 0.53 |
| 0.20–0.50 | 0.64 | 0.62 |
| 0.50–0.97 (common) | **0.72** | **0.67** |

So the common-allele signal that carries most of the invasion variance (§1) rests on well-calibrated — if
anything conservative — statistics; the reported λ≈0.5 is not masking a common-variant inflation. A
**within-lineage permutation null** confirms this is genuine structure control, not an artefact of λ<1:
shuffling the phenotype within each sublineage (structure preserved, signal destroyed) and re-running the
LMM gives **λ_perm ≈ 1 at every af bin** (0.86–1.07) — the kinship absorbs the population structure, so the
real λ<1 is true conservatism. (The *unitig* axis behaves very differently — its common-af inflation is
**real accessory signal**, not a kinship artefact — see [`PROGRESS_UNITIGS.md`](PROGRESS_UNITIGS.md).)

**The four axes.** Variant/core-SNP (**this report**: chromosomal alleles, incl. SNPs within
chromosomally-integrated islands; blind to plasmids) · **independent-origin hotspot** (a sub-analysis of
the same variants, §5) · **unitig** (accessory/HGT sequence — [`PROGRESS_UNITIGS.md`](PROGRESS_UNITIGS.md)) ·
**gene presence/absence** (Panaroo — [`PROGRESS_PANAROO.md`](PROGRESS_PANAROO.md)).

---

## §1 · Invasion framing and the common-allele spectrum

pyseer reports each variant's `β`/`af` **relative to the reference allele** — an artefact of which genome
we picked, not of biology. We care about *all* invasion-priming variation, and the **common** allele is
often the one *Klebsiella* evolution has selected. So every hit is re-oriented to its **invasion allele**:

- **`invasive_af`** = frequency of the invasion-associated allele (`af` if β>0, else `1−af`). Near **1** ⇒
  the invasion-adapted allele is **common / population-wide**; near **0** ⇒ a **rare** invasion variant.
- **`|β|`** (`abs_beta`) = effect magnitude, reported positive (every hit reads "+ invasion").
- **`var_explained`** ≈ `af(1−af)·β²` (normalised by the binary-phenotype variance ≈0.249) is **invariant
  to the reference choice** — the direction-free footing that ranks hits and gates §4/§5.

**Most invasion variance sits at the common end of the spectrum.**
![invasive_af histogram + cumulative variance](progress_figures/invasive_af_histogram.png)
Weighting the `invasive_af` distribution by variance explained, **79% of blood's invasion variance (66%
respiratory) lies at `invasive_af` ≥ 0.5** — common, population-wide invasion-adapted alleles, the signal
a conventional rare-variant (β>0-only) GWAS would discard as "faeces hits". The top loci span both ends:
common — a **riboflavin-synthase** noncoding change (invasive_af 0.93, VE 8.1%), the **RcnA Ni/Co efflux**
missense (0.93, 7.5%), **focA** (0.89, 4.7%); rare — **dnaK** (0.06, 7.8%), **phoA** (0.11, 7.7%).

**Synonymous hits are removed only where they are clade diversity, not signal (the hypervariable rule).**
A synonymous SNP changes no protein, but it can *tag* a real selective sweep — so we keep synonymous hits
in ordinary genes (the iron pathway, §4b, is carried by synonymous tags) and **drop them only in
hypervariable genes** (capsule/defence; defined in §4a), where the synonymous variation is lineage
sequence diversity. In practice this sets aside the **wzi** (capsule) synonymous hit; the iron synonymous
tags are retained.
![invasion variance by consequence](progress_figures/invasion_variance_by_consequence.png)

---

## §2 · Blood ↔ respiratory concordance — the strongest evidence the signal is real

Both contrasts share faeces as control and use the invasive niche as case, so a variant's β sign is
directly comparable: β>0 ⇒ the ALT allele is the invasion allele in *both*. Taking the **union of every
Bonferroni-significant variant in either niche** (165 variants: 110 blood + 88 resp, 33 shared) and
looking each up in *both* full associations:

- of the **137 testable in both cohorts, 136 (99.3%) agree in invasion direction**;
- collapsing perfect-LD clonal blocks to independent patterns, **85 of 86 patterns are concordant**
  (binomial sign-test **p ≈ 1.1×10⁻²⁴**);
- the effect sizes correlate in **magnitude** too: **r² = 0.78** (Pearson r = 0.88) for β_blood vs β_resp.

![blood↔resp concordance union](progress_figures/blood_resp_concordance_union.png)

So the hits are a **shared invasion signal across two independent invasive niches**, not blood-specific —
even though most are *below* the per-variant genome-wide bar in the smaller respiratory cohort. The strict
subset clearing Bonferroni in **both** is a cross-lineage **adhesion + capsule** signature — the
**fimbrial/pilus usher** (`KPN_RS24485`), **capsule *wzi*** (`KPN_RS13515`), and a **btuB** variant (same
SNP, same direction). Detail: [`blood_resp_concordance_union.tsv`](visualise/faeces_resp_lmm_model/blood_resp_concordance_union.tsv).

---

## §3 · Lineage breadth — species-wide, single-sublineage, or rare

For each hit we asked how its invasion allele is distributed across Kleborate sub-lineages (SL), by
intersecting the per-sample carriage matrix with SL labels ([`lineage_breadth.tsv`](visualise/faeces_resp_lmm_model/lineage_breadth.tsv);
reference = the 13,602-isolate blood/faeces cohort, which spans SL258/147/17/307 + many rare SLs).

![lineage breadth](progress_figures/lineage_breadth.png)

- **Common invasion alleles are pan-lineage.** All **110** common (REF) invasion alleles are
  **species-wide** (carried across hundreds of SLs; e.g. the riboflavin-synthase allele in 12,665 carriers
  across 356 SLs) — consistent with §1: the population-wide invasion-adapted background.
- **Rare invasion alleles split three ways:** **17 species-wide** (a derived allele recurring across many
  distantly-related SLs — see §5, convergence), **7 single-sublineage** (e.g. a **TonB siderophore**
  variant 88% in SL307, the **dnaK** variant 84% in SL307), **8 few-sublineage**, and **23 sub-1%** (very
  rare / lineage-private).

Because the LMM over-corrects (λ≪1) and the hits replicate across niches (§2), the **single-sublineage
hits are treated as real candidate clade-specific adaptations** — plausibly to that clade's acquired/HGT
accessory content — not as confounding. They are exactly the hits the accessory axes (§6) should
illuminate.

---

## §4 · What genes the hits are in

### §4a · Frequently-mutated vs real biology (the filter)

Some genes appear as hits simply because they are **hypervariable** — they accumulate sequence diversity
for reasons unrelated to invasion (capsule immune-evasion, phage-defence arms races). We flag these with
the per-niche **distinct-locus-richness Chi-sq**: a gene whose **synonymous** diversity is as enriched as
its non-synonymous (NS/syn ≈ 1, the `density/clade` verdict) is *just frequently mutated*; a gene with
**non-syn ≫ syn** carries codon-level selection.

![frequently-mutated vs selection](progress_figures/frequently_mutated_vs_selection.png)

Six hit genes are flagged hypervariable — **capsule *wzi*, cps K-locus (*gndA*, *ugd*, polysaccharide
export), a type-I restriction-modification defence gene, a Y-family DNA polymerase** — and set aside (their
synonymous hits dropped, §1). Note this is the **inverse** of a naïve "high NS/syn = noise" reading: the
genome-wide raw NS/syn baseline is **1.68** (≡ site-normalised dN/dS ≈ 0.56), and the genes *above* it are
the real ones (§4b/§5), while the hypervariable genes sit *on* the synonymous diagonal.

### §4b · The coherent invasion themes

With the hypervariable genes set aside, the named themes carrying invasion variance are
([`cross_axis_candidates.tsv`](cross_axis_candidates.tsv), Σ VE by category):

![invasion variance by category](progress_figures/invasion_variance_by_category.png)

- **Transcriptional regulators / de-repression** (16 genes, Σ VE ≈ 21) — the lab-focus theme. **RamA**
  (efflux activator), **mgrB** (PhoPQ feedback *repressor* — its loss is the canonical PhoPQ-de-repression
  route), **phoQ**, **qseC**, **acrR** (AcrAB *repressor*), **pdhR**. The organising question (to test, not
  assert) is **de-repression**: loss-of-function/missense in a *negative* regulator de-represses a stress /
  efflux / virulence pathway. Read in the invasion frame, *does the invasion allele carry the disruption or
  the intact form?* — table: [`regulator_derepression_table.tsv`](regulator_derepression_table.tsv),
  carrying consequence, `invasive_af`, dN/dS and the recurrent-mutation flag.
- **Iron / Fe-S acquisition** (7 genes, Σ VE ≈ 13) — **TonB siderophore receptor, an iron-redox enzyme,
  NfuA, *nadB*, *btuB***. This theme **replicates in direction** across to respiratory (§2) and several of
  its genes are independently-recurrent hotspots (§5) — a coherent pathway, the pattern expected of true
  invasion determinants.
- **Adhesion / fimbrial** (3 genes, Σ VE ≈ 8) — the cross-lineage **fimbrial usher** (§2).
- A large **diffuse remainder** (Σ VE ≈ 209 across 135 genes) — metabolic, membrane, and hypothetical
  loci, individually small; not yet a theme.

(Gene mechanisms are supporting annotation; the statistics lead. Whether a noncoding/regulatory change
alters expression is the experiment, not a claim here.)

---

## §5 · Independent-origin (recurrent-mutation) hotspots

A separate question from §4's *which genes*: of the invasion-hit genes, **which arose by recurrent,
phylogenetically-independent mutation** (convergent adaptation — the same gene hit repeatedly across the
tree) **vs which were acquired once and spread widely** (single origin, clonal dissemination)? The
collaborator's genome-wide **Poisson recurrent-mutation test** answers this directly: `is_sig` flags a
gene with significantly more independent mutations than expected (768 / 5,422 genes genome-wide, 14.2%),
and `poisson_dn_ds` (raw non-syn/syn, baseline 1.68) measures the protein-level pressure.

![independent-origin hotspots](progress_figures/independent_origin_hotspots.png)

**37 of the 167 invasion-hit genes are independent-origin hotspots.** They are precisely the §4b theme
genes: **RamA (dN/dS 38.5), mgrB (6.75), phoQ (2.23), the iron-redox enzyme (3.2), the TonB siderophore
receptor (2.23)** — repeatedly, independently selected ⇒ strong evidence of convergent invasion
adaptation. By contrast the **fimbrial usher (1.23), NfuA (1.0), *nadB* (0.75)** are *not* recurrent
hotspots — single-origin alleles spread clonally (cf. §3: the single-sublineage rare hits). This is **not**
an "arms race" reading: §1 already shows much invasion-directed mutation is population-wide; §5 adds *how*
it arose — recurrent independent origin vs single-origin spread.

The per-niche distinct-locus-richness Chi-sq (§4a's tool) is kept as a secondary cross-check: it asks
whether an invasion niche carries *more* of a gene's distinct variation than the background — a modest,
respiratory-weighted signal in the same regulators (`phoQ`, `mgrB`, `ramA`, `qseC`) + `sufB`; detail in
[`visualise/source_hotspot_chisq/`](visualise/source_hotspot_chisq/).

---

## §6 · Accessory axes — see the per-axis docs

The chromosomal variant axis cannot see plasmids or accessory genes absent from MGH 78578 — yet the
single-sublineage hits (§3) and the clade-adaptation reading point exactly there. Those axes now have their
own documents:

- **Unitig (accessory/HGT) LMM → [`PROGRESS_UNITIGS.md`](PROGRESS_UNITIGS.md).** *Headline:* the unitig axis
  is calibrated at **every** af (within-lineage permutation λ_perm ≈ 1), so its common-af inflation
  (observed λ up to 24) is **real, LD-redundant accessory signal**, not structure — there is **no af
  ceiling**. Next: map the hits to **chromosome / plasmid / virus** (geNomad) — the direct
  HGT-vs-chromosomal test — and DefenseFinder.
- **Panaroo gene presence/absence GWAS → [`PROGRESS_PANAROO.md`](PROGRESS_PANAROO.md).** Placeholder; **may
  not run** (Panaroo doesn't scale to ~80k genomes, and the unitig axis may already capture the accessory
  signal).

---

## §7 · Caveats & status

- **Between-strain association**, not a within-host gut→blood transition; chromosome-anchored only
  (plasmids invisible, §6); small per-variant effects (leads, not a classifier).
- **Population structure**: the LMM is conservative (λ<1), so clade-attributed hits are treated as
  plausibly real. The **variant** axis is **permutation-backed** — a within-lineage permutation null gives
  λ_perm≈1 at every af (§0), so residual *between-lineage* confounding is excluded (within-lineage and
  duplicate-isolate effects are not addressed by it). (The unitig axis is likewise validated —
  [`PROGRESS_UNITIGS.md`](PROGRESS_UNITIGS.md).)
- **Consequence ≠ causality**; **`invasive_af` ≠ mechanism** — common vs rare, and the consequence label,
  do not alone separate causal from regulatory from background-tag.
- **MDS (fixed-effects) was set aside at λ=4.34 — REVISIT; we may be kneecapping the variant results.**
  λ=4.34 was read as severe under-correction, but the unitig work showed a **high λ can be LD-redundancy of
  *real* signal** (clonal co-inheritance), not p-value inflation — so the MDS run may have been discarded
  prematurely, forcing the more conservative LMM-only (λ<1) view that could be **over-correcting real
  signal**. **Open question for the variant work:** recover the MDS run (`mds_cache` may still exist on RDS),
  test whether its λ=4.34 is redundancy vs genuine confounding (its own within-lineage permutation, exactly
  as we did for unitigs), and reconsider whether fixed-effects MDS recovers signal the LMM over-corrects.

**Status (variant axis).** Variant LMM (both contrasts) + invasion orientation + per-hit consequence +
blood↔resp concordance + lineage breadth + independent-origin hotspots: **done**; variant axis
**permutation-validated** (λ_perm≈1 at every af, §0). **Open / next (variant work — likely a dedicated
agent):** (1) **revisit the MDS abandonment** (above — λ=4.34 may be redundancy, not under-correction);
(2) deeper **hotspot mapping**; (3) a **Bakta re-annotation** of the reference for richer gene symbols.
Accessory axes are tracked separately: [`PROGRESS_UNITIGS.md`](PROGRESS_UNITIGS.md),
[`PROGRESS_PANAROO.md`](PROGRESS_PANAROO.md). Cross-task tracker: [`../../../ToDo.md`](../../../ToDo.md).
