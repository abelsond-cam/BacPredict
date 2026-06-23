# *Klebsiella* invasion GWAS — progress report

Single hub for the pyseer/GWAS analysis of *Klebsiella pneumoniae* **invasion** (blood / respiratory vs
faeces isolation source). The per-phase READMEs under [`visualise/`](visualise/) hold the method detail;
this page is the narrative + the cross-axis synthesis. **Posture: the tables and figures lead; the
statistics are the evidence; known gene mechanisms are annotation; conclusions are kept tentative.**

---

## 1. Purpose & question

Does genomic variation associate with the isolation source we use as an invasion proxy (blood and
respiratory = invasive niches; faeces = gut carriage)? The programme's larger question is **whether
invasion is driven more by horizontally-acquired (accessory/HGT) determinants or by chromosomal
changes** — so we look along several independent axes (§3) and ask which signals each surfaces.

## 2. Data & cohorts

One common reference for every sample: **`NC_009648`** (*K. pneumoniae* MGH 78578 chromosome,
5,315,120 bp). Variant calls are anchored to this chromosome **only** — the reference's five plasmids
(`NC_009649`–`53`) are **not** in the call set, so plasmid-borne accessory content is invisible to the
variant axis (§3). Per-contrast orientation ([`orientation_table.tsv`](orientation_table.tsv)):

| contrast | method | n | λ | variants tested | hits | invasion-direction hits |
|---|---|--:|--:|--:|--:|--:|
| blood vs faeces | LMM (FaST-LMM, core-SNP kinship) | 13,602 (faeces 6,426 / blood 7,176) | **0.562** | 372,238 | 110 | 18 |
| resp vs faeces | LMM (FaST-LMM, core-SNP kinship) | 9,169 (faeces 4,737 / resp 4,432) | **0.498** | 326,146 | 88 | 40 |

**λ < 1 in both.** The LMM random effect (a kinship built from the core SNPs themselves) is a
*conservative* — arguably *over-* — correction for population structure. A practical consequence worth
keeping in mind throughout: signals that survive this correction, including **clade-/lineage-attributed
hits, are unlikely to be pure structure artefacts** and deserve to be examined, not discounted.

## 3. The four analysis axes — what each can and cannot see

| axis | feature | sees | blind to | status |
|---|---|---|---|---|
| **Variant (core SNP)** | reference-anchored SNP/indel presence | chromosomal alleles, incl. SNPs *within* chromosomally-integrated islands | the 5 reference plasmids; any accessory gene absent from MGH 78578 | **done** (this report) |
| **Unitig (accessory/HGT)** | GGCAT coloured de-Bruijn unitigs | accessory sequence incl. plasmid/MGE content | — | in progress (matrices built; LMM run needs more memory) |
| **Hotspot (per-source diversifying)** | distinct-locus richness per gene per source | whether a niche accrues more variation in a gene | causality | **done** — secondary/orthogonal (§6) |
| **Gene presence/absence** | Panaroo GPA Rtab | gene gain/loss | within-gene change | planned |

The variant axis is the **chromosomal** view. The unitig and GPA axes are where **plasmid-borne /
accessory** adaptation should appear — the direct test of the HGT-vs-chromosomal question.

---

## 4. PRIMARY — niche-specific phenotype GWAS (the variant axis)

Method of record is the **LMM** (`pyseer --lmm`, FaST-LMM kinship); the fixed-effects MDS attempt was
abandoned (λ=4.34, [`visualise/mds_model/`](visualise/mds_model/)). Hits are ranked by variance
explained. Full detail: [`visualise/lmm_model/`](visualise/lmm_model/) (blood/faeces) and
[`visualise/faeces_resp_lmm_model/`](visualise/faeces_resp_lmm_model/) (resp/faeces, the replication
contrast). β>0 = the invasive side.

### 4.1 What associates

- **A cross-lineage adhesion + capsule signature replicates across both invasive niches**: the
  **fimbrial/pilus outer-membrane usher** (`KPN_RS24485`) and **capsule-assembly *wzi*** (`KPN_RS13515`)
  are significant with β>0 in *both* blood/faeces and resp/faeces — same SNP, same direction
  ([`replication_scatter`](progress_figures/replication_scatter.png),
  [`cross_contrast_overlap_blood_vs_resp.tsv`](visualise/faeces_resp_lmm_model/cross_contrast_overlap_blood_vs_resp.tsv)).
  Of the 18 blood-invasion hits, these two are the only ones reaching genome-wide significance in the
  invasion direction in respiratory.
- **Blood additionally carries an iron / Fe-S theme** (siderophore receptor, iron-redox enzyme, NfuA,
  BtuB, *nadB*) that does **not** replicate in respiratory — niche-specific, to be read alongside §4.2.
- Many hits are **single-Sublineage-attributed** (e.g. SL258, SL147, SL17, SL307). Given λ<1 (§2) these
  are treated as **plausibly real clade-level signals** — candidate clade-specific adaptation — not
  dismissed as confounding.

### 4.2 The consequence spectrum of the hits (the evidence)

Each hit's associated SNP is labelled by its reference consequence (SnpEff effect map). The breakdown
([`hit_consequence_spectrum.png`](progress_figures/hit_consequence_spectrum.png)):

| contrast · direction | synonymous | noncoding | missense | LoF |
|---|--:|--:|--:|--:|
| **blood — invasion (β>0)** | 10 | 8 | **0** | **0** |
| blood — faeces (β<0) | 36 | 14 | 40 | 2 |
| **resp — invasion (β>0)** | 15 | 15 | 7 | 3 |
| resp — faeces (β<0) | 16 | 10 | 22 | 0 |

**The observation** (a factual breakdown, *not* a conclusion): the **blood-invasion** hits carry **no
protein-coding changes** — every one is synonymous or noncoding — whereas the faeces direction and the
respiratory-invasion direction both include missense/LoF changes.

**What this does and does not mean.** It does **not** follow that the blood-invasion hits are
"non-functional" or mere lineage tags: **synonymous** SNPs can affect translation/mRNA, and **noncoding**
SNPs can hit promoters, RBSs, operators or sRNAs and be the actual driver of an expression-level
phenotype. Equally, any of them *may* tag an unobserved causal variant or a genetic background. The
consequence label alone cannot decide between these. Tentative readings to **test, not conclude**:

1. the blood-invasion association may act through **regulatory / expression-level** changes (the
   noncoding hits — see the regulatory-context columns in the hit table) rather than convergent
   protein-coding adaptation;
2. and/or it may reflect **clade-/accessory-background** structure that the chromosomal scan can only
   tag — directly testable on the unitig/GPA axes (§3, §7).

Respiratory's inclusion of missense/LoF in the invasion direction (e.g. an MdtB/MuxB efflux LoF, a
carbohydrate porin, *citC*; mostly SL258) is the complementary observation — there, coding-level change
*is* part of the invasion-associated signal.

---

## 5. Loss-of-function & regulator hits (focus)

Regulators/repressors that surface across the variant GWAS and the hotspot Chi-sq, with the consequence
of the associated change ([`regulator_derepression_table.tsv`](regulator_derepression_table.tsv); 16
hits). The organising question — to test, not assert — is **de-repression**: loss-of-function or
missense change in a *negative* regulator can de-repress a stress / efflux / virulence pathway.

- **From the hotspot Chi-sq (non-synonymous-enriched):** **`phoQ`** (PhoPQ sensor kinase), **`mgrB`**
  (PhoPQ feedback *repressor* — its LoF is the canonical colistin/PhoPQ-de-repression route), **`ramA`**
  (AcrAB-TolC efflux activator), **`qseC`** (quorum-sensing sensor kinase).
- **From the phenotype GWAS:** **`acrR`** (AcrAB efflux *repressor*; noncoding change), **`pdhR`**
  (pyruvate-dehydrogenase *repressor*; noncoding), `ecpR`, and several LysR/LacI/MerR-family regulators
  (missense); plus a respiratory **MdtB/MuxB efflux** LoF.

These are candidate de-repression events; whether the noncoding/regulatory changes alter expression is
the experiment, not a claim here.

---

## 6. SECONDARY (orthogonal) — hotspot 'arms-race' Chi-sq

A different question from §4: *does an invasion niche accrue **more distinct functional variation** in a
gene than the background — i.e. is there diversifying selection / an "arms race" behind the phenotype?*
Distinct-locus richness per gene per source, two controls — **population-vs-niche-specific** (niche-pair
vs one-vs-rest contrasts) and **synonymous** (neutral baseline). Detail + the ancestry-correction note:
[`visualise/source_hotspot_chisq/`](visualise/source_hotspot_chisq/). Evidence
([`hotspot_codon_vs_clade.png`](progress_figures/hotspot_codon_vs_clade.png)) splits the 42 significant
genes into two classes:

- **Codon-level functional** (non-syn share-ratio ≫ synonymous): the regulators of §5 (`phoQ`, `mgrB`,
  `ramA`, `qseC`) + **`sufB`** (Fe-S; the one cross-niche functional hit). Modest effects (1.3–2.8×),
  synonymous-clean — a respiratory-weighted signal of functional diversification.
- **Clade-linked sequence diversity** (non-syn ≈ synonymous): the **cps K-locus** (~2.72–2.75 Mb),
  a chromosomally-integrated **pathogenicity/defence island** (~5.14 Mb: a *"STY4528 pathogenicity-island
  replication protein"* + ParB + the **DISARM** anti-phage system + RM + an Abi toxin), and a **prophage**
  gene (~1.59 Mb). The synonymous control shows these are **not** under codon-level diversifying selection,
  but they *are* real between-clade diversity in integrated mobile elements — consistent with §2/§4.1's
  clade-adaptation reading, not a confound to discard.

This axis is **secondary**: it qualifies the phenotype hits (which are diversifying vs which are
clade-structured), it does not establish association.

## 7. Unitig (in progress) & Panaroo GPA (planned)

The accessory/HGT axes — where plasmid-borne adaptation invisible to §4 should surface. Unitig matrices
are built (per cohort, GGCAT, 1% MAF); the blood/faeces unitig LMM run OOM'd at 128 GB and needs a larger
allocation (or a harder MAF pre-filter). GPA awaits the per-SL Panaroo outputs. **Placeholder** — fill on
completion.

## 8. Cross-axis synthesis

[`cross_axis_candidates.tsv`](cross_axis_candidates.tsv) — one row per gene across the union of
significant hits (167 genes): variant blood/resp β, VE, **consequence**, lineage; replication flag;
`is_regulator` / `mobile_element` flags; hotspot verdict; unitig placeholder. The two genes significant
and invasion-concordant across both variant contrasts are the **fimbrial usher** and **Wzi**.

## 9. Caveats

- **Population structure.** The LMM is conservative (λ<1, §2), so clade-attributed hits are treated as
  plausibly real; but residual structure cannot be *fully* excluded — the unitig/GPA axes and explicit
  clade analyses are the resolving follow-ups.
- **Chromosome-anchored.** The variant axis cannot see the reference plasmids or accessory genes absent
  from MGH 78578 (§3).
- **Consequence ≠ causality.** §4.2 — synonymous/noncoding are not assumed non-functional.
- **Modest hotspot effects** (§6), and it is a screen, not proof.

## 10. Status & next

Variant LMM (both contrasts) + per-hit consequence + hotspot Chi-sq: **done**. Next: re-run the unitig
LMM (more memory), then GPA; Bakta re-annotation of the reference for richer gene symbols. Cross-task
tracker: [`../../../ToDo.md`](../../../ToDo.md).
