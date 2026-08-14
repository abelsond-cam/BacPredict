# Where the *Klebsiella* invasion unitig signal sits — IGR vs CDS, and plasmid vs chromosome

> **Living doc.** The genomic-context analysis of the blood/faeces unitig-GWAS hits. Three panels:
> (1) IGR-vs-CDS bias with a significant-vs-non-significant control, (2) plasmid/prophage enrichment
> vs base rate, (3) the two axes crossed. Companion to [`PROGRESS_UNITIGS.md`](PROGRESS_UNITIGS.md)
> (the MGE/geNomad + ISEScan mapping). **These are statistics + hypotheses — mechanism is untested.**

## Question & method

The unitig axis of the invasion GWAS is real accessory signal, concentrated at common allele frequency
([`PROGRESS_UNITIGS.md`](PROGRESS_UNITIGS.md)). Here we ask **what kind of DNA carries it** — coding
(CDS) or intergenic (IGR), and on which replicon — by measuring, for every hit unitig in every carrier
that has it, the **base-pair fraction of the placement that lies in IGR** (`igr_frac`), then reporting
robustly across thresholds: `entirely-CDS` (igr_frac = 0), `touch` (>0), `significant` (≥0.25),
`predominant` (≥0.5), `entirely-IGR`. Unit = the hit unitig (each once, by its behaviour across
carriers) with a placement-weighted view alongside.

Every observed number is weighed against a matched **uniform-placement null** — if the GWAS were
spatially random, how would `igr_frac` fall? — computed by sliding each hit-unitig length across each
genome's coding/non-coding architecture. Two further controls: an **af-matched non-significant unitig
set** (is the bias specific to the hits, or general to divergent unitigs?), and **genome base-rate bp
fractions** (plasmid/prophage/chromosome, and CDS/IGR within each) for the MGE enrichment.

Pipeline (all reuse the generic `unitig_placement` select/placement engine; the coding classifier is
`genome_prep.CodingIndex`): `annotate_unitig_coding` (coverage), `coding_null_model` (uniform null),
`sample_nonsig_unitigs` + `nonsig_coding.sh` (control), `genome_coding_fraction` /
`genome_mge_fraction` (base rates), `coding_by_mge` (the cross). Bakta GFF3 (BakRep) `seqID`s are
concordant with the seb assemblies the unitigs were placed on (verified; 0 discordance / 670 pairs).

Scale: significant set **33,039 unitigs × 13,171 carriers = 108,796,538 placements** (15 no-asm-hit,
100% Bakta); non-significant control **100,005 unitigs × 13,171 carriers = 329,387,330 placements**.
Genome bp baseline **87.4% CDS / 12.6% IGR** (12.1% of it Bakta-unclassified — where promoters live).

## Panel 1 — the invasion signal is depleted from coding sequence, and it is specific to the hits

Placement-weighted fraction (per-unitig, majority-of-carriers, in parentheses):

| threshold | uniform null | non-significant | **significant hits** | sig ÷ null | sig ÷ non-sig |
|---|--:|--:|--:|--:|--:|
| entirely-CDS | 0.848 | 0.834 | **0.645** (0.724) | — | — |
| touch IGR | 0.152 | 0.166 | **0.355** (0.276) | **2.3×** | **2.1×** |
| ≥0.25 IGR | 0.137 | 0.153 | **0.338** (0.257) | 2.5× | 2.2× |
| predominant ≥0.5 | 0.124 | 0.141 | **0.314** (0.239) | 2.5× | 2.2× |
| entirely IGR | 0.102 | 0.120 | **0.250** (0.199) | 2.5× | 2.1× |

- **The enrichment is ~2.3–2.5× the null and flat across every threshold** — including unitigs lying
  *wholly* in IGR (~2.5×). By any reasonable measure ≈28–36% of hit unitigs cover intergenic DNA and
  ≈20–25% sit entirely in it, versus a ~15% null.
- **It is specific to the significant hits.** The af-matched non-significant set sits **at the null**
  (0.166 vs 0.152 touch — 1.09×), while the hits are **2.1× the non-significant set**. af-matching was
  exact (both sets 35.4 / 34.6 / 8.9 / 3.0 / 18.0 % across the af bins), so this is not an af artifact.
  So the IGR bias is **not** a general property of divergent unitigs — random divergence sits ≈at
  genome proportions; only the phenotype-associated signal avoids coding sequence.
- **Blood > faeces**, both far above null (placement entirely-CDS blood 0.627 / faeces 0.685; nulls
  identical at ~0.85, so the difference is real). Flat across every sublineage / clonal group.

## Panel 2 — plasmid/prophage enrichment splits by direction (faeces MGE-borne, blood not)

Base rate (bp): chromosome **0.806**, plasmid **0.164**, prophage **0.030**. Unitig placement fraction
(geNomad, [`PROGRESS_UNITIGS.md`](PROGRESS_UNITIGS.md)) ÷ base rate = enrichment:

| direction | chromosomal (obs / enrich) | plasmid (obs / enrich) | prophage (obs / enrich) |
|---|--:|--:|--:|
| ALL | 0.614 / 0.76× | 0.327 / **2.0×** | 0.043 / 1.4× |
| blood (invasion) | 0.823 / **1.02×** | 0.172 / **1.05×** | 0.005 / 0.2× |
| faeces | 0.153 / 0.19× | 0.669 / **4.1×** | 0.128 / **4.3×** |

- **The plasmid/prophage (HGT) enrichment is a faeces phenomenon** — faeces hits are ~4× enriched on
  both. **The blood/invasion signal is *not* plasmid-enriched** — it sits at the genome base rate for
  plasmid (1.05×) and chromosome (1.02×), i.e. essentially chromosomal.
- Combined with Panel 1: **blood/invasion = chromosomal + intergenic** (IGR-enriched but not
  plasmid-enriched), whereas **faeces = MGE-borne *and* intergenic**.

## Panel 3 — IGR enrichment holds on both the chromosome and plasmids

Per-partition IGR bp base rate: chromosome **0.121**, plasmid **0.153** (plasmids are slightly more
intergenic at baseline). Observed touch-IGR within each geNomad class (placement-weighted):

| geNomad class | ALL | blood | faeces | partition IGR base |
|---|--:|--:|--:|--:|
| chromosomal | 0.392 | 0.377 | 0.558 | 0.121 |
| plasmid | 0.314 | 0.363 | 0.288 | 0.153 |
| prophage | 0.152 | 0.097 | 0.157 | — |

- **The signal avoids CDS on both replicon types** — touch-IGR is ~2.6–3× the base rate on the
  chromosome and ~1.7–2× on plasmids. So "the signal is intergenic" is **replicon-independent**, not a
  chromosome-only artifact; it is strongest on the chromosome.
- Faeces chromosomal placements are **>50% IGR-touching** (0.558); blood prophage placements are almost
  entirely coding (0.90 entirely-CDS).

## A note on counts (placements vs unique unitigs)

The stratify tables show blood ≈2–3× faeces by `n_pairs`, which can mislead. Unique hit unitigs are
**faeces-heavy** (25,277 vs 7,762, 3.3×); *placements* are **blood-heavy** (74.9M vs 33.9M, 2.2×). The
flip is the common-vs-rare af structure — blood hits are common (many carriers each), faeces hits rare
— already documented; it is not a separate signal.

## Working hypotheses (untested — for discussion, not conclusions)

1. **The invasion signal carries a selected intergenic/regulatory component.** ~2.3–2.5× IGR over
   chance, specific to the significant hits (non-sig at null), on both chromosome and plasmid —
   consistent with selection on expression-level / regulatory variation, not only protein change. We
   cannot yet say "promoter": the IGR is overwhelmingly Bakta-*unclassified* (promoter-candidate) —
   the deferred promoter-annotation step.
2. **Two distinct niche signatures.** Blood/invasion = chromosomal + intergenic (not HGT/plasmid);
   faeces/commensal = MGE-borne (plasmid+prophage ~4×) *and* intergenic. The "acquired/HGT" reading
   applies to the faeces side, not the invasion side.
3. **A candidate general property to test next:** significant unitigs avoid coding sequence while
   random divergence does not — David's "coding conserved" intuition may live at the *selected-signal*
   level rather than the *all-divergence* level. Core-vs-accessory (Nuna pangenome) would test whether
   core coding is even more conserved and CDS hits largely reflect accessory genes.

## Next
- **Promoter annotation** of the unclassified-IGR hits (software that calls promoters, which Bakta does
  not) — is the intergenic signal promoter-concentrated?
- **Variant-axis IGR** on the all-variants `.assoc` — the correct instrument for "are the recurrent /
  Poisson hotspots IGR-enriched?" (the per-gene Poisson table is CDS/RNA-only and cannot show IGR).

## Provenance
Outputs on project_k `…/blood_faeces/sampled_country_2_1_all/gwas_unitig_lmm/`:
`coding_mapping/` (significant: `coding_overall.tsv`, `coding_by_{sublineage,clonal_group}.tsv`,
`coding_hits.parquet/`, `coding_null_allthresholds.json`, `genome_coding_baseline.json`,
`genome_mge_baseline.json`, `genome_coding_by_mge_baseline.json`, `coding_by_mge.tsv`),
`coding_mapping_nonsig/` (the af-matched control). Code: `src/bac_pyseer/kleb_iso_source/` commits
`b0c1d10` (unitig_placement extraction) · `8bb3832` (all-threshold null) · `9423959` (non-sig control)
· `0c1ae5e` (MGE base rate + cross).
