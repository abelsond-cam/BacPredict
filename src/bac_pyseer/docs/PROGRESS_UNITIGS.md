# *Klebsiella* invasion GWAS — unitig (accessory / HGT) axis

> **Axis docs.** This is the **unitig (accessory/HGT)** axis. The chromosomal-variant axis is
> [`PROGRESS_VARIANTS.md`](PROGRESS_VARIANTS.md); gene presence/absence is
> [`PROGRESS_PANAROO.md`](PROGRESS_PANAROO.md); cross-axis overview is [`PROGRESS.md`](PROGRESS.md).
> Pipeline/run detail: [`../kleb_iso_source/CLAUDE.md`](../kleb_iso_source/CLAUDE.md).

**Why this axis.** The variant axis is anchored to one chromosome (MGH 78578) and is **blind to plasmids and
to accessory genes absent from that reference** — exactly where the single-sublineage / clade-adaptation hits
(PROGRESS_VARIANTS §3) point. The unitig axis searches the **whole accessory/HGT sequence space**: GGCAT
coloured de-Bruijn **unitigs (6.28M)** across the blood/faeces cohort, tested by `pyseer --lmm` with the
**same core-SNP kinship as the variant axis** (so the two are directly comparable; the core-SNP kinship
corrects clonal/phylogenetic structure while leaving accessory content free to associate).

## How it's run — chunked for memory

pyseer's `--kmers` mode **accumulates memory across the scan** and OOMs on the full 6.28M in one process (even
at 200 GB). So every unitig LMM — the production run *and* the permutation null — **primes the kinship cache
once, then scans ~100k-unitig chunks as separate processes** (memory freed between, ~26 GB peak each) and
concatenates the per-unitig results. Practical settings that matter: `--cpu ≈ 8` (heavy per-worker unitig
blocks), **gzipped** chunk input (pyseer `--kmers` requires gzip), and a **fresh** `--save-lmm` prime +
`--load-lmm` per chunk. Scripts: [`../kleb_iso_source/scripts/run_unitig_lmm_sharded.sh`](../kleb_iso_source/scripts/run_unitig_lmm_sharded.sh)
(production), [`../kleb_iso_source/scripts/permute_unitig_lambda.sh`](../kleb_iso_source/scripts/permute_unitig_lambda.sh) (permutation null).

## Calibration — which population correction, and does it actually work?

**Two candidate corrections for clonal structure.** pyseer offers two ways to stop lineage from driving false
associations, and we compared them:

1. **LMM (linear mixed model)** — the FaST-LMM engine (`pyseer --lmm --similarity K`). Structure enters as a
   **random effect** through a kinship matrix `K`; here `K` is the **core-SNP kinship** (identical to the
   variant axis). This is the method we adopted.
2. **MDS fixed-effects** — the SEER-style model (`pyseer --distances D --max-dimensions N`). Structure enters
   as **fixed covariates**: the leading metric-MDS dimensions of a genome-distance matrix are regressed out
   before testing. This is the alternative we tested against, and **rejected** (below).

**The test — a within-lineage shuffle (permutation null).** Genome-wide λ = median(observed χ²)/median(null χ²)
mixes *real signal* with *residual structure*, so λ on the observed data cannot tell them apart. The decisive
test destroys the signal while **keeping the structure**: shuffle the blood/faeces label **within each
lineage**, re-fit the model **fresh**, and recompute a **structure-only** λ_perm. If the correction is
working, λ_perm collapses to ~1 (no structure leaks through); if structure is leaking, λ_perm stays inflated.
We ran this shuffle at **two resolutions of "lineage"** — **sublineage (SL)** and **clonal group (CG)** — as
independent checks that the result isn't an artifact of how coarsely lineage is defined.

**Result — the LMM controls structure at both SL and CG; the MDS model does not.**

*LMM, sublineage shuffle* (1,345 SLs; between-SL structure preserved, within-SL signal destroyed):

| af bin | 0.01–0.05 | 0.05–0.10 | 0.10–0.15 | 0.15–0.20 | 0.20–0.50 | 0.50–0.70 | 0.70–1.0 |
|---|--:|--:|--:|--:|--:|--:|--:|
| observed λ | 0.80 | 1.09 | 1.91 | 2.32 | 2.7→4.2 | 6.81 | 23.8 |
| **LMM null λ_perm** | 0.91 | 0.80 | 1.05 | 1.29 | 1.1–1.3 | 1.06 | **0.43** |

The **LMM null never runs away** — λ_perm ≈ 1 (0.43–1.29) at *every* af, including the common end where
*observed* λ hits 24. Because the shuffle *preserves* between-lineage structure, a structure artifact would
keep λ_perm inflated there; it collapses instead. **Repeating the shuffle at clonal-group (CG) resolution gave
the same collapse** — so the conclusion is not sensitive to the SL-vs-CG definition of lineage. In contrast,
the **MDS fixed-effects correction did *not* ablate** under the same permutation (its structure-only λ_perm
stayed well above 1, ~3.5–4 at the common end vs the LMM's ~1) — i.e. MDS covariates leave residual
lineage signal in the test, so the MDS model is **not** an adequate correction here. **The LMM (core-SNP
kinship) is the method of record on both axes**; the variant axis is permutation-validated the same way
(PROGRESS_VARIANTS §0).

> **Naming, for the record** (David's question): the two models are the **LMM / mixed-model with kinship**
> (`--lmm --similarity`, adopted) and the **MDS fixed-effects / distance-covariate model**
> (`--distances --max-dimensions`, rejected). "LM"/"MDS" in earlier notes refer to these. The MDS-model
> λ_perm numbers are Agent B's permutation outputs; the of-record LMM table above is
> [`genomic_inflation_by_af.tsv`](genomic_inflation_by_af.tsv) (rows `blood_vs_faeces_unitig_fine`,
> `unitig_permnull`); tool [`../kleb_iso_source/genomic_inflation_by_af.py`](../kleb_iso_source/genomic_inflation_by_af.py).

**So the common-af inflation is genuine signal, not structure.** There is **no structure-driven af ceiling** —
common-af accessory unitigs are exactly where the invasion signal concentrates (the observed-vs-null gap is
largest at the common end). *(A **mash** whole-genome k-mer kinship was also tried as an accessory-aware
control and was a **trade-off, not a fix** — common 21→3.6 but rare 0.80→4.4 breaks, h²=0.60 — moot now that
the common end is signal, not structure to "fix".)*

## How to read the hits — LD-redundant, not independent

*Klebsiella* is clonal and accessory DNA travels in large co-inherited blocks: a single niche-associated
**megaplasmid (1–2 Mb)** contributes a unitig — often several, from within-plasmid mutational variants — at
*each* locus, so **λ = 24 is one (or a few) biological events multiplied across thousands of co-inherited
unitigs**, not 24× independent findings. Report at the **independent-pattern / locus** level (pyseer's
pattern-count Bonferroni already does this for significance), never per-unitig.

## What the unitigs are — mapping the hits to their genomic home (geNomad + ISEScan)

Calibration establishes the signal is real; the next question is **what kind of DNA carries it** — the direct
test of the programme's central HGT-vs-chromosomal hypothesis. We took **all 33,039 significant hit unitigs**
and located every one in **every carrier that has it** — **13,171 carriers, 108,796,553 unitig×carrier
placements, ASM-recall 1.0** (every expected unitig found; 0 found-not-expected). This is exhaustive, not a
sample.

**Method.** A unitig from a coloured de-Bruijn graph is an **exact substring** of the assemblies that carry
it, so mapping is exact-match, not alignment. One **Aho-Corasick** automaton holds all 33,039 unitigs on both
strands; each carrier's genome is streamed through it once (cost ∝ genome size, not × unitigs). Each placement
is then classified against that carrier's **geNomad** output — **plasmid > prophage > chromosome** (geNomad
detects plasmids and integrated prophages only) — and, independently, checked for overlap with that carrier's
**ISEScan** IS-element calls (short-read ISEScan, 100% carrier coverage). Coordinate overlap was independently
cross-checked (0 discordance in 4,800 spot-checks). Module:
[`../kleb_iso_source/map_unitig_hits_genomad.py`](../kleb_iso_source/map_unitig_hits_genomad.py); driver
[`../kleb_iso_source/scripts/map_unitig_hits_genomad.sh`](../kleb_iso_source/scripts/map_unitig_hits_genomad.sh).
Outputs on project_k under `…/gwas_unitig_lmm/mge_mapping/` (`mge_overall.tsv`, `mge_unitig_class.tsv`,
`mge_pattern_group.tsv`, `mge_by_{sublineage,clonal_group}.tsv`, `mge_hits.parquet/`, `combine_manifest.json`).

### geNomad partition — the blood/faeces contrast is chromosome-vs-plasmid, and af-driven

| direction | chromosomal | plasmid | prophage |
|---|--:|--:|--:|
| **blood (invasion)** | **0.823** | 0.172 | 0.005 |
| **faeces** | 0.153 | **0.669** | **0.128** |

| af bin | chromosomal | plasmid | prophage |
|---|--:|--:|--:|
| (0.0, 0.05] | 0.067 | 0.582 | 0.329 |
| (0.05, 0.2] | 0.052 | 0.647 | 0.225 |
| (0.2, 0.5] | 0.165 | 0.798 | 0.007 |
| (0.5, 0.7] | 0.615 | 0.363 | 0.003 |
| (0.7, 1.0] | 0.848 | 0.150 | 0.002 |

Blood-favouring unitigs are ~82% **chromosomal**; faeces-favouring unitigs are ~67% **plasmid** + ~13%
**prophage**. This tracks almost perfectly with allele frequency: the chromosomal fraction rises monotonically
with af (7% at rare → 85% at common), and the blood/faeces label is largely the same af axis seen before
(blood ≈ common/reference-like; faeces ≈ derived/rare). Faeces MGE *type* (plasmid vs prophage) varies by
clonal group; see `mge_by_clonal_group.tsv`.

### ISEScan IS-element layer — IS is **not** the hidden home of the "chromosomal" signal

geNomad sees plasmids and prophages but **not IS elements / transposons**, so "chromosomal" could silently
absorb IS-borne unitigs — the obvious place a mobile home for the derived signal could hide. Splitting each
class by IS overlap (108.8M placements):

| direction | chromosomal | **chromosomal_IS** | plasmid | plasmid_IS | prophage |
|---|--:|--:|--:|--:|--:|
| all | 0.614 | **0.0032** | 0.327 | 0.0123 | 0.043 |
| **blood (invasion)** | 0.823 | **0.0001** | 0.172 | 0.0001 | 0.005 |
| **faeces** | 0.153 | **0.0101** | 0.669 | 0.039 | 0.128 |

IS overlap is **small everywhere and near-zero exactly where it would have mattered.** The blood/invasion
chromosomal fraction has **0.01%** IS overlap — the "chromosomal" 82% is genuinely chromosomal, not an
ISEScan blind spot. What little IS signal exists is *relatively* enriched on the **derived/faeces/low-af**
side (peaks at af 0.05–0.2: chromosomal_IS 1.7%, plasmid_IS 5.8%), consistent with recent, rare, lineage-
restricted IS insertions. Copy-number agrees: median unitig is single-copy (mean `mean_copies` 1.03), with a
thin multi-copy IS-borne tail. Dominant IS families among the ~26k IS-overlapping unitigs are the usual
Enterobacteriaceae mobile families (IS5, IS3, IS1, IS21, IS6, IS66). **One lineage stands out — SL147/CG147**
carries ~4% chromosomal_IS and ~10% plasmid_IS on the faeces side (vs ~1% elsewhere), the only clonal group
where IS is a non-trivial home.

### Working hypotheses (untested — for discussion, not conclusions)

Per the programme's "gather → hypothesise → stop" discipline, these are candidate readings to take away, **not
findings**:

- **IS elements are not the mobile carrier of the invasion signal.** Blood-favouring unitigs sit on true
  chromosome or plasmid, essentially never on IS — so the geNomad "chromosomal" bucket is real, not an IS
  artifact.
- **The invasion (blood, common-af) signal is split between chromosome (~82%) and plasmid (~17%).** Whether
  the plasmid share is *causal* (acquired/HGT) or a *lineage correlate* (co-inherited megaplasmid marking a
  clone) is **not** resolved by this mapping — it is the LD-redundancy caveat above, and remains open.
- **The faeces / derived / low-af signal is predominantly MGE-borne** (plasmid + prophage), with a small,
  lineage-restricted IS component — an accessory-genome phenomenon on the commensal side.
- **SL147/CG147 merits a closer look** as a lineage with unusually active IS mobilisation.

## Next work

1. **DefenseFinder** mapping of the hit regions — anti-phage / defence systems carried on the accessory signal.
2. **Unitig blood↔resp concordance** (mirror PROGRESS_VARIANTS §2) — the check that the signal is *invasion*,
   not clonal sampling; run the **faeces↔respiratory** unitig LMM and compare β direction/magnitude.
3. **Pattern/locus-level annotation** — collapse to independent patterns, map unitig → gene/region, reconcile
   against the variant-axis hit genes; resolve the causal-vs-lineage question for the plasmid share.

**Planned figures:** unitig Manhattan + unitig↔variant gene overlap; unitig blood↔resp concordance; an
**UpSet** of hit overlap across axes; the **chromosome-vs-plasmid-vs-prophage partition** of the invasion
signal (geNomad, above) and its **IS split** (ISEScan).

## Status

Unitig LMM (blood/faeces): **run + calibration resolved + hits mapped.** Calibration — LMM (core-SNP kinship)
permutation-validated at SL **and** CG resolution; MDS fixed-effects rejected; no af ceiling; signal
concentrated at common af and LD-redundant. Mapping — **all 33,039 hits placed across all 13,171 carriers**
(108.8M placements, ASM-recall 1.0): invasion signal ~82% chromosomal / ~17% plasmid, faeces signal
MGE-borne, and **IS elements are not the hidden home of the chromosomal fraction**. **Open:** DefenseFinder (1),
unitig blood↔resp concordance + faeces↔resp run (2), locus-level annotation + causal-vs-lineage on the plasmid
share (3). Cross-task tracker: [`../../../ToDo.md`](../../../ToDo.md).
