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

## Calibration — the common-af inflation is real signal, not structure (resolved)

The blood/faeces run completed cleanly but the genome-wide **λ = 3.42** is inflated, rising with af on the
**observed** data. Observed λ conflates *real signal* with *structure*, so the decisive test was the
**within-lineage permutation null on the unitig axis** — shuffle the phenotype within each of 1,345
sublineages (between-lineage structure **preserved**, real signal **destroyed**), re-fit the LMM fresh,
recompute the **structure-only** λ_perm:

| af bin | 0.01–0.05 | 0.05–0.10 | 0.10–0.15 | 0.15–0.20 | 0.20–0.50 | 0.50–0.70 | 0.70–1.0 |
|---|--:|--:|--:|--:|--:|--:|--:|
| observed λ | 0.80 | 1.09 | 1.91 | 2.32 | 2.3→4.2 | 6.81 | 23.8 |
| **null λ_perm** | 0.91 | 0.80 | 1.05 | 1.29 | 1.1–1.3 | 1.06 | **0.43** |

**The null never runs away** — λ_perm ≈ 1 (0.43–1.29) at *every* af, including the common end where observed
λ hits 24. Because the permutation *preserves* the between-lineage structure, a structure artefact would keep
λ_perm inflated there; it collapses to ~1 instead. So **the core-SNP kinship controls population structure at
all af**, and the observed common-af inflation is **genuine within-lineage invasion signal**, not confounding.
**There is no structure-driven af ceiling** — common-af accessory unitigs are exactly where the invasion
signal concentrates (the observed-vs-null gap is largest at the common end). The variant axis is likewise
permutation-validated (PROGRESS_VARIANTS §0), so the kinship is sound on both.
Data: [`genomic_inflation_by_af.tsv`](genomic_inflation_by_af.tsv) (rows `blood_vs_faeces_unitig_fine`,
`unitig_permnull`); tool: [`../kleb_iso_source/genomic_inflation_by_af.py`](../kleb_iso_source/genomic_inflation_by_af.py).

*(A **mash** whole-genome k-mer kinship was tried as an accessory-aware control and was a **trade-off, not a
fix** — common 21→3.6 but rare 0.80→4.4 breaks, h²=0.60 — moot now that the common end is signal, not
structure to "fix".)*

## How to read the hits — LD-redundant, not independent

*Klebsiella* is clonal and accessory DNA travels in large co-inherited blocks: a single niche-associated
**megaplasmid (1–2 Mb)** contributes a unitig — often several, from within-plasmid mutational variants — at
*each* locus, so **λ = 24 is one (or a few) biological events multiplied across thousands of co-inherited
unitigs**, not 24× independent findings. Report at the **independent-pattern / locus** level (pyseer's
pattern-count Bonferroni already does this for significance), never per-unitig.

## Next work

1. **Map hits to chromosome / plasmid / virus with geNomad** (already run on the cohort) — the **direct
   HGT-vs-chromosomal test**, and the through-line to the programme's central hypothesis: is the common-af
   invasion signal **plasmid/MGE-borne (acquired/HGT)** or chromosomal?
2. **DefenseFinder** mapping of the hit regions — anti-phage / defence systems carried on the accessory signal.
3. **Unitig blood↔resp concordance** (mirror PROGRESS_VARIANTS §2) — the check that the signal is *invasion*,
   not clonal sampling; run the **faeces↔respiratory** unitig LMM and compare β direction/magnitude.
4. **Pattern/locus-level annotation** — collapse to independent patterns, map unitig → gene/region (deferred
   bwa step), reconcile against the variant-axis hit genes.

**Planned figures:** unitig Manhattan + unitig↔variant gene overlap; unitig blood↔resp concordance; an
**UpSet** of hit overlap across axes; and a **chromosome-vs-plasmid-vs-virus partition** of the invasion
signal (geNomad).

## Status

Unitig LMM (blood/faeces): **run + calibration resolved** — permutation-validated, no af ceiling, signal
concentrated at common af and LD-redundant. **Open:** geNomad chromosome/plasmid/virus mapping (1),
DefenseFinder (2), unitig blood↔resp concordance + the faeces↔resp run (3), locus-level annotation (4).
Cross-task tracker: [`../../../ToDo.md`](../../../ToDo.md).
