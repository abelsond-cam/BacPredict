# Hotspot / dN-dS annotation of the invasion-GWAS hits

Annotates the LMM variant-GWAS hits with a **combined whole-population per-gene dN/dS +
variant-density hotspot table**, and tests whether invasion-associated genes are under stronger
positive selection / more often mutational hotspots than the genome background.

Built by [`annotate_hits_with_hotspots.py`](../../../kleb_iso_source/annotate_hits_with_hotspots.py).

## Inputs

- **Hotspot table** — `src/bac_pyseer/data/combined_poisson_test_variant_hotspots.txt` (Aaron's
  combined Poisson test): one row per gene on the **MGH 78578 / `KPN_RS`** annotation shared with
  the variant GWAS. 5,422 genes; **768 flagged `is_sig`** (carries more variants than the Poisson
  background expects). `dn_ds = n.y/n.x` = non-synonymous / synonymous variant count (a crude dN/dS
  proxy); `padj` = BH-adjusted hotspot p.
- **GWAS hit tables** — the two LMM contrasts: `lmm_model/blood_vs_faeces_hits_annotated.tsv`
  (110 hits) and `faeces_resp_lmm_model/respiratory_vs_faeces_hits_annotated.tsv` (88 hits).

## Method

Merge each hit onto the hotspot table by `locus_tag` → hotspot columns prefixed `hot_`
(`hot_dn_ds`, `hot_is_sig`, `hot_padj`, `hot_unique_variants`, syn/non-syn counts). Enrichment over
the set of **distinct hit genes** vs the **tested-gene background** (the 5,422 genes the hotspot
pipeline saw): Fisher 2×2 on `is_sig` + Mann–Whitney on `dn_ds`, run for **all hit genes** and for
**invasion-direction** (β>0) hit genes. `dn_ds = inf` (syn = 0) dropped from the MWU.

> **Confound (documented).** Hit genes are by definition *variable* genes, and `is_sig` also keys on
> variant count — so some hit↔hotspot association is mechanical. Using the tested-gene set as
> background (already conditioned on being polymorphic) partially controls this; the dN/dS shift is
> the less-confounded signal. The **per-isolation-source hotspot-rate Chi-sq** (planned next) is the
> clean, invasion-specific test.

## Result — two distinct selective regimes among the blood-invasion hits

**blood/faeces hit genes are enriched for hotspots overall** — 27.3% `is_sig` vs 13.9% background
(OR 2.3, **Fisher p = 4.3e-4**); dN/dS marginally higher (1.86 vs 1.69, MWU p = 0.07). **But this is
*not* specifically an invasion signal:** restricted to invasion-direction (β>0) genes it is
underpowered/null (15 genes, p = 0.15; dN/dS *not* elevated, median 1.39). The aggregate enrichment
is largely carried by **faeces-direction (gut sub-clade) markers** — 26 of them are hotspot-`sig`
(e.g. RcnA Ni/Co efflux, a MerR regulator at dN/dS 4.6, a fimbrial usher, peptidases).

Within the invasion-direction hits the signal **splits by function**:

- **Iron-acquisition surface receptors ARE positively-selected hotspots** — a host-iron-restriction
  arms-race signature that refines the original blood iron story:
  | gene | product | β | dN/dS | hotspot padj |
  |---|---|--:|--:|--:|
  | KPN_RS09430 | iron-containing redox enzyme | +0.29 | **3.20** | 5.3e-4 |
  | KPN_RS11350 | TonB-dependent siderophore receptor | +0.39 | **2.23** | 0.014 |
  | KPN_RS11695 | hypothetical protein | +0.10 | 2.52 | 2e-6 |
  | hpxZ | oxalurate catabolism (purine) | +0.16 | 2.50 | 7e-3 |

- **Capsule / adhesion / chaperone invasion genes are NOT hotspots and are ~conserved** — their
  invasion-associated SNPs are *specific functional alleles in conserved genes*, not diversifying
  hotspots: `wzi` (capsule, padj 1.0), fimbrial usher KPN_RS24485 (dN/dS 1.23), `dnaK` (0.93,
  purifying), `phoA` (1.13), `nfuA` (1.0).

**faeces/respiratory: no hotspot enrichment** (all hits 18% vs 14%, p = 0.21; invasion-direction 16%
vs 14%, p = 0.45). dN/dS for all hits is marginally elevated (1.75 vs 1.69, MWU p = 0.046) but the
invasion-direction subset is not (p = 0.22).

**Takeaway.** The blood-invasion hits carry *two* selective regimes — iron-uptake surface receptors
under positive/diversifying selection (mutational hotspots), versus capsule/adhesion alleles sitting
in conserved genes. The genome-wide "hit genes are hotspots" enrichment is real but is mostly a
gut-clade (faeces-direction) phenomenon, **not** a general "invasion genes are hotspots" result. To
test invasion-specificity directly — per-source hotspot rate vs the whole-population background — see
the planned per-isolation-source Chi-sq (uses our per-sample variant→loci collation).

## Files

| File | What |
|---|---|
| `blood_faeces_hits_with_hotspots.tsv` | 110 blood/faeces hits + `hot_*` columns, sorted by variance explained |
| `faeces_respiratory_hits_with_hotspots.tsv` | 88 faeces/resp hits + `hot_*` columns |
| `hotspot_enrichment_summary.json` / `.md` | Fisher + Mann–Whitney, all-genes and invasion-direction, per contrast |
