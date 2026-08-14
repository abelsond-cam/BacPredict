---
name: gwas-calibration-reliability-protocol
description: How to judge a pyseer LMM GWAS trustworthy — locus-universe filter + af-stratified genomic-inflation λ + within-lineage permutation null (reuse for every GWAS in the programme)
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 43f9dbd0-7aa0-4579-ab80-2e7170ce8eb3
---

The confirmed protocol for deciding whether a pyseer LMM GWAS is trustworthy and **how high in allele
frequency** its calls can be trusted. Established on the Kp invasion variant + unitig axes; **apply to every
GWAS** (Kp, TB-AST, future).

1. **Locus universe — filter to ≥1% penetrance.** Call each sample vs one reference (NC_009648 for Kp) with
   snippy (acceptance `GT="1/1" && QUAL≥100 && DP≥3`); this gives **~2.04M** candidate SNP+indel loci, which
   we cut to those carried by **≥1% of isolates** (≥137 of 13,602) → **~372k** loci analysed (af 1–99%
   window). Ultra-rare per-sample noise is filtered, not modelled — so the analysis universe is
   common-to-intermediate variation, which is why af-stratified calibration is the right lens.

2. **Genomic-inflation λ, stratified by allele frequency** (`genomic_inflation_by_af.py --bins`). A single
   genome-wide λ hides *where* miscalibration sits. λ≈1 = calibrated; **λ<1 = conservative** (kinship
   over-corrects → clade-restricted hits are real, not artefacts); **λ>1 = inflated OR LD-redundant real
   signal** — in a clonal organism / on accessory features a high λ can be one biological event multiplied
   across thousands of co-inherited features, *not* p-value inflation. So **do NOT reject a correction method
   on raw λ alone** (the Kp unitig common-af λ=24 was **interpreted as** real LD-redundant signal — a HYPOTHESIS
   still under test, geNomad mapping + CG-level shuffle pending, see [[bac-pyseer-unitig-lambda-investigation]];
   the variant fixed-effects MDS was set aside at λ=4.34 — **its SL+CG within-lineage permutation is now running
   (agent B) to test whether that λ=4.34 is redundant signal, not confounding. If it ablates, the LMM
   overcorrected and "LMM is the method of record" is itself provisional**). Step 3 (permutation) distinguishes
   the two. Kp variant (LMM) axis: genome-wide **λ=0.562**, conservative everywhere (committed
   PROGRESS_VARIANTS.md). Kp unitig axis: 0.80 (rare) → 24 (common) — inflation localised to the common end.

3. **Within-lineage permutation null** (`permute_phenotype_within_lineage.py`) — the decisive
   structure-confounding test, because observed λ conflates real signal + structure (both inflate). Shuffle
   the phenotype **within each sublineage** (preserves between-lineage structure, destroys within-lineage
   signal), **re-fit the LMM FRESH** (`--save-lmm`; pyseer bakes h² into the cache, so `--load-lmm` would
   apply the *real* phenotype's h² — invalid), recompute af-stratified λ_perm. λ_perm≈1 ⇒ kinship genuinely
   controls structure (the real run's λ is trustworthy, not confounded); λ_perm≫1 in a bucket ⇒ that af's
   inflation is structure ⇒ unreliable. The **reliable af ceiling = highest bucket where λ_perm≈1**;
   common-af buckets double as a structure positive control. Kp variant axis: λ_perm≈1 at all af (0.86–1.07;
   h² 0.83→0.24 under the shuffle). A *plain* (unrestricted) shuffle only tests test-machinery calibration —
   it breaks the structure correlation, so it cannot diagnose confounding; the *within-lineage* shuffle can.

**Why:** structure correction (the kinship) is the make-or-break of a bacterial GWAS; these three steps
separate genuine conservatism from uncontrolled confounding and give a *non-arbitrary* af cutoff instead of
a hand-picked 0.05/0.10.

**How to apply:** run all three before trusting hits; gate reported hits to the calibrated af range; always
re-fit fresh (never `--load-lmm`) for a permuted phenotype. HPC gotcha: pyseer `--kmers` (unitigs)
accumulates memory across the scan and OOMs on the full set in one process (even at 200 GB) — **chunk it to
~100k unitigs/process** (prime the kinship cache once, scan each chunk as a separate `--load-lmm` process,
memory freed between, ~26 GB peak each), then concatenate. This is how the production unitig run is sharded
(64×~100k) and how the permutation null is run. See [[bac-pyseer-unitig-lambda-investigation]] (the
findings) and [[hpc-no-data-in-home-use-rds-scratch]].
