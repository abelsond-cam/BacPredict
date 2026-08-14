---
name: bac-pyseer-unitig-lambda-investigation
description: Kp invasion-GWAS calibration — within-lineage permutation nulls DONE at BOTH SL and CG (2026-07-29, agent B, single seed, David: conclusive). Unitig common-af inflation is NOT structure at SL or CG (real accessory signal); variant MDS λ=4.34 does NOT ablate (genuine under-correction) → LMM confirmed method of record. Accessory-genome/within-CG-HGT mechanism is David's HYPOTHESIS, still open; geNomad (agent C) is only ONE test of it (virus/prophage — NOT plasmids/integrons/IS/other HGT).
metadata: 
  node_type: memory
  type: project
  originSessionId: 43f9dbd0-7aa0-4579-ab80-2e7170ce8eb3
  modified: 2026-08-12T01:41:42.386Z
---

⚠️ **2026-07-29: the CG-level shuffle (agent B) is DONE — measurements below now cover SL AND CG. One test still
open: geNomad virus/prophage mapping** (agent C — ONE test only; geNomad does NOT detect plasmids/integrons/IS/other
HGT, so it cannot alone resolve the accessory-genome/HGT reading) ([[real-numbers-causal-lr-plan]] plan file
`our-next-task-is-cuddly-cosmos.md`). Keep separating the measurements (settled) from the mechanism
interpretation (David's HYPOTHESIS, still pending). Precedent for this discipline: [[dont-conflate-penetrance-with-lineage]].

**CG-LEVEL PERMUTATION RESULTS (agent B, 2026-07-29, single seed — David: conclusive, don't chase extra seeds).**
Ran the full matrix `MODEL{lmm,mds} × LEVEL{sl,cg}` via the env-toggled wrappers (commit `22cb7ce` on branch
`refactor/consolidate-engine`; results on CSD3 project_k `…/pyseer_iso_source/blood_faeces/sampled_country_2_1_all/gwas_lmm_permnull/`,
stems `_${MODEL}_${LEVEL}_seed1`). CG grouping = 2,270 "Clonal group" clusters (FINER than the 1,345 SLs → stricter null).
- **Unitig LMM · CG: λ_perm overall 0.85, flat by af** (common bins 0.50–0.70 = 1.0, 0.70–1.0 = 0.56 — the last bin is
  sparse; read as flat throughout). The real-run common-af λ≈24 does NOT reappear → **the common-af inflation is real
  accessory signal, not structure, at CG as well as SL** (SL null was 0.72 overall / 0.43 at 0.70–1.0). Mild mid-af
  bump (λ_perm ~1.1–1.3 at af 0.10–0.25) recurs at both levels — David: not a concern.
- **Variant MDS · SL = 3.49, MDS · CG = 4.12** (real MDS λ=4.34): the null does NOT ablate — the CG null reproduces
  ~95% of the real inflation across all af. So **MDS (K=10) genuinely UNDER-corrects structure; λ=4.34 is NOT
  interpretable within-lineage signal.** The "revisit MDS abandonment" thread is closed.
- **Variant LMM · CG: λ_perm 0.94, flat** (matches SL) → the core-SNP kinship absorbs structure at CG too. **LMM
  CONFIRMED as method of record** for the variant axis (its real λ=0.562<1 is legitimate conservatism).
- **Cross-pattern co-occurrence (descriptive, not a gate):** the top-4 common-af unitig pattern groups are each carried
  by ~11.4–11.8k of 13,602 genomes (af ~0.85) with pairwise carrier Jaccard **0.83–0.98** → they co-occur in one
  ~11.5k-genome background (LD-redundancy spans "independent" patterns). Gene labels blank for these top patterns.
  Output: `…/gwas_unitig_lmm/cross_pattern/`.

**DAVID'S MECHANISM TAKE (2026-07-29 — HYPOTHESIS, PENDING, do NOT assert):** the *contrast* — variant LMM
over-corrects (conservative λ=0.56) while the unitig axis shows huge apparent under-correction (λ≈24 that the null
proves is real) — hints that the invasion changes live in the **accessory genome**. "Some form of population change
**within a CG level** accounts for such huge population bias within CGs" — still **likely HGT-related, but untested**.
**geNomad (agent C) is only ONE test of this, NOT a resolver:** it detects **virus/prophage** and does NOT cover
plasmids, integrons, IS elements, or other HGT — those routes need their own tools/tests, so geNomad cannot alone
confirm/refute the accessory-genome/HGT reading. Do not conflate this with lineage/state as fact
([[dont-conflate-penetrance-with-lineage]], [[frame-gwas-results-as-invasion]]).

**MEASUREMENTS (trustworthy).** bac_pyseer unitig LMM (Kp blood-vs-faeces invasion GWAS) has genome-wide
genomic-inflation **λ=3.42**, rising with af on the **observed** data (fine bins: 0.01–0.05=0.80, 0.05–0.10=1.09,
0.10–0.15=1.91, … 0.50–0.70=6.81, 0.70–1.0=**23.8**). The **within-sublineage (SL)** permutation null (shuffle
phenotype within each of 1,345 sublineages → between-lineage structure preserved, within-lineage signal destroyed;
fresh re-fit, h² 0.83→0.24) gives **λ_perm ≈ 1 at EVERY af** (0.91, 0.80, 1.05, 1.29, 1.1–1.3 across 0.20–0.50,
1.06, 0.43 at 0.70–1.0). So the null never runs away **at the SL level** → the core-SNP kinship controls
**SL-level** structure at all af. The **variant** within-lineage permutation likewise gives λ_perm≈1 at all af.

**INTERPRETATIONS (HYPOTHESES — under test, do NOT assert as fact):**
- *Hypothesis:* the observed common-af inflation is **real within-lineage invasion signal, not confounding.* What
  is actually shown is only that it is **not SL-level** structure — **CG (clonal-group) level is not yet tested**
  (agent B). CG may carry population bias that SL does not; if λ_perm(CG)≫1 while λ_perm(SL)≈1, that is itself a
  **major finding**, not a refutation. Do not claim "no af ceiling" as settled — it is provisional pending the CG
  shuffle.
- *Hypothesis:* the magnitude is **LD-redundant, not independent** — Kp is clonal, accessory DNA travels in big
  co-inherited blocks, so one niche-associated **megaplasmid or prophage** could yield a unitig (often several)
  per locus, making λ=24 one/a few biological events × thousands of correlated unitigs. **geNomad (agent C) tests only
  the virus/prophage part** of this — it does NOT detect plasmids, integrons, IS elements or other HGT, so a
  prophage-negative result does not rule out a plasmid/other-MGE driver; those need separate tools. Not yet demonstrated.
- Reporting convention that DOES follow safely: read hits at the **independent-pattern / locus** level (pyseer
  pattern-count Bonferroni), never per-unitig — LD-redundancy inflates raw counts regardless of which hypothesis
  wins.

**Variant-axis thread — now RESOLVED (see CG-LEVEL RESULTS above):** the MDS λ=4.34 did NOT ablate under the
SL/CG permutation (λ_perm 3.49/4.12) → genuine under-correction, LMM confirmed as method of record. See
[[gwas-calibration-reliability-protocol]].

**⚠ "λ=24" is the COMMON-AF stratum, never the headline — do not compress it to "the unitig λ".** The
blood/faeces unitig LMM's **overall** genomic-inflation λ is **3.42** (`blood_vs_faeces_unitig_gwas_summary.json`,
full cohort n=13,602, 33,039 hits over 6.07M unique patterns, Bonferroni 8.23e-09). Verified 2026-08-12 by the
holdout-free re-run (train+validate, `sampled_country_2_1_all_trainval`, n=10,887): **λ=3.18**, 19,622 hits over
5.64M patterns, threshold 8.86e-09. Dropping 20% of genomes barely moves λ, so the leaky and honest hit sets are
comparably calibrated and the hit-count fall (33,039 → 19,622, −41%) is ordinary power loss, not a regime change.

**How to apply:** cite the measurements; present every interpretation as a hypothesis with its pending test named;
report at the pattern/locus level. Method: [[gwas-calibration-reliability-protocol]]; framing:
[[frame-gwas-results-as-invasion]].
