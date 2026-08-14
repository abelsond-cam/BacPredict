---
name: pyseer-invasion-structure-tests-program
description: LIVE program (2026-07-28) — parallel tests of whether the Kp invasion-GWAS common-af inflation is real signal vs structure, and plasmid/prophage vs chromosomal. Agent division + result landing slots (write these memories only when results return, as measurement + hypothesis).
metadata: 
  node_type: memory
  type: project
  originSessionId: 43f9dbd0-7aa0-4579-ab80-2e7170ce8eb3
---

Live multi-agent program launched 2026-07-28 to **test** (not assume) the interpretations that had been written
into memory as fact. Plan: `~/.claude/plans/our-next-task-is-cuddly-cosmos.md`. Branch `dev`; files kept disjoint
per agent; commits on explicit user go. Deliverable 1 (within-pattern carrier Jaccard) was **dropped** as
tautological (pyseer collapses `pattern_group` on rounded `(af, β, p)` — a proxy for identical carrier sets).

**Agent division:**
- **Agent 0 (memories/supervision)** — this session. Cleaned the pyseer memories so no pending test is pre-empted
  by a conclusion-as-fact (see [[bac-pyseer-unitig-lambda-investigation]], [[gwas-calibration-reliability-protocol]],
  [[frame-gwas-results-as-invasion]], and the AMR-scoping fix on [[tb-vs-kp-chromosomal-hgt-contrast]]). Owns all
  future `docs/PROGRESS_*.md` + result-memory writes.
- **Agent B — permutation-null program** (matrix + lineage columns, no minimap2): variant **MDS (λ=4.34)** at
  **SL and CG** (the decisive test — never permutation-tested; if it ablates, λ=4.34 was interpretable signal and
  the LMM λ=0.562 overcorrected), variant **LMM at CG**, and **unitig LMM at CG**. Optional descriptive
  cross-pattern carrier co-occurrence. Not a gate.
- **Agent C — geNomad virus/prophage mapping** (minimap2 into per-sample geNomad extracts + assembly): do the
  invasion-hit unitigs of a pattern land on the **same prophage** (and, where geNomad flags them, plasmid) across
  carriers, or scatter / chromosomal? Independent of B. Includes scale-up timing for the full thousands×thousands job.
  **Scope caveat (David 2026-07-29): geNomad = virus/prophage; it does NOT cover plasmids/integrons/IS/other HGT — so
  this is ONE test of the accessory-genome/HGT hypothesis, not a resolver (see slot 4).**

**RESULT LANDING SLOTS — write/update these ONLY when results return, each as measurement + hypothesis, never a
bare conclusion:**
1. ✅ **LANDED 2026-07-29 (agent B, [[bac-pyseer-unitig-lambda-investigation]]).** Variant MDS did NOT ablate
   (λ_perm SL 3.49 / CG 4.12 vs real 4.34) → genuine under-correction, **LMM confirmed as method of record** (not
   flipped). Single seed; David: conclusive.
2. ✅ **LANDED 2026-07-29 (agent B).** Unitig LMM · CG λ_perm 0.85, flat by af (common bins ~1.0 / 0.56-sparse) →
   the common-af inflation is **real accessory signal at CG too**, not CG-level bias. Not ≫1.
3. ✅ **LANDED 2026-07-29 (agent B).** Cross-pattern: top-4 common-af patterns co-occur in ~11.5k/13,602 genomes,
   pairwise carrier Jaccard 0.83–0.98 (LD-redundancy spans "independent" patterns). Descriptive.
   → New pending thread from David: the accessory-genome / within-CG-HGT mechanism reading (HYPOTHESIS). geNomad is
   only ONE test of it, NOT a resolver — see slot 4.
4. **geNomad virus/prophage mapping** (agent C, STILL OPEN) → new memory + `PROGRESS_UNITIGS.md`: partition of invasion
   signal at the independent-pattern / var_explained level (never raw unitig count). **Scope caveat (David 2026-07-29):
   geNomad detects VIRUS/PROPHAGE only — it does NOT cover plasmids, integrons, IS elements or other HGT, so it is
   ONE test, not the full MGE/HGT partition.** A prophage-negative / chromosomal-or-scattered result is reportable but
   does NOT refute the accessory-genome/HGT reading (plasmid/integron/IS drivers need their own tools).
