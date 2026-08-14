---
name: frame-gwas-results-as-invasion
description: "Present pyseer/GWAS hits oriented to the invasion (phenotype) allele, not the reference-allele β direction"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 43f9dbd0-7aa0-4579-ab80-2e7170ce8eb3
---

Frame every pyseer/GWAS result as **invasion**, independent of the β sign. pyseer's `β`/`af` are relative to the *reference* allele (an arbitrary choice), so β>0 only means "the non-reference variant points toward invasion." We care about *all* invasion-priming variation — and the **common** allele *may* be the evolution-selected invasion-adapted one (an interpretive prior, **not established** — orienting to it is a reporting choice, not a claim that common = causal). So re-orient every hit to its invasion allele and report **`invasive_af`** (= `af` if β>0 else `1−af`), **`abs_beta`** (|β|), and **`invasion_allele`** (ALT/REF); lead with **`var_explained ≈ af(1−af)·β²`**, which is invariant to the reference choice (`af(1−af)=invasive_af(1−invasive_af)`, `β²=|β|²`) and so is the right direction-free footing that gates the consequence and hotspot sub-analyses.

**Why:** a β-sign GWAS discards the population-wide common-allele invasion signal as "faeces hits." Reframing surfaced that ~79% of blood's invasion variance — including *all* 40 missense + 2 LoF coding hits — sits in the **common** allele (median invasive_af 0.98), the opposite of the "no coding invasion signal" a rare-variant-only read gave.

**How to apply:** `src/bac_pyseer/kleb_iso_source/pyseer_postprocess.py` now emits `invasive_af`/`abs_beta`/`invasion_allele`; collapse β>0/β<0 tables into one invasion table banded by `invasive_af` (<0.5 rare vs ≥0.5 common invasion allele). The same orientation applies to the unitig, Panaroo-GPA, and TB-AST GWAS. Worked example + the source-of-truth narrative: `src/bac_pyseer/docs/PROGRESS.md` (§2.1 + §4). Related: the variant axis was validated by 83/84 independent blood patterns replicating in *direction* in respiratory (binomial p≈4e-24) — directional concordance, not per-variant genome-wide significance, is the strong replication test.
