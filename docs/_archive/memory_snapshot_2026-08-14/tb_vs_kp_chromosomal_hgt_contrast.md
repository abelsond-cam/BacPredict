---
name: tb-vs-kp-chromosomal-hgt-contrast
description: "AMR/AST ONLY — TB AST Stage C performs much worse than Kp, an early observation CONSISTENT WITH (not proof of) the chromosomal-vs-HGT resistance-mechanism hypothesis. Scoped strictly to AMR; must NOT be imported into the invasion (blood/faeces) GWAS, whose plasmid/prophage question is separate and still untested."
metadata: 
  node_type: memory
  type: project
  originSessionId: 73acce67-3052-4ebe-9c0f-d5a392125f96
---

**Scope — AMR/AST only.** This memory is about the **antibiotic-resistance (AST) prediction** task and nothing
else. It is **not** evidence about the *Klebsiella* **invasion (blood-vs-faeces) GWAS**, and the phrase "HGT-vs-
chromosomal" here refers to the AMR resistance-mechanism contrast — it must **not** be borrowed to interpret the
invasion unitig/plasmid work, which has its own separate, still-untested plasmid/prophage hypothesis
([[bac-pyseer-unitig-lambda-investigation]]). Keep the two domains disconnected.

As of 2026-05-29, user reports TB AST Stage C results are "terrible" compared to Kp's strong performance (most Kp fan-out drugs in the 0.92–0.99 AUROC range; TB markedly lower). This matches the central programme hypothesis: TB resistance is dominated by **chromosomal point mutations** (rpoB, katG, gyrA, etc. — discrete SNPs where there's little for a genome-context embedding model to add beyond the known markers), while Kp resistance is dominated by **HGT / gene acquisition** (ESBLs, carbapenemases, *mcr*, AMEs on plasmids and ICEs — exactly the kind of co-occurrence / linkage signal Bacformer captures).

**Why this matters.** This is an early AST-only observation **consistent with** (not proof of) the cross-organism resistance-mechanism contrast — a **hypothesis**, not a settled result; a fair comparison also needs the refreshed base model and matched cohorts before the gap is attributed to mechanism. It also predicts which Kp drugs will be hardest within the species (chromosomal-mechanism ones like colistin via *mgrB*/*pmrAB*; FQs partly via gyrA/parC) and why — see [[kp-cipro-beats-kleborate-marker-model]].

**How to apply.**
- When discussing inter-organism comparisons, frame the TB-Kp gap as a *feature of the resistance-mechanism landscape*, not a model failure. TB is expected to remain harder.
- Within Kp, expect colistin and (partly) FQs to be the softest performers in the panel — consistent with the chromosomal arm.
- For the deferred mechanism stratification milestone, the cross-organism contrast is the headline; within-Kp HGT-vs-chromosomal stratification (Kleborate-labelled — see [[kleborate-for-mechanism-stratification]]) is the granular evidence.
