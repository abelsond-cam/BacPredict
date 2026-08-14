---
name: amr-ladder-raw-recovery-framing
description: The AMR concat-ladder tests RAW AUROC recovery with simple measures — do NOT net out lineage/confounds
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 287d1555-8fdc-4241-9446-ecac54a396be
---

For the AMR concat-ladder deliverable ([[real-numbers-causal-lr-plan]]), the goal is to measure **how much
AUROC we can recover with simple measures** (FT genome-mean ⊕ best coding gene ⊕ best non-coding region, vs the
catalogue one-hot ceiling). It is NOT a causal/mechanism-purity argument.

**Why:** David (2026-07-19). When I recommended "net out lineage" — subtract the rif/cipro control baseline
because clonal population structure inflates core-region AUROC — he said this is NOT relevant to what we're
testing. We report the RAW recovered AUROC.

**How to apply:** Report raw ladder AUROCs. Keep rif/cipro in the panel as controls (they show the null/baseline
lift a non-coding rung gets from structure alone) but do NOT subtract, adjust, regress-out, or "net out" any
lineage/confound term from the reported numbers, and don't frame ladder results as lineage-corrected. Same spirit
applies to how the non-coding rung is selected: pick the best-recovering region, don't try to purify it of
lineage.
