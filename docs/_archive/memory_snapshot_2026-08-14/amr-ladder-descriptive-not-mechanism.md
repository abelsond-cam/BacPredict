---
name: amr-ladder-descriptive-not-mechanism
description: "The AMR ladder deliverable is a DESCRIPTIVE gene/weight comparison (CARD vs LR vs ladder), NOT causal-vs-phylogeny interpretation"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: aac091b8-20e4-4661-ab5d-762fe4b1c697
---

For the Kp/TB AMR ladder work the deliverable David wants is a **detailed description of which
genes the model finds and at what weights**, comparing the **CARD determinant one-hot → the
per-gene LR ranking (baclm/ESM coding) → the ladder** (ft_mean → +coding → +non-coding → +both →
vs ceiling). Concretely: which genes appear in each, their weights/AUROCs, and how the LR ranking
lines up against the CARD determinant list.

**Why:** We are **NOT** interpreting causal-vs-phylogeny (lineage/clonal) to judge the model's
worth. David said this sharply and repeatedly. Do **not** editorialise "this gene is causal / that
one is a lineage confound," do not offer mechanism hypotheses as the point of the panel, and do not
frame results as "the model works because X mechanism."

**How to apply:** Present the ladder + rankings as concrete tables — genes, weights (lr_auroc /
eval_auroc / prevalence / CARD coefficient), and the CARD↔LR cross-reference (does the LR surface
the CARD gene, at what weight). Report raw recovered AUROC ([[amr-ladder-raw-recovery-framing]]).
Never call a gene a "lineage correlate" ([[dont-conflate-penetrance-with-lineage]]). Save the
biology interpretation for David to drive.
