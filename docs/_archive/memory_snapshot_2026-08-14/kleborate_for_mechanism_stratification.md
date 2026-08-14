---
name: kleborate-for-mechanism-stratification
description: "For the deferred HGT-vs-chromosomal mechanism stratification milestone, drive the per-isolate mechanism labels off Kleborate's gene calls."
metadata: 
  node_type: memory
  type: project
  originSessionId: 73acce67-3052-4ebe-9c0f-d5a392125f96
---

For the HGT-vs-chromosomal mechanism stratification (Task 2 milestone 6 in [src/kleb_ast/CLAUDE.md](../../../../../../developer/BacPredict/src/kleb_ast/CLAUDE.md); currently **deferred**), use **Kleborate**'s per-isolate gene-calling output as the primary source for labelling each isolate's resistance mechanism per drug — acquired gene (HGT) vs chromosomal point mutation vs mixed. Kleborate is preferred over AMRFinderPlus alone for this task because (a) it is Kp-specific and ships the curated Kleborate AMR reference set the rest of the BacHGT/BacPredict stack already vendors ([src/bac_kleborate/](../../../../../../developer/BacHGT/src/bac_kleborate/)), and (b) it also publishes its own marker-based AMR predictor that Bacformer is now beating on cipro (see [[kp-cipro-beats-kleborate-marker-model]]) — running Kleborate gives us both the mechanism labels *and* the comparator baseline in one pass.

**Why this matters.** The stratified delta (Bacformer's gain over baseline on HGT-resistant vs vertically-resistant isolates) is the headline figure for the paper's central hypothesis. The data source for the stratum labels must be robust and reproducible.

**How to apply.**
- When the deferred milestone is picked up, build the mechanism-stratification pipeline off Kleborate output, not AMRFinderPlus.
- Per-drug, partition evaluate-set isolates into {HGT-only, chromosomal-only, mixed} and re-report AUROC / sens / spec per stratum + the delta vs baseline.
- See [[tb-vs-kp-chromosomal-hgt-contrast]] for the inter-organism version of the same evidence.
