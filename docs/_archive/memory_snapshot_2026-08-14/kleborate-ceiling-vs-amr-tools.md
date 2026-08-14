---
name: kleborate-ceiling-vs-amr-tools
description: "For the Kp AST determinant \"ceiling\", Kleborate v3 KpSC is CARD-derived (not AMRFinderPlus) and other AMR tools don't improve it"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 3240ca4c-5459-4b1e-97f4-9f2775c0f04c
---

For BacPredict Task-2 (`kleb_ast`) the determinant **"ceiling"** (the Kp analogue of TB's WHO
catalogue, against which Bacformer-concat is compared) is built from **Kleborate** columns already in
`metadata_v2`. Methodology question resolved 2026-06-19 (does Kleborate miss SNP/non-coding determinants
that CARD / AMRFinderPlus / ResFinder catch?):

- **Kleborate v3 (v3.2.4 = what produced metadata_v2) KpSC AMR is CARD-derived** — "AMR database updated
  based on CARD v3.2.9 (June 2024)" — with Kleborate's **own BioPython-alignment logic** for the curated
  chromosomal set (gyrA 83/87, parC 80/84, mgrB/pmrB truncation, OmpK35/36, SHV). It is **NOT** built on
  AMRFinderPlus for KpSC; AMRFinderPlus is the engine for Kleborate's *Escherichia* module, not Kp.
  (Correction to the prior assumption that "Kleborate is from AMRFinderPlus".) Source: Kleborate v3
  readthedocs.
- **Other tools do not materially improve the Kp determinant ceiling.** Comparative assessment of 8 tools
  (Kleborate, AMRFinderPlus, RGI/CARD, ResFinder/PointFinder, DeepARG, Abricate, StarAMR, SraX) over Kp
  with an ML-AUC read-out: Kleborate detects the most ARGs and gave the most consistent ML usefulness;
  **integrating all tools did NOT beat the best minimal model** ("fewer, well-curated features outperform
  quantity"). RGI/CARD is the most conservative (fewest markers); ResFinder misses cefazolin/levofloxacin/
  cefuroxime. Source: *Sci Rep* 2025, s41598-025-24333-9 (PMC12627748).
- **The weak β-lactam/tetracycline drugs (cefepime, cefoxitin, cefuroxime, pip-tazo, amp-sulbactam,
  tetracycline) are literature-wide catalogue knowledge gaps** — *no* tool predicts them well. So a low
  Kleborate ceiling there is a faithful representation of the determinant-catalogue ceiling, **not** a
  Kleborate artefact → if Bacformer-concat beats it, that is a genuine "embeddings capture what catalogues
  miss" result (the Kp analogue of TB pyrazinamide). The paper independently establishes these are the gaps.
- Residual deep-research item (off critical path): ResFinder/PointFinder vs Kleborate point-mutation
  coverage for **colistin / azithromycin** (neither was in the 2025 study; colistin = mgrB/pmrB so
  Kleborate is mechanistically strong; azithromycin has no clean determinant in any catalogue).

**Decision (REVISED 2026-07, engine consolidation): CARD is the DEFAULT Kp ceiling; Kleborate is a
retained comparator.** The earlier "Kleborate alone" decision was reversed. Reasons: (1) CARD (our own
minimap sidecars → `card_determinant_lr`) resolves to **specific mutations**, which Kleborate's
per-isolate determinant calls cannot; (2) CARD is *also* load-bearing as the acquired-gene **locator**
(`card_gene_locator` finds the flat protein index of blaKPC/armA/AAC(6′) that Bakta under-annotates —
you cannot embed a gene you cannot locate). Kleborate (`kleborate_determinant_lr`) is **kept as a
comparator** because many readers treat it as the gold standard — it lets us show our results vs
Kleborate. The factual points above still hold (Kleborate is CARD-derived for KpSC; integrating tools
doesn't help). Both ceiling modules now share `engine/catalogue/base.score_onehot_frame`. Recorded in
`src/bacpredict/apps/kleb/CLAUDE.md`. Links: [[kleborate-for-mechanism-stratification]],
[[tb-vs-kp-chromosomal-hgt-contrast]].
