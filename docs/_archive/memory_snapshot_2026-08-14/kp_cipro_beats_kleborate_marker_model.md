---
name: kp-cipro-beats-kleborate-marker-model
description: Bacformer Kp ciprofloxacin (still mid-training) is already beating the Kleborate published marker-based AMR predictor — a notable comparator for the paper.
metadata: 
  node_type: memory
  type: project
  originSessionId: 73acce67-3052-4ebe-9c0f-d5a392125f96
---

At epoch 35 of Stage C training (job 29824391, fan-out submitted 2026-05-29), Kp **ciprofloxacin** in-training validation AUROC reached **0.979** — roughly **9 percentage points above** the Kleborate-published marker-based AMR predictor that ships with Kleborate. Cipro was still training (panel-eval job 29837108 will produce the final held-out evaluate-set number + a Youden operating point).

**Why this matters.** Cipro resistance in Kp is partly chromosomal (gyrA/parC QRDR) and partly HGT (qnr, aac(6')-Ib-cr). Bacformer leveraging the genomic context beyond known markers — and beating Kleborate's marker rules on the same task — is a strong positive comparator for the paper's central HGT-aware-prediction story. See [[tb-vs-kp-chromosomal-hgt-contrast]].

**How to apply.** When framing Kp AMR results, position Bacformer's cipro performance against the Kleborate marker-based model as the operative baseline (Kleborate is the established tool clinicians/researchers already use). Use the *final* (panel-eval) AUROC for any quoted comparison, not the in-training interim. For the mechanism stratification milestone, cipro is a good candidate for an HGT-vs-chromosomal stratified analysis since it has both modes — see [[kleborate-for-mechanism-stratification]].
