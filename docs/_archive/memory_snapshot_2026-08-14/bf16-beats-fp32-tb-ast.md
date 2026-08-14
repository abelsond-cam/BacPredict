---
name: bf16-beats-fp32-tb-ast
description: "Controlled A/B finding — bf16 fine-tuning beats fp32 by ~5pts AUROC on TB rifampin AST; higher precision is WORSE, so fp32 is not an upgrade"
metadata: 
  node_type: memory
  type: project
  originSessionId: 287d1555-8fdc-4241-9446-ecac54a396be
---

Controlled A/B on Isambard (2026-07-21): TB **rifampin** AST fine-tune, **bf16 beats fp32 by ~5 points
AUROC**. A genuine finding, not a measurement confound — verified against the split/model/LR provenance.

**The clean like-for-like** (fold-0 *validation*-peak, the same 5,702-sample val set — this is the
same-role comparison; do NOT compare fp32 val-peak against bf16's evaluate-holdout):
- **bf16** (job `5661316`): val-peak **0.9600** @ epoch 10.9 → evaluate-holdout (n=7,127) **0.9642**
- **fp32** (job `5734578`): val-peak **0.9109** @ epoch 13.7 → **cancelled ~16.75h in** (well-known
  bf16>fp32 phenomenon; not worth ~6 more GH200-h for an overfit-plateau eval-holdout). Finding rests on
  the val-peak comparison; fp32 eval-holdout never scored but ~0.91 by val≈eval.

**Why it's controlled, not a [[tb_mini_set_0977_confound_vs_bug]]-style artifact:** both runs read the
SAME split CSV (`processed/train_tb_ast/binary_ast_with_split.csv`, kfold fold0/seed1/evaluate_seed1,
~35.6k cohort), the SAME refreshed model (`macwiatrak/bacformer-large-masked-complete-genomes` snapshot
`ab3a91a2`), the SAME LR 0.00015; fp32 writes a separate `_fp32_` dir (bf16 checkpoint NOT overwritten),
only `Precision (master weights)` differs. bf16's evaluate (0.9642) ≈ its val-peak (0.960) confirms 0.96
is real, not a lucky holdout; fp32's 0.89–0.91 band never overlaps bf16's 0.95–0.96 band → convincing
even at n=1 seed (multi-seed only needed for external publication).

**Surprising direction — higher precision is WORSE.** Leading hypothesis (UNTESTED): bf16 mixed-precision
acts as a regularizer / fp32 converges to a sharper, worse-generalizing minimum — fp32 visibly overfits
harder (eval_loss climbs 0.355→0.61 past its peak, val AUROC drifts 0.911→~0.89). NOT a pure numerics story.

**Practical takeaway:** the repo's bf16 default (CLAUDE.md — "both organisms train in bf16") is the RIGHT
default, not merely the memory-cheap one; do NOT switch to fp32 expecting a precision upgrade — it costs
~5pts here. Only rifampin tested; likely holds for TB broadly but untested on other drugs / Kp. Related:
[[tb_vs_kp_chromosomal_hgt_contrast]], [[bacpredict-isambard-validation-program]].
