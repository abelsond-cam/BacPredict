---
name: kleb-ast-always-bf16-isosource-is-fp32
description: "Kp AST trainer used bf16 in EVERY git version (so the azithromycin Cambridge-vs-Isambard gap is NOT the fp32 bug); kleb_iso_source is the real fp32 exposure via dtype=\"auto\""
metadata: 
  node_type: memory
  type: project
  originSessionId: aac091b8-20e4-4661-ab5d-762fe4b1c697
  modified: 2026-08-11T16:10:32.246Z
---

Git-verified precision audit (2026-07-22), settling whether the Kp azithromycin gap
(Cambridge 0.822 vs Isambard 0.918) is the same fp32 bug that cost TB rifampin ~7pp.

**Kp AST trainer was bf16 in every version in history — the gap is NOT precision.**
- `dtype="auto"` (the fp32 path) NEVER appears in any Kp AST trainer path across all of
  git history. Only TB and `kleb_iso_source` ever used it.
- Every Kp AST trainer cast the model with `.to(torch.bfloat16)`: earliest
  `src/kleb_ast/train_amr.py` (commit 6a97c77) line 290; pre-consolidation
  `src/bacpredict/apps/kleb/train_amr.py` (@ b047ed8~1) line 260.
- The b047ed8 consolidation message states it plainly: "was fp32 'auto' for TB, bf16 for
  Kp … Kp unchanged". The azithromycin ladder CSV predates b047ed8 (Jul 11 ~12:46, before
  the 15:27 consolidation) → it ran on the bf16 `apps/kleb`/`kleb_ast` trainer.
- **Consequence:** Cambridge azithromycin already ran bf16, same as Isambard. Rerunning
  azithromycin `--precision fp32` on Isambard CANNOT reproduce the 0.822 and is the wrong
  test for this gap. David's original belief ("Kleb always used bf16 on HPC") was correct.

**Remaining live causes of the azithromycin gap** (the TB triad minus precision):
1. Base-model REVISION. Both clusters default to the same model *ID*
   (`macwiatrak/bacformer-large-masked-complete-genomes`, [[refreshed-bacformer-complete-genomes-model]]),
   but the HF revision is NOT pinned — an early-July Cambridge pull could be a pre-fix
   snapshot vs a later Isambard pull (CLAUDE.md §0.1 "earlier weights had defects"). This
   is the strongest remaining candidate. `results.json` `model.revision` (added 0167e1c)
   settles it but only exists for post-0167e1c runs.
2. Splits/seed and eval-set/metric definition.

**kleb_iso_source IS the real fp32 exposure. → FIXED 2026-08-11, see
[[invasion-model-was-fp32-not-bf16]].** The bf16 cast landed in `a817ac2` (2026-07-26);
`--precision` + `run_config.precision` in results.json landed 2026-08-11; bf16 re-runs of all
three cohorts are in flight. Every published iso-source number (0.786/0.762/0.827) remains an
fp32 result. The paragraph below is the original diagnosis, kept for the causal chain.

`src/kleb_iso_source/train_isolation_source.py`
line 307 uses `dtype="auto"` — changed FROM `.to(torch.bfloat16)` in commit 2d5866e
("use dtype='auto' so Stage A CPU smoke test works", [[bacformer-loading-idiom]]). On the
complete-genomes model, from_pretrained without a bf16 cast loads fp32 (empirically:
TB `--precision fp32` reproduced 0.90, [[bf16-beats-fp32-tb-ast]]). So iso_source has been
training fp32 — the exact condition that cost TB ~7pp. David expected iso_source was bf16;
it was, until 2d5866e flipped it. **How to apply:** to fix, mirror the AST trainer's
`--precision bf16` cast on GPU (keep a CPU/fp32 fallback only for Stage-A smokes) and
re-check iso_source's published numbers under bf16.
