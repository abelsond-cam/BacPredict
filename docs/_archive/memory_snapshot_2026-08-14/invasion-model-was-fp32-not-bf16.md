---
name: invasion-model-was-fp32-not-bf16
description: "Every published kleb_iso_source (blood-vs-faeces) result was trained in fp32, not bf16 — the bf16 cast landed after those runs; plus the CSD3 flash-attn blocker for baclm"
metadata: 
  node_type: memory
  type: project
  originSessionId: 65fba51f-bd25-41d2-813a-24d9abb4255d
  modified: 2026-08-11T16:10:18.192Z
---

**Every published blood-vs-faeces number is an fp32 result.** `train_isolation_source.py`
loaded the model with `dtype="auto"`, which resolves to **fp32** master weights for
Bacformer-large. The unconditional bf16 cast only landed in **`a817ac2` (2026-07-26)** — after
every iso-source result, all dated 2026-05. So 0.786 (pooled) / 0.762 (stratified) / 0.827
(all_samples) are fp32.

This is the *underperforming* setting elsewhere: bf16 beat fp32 by ~7 pp AUROC on TB rifampin
in a controlled A/B ([[bf16-beats-fp32-tb-ast]]). Direction for this phenotype is **unproven** —
report the delta, don't assume a win.

Fixed 2026-08-11: `--precision {bf16,fp32}` added, and `results.json` now records
`run_config.precision` + `versions`, so this can never be ambiguous again. Older `results.json`
files are schema 1.1 and carry **no** precision field — absence means fp32 for iso-source runs
predating a817ac2. Docs that claimed the opposite (task CLAUDE.md, ToDo.md) are corrected.

New parameterised driver `scripts/train_isolation_source_cohort.sh` supersedes the three
`stage_c{,_pooled,_stratified}.sh` copies, which all **hardcoded `output_dir=<cohort>/models`**
— re-running one would have overwritten the deployed fp32 checkpoint in place. It writes to
`models_bf16/` and sends SLURM logs off the git tree.

**⚠ flash-attn CANNOT be installed on CSD3 — it is a glibc wall, not a torch-version one.**
The wheel installs fine and then fails at import: ``GLIBC_2.32 not found``. CSD3 runs **glibc
2.28** and every recent flash-attn release is built against 2.32, so **no prebuilt wheel of any
torch/CUDA tag will ever load there**. Don't retry wheels. Remaining options were a source build
(CSD3's newest nvcc module is 12.1 against a cu128 torch) or BacLM's own
`F.scaled_dot_product_attention` fallback (its modeling code branches `if fa_func is not None`).
**David's decision (2026-08-11): the fallback is too slow — do NOT use it.** Instead port
assemblies+GFFs to Isambard, embed there under the working GH200 env, and bring the store back;
explicitly a stop-gap, and **demoted to the last job**. Inputs are ready: 13,980/14,119 (99%) of
the pooled cohort have both `sr_assembly_file` and `sr_gff_file` (~43 GPU-h coding-only at the
measured ~11 s/genome; non-coding adds ~65% more sequences).

→ [[invasion-comparators-2026-08]] · [[bf16-beats-fp32-tb-ast]] · [[kleb-ast-always-bf16-isosource-is-fp32]]
