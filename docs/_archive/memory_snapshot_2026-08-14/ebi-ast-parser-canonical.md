---
name: ebi-ast-parser-canonical
description: Canonical EBI AST→binary parser location + the recipe to regenerate TB labels on Isambard
metadata: 
  node_type: memory
  type: reference
  originSessionId: f3c1d41f-8ba4-45e4-98f4-8a0db0e1b64a
---

The canonical, **organism-agnostic** EBI AST → binary-label parser is
`src/pangena_predict/parse_ebi_ast_to_binary.py` (function `parse_ebi_ast_to_binary`; old name
`process_klebsiella_ast_data` kept as a back-compat alias). It was the misleadingly-named
`kleb_ast/convert_ast_data.py` — moved this session. Do NOT reconstruct the parsing; reuse this.

**Encoded nuance:** resistant→1, susceptible→0, intermediate→NaN; MIC→log with `<`/`>`/`>,<` censoring;
**repeat tests per sample×antibiotic are averaged** (`pivot_table(aggfunc="mean")`) → conflicting DSTs
become fractional labels, dropped downstream as ambiguous by `resolve_clean_splits`. TB has ~8 tests/
sample but almost never conflicts (rifampin: 16 fractional of 38,758).

**Regenerate TB labels on Isambard (two canonical steps):**
```
SC=$SCRATCHDIR; export PYTHONPATH=$HOME/BacPredict/src SCRATCHDIR=$SC MPLBACKEND=Agg
# 1) raw EBI long-format → binary_ast.csv (+ metadata, regression_log_mic, stats)
$SC/envs/bacpredict-gpu-venv/bin/python src/pangena_predict/parse_ebi_ast_to_binary.py \
  --input $SC/raw/tb/ebi_tb_amr_records.csv --output-dir $SC/processed/train_tb_ast \
  --viz-dir $SC/results_visualisations/tb
# 2) add_splits(seed=1) + prune to embedded samples → binary_ast_with_split.csv (~17s inline; NOT the
#    "10-min crawl" the old note feared). MUST pass --output-base to a live path (default is dead $RDS).
$SC/envs/.../python src/tb_ast/prepare_esmc_embeddings_and_labels_to_finetune_amr.py \
  --ast-csv $SC/processed/train_tb_ast/binary_ast.csv --embeddings-dir $SC/processed/train_tb_ast/esm \
  --output-base $SC/processed/train_tb_ast/ast_training --seed 1
```
Kp equivalent: `--input $SC/raw/kleb_ast/…` → `train_kleb_ast`. Result on Isambard:
`binary_ast_with_split.csv` = 36,692 TB samples, drug col `rifampin` (US spelling). See
[[pangena-predict-stage2-state]], [[tb_drug_us_spellings]].
