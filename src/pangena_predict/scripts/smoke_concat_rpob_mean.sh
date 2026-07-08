#!/bin/bash
# Stage-A smoke for the concat probe — CPU, login node (<15 min, CUDA disabled).
#
# Proves the pipeline runs end-to-end: genotype rpoB sequences, pull the ESM-C rpoB vector (mmap),
# compute the frozen Bacformer mean on CPU (dtype="auto" — no manual bf16 cast), assemble the 1,920-d
# concat, fit + score the three LR steps, AND route the same three frames through the k-fold × m-seed
# harness. AUROC on a smoke is meaningless, so the ablation sanity check is auto-skipped; we only check
# it runs end-to-end (incl. the harness) and writes the JSON with a `kfold` block.
#
# n bumped to 60 so the 2-fold × 2-seed harness has a usable universe (it is skipped if too small).
# Usage (on the HPC login node):  bash src/pangena_predict/scripts/smoke_concat_rpob_mean.sh
set -euo pipefail

cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""   # force CPU — Stage-A must run with CUDA disabled

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
OUT_DIR=$RDS/pangena_predict/concat_rpob_mean_smoke
mkdir -p "$OUT_DIR"

uv run python src/pangena_predict/concatenate_bacformer_genome_esm_protein_emb.py \
    --ast-sheet-path "$RDS/binary_ast_with_split.csv" \
    --parquet-dir "$RDS/tb_protein_sequences" \
    --esm-store-dir "$RDS/tb_esm_embeddings" \
    --output-json "$OUT_DIR/concat_rpob_mean_smoke.json" \
    --qc-log "$OUT_DIR/rpob_copy_qc_smoke.log" \
    --drug rifampin \
    --device cpu \
    --max-samples 60 \
    --kfold 2 --seeds 1 2

echo "Smoke finished — JSON at $OUT_DIR/concat_rpob_mean_smoke.json"
