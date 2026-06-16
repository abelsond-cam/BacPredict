#!/bin/bash
# Stage-A smoke for E1 (concat probe) — n=10, CPU, login node (<15 min, CUDA disabled).
#
# Proves the pipeline runs end-to-end: genotype 10 rpoB sequences, pull the ESM-C rpoB vector
# (mmap), compute the frozen Bacformer mean on CPU (dtype="auto" — no manual bf16 cast), assemble
# the 1,920-d concat, fit + score the three LR steps. AUROC on n≈2 evaluate is meaningless, so the
# ablation sanity check is auto-skipped on a smoke; we only check it runs and writes the JSON.
#
# Usage (on the HPC login node):  bash src/snp_embeddings/scripts/smoke_concat_rpob_mean.sh
set -euo pipefail

cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""   # force CPU — Stage-A must run with CUDA disabled

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
OUT_DIR=$RDS/snp_embeddings/concat_rpob_mean_smoke
mkdir -p "$OUT_DIR"

uv run python src/snp_embeddings/eval_concat_rpob_mean.py \
    --ast-sheet-path "$RDS/binary_ast_with_split.csv" \
    --parquet-dir "$RDS/tb_protein_sequences" \
    --esm-store-dir "$RDS/tb_esm_embeddings" \
    --output-json "$OUT_DIR/concat_rpob_mean_smoke.json" \
    --qc-log "$OUT_DIR/rpob_copy_qc_smoke.log" \
    --drug rifampin \
    --device cpu \
    --max-samples 10

echo "Smoke finished — JSON at $OUT_DIR/concat_rpob_mean_smoke.json"
