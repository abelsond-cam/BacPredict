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
# Usage (on the HPC login node):  bash src/bacpredict/engine/scripts/smoke_concat_rpob_mean.sh
set -euo pipefail

# Data root + env — cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""   # force CPU — Stage-A must run with CUDA disabled

RDS=$D/processed/train_tb_ast
OUT_DIR=$RDS/pangena_predict/concat_rpob_mean_smoke
mkdir -p "$OUT_DIR"

"$PY" -m bacpredict.engine.concat.concatenate_bacformer_genome_esm_protein_emb \
    --ast-sheet-path "$RDS/binary_ast_with_split.csv" \
    --parquet-dir "$RDS/protein_sequences" \
    --esm-store-dir "$RDS/esm" \
    --output-json "$OUT_DIR/concat_rpob_mean_smoke.json" \
    --qc-log "$OUT_DIR/rpob_copy_qc_smoke.log" \
    --drug rifampin \
    --device cpu \
    --max-samples 60 \
    --kfold 2 --seeds 1 2

echo "Smoke finished — JSON at $OUT_DIR/concat_rpob_mean_smoke.json"
