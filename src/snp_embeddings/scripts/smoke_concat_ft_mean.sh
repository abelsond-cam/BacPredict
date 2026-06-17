#!/bin/bash
# Stage-A smoke for A.1.i (FT-mean concat) — n=10, CPU, login node (<15 min, CUDA disabled).
#
# Proves the fine-tuned path runs end-to-end: load the *fine-tuned* Bacformer backbone from the
# deployed 0.905 AMR checkpoint (29776879), run it forward on CPU (dtype="auto" → .float()), pull the
# FT genome-mean, concat with the ESM-C rpoB vector, fit + score the three LR steps. AUROC on n≈2
# evaluate is meaningless (ablation sanity auto-skipped on a smoke) — we only check it runs and writes
# the JSON with mean_variant="finetuned". The contrast vs the frozen smoke is the FT-backbone load.
#
# Usage (on the HPC login node):  bash src/snp_embeddings/scripts/smoke_concat_ft_mean.sh
set -euo pipefail

cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""   # force CPU — Stage-A must run with CUDA disabled

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
OUT_DIR=$RDS/snp_embeddings/concat_ft_mean_smoke
mkdir -p "$OUT_DIR"

# The deployed RIF mean-pool checkpoint (job 29776879, ~0.905). resolve_checkpoint_dir picks the
# best checkpoint-*/ subdir inside it; glob the run dir so we need not hardcode the species prefix.
CKPT=$(ls -d "$RDS"/checkpoints/*rifampin_stage_c_29776879* 2>/dev/null | head -1)
if [[ -z "$CKPT" ]]; then
    echo "ERROR: could not find the rifampin_stage_c_29776879 checkpoint under $RDS/checkpoints/" >&2
    exit 1
fi
echo "Fine-tuned checkpoint: $CKPT"

uv run python src/snp_embeddings/concatenate_bacformer_genome_esm_protein_emb.py \
    --ast-sheet-path "$RDS/binary_ast_with_split.csv" \
    --parquet-dir "$RDS/tb_protein_sequences" \
    --esm-store-dir "$RDS/tb_esm_embeddings" \
    --output-json "$OUT_DIR/concat_ft_mean_smoke.json" \
    --qc-log "$OUT_DIR/rpob_copy_qc_smoke.log" \
    --drug rifampin \
    --device cpu \
    --bacformer-checkpoint "$CKPT" \
    --max-samples 10

echo "Smoke finished — JSON at $OUT_DIR/concat_ft_mean_smoke.json"
