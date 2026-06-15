#!/bin/bash
#SBATCH --job-name=intrinsic_attn
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00

# Part 1 — intrinsic-attention diagnostic (Maciej's method): does the FROZEN pretrained
# Bacformer already attend to rpoB? Reads the model's own self-attention
# (return_attn_weights=True), averages over the 15 heads, and ranks rpoB's received
# attention among the genome's proteins, resistant vs WT, per layer + mean-over-layers.
# Materialises [n_heads, N, N] per layer (bf16) — generous --mem/GPU for the ~4000-protein
# TB genomes.
#
# 1st positional arg = max genomes (smoke). Omit for the full 1000 manifest.
#   sbatch src/snp_embeddings/scripts/phase1_intrinsic_attention.sh 20    # smoke
#   sbatch src/snp_embeddings/scripts/phase1_intrinsic_attention.sh       # full 1000

cd /home/dca36/workspace/BacPredict

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4
export PYTHONUNBUFFERED=1

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
SCAN_DIR=$RDS/snp_embeddings/unmasked_surprisal_scan
OUT_DIR=$RDS/snp_embeddings/intrinsic_attention

max_samples=${1:-}
extra=""
if [ -n "$max_samples" ]; then
    extra="--max-samples $max_samples"
    OUT_DIR=${OUT_DIR}_smoke
fi

echo "Intrinsic-attention probe — manifest=$SCAN_DIR/manifest.csv  out=$OUT_DIR  $extra"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python src/snp_embeddings/intrinsic_attention_probe.py \
    --manifest-csv "$SCAN_DIR/manifest.csv" \
    --esm-store-dir "$RDS/tb_esm_embeddings" \
    --output-dir "$OUT_DIR" \
    --device cuda:0 \
    --max-proteins 8000 \
    $extra

echo "Done — figure + JSON in $OUT_DIR"
