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

# Intrinsic-attention diagnostic (Maciej's method): read the model's OWN self-attention
# (return_attn_weights=True), average over the 15 heads, and rank rpoB's received attention among
# the genome's proteins, resistant vs WT, per layer + mean-over-layers. Materialises [n_heads,N,N]
# per layer (bf16) — generous --mem/GPU for the ~4000-protein TB genomes.
#
#   D2: pass a fine-tuned checkpoint to probe its .bacformer backbone (does fine-tuning KEEP rpoB
#       attended internally while the mean pool obliterates it?).
#   D3: always names the top-K most-attended genes per genome (--protein-parquet-dir).
#
#   $1 = label (required)        names the output subdir, e.g. frozen / e2e_gated_mil_0868 / meanpool_0905
#   $2 = checkpoint dir          optional; "-" or "frozen" → frozen pretrained complete-genomes model
#   $3 = max genomes             optional smoke cap; omit for the full 1000 manifest
#
#   sbatch src/snp_embeddings/scripts/phase1_intrinsic_attention.sh frozen - 20                 # frozen smoke
#   sbatch src/snp_embeddings/scripts/phase1_intrinsic_attention.sh frozen                      # frozen full
#   sbatch src/snp_embeddings/scripts/phase1_intrinsic_attention.sh e2e_gated_mil_0868 /path/ckpt  # FT full (D2)

cd /home/dca36/workspace/BacPredict

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4
export PYTHONUNBUFFERED=1

LABEL=${1:?usage: $0 <label> [checkpoint_dir|-] [max_samples]}
CKPT=${2:-}
max_samples=${3:-}

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
SCAN_DIR=$RDS/snp_embeddings/unmasked_surprisal_scan
PARQUET_DIR=$RDS/tb_protein_sequences
OUT_DIR=$RDS/snp_embeddings/intrinsic_attention/$LABEL

ckpt_arg=""
if [ -n "$CKPT" ] && [ "$CKPT" != "-" ] && [ "$CKPT" != "frozen" ]; then
    ckpt_arg="--checkpoint-dir $CKPT"
fi
extra=""
if [ -n "$max_samples" ]; then
    extra="--max-samples $max_samples"
    OUT_DIR=${OUT_DIR}_smoke
fi

echo "Intrinsic-attention probe — label=$LABEL  ckpt=${CKPT:-frozen}  out=$OUT_DIR  $extra"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python src/snp_embeddings/intrinsic_attention_probe.py \
    --manifest-csv "$SCAN_DIR/manifest.csv" \
    --esm-store-dir "$RDS/tb_esm_embeddings" \
    --protein-parquet-dir "$PARQUET_DIR" \
    --output-dir "$OUT_DIR" \
    --device cuda:0 \
    --max-proteins 8000 \
    --top-k 20 \
    $ckpt_arg $extra

echo "Done — figure + JSON in $OUT_DIR"
