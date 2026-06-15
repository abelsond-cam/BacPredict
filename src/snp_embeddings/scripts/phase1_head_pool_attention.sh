#!/bin/bash
#SBATCH --job-name=head_pool_attn
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

# D1 — head-pool diagnostic: does the *predictive head's* learned pool attend to rpoB?
# Loads a TRAINED BacformerAttnPoolForGenomeClassification checkpoint, runs it over the 1000
# manifest genomes, reads the pool's per-protein weight (model.last_attention_weights), and
# ranks rpoB's weight among the genome's proteins, resistant vs WT. A plain mean pool would put
# rpoB at percentile 0.5 (uniform) — separation from 0.5, especially R>WT, is the signature of a
# head that actually routes to rpoB. Much lighter than the intrinsic probe (no [H,N,N] matrices).
#
#   $1 = checkpoint dir (required)   e.g. .../<run>/checkpoint-<step>
#   $2 = label (required)            e.g. e2e_gated_mil_0868   (names the output subdir)
#   $3 = max genomes (optional)      smoke; omit for the full 1000 manifest
#
#   sbatch src/snp_embeddings/scripts/phase1_head_pool_attention.sh /path/ckpt e2e_gated_mil_0868 20   # smoke
#   sbatch src/snp_embeddings/scripts/phase1_head_pool_attention.sh /path/ckpt e2e_gated_mil_0868       # full

cd /home/dca36/workspace/BacPredict

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4
export PYTHONUNBUFFERED=1

CKPT=${1:?usage: $0 <checkpoint_dir> <label> [max_samples]}
LABEL=${2:?usage: $0 <checkpoint_dir> <label> [max_samples]}
max_samples=${3:-}

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
SCAN_DIR=$RDS/snp_embeddings/unmasked_surprisal_scan
OUT_DIR=$RDS/snp_embeddings/head_pool_attention/$LABEL

extra=""
if [ -n "$max_samples" ]; then
    extra="--max-samples $max_samples"
    OUT_DIR=${OUT_DIR}_smoke
fi

echo "Head-pool probe — manifest=$SCAN_DIR/manifest.csv  ckpt=$CKPT  label=$LABEL  out=$OUT_DIR  $extra"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python src/snp_embeddings/head_pool_attention_probe.py \
    --manifest-csv "$SCAN_DIR/manifest.csv" \
    --esm-store-dir "$RDS/tb_esm_embeddings" \
    --checkpoint-dir "$CKPT" \
    --label "$LABEL" \
    --output-dir "$OUT_DIR" \
    --device cuda:0 \
    --max-proteins 8000 \
    $extra

echo "Done — figure + JSON in $OUT_DIR"
