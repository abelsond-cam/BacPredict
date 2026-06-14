#!/bin/bash
# Experiment-4 unmasked-surprisal scan — STEP 2: sharded GPU pass over the manifest.
#
# One forward per protein (cheap unmasked surprisal) for every genome in this shard, writing
# a per-shard stats parquet, a per-shard autocorrelation NPZ, and a raw per-residue dump.
# 20 shards over ~1,000 genomes (~50 genomes / ~200k proteins each, ~2.5 h at ~22 prot/s);
# 12 h wall is deliberately generous. The module preamble (purge -> cuda/12.4 -> cudnn) is
# the lesson from the 6-second crash.
#
# Usage:  sbatch src/snp_embeddings/scripts/unmasked_surprisal_scan.sh   # after the manifest
#
#SBATCH --job-name=usurp_scan
#SBATCH --output=usurp_scan_%A_%a.out
#SBATCH --error=usurp_scan_%A_%a.err
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-19

cd /home/dca36/workspace/BacPredict

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4
export PYTHONUNBUFFERED=1

N_SHARDS=20
RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
PARQUET_DIR=$RDS/tb_protein_sequences
OUT_DIR=$RDS/snp_embeddings/unmasked_surprisal_scan
MANIFEST=$OUT_DIR/manifest.csv
OUT_PREFIX=$OUT_DIR/scan

echo "Shard ${SLURM_ARRAY_TASK_ID}/${N_SHARDS}  manifest=$MANIFEST  prefix=$OUT_PREFIX"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python src/snp_embeddings/unmasked_surprisal_scan.py \
    --mode scan \
    --manifest-csv "$MANIFEST" \
    --parquet-dir "$PARQUET_DIR" \
    --out-prefix "$OUT_PREFIX" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "$N_SHARDS" \
    --device cuda:0 \
    --max-lag 20 \
    --window 25 \
    --dump-raw

echo "Shard ${SLURM_ARRAY_TASK_ID} finished"
