#!/bin/bash
#SBATCH --job-name=scan_surprisal_full_38k
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-255

# Genome-wide unmasked-surprisal pass over the FULL labelled TB split (~35.6k genomes), for the
# full-eval surprisal-panel experiment. One cheap forward per protein -> per-shard stats parquet
# (the per-protein PANEL columns) + ACF NPZ. NO raw per-residue dump (200 GB at this scale) — the
# panel store is built with build_surprisal_store.py --source parquet from the stats parquets.
#
# Cost: ~3 min/genome (4k proteins x 1 forward at ~22 prot/s) -> ~1,800 A100-hours total, spread
# over the array. 256 shards x ~140 genomes ~= 7 h/shard (12 h wall is generous).
#
# Build the manifest first (login node, <1 min) — all clean-label samples with dummy rpoB cols:
#   uv run python -c '...'  (see scripts comment / the plan) -> $OUT_DIR/full_scan_manifest.csv
# Then:
#   sbatch src/pangena_predict/scripts/scan_surprisal_full_38k.sh
# After all shards: build_surprisal_store.py --source parquet --scan-stats-glob "$OUT_DIR/scan_stats_shard*.parquet"

cd /home/dca36/workspace/BacPredict

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4
export PYTHONUNBUFFERED=1

N_SHARDS=256
RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
PARQUET_DIR=$RDS/tb_protein_sequences
OUT_DIR=$RDS/pangena_predict/unmasked_surprisal_scan_full
MANIFEST=$OUT_DIR/full_scan_manifest.csv
OUT_PREFIX=$OUT_DIR/scan

echo "Full scan shard ${SLURM_ARRAY_TASK_ID}/${N_SHARDS}  manifest=$MANIFEST  prefix=$OUT_PREFIX"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python src/pangena_predict/unmasked_surprisal_scan.py \
    --mode scan \
    --manifest-csv "$MANIFEST" \
    --parquet-dir "$PARQUET_DIR" \
    --out-prefix "$OUT_PREFIX" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "$N_SHARDS" \
    --device cuda:0 \
    --max-lag 20 \
    --window 25 \
    --no-dump-raw

echo "Full scan shard ${SLURM_ARRAY_TASK_ID} finished"
