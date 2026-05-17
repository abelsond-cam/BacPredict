#!/bin/bash
#SBATCH --job-name=bacformer_array_tb
#SBATCH --output=bacformer_array_tb_%A_%a.out
#SBATCH --error=bacformer_array_tb_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --array=0-63

# TB Bacformer embedding generation (GPU array job).
#
# Mirrors run_bacformer_embeddings_array.sh but points at the TB processed
# dirs via the --input-dir/--esm-dir/--bacformer-dir CLI args (organism-
# agnostic; Klebsiella behaviour is unchanged because those args default to
# the klebsiella_* constants when omitted).
#
# Sizing vs the 36h GPU per-job wall-clock limit: ~38k genomes * 15s ~= 158
# GPU-h. 64 array tasks -> ~600 genomes/task -> ~2.5h compute; --time=04:00:00
# leaves large headroom and no single task can approach 36h. --skip-existing
# filters before index slicing so a maintenance kill/requeue resumes for free.
#
# Usage:
#   sbatch slurm_scripts/run_bacformer_embeddings_array_tb.sh
#   sbatch --array=0-127 --time=02:00:00 slurm_scripts/run_bacformer_embeddings_array_tb.sh

module purge
module load cuda/12.4 2>/dev/null || echo "CUDA module not found, using system CUDA"
module load cudnn/8.9_cuda-12.4 2>/dev/null || echo "cuDNN module not found, using system cuDNN"

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
export HF_HOME=/home/dca36/rds/hpc-work/.huggingface_cache
export TRANSFORMERS_CACHE=/home/dca36/rds/hpc-work/.huggingface_cache
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

cd /home/dca36/workspace/predict_kleb_by_bacformer

RDS_ROOT=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david
TB_INPUT_DIR="${RDS_ROOT}/processed/tb/tb_protein_sequences"
TB_ESM_DIR="${RDS_ROOT}/processed/tb/tb_esm_embeddings"
TB_BACFORMER_DIR="${RDS_ROOT}/processed/tb/tb_bacformer_embeddings"

echo "=========================================="
echo "TB Bacformer Embedding Generation (Array Job)"
echo "Job ID: $SLURM_JOB_ID  Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Count unprocessed protein parquets (both .pt outputs must exist to skip).
TOTAL_FILES=$(uv run python -c "
from pathlib import Path
inp = Path('${TB_INPUT_DIR}')
esm = Path('${TB_ESM_DIR}')
bac = Path('${TB_BACFORMER_DIR}')
files = sorted(inp.glob('*_protein_sequences.parquet'))
todo = [
    f for f in files
    if not ((esm / f'{f.stem.replace(\"_protein_sequences\", \"\")}_esm_embeddings.pt').exists() and
            (bac / f'{f.stem.replace(\"_protein_sequences\", \"\")}_bacformer_embeddings.pt').exists())
]
print(len(todo))
")
echo "Total unprocessed files: $TOTAL_FILES"

# Derive chunk size from the array task count (set by Slurm, overridable via
# sbatch --array=...); fall back to 64 if unset.
NTASKS=${SLURM_ARRAY_TASK_COUNT:-64}
CHUNK_SIZE=$((TOTAL_FILES / NTASKS + 1))
START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * CHUNK_SIZE))
if [ $END_IDX -gt $TOTAL_FILES ]; then END_IDX=$TOTAL_FILES; fi
echo "Tasks: $NTASKS  Chunk: $CHUNK_SIZE  Indices: $START_IDX..$END_IDX"
echo "=========================================="

uv run python src/predict_kleb_by_bacformer/pp/generate_bacformer_embeddings.py \
    --input-dir "$TB_INPUT_DIR" \
    --esm-dir "$TB_ESM_DIR" \
    --bacformer-dir "$TB_BACFORMER_DIR" \
    --skip-existing \
    --start-idx $START_IDX \
    --end-idx $END_IDX

echo "=========================================="
echo "End time: $(date)  Array Task $SLURM_ARRAY_TASK_ID completed"
echo "=========================================="
