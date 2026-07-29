#!/bin/bash
#SBATCH --job-name=embeddings_array
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%A_%a.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%A_%a.out
#SBATCH --time=0:45:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --array=0-29
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → embeddings_array_%A_%a.out/.err (repo-relative), and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.

# Array job: ESM-C embeddings for Klebsiella, split across parallel GPU tasks.
# Add --bacformer-embeddings (and --bacformer-dir …) to the invocation below
# to also produce Bacformer contextualised outputs.
#
# Usage:
#   sbatch src/bacpredict/engine/embedding/scripts/run_embeddings_array.sh

# CUDA comes from the Isambard Cray PE + the venv — no `module load` needed.
# HF_HOME/TORCH_HOME are set to the persistent PROJECTDIR cache via ~/.bashrc.

set -uo pipefail
# Data root + env — cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction"}"
D="$BACPREDICT_DATA_ROOT"
PY="$HOME/workspace/BacPredict/.venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

# Force Python unbuffered output for real-time logging
export PYTHONUNBUFFERED=1
# Set transformers verbosity for better logging
export TRANSFORMERS_VERBOSITY=info

echo "=========================================="
echo "Klebsiella Embedding Generation (Array Job; ESM-C default)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Chunk by the TOTAL parquet count (stable across tasks), not the unprocessed
# count. The unprocessed count is racy — late-starting tasks would see a
# shrunken list and slice into the wrong index space, leaving gaps. Each task
# now slices [start..end) of the full sorted parquet list; `--skip-existing`
# inside the Python script then filters within the slice.
read TOTAL_FILES UNPROCESSED <<< "$("$PY" -c "
from pathlib import Path

root = Path('$D')
input_dir = root / 'processed' / 'train_kleb_ast' / 'protein_sequences'
esm_dir = root / 'processed' / 'train_kleb_ast' / 'esm'

protein_files = sorted(input_dir.glob('*_protein_sequences.parquet'))
unprocessed = [
    f for f in protein_files
    if not (esm_dir / f'{f.stem.replace(\"_protein_sequences\", \"\")}_esm_embeddings.pt').exists()
]
print(len(protein_files), len(unprocessed))
")"

echo "Total parquet files: $TOTAL_FILES  (unprocessed: $UNPROCESSED)"

# Calculate chunk size from total parquet count (divide by array task count)
NTASKS=${SLURM_ARRAY_TASK_COUNT:-30}
CHUNK_SIZE=$((TOTAL_FILES / NTASKS + 1))
START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * CHUNK_SIZE))

# Make sure we don't go past the end
if [ $END_IDX -gt $TOTAL_FILES ]; then
    END_IDX=$TOTAL_FILES
fi

echo "Tasks: $NTASKS  Chunk: $CHUNK_SIZE  Indices: $START_IDX..$END_IDX"
echo "=========================================="

# Run the Python script with array job parameters
"$PY" -m bacpredict.engine.embedding.generate_embeddings \
    --skip-existing \
    --start-idx $START_IDX \
    --end-idx $END_IDX

echo "=========================================="
echo "End time: $(date)"
echo "Array Task $SLURM_ARRAY_TASK_ID completed"
echo "=========================================="
