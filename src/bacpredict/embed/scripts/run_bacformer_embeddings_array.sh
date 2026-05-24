#!/bin/bash
#SBATCH --job-name=bacformer_array
#SBATCH --output=bacformer_array_%A_%a.out
#SBATCH --error=bacformer_array_%A_%a.err
#SBATCH --time=0:45:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --array=0-29

# Array job script to run Bacformer embedding generation on HPC with GPU
# This splits the workload across multiple parallel jobs
#
# Usage:
#   sbatch src/bacpredict/embed/scripts/run_bacformer_embeddings_array.sh
#
# The array range (--array=0-59) means 60 tasks, each processing ~1000 genomes
# With 64 GPU limit on SL2, up to 60 can run in parallel
# Each task should complete in ~4 hours (1000 genomes * 15s = ~4.2 hours)

# Load required modules
module purge
# Try to load CUDA modules (may not be available on all systems)
module load cuda/12.4 2>/dev/null || echo "CUDA module not found, using system CUDA"
module load cudnn/8.9_cuda-12.4 2>/dev/null || echo "cuDNN module not found, using system cuDNN"

# Force Python unbuffered output for real-time logging
export PYTHONUNBUFFERED=1
# Set transformers verbosity for better logging
export TRANSFORMERS_VERBOSITY=info
# Use work directory for UV cache to avoid disk space issues in home directory
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
# Use RDS for HuggingFace cache (much faster I/O than home directory)
export HF_HOME=/home/dca36/rds/hpc-work/.huggingface_cache
export TRANSFORMERS_CACHE=/home/dca36/rds/hpc-work/.huggingface_cache

# Ensure uv is in PATH (adjust this path to where uv is installed)
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Change to project directory
cd /home/dca36/workspace/BacPredict

echo "=========================================="
echo "Bacformer Embedding Generation (Array Job)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Calculate file indices for this array task
# We need to get the total number of files first
# Run a quick count with Python
TOTAL_FILES=$(uv run python -c "
from pathlib import Path

root = Path('/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw')
input_dir = root / 'david' / 'processed' / 'klebsiella_protein_sequences'
esm_dir = root / 'david' / 'processed' / 'klebsiella_esm_embeddings'
bacformer_dir = root / 'david' / 'processed' / 'klebsiella_bacformer_embeddings'

protein_files = sorted(input_dir.glob('*_protein_sequences.parquet'))
# Filter to only unprocessed files
unprocessed = [
    f for f in protein_files 
    if not ((esm_dir / f'{f.stem.replace(\"_protein_sequences\", \"\")}_esm_embeddings.pt').exists() and 
            (bacformer_dir / f'{f.stem.replace(\"_protein_sequences\", \"\")}_bacformer_embeddings.pt').exists())
]
print(len(unprocessed))
")

echo "Total unprocessed files: $TOTAL_FILES"

# Calculate chunk size (divide total files by number of array tasks)
CHUNK_SIZE=$((TOTAL_FILES / 30 + 1))
START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * CHUNK_SIZE))

# Make sure we don't go past the end
if [ $END_IDX -gt $TOTAL_FILES ]; then
    END_IDX=$TOTAL_FILES
fi

echo "Processing files from index $START_IDX to $END_IDX"
echo "Chunk size: $CHUNK_SIZE files"
echo "=========================================="

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found in PATH"
    echo "PATH: $PATH"
    echo "Trying to use python directly from virtual environment..."
    
    # Try to use existing virtual environment
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        python src/bacpredict/embed/generate_bacformer_embeddings.py \
            --skip-existing \
            --start-idx $START_IDX \
            --end-idx $END_IDX
    else
        echo "ERROR: No .venv found and uv not available"
        exit 1
    fi
else
    echo "Using uv: $(which uv)"
    # Run the Python script with array job parameters
    uv run python src/bacpredict/embed/generate_bacformer_embeddings.py \
        --skip-existing \
        --start-idx $START_IDX \
        --end-idx $END_IDX
fi

echo "=========================================="
echo "End time: $(date)"
echo "Array Task $SLURM_ARRAY_TASK_ID completed"
echo "=========================================="
