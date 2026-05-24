#!/bin/bash
#SBATCH --job-name=bacformer_embed_klebsiella
#SBATCH --output=bacformer_embed_%A.out
#SBATCH --error=bacformer_embed_%A.err
#SBATCH --time=36:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G

# Script to run Bacformer embedding generation on HPC with GPU
# Usage:
#   sbatch src/bacpredict/embed/scripts/run_bacformer_embeddings.sh --n 10  # Test with 10 files
#   sbatch src/bacpredict/embed/scripts/run_bacformer_embeddings.sh         # Process all files
#   sbatch src/bacpredict/embed/scripts/run_bacformer_embeddings.sh --skip-existing  # Resume

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

# Ensure uv is in PATH (adjust this path to where uv is installed)
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Change to project directory
cd /home/dca36/workspace/BacPredict

echo "=========================================="
echo "Bacformer Embedding Generation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Arguments: $@"
echo "=========================================="

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found in PATH"
    echo "PATH: $PATH"
    echo "Trying to use python directly from virtual environment..."
    
    # Try to use existing virtual environment
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        python src/bacpredict/embed/generate_bacformer_embeddings.py "$@"
    else
        echo "ERROR: No .venv found and uv not available"
        exit 1
    fi
else
    echo "Using uv: $(which uv)"
    # Run the Python script with all passed arguments
    uv run python src/bacpredict/embed/generate_bacformer_embeddings.py "$@"
fi

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
