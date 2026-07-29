#!/bin/bash
#SBATCH --job-name=embeddings_klebsiella
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --time=36:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → embeddings_%A.out/.err (repo-relative), and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.

# Script to run ESM-C embedding generation on HPC with GPU (single-job).
# Pass --bacformer-embeddings to also produce Bacformer contextualised outputs.
# Usage:
#   sbatch src/bacpredict/engine/embedding/scripts/run_embeddings.sh --n 10                  # Test with 10 files
#   sbatch src/bacpredict/engine/embedding/scripts/run_embeddings.sh                         # ESM-C only, all files
#   sbatch src/bacpredict/engine/embedding/scripts/run_embeddings.sh --skip-existing         # Resume
#   sbatch src/bacpredict/engine/embedding/scripts/run_embeddings.sh --bacformer-embeddings  # Add Bacformer outputs

# CUDA comes from the Isambard Cray PE + the venv — no `module load` needed.

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
echo "Embedding Generation (ESM-C; Bacformer opt-in)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Arguments: $@"
echo "=========================================="

# Run the Python script with all passed arguments
"$PY" -m bacpredict.engine.embedding.generate_embeddings "$@"

echo "=========================================="
echo "End time: $(date)"
echo "=========================================="
