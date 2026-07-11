#!/bin/bash
#SBATCH --job-name=genome_embeddings
#SBATCH --output=genome_embeddings_%j.out
#SBATCH --error=genome_embeddings_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --account=FLOTO-SL2-CPU

# Change to the workspace directory
cd /home/dca36/workspace/BacPredict

# Run the script with uv
# Using 32 workers to process ~65,000 files efficiently
echo "Starting genome embeddings generation at $(date)"
echo "Using 32 parallel workers"
echo ""

uv run python src/tl/embed/genome_assemblies_from_bacformer_embeddings.py \
    --workers 32

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Job completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
