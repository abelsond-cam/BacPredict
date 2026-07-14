#!/bin/bash
#SBATCH --job-name=genome_embeddings
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=00:25:00
# CSD3/UoHPC variant (when it returns): --partition=icelake --account=FLOTO-SL2-CPU,
#   logs → genome_embeddings_%j.out/.err (repo-relative).

set -uo pipefail
# Data root + env — cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

# Run the script — 32 workers to process ~65,000 files efficiently
echo "Starting genome embeddings generation at $(date)"
echo "Using 32 parallel workers"
echo ""

"$PY" -m bacpredict.engine.embedding.genome_assemblies_from_bacformer_embeddings \
    --workers 32

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Job completed at $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
