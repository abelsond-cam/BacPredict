#!/bin/bash
#SBATCH --job-name=kleb_bacformer_mean
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → a project-tier logs dir, and `module purge; module load cuda/12.4 cudnn/8.9_cuda-12.4`.
#
# Cache the Kp frozen Bacformer genome-mean for the whole cohort (~6.8k genomes), one forward each.
# Drug-agnostic; written once so every per-drug concat probe + ladder runs CPU-only thereafter.
# ~10-20 min of compute, but time is generous per the never-under-call rule.
#
# Submit (ensure the HPC checkout is on the right branch + up to date first):
#   sbatch src/bacpredict/apps/kleb/scripts/cache_bacformer_genome_mean.sh

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction"}"
D="$BACPREDICT_DATA_ROOT"
PY="$HOME/workspace/BacPredict/.venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"$PY" -m bacpredict.apps.kleb.cache_bacformer_genome_mean --device cuda:0
echo "BACFORMER_MEAN_CACHE_DONE"
