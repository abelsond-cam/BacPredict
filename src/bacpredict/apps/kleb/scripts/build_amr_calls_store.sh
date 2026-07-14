#!/bin/bash
# Concatenate the ~6.4k {Sample}_amr.parquet sidecars into one amr_calls_all.parquet store
# (bacpredict.apps.kleb.build_amr_calls_store). I/O-bound (~17 min wall, low CPU) — past the login ceiling, so submit it.
# Run once after the sidecar array; downstream modules (card_determinant_lr, ladders) then read it in seconds.
#
#     sbatch src/bacpredict/apps/kleb/scripts/build_amr_calls_store.sh
#
#SBATCH --job-name=kleb_amr_calls_store
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=icelake-himem --account=FLOTO-PROJECT-K-SL2-CPU,
#   logs → a project-tier logs dir (e.g. ~/rds/hpc-work/logs/%x-%j.out).

set -euo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== build combined AMR-calls store ==="
"$PY" -m bacpredict.apps.kleb.build_amr_calls_store
echo "done -> amr_annotation/amr_calls_all.parquet"
