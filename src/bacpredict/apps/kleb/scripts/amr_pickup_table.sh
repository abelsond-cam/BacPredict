#!/bin/bash
# CARD vs Kleborate vs Bakta AMR-gene pickup table (bacpredict.apps.kleb.amr_pickup_table).
#
# I/O-bound: concatenates ~6.4k {Sample}_amr.parquet sidecars off scratch (~17 min wall, low CPU), so it runs
# past the login-node watchdog ceiling — submit it as a job rather than running it on the login node.
#
#     sbatch src/bacpredict/apps/kleb/scripts/amr_pickup_table.sh
#
#SBATCH --job-name=kleb_amr_pickup_table
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

echo "=== CARD vs Kleborate vs Bakta pickup table ==="
"$PY" -m bacpredict.apps.kleb.amr_pickup_table
echo "pickup table finished -> src/bacpredict/visualisations/kp/amr_annotation/card_vs_kleborate_vs_bakta_pickup.{csv,md}"
