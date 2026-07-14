#!/bin/bash
# Kp AST cohort: reliable Kleborate-style AMR labels → per-sample {Sample}_amr.parquet sidecars.
#
# CPU array, one CHUNK of genomes per array task. Each task minimap2-aligns the vendored CARD +
# chromosomal AMR refs against each genome's assembly and writes a flat-index-keyed AMR sidecar
# (no protein parquet rewrite, no re-embedding). Python runs under the venv $PY; the minimap2 binary
# comes from this task's pixi env (never `module load` — see csd3-spack-module-breaks-uv).
#
# One-time setup (login node, needs internet):
#     cd "$HOME/BacPredict/src/bacpredict/apps/kleb" && pixi install
#
# Size the array first (login node, seconds):
#     MM2=$(cd "$HOME/BacPredict/src/bacpredict/apps/kleb" && pixi run -- which minimap2)
#     "$PY" -m bacpredict.apps.kleb.annotate_amr_sidecar --dry-run --minimap2-bin "$MM2"
#   -> prints "worklist size = N". With CHUNK=200, set --array=0-$(( (N+199)/200 - 1 )).
#
# Smoke (5 known carbapenemase carriers; login node or a short interactive — minutes):
#     "$PY" -m bacpredict.apps.kleb.annotate_amr_sidecar --minimap2-bin "$MM2" \
#         --samples SAMPLE_A SAMPLE_B ... --workers 5
#
# Full run:
#     sbatch --array=0-NN src/bacpredict/apps/kleb/scripts/annotate_amr_parquet.sh
#
#SBATCH --job-name=kleb_amr_annotate
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=icelake-himem --account=FLOTO-PROJECT-K-SL2-CPU,
#   logs → a project-tier logs dir (e.g. ~/rds/hpc-work/logs/%x-%A_%a.out).

set -euo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

CHUNK=${CHUNK:-200}
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

# minimap2 from the kleb pixi env (absolute path; no PATH/module fiddling). Prefer a
# pre-resolved $MM2 (passed via --export) so array tasks don't each invoke pixi on a compute node.
MM2=${MM2:-$(cd "$HOME/BacPredict/src/bacpredict/apps/kleb" && pixi run -- which minimap2)}
echo "=== Kp AMR annotate — array task $SLURM_ARRAY_TASK_ID, chunk [$START:$((START+CHUNK))], minimap2=$MM2 ==="

# 32 cores: 8 genome workers × 4 minimap2 threads each.
"$PY" -m bacpredict.apps.kleb.annotate_amr_sidecar \
    --minimap2-bin "$MM2" \
    --workers 8 --threads 4 \
    --start "$START" --count "$CHUNK" \
    --skip-existing

echo "Kp AMR annotate (task $SLURM_ARRAY_TASK_ID) finished."
