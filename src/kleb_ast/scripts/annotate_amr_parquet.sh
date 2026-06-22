#!/bin/bash
# Kp AST cohort: reliable Kleborate-style AMR labels → per-sample {Sample}_amr.parquet sidecars.
#
# CPU array, one CHUNK of genomes per array task. Each task minimap2-aligns the vendored CARD +
# chromosomal AMR refs against each genome's assembly and writes a flat-index-keyed AMR sidecar
# (no protein parquet rewrite, no re-embedding). Python runs under `uv run`; the minimap2 binary
# comes from this task's pixi env (never `module load` — see csd3-spack-module-breaks-uv).
#
# One-time setup (login node, needs internet):
#     cd /home/dca36/workspace/BacPredict/src/kleb_ast && pixi install
#
# Size the array first (login node, seconds):
#     cd /home/dca36/workspace/BacPredict
#     MM2=$(cd src/kleb_ast && pixi run -- which minimap2)
#     uv run python -m kleb_ast.annotate_amr_sidecar --dry-run --minimap2-bin "$MM2"
#   -> prints "worklist size = N". With CHUNK=200, set --array=0-$(( (N+199)/200 - 1 )).
#
# Smoke (5 known carbapenemase carriers; login node or a short interactive — minutes):
#     uv run python -m kleb_ast.annotate_amr_sidecar --minimap2-bin "$MM2" \
#         --samples SAMPLE_A SAMPLE_B ... --workers 5
#
# Full run:
#     sbatch --array=0-NN src/kleb_ast/scripts/annotate_amr_parquet.sh
#
#SBATCH --job-name=kleb_amr_annotate
#SBATCH --output=kleb_amr_annotate_%A_%a.out
#SBATCH --error=kleb_amr_annotate_%A_%a.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --open-mode=append

set -euo pipefail
cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

CHUNK=${CHUNK:-200}
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

# minimap2 from the kleb_ast pixi env (absolute path; no PATH/module fiddling).
MM2=$(cd src/kleb_ast && pixi run -- which minimap2)
echo "=== Kp AMR annotate — array task $SLURM_ARRAY_TASK_ID, chunk [$START:$((START+CHUNK))], minimap2=$MM2 ==="

# 32 cores: 8 genome workers × 4 minimap2 threads each.
uv run python -m kleb_ast.annotate_amr_sidecar \
    --minimap2-bin "$MM2" \
    --workers 8 --threads 4 \
    --start "$START" --count "$CHUNK" \
    --skip-existing

echo "Kp AMR annotate (task $SLURM_ARRAY_TASK_ID) finished."
