#!/usr/bin/env bash
# Stage 2a — full baclm-vs-ESM coding panel (k=5 × s=3), the go/no-go validation that baclm embeds
# coding regions with the information ESM holds. CPU-only (sklearn LR over precomputed 960-vectors);
# the cost is ~38k single-row mmap .pt reads per gene, parallelised by --pool-workers.
#
#   sbatch --export=ALL,TASK=tb   -J coding-amr-tb   src/pangena_predict/scripts/coding_amr_lr_panel.sh
#   sbatch --export=ALL,TASK=kleb -J coding-amr-kleb src/pangena_predict/scripts/coding_amr_lr_panel.sh
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
# CPU-only — NO --gres=gpu. A no-GPU job schedules normally on workq; the PENDING(None) right after
# submit is transient, not a stall. Memory defaults are GPU-tied (DefMemPerGPU) so a GPU-less job MUST
# set --mem. The whole panel over ~38k genomes runs well under an hour per gene; 8 h is headroom.
set -uo pipefail
: "${SCRATCHDIR:?}" "${TASK:=tb}"
S="$SCRATCHDIR"
PY="$S/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
case "$TASK" in
  tb)   SPECIES=tb ;;
  kleb) SPECIES=kp ;;
  *) echo "unknown TASK=$TASK (want tb|kleb)"; exit 1 ;;
esac
OUT="$S/processed/train_${TASK}_ast/pangena_predict/coding_amr_lr/panel_${SPECIES}_${SLURM_JOB_ID:-local}.json"

echo "=== coding_amr_lr panel: species=$SPECIES workers=${SLURM_CPUS_PER_TASK:-8} ==="
"$PY" "$HOME/BacPredict/src/pangena_predict/coding_amr_lr.py" \
  --species "$SPECIES" --panel --n-folds 5 --seeds 1,2,3 \
  --pool-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --output "$OUT"
echo "panel JSON -> $OUT"
