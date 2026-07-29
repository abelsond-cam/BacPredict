#!/usr/bin/env bash
# Stage 2a — baclm-vs-ESM coding LEARNING CURVE (AUROC vs training-set size), the whole panel in one
# parquet sweep. Fixed evaluate holdout; per rung a stratified subsample of the shared train pool feeds
# the SAME rows to the ESM and baclm LRs (paired Δ). Distinguishes "baclm gap closes with data"
# (data-hungry embedding) from "baclm has a lower ceiling" (persistent gap). CPU-only.
#
#   sbatch --export=ALL,TASK=tb   -J coding-ladder-tb   src/bacpredict/engine/scripts/coding_amr_ladder.sh
#   sbatch --export=ALL,TASK=kleb -J coding-ladder-kleb src/bacpredict/engine/scripts/coding_amr_ladder.sh
#SBATCH --partition=icelake-himem
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
# CPU-only — NO --gres=gpu. A no-GPU job on workq schedules normally (reaches PENDING(Priority) and
# runs); the PENDING(None) seen right after submit is transient, not a stall — do not add a GPU handle
# to "fix" it. Memory defaults are GPU-tied here (DefMemPerGPU), so a GPU-less job MUST set --mem
# explicitly. LR at large n dominates — 12 h is headroom.
set -uo pipefail
: "${SCRATCHDIR:?}" "${TASK:=tb}"
S="$SCRATCHDIR"
PY="$S/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg
case "$TASK" in
  tb)   SPECIES=tb ;;
  kleb) SPECIES=kp ;;
  *) echo "unknown TASK=$TASK (want tb|kleb)"; exit 1 ;;
esac
# Fine 500-increments up to n=6000 (the steep, informative low-n regime), then 4x coarser to the full
# pool. Override STEP/FINE_UNTIL to change; FINE_UNTIL >= pool forces literal 500-increments throughout.
STEP="${STEP:-500}"
FINE_UNTIL="${FINE_UNTIL:-6000}"
OUT="$S/processed/train_${TASK}_ast/pangena_predict/coding_amr_lr/ladder_${SPECIES}_${SLURM_JOB_ID:-local}.json"

echo "=== coding_amr_lr ladder: species=$SPECIES step=$STEP fine_until=$FINE_UNTIL workers=${SLURM_CPUS_PER_TASK:-8} ==="
"$PY" "$HOME/BacPredict/src/bacpredict/engine/gene_lr/coding_amr_lr.py" \
  --species "$SPECIES" --panel --ladder \
  --ladder-step "$STEP" --ladder-fine-until "$FINE_UNTIL" \
  --seeds 1,2,3 --plot \
  --pool-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --output "$OUT"
echo "ladder JSON -> $OUT"
echo "ladder PNG  -> ${OUT%.json}.png"
