#!/usr/bin/env bash
# Stage 2c — baclm PROMOTER (intergenic) → AMR learning curves. For each flank gene (fabG1/eis/pncA)
# it locates the intergenic region abutting the gene's 5' end (strand from the Bakta GFF), pulls that
# baclm intergenic_embeddings row, and runs the AUROC-vs-training-size ladder against the drug label.
# Folds in the build-quality audit (CDS-flanked vs RNA-abutting, truncation). CPU-only.
#
#   sbatch --export=ALL,TASK=tb   -J igr-amr-tb   src/pangena_predict/scripts/igr_amr_lr.sh
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
# CPU-only — NO --gres=gpu (a no-GPU job schedules normally on workq; --mem is required since memory
# defaults are GPU-tied). Cost is one GFF parse + one .pt load per genome, parallelised by workers.
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
STEP="${STEP:-500}"
FINE_UNTIL="${FINE_UNTIL:-6000}"
OUT="$S/processed/train_${TASK}_ast/pangena_predict/igr_amr_lr/promoter_${SPECIES}_${SLURM_JOB_ID:-local}.json"

echo "=== igr_amr_lr promoter: species=$SPECIES step=$STEP fine_until=$FINE_UNTIL workers=${SLURM_CPUS_PER_TASK:-8} ==="
"$PY" "$HOME/BacPredict/src/pangena_predict/igr_amr_lr.py" \
  --species "$SPECIES" \
  --ladder-step "$STEP" --ladder-fine-until "$FINE_UNTIL" \
  --seeds 1,2,3 --plot \
  --pool-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --output "$OUT"
echo "IGR JSON -> $OUT"
echo "IGR PNG  -> ${OUT%.json}.png"
