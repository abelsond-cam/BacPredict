#!/usr/bin/env bash
# Full-cohort ESM-C + Bacformer embeddings (GPU array). Each array task embeds a fixed slice
# of the sorted protein-parquet list (slice = ceil(TOTAL/NTASKS)); --skip-existing makes it
# re-runnable. Parameterised by TASK + array size; submit with e.g.:
#   sbatch --export=ALL,TASK=tb   -J embed-esmbac-tb   --array=0-31 setup/isambard/embed_esm_bacformer.sbatch
#   sbatch --export=ALL,TASK=kleb -J embed-esmbac-kleb --array=0-7  setup/isambard/embed_esm_bacformer.sbatch
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
set -uo pipefail
: "${SCRATCHDIR:?}" "${TASK:?set TASK=tb|kleb}"
S="$SCRATCHDIR"
PY="$S/envs/bacpredict-gpu-venv/bin/python"
export HF_HOME="$S/cache/hf" TORCH_HOME="$S/cache/torch"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
EMB="$HOME/BacPredict/src/tl/embed"

case "$TASK" in
  tb)   PROC="$S/processed/train_tb_ast" ;;
  kleb) PROC="$S/processed/train_kleb_ast" ;;
  *) echo "unknown TASK=$TASK"; exit 1 ;;
esac
cd "$S/tmp"

# Fixed contiguous slice of the sorted parquet list for this array task.
TOTAL=$(find "$PROC/protein_sequences" -name '*_protein_sequences.parquet' -type f | wc -l)
NT="${SLURM_ARRAY_TASK_COUNT:-1}"; ID="${SLURM_ARRAY_TASK_ID:-0}"
CHUNK=$(( (TOTAL + NT - 1) / NT ))
START=$(( ID * CHUNK )); END=$(( START + CHUNK )); [ "$END" -gt "$TOTAL" ] && END=$TOTAL
echo "=== $TASK task $ID/$NT : TOTAL=$TOTAL slice [$START:$END] ==="
[ "$START" -ge "$TOTAL" ] && { echo "slice past end, nothing to do"; exit 0; }

"$PY" "$EMB/generate_embeddings.py" \
  --input-dir "$PROC/protein_sequences" --esm-dir "$PROC/esm" \
  --bacformer-dir "$PROC/bacformer" --bacformer-embeddings \
  --start-idx "$START" --end-idx "$END" --skip-existing --device cuda:0
