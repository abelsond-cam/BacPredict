#!/usr/bin/env bash
# Full-cohort baclm coding + non-coding embeddings (GPU array). Each array task embeds a fixed
# slice of the sorted input-CSV rows; --skip-existing makes it re-runnable. Parameterised by
# TASK + array size; submit with e.g.:
#   sbatch --export=ALL,TASK=tb   -J embed-baclm-tb   --array=0-31 setup/isambard/embed_baclm.sbatch
#   sbatch --export=ALL,TASK=kleb -J embed-baclm-kleb --array=0-7  setup/isambard/embed_baclm.sbatch
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=18:00:00
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
set -uo pipefail
: "${SCRATCHDIR:?}" "${TASK:?set TASK=tb|kleb}"
S="$SCRATCHDIR"
# baclm needs flash-attn (built for GH200). Our lean venv has no aarch64 flash-attn wheel and the
# system gcc is too old to build it — so run baclm with Maciej Wiatrak's shared bacformer env, which
# already has torch 2.9.1 + flash_attn 2.8.3 (matches our torch pin). Without flash-attn baclm's
# attention falls back to a dense O((batch×len)²) path that OOMs / runs ~100× slower.
PY="/projects/u6fp/public/micromamba/envs/bacformer/bin/python"
export HF_HOME="$S/cache/hf" TORCH_HOME="$S/cache/torch"
# On Isambard $HOME/BacPredict is the stale pre-consolidation `dev` checkout (no bacpredict pkg); the
# consolidated code lives in the worktree. Override REPO at submit if elsewhere.
REPO="${REPO:-$SCRATCHDIR/worktrees/consolidate}"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EMB="$REPO/src/bacpredict/engine/embedding"

case "$TASK" in
  tb)   PROC="$S/processed/train_tb_ast" ;;
  kleb) PROC="$S/processed/train_kleb_ast" ;;
  *) echo "unknown TASK=$TASK"; exit 1 ;;
esac
cd "$S/tmp"

# baclm reads the shared protein + intergenic parquet stores (extract_proteins.sbatch); slice the
# sorted protein-parquet list so each array task owns a fixed set regardless of scheduling.
TOTAL=$(find "$PROC/protein_sequences" -name '*_protein_sequences.parquet' -type f | wc -l)
NT="${SLURM_ARRAY_TASK_COUNT:-1}"; ID="${SLURM_ARRAY_TASK_ID:-0}"
CHUNK=$(( (TOTAL + NT - 1) / NT ))
START=$(( ID * CHUNK )); END=$(( START + CHUNK )); [ "$END" -gt "$TOTAL" ] && END=$TOTAL
echo "=== $TASK baclm task $ID/$NT : TOTAL=$TOTAL slice [$START:$END] ==="
[ "$START" -ge "$TOTAL" ] && { echo "slice past end, nothing to do"; exit 0; }

# FRESH=1 drops --skip-existing to force a full re-embed (overwrite stale baclm/*.pt in place) — used
# for the non-coding-channel re-embed; default keeps --skip-existing for re-runnable incremental jobs.
SKIP="--skip-existing"; [ "${FRESH:-0}" = 1 ] && SKIP=""
"$PY" "$EMB/baclm_embed.py" \
  --protein-dir "$PROC/protein_sequences" --intergenic-dir "$PROC/intergenic" \
  --output-dir "$PROC/baclm" \
  --start-idx "$START" --end-idx "$END" $SKIP --device cuda:0   # default batch-size (128, flash-attn)
