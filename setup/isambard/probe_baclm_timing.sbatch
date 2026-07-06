#!/usr/bin/env bash
# Honest baclm throughput probe: sweep linear token budgets (n=3 each, fresh process so peak mem
# resets) and report s/genome + peak GPU mem per budget, stopping at the first OOM. The smoke
# already measured 8192 -> ~332 s/genome, so start above that to see how hard we can safely push.
#SBATCH --job-name=probe-baclm-timing
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
set -uo pipefail
: "${SCRATCHDIR:?}"
S="$SCRATCHDIR"; PY="$S/envs/bacpredict-gpu-venv/bin/python"
export HF_HOME="$S/cache/hf" TORCH_HOME="$S/cache/torch"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$S/tmp"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for TB in 12288 16384 24576 32768; do
  echo "##################### token_budget=$TB (n=3) #####################"
  rm -f "$S/processed/baclm_probe"/*.pt 2>/dev/null || true
  OUT=$("$PY" "$HOME/BacPredict/src/tl/embed/baclm_embed.py" \
        --input-csv "$S/processed/smoke_kleb/input_smoke.csv" --output-dir "$S/processed/baclm_probe" \
        --n 3 --token-budget "$TB" --device cuda:0 2>&1)
  echo "$OUT" | grep -iE "DONE|peak GPU|out of memory|ok=|fail=" | tail -4
  if echo "$OUT" | grep -qi "out of memory"; then echo ">>> OOM at token_budget=$TB — stopping sweep"; break; fi
done
echo "=== probe sweep done ==="
