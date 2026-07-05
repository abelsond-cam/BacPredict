#!/usr/bin/env bash
# Quick baclm throughput probe (n=3, GPU): measure s/genome at a high token budget to size the
# full-cohort GPU-hour cost, and confirm the raised budget doesn't OOM. Reuses the smoke input CSV.
#SBATCH --job-name=probe-baclm-timing
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
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
rm -f "$S/processed/baclm_probe"/*.pt 2>/dev/null || true  # clean prior partial outputs
"$PY" "$HOME/BacPredict/src/tl/embed/baclm_embed.py" \
  --input-csv "$S/processed/smoke_kleb/input_smoke.csv" --output-dir "$S/processed/baclm_probe" \
  --n 3 --attn-budget "${ATTN_BUDGET:-30000000}" --device cuda:0
