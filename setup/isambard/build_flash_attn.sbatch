#!/usr/bin/env bash
# Install flash-attn into the aarch64 GPU venv so baclm uses its efficient variable-length attention
# (flash_attn_varlen_qkvpacked_func). Without it, baclm falls back to a dense "packed SDPA with
# block-diagonal mask" path whose memory is O((batch × length)²) — 100× slower and OOMs (measured).
# No nvcc on the login node, so this MUST be a job: it module-loads cuda/12.6 (matches torch's
# cu126), adds the build deps the venv lacks (setuptools/ninja/…), and builds for GH200 (sm_90).
# Source build is slow; budget is generous. Submit: sbatch setup/isambard/build_flash_attn.sbatch
#SBATCH --job-name=build-flash-attn
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
set -euo pipefail
: "${SCRATCHDIR:?}"
S="$SCRATCHDIR"
VENV="$S/envs/bacpredict-gpu-venv"
PY="$VENV/bin/python"
export UV_CACHE_DIR="$S/cache/uv" HF_HOME="$S/cache/hf" TORCH_HOME="$S/cache/torch"
export TMPDIR="$S/tmp"; mkdir -p "$TMPDIR"

module load cuda/12.6
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
echo "nvcc: $(command -v nvcc)"; nvcc --version | tail -2
echo "CUDA_HOME=$CUDA_HOME"

# GH200 = Hopper sm_90. Bound compile parallelism so nvcc's per-file RAM doesn't blow the allocation.
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=32

echo "=== build deps into the venv (needed for --no-build-isolation) ==="
uv pip install --python "$PY" setuptools wheel ninja packaging psutil

echo "=== install flash-attn (aarch64 wheel if one resolves, else source build) ==="
time uv pip install --python "$PY" --no-build-isolation flash-attn

echo "=== verify the exact entrypoint baclm calls ==="
"$PY" - <<'PY'
import torch, flash_attn
from flash_attn import flash_attn_varlen_qkvpacked_func  # noqa: F401  (the fn baclm's attention uses)
print("flash_attn", flash_attn.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda)
print("flash-attn import + varlen entrypoint OK")
PY
echo "=== build-flash-attn done ==="
