#!/bin/bash
#SBATCH --job-name=setup_baclm_env
#SBATCH --output=/rds/user/dca36/hpc-work/logs/setup_baclm_env_%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/setup_baclm_env_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU

# Build a DEDICATED CSD3 environment for baclm inference, with flash-attn.
#
# Why a separate env rather than adding flash-attn to the shared .venv:
#   * The shared venv runs torch 2.10.0+cu128. Dao-AILab publish prebuilt flash-attn wheels only up
#     to torch 2.9 (cu13) / torch 2.8 (cu12) for cp312 — there is no torch-2.10 wheel, and the wheels
#     are compiled against a specific torch ABI, so a 2.9 wheel on 2.10 fails at import.
#   * The only ways to get flash-attn onto torch 2.10 are a source build (CSD3's newest nvcc module is
#     12.1 against a cu128 torch, ~1-3 h and fragile) or downgrading torch in the shared venv — which
#     would disturb every other job using it, including live GPU fine-tunes.
#   * Isambard already solves this the same way: baclm runs under its own env there (Maciej's shared
#     bacformer env), not the project venv. This mirrors that.
#
# So: pin torch 2.8.0+cu128 in an isolated env and install the matching prebuilt wheel. No build,
# no compiler, no risk to the shared venv. baclm inference needs only torch + transformers +
# flash-attn + parquet IO, so the env stays small.
#
# Usage: sbatch setup/csd3/setup_baclm_env.sh
#        then verify on a GPU node: sbatch setup/csd3/verify_baclm_env.sh

set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/rds/user/dca36/hpc-work/.uv_cache
unset PYTHONPATH PYTHONHOME

ENV_DIR=${ENV_DIR:-/rds/user/dca36/hpc-work/.venv/baclm}
TORCH_VER=${TORCH_VER:-2.8.0}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
FA_VER=${FA_VER:-2.8.3.post1}
FA_TAG=${FA_TAG:-cu12torch2.8cxx11abiTRUE}
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/flash_attn-${FA_VER}+${FA_TAG}-cp312-cp312-linux_x86_64.whl"

echo "env=$ENV_DIR torch=$TORCH_VER flash-attn=$FA_VER ($FA_TAG)"
mkdir -p "$(dirname "$ENV_DIR")" /rds/user/dca36/hpc-work/logs

uv venv --python 3.12 "$ENV_DIR"
export VIRTUAL_ENV="$ENV_DIR"

echo "=== torch ${TORCH_VER} from ${TORCH_INDEX} ==="
uv pip install --python "$ENV_DIR/bin/python" "torch==${TORCH_VER}" --index-url "$TORCH_INDEX"

echo "=== runtime deps ==="
uv pip install --python "$ENV_DIR/bin/python" \
    "transformers>=4.44" "tokenizers" "safetensors" "huggingface_hub" \
    "pandas" "pyarrow" "numpy" "einops" "tqdm"

echo "=== flash-attn prebuilt wheel ==="
echo "  $FA_URL"
uv pip install --python "$ENV_DIR/bin/python" --no-deps "$FA_URL"

echo "=== verify (CPU import only; the CUDA kernels need a GPU node) ==="
"$ENV_DIR/bin/python" - <<'PY'
import importlib.util as u, torch
print("torch", torch.__version__, "cuda-built", torch.version.cuda)
spec = u.find_spec("flash_attn")
print("flash_attn spec:", "FOUND" if spec else "MISSING")
import flash_attn
print("flash_attn", flash_attn.__version__)
from flash_attn import flash_attn_varlen_func  # the entry point baclm_embed relies on
print("flash_attn_varlen_func import OK")
PY

echo "=== done: $ENV_DIR ==="
echo "Point baclm at it with:  BACLM_PYTHON=$ENV_DIR/bin/python"
