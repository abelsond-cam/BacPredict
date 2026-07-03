#!/usr/bin/env bash
# Build the lean bacpredict-gpu env on Isambard-AI (aarch64) by uv-syncing the
# dedicated project at setup/isambard/gpu/pyproject.toml (bacformer[git] + torch cu126
# + transformers<5), bootstrapped from nuna's proven pins. ESM-C + frozen Bacformer +
# baclm-350m-masked forward passes only. Then caches all three models to $HF_HOME.
#
# aarch64 specifics (learned from nuna's builds): steer torch to cu126 wheels (node CUDA
# toolkit 12.6); force a uv-MANAGED standalone Python (system /usr/bin/python3.11 lacks
# stdlib sqlite3, which the ESM-C trust_remote_code modeling file imports); relocate the
# multi-GB venv + uv cache off $HOME (100 GiB; a full home locks SSH login).
set -uo pipefail
: "${SCRATCHDIR:?SCRATCHDIR must be set (Isambard personal 5 TiB space)}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"   # uv

# Everything BacPredict lives on the personal 5 TiB $SCRATCHDIR (persistent to project
# end on Isambard-AI; NOT the shared /projects/u6fp group allocation). Force HF/TORCH
# caches here too (override any shared-drive default in .bashrc) so the project is
# self-contained on David's own space.
PROJ="$HOME/BacPredict/setup/isambard/gpu"
export UV_PROJECT_ENVIRONMENT="$SCRATCHDIR/envs/bacpredict-gpu-venv"
export UV_PYTHON_PREFERENCE=only-managed   # managed python-build-standalone bundles sqlite3
export UV_PYTHON=3.11
export HF_HOME="$SCRATCHDIR/cache/hf"
export TORCH_HOME="$SCRATCHDIR/cache/torch"

echo "uv: $(uv --version)"
echo "project -> $PROJ"
echo "venv    -> $UV_PROJECT_ENVIRONMENT"
echo "HF_HOME -> $HF_HOME"
cd "$PROJ"

echo "=== ensure a managed standalone Python 3.11 (has sqlite3) ==="
uv python install 3.11 || { echo "uv python install FAILED"; exit 1; }
if [ -e "$UV_PROJECT_ENVIRONMENT/bin/python" ] && ! "$UV_PROJECT_ENVIRONMENT/bin/python" -c "import sqlite3" 2>/dev/null; then
  echo "existing venv lacks sqlite3 (system Python) -> removing to rebuild on managed"
  rm -rf "$UV_PROJECT_ENVIRONMENT"
fi

echo "=== uv sync (torch-cu126 + bacformer[git] + tree; managed Python) ==="
uv sync || { echo "UV SYNC FAILED"; exit 1; }
uv run python -c "import sqlite3; print('sqlite3 OK', sqlite3.sqlite_version)" \
  || { echo "sqlite3 STILL MISSING after managed-Python rebuild"; exit 1; }

echo "=== CPU-safe import smoke (GPU + model load checked later on a GH200) ==="
uv run python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| cuda_build", torch.version.cuda)
print("transformers", transformers.__version__)
import bacformer, bacformer.pp  # noqa: F401
from bacformer.pp import compute_genome_protein_embeddings, protein_embeddings_to_inputs  # noqa: F401
print("bacformer + bacformer.pp import OK")
PY

echo "=== cache all three models to \$HF_HOME (persistent; CPU node has egress) ==="
uv run python "$HOME/BacPredict/setup/isambard/cache_models.py" || { echo "MODEL CACHE FAILED"; exit 1; }
echo "=== DONE bacpredict-gpu env build + model cache ==="
