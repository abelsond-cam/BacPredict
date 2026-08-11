#!/bin/bash
#SBATCH --job-name=verify_baclm_env
#SBATCH --output=/rds/user/dca36/hpc-work/logs/verify_baclm_env_%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/verify_baclm_env_%j.err
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# Verify the dedicated baclm env on a GPU node. An import check on a login node proves nothing:
# flash-attn's CUDA kernels are only exercised on device, and the classic failure mode of a
# prebuilt wheel is an undefined-symbol or arch-mismatch error at first kernel launch, not import.
# So this actually runs flash_attn_varlen_func and then a real baclm forward pass, and prints the
# peak memory — the number that decides whether batch 128 is safe (the docs' figure is ~8.5 GiB).
#
# Usage: sbatch setup/csd3/verify_baclm_env.sh

set -uo pipefail
ENV_DIR=${ENV_DIR:-/rds/user/dca36/hpc-work/.venv/baclm}
PY="$ENV_DIR/bin/python"
[ -x "$PY" ] || { echo "no interpreter at $PY — run setup_baclm_env.sh first"; exit 1; }

module purge
export PYTHONUNBUFFERED=1
unset PYTHONPATH PYTHONHOME

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "| cuda avail:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))

import flash_attn
from flash_attn import flash_attn_varlen_func
print("flash_attn", flash_attn.__version__)

# 1. Exercise the kernel directly — this is where an ABI/arch mismatch actually surfaces.
d, nh = 64, 8
lens = [512, 1024, 2048]
cu = torch.tensor([0, *torch.cumsum(torch.tensor(lens), 0).tolist()], dtype=torch.int32, device="cuda")
tot = sum(lens)
q, k, v = (torch.randn(tot, nh, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
out = flash_attn_varlen_func(q, k, v, cu, cu, max(lens), max(lens))
assert out.shape == (tot, nh, d), out.shape
assert torch.isfinite(out).all(), "non-finite output from flash_attn_varlen_func"
print("flash_attn_varlen_func OK ->", tuple(out.shape))

# 2. A real baclm forward pass at the batch size the pipeline uses.
from transformers import AutoModel, AutoTokenizer
MID = "macwiatrak/baclm-350m-masked"
tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
model = AutoModel.from_pretrained(MID, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
print("attn impl:", getattr(model.config, "_attn_implementation", None))

seqs = ["MKT" * 300] * 128                      # 128 x ~900 residues, the pipeline's batch shape
enc = tok(seqs, return_tensors="pt", padding=True, truncation=True, max_length=2048)
enc = {k_: v_.to("cuda") for k_, v_ in enc.items()}
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    o = model(**enc)
h = o.last_hidden_state
assert torch.isfinite(h).all(), "non-finite hidden states"
print("baclm forward OK ->", tuple(h.shape))
print(f"peak GPU mem: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB (docs: ~8.5 GiB at batch 128/len 2048)")
PY
status=$?
[ "$status" -ne 0 ] && { echo "VERIFY FAILED (exit $status)"; exit "$status"; }
echo "=== baclm env verified on GPU ==="
