#!/usr/bin/env bash
# Stage-A smoke (n=10, GPU): exercise the whole Isambard embedding pipeline end-to-end on 10
# Kp genomes — build input CSV -> extract proteins -> ESM-C + Bacformer -> baclm coding+non-coding
# -> assert every output exists with the right shapes. Pinned ESM-C + Bacformer + baclm all load
# from the pre-cached $HF_HOME (no egress needed). Minimal GPU spend (<~15 min of a 1 h budget).
#SBATCH --job-name=smoke-embed-kleb
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
set -uo pipefail
: "${SCRATCHDIR:?}"
S="$SCRATCHDIR"
PY="$S/envs/bacpredict-gpu-venv/bin/python"
export HF_HOME="$S/cache/hf" TORCH_HOME="$S/cache/torch"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SMOKE="$S/processed/smoke_kleb"
EMB="$HOME/BacPredict/src/tl/embed"
mkdir -p "$SMOKE" "$S/tmp"
cd "$S/tmp"   # keep the scripts' CWD log files off $HOME / the repo

echo "=== node $(hostname) ; gpu ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== 1. build input CSV (Kp) + take first 10 ==="
"$PY" "$EMB/build_ast_input_csv.py" --base "$S/raw/kleb_ast" --out "$SMOKE/input_full.csv"
head -n 11 "$SMOKE/input_full.csv" > "$SMOKE/input_smoke.csv"
echo "smoke rows: $(( $(wc -l < "$SMOKE/input_smoke.csv") - 1 ))"

echo "=== 2. extract protein sequences ==="
"$PY" "$EMB/preprocess_assemblies_to_protein_sequences.py" \
  --input-csv "$SMOKE/input_smoke.csv" --output-dir "$SMOKE/protein_sequences" --workers 8

echo "=== 3. ESM-C + Bacformer embeddings ==="
"$PY" "$EMB/generate_embeddings.py" \
  --input-dir "$SMOKE/protein_sequences" --esm-dir "$SMOKE/esm" \
  --bacformer-dir "$SMOKE/bacformer" --bacformer-embeddings --device cuda:0

echo "=== 4. baclm coding + non-coding embeddings ==="
"$PY" "$EMB/baclm_embed.py" \
  --input-csv "$SMOKE/input_smoke.csv" --output-dir "$SMOKE/baclm" --device cuda:0

echo "=== 5. assert outputs + shapes ==="
"$PY" - "$SMOKE" <<'PY'
import sys, glob, torch
S = sys.argv[1]
def one(pat):
    fs = sorted(glob.glob(pat)); assert fs, f"no files: {pat}"; return fs
esm = one(f"{S}/esm/*_esm_embeddings.pt")
bac = one(f"{S}/bacformer/*_bacformer_embeddings.pt")
bcl = one(f"{S}/baclm/*_baclm_embeddings.pt")
print(f"counts: esm={len(esm)} bacformer={len(bac)} baclm={len(bcl)}")
e = torch.load(esm[0], map_location="cpu")
assert "protein_embeddings" in e, f"esm keys: {list(e)}"
print("esm  protein_embeddings:", tuple(e["protein_embeddings"].shape), e["protein_embeddings"].dtype)
b = torch.load(bac[0], map_location="cpu")
assert b.shape[-1] == 960, b.shape
print("bacformer last_hidden_state:", tuple(b.shape), b.dtype)
c = torch.load(bcl[0], map_location="cpu")
assert c["protein_embeddings"].shape[-1] == 960 and c["intergenic_embeddings"].shape[-1] == 960, c
print("baclm protein:", tuple(c["protein_embeddings"].shape),
      "| intergenic:", tuple(c["intergenic_embeddings"].shape), c["protein_embeddings"].dtype)
assert len(esm) == 10 == len(bac) == len(bcl), "expected 10 of each"
print("=== SMOKE PASS: proteins + ESM-C + Bacformer + baclm(coding+noncoding) OK on GH200, 960-d ===")
PY
echo "=== smoke done (rc above) ==="
