#!/usr/bin/env bash
# Stage-A smoke (n=10, GPU): exercise the whole Isambard embedding pipeline end-to-end on 10
# Kp genomes — build input CSV -> extract proteins + intergenic parquets -> ESM-C + Bacformer ->
# EQUIVALENCE GATE (our fast baclm token_type_ids == the model's own) -> baclm coding+non-coding
# from parquet (timed) -> assert every output exists with the right shapes. Pinned ESM-C + Bacformer
# + baclm all load from the pre-cached $HF_HOME (no egress). Minimal GPU spend (<~15 min of 1 h).
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
PY="$S/envs/bacpredict-gpu-venv/bin/python"        # our pinned env: ESM-C + Bacformer (parity)
BACLM_PY="/projects/u6fp/public/micromamba/envs/bacformer/bin/python"  # Maciej's env: baclm (flash-attn)
export HF_HOME="$S/cache/hf" TORCH_HOME="$S/cache/torch"
# On Isambard $HOME/BacPredict is the stale pre-consolidation `dev` checkout; consolidated code is in
# the worktree. Override REPO at submit if elsewhere.
REPO="${REPO:-$SCRATCHDIR/worktrees/consolidate}"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SMOKE="$S/processed/smoke_kleb"
EMB="$REPO/src/bacpredict/engine/embedding"
mkdir -p "$SMOKE" "$S/tmp"
cd "$S/tmp"   # keep the scripts' CWD log files off $HOME / the repo

echo "=== node $(hostname) ; gpu ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "=== 1. build input CSV (Kp) + take first 10 ==="
"$PY" "$EMB/build_ast_input_csv.py" --base "$S/raw/kleb_ast" --out "$SMOKE/input_full.csv"
head -n 11 "$SMOKE/input_full.csv" > "$SMOKE/input_smoke.csv"
echo "smoke rows: $(( $(wc -l < "$SMOKE/input_smoke.csv") - 1 ))"

echo "=== 2. extract protein + intergenic parquets (CPU) ==="
"$PY" "$EMB/preprocess_assemblies_to_protein_sequences.py" \
  --input-csv "$SMOKE/input_smoke.csv" --output-dir "$SMOKE/protein_sequences" --workers 8
"$PY" "$EMB/extract_intergenic_to_parquet.py" \
  --input-csv "$SMOKE/input_smoke.csv" --output-dir "$SMOKE/intergenic" --workers 8

echo "=== 3. ESM-C + Bacformer embeddings ==="
"$PY" "$EMB/generate_embeddings.py" \
  --input-dir "$SMOKE/protein_sequences" --esm-dir "$SMOKE/esm" \
  --bacformer-dir "$SMOKE/bacformer" --bacformer-embeddings --device cuda:0

echo "=== 4. EQUIVALENCE GATE: our fast token_type_ids/input_ids == baclm's own (byte-identical) ==="
"$BACLM_PY" - "$SMOKE" <<'PY'
import glob, sys
import pandas as pd, torch
import torch.nn.functional as F
from bacpredict.engine.embedding.baclm_embed import _MODALITY, _encode_fast, load_baclm, mean_pool_embeddings
S = sys.argv[1]
model, tok = load_baclm("cuda:0")
special = torch.tensor(sorted(tok.all_special_ids))
pf = sorted(glob.glob(f"{S}/protein_sequences/*_protein_sequences.parquet"))[0]
proteins = [p for c in pd.read_parquet(pf)["protein_sequence"].iloc[0] for p in c][:64]
igf = sorted(glob.glob(f"{S}/intergenic/*_intergenic.parquet"))[0]
igs = list(pd.read_parquet(igf)["noncoding_sequence"].iloc[0])[:64]

for modality, seqs in [("protein", proteins), ("dna", igs)]:
    if not seqs:
        print(f"{modality}: no sequences in this genome, skipping"); continue
    mod_id = _MODALITY[modality]
    fast = _encode_fast(tok, seqs)
    fast_tt = torch.where(torch.isin(fast["input_ids"], special), 2, mod_id)
    slow = tok.batch_encode_plus(seqs, padding="longest", truncation=True, max_length=2048, return_tensors="pt")
    assert torch.equal(fast["input_ids"], slow["input_ids"]), f"{modality}: input_ids differ"
    assert torch.equal(fast_tt, slow["token_type_ids"]), f"{modality}: token_type_ids differ"
    # embeddings: identical inputs -> identical forward. Cross-check fast pool vs a slow-path forward.
    with torch.no_grad():
        d = {k: v.to("cuda:0") for k, v in slow.items()}
        h = model(input_ids=d["input_ids"], token_type_ids=d["token_type_ids"],
                  attention_mask=d["attention_mask"]).last_hidden_state
        m = d["attention_mask"].unsqueeze(-1).to(h.dtype)
        slow_pool = ((h * m).sum(1) / m.sum(1).clamp_min(1)).to(torch.bfloat16).cpu()
    fast_pool = mean_pool_embeddings(seqs, model, tok, "cuda:0", modality)
    cos = F.cosine_similarity(fast_pool.float(), slow_pool.float(), dim=1)
    assert cos.min() > 0.999, f"{modality}: min cosine {cos.min():.5f}"
    print(f"{modality}: input_ids+token_type_ids byte-identical, pooled cos.min={cos.min():.5f} "
          f"({tuple(fast['input_ids'].shape)})")
print("=== EQUIVALENCE GATE PASS: fast baclm path == developers' tokenizer ===")
PY

echo "=== 5. baclm coding + non-coding embeddings from parquet (timed) ==="
rm -rf "$SMOKE/baclm"   # clear any stale .pt so step 6 can't pass on old outputs
"$BACLM_PY" "$EMB/baclm_embed.py" \
  --protein-dir "$SMOKE/protein_sequences" --intergenic-dir "$SMOKE/intergenic" \
  --output-dir "$SMOKE/baclm" --device cuda:0   # default batch-size (128), flash-attn env

echo "=== 6. assert outputs + shapes ==="
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
for k in ("protein_embeddings", "noncoding_embeddings", "fragment_embeddings", "feature_embeddings"):
    assert k in c, f"baclm .pt missing {k}: {list(c)}"
    assert c[k].shape[-1] == 960, f"{k}: {c[k].shape}"
print("baclm protein:", tuple(c["protein_embeddings"].shape),
      "| noncoding:", tuple(c["noncoding_embeddings"].shape),
      "| fragment:", tuple(c["fragment_embeddings"].shape),
      "| feature:", tuple(c["feature_embeddings"].shape),
      "| feature_types:", sorted(set(c["feature_type"])), c["protein_embeddings"].dtype)
assert len(esm) == 10 == len(bac) == len(bcl), "expected 10 of each"
print("=== SMOKE PASS: proteins + ESM-C + Bacformer + baclm(coding + 3 non-coding channels) OK on GH200, 960-d ===")
PY
echo "=== smoke done (rc above) ==="
