#!/bin/bash
#SBATCH --job-name=bench_baclm_attn
#SBATCH --output=/rds/user/dca36/hpc-work/logs/bench_baclm_attn_%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/bench_baclm_attn_%j.err
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00

# Is flash-attn actually required on CSD3, and at what batch size?
#
# Prebuilt flash-attn wheels are unusable here: every recent release is built against GLIBC 2.32 and
# CSD3 has 2.28, so the .so fails to load regardless of the torch/CUDA tags. That leaves a source
# build (nvcc 12.1 against a cu128 torch, hours, fragile) — or not needing it.
#
# BacLM's own modeling code already branches: `if fa_func is not None` uses flash_attn_varlen, else
# it falls back to torch's F.scaled_dot_product_attention. Crucially, our pipeline length-sorts
# sequences into contiguous batches, so many batches are fully packed; BacLM then passes
# attn_mask=None and SDPA can dispatch to torch's OWN bundled FlashAttention-2 kernel, which is
# linear in memory. The repo's "~1 TB, 100x slower" warning describes the masked/math backend, and
# may simply not apply once batches are packed.
#
# That is an empirical question, so this measures it on REAL Kp genomes rather than guessing:
# throughput and peak memory at several batch sizes, using the shared venv (torch 2.10, no
# flash-attn), and projects the GPU-hours for the full 13,980-genome cohort.
#
# Usage: sbatch setup/csd3/benchmark_baclm_attention.sh

set -uo pipefail
cd /home/dca36/workspace/BacPredict
module purge
module load cuda/12.1 || echo "WARN: cuda module missing (torch bundles its own runtime)"
export PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-/rds/user/dca36/hpc-work/.huggingface_cache}
unset PYTHONPATH PYTHONHOME

PARQ=${PARQ:-/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/bacformer_processed/protein_sequences/klebsiella}
N_GENOMES=${N_GENOMES:-6}
COHORT_N=${COHORT_N:-13980}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run python - "$PARQ" "$N_GENOMES" "$COHORT_N" <<'PY'
import sys, time, glob
import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoTokenizer

parq_dir, n_genomes, cohort_n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

import importlib.util as u
print("flash_attn present:", u.find_spec("flash_attn") is not None, "| torch", torch.__version__)

files = sorted(glob.glob(f"{parq_dir}/*_protein_sequences.parquet"))[:n_genomes]
seqs_per_genome = []
for f in files:
    df = pd.read_parquet(f)
    col = next(c for c in df.columns if "seq" in c.lower() or df[c].dtype == object)
    seqs_per_genome.append([str(s) for s in df[col].tolist()])
lens = [len(s) for g in seqs_per_genome for s in g]
print(f"{len(files)} genomes | {sum(len(g) for g in seqs_per_genome)} proteins | "
      f"len median {int(np.median(lens))} p95 {int(np.percentile(lens,95))} max {max(lens)}")

MID = "macwiatrak/baclm-350m-masked"
tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
model = AutoModel.from_pretrained(MID, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()
MAX_LEN = 2048

def run(seqs, bs):
    """Length-sorted contiguous batches — the pipeline's own strategy, which is what packs batches."""
    order = np.argsort([len(s) for s in seqs])
    ordered = [seqs[i] for i in order]
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    n_tok = 0
    with torch.no_grad():
        for i in range(0, len(ordered), bs):
            chunk = ordered[i:i+bs]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
            enc = {k: v.to("cuda") for k, v in enc.items()}
            n_tok += int(enc["attention_mask"].sum())
            out = model(**enc)
            h = out.last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            (h * m).sum(1) / m.sum(1).clamp(min=1)   # the pipeline's masked mean-pool
    torch.cuda.synchronize()
    return time.time() - t0, torch.cuda.max_memory_allocated() / 2**30, n_tok

all_seqs = [s for g in seqs_per_genome for s in g]
print(f"\n{'batch':>6} {'sec':>8} {'seq/s':>9} {'peakGiB':>9}   projected full-cohort GPU-h")
best = None
for bs in (8, 16, 32, 64, 128):
    try:
        dt, peak, _ = run(all_seqs, bs)
        rate = len(all_seqs) / dt
        per_genome = dt / len(files)
        gpu_h = per_genome * cohort_n / 3600
        print(f"{bs:>6} {dt:>8.1f} {rate:>9.1f} {peak:>9.2f}   {gpu_h:>8.1f} h  ({per_genome:.2f} s/genome, coding only)")
        if best is None or rate > best[1]:
            best = (bs, rate, peak, gpu_h)
    except torch.cuda.OutOfMemoryError:
        print(f"{bs:>6}   OOM")
        torch.cuda.empty_cache()

if best:
    bs, rate, peak, gpu_h = best
    print(f"\nBEST: batch {bs} -> {rate:.0f} seq/s, peak {peak:.2f} GiB, ~{gpu_h:.1f} GPU-h coding-only "
          f"for {cohort_n} genomes.")
    print("Reference (Isambard, WITH flash-attn): ~700-800 seq/s, ~8.5 GiB at batch 128.")
    print("Non-coding adds ~65% more sequences, so multiply the projection by ~1.65 for a full pass.")
PY
