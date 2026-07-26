#!/bin/bash
# E1 — Concat probe: ESM-C rpoB (960) ⊕ frozen Bacformer genome-mean (960) → LR on the full eval.
#
# Bypasses the broken prediction head: instead of asking the gated-MIL pool to route to rpoB,
# we hand the rpoB signal to a plain logistic regression alongside the genome mean. Three steps,
# one common evaluate set:
#   esm_rpob_only              frozen ESM-C mean-pooled rpoB 960-vector        (ladder ~0.971)
#   bacformer_mean_only        frozen Bacformer genome-mean 960-vector         (ladder ~0.788)
#   concat_esm_rpob_plus_mean  the two concatenated (1,920-d)                  the test
# The two ablations are the harness sanity check (must reproduce the ladder before the concat
# is trusted). GPU: one frozen Bacformer forward per genome to build the mean (ESM-C inputs are
# precomputed — NOT re-embedded), ~0.3 s/genome → ~2-3 GPU-h over the genotyped cohort.
#
# If a full-cohort frozen-Bacformer mean-vectors NPZ already exists, pass it as BAC_NPZ below and
# switch the directives to a CPU partition — the whole probe then runs without a GPU.
#
# Usage:  sbatch src/bacpredict/engine/scripts/run_concat_rpob_mean.sh
#
#SBATCH --job-name=concat_rpob_mean
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → concat_rpob_mean_%j.out/.err (repo-relative), and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.
# ~2-3 GPU-h estimated (genotype ~30k parquets + one Bacformer forward each); 24 h budget — never
# under-call walltime (charged on time used, not requested). --pool-workers parallelises the ESM-C
# rpoB .pt reads across the 8 cores.

set -uo pipefail
# Data root + env — cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
# CUDA comes from the Isambard Cray PE + the venv — no `module load` needed.
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1

RDS=$D/processed/train_tb_ast
SHEET=$RDS/binary_ast_with_split.csv
PARQUET_DIR=$RDS/protein_sequences
ESM_STORE_DIR=$RDS/esm
OUT_DIR=$RDS/pangena_predict/concat_rpob_mean
OUT_JSON=$OUT_DIR/concat_rpob_mean_${SLURM_JOB_ID}.json
QC_LOG=$OUT_DIR/rpob_copy_qc_${SLURM_JOB_ID}.log
SAVE_NPZ=$OUT_DIR/frozen_bacformer_vectors_${SLURM_JOB_ID}.npz   # cache rpoB-token + mean for reuse

# Pre-computed frozen-Bacformer mean-vectors NPZ (from bacformer_genome_vectors.py). If set
# and present, reuse it (CPU-only — also switch the directives above to icelake) instead of the GPU
# forward; leave empty to compute + cache to SAVE_NPZ.
BAC_NPZ=""

mkdir -p "$OUT_DIR"

echo "========================================================================"
echo "E1 concat probe (ESM-rpoB ⊕ Bacformer-mean → LR)"
echo "Split sheet: $SHEET"
echo "Output JSON: $OUT_JSON"
echo "Job ID:      $SLURM_JOB_ID"
echo "========================================================================"

BAC_ARG=()
if [[ -n "$BAC_NPZ" && -f "$BAC_NPZ" ]]; then
    BAC_ARG=(--bacformer-vectors "$BAC_NPZ")
    echo "Reusing frozen-Bacformer vectors: $BAC_NPZ"
else
    BAC_ARG=(--save-bacformer-vectors "$SAVE_NPZ")
    echo "Computing frozen-Bacformer mean on GPU; caching to $SAVE_NPZ"
fi

"$PY" -m bacpredict.engine.segment_amr_lr.concat.concatenate_bacformer_genome_esm_protein_emb \
    --ast-sheet-path "$SHEET" \
    --parquet-dir "$PARQUET_DIR" \
    --esm-store-dir "$ESM_STORE_DIR" \
    --output-json "$OUT_JSON" \
    --qc-log "$QC_LOG" \
    --drug rifampin \
    --device cuda:0 \
    --pool-workers "${SLURM_CPUS_PER_TASK:-8}" \
    "${BAC_ARG[@]}"

echo "E1 concat probe finished — JSON at $OUT_JSON"
