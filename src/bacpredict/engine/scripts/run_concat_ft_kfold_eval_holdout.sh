#!/bin/bash
# Honest FT-mean concat k-fold — error bars for A.1.i WITHOUT re-fine-tuning (CPU, no GPU).
#
# The A.1.i headline (run_concat_ft_mean.sh, job 30673823) is k=1/m=1: ESM-rpoB ⊕ the *fine-tuned*
# Bacformer genome-mean → LR on the canonical evaluate holdout. To put error bars on it we CANNOT
# k-fold the cached FT NPZ over the whole cohort — the backbone was fine-tuned on the original TRAIN
# labels, so re-splitting would drop FT-training genomes into the new evaluate fold (representation
# leakage → optimistic). The honest, GPU-free alternative: restrict the fold universe to the canonical
# evaluate holdout (~7k genomes the FT backbone was held out from). On those FT-unseen genomes the
# fine-tuned mean is once again a label-blind feature, so re-splitting them is a valid k-fold.
#   --kfold 5 --seeds 1 2 3 --kfold-on-eval-holdout  → mean ± sd + paired AUROC deltas, all honest.
# Re-uses the FT vectors NPZ cached by 30673823 (no GPU, no re-tuning).
#
# Usage:  sbatch src/bacpredict/engine/scripts/run_concat_ft_kfold_eval_holdout.sh
#
#SBATCH --job-name=concat_ft_kfold_evalhold
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=icelake-himem --account=FLOTO-SL2-CPU,
#   logs → concat_ft_kfold_evalhold_%j.out/.err (repo-relative).
# CPU-only (cached FT mean): genotype ~38k parquets + mmap ESM-gene reads (parallel over the cores) +
# 3 frames × (1 single-split + 15 k-fold) logistic-regression fits over the ~7k eval-holdout universe.
# icelake-himem, 24 h budget — never under-call walltime (charged on time used, not requested).

set -uo pipefail
# Data root + env — cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1

RDS=$D/processed/train_tb_ast
SHEET=$RDS/binary_ast_with_split.csv
PARQUET_DIR=$RDS/protein_sequences
ESM_STORE_DIR=$RDS/esm
OUT_DIR=$RDS/pangena_predict/concat_ft_mean
OUT_JSON=$OUT_DIR/concat_ft_kfold_evalhold_${SLURM_JOB_ID}.json
QC_LOG=$OUT_DIR/gene_presence_qc_${SLURM_JOB_ID}.log
FT_NPZ=$OUT_DIR/finetuned_bacformer_vectors_30673823.npz   # cached FT rpoB-token + genome-mean (job 30673823)

if [[ ! -f "$FT_NPZ" ]]; then
    echo "ERROR: cached fine-tuned-mean NPZ not found at $FT_NPZ" >&2
    echo "       (it is written by the A.1.i GPU job 30673823 — wait for that to finish.)" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "========================================================================"
echo "Honest FT-mean concat k-fold — error bars over the canonical evaluate holdout"
echo "Cached FT NPZ: $FT_NPZ"
echo "Output JSON:   $OUT_JSON"
echo "Job ID:        $SLURM_JOB_ID"
echo "========================================================================"

"$PY" -m bacpredict.engine.segment_amr_lr.concat.concatenate_bacformer_genome_esm_protein_emb \
    --ast-sheet-path "$SHEET" \
    --parquet-dir "$PARQUET_DIR" \
    --esm-store-dir "$ESM_STORE_DIR" \
    --output-json "$OUT_JSON" \
    --qc-log "$QC_LOG" \
    --drug rifampin \
    --device cpu \
    --bacformer-vectors "$FT_NPZ" \
    --mean-is-finetuned \
    --kfold 5 --seeds 1 2 3 --kfold-on-eval-holdout \
    --pool-workers "${SLURM_CPUS_PER_TASK:-32}"

echo "Honest FT concat k-fold finished — JSON at $OUT_JSON"
echo
echo "These are honest error bars: the universe is the canonical evaluate holdout, which the FT backbone"
echo "was held out from, so the fine-tuned mean is label-blind there. finetuned_mean_leakage_warning=false,"
echo "restricted_to_eval_holdout=true. Directly error-bars the A.1.i k=1/m=1 headline (job 30673823)."
