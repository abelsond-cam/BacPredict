#!/bin/bash
# Phase 0 surprise diagnostic (experiment 4) — masked-vs-unmasked proxy + distributions.
#
#   0A  windowed masked-vs-unmasked proxy: at a resistant isolate's rpoB hotspot
#       ±W residues, compute masked (ablation) AND unmasked (cheap) surprise; show
#       the masked trough is only at the SNP, correlate masked vs unmasked, report
#       the SNP's z-score/rank. A WT isolate's same window is the negative control.
#   0B  full-gene + 2-neighbour unmasked distributions: pick the per-protein
#       outlier statistic (max / top-k / skew) for the anomaly channel.
#
# Read-only — no training, no embedding store; runs straight from the protein
# parquets + the pinned ESM-C MLM. GPU because it forwards ESM-C as a masked LM
# (0A: ~2W+1 masked forwards/isolate). Cheap (minutes) but the model load > 128 MB
# rules out the login node, so it goes through SLURM. The module preamble
# (purge -> cuda/12.4 -> cudnn) is the lesson from the 6-second crash.
#
# Usage:  sbatch src/snp_embeddings/scripts/llr_distribution_probe.sh [WINDOW]
#         WINDOW defaults to 25; pass 100 for the stronger proxy test.
#
#SBATCH --job-name=llr_dist_probe
#SBATCH --output=llr_dist_probe_%j.out
#SBATCH --error=llr_dist_probe_%j.err
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append

cd /home/dca36/workspace/BacPredict

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4

export PYTHONUNBUFFERED=1

WINDOW=${1:-25}

# --- Data paths (TB AST cohort; the deployed model's canonical split) ----------
RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
SHEET=$RDS/binary_ast_with_split.csv
PARQUET_DIR=$RDS/tb_protein_sequences
OUT_DIR=$RDS/snp_embeddings/llr_distribution_probe
OUT_JSON=$OUT_DIR/llr_distribution_probe_${SLURM_JOB_ID:-local}_w${WINDOW}.json
QC_LOG=$OUT_DIR/rpob_copy_qc_${SLURM_JOB_ID:-local}.log

mkdir -p "$OUT_DIR"

echo "========================================================================"
echo "Phase 0 surprise diagnostic (0A proxy + 0B distributions)"
echo "Split sheet: $SHEET"
echo "Parquets:    $PARQUET_DIR"
echo "Window:      +/- $WINDOW"
echo "Output JSON: $OUT_JSON"
echo "Plots dir:   $OUT_DIR"
echo "Job ID:      $SLURM_JOB_ID"
echo "========================================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python src/snp_embeddings/llr_distribution_probe.py \
    --ast-sheet-path "$SHEET" \
    --parquet-dir "$PARQUET_DIR" \
    --output-json "$OUT_JSON" \
    --output-dir "$OUT_DIR" \
    --qc-log "$QC_LOG" \
    --drug rifampin \
    --device cuda:0 \
    --window "$WINDOW" \
    --n-resistant 3 \
    --n-wt 1 \
    --pool-size 300

echo "Phase 0 surprise diagnostic finished — JSON + plots in $OUT_DIR"
