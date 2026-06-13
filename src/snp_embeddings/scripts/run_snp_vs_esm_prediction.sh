#!/bin/bash
# Stage 1.1 — rpoB / rifampicin phenotype-ceiling ladder.
#
# Default phase (this script): predictors 1 (one-hot RRDR) + 3 (frozen pooled
# ESM-C rpoB vector) only — the head-line AUROC(1) - AUROC(3). Pure CPU: it
# aligns ~30k rpoB sequences and stat+loads ~30k .pt embedding files, so it runs
# as a CPU sbatch job, NOT on the login node (same reasoning as the prepare/split
# job — it crawls the embedding store).
#
# Predictor 2 (masked-marginal LLR) needs ESM-C forward passes; run that as the
# GPU variant below (drop --skip-masked-marginal, add --device cuda:0, switch to
# the ampere block). Read predictors 1+3 first; only spend the GPU once the
# head-line gap is in hand.
#
# Usage:  sbatch src/snp_embeddings/scripts/run_ceiling_ladder.sh
#
#SBATCH --job-name=snp_ceiling_ladder
#SBATCH --output=snp_ceiling_ladder_%j.out
#SBATCH --error=snp_ceiling_ladder_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
# Timing (measured 2026-06-12 smoke): ~0.3 s per .pt read, sequential loop over
# ~38k labelled samples → ~4-5 h. 12 h budget per the never-under-call rule.
# (If this ever needs to be faster, predictor 3's .pt reads are embarrassingly
# parallel — a multiprocessing Pool over the 32 cores would cut it to ~10 min.)
#SBATCH --open-mode=append

cd /home/dca36/workspace/BacPredict

export PYTHONUNBUFFERED=1

# --- Data paths (TB AST cohort; verified on HPC 2026-06-12) ----------------
# 38,248 parquets + 38,248 esm .pt; binary_ast.csv has 38,758 non-null rifampin
# labels (26,147 S / 12,595 R; 16 ambiguous 0.5 dropped in code). Sample-ID
# column is 'phenotype-BioSample_ID' (SAMEA... = parquet stems).
RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
AST_CSV=$RDS/binary_ast.csv
PARQUET_DIR=$RDS/tb_protein_sequences
ESM_STORE_DIR=$RDS/tb_esm_embeddings
OUT_JSON=$RDS/snp_embeddings/ceiling_ladder_${SLURM_JOB_ID}.json
SAMPLE_COL="phenotype-BioSample_ID"

echo "========================================================================"
echo "Stage 1.1 ceiling ladder (predictors 1 + 3; masked-marginal skipped)"
echo "AST CSV:     $AST_CSV"
echo "Parquets:    $PARQUET_DIR"
echo "ESM store:   $ESM_STORE_DIR"
echo "Output JSON: $OUT_JSON"
echo "Job ID:      $SLURM_JOB_ID"
echo "========================================================================"

uv run python src/snp_embeddings/ceiling_ladder.py \
    --ast-csv "$AST_CSV" \
    --parquet-dir "$PARQUET_DIR" \
    --esm-store-dir "$ESM_STORE_DIR" \
    --output-json "$OUT_JSON" \
    --sample-column "$SAMPLE_COL" \
    --label-column rifampin \
    --skip-masked-marginal

echo "Ceiling ladder finished — JSON at $OUT_JSON"

# --- GPU variant (full ladder incl. masked-marginal predictor 2) ----------
# Switch the directives to:
#   #SBATCH --partition=ampere
#   #SBATCH --account=FLOTO-SL2-GPU
#   #SBATCH --gres=gpu:1
#   #SBATCH --cpus-per-task=8
#   #SBATCH --mem=128G
#   #SBATCH --time=08:00:00
# and run without --skip-masked-marginal, with --device cuda:0:
#
#   module load cuda/12.4 cudnn/8.9_cuda-12.4
#   uv run python src/snp_embeddings/ceiling_ladder.py \
#       --ast-csv "$AST_CSV" --parquet-dir "$PARQUET_DIR" \
#       --esm-store-dir "$ESM_STORE_DIR" --output-json "$OUT_JSON" \
#       --sample-column "$SAMPLE_COL" --label-column rifampin \
#       --device cuda:0 --masked-marginal-codons panel
