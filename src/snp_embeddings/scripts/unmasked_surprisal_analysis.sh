#!/bin/bash
# Experiment-4 unmasked-surprisal scan — STEP 3: scaled analysis (CPU, no GPU).
#
# Reads the per-shard stats parquets + ACF NPZs from STEP 2 (millions of protein rows → too
# big for the login node, so a CPU sbatch), and writes the per-protein statistic histogram
# grid, the genome-wide spatial-autocorrelation figure (with the rpoB-only 0A ACF overlaid),
# the stat correlation heatmap, and a results JSON.
#
# Usage:  sbatch src/snp_embeddings/scripts/unmasked_surprisal_analysis.sh   # after the array
#
#SBATCH --job-name=usurp_analysis
#SBATCH --output=usurp_analysis_%j.out
#SBATCH --error=usurp_analysis_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
SCAN_DIR=$RDS/snp_embeddings/unmasked_surprisal_scan
ANALYSIS_DIR=$SCAN_DIR/analysis
POINTS_NPZ=$RDS/snp_embeddings/llr_distribution_probe/llr_phase0_0a_30535846_0a_points.npz
mkdir -p "$ANALYSIS_DIR"

uv run python src/snp_embeddings/surprisal_analysis.py \
    --scan-stats-glob "$SCAN_DIR/scan_stats_shard*.parquet" \
    --scan-acf-glob "$SCAN_DIR/scan_acf_shard*.npz" \
    --points-npz "$POINTS_NPZ" \
    --output-dir "$ANALYSIS_DIR"

echo "Scaled analysis done — figures + JSON in $ANALYSIS_DIR"
