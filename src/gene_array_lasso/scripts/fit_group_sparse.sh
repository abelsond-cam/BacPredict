#!/bin/bash
# Step D — fit the group-sparse models (groupyr SGL) on a Step C array for one drug.
#
# groupyr requires DENSE X, so each split is densified — himem job. Output -> fits/<tag>/ on RDS
# (results.json + selected_genes.csv), tag = array dir name.
#
# Usage:  sbatch src/gene_array_lasso/scripts/fit_group_sparse.sh <drug> <array_tag> [l1_ratios] [alphas]
#   e.g.  sbatch src/gene_array_lasso/scripts/fit_group_sparse.sh tetracycline tetracycline_p5
#         sbatch src/gene_array_lasso/scripts/fit_group_sparse.sh colistin colistin_p1 "0.5 0.9" "0.3 0.1 0.03"
#
#SBATCH --job-name=gal_fit_sgl
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=420G
#SBATCH --time=24:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --output=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/gene_array_lasso/logs/gal_fit_sgl_%j.out
#SBATCH --error=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/gene_array_lasso/logs/gal_fit_sgl_%j.err
set -euo pipefail

DRUG="${1:?usage: fit_group_sparse.sh <drug> <array_tag> [l1_ratios] [alphas]}"
TAG="${2:?give array_tag, e.g. tetracycline_p5}"
L1="${3:-0.5 0.9}"
ALPHAS="${4:-0.3 0.1 0.03 0.01}"

cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
GAL=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/gene_array_lasso

echo "=== fit SGL: drug=$DRUG tag=$TAG l1=[$L1] alphas=[$ALPHAS] ==="
uv run python src/gene_array_lasso/fit_group_sparse.py \
  --drug "$DRUG" \
  --array-dir "$GAL/gene_arrays/$TAG" \
  --out-dir "$GAL/fits/$TAG" \
  --l1-ratios $L1 \
  --alphas $ALPHAS
echo "Done -> $GAL/fits/$TAG"
