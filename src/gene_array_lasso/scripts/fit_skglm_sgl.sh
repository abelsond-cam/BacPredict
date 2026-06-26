#!/bin/bash
# Step C/F — fit the skglm sparse-group lasso on a gene-embedding array (sparse CSR, no densify).
#
# himem because the array CSR is loaded into RAM (~45 GB at >5%, ~50 GB at >1%) plus the train-subset copy.
# skglm operates on the CSR directly (does NOT densify); the working set is the small active gene set.
#
# Usage:
#   sbatch src/gene_array_lasso/scripts/fit_skglm_sgl.sh <drug> <array_tag> fit   [extra fit args]
#   sbatch src/gene_array_lasso/scripts/fit_skglm_sgl.sh <drug> <array_tag> smoke [extra smoke args]
#   e.g. sbatch src/gene_array_lasso/scripts/fit_skglm_sgl.sh colistin colistin_p5 fit --tau 0.05
#
#SBATCH --job-name=gal_skglm
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --output=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/gene_array_lasso/logs/gal_skglm_%j.out
#SBATCH --error=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/gene_array_lasso/logs/gal_skglm_%j.err
set -euo pipefail

DRUG="${1:?usage: fit_skglm_sgl.sh <drug> <array_tag> <fit|smoke> [args]}"
TAG="${2:?give array_tag, e.g. colistin_p5}"
MODE="${3:?give fit or smoke}"
shift 3 || true

cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
GAL=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/gene_array_lasso

ARGS=(--drug "$DRUG" --array-dir "$GAL/gene_arrays/$TAG")
if [[ "$MODE" == "smoke" ]]; then
  ARGS+=(--smoke)
else
  ARGS+=(--out-dir "$GAL/fits/$TAG")
fi
echo "=== skglm $MODE: drug=$DRUG tag=$TAG args=${*:-} ==="
uv run python src/gene_array_lasso/fit_skglm_sgl.py "${ARGS[@]}" "$@"
echo "Done."
