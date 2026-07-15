#!/bin/bash
# Kp: reliable per-gene ESM-vs-frozen-vs-FT LR + FT-mean ⊕ best-gene concat, on the CARD reliable
# carriers (reliable_ft_concat over the token caches from cache_amr_tokens_kleb.sh). CPU, no forward
# pass. Writes reliable_esm_vs_ft_per_gene_<drug>.csv + reliable_concat_<drug>.csv (the blue concat
# number for the summary panel). Run after the token cache lands.
#
#     DRUG=ciprofloxacin sbatch --export=ALL,DRUG=ciprofloxacin src/bacpredict/apps/kleb/scripts/reliable_ft_concat_kleb.sh
#
#SBATCH --job-name=kleb_reliable_ft_concat
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=icelake-himem --account=FLOTO-PROJECT-K-SL2-CPU.

set -euo pipefail

: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

DRUG=${DRUG:-ciprofloxacin}
RDS="$D/processed/train_kleb_ast"

echo "=== Kp reliable FT concat — drug=$DRUG ==="
"$PY" -m bacpredict.apps.kleb.reliable_ft_concat \
    --drug "$DRUG" \
    --ft-cache-dir "$RDS/ft_amr_cache/$DRUG" \
    --frozen-cache-dir "$RDS/frozen_amr_cache/$DRUG" \
    --out-dir "$RDS/pangena_predict/reliable_concat/$DRUG" \
    --n-folds 5 --seed 1

echo "=== reliable_concat_${DRUG}.csv ==="
cat "$RDS/pangena_predict/reliable_concat/$DRUG/reliable_concat_${DRUG}.csv" 2>/dev/null
echo "reliable FT concat ($DRUG) finished."
