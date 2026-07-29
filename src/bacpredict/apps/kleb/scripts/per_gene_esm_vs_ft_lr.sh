#!/bin/bash
# Kp per-gene ESM-LR vs Bacformer-FT-LR — CPU array, one drug per task.
#
# For each drug's top genes (from the FT cache manifest), compute ESM-LR and FT-LR AUROC over the eval
# holdout (same samples, same zero-imputed out-of-fold k-fold) -> esm_vs_ft_per_gene_<drug>.csv. No forward
# pass: ESM vectors come from the store, FT vectors from the cached ft_bacformer_cache/<drug>/gene_emb/.
#
# Usage:  sbatch src/bacpredict/apps/kleb/scripts/per_gene_esm_vs_ft_lr.sh
#
#SBATCH --job-name=kleb_esm_vs_ft
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%A_%a.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%A_%a.out
#SBATCH --array=0-21
#SBATCH --partition=icelake-himem
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --open-mode=append
# CPU-only: the only I/O is the eval-holdout .pt/parquet reads for the ESM extraction (~280-940/drug);
# the FT vectors are loaded from the cache.
# CSD3/UoHPC variant (when it returns): --partition=icelake-himem --account=FLOTO-PROJECT-K-SL2-CPU,
#   logs → relative or ~/rds/hpc-work/logs/.

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction"}"
D="$BACPREDICT_DATA_ROOT"
PY="$HOME/workspace/BacPredict/.venv/bin/python"
export PYTHONPATH="${BACPREDICT_REPO:-$HOME/workspace/BacPredict}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
       gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
       levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2; exit 1; fi

FTC=$D/processed/train_kleb_ast/pangena_predict/ft_bacformer_cache/$DRUG
FRC=$D/processed/train_kleb_ast/pangena_predict/frozen_bacformer_cache/$DRUG
OUT=$D/processed/train_kleb_ast/pangena_predict/esm_vs_ft_per_gene/$DRUG
mkdir -p "$OUT"
if [[ ! -f "$FTC/top_gene_manifest_${DRUG}.csv" ]]; then echo "ERROR: FT cache manifest missing: $FTC" >&2; exit 1; fi
[[ -d "$FRC/gene_emb" ]] || echo "WARN: frozen gene cache missing ($FRC) — frozen_lr_auroc skipped"

echo "=== Kp ESM-vs-frozen-vs-FT per-gene LR — drug=$DRUG (task $SLURM_ARRAY_TASK_ID) ==="

"$PY" -m bacpredict.engine.gene_lr.segment_vs_ft \
    --split-table "$D/processed/train_kleb_ast/splits/${DRUG}_split.csv" \
    --drug "$DRUG" \
    --parquet-dir "$D/processed/train_kleb_ast/protein_sequences" \
    --esm-store-dir "$D/processed/train_kleb_ast/esm" \
    --ft-cache-dir "$FTC" \
    --frozen-cache-dir "$FRC" \
    --out-dir "$OUT" --n-folds 5 --seed 1

echo "Kp ESM-vs-FT per-gene LR ($DRUG) finished — $OUT"
