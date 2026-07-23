#!/bin/bash
# Kp FROZEN Bacformer gene cache (GPU) — the mode="frozen" counterpart of cache_ft_bacformer_gene_embeddings.sh.
# One base-backbone forward per eval genome saves the per-gene *frozen* contextualised tokens for the same
# top-N (AUROC>0.6) genes of the ESM screen, so per_gene_esm_vs_ft_lr can add a frozen_lr_auroc column and
# Plot #1 can show ESM -> frozen -> fine-tuned for the non-AMR lineage genes too (not just AMR genes).
#
# Usage:  sbatch src/bacpredict/apps/kleb/scripts/cache_frozen_bacformer_gene_embeddings.sh
#
#SBATCH --job-name=kleb_frozen_gene_cache
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --array=0-21
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → a project-tier logs dir, and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.
# Same cost profile as the FT gene cache (~0.5 s/genome on GPU); --scope eval = the k-fold holdout only.
# NOTE: frozen mode has no checkpoint, so the holdout falls back to the CSV single-split (logged warning) —
# aligning the frozen cache to the deployed FT k-fold holdout is a fan-out follow-up (pass the FT run dir).

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
       gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
       levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2; exit 1; fi

RANK=$D/processed/train_kleb_ast/pangena_predict/per_gene_lr_ranking_imputed/$DRUG/per_gene_lr_${DRUG}.csv
OUT=$D/processed/train_kleb_ast/pangena_predict/frozen_bacformer_cache/$DRUG
mkdir -p "$OUT"
if [[ ! -f "$RANK" ]]; then echo "ERROR: ranking CSV missing: $RANK" >&2; exit 1; fi

echo "=== Kp FROZEN Bacformer gene cache — drug=$DRUG (task $SLURM_ARRAY_TASK_ID) ==="
echo "rank=$RANK"; echo "out=$OUT"

"$PY" -m bacpredict.engine.concat.cache_bacformer_gene_embeddings \
    --ast-sheet-path "$D/processed/train_kleb_ast/binary_ast_with_split.csv" \
    --drug "$DRUG" \
    --parquet-dir "$D/processed/train_kleb_ast/protein_sequences" \
    --esm-store-dir "$D/processed/train_kleb_ast/esm" \
    --ranking-csv "$RANK" \
    --out-dir "$OUT" \
    --mode frozen --auroc-threshold 0.6 --top-n 50 --device cuda:0 --scope eval

echo "Kp FROZEN Bacformer gene cache ($DRUG) finished — $OUT"
