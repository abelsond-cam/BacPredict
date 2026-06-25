#!/bin/bash
# Kp FROZEN Bacformer gene cache (GPU) — the mode="frozen" counterpart of cache_ft_bacformer_gene_embeddings.sh.
# One base-backbone forward per eval genome saves the per-gene *frozen* contextualised tokens for the same
# top-N (AUROC>0.6) genes of the ESM screen, so per_gene_esm_vs_ft_lr can add a frozen_lr_auroc column and
# Plot #1 can show ESM -> frozen -> fine-tuned for the non-AMR lineage genes too (not just AMR genes).
#
# Usage:  sbatch src/kleb_ast/scripts/cache_frozen_bacformer_gene_embeddings.sh
#
#SBATCH --job-name=kleb_frozen_gene_cache
#SBATCH --output=kleb_frozen_gene_cache_%A_%a.out
#SBATCH --error=kleb_frozen_gene_cache_%A_%a.err
#SBATCH --array=0-21
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --open-mode=append
# Same cost profile as the FT gene cache (~0.5 s/genome on GPU, eval-only); no checkpoint (base backbone).

cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
       gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
       levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2; exit 1; fi

D=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed
RANK=$D/train_kleb_ast/snp_embeddings/per_gene_lr_ranking_imputed/$DRUG/per_gene_lr_${DRUG}.csv
OUT=$D/train_kleb_ast/snp_embeddings/frozen_bacformer_cache/$DRUG
mkdir -p "$OUT"
if [[ ! -f "$RANK" ]]; then echo "ERROR: ranking CSV missing: $RANK" >&2; exit 1; fi

echo "=== Kp FROZEN Bacformer gene cache — drug=$DRUG (task $SLURM_ARRAY_TASK_ID) ==="
echo "rank=$RANK"; echo "out=$OUT"

uv run python src/kleb_ast/cache_ft_bacformer_gene_embeddings.py \
    --ast-sheet-path "$D/train_kleb_ast/binary_ast_with_split.csv" \
    --drug "$DRUG" \
    --parquet-dir "$D/klebsiella_protein_sequences" \
    --esm-store-dir "$D/klebsiella_esm_embeddings" \
    --ranking-csv "$RANK" \
    --out-dir "$OUT" \
    --mode frozen --auroc-threshold 0.6 --top-n 50 --device cuda:0 --eval-only

echo "Kp FROZEN Bacformer gene cache ($DRUG) finished — $OUT"
