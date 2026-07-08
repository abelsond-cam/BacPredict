#!/bin/bash
# Phase 2a: per-gene ESM-LR on reliable Kleborate/CARD AMR labels, one drug per array task.
#
# For each drug, over the evaluate holdout, per AMR gene-family: ESM-C LR on the reliable carrier set
# (every CARD-identified carrier) vs the Bakta-named subset -> reliable_per_gene_esm_lr_<drug>.csv. Reads
# the {Sample}_amr.parquet sidecars (Phase 1) + the ESM store; CPU only, no forward pass.
#
# Smoke one drug first (trimethoprim-sulfamethoxazole = index 11):
#     sbatch --array=11 src/kleb_ast/scripts/per_gene_lr_from_annotation.sh
# Full panel:
#     sbatch --array=0-22 src/kleb_ast/scripts/per_gene_lr_from_annotation.sh
#
#SBATCH --job-name=kleb_reliable_per_gene_lr
#SBATCH --output=kleb_reliable_per_gene_lr_%A_%a.out
#SBATCH --error=kleb_reliable_per_gene_lr_%A_%a.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --open-mode=append

set -euo pipefail
cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
       gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
       levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2; exit 1; fi

D=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed
OUT=$D/train_kleb_ast/pangena_predict/reliable_per_gene_esm_lr/$DRUG
mkdir -p "$OUT"
echo "=== Kp reliable per-gene ESM-LR — drug=$DRUG (task $SLURM_ARRAY_TASK_ID) ==="

uv run python -m kleb_ast.per_gene_lr_from_annotation \
    --drug "$DRUG" \
    --ast-sheet-path "$D/train_kleb_ast/binary_ast_with_split.csv" \
    --sidecar-dir "$D/train_kleb_ast/amr_annotation" \
    --esm-store-dir "$D/klebsiella_esm_embeddings" \
    --parquet-dir "$D/klebsiella_protein_sequences" \
    --out-dir "$OUT" --grain family --n-folds 5 --seed 1

echo "Kp reliable per-gene ESM-LR ($DRUG) finished — $OUT"
