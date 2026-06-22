#!/bin/bash
# Phase 2b (GPU): cache the FT genome-mean + per-AMR-gene contextualised tokens at the RELIABLE CARD
# flat-indices, one drug per array task. One FT forward per eval genome; extracts last_hidden_state at
# each AMR protein the {Sample}_amr.parquet sidecar identifies -> ft_amr_cache/<drug>/.
#
# SL2-GPU (personal account; SL3-GPU never schedules). Eval-only (FT-unseen, honest scope, ~5x cheaper).
#
# Smoke one drug first (trimethoprim-sulfamethoxazole = index 11):
#     sbatch --array=11 src/kleb_ast/scripts/cache_ft_amr_proteins.sh
# Full panel (after the smoke + a cost check):
#     sbatch --array=0-21 src/kleb_ast/scripts/cache_ft_amr_proteins.sh
#
#SBATCH --job-name=kleb_ft_amr_cache
#SBATCH --output=kleb_ft_amr_cache_%A_%a.out
#SBATCH --error=kleb_ft_amr_cache_%A_%a.err
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --open-mode=append

set -euo pipefail
cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1

DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
       gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
       levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2; exit 1; fi

D=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed
CKPT=$D/train_kleb_ast/models/finetune/klebsiella_pneumoniae_${DRUG}_lr_0.00015_finetuned_fold00_seed1
OUT=$D/train_kleb_ast/snp_embeddings/ft_amr_cache
mkdir -p "$OUT"
if [[ ! -d "$CKPT" ]]; then echo "ERROR: FT checkpoint missing: $CKPT" >&2; exit 1; fi
echo "=== Kp FT AMR-token cache — drug=$DRUG (task $SLURM_ARRAY_TASK_ID), ckpt=$CKPT ==="

uv run python -m kleb_ast.cache_ft_amr_proteins \
    --drug "$DRUG" \
    --ast-sheet-path "$D/train_kleb_ast/binary_ast_with_split.csv" \
    --parquet-dir "$D/klebsiella_protein_sequences" \
    --esm-store-dir "$D/klebsiella_esm_embeddings" \
    --sidecar-dir "$D/train_kleb_ast/amr_annotation" \
    --bacformer-checkpoint "$CKPT" \
    --out-dir "$OUT" --grain family --device cuda:0

echo "Kp FT AMR-token cache ($DRUG) finished — $OUT/$DRUG"
