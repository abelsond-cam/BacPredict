#!/bin/bash
# Plot #5 (GPU): cache the FROZEN Bacformer genome-mean + per-AMR-gene tokens at the reliable CARD
# flat-indices, one drug per array task (kleb_ast.cache_frozen_amr_proteins). Mirrors the Phase-2b FT
# cache but through the base backbone (no checkpoint) -> frozen_amr_cache/<drug>/. Eval-holdout only.
#
# Smoke one drug first (trimethoprim-sulfamethoxazole = index 11):
#     sbatch --array=11 src/kleb_ast/scripts/cache_frozen_amr_proteins.sh
# Full panel (after the smoke + a cost check):
#     sbatch --array=0-21 src/kleb_ast/scripts/cache_frozen_amr_proteins.sh
#
#SBATCH --job-name=kleb_frozen_amr_cache
#SBATCH --output=kleb_frozen_amr_cache_%A_%a.out
#SBATCH --error=kleb_frozen_amr_cache_%A_%a.err
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
GRAIN=${GRAIN:-family}

D=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed
OUT=$D/train_kleb_ast/snp_embeddings/frozen_amr_cache
mkdir -p "$OUT"
echo "=== Kp FROZEN AMR-token cache — drug=$DRUG grain=$GRAIN (task $SLURM_ARRAY_TASK_ID) ==="

uv run python -m kleb_ast.cache_frozen_amr_proteins \
    --drug "$DRUG" \
    --ast-sheet-path "$D/train_kleb_ast/binary_ast_with_split.csv" \
    --parquet-dir "$D/klebsiella_protein_sequences" \
    --esm-store-dir "$D/klebsiella_esm_embeddings" \
    --sidecar-dir "$D/train_kleb_ast/amr_annotation" \
    --out-dir "$OUT" --grain "$GRAIN" --device cuda:0

echo "Kp frozen AMR-token cache ($DRUG) finished — $OUT/$DRUG"
