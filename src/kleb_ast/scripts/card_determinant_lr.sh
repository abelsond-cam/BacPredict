#!/bin/bash
# Plot #2 data: per-CARD-gene one-hot LR + __ALL_CARD__ ceiling, all 22 drugs × both grains in one job
# (kleb_ast.card_determinant_lr). Reads the combined amr_calls_all.parquet store (build it first), so the
# I/O is seconds; the cost is the k-fold × m-seed LRs. Chain it after the store build:
#
#     sbatch src/kleb_ast/scripts/build_amr_calls_store.sh                 # -> jid
#     sbatch --dependency=afterok:<jid> src/kleb_ast/scripts/card_determinant_lr.sh
#
#SBATCH --job-name=kleb_card_determinant_lr
#SBATCH --output=kleb_card_determinant_lr_%j.out
#SBATCH --error=kleb_card_determinant_lr_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --open-mode=append

set -euo pipefail
cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
       gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
       levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)

echo "=== CARD determinant LR (one-hot ceiling) — ${#DRUGS[@]} drugs × {family,allele} ==="
uv run python -m kleb_ast.card_determinant_lr --drugs "${DRUGS[@]}" --grains family allele
echo "done -> docs/visualisations/amr_per_abx/kp_<drug>/card_determinant_lr_<drug>_<grain>.csv"
