#!/bin/bash
# Plot #2 data: per-CARD-gene one-hot LR + __ALL_CARD__ ceiling, all 22 drugs × both grains in one job
# (bacpredict.apps.kleb.card_determinant_lr). Reads the combined amr_calls_all.parquet store (build it first), so the
# I/O is seconds; the cost is the k-fold × m-seed LRs. Chain it after the store build:
#
#     sbatch src/bacpredict/apps/kleb/scripts/build_amr_calls_store.sh                 # -> jid
#     sbatch --dependency=afterok:<jid> src/bacpredict/apps/kleb/scripts/card_determinant_lr.sh
#
#SBATCH --job-name=kleb_card_determinant_lr
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --partition=icelake-himem
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=icelake-himem --account=FLOTO-PROJECT-K-SL2-CPU,
#   logs → a project-tier logs dir (e.g. ~/rds/hpc-work/logs/%x-%j.out).

set -euo pipefail

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

echo "=== CARD determinant LR (one-hot ceiling) — ${#DRUGS[@]} drugs × {family,allele} ==="
"$PY" -m bacpredict.apps.kleb.card_determinant_lr --drugs "${DRUGS[@]}" --grains family allele
echo "done -> visualisations/kp/<drug>/card_determinant_lr_<drug>_<grain>.csv"
