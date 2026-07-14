#!/bin/bash
# Plot #5 (GPU): cache the FROZEN Bacformer genome-mean + per-AMR-gene tokens at the reliable CARD
# flat-indices, one drug per array task (bacpredict.apps.kleb.cache_frozen_amr_proteins). Mirrors the Phase-2b FT
# cache but through the base backbone (no checkpoint) -> frozen_amr_cache/<drug>/. Eval-holdout only.
#
# Smoke one drug first (trimethoprim-sulfamethoxazole = index 11):
#     sbatch --array=11 src/bacpredict/apps/kleb/scripts/cache_frozen_amr_proteins.sh
# Full panel (after the smoke + a cost check):
#     sbatch --array=0-21 src/bacpredict/apps/kleb/scripts/cache_frozen_amr_proteins.sh
#
#SBATCH --job-name=kleb_frozen_amr_cache
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → a project-tier logs dir, and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.

set -euo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
       gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
       levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2; exit 1; fi
GRAIN=${GRAIN:-family}

OUT=$D/processed/train_kleb_ast/pangena_predict/frozen_amr_cache
mkdir -p "$OUT"
echo "=== Kp FROZEN AMR-token cache — drug=$DRUG grain=$GRAIN (task $SLURM_ARRAY_TASK_ID) ==="

"$PY" -m bacpredict.apps.kleb.cache_frozen_amr_proteins \
    --drug "$DRUG" \
    --ast-sheet-path "$D/processed/train_kleb_ast/binary_ast_with_split.csv" \
    --parquet-dir "$D/processed/train_kleb_ast/protein_sequences" \
    --esm-store-dir "$D/processed/train_kleb_ast/esm" \
    --sidecar-dir "$D/processed/train_kleb_ast/amr_annotation" \
    --out-dir "$OUT" --grain "$GRAIN" --device cuda:0

echo "Kp frozen AMR-token cache ($DRUG) finished — $OUT/$DRUG"
