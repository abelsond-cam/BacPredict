#!/bin/bash
# Phase 2b (GPU): cache the FT genome-mean + per-AMR-gene contextualised tokens at the RELIABLE CARD
# flat-indices, one drug per array task. One FT forward per eval genome; extracts last_hidden_state at
# each AMR protein the {Sample}_amr.parquet sidecar identifies -> ft_amr_cache/<drug>/.
#
# Eval-only (FT-unseen, honest scope, ~5x cheaper).
#
# Smoke one drug first (trimethoprim-sulfamethoxazole = index 11):
#     sbatch --array=11 src/bacpredict/apps/kleb/scripts/cache_ft_amr_proteins.sh
# Full panel (after the smoke + a cost check):
#     sbatch --array=0-21 src/bacpredict/apps/kleb/scripts/cache_ft_amr_proteins.sh
#
#SBATCH --job-name=kleb_ft_amr_cache
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
export PYTHONPATH="${BACPREDICT_REPO:-$SCRATCHDIR/worktrees/consolidate}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
       gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
       levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2; exit 1; fi

CKPT=$D/processed/train_kleb_ast/models/finetune/klebsiella_pneumoniae_${DRUG}_lr_0.00015_finetuned_fold00_seed1
OUT=$D/processed/train_kleb_ast/pangena_predict/ft_amr_cache
mkdir -p "$OUT"
if [[ ! -d "$CKPT" ]]; then echo "ERROR: FT checkpoint missing: $CKPT" >&2; exit 1; fi
echo "=== Kp FT AMR-token cache — drug=$DRUG (task $SLURM_ARRAY_TASK_ID), ckpt=$CKPT ==="

"$PY" -m bacpredict.apps.kleb.cache_ft_amr_proteins \
    --drug "$DRUG" \
    --split-table "$D/processed/train_kleb_ast/splits/${DRUG}_split.csv" \
    --parquet-dir "$D/processed/train_kleb_ast/protein_sequences" \
    --esm-store-dir "$D/processed/train_kleb_ast/esm" \
    --sidecar-dir "$D/processed/train_kleb_ast/amr_annotation" \
    --bacformer-checkpoint "$CKPT" \
    --out-dir "$OUT" --grain family --device cuda:0

echo "Kp FT AMR-token cache ($DRUG) finished — $OUT/$DRUG"
