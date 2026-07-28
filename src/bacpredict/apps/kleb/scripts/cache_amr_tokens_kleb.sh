#!/bin/bash
# Kp: cache the drug's FINE-TUNED + FROZEN Bacformer genome-mean and per-CARD-gene contextualised
# tokens — one forward pass per eval genome each (bacformer_token_cache via the thin CARD CLIs
# cache_ft_amr_proteins / cache_frozen_amr_proteins). GPU. The CPU concat (reliable_ft_concat) then
# reads these to compute the FT-mean ⊕ best-gene concat + the per-gene ESM-vs-frozen-vs-FT LR.
#
# Needs the CARD sidecars (amr_annotation/) — run after annotate_amr_parquet.sh + build_amr_calls_store.
# DRUG selects the antibiotic (default ciprofloxacin, the canonical Kp test case); the FT checkpoint is
# auto-discovered as the highest checkpoint-* under the drug's finetuned_fold00_seed1 dir.
#
#     DRUG=ciprofloxacin sbatch --export=ALL,DRUG=ciprofloxacin src/bacpredict/apps/kleb/scripts/cache_amr_tokens_kleb.sh
#
#SBATCH --job-name=kleb_amr_token_cache
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-PROJECT-K-SL2-GPU.

set -euo pipefail

: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
REPO="${REPO:-$SCRATCHDIR/worktrees/consolidate}"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

DRUG=${DRUG:-ciprofloxacin}
RDS="$D/processed/train_kleb_ast"
SPLITS="$RDS/splits"                             # per-drug <drug>_split.csv (generate_kfold_splits)

# Auto-discover the drug's deployed FT checkpoint (highest checkpoint-N under finetuned_fold00_seed1).
FT_DIR=$(ls -d "$RDS"/models/finetune/klebsiella_pneumoniae_${DRUG}_*_finetuned_fold00_seed1 2>/dev/null | head -1)
CKPT=$(ls -d "$FT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [[ -z "${CKPT:-}" ]]; then echo "ERROR: no FT checkpoint for $DRUG under $FT_DIR" >&2; exit 1; fi

echo "=== Kp AMR token cache — drug=$DRUG ==="
echo "FT checkpoint: $CKPT"
echo "Sidecar dir:   $RDS/amr_annotation"

# 1) FINE-TUNED backbone: genome-mean + per-gene FT tokens -> ft_amr_cache/<drug>/
"$PY" -m bacpredict.apps.kleb.cache_ft_amr_proteins \
    --split-table "$SPLITS/${DRUG}_split.csv" \
    --drug "$DRUG" \
    --bacformer-checkpoint "$CKPT" \
    --out-dir "$RDS/ft_amr_cache" \
    --device cuda:0

# 2) FROZEN backbone: genome-mean + per-gene frozen tokens -> frozen_amr_cache/<drug>/
"$PY" -m bacpredict.apps.kleb.cache_frozen_amr_proteins \
    --split-table "$SPLITS/${DRUG}_split.csv" \
    --drug "$DRUG" \
    --out-dir "$RDS/frozen_amr_cache" \
    --device cuda:0

echo "token cache ($DRUG) finished -> $RDS/{ft,frozen}_amr_cache/$DRUG/"
