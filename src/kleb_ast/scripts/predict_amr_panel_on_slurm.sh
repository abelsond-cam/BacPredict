#!/bin/bash
#SBATCH --job-name=kleb_amr_predict
#SBATCH --output=kleb_amr_predict_%A_%a.out
#SBATCH --error=kleb_amr_predict_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-21
#SBATCH --open-mode=append
#
# Score every kpsc_final_list sample with the fine-tuned Bacformer head for one
# drug per array task. Per-drug parquets land at:
#
#   $BASE/predictions_for_metadata/<drug>.parquet
#
# The downstream BacHGT merge step joins these into the v2 metadata table.
#
# Submit:   sbatch src/kleb_ast/scripts/predict_amr_panel_on_slurm.sh
# Re-run one drug only (e.g. cipro = index 3):
#           sbatch --array=3 src/kleb_ast/scripts/predict_amr_panel_on_slurm.sh

cd /home/dca36/workspace/BacPredict
git pull --ff-only || true
module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4
export PYTHONUNBUFFERED=1

BASE=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_kleb_ast
FT=$BASE/models/finetune
EMB=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_esm_embeddings
METADATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_v2_all_samples_and_columns.tsv
OUT_DIR=$BASE/predictions_for_metadata
mkdir -p "$OUT_DIR"

# Panel order matches src/kleb_ast/scripts/eval_panel_on_slurm.sh.
DRUGS=(
  gentamicin
  ceftazidime
  meropenem
  ciprofloxacin
  trimethoprim-sulfamethoxazole
  amikacin
  ceftriaxone
  piperacillin-tazobactam
  cefoxitin
  aztreonam
  cefazolin
  tobramycin
  cefepime
  imipenem
  levofloxacin
  cefotaxime
  cefuroxime
  ampicillin-sulbactam
  ertapenem
  tetracycline
  azithromycin
  colistin
)

d=${DRUGS[$SLURM_ARRAY_TASK_ID]}
CK=$FT/klebsiella_pneumoniae_${d}_lr_0.00015_finetuned_fold00_seed1
OUT=$OUT_DIR/${d}.parquet

echo "=== array task $SLURM_ARRAY_TASK_ID / drug $d ==="
echo "checkpoint: $CK"
echo "output:     $OUT"

if [ ! -d "$CK" ]; then
  echo "NO CHECKPOINT (skipping): $d"
  exit 0
fi

# Idempotent: skip drugs whose parquet already exists (set FORCE_RECOMPUTE=1 to override).
if [ -f "$OUT" ] && [ "${FORCE_RECOMPUTE:-0}" != "1" ]; then
  echo "ALREADY DONE (skip; set FORCE_RECOMPUTE=1 to re-run): $d"
  exit 0
fi

# num-workers=0 (single-process loading) — at deployment scale (~73k samples per drug)
# the file_system sharing strategy + workers>0 exhausts mmap quotas on ampere nodes.
# Fail-fast: SLURM sees a non-zero exit if the Python crashes, so failures aren't silent.
if ! uv run python src/kleb_ast/predict_amr_for_metadata.py \
  --drug "$d" \
  --checkpoint "$CK" \
  --metadata-tsv "$METADATA" \
  --embeddings-dir "$EMB" \
  --out "$OUT" \
  --batch-size 1 \
  --num-workers 0; then
  echo "PREDICT_FAILED: $d (see .err log)"
  exit 1
fi

echo "PREDICT_DONE: $d"
