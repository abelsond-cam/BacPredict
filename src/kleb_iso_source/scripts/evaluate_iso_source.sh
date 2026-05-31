#!/bin/bash
# Score a fine-tuned iso-source checkpoint on its held-out 'evaluate' split.
# Wraps src/tl/train/evaluate.py (task-agnostic). Inference on a few-thousand
# Bacformer-large genomes needs a GPU, so this is a short ampere sbatch, not login CPU.
# Edit CHECKPOINT / SHEET / OUT_DIR inline per cohort.

#SBATCH --job-name=eval_iso_blood_faeces
#SBATCH --output=eval_iso_blood_faeces_%j.out
#SBATCH --error=eval_iso_blood_faeces_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --open-mode=append

cd /home/dca36/workspace/BacPredict

BASE=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_iso_source/blood_faeces
# Args: $1 cohort, $2 flavor, $3 checkpoint subdir (optional).
#   cohort       ∈ {all_samples, sampled_country_2_1_stratified, sampled_country_2_1_all}
#   flavor       ∈ {kpsc_human (default — KPSC+Sublineage+human), mixed_species (pre-fix reference)}
#   ckpt_subdir  optional, e.g. checkpoint-31000 — pins evaluate.py to a specific
#                checkpoint. If empty, evaluate.py picks the latest inside models/.
# Each flavor dir holds binary_blood_vs_faeces_with_split.csv + models/ (the checkpoint).
# Note: all_samples's ~4-5k-row evaluate split needs >1h — bump --time when scoring it.
SUB="${1:-sampled_country_2_1_all}"
FLAVOR="${2:-kpsc_human}"
CKPT_SUBDIR="${3:-}"
DIR="${BASE}/${SUB}/${FLAVOR}"
CHECKPOINT="${DIR}/models${CKPT_SUBDIR:+/$CKPT_SUBDIR}"
SHEET="${DIR}/binary_blood_vs_faeces_with_split.csv"
OUT_DIR="${DIR}/models"

EMB=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_esm_embeddings

module purge
module load cuda/12.4 || echo "WARN: cuda module not found (torch uses bundled CUDA)"
module load cudnn/8.9_cuda-12.4 || true
export PYTHONUNBUFFERED=1

echo "=== Evaluating $CHECKPOINT on evaluate split of $SHEET ==="
uv run python src/tl/train/evaluate.py \
  --checkpoint "$CHECKPOINT" \
  --drug blood_vs_faeces_label \
  --task kleb_iso_source \
  --ast-sheet-path "$SHEET" \
  --embeddings-dir "$EMB" \
  --out-dir "$OUT_DIR"
status=$?
if [ "$status" -ne 0 ]; then echo "EVAL FAILED (exit $status)"; exit "$status"; fi
echo "Eval done — eval_results.json + ROC/PR PNG in $OUT_DIR"
