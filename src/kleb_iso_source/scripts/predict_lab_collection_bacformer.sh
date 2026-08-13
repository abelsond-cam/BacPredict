#!/bin/bash
#SBATCH --job-name=lab_bacformer
#SBATCH --output=/rds/user/dca36/hpc-work/logs/lab_bacformer_%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/lab_bacformer_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Score the lab collection with the fine-tuned invasion head — both cohorts' models in one pass.
#
# ~673 genomes of Bacformer-large inference. The 2.8k holdout runs in well under an hour, so this is
# ~10 min per model; 1 h requested (a short job's failure is cheap to retry, so the padding is small
# by design, unlike the long cohort-scoring runs).
#
# Both models are scored deliberately: which cohort's model is "Bacformer" is decided separately by
# the per-sublineage comparison, and re-queuing a GPU job to fetch the other column would be waste.
# The extra column also shows the lab where the two models disagree.

set -uo pipefail
cd /home/dca36/workspace/BacPredict

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david
LAB=${LAB_DIR:-$DATA/processed/train_iso_source/lab_collection}
MANIFEST=${MANIFEST:-$LAB/lab_collection_manifest.csv}
OUT=${OUT:-$LAB/lab_bacformer_probs.csv}
MODELS_DIR=${MODELS_DIR:-models}   # fp32 deployed checkpoints

module purge
module load cuda/12.4 || echo "WARN: cuda module not found (torch uses bundled CUDA)"
module load cudnn/8.9_cuda-12.4 || true
export PYTHONUNBUFFERED=1

[ -s "$MANIFEST" ] || { echo "ERROR: missing manifest $MANIFEST"; exit 1; }
echo "manifest=$MANIFEST out=$OUT models_dir=$MODELS_DIR"

uv run python -m kleb_iso_source.predict_lab_collection bacformer \
  --manifest "$MANIFEST" \
  --out "$OUT" \
  --models-dir "$MODELS_DIR" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}"

status=$?
if [ "$status" -ne 0 ]; then echo "LAB BACFORMER SCORING FAILED (exit $status)"; exit "$status"; fi
echo "=== done $(date) ==="
