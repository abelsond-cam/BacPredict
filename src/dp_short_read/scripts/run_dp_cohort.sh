#!/bin/bash
# Full DefensePredictor sweep over the paired LR/SR cohort (~2,900 pairs -> ~5,800 arms),
# sharded across a SLURM GPU array. Run the smoke test FIRST to validate + time per genome,
# then size --array and --time from the measured cost before launching this.
#
# Build the full manifest once on the login node (cheap — two TSV reads):
#   PROJECT_K=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
#   src/dp_short_read/.venv-dp/bin/python src/dp_short_read/build_dp_cohort.py \
#       --paired-index ${PROJECT_K}/david/processed/complete_vs_sr_genomes/paired_index.tsv \
#       --metadata     ${PROJECT_K}/david/final/metadata_v2_all_samples_and_columns.tsv \
#       --base-dir     ${PROJECT_K} \
#       --out          ${PROJECT_K}/david/processed/defence_predictor/full/dp_manifest_full.tsv
#
# Then:  sbatch src/dp_short_read/scripts/run_dp_cohort.sh
# Each array task processes a round-robin shard; N_SHARDS must equal the array size.

#SBATCH --job-name=dp_cohort
#SBATCH --output=dp_cohort_%A_%a.out
#SBATCH --error=dp_cohort_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --array=0-19
#SBATCH --open-mode=append

set -euo pipefail
cd /home/dca36/workspace/BacPredict

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4
export PYTHONUNBUFFERED=1

PROJECT_K="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw"
OUT_DIR="${PROJECT_K}/david/processed/defence_predictor/full"
MANIFEST="${OUT_DIR}/dp_manifest_full.tsv"
PANAROO_REPO="/home/dca36/workspace/panaroo"
PY="src/dp_short_read/.venv-dp/bin/python"

N_SHARDS=20  # keep in sync with --array=0-19

echo "DP cohort shard ${SLURM_ARRAY_TASK_ID}/${N_SHARDS} | job ${SLURM_ARRAY_JOB_ID} on ${SLURMD_NODENAME}"

"${PY}" src/dp_short_read/run_defense_predictor.py \
    --manifest "${MANIFEST}" \
    --out-dir "${OUT_DIR}" \
    --panaroo-repo "${PANAROO_REPO}" \
    --n-shards "${N_SHARDS}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}"
