#!/bin/bash
# Stage A smoke test — DefensePredictor on 10 reference genomes, BOTH arms (LR + SR).
# Validates: env + weights, the Bakta->combined-GFF conversion, the DP forward pass, and the
# output layout end-to-end. Also times per-genome cost to size the full ~2,900-pair sweep.
#
# GPU job (ESM2-150M forward pass). Build the manifest inside the job so it is self-contained.
# Prereq: scripts/setup_dp_env.sh has been run once on the login node.
#
# Submit:  sbatch src/dp_short_read/scripts/smoke_dp.sh

#SBATCH --job-name=dp_smoke
#SBATCH --output=dp_smoke_%j.out
#SBATCH --error=dp_smoke_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --open-mode=append

set -euo pipefail
cd /home/dca36/workspace/BacPredict

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4

export PYTHONUNBUFFERED=1

PROJECT_K="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw"
PAIRED_INDEX="${PROJECT_K}/david/processed/complete_vs_sr_genomes/paired_index.tsv"
METADATA="${PROJECT_K}/david/final/metadata_v2_all_samples_and_columns.tsv"
PANAROO_REPO="/home/dca36/workspace/panaroo"
OUT_DIR="${PROJECT_K}/david/processed/defence_predictor/smoke"
PY="src/dp_short_read/.venv-dp/bin/python"

echo "================================================================"
echo "DefensePredictor smoke test — 10 reference genomes, LR + SR arms"
echo "Output: ${OUT_DIR}"
echo "Job:    ${SLURM_JOB_ID} on ${SLURMD_NODENAME}, GPU ${CUDA_VISIBLE_DEVICES}"
echo "================================================================"

echo "--- Building manifest (10 reference genomes) ---"
"${PY}" src/dp_short_read/build_dp_cohort.py \
    --paired-index "${PAIRED_INDEX}" \
    --metadata "${METADATA}" \
    --base-dir "${PROJECT_K}" \
    --reference-only --limit 10 \
    --out "${OUT_DIR}/dp_manifest_smoke.tsv"

echo "--- Running DefensePredictor ---"
"${PY}" src/dp_short_read/run_defense_predictor.py \
    --manifest "${OUT_DIR}/dp_manifest_smoke.tsv" \
    --out-dir "${OUT_DIR}" \
    --panaroo-repo "${PANAROO_REPO}"

echo ""
echo "Smoke test finished — inspect ${OUT_DIR}/run_manifest_shard000.tsv"
