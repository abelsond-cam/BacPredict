#!/bin/bash
#SBATCH --job-name=kleb_bacformer_mean
#SBATCH --output=kleb_bacformer_mean_%j.out
#SBATCH --error=kleb_bacformer_mean_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --open-mode=append
#
# Cache the Kp frozen Bacformer genome-mean for the whole cohort (~6.8k genomes), one forward each.
# Drug-agnostic; written once so every per-drug concat probe + ladder runs CPU-only thereafter.
# ~10-20 min of compute, but time is generous per the never-under-call rule.
#
# Submit (ensure the HPC checkout is on the right branch + up to date first):
#   sbatch src/kleb_ast/scripts/cache_bacformer_genome_mean.sh

cd /home/dca36/workspace/BacPredict
module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4
export PYTHONUNBUFFERED=1

uv run python src/kleb_ast/cache_bacformer_genome_mean.py --device cuda:0
echo "BACFORMER_MEAN_CACHE_DONE"
