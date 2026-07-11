#!/bin/bash
#SBATCH --job-name=tb_smoke_rif
#SBATCH --output=tb_smoke_rif_%j.out
#SBATCH --error=tb_smoke_rif_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Stage A smoke test for TB rifampin — pipeline correctness only.
# Originally specified as CPU-disabled (§0.2 protocol) but the login node killed
# the CPU run before the first training step. A short GPU job gives a proper
# accelerator quota; the verification target is identical: loss decreases, an
# eval step succeeds, a checkpoint saves.

cd /home/dca36/workspace/BacPredict

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

echo "TB AMR Stage A smoke test (GPU, n=10, drug=rifampin)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES"

uv run python -m bacpredict.engine.finetune.finetune_amr --task tb_ast \
    --drug rifampin \
    --n-samples 10 \
    --num-workers 0 \
    --output-dir /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast/checkpoints/smoke_rifampin_$SLURM_JOB_ID

echo "Smoke test finished."
