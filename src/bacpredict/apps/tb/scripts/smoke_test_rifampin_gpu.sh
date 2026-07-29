#!/bin/bash
#SBATCH --job-name=tb_smoke_rif
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --time=00:10:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → relative or ~/rds/hpc-work/logs/, and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.

# Stage A smoke test for TB rifampin — pipeline correctness only.
# Originally specified as CPU-disabled (§0.2 protocol) but the login node killed
# the CPU run before the first training step. A short GPU job gives a proper
# accelerator quota; the verification target is identical: loss decreases, an
# eval step succeeds, a checkpoint saves.

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction"}"
D="$BACPREDICT_DATA_ROOT"
PY="$HOME/workspace/BacPredict/.venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

echo "TB AMR Stage A smoke test (GPU, n=10, drug=rifampin)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES"

"$PY" -m bacpredict.engine.finetune.finetune_amr --task tb_ast \
    --drug rifampin \
    --n-samples 10 \
    --num-workers 0 \
    --output-dir "$D/processed/train_tb_ast/checkpoints/smoke_rifampin_$SLURM_JOB_ID"

echo "Smoke test finished."
