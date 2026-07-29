#!/bin/bash
#SBATCH --job-name=tb_smoke_attn
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --time=00:30:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → relative or ~/rds/hpc-work/logs/, and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.

# Stage A smoke for the attention-pool genome head (TB rifampin) — pipeline +
# overfit correctness only. n=10 train=val=eval, so the target is loss -> 0 /
# AUROC -> 1: that proves the gated-attention pool, the BacformerLargeTrainer
# wiring, and the backbone contig/embedding path all run end-to-end.
#
# Mode via the 1st positional arg (default 'freeze'):
#   freeze : --freeze-encoder  (pool + head only — the milestone-(a) check)
#   e2e    : full end-to-end   (also exercises the unfrozen backbone grad path)
#   sbatch src/bacpredict/apps/tb/scripts/smoke_attn_rifampin.sh            # frozen
#   sbatch src/bacpredict/apps/tb/scripts/smoke_attn_rifampin.sh e2e        # end-to-end

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction"}"
D="$BACPREDICT_DATA_ROOT"
PY="$HOME/workspace/BacPredict/.venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

mode=${1:-freeze}   # freeze | e2e
freeze_flag=""
if [ "$mode" = "freeze" ]; then
    freeze_flag="--freeze-encoder"
elif [ "$mode" != "e2e" ]; then
    echo "Unknown mode '$mode' (expected 'freeze' or 'e2e')"; exit 1
fi

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

echo "TB attention-pool smoke (GPU, n=10, drug=rifampin, mode=$mode)"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"

"$PY" -m bacpredict.engine.finetune.finetune_amr --task tb_ast \
    --drug rifampin \
    --pooling attention \
    --attn-dim 128 \
    --lr 1e-3 \
    $freeze_flag \
    --n-samples 10 \
    --num-workers 0 \
    --output-dir "$D/processed/train_tb_ast/checkpoints/smoke_attn_rifampin_${mode}_$SLURM_JOB_ID"

echo "Smoke finished — expect train loss -> ~0 and eval AUROC -> 1.0."
