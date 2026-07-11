#!/bin/bash
#SBATCH --job-name=tb_smoke_attn
#SBATCH --output=tb_smoke_attn_%j.out
#SBATCH --error=tb_smoke_attn_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Stage A smoke for the attention-pool genome head (TB rifampin) — pipeline +
# overfit correctness only. n=10 train=val=eval, so the target is loss -> 0 /
# AUROC -> 1: that proves the gated-attention pool, the BacformerLargeTrainer
# wiring, and the backbone contig/embedding path all run end-to-end.
#
# Mode via the 1st positional arg (default 'freeze'):
#   freeze : --freeze-encoder  (pool + head only — the milestone-(a) check)
#   e2e    : full end-to-end   (also exercises the unfrozen backbone grad path)
#   sbatch src/tb_ast/scripts/smoke_attn_rifampin.sh            # frozen
#   sbatch src/tb_ast/scripts/smoke_attn_rifampin.sh e2e        # end-to-end

cd /home/dca36/workspace/BacPredict

mode=${1:-freeze}   # freeze | e2e
freeze_flag=""
if [ "$mode" = "freeze" ]; then
    freeze_flag="--freeze-encoder"
elif [ "$mode" != "e2e" ]; then
    echo "Unknown mode '$mode' (expected 'freeze' or 'e2e')"; exit 1
fi

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

echo "TB attention-pool smoke (GPU, n=10, drug=rifampin, mode=$mode)"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"

uv run python -m bacpredict.engine.finetune.finetune_amr --task tb_ast \
    --drug rifampin \
    --pooling attention \
    --attn-dim 128 \
    --lr 1e-3 \
    $freeze_flag \
    --n-samples 10 \
    --num-workers 0 \
    --output-dir /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast/checkpoints/smoke_attn_rifampin_${mode}_$SLURM_JOB_ID

echo "Smoke finished — expect train loss -> ~0 and eval AUROC -> 1.0."
