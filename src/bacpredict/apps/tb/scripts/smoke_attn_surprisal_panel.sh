#!/bin/bash
#SBATCH --job-name=smoke_attn_surprisal_panel
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Stage A/B overfit for the surprisal-PANEL attention head (TB rifampin).
# n=10 train=val=eval over the class-balanced smoke sheet → target loss -> 0 / AUROC -> 1.
# Proves the full panel path: PanelInjectingFileDataset -> collate panel pad ->
# PanelBacformerLargeTrainer.compute_loss pops panel -> model concatenates panel onto the
# backbone token -> gated-attention gate. Run AFTER build_surprisal_panel_store.sh.
#
# Mode via the 1st positional arg (default 'att_head'):
#   att_head : panel steers the gate only; pooled value stays the pure backbone token;
#              backbone frozen (--freeze-encoder), so pool+head+panel train.
#   e2e      : panel carried into the pooled value + head; backbone fine-tuned end-to-end.
#   sbatch src/tb_ast/scripts/smoke_attn_surprisal_panel.sh att_head
#   sbatch src/tb_ast/scripts/smoke_attn_surprisal_panel.sh e2e

cd /home/dca36/workspace/BacPredict

mode=${1:-att_head}   # att_head | e2e
freeze_flag=""
if [ "$mode" = "att_head" ]; then
    freeze_flag="--freeze-encoder"
elif [ "$mode" != "e2e" ]; then
    echo "Unknown mode '$mode' (expected 'att_head' or 'e2e')"; exit 1
fi

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
STORE_DIR=$RDS/tb_surprisal_panel

echo "TB surprisal-panel smoke (GPU, n=10, drug=rifampin, panel-mode=$mode)"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"

uv run python src/tb_ast/train_amr.py \
    --drug rifampin \
    --pooling attention \
    --attn-dim 128 \
    --panel-mode $mode \
    --panel-store $STORE_DIR \
    --panel-stats $STORE_DIR/panel_standardization.json \
    --ast-sheet-path $STORE_DIR/tb_rif_smoke_split.csv \
    --embeddings-dir $RDS/tb_esm_embeddings \
    --lr 1e-3 \
    $freeze_flag \
    --n-samples 10 \
    --num-workers 0 \
    --output-dir $RDS/checkpoints/smoke_attn_surprisal_panel_${mode}_$SLURM_JOB_ID

echo "Smoke finished — expect train loss -> ~0 and eval AUROC -> 1.0 (panel path wired)."
