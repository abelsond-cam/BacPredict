#!/bin/bash
#SBATCH --job-name=train_attn_surprisal_panel_1000
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → relative or ~/rds/hpc-work/logs/, and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.

# Train the gated-MIL head with the per-protein surprisal panel on the 1000-genome manifest
# split (TB rifampin). Tests whether the surprisal panel lets the gate route to rpoB (the
# panel-less head puts rpoB at only the ~68th head-pool percentile; D1 diagnostic).
#
# Runs on the class-balanced manifest split sheet (~700 train / 100 val / 200 eval), produced by
# the split-sheet step (add_splits over manifest.csv) -> $STORE_DIR/tb_rif_1000_split.csv.
#
# mode (1st positional arg, default att_head):
#   none     : gated-MIL, NO panel, backbone FROZEN            -> same-split baseline
#   att_head : gated-MIL + panel steers the gate, backbone FROZEN, pooled value = pure token
#   e2e      : gated-MIL + panel into gate + pooled value, backbone fine-tuned end-to-end
#
#   sbatch src/bacpredict/apps/tb/scripts/train_attn_surprisal_panel_1000.sh none
#   sbatch src/bacpredict/apps/tb/scripts/train_attn_surprisal_panel_1000.sh att_head

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction"}"
D="$BACPREDICT_DATA_ROOT"
PY="$HOME/workspace/BacPredict/.venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

mode=${1:-att_head}

export PYTHONUNBUFFERED=1

RDS=$D/processed/train_tb_ast
STORE_DIR=$RDS/tb_surprisal_panel
SHEET=$STORE_DIR/tb_rif_1000_split.csv
STD=$STORE_DIR/panel_standardization.json

panel_args=""
freeze_flag=""
case "$mode" in
  none)     freeze_flag="--freeze-encoder" ;;
  att_head) freeze_flag="--freeze-encoder"; panel_args="--panel-mode att_head --panel-store $STORE_DIR --panel-stats $STD" ;;
  e2e)      panel_args="--panel-mode e2e --panel-store $STORE_DIR --panel-stats $STD" ;;
  *) echo "Unknown mode '$mode' (expected none|att_head|e2e)"; exit 1 ;;
esac

echo "TB 1000-genome surprisal-panel run — mode=$mode  job=$SLURM_JOB_ID  node=$SLURMD_NODENAME"
echo "sheet=$SHEET"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PY" -m bacpredict.engine.finetune.finetune_amr --task tb_ast \
    --drug rifampin \
    --pooling attention \
    --attn-dim 128 \
    $panel_args \
    $freeze_flag \
    --ast-sheet-path $SHEET \
    --embeddings-dir $RDS/esm \
    --lr 1e-3 \
    --max-steps 4000 \
    --eval-steps 100 \
    --early-stopping-patience 12 \
    --num-workers 8 \
    --output-dir $RDS/checkpoints/attn_surprisal_panel_1000_${mode}_$SLURM_JOB_ID

echo "Done — §0.4 results.json in the output dir (eval on the 200-genome holdout)."
