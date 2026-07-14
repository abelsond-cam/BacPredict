#!/bin/bash
#SBATCH --job-name=smoke_attn_surprisal_panel
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --time=00:30:00
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → relative or ~/rds/hpc-work/logs/, and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.

# Stage A/B overfit for the surprisal-PANEL attention head (TB rifampin).
# n=10 train=val=eval over the class-balanced smoke sheet → target loss -> 0 / AUROC -> 1.
# Proves the full panel path: PanelInjectingFileDataset -> collate panel pad ->
# PanelBacformerLargeTrainer.compute_loss pops panel -> model concatenates panel onto the
# backbone token -> gated-attention gate. The surprisal-panel STORE is built by the (now
# ARCHIVED) src/bacpredict/_archive/tb_snp_diagnostic/scripts/build_surprisal_panel_store.sh —
# run that first if the store is missing. (This smoke drives the LIVE engine trainer.)
#
# Mode via the 1st positional arg (default 'att_head'):
#   att_head : panel steers the gate only; pooled value stays the pure backbone token;
#              backbone frozen (--freeze-encoder), so pool+head+panel train.
#   e2e      : panel carried into the pooled value + head; backbone fine-tuned end-to-end.
#   sbatch src/bacpredict/apps/tb/scripts/smoke_attn_surprisal_panel.sh att_head
#   sbatch src/bacpredict/apps/tb/scripts/smoke_attn_surprisal_panel.sh e2e

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

mode=${1:-att_head}   # att_head | e2e
freeze_flag=""
if [ "$mode" = "att_head" ]; then
    freeze_flag="--freeze-encoder"
elif [ "$mode" != "e2e" ]; then
    echo "Unknown mode '$mode' (expected 'att_head' or 'e2e')"; exit 1
fi

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

RDS=$D/processed/train_tb_ast
STORE_DIR=$RDS/tb_surprisal_panel

echo "TB surprisal-panel smoke (GPU, n=10, drug=rifampin, panel-mode=$mode)"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"

"$PY" -m bacpredict.engine.finetune.finetune_amr --task tb_ast \
    --drug rifampin \
    --pooling attention \
    --attn-dim 128 \
    --panel-mode $mode \
    --panel-store $STORE_DIR \
    --panel-stats $STORE_DIR/panel_standardization.json \
    --ast-sheet-path $STORE_DIR/tb_rif_smoke_split.csv \
    --embeddings-dir $RDS/esm \
    --lr 1e-3 \
    $freeze_flag \
    --n-samples 10 \
    --num-workers 0 \
    --output-dir $RDS/checkpoints/smoke_attn_surprisal_panel_${mode}_$SLURM_JOB_ID

echo "Smoke finished — expect train loss -> ~0 and eval AUROC -> 1.0 (panel path wired)."
