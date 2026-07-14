#!/bin/bash
#SBATCH --job-name=train_attn_per_gene_lr_panel_1000
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --time=02:00:00
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → relative or ~/rds/hpc-work/logs/, and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.

# Train the gated-MIL head with the per-gene-LR probability panel on the 1000-genome manifest
# split (TB rifampin). The panel hands the gate an explicit per-protein "this protein predicts
# resistance" pointer (rpoB carries the strong signal). Compare against the panel-less baseline
# (eval AUROC 0.9768, job 30602029) on the IDENTICAL split, then read out the D1 head-pool
# probe: does the gate now route to rpoB.
#
# Build the panel store first: src/bacpredict/engine/scripts/build_per_gene_lr_panel_store.sh
#
# mode (1st positional arg, default filtered):
#   none        : gated-MIL, NO panel, backbone FROZEN          -> same-split baseline
#   filtered    : panel (genes with out-of-fold AUROC > 0.8) steers the gate, backbone FROZEN
#   unfiltered  : panel (all core genes) steers the gate, backbone FROZEN
#
#   sbatch src/bacpredict/apps/tb/scripts/train_attn_per_gene_lr_panel_1000.sh filtered
#   sbatch src/bacpredict/apps/tb/scripts/train_attn_per_gene_lr_panel_1000.sh unfiltered

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

mode=${1:-filtered}

export PYTHONUNBUFFERED=1

RDS=$D/processed/train_tb_ast
STORE_DIR=$RDS/tb_per_gene_lr_panel
SHEET=$RDS/tb_surprisal_panel/tb_rif_1000_split.csv   # SAME split as the surprisal/baseline runs

panel_args=""
case "$mode" in
  none)        : ;;
  filtered|unfiltered)
      STORE=$STORE_DIR/$mode
      panel_args="--panel-mode att_head --panel-store $STORE --panel-stats $STORE/panel_standardization.json" ;;
  *) echo "Unknown mode '$mode' (expected none|filtered|unfiltered)"; exit 1 ;;
esac

echo "TB 1000-genome per-gene-LR run — mode=$mode  job=$SLURM_JOB_ID  node=$SLURMD_NODENAME"
echo "sheet=$SHEET"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"$PY" -m bacpredict.engine.finetune.finetune_amr --task tb_ast \
    --drug rifampin \
    --pooling attention \
    --attn-dim 128 \
    $panel_args \
    --freeze-encoder \
    --ast-sheet-path $SHEET \
    --embeddings-dir $RDS/esm \
    --lr 1e-3 \
    --max-steps 4000 \
    --eval-steps 100 \
    --early-stopping-patience 12 \
    --num-workers 8 \
    --output-dir $RDS/checkpoints/attn_per_gene_lr_panel_1000_${mode}_$SLURM_JOB_ID

echo "Done — §0.4 results.json in the output dir (eval on the 200-genome holdout)."
