#!/bin/bash
#SBATCH --job-name=train_attn_per_gene_lr_panel_1000
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

# Train the gated-MIL head with the per-gene-LR probability panel on the 1000-genome manifest
# split (TB rifampin). The panel hands the gate an explicit per-protein "this protein predicts
# resistance" pointer (rpoB carries the strong signal). Compare against the panel-less baseline
# (eval AUROC 0.9768, job 30602029) on the IDENTICAL split, then read out the D1 head-pool
# probe: does the gate now route to rpoB.
#
# Build the panel store first: src/pangena_predict/scripts/build_per_gene_lr_panel_store.sh
#
# mode (1st positional arg, default filtered):
#   none        : gated-MIL, NO panel, backbone FROZEN          -> same-split baseline
#   filtered    : panel (genes with out-of-fold AUROC > 0.8) steers the gate, backbone FROZEN
#   unfiltered  : panel (all core genes) steers the gate, backbone FROZEN
#
#   sbatch src/tb_ast/scripts/train_attn_per_gene_lr_panel_1000.sh filtered
#   sbatch src/tb_ast/scripts/train_attn_per_gene_lr_panel_1000.sh unfiltered

cd /home/dca36/workspace/BacPredict
git pull --ff-only || true

mode=${1:-filtered}

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4
export PYTHONUNBUFFERED=1

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
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

uv run python -m bacpredict.engine.finetune.finetune_amr --task tb_ast \
    --drug rifampin \
    --pooling attention \
    --attn-dim 128 \
    $panel_args \
    --freeze-encoder \
    --ast-sheet-path $SHEET \
    --embeddings-dir $RDS/tb_esm_embeddings \
    --lr 1e-3 \
    --max-steps 4000 \
    --eval-steps 100 \
    --early-stopping-patience 12 \
    --num-workers 8 \
    --output-dir $RDS/checkpoints/attn_per_gene_lr_panel_1000_${mode}_$SLURM_JOB_ID

echo "Done — §0.4 results.json in the output dir (eval on the 200-genome holdout)."
