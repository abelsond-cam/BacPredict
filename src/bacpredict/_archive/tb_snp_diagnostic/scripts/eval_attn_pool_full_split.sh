#!/bin/bash
#SBATCH --job-name=eval_attn_pool_full_split
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00

# Test A — does the 1000-genome manifest baseline's eval AUROC (0.9768) survive the FULL 38k eval?
# Scores a TRAINED no-panel gated-MIL checkpoint over the full canonical `evaluate` holdout
# (binary_ast_with_split.csv), EXCLUDING the manifest's own train+validate genomes (no train-on-eval
# leakage) and re-scoring the manifest's 200-genome eval as a path self-check (must reproduce ~0.9768).
#   stays ~0.97 -> read-out genuinely that good (H2);  drops ~0.8 -> manifest was a confound (H1).
#
# One backbone forward per genome over ~7k eval genomes -> minutes-to-~1h on one A100 (6h wall generous).
#
#   $1 = checkpoint run dir (required)  e.g. .../checkpoints/attn_surprisal_panel_1000_none_30602029
#   $2 = label (optional)               output subdir name (default: the checkpoint dir name)
#   $3 = max samples (optional)         smoke cap per split
#
#   sbatch src/pangena_predict/scripts/eval_attn_pool_full_split.sh \
#       $RDS/checkpoints/attn_surprisal_panel_1000_none_30602029 manifest_baseline_30602029

cd /home/dca36/workspace/BacPredict

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4
export PYTHONUNBUFFERED=1

CKPT=${1:?usage: $0 <checkpoint_run_dir> [label] [max_samples]}
LABEL=${2:-$(basename "$CKPT")}
max_samples=${3:-}

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast
AST_SHEET=$RDS/binary_ast_with_split.csv
EMB=$RDS/tb_esm_embeddings
MANIFEST_SPLIT=$RDS/tb_surprisal_panel/tb_rif_1000_split.csv
OUT_DIR=$RDS/pangena_predict/full_eval_attn_pool/$LABEL

extra=""
if [ -n "$max_samples" ]; then
    extra="--max-samples $max_samples"
    OUT_DIR=${OUT_DIR}_smoke
fi

echo "Full-eval A — ckpt=$CKPT  label=$LABEL  ast=$AST_SHEET  manifest=$MANIFEST_SPLIT  out=$OUT_DIR  $extra"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python src/pangena_predict/eval_attn_pool_on_full_split.py \
    --ast-sheet-path "$AST_SHEET" \
    --esm-store-dir "$EMB" \
    --checkpoint-dir "$CKPT" \
    --manifest-split-csv "$MANIFEST_SPLIT" \
    --label "$LABEL" \
    --output-dir "$OUT_DIR" \
    --drug rifampin \
    --device cuda:0 \
    $extra

echo "=== full_eval_summary.json ==="
cat "$OUT_DIR/full_eval_summary.json"
echo
echo "Done — results in $OUT_DIR"
