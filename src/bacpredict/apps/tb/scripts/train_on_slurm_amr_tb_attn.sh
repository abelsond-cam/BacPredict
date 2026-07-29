#!/bin/bash
#SBATCH --job-name=tb_attn_stagec
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --time=36:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → relative or ~/rds/hpc-work/logs/, and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.

# Stage C (§0.2) for the attention-pool genome head — single split, 1 fold x 1 seed.
# Identical data / split / hyperparameters to train_on_slurm_amr_tb_stage_c.sh, but
# --pooling attention swaps the stock mask-mean genome head for a gated-attention MIL
# pool (src/bacpredict/engine/finetune/attention_pool.py) over the contig-aware BacformerLarge backbone.
# train_amr.py reads train/validate/evaluate from binary_ast_with_split.csv
# (split_source="csv") and writes the §0.4 results.json on the evaluate holdout.
#
# Args: 1=drug (default rifampin), 2=mode (frozen | e2e, default frozen)
#   frozen : --freeze-encoder  -> train pool+head only; targets the frozen
#            rpoB-token ceiling (~0.95) vs the mean-pool baseline 0.905.
#   e2e    : full end-to-end fine-tune -> apples-to-apples vs the 0.905 mean-pool run.
# Override the job name per run so %x logs stay distinct:
#   sbatch --job-name=tb_rif_attn_frozen src/bacpredict/apps/tb/scripts/train_on_slurm_amr_tb_attn.sh rifampin frozen
#   sbatch --job-name=tb_rif_attn_e2e    src/bacpredict/apps/tb/scripts/train_on_slurm_amr_tb_attn.sh rifampin e2e

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction"}"
D="$BACPREDICT_DATA_ROOT"
PY="$HOME/workspace/BacPredict/.venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

species=mycobacterium_tuberculosis
drug=${1:-rifampin}     # TB binary_ast.csv uses US spelling (rifampin)
mode=${2:-frozen}       # frozen | e2e
warmup_proportion=0.1
eval_steps=250
attn_dim=128
model_name_or_path="macwiatrak/bacformer-large-masked-complete-genomes"

# Fresh pool+head wants a higher LR than the backbone fine-tune. frozen trains only
# the new head (5e-4); e2e fine-tunes the pretrained backbone too (1.5e-4, matching
# the stock mean-pool run for an apples-to-apples comparison).
freeze_flag=""
if [ "$mode" = "frozen" ]; then
    freeze_flag="--freeze-encoder"
    lr=5e-4
elif [ "$mode" = "e2e" ]; then
    lr=0.00015
else
    echo "Unknown mode '$mode' (expected 'frozen' or 'e2e')"; exit 1
fi

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

echo "TB AMR attention-pool Stage C — drug=$drug mode=$mode attn_dim=$attn_dim lr=$lr"
echo "Job ID: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"

embeddings_dir="$D/processed/train_tb_ast/esm"

"$PY" -m bacpredict.engine.finetune.finetune_amr --task tb_ast \
--embeddings-dir $embeddings_dir \
--ast-sheet-path "$D/processed/train_tb_ast/binary_ast_with_split.csv" \
--lr $lr \
--model-name-or-path $model_name_or_path \
--warmup-proportion $warmup_proportion \
--drug ${drug} \
--pooling attention \
--attn-dim $attn_dim \
$freeze_flag \
--num-workers 15 \
--grad-accumulation-steps 8 \
--batch-size 1 \
--eval-steps $eval_steps \
--max-steps 100000 \
--early-stopping-patience 30 \
--output-dir "$D/processed/train_tb_ast/checkpoints/${species}_${drug}_attn_${mode}_${SLURM_JOB_ID}"

echo "End of script — check .out/.err for progress + results.json in the output dir."

# Check progress with: squeue -u dca36
