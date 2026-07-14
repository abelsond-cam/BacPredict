#!/bin/bash
#SBATCH --job-name=tb_rif_stagec
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --time=36:00:00
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → relative or ~/rds/hpc-work/logs/, and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.

# Stage C (§0.2) — single split, 1 fold × 1 seed. Per-drug single-split run.
# No --array / --n-folds / --fold / --seed: train_amr.py reads the train/validate/
# evaluate assignment straight from binary_ast_with_split.csv (split_source="csv")
# and writes the §0.4 results.json on the evaluate holdout at the end.
# The k-fold array sweep (publication only) lives in train_on_slurm_amr_tb.sh.
#
# Drug is the first positional arg (defaults to rifampin). Override the SLURM
# job name per drug so the %x-based log files stay distinct, e.g.:
#   sbatch --job-name=tb_isoniazid_stagec train_on_slurm_amr_tb_stage_c.sh isoniazid
# Drug names must match the binary_ast.csv columns (US spellings — rifampin etc.).

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"

species=mycobacterium_tuberculosis
drug=${1:-rifampin}  # TB binary_ast.csv uses US spelling (rifampin, not rifampicin)
warmup_proportion=0.1
lr=0.00015
eval_steps=250
model_name_or_path="macwiatrak/bacformer-large-masked-complete-genomes"

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

echo "Training TB AMR model — Stage C single split (Bacformer finetuning, linear head)"
echo "drug: $drug"
echo "species: $species"
echo "eval_steps: $eval_steps"
echo "Learning rate: $lr"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES"

embeddings_dir="$D/processed/train_tb_ast/esm"

"$PY" -m bacpredict.engine.finetune.finetune_amr --task tb_ast \
--embeddings-dir $embeddings_dir \
--ast-sheet-path "$D/processed/train_tb_ast/binary_ast_with_split.csv" \
--lr $lr \
--model-name-or-path $model_name_or_path \
--warmup-proportion $warmup_proportion \
--drug ${drug} \
--num-workers 15 \
--grad-accumulation-steps 8 \
--batch-size 1 \
--eval-steps $eval_steps \
--max-steps 100000 \
--early-stopping-patience 30 \
--output-dir "$D/processed/train_tb_ast/checkpoints/${species}_${drug}_stage_c_${SLURM_JOB_ID}"

echo "End of script... check the .out and .err logs for any errors and for training progress"


# Run with: sbatch src/bacpredict/apps/tb/scripts/train_on_slurm_amr_tb_stage_c.sh
# Check on progress with: squeue -u dca36
