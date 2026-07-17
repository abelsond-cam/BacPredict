#!/bin/bash
#SBATCH --job-name=ceftriaxone
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
# QOS 'normal' on partition workq caps wall at 24h; --qos=restricted48 allows 48h.
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --open-mode=append
#SBATCH --array=0-14   # 5 folds × 3 seeds = 15 jobs.
# Stage C (single canonical run): submit with `sbatch --array=0 train_on_slurm_amr.sh`
# (= fold 0, seed 1). Restore the full 0-14 sweep only for the publication k-fold.
# CSD3/UoHPC variant (when it returns): --partition=ampere --account=FLOTO-SL2-GPU,
#   logs → relative or ~/rds/hpc-work/logs/, and `module load cuda/12.4 cudnn/8.9_cuda-12.4`.

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="${BACPREDICT_REPO:-$HOME/BacPredict}/src:${PYTHONPATH:-}"

# K-fold settings — SLURM_ARRAY_TASK_ID encodes fold×seed
N_FOLDS=5
FOLD=$(( SLURM_ARRAY_TASK_ID % N_FOLDS ))
SEED=$(( SLURM_ARRAY_TASK_ID / N_FOLDS + 1 ))

species=klebsiella_pneumoniae
# Drug defaults to ceftriaxone (Stage A/B/C canonical). For the fan-out, override
# per job: sbatch --array=0 --job-name=meropenem --export=ALL,DRUG=meropenem train_on_slurm_amr.sh
drug=${DRUG:-ceftriaxone}
warmup_proportion=0.1 # (default)
lr=0.00015 # Use this is finetuning the encoder (freeze-encoder not called)
eval_steps=250 # This is 3-8 per epoch with training set size 1500-4000 samples!
# Default = refreshed complete-genomes weights (sub-step 2).
# For sub-step 3 (MAG contrast), switch to "macwiatrak/bacformer-large-masked-MAG".
# Restart training from a checkpoint (example):
# model_name_or_path=<checkpoint-dir>/checkpoint-19250
model_name_or_path="macwiatrak/bacformer-large-masked-complete-genomes"

# Force Python unbuffered output for real-time logging
export PYTHONUNBUFFERED=1
# (optional but nice) turn on tqdm in non-interactive envs
export TRANSFORMERS_VERBOSITY=info

echo "Training AMR model from pytorch (.pt) files (Bacformer finetuning, linear head)"
echo "drug: $drug"
echo "species: $species"
echo "K-fold: n_folds=$N_FOLDS, fold=$FOLD, seed=$SEED"
echo "eval_steps: $eval_steps"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Learning rate: $lr, Drug: $drug"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES"

embeddings_dir="$D/processed/train_kleb_ast/esm"

"$PY" -m bacpredict.engine.finetune.finetune_amr --task kleb_ast \
--embeddings-dir "$embeddings_dir" \
--ast-sheet-path "$D/processed/train_kleb_ast/binary_ast_with_split.csv" \
--lr $lr \
--model-name-or-path $model_name_or_path \
--warmup-proportion $warmup_proportion \
--drug ${drug} \
--num-workers 15 \
--grad-accumulation-steps 8 \
--batch-size 1 \
--eval-steps $eval_steps \
--max-steps 100000 \
--early-stopping-patience 15 \
--n-folds $N_FOLDS \
--fold $FOLD \
--seed $SEED \
--output-dir "$D/processed/train_kleb_ast/models/finetune/${species}_${drug}_lr_${lr}_finetuned"

echo "End of script... check the .out and .err logs for any errors and for training progress"

# Run with: sbatch src/bacpredict/apps/kleb/scripts/train_on_slurm_amr.sh
# Check on progress with: squeue -u dca36.u6fp
