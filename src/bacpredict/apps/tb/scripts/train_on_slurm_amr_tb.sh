#!/bin/bash
#SBATCH --job-name=tb_rifampin
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
#SBATCH --array=0-14   # 5 folds × 3 seeds = 15 jobs (comment out to run single-split mode)
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

species=mycobacterium_tuberculosis
drug=${DRUG:-rifampin}  # US spelling (rifampin, not rifampicin); override per-drug via --export=ALL,DRUG=<drug>
warmup_proportion=0.1
lr=0.00015
# TB's AST cohort is ~10x Kp's, so a step-based early-stopping patience buys ~10x fewer
# epochs than the same setting does for Kp. At ~2,850 steps/epoch, eval every 1000 steps
# (~2.8 evals/epoch) with patience 45 gives TB a ~15-epoch no-improvement window (matches
# root §0.2 "early-stopping ~15 epochs"), vs the ~2.5 epochs the old 250/30 gave. GH200 +
# 24h wall covers it; checkpoints save each eval so a wall-hit run can --resume-from-checkpoint.
eval_steps=${EVAL_STEPS:-1000}
patience=${PATIENCE:-45}
model_name_or_path="macwiatrak/bacformer-large-masked-complete-genomes"

# Force Python unbuffered output for real-time logging
export PYTHONUNBUFFERED=1
# (optional but nice) turn on tqdm in non-interactive envs
export TRANSFORMERS_VERBOSITY=info

echo "Training TB AMR model (Bacformer finetuning, linear head)"
echo "drug: $drug"
echo "species: $species"
echo "K-fold: n_folds=$N_FOLDS, fold=$FOLD, seed=$SEED"
echo "eval_steps: $eval_steps"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Learning rate: $lr, Drug: $drug"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES"

embeddings_dir="$D/processed/train_tb_ast/esm"

"$PY" -m bacpredict.engine.finetune.finetune_amr --task tb_ast \
--embeddings-dir "$embeddings_dir" \
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
--early-stopping-patience $patience \
--n-folds $N_FOLDS \
--fold $FOLD \
--seed $SEED \
--output-dir "$D/processed/train_tb_ast/checkpoints/${species}_${drug}_lr_${lr}_finetuned"

echo "End of script... check the .out and .err logs for any errors and for training progress"

# Run with: sbatch src/bacpredict/apps/tb/scripts/train_on_slurm_amr_tb.sh
# Check on progress with: squeue -u dca36.u6fp
