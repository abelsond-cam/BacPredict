#!/bin/bash
#SBATCH --job-name=tb_rifampin
#SBATCH --output=tb_rifampin_%A_%a.out
#SBATCH --error=tb_rifampin_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --open-mode=append
#SBATCH --array=0-14   # 5 folds × 3 seeds = 15 jobs (comment out to run single-split mode)

cd /home/dca36/workspace/BacPredict

# K-fold settings — SLURM_ARRAY_TASK_ID encodes fold×seed
N_FOLDS=5
FOLD=$(( SLURM_ARRAY_TASK_ID % N_FOLDS ))
SEED=$(( SLURM_ARRAY_TASK_ID / N_FOLDS + 1 ))

species=mycobacterium_tuberculosis
drug=rifampin  # TB binary_ast.csv uses US spelling (rifampin, not rifampicin)
warmup_proportion=0.1
lr=0.00015
eval_steps=250
model_name_or_path="macwiatrak/bacformer-large-masked-complete-genomes"

# Load any necessary modules
module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4

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

embeddings_dir="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast/tb_esm_embeddings"

uv run python src/tb_ast/train_amr.py \
--embeddings-dir $embeddings_dir \
--ast-sheet-path /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast/binary_ast_with_split.csv \
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
--n-folds $N_FOLDS \
--fold $FOLD \
--seed $SEED \
--output-dir /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast/checkpoints/${species}_${drug}_lr_${lr}_finetuned

echo "End of script... check the .out and .err logs for any errors and for training progress"


# Run with: sbatch src/tb_ast/scripts/train_on_slurm_amr_tb.sh
# Check on progress with: squeue -u dca36
