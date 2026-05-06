#!/bin/bash
#SBATCH --job-name=amikacin
#SBATCH --output=amikacin_%A_%a.out
#SBATCH --error=amikacin_%A_%a.err
#SBATCH --time=36:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --open-mode=append

cd /home/dca36/workspace/predict_kleb_by_bacformer

species=klebsiella_pneumoniae
drug=amikacin
warmup_proportion=0.1 # (default)
lr=0.00015 # Use this is finetuning the encoder (freeze-encoder not called)
eval_steps=250 # This is 3-8 per epoch with training set size 1500-4000 samples!
#Restart training from a checkpoint
#model_name_or_path=/home/dca36/rds/hpc-work/data/BacFormer/models/finetune/acinetobacter_baumannii_ceftazidime_lr_0.00015_finetuned/checkpoint-19250
model_name_or_path="macwiatrak/bacformer-large-masked-MAG"

# declare -a learning_rates=0.00015
# lr_idx=$SLURM_ARRAY_TASK_ID
# lr=${learning_rates[$lr_idx]}

# Load any necessary modules
module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4

# Force Python unbuffered output for real-time logging
export PYTHONUNBUFFERED=1
# (optional but nice) turn on tqdm in non-interactive envs
export TRANSFORMERS_VERBOSITY=info

echo "Training AMR model from pytorch (.pt) files (Bacformer finetuning, linear head)"
echo "drug: $drug"
echo "species: $species"
echo "Training set 70%, validation set 10%, test set 20%"
echo "eval_steps: $eval_steps"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Learning rate: $lr, Drug: $drug"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES"


echo "Finetuned model from pytorch (.pt) files (Bacformer finetuning, linear head)"
embeddings_dir="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_esm_embeddings"

uv run python src/predict_kleb_by_bacformer/tl/train_amr.py  \
--embeddings-dir $embeddings_dir \
--ast-sheet-path /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/binary_ast_with_split.csv \
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
--output-dir /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/ast_training/models/finetune/${species}_${drug}_lr_${lr}_finetuned

echo "End of script... check the .out and .err logs for any errors and for training progress"


# Run with: sbatch train_on_slurm_finetune.sh
# Check on progress with: squeue -u dca36

