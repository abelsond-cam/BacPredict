#!/bin/bash
#SBATCH --job name=tr_complete_genomes_blood_faeces
#SBATCH --output=tr_complete_genomes_blood_faeces_%j.out
#SBATCH --error=tr_complete_genomes_blood_faeces_%j.err
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

# Script to train the model
python_script="src/predict_kleb_by_bacformer/tl/train_isolation_source.py"
# Isolation sources are the CLI tokens, which are slugified to training_{slug1}_{slug2}
isolation_sources="blood faeces"
# Base dir from these is training_{slug1}_{slug2}, where it finds the train and validate directories
output_dir="complete_genomes_blood_faeces" # Directory is base_dir/output_dir
# If none, then uses /base_dir/bacformer_finetuned_lr_{$lr}

warmup_proportion=0.1
lr=0.00015
eval_steps=1000  # There are ~ 7,000 in training set, so this is one epoch, with step size of 8
# Train from complete genomes model
model_name_or_path="macwiatrak/bacformer-large-masked-complete-genomes"
# Use this to continue training from a checkpoint
#model_name_or_path="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/#training_faeces_respiratory/bacformer_finetuned_lr_0.00015/checkpoint-18000"  
# or to train from scratch (from mags model)
#model_name_or_path="macwiatrak/bacformer-large-masked-MAG"

# Load any necessary modules
module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4

# Force Python unbuffered output for real-time logging
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

echo "========================================================================"
echo "Fine-tuning Bacformer for isolation-source pair prediction (.pt files)"
echo "Isolation sources: $isolation_sources"
echo "Python script: $python_script"
echo "eval_steps: $eval_steps"
echo "Learning rate: $lr"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME, GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================================================"
echo ""

embeddings_dir="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_esm_embeddings"

uv run python "$python_script" \
  --isolation-sources $isolation_sources \
  --embeddings-dir "$embeddings_dir" \
  --lr "$lr" \
  --output-dir "$output_dir" \
  --model-name-or-path "$model_name_or_path" \
  --warmup-proportion "$warmup_proportion" \
  --num-workers 15 \
  --grad-accumulation-steps 8 \
  --batch-size 1 \
  --eval-steps "$eval_steps" \
  --max-steps 100000 \
  --early-stopping-patience 30

echo ""
echo "End of script... check the .out and .err logs for any errors and for training progress"
