#!/bin/bash
#SBATCH --job-name=tb_rif_stagec
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=36:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --open-mode=append

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

cd /home/dca36/workspace/BacPredict

species=mycobacterium_tuberculosis
drug=${1:-rifampin}  # TB binary_ast.csv uses US spelling (rifampin, not rifampicin)
warmup_proportion=0.1
lr=0.00015
eval_steps=250
model_name_or_path="macwiatrak/bacformer-large-masked-complete-genomes"

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

echo "Training TB AMR model — Stage C single split (Bacformer finetuning, linear head)"
echo "drug: $drug"
echo "species: $species"
echo "eval_steps: $eval_steps"
echo "Learning rate: $lr"
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
--output-dir /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_tb_ast/checkpoints/${species}_${drug}_stage_c_${SLURM_JOB_ID}

echo "End of script... check the .out and .err logs for any errors and for training progress"


# Run with: sbatch src/tb_ast/scripts/train_on_slurm_amr_tb_stage_c.sh
# Check on progress with: squeue -u dca36
