#!/bin/bash
#SBATCH --job-name=iso_source_cohort
#SBATCH --output=/rds/user/dca36/hpc-work/logs/iso_source_%x_%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/iso_source_%x_%j.err
#SBATCH --time=36:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --open-mode=append

# Stage C fine-tune for one isolation-source cohort, parameterised.
#
# Supersedes the three near-identical copies (train_isolation_source_stage_c{,_pooled,_stratified}.sh)
# which differed only in cohort_dir, eval_steps, and job name — and which all hardcoded
# output_dir=<cohort>/models, so re-running one would overwrite the deployed checkpoint in place.
# Here OUTPUT_SUBDIR defaults to models_bf16, keeping the fp32 checkpoints intact for the A/B.
#
# Logs go to the persistent project tier, never the git working tree (the old copies wrote
# stage_c_*_%j.out into the repo checkout).
#
# Usage:
#   COHORT=sampled_country_2_1_all sbatch -J bf16_pooled  .../train_isolation_source_cohort.sh
#   COHORT=sampled_country_2_1_stratified sbatch -J bf16_strat .../train_isolation_source_cohort.sh
#   COHORT=all_samples PRECISION=bf16 sbatch -J bf16_all .../train_isolation_source_cohort.sh
#   # reproduce the fp32 condition (what produced the 2026-05 numbers):
#   COHORT=... PRECISION=fp32 OUTPUT_SUBDIR=models_fp32_repro sbatch ...
#   # the publication k-fold x seed sweep (FOLD = task % N_FOLDS, SEED = task / N_FOLDS + 1):
#   N_FOLDS=5 OUTPUT_SUBDIR=models_kfold sbatch --array=0-14 -J kfold_pooled .../train_isolation_source_cohort.sh

set -uo pipefail
cd /home/dca36/workspace/BacPredict

COHORT=${COHORT:-sampled_country_2_1_all}
PAIR=${PAIR:-blood_faeces}
FLAVOR=${FLAVOR:-kpsc_human}
PRECISION=${PRECISION:-bf16}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-models_bf16}

processed_base_dir="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_iso_source"
cohort_dir="${processed_base_dir}/${PAIR}/${COHORT}/${FLAVOR}"
sheet_path="${cohort_dir}/binary_blood_vs_faeces_with_split.csv"
embeddings_dir="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_esm_embeddings"
output_dir="${cohort_dir}/${OUTPUT_SUBDIR}"
model_name_or_path="macwiatrak/bacformer-large-masked-complete-genomes"

# eval_steps ~= every half epoch. batch=1 x grad_accum=8, so steps/epoch = n_train/8:
#   all_samples          train ~18k -> ~2.3k steps/epoch -> 1000
#   sampled_country_2_1_all      ~9.9k -> ~1.24k          ->  700
#   sampled_country_2_1_stratified ~7.4k -> ~920          ->  500
case "$COHORT" in
  all_samples)                    default_eval_steps=1000 ;;
  sampled_country_2_1_all)        default_eval_steps=700  ;;
  sampled_country_2_1_stratified) default_eval_steps=500  ;;
  *)                              default_eval_steps=700  ;;
esac
eval_steps=${EVAL_STEPS:-$default_eval_steps}

# K-fold sweep. Off unless N_FOLDS is set AND this is an array task, so every existing single-run
# caller is byte-for-byte unchanged. The grid is the repo convention (root CLAUDE.md, "K-fold CV and
# split semantics"): FOLD = task % N_FOLDS, SEED = task / N_FOLDS + 1.
#
# EVALUATE_SEED is pinned and must stay pinned: it alone fixes the holdout, so every (fold, seed) run
# is scored on the same genomes and the 15 AUROCs are a distribution rather than 15 different
# questions. Changing it mid-sweep silently makes the runs incomparable.
N_FOLDS=${N_FOLDS:-}
EVALUATE_SEED=${EVALUATE_SEED:-1}
# Expanded below as ${kfold_args[@]+"..."} — `set -u` treats an empty array as unset on bash < 4.4,
# so the bare "${kfold_args[@]}" would abort every non-k-fold run on an older login image.
kfold_args=()
seed=${SEED:-1}
if [ -n "$N_FOLDS" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  fold=$(( SLURM_ARRAY_TASK_ID % N_FOLDS ))
  seed=$(( SLURM_ARRAY_TASK_ID / N_FOLDS + 1 ))
  kfold_args=(--n-folds "$N_FOLDS" --fold "$fold" --evaluate-seed "$EVALUATE_SEED")
  # The trainer appends _fold{NN}_seed{S} itself, so all 15 runs can share one OUTPUT_SUBDIR without
  # any of them overwriting another -- or the deployed checkpoint in models/.
elif [ -n "$N_FOLDS" ]; then
  echo "N_FOLDS=$N_FOLDS set but this is not an array task -- submit with --array=0-\$((N_FOLDS*3-1))"; exit 1
fi
lr=${LR:-0.00015}
warmup_proportion=0.1
# ⚠ max_steps is NOT just a cap — it defines the LR SCHEDULE. warmup_steps = max_steps *
# warmup_proportion and the LR decays to zero at max_steps, so lowering it to "save time" changes the
# learning-rate trajectory and therefore the model. Every result to date (fp32 and bf16) used
# 100,000; changing it makes a different experiment, not the same one stopped earlier. Leave it.
max_steps=100000
# The best-model objective is eval_auroc (metric_for_best_model in train_isolation_source.py), so
# patience counts evals with no AUROC improvement — not epochs, not loss.
#
# 30 was too long to ever fire: on the 2026-08-11 bf16 runs validation AUROC peaked at steps
# 31,500 / 30,500 / 15,000 and never recovered, yet only 9 / 15 / 23 non-improving evals had accrued
# when the 36 h wall killed all three. Early stopping was configured and structurally could not
# trigger inside the budget; ~108 GPU-hours ran past the useful point.
#
# 12 (David's call): deliberately still generous — safer than a tight value on jobs whose AUROC
# plateaus and then recovers. It would have stopped these three ~8,400 / 6,000 / 12,000 steps past
# their peaks, i.e. inside the wall with room to spare.
early_stopping_patience=12

[ -f "$sheet_path" ] || { echo "MISSING split CSV: $sheet_path"; exit 1; }

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4

export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=info

echo "========================================================================"
echo "Stage C — isolation source (blood vs faeces)"
echo "Cohort:            ${COHORT}/${FLAVOR}"
echo "Precision:         ${PRECISION}   (recorded in results.json run_config)"
echo "Sheet:             $sheet_path"
echo "Output:            $output_dir"
echo "eval_steps:        $eval_steps   seed: $seed   lr: $lr"
if [ ${#kfold_args[@]} -gt 0 ]; then
  echo "K-fold:            task ${SLURM_ARRAY_TASK_ID} -> n_folds=$N_FOLDS fold=$fold seed=$seed evaluate_seed=$EVALUATE_SEED"
  echo "                   holdout is fixed by evaluate_seed alone -- identical genomes in all runs"
fi
echo "Job ID:            $SLURM_JOB_ID  Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES"
echo "========================================================================"

uv run python src/kleb_iso_source/train_isolation_source.py \
  --isolation-sources blood faeces \
  --processed-base-dir "$processed_base_dir" \
  --sheet-path "$sheet_path" \
  --embeddings-dir "$embeddings_dir" \
  --output-dir "$output_dir" \
  --model-name-or-path "$model_name_or_path" \
  --precision "$PRECISION" \
  --lr "$lr" \
  --warmup-proportion "$warmup_proportion" \
  --batch-size 1 \
  --grad-accumulation-steps 8 \
  --num-workers 15 \
  --eval-steps "$eval_steps" \
  --max-steps "$max_steps" \
  --early-stopping-patience "$early_stopping_patience" \
  --seed "$seed" \
  ${kfold_args[@]+"${kfold_args[@]}"}
status=$?

if [ "$status" -ne 0 ]; then
  echo "Stage C ($COHORT, $PRECISION) FAILED with exit code $status — inspect the .err log."
  exit "$status"
fi
echo "Stage C ($COHORT, $PRECISION) finished — checkpoint + results.json in $output_dir."
echo "Compare results.json auroc against the fp32 run in ${cohort_dir}/models/results.json."
