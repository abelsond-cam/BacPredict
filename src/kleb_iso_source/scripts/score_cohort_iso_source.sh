#!/bin/bash
#SBATCH --job-name=score_cohort_iso
#SBATCH --output=/rds/user/dca36/hpc-work/logs/score_cohort_iso_%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/score_cohort_iso_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=ampere
#SBATCH --account=FLOTO-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

# Score EVERY labelled genome in an iso-source cohort with a deployed checkpoint, keeping each
# genome's split label, so per-stratum tables can be built at full n rather than on the 20% holdout.
# Only 5 sublineages and 4 clonal groups clear n>=100 inside the holdout; on the whole cohort it is
# 20 and 15. The train rows are fitted-on and their AUROC is NOT a measurement — score_cohort and
# stratified_metrics both say so loudly, and --restrict-split evaluate is what gets quoted.
#
# ~14.1k genomes of Bacformer-large inference. The 2.8k holdout takes well under an hour, so ~3-4 h
# here; 8 h requested because a wall-clock kill costs the whole run and another GPU queue wait.
#
# Usage: COHORT=sampled_country_2_1_all sbatch .../score_cohort_iso_source.sh
#        CKPT_SUBDIR=models_bf16 COHORT=... sbatch ...   # once the bf16 runs land

set -uo pipefail
cd /home/dca36/workspace/BacPredict

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david
COHORT=${COHORT:-sampled_country_2_1_all}
PAIR=${PAIR:-blood_faeces}
FLAVOR=${FLAVOR:-kpsc_human}
CKPT_SUBDIR=${CKPT_SUBDIR:-models}
LABEL_COL=${LABEL_COL:-blood_vs_faeces_label}

C=$DATA/processed/train_iso_source/$PAIR/$COHORT/$FLAVOR
EMB=${EMB:-$DATA/processed/klebsiella_esm_embeddings}
OUT=${OUT:-$C/$CKPT_SUBDIR/cohort_scores.npz}

module purge
module load cuda/12.4
module load cudnn/8.9_cuda-12.4
export PYTHONUNBUFFERED=1

echo "cohort=$COHORT ckpt=$C/$CKPT_SUBDIR out=$OUT"
uv run python -m bacpredict.engine.finetune.score_cohort \
  --checkpoint "$C/$CKPT_SUBDIR" \
  --split-csv "$C/binary_blood_vs_faeces_with_split.csv" \
  --label-column "$LABEL_COL" \
  --embeddings-dir "$EMB" \
  --out "$OUT" \
  --num-workers 15
