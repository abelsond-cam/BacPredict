#!/bin/bash
#SBATCH --job-name=pyseer_build_matrix
#SBATCH --output=pyseer_build_matrix_%j.out
#SBATCH --error=pyseer_build_matrix_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU

# Reduce the per-sample locus cache into pyseer inputs for one cohort: the variant
# presence Rtab (<1% loci dropped), Jaccard distances, phenotype, and manifest.
# Generous on memory + time; the dense Jaccard is the only heavy step (fine at the
# Tier-1 14-21k scale on a himem node).
#
# Edit COHORT below to retarget (sampled_country_2_1_all is the chosen first cohort).
# Usage: sbatch src/bac_pyseer/kleb_iso_source/scripts/build_matrix_and_distances.sh

set -euo pipefail
# Pure-Python reduce (scipy/sklearn) under uv — no bcftools, no modules. Clear PYTHONPATH
# so a stray spack/module leak in the environment can't shadow uv's numpy.
export PYTHONUNBUFFERED=1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
unset PYTHONPATH PYTHONHOME
cd /home/dca36/workspace/BacPredict

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
PYSEER=$DATA/david/processed/pyseer_iso_source
CACHE_DIR=$PYSEER/locus_cache

# Toggle these (env) to retarget a pair/cohort; defaults = blood/faeces sampled_country_2_1_all.
# e.g.  PAIR=faeces_respiratory LABEL_COL=respiratory_vs_faeces_label \
#       COHORT_CSV=…/faeces_respiratory/$COHORT/kpsc_human/binary_respiratory_vs_faeces_labels.csv \
#       RESOLUTION_TSV=$PYSEER/resolution/faeces_respiratory_resolution.tsv sbatch … build_matrix_and_distances.sh
PAIR=${PAIR:-blood_faeces}
COHORT=${COHORT:-sampled_country_2_1_all}   # pooled country-balanced
LABEL_COL=${LABEL_COL:-blood_vs_faeces_label}
RESOLUTION_TSV=${RESOLUTION_TSV:-$PYSEER/resolution/blood_faeces_union_resolution.tsv}
COHORT_CSV=${COHORT_CSV:-$DATA/david/processed/train_iso_source/$PAIR/$COHORT/kpsc_human/binary_blood_vs_faeces_with_split.csv}
OUT_DIR=$PYSEER/$PAIR/$COHORT

echo "Job $SLURM_JOB_ID  Node $SLURMD_NODENAME  pair=$PAIR  cohort=$COHORT  label=$LABEL_COL  $(date)"

uv run python src/bac_pyseer/kleb_iso_source/build_presence_and_distances.py \
    --cohort-csv "$COHORT_CSV" \
    --cache-dir "$CACHE_DIR" \
    --out-dir "$OUT_DIR" \
    --resolution-tsv "$RESOLUTION_TSV" \
    --label-col "$LABEL_COL" \
    --min-freq 0.01 \
    --n-jobs -1 \
    --min-qual 100 --min-dp 3

echo "Done  $(date)"
