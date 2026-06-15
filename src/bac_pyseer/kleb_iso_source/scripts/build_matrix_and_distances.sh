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
module purge

export PYTHONUNBUFFERED=1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
cd /home/dca36/workspace/BacPredict

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
PYSEER=$DATA/david/processed/pyseer_iso_source
CACHE_DIR=$PYSEER/locus_cache
RESOLUTION_TSV=$PYSEER/resolution/blood_faeces_union_resolution.tsv

COHORT=sampled_country_2_1_all   # first target (pooled country-balanced, ~14.2k)
COHORT_CSV=$DATA/david/processed/train_iso_source/blood_faeces/$COHORT/kpsc_human/binary_blood_vs_faeces_with_split.csv
OUT_DIR=$PYSEER/blood_faeces/$COHORT

echo "Job $SLURM_JOB_ID  Node $SLURMD_NODENAME  cohort=$COHORT  $(date)"

uv run python src/bac_pyseer/kleb_iso_source/build_presence_and_distances.py \
    --cohort-csv "$COHORT_CSV" \
    --cache-dir "$CACHE_DIR" \
    --out-dir "$OUT_DIR" \
    --resolution-tsv "$RESOLUTION_TSV" \
    --label-col blood_vs_faeces_label \
    --min-freq 0.01 \
    --n-jobs -1 \
    --min-qual 100 --min-dp 10 --min-altfrac 0.9

echo "Done  $(date)"
