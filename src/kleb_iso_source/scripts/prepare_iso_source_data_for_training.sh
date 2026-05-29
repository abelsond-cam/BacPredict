#!/bin/bash
# Build the iso-source split CSV (binary_<pair>_with_split.csv) for one cohort.
# Writes into <cohort>/kpsc_human/ (the KPSC-clean filter flavor; mixed_species/
# is the pre-fix run we keep as reference).
#
# Usage:  sbatch [--job-name=prep_<cohort>] prepare_iso_source_data_for_training.sh <cohort>
#   cohort ∈ {all_samples, sampled_country_2_1_stratified, sampled_country_2_1_all}
#
# Input metadata:
#   - all_samples            → v2 metadata directly (no upstream stratified TSV);
#                              prepare's host + KPSC + Sublineage filters define the cohort.
#   - sampled_country_2_1_*  → the sampler's stratified TSV under <cohort>/kpsc_human/
#                              (already country-capped + thread-handled + KPSC-filtered).

#SBATCH --job-name=prep_iso_source
#SBATCH --output=prep_iso_source_%j.out
#SBATCH --error=prep_iso_source_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --open-mode=append

cd /home/dca36/workspace/BacPredict

COHORT="${1:?usage: sbatch $0 <cohort>}"
case "$COHORT" in
  all_samples|sampled_country_2_1_stratified|sampled_country_2_1_all) ;;
  *) echo "ERROR: unknown cohort '$COHORT'" >&2; exit 2 ;;
esac

BASE=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_iso_source/blood_faeces
DIR=$BASE/$COHORT/kpsc_human
mkdir -p "$DIR"

if [ "$COHORT" = "all_samples" ]; then
  INPUT=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_v2_all_samples_and_columns.tsv
else
  INPUT=$DIR/stratified_selected_isolation_source_metadata.tsv
fi
if [ ! -f "$INPUT" ]; then
  echo "ERROR: input metadata not found at $INPUT" >&2; exit 3
fi

export PYTHONUNBUFFERED=1

echo "========================================================================"
echo "Prep iso-source data — cohort=$COHORT"
echo "Input:  $INPUT"
echo "Output: $DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================================================"

uv run python src/kleb_iso_source/prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py \
  --isolation-sources blood faeces \
  --input-metadata-file "$INPUT" \
  --output-dir "$DIR"
status=$?

if [ "$status" -ne 0 ]; then
  echo "PREP FAILED with exit $status — inspect .err"; exit "$status"
fi
echo "Prep done — split CSV at $DIR/binary_blood_vs_faeces_with_split.csv"
