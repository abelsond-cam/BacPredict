#!/bin/bash
#SBATCH --job-name=pyseer_qc
#SBATCH --output=pyseer_qc_%j.out
#SBATCH --error=pyseer_qc_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU

# QC the pyseer inputs before the GWAS — two steps, one job:
#   1) variant frequency spectrum (rebuilds the PRE-filter per-locus freq from the cache;
#      ~63 GB peak, the only heavy step) → frequency histogram + per-position scatter.
#   2) UMAP of the Jaccard distances (light) → colored by Sublineage and by phenotype.
# Figures + data npz land in the cohort's qc/ dir on RDS; the small PNGs are scp'd to the
# repo's docs/figures from the laptop afterwards (commit from local, per convention).
#
# Usage: sbatch src/bac_pyseer/kleb_iso_source/scripts/run_qc.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
unset PYTHONPATH PYTHONHOME
cd /home/dca36/workspace/BacPredict

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
PYSEER=$DATA/david/processed/pyseer_iso_source
CACHE_DIR=$PYSEER/locus_cache
COHORT=sampled_country_2_1_all
COHORT_CSV=$DATA/david/processed/train_iso_source/blood_faeces/$COHORT/kpsc_human/binary_blood_vs_faeces_with_split.csv
OUT_DIR=$PYSEER/blood_faeces/$COHORT
QC_DIR=$OUT_DIR/qc
mkdir -p "$QC_DIR"

echo "Job $SLURM_JOB_ID  Node $SLURMD_NODENAME  cohort=$COHORT  $(date)"

# 1) frequency spectrum + per-position scatter (pre-filter freq rebuilt from cache).
uv run python src/bac_pyseer/kleb_iso_source/qc_variant_spectrum.py \
    --cohort-csv "$COHORT_CSV" --cache-dir "$CACHE_DIR" \
    --label-col blood_vs_faeces_label --min-freq 0.01 --n-jobs -1 \
    --spectrum-npz "$QC_DIR/prefilter_locus_spectrum.npz" \
    --out-fig-dir "$QC_DIR"

# 2) UMAP of the Jaccard distances, colored by Sublineage + phenotype.
uv run python src/bac_pyseer/kleb_iso_source/qc_distance_umap.py \
    --distances-npz "$OUT_DIR/jaccard_distances.npz" \
    --split-csv "$COHORT_CSV" \
    --sl-col Sublineage --label-col blood_vs_faeces_label \
    --top-n 10 --n-neighbors 15 --seed 42 \
    --coords-npz "$QC_DIR/umap_coords.npz" \
    --out-fig-dir "$QC_DIR"

echo "QC figures + npz in $QC_DIR"
ls -lh "$QC_DIR"
echo "Done  $(date)"
