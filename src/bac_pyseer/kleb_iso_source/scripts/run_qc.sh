#!/bin/bash
#SBATCH --job-name=pyseer_qc
#SBATCH --output=pyseer_qc_%j.out
#SBATCH --error=pyseer_qc_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU

# QC the pyseer inputs before the GWAS — two steps, one light job (no himem):
#   1) variant frequency spectrum (>=1% loci streamed straight from the post-filter Rtab —
#      one sequential pass, <1 GB; the <1% count comes from the manifest) → frequency
#      histogram + per-position allele-frequency scatter.
#   2) UMAP of the Jaccard distances → colored by Sublineage and by phenotype.
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
COHORT=sampled_country_2_1_all
COHORT_CSV=$DATA/david/processed/train_iso_source/blood_faeces/$COHORT/kpsc_human/binary_blood_vs_faeces_with_split.csv
OUT_DIR=$PYSEER/blood_faeces/$COHORT
QC_DIR=$OUT_DIR/qc
mkdir -p "$QC_DIR"

echo "Job $SLURM_JOB_ID  Node $SLURMD_NODENAME  cohort=$COHORT  $(date)"

# 1) frequency spectrum (>=1% loci) + per-position scatter — streamed from the Rtab.
uv run python src/bac_pyseer/kleb_iso_source/qc_variant_spectrum.py \
    --rtab "$OUT_DIR/variant_by_loci_presence.Rtab" \
    --manifest "$OUT_DIR/collation_manifest.json" \
    --min-freq 0.01 --contig NC_009648 \
    --spectrum-npz "$QC_DIR/postfilter_locus_spectrum.npz" \
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
