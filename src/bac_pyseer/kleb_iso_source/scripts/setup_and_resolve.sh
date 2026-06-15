#!/bin/bash
#SBATCH --job-name=pyseer_setup_resolve
#SBATCH --output=pyseer_setup_resolve_%j.out
#SBATCH --error=pyseer_setup_resolve_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU

# One-time setup for the bac_pyseer / kleb_iso_source collation:
#   (1) stage the shared reference FASTA (NC_009648, MGH 78578) + faidx it, and
#   (2) resolve every Sample in the blood/faeces union to its raw snippy VCF
#       -> the resolution TSV the extraction array chunks over.
#
# Usage: sbatch src/bac_pyseer/kleb_iso_source/scripts/setup_and_resolve.sh

set -euo pipefail
# samtools (for faidx) comes from the bac_pyseer pixi env, not a spack module (which would
# leak python-3.9 site-packages onto PYTHONPATH and break uv). Run `pixi install` in
# src/bac_pyseer/ once before this job.
export PYTHONUNBUFFERED=1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
unset PYTHONPATH PYTHONHOME
cd /home/dca36/workspace/BacPredict

SAMTOOLS=$PWD/src/bac_pyseer/.pixi/envs/default/bin/samtools

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
PHYLO=$DATA/klebsiella/phylogeny
PYSEER=$DATA/david/processed/pyseer_iso_source
REF_DIR=$PYSEER/ref
RESOLUTION_DIR=$PYSEER/resolution

# Tier-1 union = every KPSC-human blood/faeces sample (superset of all cohorts).
UNION_CSV=$DATA/david/processed/train_iso_source/blood_faeces/all_samples/kpsc_human/binary_blood_vs_faeces_with_split.csv
RESOLUTION_TSV=$RESOLUTION_DIR/blood_faeces_union_resolution.tsv

echo "=== (1) stage reference FASTA ==="
mkdir -p "$REF_DIR"
if [ ! -s "$REF_DIR/ref.fa" ]; then
    # Any snippy_ncbi sample carries reference/ref.fa = NC_009648 (the common reference).
    SRC_REF=$(find "$PHYLO/snippy_ncbi" -maxdepth 2 -name ref.fa 2>/dev/null | head -1)
    echo "Copying reference from: $SRC_REF"
    cp "$SRC_REF" "$REF_DIR/ref.fa"
    "$SAMTOOLS" faidx "$REF_DIR/ref.fa"
fi
echo "Reference: $REF_DIR/ref.fa"; head -1 "$REF_DIR/ref.fa.fai"

echo "=== (2) resolve Sample -> raw snippy VCF ==="
uv run python src/bac_pyseer/kleb_iso_source/resolve_snippy_paths.py \
    --sample-csv "$UNION_CSV" \
    --out-tsv "$RESOLUTION_TSV"

echo "Done. Resolution TSV: $RESOLUTION_TSV"
echo "Rows: $(($(wc -l < "$RESOLUTION_TSV") - 1))"
