#!/bin/bash
#SBATCH --job-name=locus_consequence
#SBATCH --output=locus_consequence_%j.out
#SBATCH --error=locus_consequence_%j.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU

# Annotate the variant LOCUS UNIVERSE with SnpEff consequence (synonymous / missense / LoF), once —
# the per-locus input the per-source hotspot Chi-sq needs. Effect is reference-determined, so we
# annotate the union of distinct (POS,REF,ALT) across the cohort caches a single time, REUSING
# snippy's prebuilt SnpEff DB for NC_009648 (the same DB behind Aaron's hotspot table → consistent).
#
# Crawls the ~18-20k per-sample caches (I/O heavy), so it runs as a job, not on the login node.
# Usage: sbatch src/bac_pyseer/kleb_iso_source/scripts/run_locus_consequence.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
unset PYTHONPATH PYTHONHOME
REPO=/home/dca36/workspace/BacPredict
PIXI_MANIFEST=$REPO/src/bac_pyseer/pixi.toml
cd "$REPO"

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
TRAIN=$DATA/david/processed/train_iso_source
PYSEER=$DATA/david/processed/pyseer_iso_source
CACHE_DIR=$PYSEER/locus_cache
BF_CSV=$TRAIN/blood_faeces/sampled_country_2_1_all/kpsc_human/binary_blood_vs_faeces_with_split.csv
RF_CSV=$TRAIN/faeces_respiratory/sampled_country_2_1_all/kpsc_human/binary_respiratory_vs_faeces_labels.csv
# snippy's prebuilt SnpEff DB for NC_009648 (genome name "ref"); any snippy_ncbi sample's reference/
# holds the identical DB. GCF_000009885.1 is the stable fidelity-check sample.
REFD=$DATA/klebsiella/phylogeny/snippy_ncbi/GCF_000009885.1/reference

OUT=$PYSEER/source_hotspot
mkdir -p "$OUT"
VCF=$OUT/locus_universe.vcf
ANN=$OUT/locus_universe.snpeff.vcf.gz
MAP=$OUT/locus_effect_map.tsv.gz

echo "Job $SLURM_JOB_ID  Node $SLURMD_NODENAME  $(date)"

echo "=== (1) build union-loci VCF from the cohort caches ==="
uv run python src/bac_pyseer/kleb_iso_source/annotate_locus_consequence.py build-vcf \
    --cohort-csv "$BF_CSV" "$RF_CSV" --cache-dir "$CACHE_DIR" --out-vcf "$VCF"
[ -s "$VCF" ] || { echo "ERROR: empty locus VCF"; exit 1; }
echo "VCF lines: $(grep -vc '^#' "$VCF")"

echo "=== (2) SnpEff annotate (reuse snippy's NC_009648 DB; snpeff 5.2 via pixi exec) ==="
export _JAVA_OPTIONS="-Xmx16g"
pixi exec -c bioconda -c conda-forge --spec "snpeff=5.2" -- \
    snpEff -noStats -c "$REFD/snpeff.config" -dataDir "$REFD" ref "$VCF" | gzip > "$ANN"
[ -s "$ANN" ] || { echo "ERROR: empty SnpEff output"; exit 1; }
echo "annotated: $(zcat "$ANN" | grep -vc '^#') variants"

echo "=== (3) parse ANN -> (POS,REF,ALT)->effect/impact/class/gene map ==="
uv run python src/bac_pyseer/kleb_iso_source/annotate_locus_consequence.py parse \
    --ann-vcf "$ANN" --out-tsv "$MAP"
[ -s "$MAP" ] || { echo "ERROR: empty effect map"; exit 1; }

echo "=== done  $(date) ==="
ls -lh "$OUT"
echo "effect-class distribution:"
zcat "$MAP" | tail -n +2 | cut -f6 | sort | uniq -c | sort -rn
echo "rm the uncompressed VCF (keep the .gz annotated + map)"; rm -f "$VCF"
