#!/bin/bash
#SBATCH --job-name=source_hotspot_chisq
#SBATCH --output=source_hotspot_chisq_%j.out
#SBATCH --error=source_hotspot_chisq_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU

# Per-source variant-hotspot enrichment: does an invasion niche (blood / respiratory) carry a larger
# SHARE of its functional (non-synonymous) variant repertoire in a gene than the gut / the other
# niches? Distinct-locus richness per gene per source (clonal-expansion-safe), share-based 2×2 Fisher,
# four consequence subsets (synonymous control / non_syn primary / LoF breakout / all_coding sanity),
# four contrasts (blood|resp vs faeces; blood|resp vs rest). See source_hotspot_chisq.py.
#
# Reads the locus effect map + the per-sample caches (I/O over ~25k small gzips → a job, not login).
# Smoke first: SMOKE=1 caps samples per source (login-node-safe, < 15 min).
# Usage:  sbatch src/bac_pyseer/kleb_iso_source/scripts/run_source_hotspot_chisq.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
unset PYTHONPATH PYTHONHOME
REPO=/home/dca36/workspace/BacPredict
cd "$REPO"

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
TRAIN=$DATA/david/processed/train_iso_source
PYSEER=$DATA/david/processed/pyseer_iso_source
CACHE_DIR=$PYSEER/locus_cache
EFFECT_MAP=$PYSEER/source_hotspot/locus_effect_map.tsv.gz
BF_CSV=$TRAIN/blood_faeces/sampled_country_2_1_all/kpsc_human/binary_blood_vs_faeces_with_split.csv
RF_CSV=$TRAIN/faeces_respiratory/sampled_country_2_1_all/kpsc_human/binary_respiratory_vs_faeces_labels.csv
GENE_ANN=$REPO/src/bac_pyseer/data/combined_poisson_test_variant_hotspots.txt
BF_HITS=$REPO/src/bac_pyseer/docs/visualise/lmm_model/blood_vs_faeces_hits_annotated.tsv
RF_HITS=$REPO/src/bac_pyseer/docs/visualise/faeces_resp_lmm_model/respiratory_vs_faeces_hits_annotated.tsv

OUT=$PYSEER/source_hotspot/chisq
mkdir -p "$OUT"

MAXPG=${MAXPG:-0}
if [ "${SMOKE:-0}" = "1" ]; then MAXPG=${MAXPG:-400}; OUT=$OUT/smoke; mkdir -p "$OUT"; fi

echo "Job ${SLURM_JOB_ID:-login}  Node ${SLURMD_NODENAME:-$(hostname)}  $(date)  max-per-group=$MAXPG"

uv run python src/bac_pyseer/kleb_iso_source/source_hotspot_chisq.py \
    --cache-dir "$CACHE_DIR" --effect-map "$EFFECT_MAP" \
    --bf-csv "$BF_CSV" --rf-csv "$RF_CSV" \
    --gene-annotation "$GENE_ANN" \
    --gwas-hits "blood_vs_faeces=$BF_HITS" "respiratory_vs_faeces=$RF_HITS" \
    --max-per-group "$MAXPG" --out-dir "$OUT"

echo "=== done  $(date) ==="
ls -lh "$OUT"
echo "--- significant_hits.tsv head ---"
{ head -20 "$OUT/significant_hits.tsv"; } || true
