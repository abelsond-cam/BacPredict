#!/bin/bash
#SBATCH --job-name=pyseer_extract
#SBATCH --output=pyseer_extract_%A_%a.out
#SBATCH --error=pyseer_extract_%A_%a.err
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --array=0-39

# Per-sample snippy raw-VCF -> filtered SNP/indel locus cache, fanned out over the
# resolution TSV. Massively over-budgeted on time per project policy (24 h; finishes
# far sooner). Idempotent: --skip-existing means a re-run (or the Tier-2 ~79k run)
# only extracts samples not already cached.
#
# Tier 1 (blood/faeces union, ~21.5k): set RESOLUTION_TSV to the union resolution.
# Tier 2 (~79k kpsc_final_list): re-run setup_and_resolve with --all-kpsc, point
#         RESOLUTION_TSV at that table, widen --array, resubmit. Same cache.
#
# Usage: sbatch src/bac_pyseer/kleb_iso_source/scripts/extract_variants_array.sh

set -euo pipefail
# bcftools comes from the bac_pyseer pixi env (NOT a spack module — that would leak a
# python-3.9 site-packages dir onto PYTHONPATH and break uv's python). unset PYTHONPATH
# belt-and-braces; the conda bcftools finds htslib via RPATH so no module/LD path needed.
export PYTHONUNBUFFERED=1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
unset PYTHONPATH PYTHONHOME
cd /home/dca36/workspace/BacPredict

BCFTOOLS=$PWD/src/bac_pyseer/.pixi/envs/default/bin/bcftools

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
PYSEER=$DATA/david/processed/pyseer_iso_source
REF=$PYSEER/ref/ref.fa
CACHE_DIR=$PYSEER/locus_cache
# Toggle RESOLUTION_TSV (env) to extract a different pair/cohort; the per-sample cache is shared,
# so --skip-existing means already-cached samples (e.g. faeces) are reused and only new ones extracted.
RESOLUTION_TSV=${RESOLUTION_TSV:-$PYSEER/resolution/blood_faeces_union_resolution.tsv}

echo "Job $SLURM_JOB_ID  Array task $SLURM_ARRAY_TASK_ID  Node $SLURMD_NODENAME  $(date)"
echo "bcftools: $BCFTOOLS  $($BCFTOOLS --version | head -1)"

# Chunk by the stable TOTAL resolution-row count (header excluded).
TOTAL=$(($(wc -l < "$RESOLUTION_TSV") - 1))
NTASKS=${SLURM_ARRAY_TASK_COUNT:-40}
CHUNK=$(( TOTAL / NTASKS + 1 ))
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END=$(( (SLURM_ARRAY_TASK_ID + 1) * CHUNK ))
if [ $END -gt $TOTAL ]; then END=$TOTAL; fi
echo "TOTAL=$TOTAL NTASKS=$NTASKS CHUNK=$CHUNK -> [$START:$END)"

uv run python src/bac_pyseer/kleb_iso_source/extract_sample_loci.py \
    --resolution-tsv "$RESOLUTION_TSV" \
    --ref "$REF" \
    --cache-dir "$CACHE_DIR" \
    --start-idx "$START" \
    --end-idx "$END" \
    --min-qual 100 --min-dp 3 \
    --bcftools "$BCFTOOLS" \
    --skip-existing

echo "Array task $SLURM_ARRAY_TASK_ID done  $(date)"
