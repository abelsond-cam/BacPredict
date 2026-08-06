#!/bin/bash
#SBATCH --job-name=coding_map
#SBATCH --output=/home/dca36/rds/hpc-work/pyseer_scratch/coding_%A_%a.out
#SBATCH --error=/home/dca36/rds/hpc-work/pyseer_scratch/coding_%A_%a.err
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#
# Classify every invasion-GWAS unitig hit as CDS vs IGR across all carriers — a self-submitting driver.
# REUSES the geNomad job's cached `select` artifacts (id_map / carriers.resolved / hits_submatrix in
# $SELECT_DIR), so it does NOT re-scan the 77 GB matrix. It maps each unitig placement through the
# shared genome_prep.CodingIndex built from the carrier's Bakta GFF3.
#
#   Login node, no PHASE -> ORCHESTRATOR: sbatch align(array) -> combine -> stratify (afterok chain).
#   Login node, SMOKE=1  -> ORCHESTRATOR: a single PHASE=smoke job (K carriers, timed).
#   Inside a job, PHASE  -> WORKER: run that phase of annotate_unitig_coding.py.
#
# Usage (login node):
#   bash src/bac_pyseer/kleb_iso_source/scripts/annotate_unitig_coding.sh          # full chain
#   SMOKE=1 SMOKE_K=30 bash src/bac_pyseer/kleb_iso_source/scripts/annotate_unitig_coding.sh   # smoke first

set -euo pipefail
export PYTHONUNBUFFERED=1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
unset PYTHONPATH PYTHONHOME

REPO=/home/dca36/workspace/BacPredict
ACCT=FLOTO-PROJECT-K-SL2-CPU
cd "$REPO"

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
P=$DATA/david/processed/pyseer_iso_source
PAIR=${PAIR:-blood_faeces}
COHORT=${COHORT:-sampled_country_2_1_all}
GD=$P/$PAIR/$COHORT/gwas_unitig_lmm
SELECT_DIR=${SELECT_DIR:-$GD/mge_mapping}                     # reuse geNomad select artifacts
OUT=${OUT:-$GD/coding_mapping}                                # durable results on project_k
SCRATCH=${SCRATCH:-/home/dca36/rds/hpc-work/coding_mapping_shards/$PAIR}
BAKTA_LOOKUP=${BAKTA_LOOKUP:-$OUT/bakta_gff_lookup.tsv}       # Sample<TAB>path -> Bakta GFF3
STRATA_CSV=${STRATA_CSV:-$DATA/david/processed/train_iso_source/$PAIR/$COHORT/kpsc_human/binary_blood_vs_faeces_with_split.csv}

NSHARDS=${NSHARDS:-8}
SMOKE_K=${SMOKE_K:-30}
mkdir -p "$OUT" "$SCRATCH"

MOD=src/bac_pyseer/kleb_iso_source/annotate_unitig_coding.py
COMMON=(--select-dir "$SELECT_DIR" --out-dir "$OUT" --scratch-dir "$SCRATCH")
py_run () { uv run python "$MOD" "$@"; }

# --- ORCHESTRATOR (login node, no PHASE) ----------------------------------------------------------
if [ -z "${PHASE:-}" ]; then
    echo "orchestrator: PAIR=$PAIR NSHARDS=$NSHARDS SMOKE=${SMOKE:-0}"
    echo "  SELECT_DIR=$SELECT_DIR"; echo "  BAKTA_LOOKUP=$BAKTA_LOOKUP"; echo "  OUT=$OUT"
    if [ "${SMOKE:-0}" = "1" ]; then
        JOB=$(sbatch --parsable --account=$ACCT --partition=icelake --nodes=1 --ntasks=1 \
            --cpus-per-task=8 --mem=48G --time=1:00:00 --job-name="coding_smoke_$PAIR" \
            --export=ALL,PHASE=smoke "$0")
        echo "smoke   : $JOB"; exit 0
    fi
    ALIGN=$(sbatch --parsable --account=$ACCT --partition=icelake --nodes=1 --ntasks=1 \
        --cpus-per-task=4 --mem=32G --time=2:00:00 --array=0-$((NSHARDS-1)) \
        --job-name="coding_align_$PAIR" --export=ALL,PHASE=align "$0")
    echo "align   : $ALIGN  (0-$((NSHARDS-1)))"
    COMBINE=$(sbatch --parsable --account=$ACCT --partition=icelake --nodes=1 --ntasks=1 \
        --cpus-per-task=4 --mem=32G --time=1:00:00 --dependency=afterok:"$ALIGN" \
        --job-name="coding_combine_$PAIR" --export=ALL,PHASE=combine "$0")
    echo "combine : $COMBINE"
    STRAT=$(sbatch --parsable --account=$ACCT --partition=icelake --nodes=1 --ntasks=1 \
        --cpus-per-task=4 --mem=32G --time=1:00:00 --dependency=afterok:"$COMBINE" \
        --job-name="coding_strat_$PAIR" --export=ALL,PHASE=stratify "$0")
    echo "stratify: $STRAT"
    echo "chain submitted: align $ALIGN -> combine $COMBINE -> stratify $STRAT"
    exit 0
fi

# --- WORKER (inside a job, PHASE set) -------------------------------------------------------------
echo "PHASE=$PHASE  job=${SLURM_JOB_ID:-none}  task=${SLURM_ARRAY_TASK_ID:-none}  $(date)"
case "$PHASE" in
align)
    py_run --phase align --bakta-lookup "$BAKTA_LOOKUP" \
        --carrier-shard-index "${SLURM_ARRAY_TASK_ID:?align needs --array}" --n-shards "$NSHARDS" "${COMMON[@]}"
    ;;
combine)
    py_run --phase combine "${COMMON[@]}"
    echo "=== outputs ==="; ls -lh "$OUT"
    ;;
stratify)
    py_run --phase stratify --strata-csv "$STRATA_CSV" "${COMMON[@]}"
    echo "=== stratified ==="; ls -lh "$OUT"/coding_by_*.tsv
    ;;
smoke)
    py_run --phase smoke --bakta-lookup "$BAKTA_LOOKUP" --smoke "$SMOKE_K" "${COMMON[@]}"
    echo "=== smoke outputs ==="; ls -lh "$OUT"
    ;;
*) echo "unknown PHASE=$PHASE"; exit 1 ;;
esac
echo "=== $PHASE done  $(date) ==="
