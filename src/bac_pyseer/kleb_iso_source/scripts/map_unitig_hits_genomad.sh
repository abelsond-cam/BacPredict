#!/bin/bash
#SBATCH --job-name=mge_map
#SBATCH --output=/home/dca36/rds/hpc-work/pyseer_scratch/mge_%A_%a.out
#SBATCH --error=/home/dca36/rds/hpc-work/pyseer_scratch/mge_%A_%a.err
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
# Resource directives (partition/cpus/mem/time/array) are set PER PHASE by the orchestrator branch
# (this same script run on the login node with no $PHASE) via sbatch CLI overrides.
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#
# geNomad MGE mapping of ALL invasion unitig hits — ONE self-submitting script (folds the
# orchestrator + worker of the unitig-LMM pattern into a single file).
#
#   Login node, no PHASE  -> ORCHESTRATOR: sbatch select -> align(array) -> combine (afterok chain).
#   Login node, SMOKE=1   -> ORCHESTRATOR: sbatch a single PHASE=smoke job (K carriers, timed).
#   Inside a job, PHASE set-> WORKER: run that phase of map_unitig_hits_genomad.py.
#
#   select : id_map for ALL ~33k hit unitigs; extract the cached hit sub-matrix (one 77 GB pigz|awk
#            pass, reused thereafter); resolve carriers -> assembly paths; probe the geNomad layout.
#   align  : --array shard of the carrier list; stream each carrier's plasmid/virus/assembly through
#            one Aho-Corasick automaton of all unitigs (both strands); classify PLASMID>VIRUS>ASM;
#            attach geNomad taxonomy/MOB; emit per-unitig aggregates + a per-(unitig,carrier) parquet part.
#   combine: sum aggregates -> per-unitig / per-pattern_group / overall tables; assemble the parquet
#            dataset; geNomad spot-check; manifest (ASM-recall, discordances, class distribution).
#   smoke  : select + align(K carriers) + combine inline; prints ASM-recall + sec/genome to validate.
#
# Usage (login node):
#   bash src/bac_pyseer/kleb_iso_source/scripts/map_unitig_hits_genomad.sh            # full chain
#   SMOKE=1 SMOKE_K=30 bash src/bac_pyseer/kleb_iso_source/scripts/map_unitig_hits_genomad.sh   # smoke first
# Knobs (env): NSHARDS (default 8), SMOKE_K (30), THREADS (from cpus).

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
HITS=${HITS:-$GD/blood_vs_faeces_unitig_hits_annotated.tsv}
MATRIX=${MATRIX:-$P/unitigs/$PAIR/unitigs.pyseer.gz}
GENOMAD=${GENOMAD:-$DATA/david/processed/genomad}
METADATA=${METADATA:-$DATA/david/final/metadata_v2_all_samples_and_columns.tsv}
STRATA_CSV=${STRATA_CSV:-$DATA/david/processed/train_iso_source/$PAIR/$COHORT/kpsc_human/binary_blood_vs_faeces_with_split.csv}
ISESCAN_LOOKUP=${ISESCAN_LOOKUP:-$DATA/david/raw/isescan_csv.tsv}   # Sample<TAB>path -> seb SR ISEScan .fa.csv
OUT=${OUT:-$GD/mge_mapping}                                   # durable results on project_k
SCRATCH=${SCRATCH:-/home/dca36/rds/hpc-work/mge_mapping_shards/$PAIR}

NSHARDS=${NSHARDS:-8}
SMOKE_K=${SMOKE_K:-30}
THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:-4}}
mkdir -p "$OUT" "$SCRATCH"

MOD=src/bac_pyseer/kleb_iso_source/map_unitig_hits_genomad.py
py_run () { uv run python "$MOD" "$@"; }

# ---------------------------------------------------------------------------------------------------
# ORCHESTRATOR — login node (no PHASE set): submit the chain (or a single smoke job).
# ---------------------------------------------------------------------------------------------------
if [ -z "${PHASE:-}" ]; then
    echo "orchestrator: PAIR=$PAIR NSHARDS=$NSHARDS SMOKE=${SMOKE:-0}"
    echo "  HITS=$HITS"
    echo "  OUT=$OUT"
    if [ "${SMOKE:-0}" = "1" ]; then
        JOB=$(sbatch --parsable --account=$ACCT --partition=icelake --nodes=1 --ntasks=1 \
            --cpus-per-task=8 --mem=48G --time=2:00:00 --job-name="mge_smoke_$PAIR" \
            --export=ALL,PHASE=smoke "$0")
        echo "smoke   : $JOB"
        exit 0
    fi
    if [ "${STRATIFY:-0}" = "1" ]; then   # re-aggregate the existing parquet by sublineage / clonal group
        JOB=$(sbatch --parsable --account=$ACCT --partition=icelake --nodes=1 --ntasks=1 \
            --cpus-per-task=8 --mem=48G --time=1:00:00 --job-name="mge_strat_$PAIR" \
            --export=ALL,PHASE=stratify "$0")
        echo "stratify: $JOB"
        exit 0
    fi
    SELECT=$(sbatch --parsable --account=$ACCT --partition=icelake --nodes=1 --ntasks=1 \
        --cpus-per-task=8 --mem=32G --time=2:00:00 --job-name="mge_select_$PAIR" \
        --export=ALL,PHASE=select "$0")
    echo "select  : $SELECT"
    ALIGN=$(sbatch --parsable --account=$ACCT --partition=icelake --nodes=1 --ntasks=1 \
        --cpus-per-task=4 --mem=32G --time=2:00:00 --array=0-$((NSHARDS-1)) \
        --dependency=afterok:"$SELECT" --job-name="mge_align_$PAIR" \
        --export=ALL,PHASE=align "$0")
    echo "align   : $ALIGN  (0-$((NSHARDS-1)))"
    COMBINE=$(sbatch --parsable --account=$ACCT --partition=icelake --nodes=1 --ntasks=1 \
        --cpus-per-task=4 --mem=32G --time=1:00:00 --dependency=afterok:"$ALIGN" \
        --job-name="mge_combine_$PAIR" --export=ALL,PHASE=combine "$0")
    echo "combine : $COMBINE"
    echo "chain submitted: select $SELECT -> align $ALIGN -> combine $COMBINE"
    exit 0
fi

# ---------------------------------------------------------------------------------------------------
# WORKER — inside a job ($PHASE set). Exact-substring matching via Aho-Corasick (all-Python under uv;
# no external aligner). Only the one-time matrix scan shells out (pigz/gzip + awk).
# ---------------------------------------------------------------------------------------------------
echo "PHASE=$PHASE  job=${SLURM_JOB_ID:-none}  task=${SLURM_ARRAY_TASK_ID:-none}  $(date)"

COMMON=(--genomad-root "$GENOMAD" --out-dir "$OUT" --scratch-dir "$SCRATCH" --decomp-threads "$THREADS")

case "$PHASE" in
select)
    py_run --phase select --hits-tsv "$HITS" --unitig-matrix "$MATRIX" --metadata "$METADATA" "${COMMON[@]}"
    ;;
align)
    py_run --phase align --isescan-lookup "$ISESCAN_LOOKUP" \
        --carrier-shard-index "${SLURM_ARRAY_TASK_ID:?align needs --array}" --n-shards "$NSHARDS" "${COMMON[@]}"
    ;;
combine)
    py_run --phase combine "${COMMON[@]}"
    echo "=== outputs ==="; ls -lh "$OUT"
    ;;
stratify)
    py_run --phase stratify --strata-csv "$STRATA_CSV" "${COMMON[@]}"
    echo "=== stratified outputs ==="; ls -lh "$OUT"/mge_by_*.tsv
    ;;
smoke)
    py_run --phase smoke --hits-tsv "$HITS" --unitig-matrix "$MATRIX" --metadata "$METADATA" \
        --isescan-lookup "$ISESCAN_LOOKUP" --smoke "$SMOKE_K" "${COMMON[@]}"
    echo "=== smoke outputs ==="; ls -lh "$OUT"
    ;;
*) echo "unknown PHASE=$PHASE"; exit 1 ;;
esac
echo "=== $PHASE done  $(date) ==="
