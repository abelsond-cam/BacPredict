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
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#
# geNomad MGE mapping of invasion unitig hits — ONE self-submitting script (folds the
# orchestrator + worker of the unitig-LMM pattern into a single file).
#
#   Login node, no PHASE  -> ORCHESTRATOR: sbatch select -> align(array) -> combine (afterok chain).
#   Login node, SMOKE=1   -> ORCHESTRATOR: sbatch a single PHASE=smoke job (few carriers, timed).
#   Inside a job, PHASE set-> WORKER: run that phase of map_unitig_hits_genomad.py.
#
#   select : pick top-N invasion pattern_groups -> query FASTA; scan the 6.28M-row unitig matrix once
#            for carrier sets; resolve carriers -> assembly paths; probe the geNomad layout.
#   align  : --array shard of the carrier list; build each carrier's tagged plasmid+virus+assembly
#            target, one minimap2 over all unitigs, classify -> per-shard long TSV in scratch.
#   combine: concat shards -> master long TSV + per-group summary + geNomad spot-check + Phase-2 estimate.
#   smoke  : select + align(K carriers of the top group) + combine inline; prints sec/genome & ASM recall.
#
# Usage (login node):
#   bash src/bac_pyseer/kleb_iso_source/scripts/map_unitig_hits_genomad.sh            # full chain
#   SMOKE=1 bash src/bac_pyseer/kleb_iso_source/scripts/map_unitig_hits_genomad.sh    # smoke first (recommended)
# Knobs (env): NSHARDS (default 16), TOP_N (3), MIN_AF (0.05), SMOKE_K (5), THREADS (2).

set -euo pipefail
export PYTHONUNBUFFERED=1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
unset PYTHONPATH PYTHONHOME

REPO=/home/dca36/workspace/BacPredict
PIXI_MANIFEST=$REPO/src/bac_pyseer/pixi.toml
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
OUT=${OUT:-$GD/mge_mapping}                                   # durable results on project_k
SCRATCH=${SCRATCH:-/home/dca36/rds/hpc-work/mge_mapping_shards/$PAIR}

NSHARDS=${NSHARDS:-16}
TOP_N=${TOP_N:-3}
MIN_AF=${MIN_AF:-0.05}
SMOKE_K=${SMOKE_K:-5}
THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:-2}}
mkdir -p "$OUT" "$SCRATCH"

MOD=src/bac_pyseer/kleb_iso_source/map_unitig_hits_genomad.py
py_run () { uv run python "$MOD" "$@"; }

# ---------------------------------------------------------------------------------------------------
# ORCHESTRATOR — login node (no PHASE set): submit the chain (or a single smoke job).
# ---------------------------------------------------------------------------------------------------
if [ -z "${PHASE:-}" ]; then
    echo "orchestrator: PAIR=$PAIR NSHARDS=$NSHARDS TOP_N=$TOP_N MIN_AF=$MIN_AF SMOKE=${SMOKE:-0}"
    echo "  HITS=$HITS"
    echo "  OUT=$OUT"
    if [ "${SMOKE:-0}" = "1" ]; then
        JOB=$(sbatch --parsable --account=$ACCT --partition=icelake --nodes=1 --ntasks=1 \
            --cpus-per-task=8 --mem=32G --time=2:00:00 --job-name="mge_smoke_$PAIR" \
            --export=ALL,PHASE=smoke "$0")
        echo "smoke   : $JOB"
        exit 0
    fi
    SELECT=$(sbatch --parsable --account=$ACCT --partition=icelake --nodes=1 --ntasks=1 \
        --cpus-per-task=8 --mem=32G --time=2:00:00 --job-name="mge_select_$PAIR" \
        --export=ALL,PHASE=select "$0")
    echo "select  : $SELECT"
    ALIGN=$(sbatch --parsable --account=$ACCT --partition=icelake --nodes=1 --ntasks=1 \
        --cpus-per-task="$THREADS" --mem=16G --time=4:00:00 --array=0-$((NSHARDS-1)) \
        --dependency=afterok:"$SELECT" --job-name="mge_align_$PAIR" \
        --export=ALL,PHASE=align "$0")
    echo "align   : $ALIGN  (0-$((NSHARDS-1)))"
    COMBINE=$(sbatch --parsable --account=$ACCT --partition=icelake --nodes=1 --ntasks=1 \
        --cpus-per-task=4 --mem=16G --time=1:00:00 --dependency=afterok:"$ALIGN" \
        --job-name="mge_combine_$PAIR" --export=ALL,PHASE=combine "$0")
    echo "combine : $COMBINE"
    echo "chain submitted: select $SELECT -> align $ALIGN -> combine $COMBINE"
    exit 0
fi

# ---------------------------------------------------------------------------------------------------
# WORKER — inside a job ($PHASE set). Unitigs are exact substrings of their carriers' assemblies, so
# classification is exact-substring matching (no aligner); only the one-time matrix scan uses a binary.
# ---------------------------------------------------------------------------------------------------
echo "PHASE=$PHASE  job=${SLURM_JOB_ID:-none}  task=${SLURM_ARRAY_TASK_ID:-none}  $(date)"

COMMON=(--genomad-root "$GENOMAD" --out-dir "$OUT" --scratch-dir "$SCRATCH" \
        --top-n-groups "$TOP_N" --min-af "$MIN_AF" --decomp-threads "$THREADS")

case "$PHASE" in
select)
    py_run --phase select --hits-tsv "$HITS" --unitig-matrix "$MATRIX" --metadata "$METADATA" "${COMMON[@]}"
    ;;
align)
    py_run --phase align \
        --carrier-shard-index "${SLURM_ARRAY_TASK_ID:?align needs --array}" --n-shards "$NSHARDS" "${COMMON[@]}"
    ;;
combine)
    py_run --phase combine "${COMMON[@]}"
    echo "=== outputs ==="; ls -lh "$OUT"
    ;;
smoke)
    py_run --phase smoke --hits-tsv "$HITS" --unitig-matrix "$MATRIX" --metadata "$METADATA" \
        --smoke "$SMOKE_K" "${COMMON[@]}"
    echo "=== smoke outputs ==="; ls -lh "$OUT"
    ;;
*) echo "unknown PHASE=$PHASE"; exit 1 ;;
esac
echo "=== $PHASE done  $(date) ==="
