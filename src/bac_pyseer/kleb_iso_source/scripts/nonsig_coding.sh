#!/bin/bash
#SBATCH --job-name=nonsig_coding
#SBATCH --output=/home/dca36/rds/hpc-work/pyseer_scratch/nonsig_%A_%a.out
#SBATCH --error=/home/dca36/rds/hpc-work/pyseer_scratch/nonsig_%A_%a.err
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#
# The NON-SIGNIFICANT control for the IGR-coverage analysis (three-way: uniform null / non-sig / sig).
# Self-submitting: sample af-matched non-sig unitigs → generic unitig_placement SELECT (no geNomad name)
# → annotate_unitig_coding align/combine. Writes to a SEPARATE coding_mapping_nonsig/ dir — never
# touches the significant coding_mapping/.
#
#   Login node, no PHASE -> ORCHESTRATOR: sbatch selectnonsig -> align(array) -> combine (afterok chain).
#   Inside a job, PHASE  -> WORKER.
#
# Usage: bash src/bac_pyseer/kleb_iso_source/scripts/nonsig_coding.sh

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
ASSOC=${ASSOC:-$GD/blood_vs_faeces_unitig.assoc}
HITS=${HITS:-$GD/blood_vs_faeces_unitig_hits_annotated.tsv}
MATRIX=${MATRIX:-$P/unitigs/$PAIR/unitigs.pyseer.gz}
METADATA=${METADATA:-$DATA/david/final/metadata_v2_all_samples_and_columns.tsv}
BAKTA_LOOKUP=${BAKTA_LOOKUP:-$GD/coding_mapping/bakta_gff_lookup.tsv}
OUT=${OUT:-$GD/coding_mapping_nonsig}                         # separate — preserve significant results
SCRATCH=${SCRATCH:-/home/dca36/rds/hpc-work/coding_nonsig_shards/$PAIR}
NONSIG_HITS=$OUT/nonsig_hits.tsv
N_TARGET=${N_TARGET:-100000}
NSHARDS=${NSHARDS:-8}
mkdir -p "$OUT" "$SCRATCH"

SAMP=src/bac_pyseer/kleb_iso_source/sample_nonsig_unitigs.py
SEL=src/bac_pyseer/kleb_iso_source/unitig_placement.py
COD=src/bac_pyseer/kleb_iso_source/annotate_unitig_coding.py
COMMON=(--select-dir "$OUT" --out-dir "$OUT" --scratch-dir "$SCRATCH")

if [ -z "${PHASE:-}" ]; then
    echo "orchestrator: nonsig control  N_TARGET=$N_TARGET NSHARDS=$NSHARDS  OUT=$OUT"
    SELJOB=$(sbatch --parsable --account=$ACCT --partition=icelake --nodes=1 --ntasks=1 \
        --cpus-per-task=8 --mem=48G --time=2:00:00 --job-name="nonsig_select_$PAIR" \
        --export=ALL,PHASE=selectnonsig "$0")
    echo "select  : $SELJOB"
    ALIGN=$(sbatch --parsable --account=$ACCT --partition=icelake --nodes=1 --ntasks=1 \
        --cpus-per-task=4 --mem=32G --time=2:00:00 --array=0-$((NSHARDS-1)) \
        --dependency=afterok:"$SELJOB" --job-name="nonsig_align_$PAIR" --export=ALL,PHASE=align "$0")
    echo "align   : $ALIGN  (0-$((NSHARDS-1)))"
    COMBINE=$(sbatch --parsable --account=$ACCT --partition=icelake --nodes=1 --ntasks=1 \
        --cpus-per-task=4 --mem=32G --time=1:00:00 --dependency=afterok:"$ALIGN" \
        --job-name="nonsig_combine_$PAIR" --export=ALL,PHASE=combine "$0")
    echo "combine : $COMBINE"
    echo "chain: select $SELJOB -> align $ALIGN -> combine $COMBINE"
    exit 0
fi

echo "PHASE=$PHASE  job=${SLURM_JOB_ID:-none}  task=${SLURM_ARRAY_TASK_ID:-none}  $(date)"
THREADS=${SLURM_CPUS_PER_TASK:-4}
case "$PHASE" in
selectnonsig)
    uv run python "$SAMP" --assoc "$ASSOC" --hits-tsv "$HITS" --n-target "$N_TARGET" --out "$NONSIG_HITS"
    uv run python "$SEL" --phase select --hits-tsv "$NONSIG_HITS" --unitig-matrix "$MATRIX" \
        --metadata "$METADATA" --out-dir "$OUT" --scratch-dir "$SCRATCH" --decomp-threads "$THREADS"
    ;;
align)
    uv run python "$COD" --phase align --bakta-lookup "$BAKTA_LOOKUP" \
        --carrier-shard-index "${SLURM_ARRAY_TASK_ID:?align needs --array}" --n-shards "$NSHARDS" "${COMMON[@]}"
    ;;
combine)
    uv run python "$COD" --phase combine "${COMMON[@]}"
    echo "=== nonsig outputs ==="; ls -lh "$OUT"
    ;;
*) echo "unknown PHASE=$PHASE"; exit 1 ;;
esac
echo "=== $PHASE done  $(date) ==="
