#!/bin/bash
# Orchestrate the SHARDED unitig LMM GWAS — submit prep -> array -> combine with afterok deps so
# SLURM sequences them (no polling). RUN ON THE LOGIN NODE: it only calls sbatch.
#
#   prep    (himem, ${PREP_CPUS}c/${PREP_MEM}/6h)  : clusters + prime LMM cache + split matrix -> N chunks
#   array   (himem, ${CPU}c/${ARRAY_MEM}/24h)      : --array=0..N-1, pyseer per chunk (--load-lmm)
#   combine (himem, ${COMB_CPUS}c/${COMB_MEM}/3h)  : cat .assoc + union patterns -> pyseer_postprocess
#
# ⚠ SHARED INFRASTRUCTURE — the invasion GWAS and both AMR arms drive this script. Every knob below
# defaults to the value that was previously hardcoded, so existing callers are byte-for-byte unchanged.
#
# Memory (calibrated 2026-06-24, cpu=8, lineage on): 10k/50k/100k unitigs/shard peak 9/21/26 GB — memory
# is sub-linear in shard size. Default NSHARDS=64 (~100k unitigs/shard for this cohort) ⇒ ~26 GB peak, so
# --mem=128G on the array is a generous ~5x margin. (A single-process run OOM'd at 134 GB: peak ≈ cpu×n²;
# an earlier 16-shard run OOM'd because the array was under-requested at 48 GB — never under-call memory.)
#
# ⚠ That ~5x margin is a fact about THAT cohort (n=13,602), and the calibration says nothing about how
# peak scales with n. At TB's rifampin (n=28,508) the same 128G is somewhere between comfortable and
# insufficient, so this script now GATES on `bac_pyseer.ast_gwas.shard_memory` before submitting
# anything, and CANARY=1 runs one shard first so the answer costs ~6 min instead of a whole array.
#
# Sharding is mathematically exact here: the n×n LMM rotation + null h^2 are computed ONCE (prep,
# --save-lmm) and reused verbatim by every shard (--load-lmm); per-unitig tests are independent; only
# the Bonferroni pattern count is combined (a set union). See unitig_lmm_sharded_job.sh.
#
# Usage:
#   blood/faeces (default; cache already exists from the earlier run):
#     bash src/bac_pyseer/kleb_iso_source/scripts/run_unitig_lmm_sharded.sh
#   one shard first, to measure before committing the array:
#     CANARY=1 ... bash .../run_unitig_lmm_sharded.sh
#   faeces/respiratory (primes its own cache in prep):
#     PAIR=faeces_respiratory COHORT=sampled_country_2_1_all LABEL_COL=respiratory_vs_faeces_label \
#       OUT_STEM=respiratory_vs_faeces_unitig POS_LABEL='respiratory (invasion)' \
#       PAIR_TITLE='faeces vs respiratory (unitigs)' \
#       COHORT_CSV=$TRAIN/faeces_respiratory/sampled_country_2_1_all/kpsc_human/binary_respiratory_vs_faeces_labels.csv \
#       bash src/bac_pyseer/kleb_iso_source/scripts/run_unitig_lmm_sharded.sh

set -euo pipefail
REPO=${REPO:-/home/dca36/workspace/BacPredict}
JOB=$REPO/src/bac_pyseer/kleb_iso_source/scripts/unitig_lmm_sharded_job.sh
# Submission knobs. run_drug.sh already hands these down; before the AMR fan-out this script
# hardcoded them and silently dropped what it was given, which worked only because the values
# happened to agree. Defaults are the previous hardcoded values, so existing callers are unchanged.
ACCT=${ACCT:-FLOTO-PROJECT-K-SL2-CPU}
PART=${PART:-icelake-himem}
# QOS and LOGDIR were passed down by run_drug.sh and then dropped on the floor -- nothing consumed
# them, so every log landed at the job script's hardcoded #SBATCH --output regardless, and the
# careful QOS= / ${QOS-normal} distinction in run_drug.sh was dead code. Both are honoured now.
QOS=${QOS:-}
LOGDIR=${LOGDIR:-}
SB_COMMON=(--account="$ACCT" --partition="$PART" --nodes=1 --ntasks=1)
if [ -n "$QOS" ]; then SB_COMMON+=(--qos="$QOS"); fi
if [ -n "$LOGDIR" ]; then mkdir -p "$LOGDIR"; fi
# %j for the single jobs, %A_%a for the array -- an array writing to %j collapses every task's log
# into one file, which is how a failed shard becomes invisible.
log_flags () { if [ -n "$LOGDIR" ]; then echo "--output=$LOGDIR/%x-$1.out --error=$LOGDIR/%x-$1.out"; fi; }
# Walltime per phase. The array's 24 h was the blocker for a 22-drug fan-out: 22 x 64 shards x 8 cpu
# x 24 h *reserves* 270,336 core-h against ~106,371 available, so every chain sits on
# AssocGrpCPUMinutesLimit and none of them start. Reservation is what gates scheduling; billing is on
# elapsed. Measured shard max is 6.2 min, so 2 h is ~19x headroom and reserves ~22,500 core-h.
PREP_TIME=${PREP_TIME:-6:00:00}
ARRAY_TIME=${ARRAY_TIME:-24:00:00}
COMB_TIME=${COMB_TIME:-3:00:00}
# Per-phase cpu/mem. Previously hardcoded; TB needs them raised, and the prep pair in particular was
# sized for n≈13.6k and has never run against a ~6 GB LMM cache.
PREP_CPUS=${PREP_CPUS:-16};  PREP_MEM=${PREP_MEM:-96G}
ARRAY_MEM=${ARRAY_MEM:-128G}
COMB_CPUS=${COMB_CPUS:-8};   COMB_MEM=${COMB_MEM:-96G}
TRAIN=${TRAIN:-/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_iso_source}

# Cohort knobs (defaults = blood/faeces) — exported so --export=ALL carries them to every phase.
export PAIR=${PAIR:-blood_faeces}
export COHORT=${COHORT:-sampled_country_2_1_all}
export LABEL_COL=${LABEL_COL:-blood_vs_faeces_label}
export OUT_STEM=${OUT_STEM:-blood_vs_faeces_unitig}
export POS_LABEL="${POS_LABEL:-blood (invasion)}"
export NEG_LABEL="${NEG_LABEL:-faeces}"   # AMR passes "susceptible"; was hardcoded in the job script
export PAIR_TITLE="${PAIR_TITLE:-blood vs faeces (unitigs)}"
export COHORT_CSV=${COHORT_CSV:-$TRAIN/$PAIR/$COHORT/kpsc_human/binary_blood_vs_faeces_with_split.csv}
export NSHARDS=${NSHARDS:-64}   # ~100k unitigs/shard for the ~6.3M-unitig matrix (calibrated: ~26 GB peak)
export CPU=${CPU:-8}
# pyseer's worker count, decoupled from the allocation. They were the same variable, so buying memory
# with cores also multiplied whatever pyseer holds per worker — no way to ask for a big node and run
# few workers on it, which is the one lever the big TB drugs have.
export PYSEER_CPU=${PYSEER_CPU:-$CPU}

echo "PAIR=$PAIR  COHORT=$COHORT  NSHARDS=$NSHARDS  cores/shard=$CPU  pyseer --cpu=$PYSEER_CPU  out_stem=$OUT_STEM"
echo "acct=$ACCT  part=$PART  prep=$PREP_CPUS c/$PREP_MEM  array=$CPU c/$ARRAY_MEM  combine=$COMB_CPUS c/$COMB_MEM"
echo "walltime  prep=$PREP_TIME  array=$ARRAY_TIME  combine=$COMB_TIME"

# ---------------------------------------------------------------- memory gate + honest reservation
# MEM_GATE: enforce (default) | warn | off. `enforce` refuses to submit when the estimate does not
# clear ARRAY_MEM by MEM_MARGIN. Kp and the invasion cohorts clear it by an order of magnitude, so
# this changes nothing for them; it exists for the cohorts nobody has measured.
MEM_GATE=${MEM_GATE:-enforce}
MEM_MARGIN=${MEM_MARGIN:-1.25}
CALIBRATION=${CALIBRATION:-$REPO/src/bac_pyseer/ast_gwas/shard_memory_observations.json}
# n = phenotyped genomes actually entering the LMM. PHENO is set by run_drug.sh; header excluded.
GWAS_N=${GWAS_N:-0}
if [ "$GWAS_N" = "0" ] && [ -n "${PHENO:-}" ] && [ -s "${PHENO:-}" ]; then
    GWAS_N=$(( $(wc -l < "$PHENO") - 1 ))
fi
# Unitigs per shard drives the (well-determined) shard-size term. TOTAL_UNITIGS is known from the
# GGCAT build; without it we assume the calibrated 100k and say so, rather than inventing a number.
if [ -n "${TOTAL_UNITIGS:-}" ]; then
    UNITIGS_PER_SHARD=${UNITIGS_PER_SHARD:-$(( TOTAL_UNITIGS / NSHARDS ))}
    _u_src="TOTAL_UNITIGS=$TOTAL_UNITIGS / NSHARDS=$NSHARDS"
else
    UNITIGS_PER_SHARD=${UNITIGS_PER_SHARD:-100000}
    _u_src="assumed (set TOTAL_UNITIGS for a real figure)"
fi

_mem_num () { printf '%s' "${1%[Gg]}"; }   # "128G" -> "128"
ARRAY_MEM_GB=$(_mem_num "$ARRAY_MEM")

if [ "$MEM_GATE" = "off" ]; then
    echo "memory gate: OFF (MEM_GATE=off)"
elif [ "$GWAS_N" -le 0 ]; then
    echo "memory gate: SKIPPED — cannot determine n (set GWAS_N, or PHENO to the phenotype TSV)" >&2
    [ "$MEM_GATE" != "enforce" ] || { echo "MEM_GATE=enforce and n is unknown — refusing to submit blind." >&2; exit 1; }
else
    echo "memory gate: n=$GWAS_N  unitigs/shard=$UNITIGS_PER_SHARD  [$_u_src]"
    set +e
    ( cd "$REPO" && uv run python -m bac_pyseer.ast_gwas.shard_memory \
        --n "$GWAS_N" --cpu "$PYSEER_CPU" --unitigs-per-shard "$UNITIGS_PER_SHARD" \
        --mem-gb "$ARRAY_MEM_GB" --margin "$MEM_MARGIN" --calibration "$CALIBRATION" )
    _gate=$?
    set -e
    if [ "$_gate" -ne 0 ] && [ "$MEM_GATE" = "enforce" ]; then
        echo "" >&2
        echo "REFUSING TO SUBMIT. Raise ARRAY_MEM, lower PYSEER_CPU, or raise NSHARDS (smaller shards)." >&2
        echo "Run one shard first with CANARY=1 to replace the estimate with a measurement, or set" >&2
        echo "MEM_GATE=warn to proceed anyway. This check costs nothing; an OOM costs the whole array." >&2
        exit 1
    fi
fi

# Reservation, not spend — but it is what SLURM checks against AssocGrpCPUMinutesLimit, so print it.
# Cores are max(requested, what --mem forces): CSD3 sells memory by the core, so --mem=128G at
# --cpus-per-task=8 allocates 20 and reserves against 20. Printing the requested 8 understated it 2.5x.
BILLED_CORES=$( (cd "$REPO" && uv run python -c "
import sys; sys.path.insert(0,'src')
from bac_pyseer.ast_gwas.shard_memory import cores_for_mem
print(max($CPU, cores_for_mem($ARRAY_MEM_GB)))") 2>/dev/null || echo "$CPU")
IFS=: read -r _h _m _s <<<"$ARRAY_TIME"
echo "array reserves ~$(( NSHARDS * BILLED_CORES * (10#$_h * 3600 + 10#$_m * 60 + 10#$_s) / 3600 )) core-h" \
     "(${BILLED_CORES} cores/shard billed; ${CPU} requested)"

# ----------------------------------------------------------------------------------- submit
# shellcheck disable=SC2046  # log_flags is deliberately word-split: it is empty when LOGDIR is unset
PREP=$(sbatch --parsable "${SB_COMMON[@]}" $(log_flags '%j') \
    --cpus-per-task="$PREP_CPUS" --mem="$PREP_MEM" --time=$PREP_TIME --job-name="uprep_$PAIR" \
    --export=ALL,PHASE=prep "$JOB")
echo "prep    : $PREP"

# CANARY=1 submits ONE shard and stops. The array's memory behaviour at an unmeasured cohort size is
# then a ~6-minute question rather than a whole-array one, and the answer feeds back into the
# calibration file so the next drug is sized from data.
ARRAY_SPEC=$([ -n "${CANARY:-}" ] && echo "0" || echo "0-$((NSHARDS-1))")
# shellcheck disable=SC2046
ARRAY=$(sbatch --parsable "${SB_COMMON[@]}" $(log_flags '%A_%a') \
    --cpus-per-task="$CPU" --mem="$ARRAY_MEM" --time=$ARRAY_TIME --array="$ARRAY_SPEC" \
    --dependency=afterok:"$PREP" --job-name="utask_$PAIR" \
    --export=ALL,PHASE=task "$JOB")
echo "array   : $ARRAY  ($ARRAY_SPEC)"

if [ -n "${CANARY:-}" ]; then
    cat <<EOF
canary submitted: prep $PREP -> shard 0 only. No combine (the .assoc would be 1/$NSHARDS of the GWAS).

When it finishes, record what it actually used and then run the full array:
  sacct -j ${ARRAY}_0 --format=JobID,State,MaxRSS,Elapsed
  uv run python -m bac_pyseer.ast_gwas.shard_memory --n $GWAS_N --cpu $PYSEER_CPU \\
      --unitigs-per-shard $UNITIGS_PER_SHARD --calibration $CALIBRATION \\
      --record-max-rss-gb <MaxRSS in GB> --source "${PAIR} ${ARRAY}_0"
  <same env, without CANARY=1> bash \$0
Shard 0's chunk_00.assoc is kept, so the full array re-runs it; that is one shard, not a set.
EOF
    exit 0
fi

# shellcheck disable=SC2046
COMBINE=$(sbatch --parsable "${SB_COMMON[@]}" $(log_flags '%j') \
    --cpus-per-task="$COMB_CPUS" --mem="$COMB_MEM" --time=$COMB_TIME --dependency=afterok:"$ARRAY" \
    --job-name="ucomb_$PAIR" --export=ALL,PHASE=combine "$JOB")
echo "combine : $COMBINE"
echo "chain submitted: prep $PREP -> array $ARRAY -> combine $COMBINE"
