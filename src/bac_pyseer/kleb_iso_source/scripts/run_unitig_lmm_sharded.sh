#!/bin/bash
# Orchestrate the SHARDED unitig LMM GWAS — submit prep -> array -> combine with afterok deps so
# SLURM sequences them (no polling). RUN ON THE LOGIN NODE: it only calls sbatch.
#
#   prep    (himem, 16c/96G/6h)        : sublineage clusters + prime LMM cache + split matrix -> N chunks
#   array   (himem, ${CPU}c/128G/24h)  : --array=0..N-1, pyseer per chunk reusing the cache (--load-lmm)
#   combine (himem, 8c/96G/3h)         : cat .assoc + union patterns -> pyseer_postprocess --feature-mode unitigs
#
# Memory (calibrated 2026-06-24, cpu=8, lineage on): 10k/50k/100k unitigs/shard peak 9/21/26 GB — memory
# is sub-linear in shard size. Default NSHARDS=64 (~100k unitigs/shard for this cohort) ⇒ ~26 GB peak, so
# --mem=128G on the array is a generous ~5x margin. (A single-process run OOM'd at 134 GB: peak ≈ cpu×n²;
# an earlier 16-shard run OOM'd because the array was under-requested at 48 GB — never under-call memory.)
#
# Sharding is mathematically exact here: the n×n LMM rotation + null h^2 are computed ONCE (prep,
# --save-lmm) and reused verbatim by every shard (--load-lmm); per-unitig tests are independent; only
# the Bonferroni pattern count is combined (a set union). See unitig_lmm_sharded_job.sh.
#
# Usage:
#   blood/faeces (default; cache already exists from the earlier run):
#     bash src/bac_pyseer/kleb_iso_source/scripts/run_unitig_lmm_sharded.sh
#   faeces/respiratory (primes its own cache in prep):
#     PAIR=faeces_respiratory COHORT=sampled_country_2_1_all LABEL_COL=respiratory_vs_faeces_label \
#       OUT_STEM=respiratory_vs_faeces_unitig POS_LABEL='respiratory (invasion)' \
#       PAIR_TITLE='faeces vs respiratory (unitigs)' \
#       COHORT_CSV=$TRAIN/faeces_respiratory/sampled_country_2_1_all/kpsc_human/binary_respiratory_vs_faeces_labels.csv \
#       bash src/bac_pyseer/kleb_iso_source/scripts/run_unitig_lmm_sharded.sh

set -euo pipefail
REPO=/home/dca36/workspace/BacPredict
JOB=$REPO/src/bac_pyseer/kleb_iso_source/scripts/unitig_lmm_sharded_job.sh
# Submission knobs. run_drug.sh already hands these down; before the AMR fan-out this script
# hardcoded them and silently dropped what it was given, which worked only because the values
# happened to agree. Defaults are the previous hardcoded values, so existing callers are unchanged.
ACCT=${ACCT:-FLOTO-PROJECT-K-SL2-CPU}
PART=${PART:-icelake-himem}
# Walltime per phase. The array's 24 h was the blocker for a 22-drug fan-out: 22 x 64 shards x 8 cpu
# x 24 h *reserves* 270,336 core-h against ~106,371 available, so every chain sits on
# AssocGrpCPUMinutesLimit and none of them start. Reservation is what gates scheduling; billing is on
# elapsed. Measured shard max is 6.2 min, so 2 h is ~19x headroom and reserves ~22,500 core-h.
PREP_TIME=${PREP_TIME:-6:00:00}
ARRAY_TIME=${ARRAY_TIME:-24:00:00}
COMB_TIME=${COMB_TIME:-3:00:00}
TRAIN=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_iso_source

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

echo "PAIR=$PAIR  COHORT=$COHORT  NSHARDS=$NSHARDS  CPU/shard=$CPU  out_stem=$OUT_STEM"
echo "acct=$ACCT  part=$PART  prep=$PREP_TIME  array=$ARRAY_TIME  combine=$COMB_TIME"
# Reservation, not spend — but it is what SLURM checks against AssocGrpCPUMinutesLimit, so print it.
IFS=: read -r _h _m _s <<<"$ARRAY_TIME"
echo "array reserves ~$(( NSHARDS * CPU * (10#$_h * 3600 + 10#$_m * 60 + 10#$_s) / 3600 )) core-h"

PREP=$(sbatch --parsable --account=$ACCT --partition=$PART --nodes=1 --ntasks=1 \
    --cpus-per-task=16 --mem=96G --time=$PREP_TIME --job-name="uprep_$PAIR" \
    --export=ALL,PHASE=prep "$JOB")
echo "prep    : $PREP"

ARRAY=$(sbatch --parsable --account=$ACCT --partition=$PART --nodes=1 --ntasks=1 \
    --cpus-per-task="$CPU" --mem=128G --time=$ARRAY_TIME --array=0-$((NSHARDS-1)) \
    --dependency=afterok:"$PREP" --job-name="utask_$PAIR" \
    --export=ALL,PHASE=task "$JOB")
echo "array   : $ARRAY  (0-$((NSHARDS-1)))"

COMBINE=$(sbatch --parsable --account=$ACCT --partition=$PART --nodes=1 --ntasks=1 \
    --cpus-per-task=8 --mem=96G --time=$COMB_TIME --dependency=afterok:"$ARRAY" \
    --job-name="ucomb_$PAIR" --export=ALL,PHASE=combine "$JOB")
echo "combine : $COMBINE"
echo "chain submitted: prep $PREP -> array $ARRAY -> combine $COMBINE"
