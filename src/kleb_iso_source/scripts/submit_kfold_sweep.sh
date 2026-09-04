#!/bin/bash
# Submit the k-fold x seed sweep as one independent CHAIN per run, on the free SL3 GPU queue.
#
# Why a chain. A pooled fine-tune runs at a measured 3.1 s/step and needs ~40,000 steps to reach its
# AUROC peak plus 12 non-improving evals, i.e. ~38 h. FLOTO-SL2-GPU (QOS gpu1) allows 36 h but has
# only ~288 h left, against the ~570 h this sweep needs. FLOTO-SL3-GPU is free with 3,000 h available
# and its QOS (gpu2) caps walltime at 12 h -- so each run is submitted as LINKS jobs that resume from
# each other's checkpoints. HF restores optimiser, LR schedule, RNG and the early-stopping counter, so
# a chained run follows the same trajectory as an uninterrupted one.
#
# ⚠ The dependency is `afterany`, not `afterok`, and that is deliberate: every link except the last is
# EXPECTED to end in TIMEOUT, and `afterok` would cancel the rest of the chain the moment the wall did
# its job. (SLURM's array-aware `aftercorr` is `afterok`-flavoured for the same reason, which is why
# these are 15 separate chains rather than 5 chained arrays.) Links after the run finishes see its
# results.json and exit 0 in a minute -- the guard is in train_isolation_source_cohort.sh.
#
# Each chain is independent: one run failing cannot stop the other fourteen.
#
# Usage:
#   bash src/kleb_iso_source/scripts/submit_kfold_sweep.sh                  # all 15 runs, 5 links each
#   TASKS="1 2 3" bash .../submit_kfold_sweep.sh                            # a subset
#   TASKS=0 LINKS=4 AFTER=34910596 bash .../submit_kfold_sweep.sh           # continue a run already going
#   TASKS=5 LINKS=2 TIME=01:00:00 ACCT=FLOTO-SL2-GPU bash .../submit_kfold_sweep.sh   # short resume test
set -euo pipefail

REPO=${REPO:-/home/dca36/workspace/BacPredict}
JOB=$REPO/src/kleb_iso_source/scripts/train_isolation_source_cohort.sh
LOGDIR=${LOGDIR:-/rds/user/dca36/hpc-work/logs}

ACCT=${ACCT:-FLOTO-SL3-GPU}      # free, low priority, 12 h QOS cap
PART=${PART:-ampere}
TIME=${TIME:-12:00:00}
CPUS=${CPUS:-32}
MEM=${MEM:-250G}                 # ampere is capped at 8,000 MB/core, so 32 cores IS 250G
# 5 links x ~12,200 usable steps (12 h minus model load and ~19 evals at 4.6 min) = ~61,000 steps of
# capacity against the ~40,000 a run needs. Spare links are nearly free: once results.json exists the
# job exits in under a minute.
LINKS=${LINKS:-5}
N_FOLDS=${N_FOLDS:-5}
TASKS=${TASKS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14"}
COHORT=${COHORT:-sampled_country_2_1_all}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-models_kfold}
# Job id whose completion the FIRST link waits on -- for grafting more links onto a run already going.
AFTER=${AFTER:-}

mkdir -p "$LOGDIR"
echo "sweep: tasks [$TASKS] x $LINKS links | acct=$ACCT part=$PART time=$TIME | cohort=$COHORT -> $OUTPUT_SUBDIR"
echo

for task in $TASKS; do
  fold=$(( task % N_FOLDS ))
  seed=$(( task / N_FOLDS + 1 ))
  dep=$AFTER
  for link in $(seq 1 "$LINKS"); do
    args=(
      --account="$ACCT" --partition="$PART" --time="$TIME"
      --cpus-per-task="$CPUS" --mem="$MEM" --gres=gpu:1
      # --no-requeue, deliberately. Observed 2026-09-04 on the unitig array: 52 of 64 tasks hit
      # `user_env_retrieval_failed_requeued_held` and sat HELD until released by hand. A held job
      # never *ends*, so an afterany dependency on it never fires and the whole chain freezes --
      # silently, for as long as nobody is watching. Without requeue the same fault ends the link in
      # FAILED instead, afterany fires, and the next link resumes from the last checkpoint. In a
      # chain, requeue buys nothing anyway: the next link already is the retry.
      --no-requeue
      --job-name="kf${task}L${link}"
      --output="$LOGDIR/kfold_t${task}_L${link}_%j.out"
      --error="$LOGDIR/kfold_t${task}_L${link}_%j.err"
      --export="ALL,TASK_ID=$task,N_FOLDS=$N_FOLDS,RESUME=auto,COHORT=$COHORT,OUTPUT_SUBDIR=$OUTPUT_SUBDIR"
    )
    [ -n "$dep" ] && args+=(--dependency="afterany:$dep")
    jid=$(sbatch --parsable "${args[@]}" "$JOB")
    printf 'JOB %-10s kf%-2s link %s/%s (fold %d seed %d) | GPU x1 | mem=%s | cores=%s | wall=%s | %s | %s%s\n' \
      "$jid" "$task" "$link" "$LINKS" "$fold" "$seed" "$MEM" "$CPUS" "$TIME" "$PART" "$ACCT" \
      "${dep:+ | after $dep}"
    dep=$jid
  done
done
