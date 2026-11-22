#!/usr/bin/env bash
# Bakta 1.9.4 annotation of a second cohort's genomes (CABBAGE v2, PROJECT_STATE §3.7), leaving every genome an
# earlier cohort built to that cohort's own annotation. Run it through a cohort wrapper, from a CSD3 login node:
#
#   <wrapper> plan              bakta/built_genomes.csv (every genome built here so far, from the build shards) and
#                               bakta/bakrep_roster.tsv (the ATB-built ones BakRep's own export lists). Re-run it, then
#                               ingest, as more genomes are built
#   <wrapper> ingest            CPU job: BakRep's GFF3 for each roster genome, accepted only on our contigs base for
#                               base; every outcome in bakta/ingest_status.tsv. Resumable
#   <wrapper> submit [all|planned|refused]
#                               CPU array: our Bakta on the COHORT genomes nothing else annotates (a new, dated,
#                               read-only run roster). planned needs no ingest status, so it can run while ingest does
#   <wrapper> collate           CPU job: exactly one current annotation per COHORT genome → genomes/bakta_annotation.csv
#                               for embedding_input.csv, bakta_annotation_<COHORT file stem>.csv for any other cohort
#   <wrapper> annotated         plan again, then genomes/preqc_annotated.csv: the genomes built here whose annotation has
#                               finished, a COHORT to collate (LIVE_RUNS_OK=1) and embed before QC exists
#   <wrapper> status
#
# The §3.6 annotation (apps/entero/scripts/annotate_entero_genomes.sh, after apps/pa/scripts/annotate_pa_genomes.sh;
# mechanics in apps/pa/CLAUDE.md *Annotation*): BakRep's GFF3 where BakRep holds the genome on our ATB contigs, our run
# of the same release, flags, tool env and database otherwise (submit refuses unless the env's fingerprint is the one
# §3.6's PsA runs recorded), all into the shared store $ROOT/raw/bakta keyed by BioSample. What differs, and why:
# - BakRep coverage is read from BakRep's export itself (BAKREP_EXPORT). §3.6's in_bakrep_ids.txt is §3.6's cohorts
#   joined to it, so it lists no genome only this cohort has.
# - A reused genome (reused_genomes.csv: an earlier cohort's manifest names it) belongs to its owner. It is never on
#   this cohort's ingest roster, and goes to our run only if no annotation record of it exists at all; otherwise the
#   owner's record stands, and collate checks it is current for the contigs the cohort holds. While any owner's Bakta
#   job is live (OWNER_BAKTA_JOBS), which could be writing the same genome's files, submit leaves reused genomes off.
# - Plan, the run roster and status are reuse_built_genomes.py subcommands (bakta-plan, bakta-roster, bakta-status).
#
# Configuration (exported by the wrapper):
#   ROOT, T            the AMR root; the cohort's folder (genomes/ under it, as the build stage left it)
#   JOB                job-name prefix: $JOB-bakta-{ingest,all,planned,refused,collate}
#   COHORT_DEFAULT     the cohort file under genomes/ that submit and collate cover (embedding_input.csv, or
#                      embedding_input_unfiltered.csv where the owner embedded every genome passing the non-species
#                      QC). COHORT (an absolute path) overrides it, e.g. bakta/built_genomes.csv before QC exists
#   OWNER_BAKTA_JOBS   comma-separated job names of every owner's Bakta ingest and run jobs (empty if none)
#   REF_RUN_ROSTER     a run roster of §3.6's whose first genome's run record fixes the fingerprint every run matches
#   BAKREP_EXPORT      BakRep's bakrep-export.tsv.gz
# collate refuses while this cohort's own run arrays are live, unless LIVE_RUNS_OK=1: for a cohort of finished genomes
# only (annotated's), since a run task skips a genome whose run is current and a genome is on one roster only.
# annotated takes HOLD (Samples to leave out: one per line, or a CSV with a Sample column such as a check's
# problems.csv) and ANNOTATED_FROM (a CSV with Sample, sr_assembly_file: consider only these genomes, e.g. the previous
# preqc_annotated.csv, so a re-run adds no genome that finished since; default every genome built here).
# Jobs run from ARRAY_REPO, a detached worktree at the SHA to run (.venv symlinked to the shared env), and refuse to
# start if its HEAD has moved.
set -euo pipefail

: "${ROOT:?}" "${T:?}" "${JOB:?}" "${COHORT_DEFAULT:?}" "${REF_RUN_ROSTER:?}" "${BAKREP_EXPORT:?}"
OWNER_BAKTA_JOBS=${OWNER_BAKTA_JOBS:-}
GEN=$T/genomes
PLAN=$GEN/bakta
STATUS=$PLAN/ingest_status.tsv
# resolved once, to an absolute path, before the cd below: submit and collate refuse if it does not exist
COHORT_ARG=${COHORT:-$GEN/$COHORT_DEFAULT}
COHORT=$(realpath -e -- "$COHORT_ARG" 2>/dev/null) || COHORT=
ANN=$ROOT/raw/bakta  # shared by every cohort, keyed by BioSample
DB=$ROOT/raw/bakta_db/v5.1_full
LOGS=/rds/user/dca36/hpc-work/logs
ACCOUNT=FLOTO-PROJECT-K-SL2-CPU
# the pixi project whose bakta env every run used: the shared checkout's, so a worktree never installs a second copy
# (an env is keyed by its project's path, and the fingerprint lists the env's exact packages)
PIXI_DIR=${PIXI_DIR:-$HOME/workspace/BacPredict/src/bacpredict/engine/genomes}
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$REPO"
# this checkout's code even though the shared .venv's editable install points at the main checkout
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

reuse() { uv run --no-sync python -m bacpredict.engine.genomes.reuse_built_genomes "$@"; }
tools_bin() { echo "$(cd "$PIXI_DIR" && pixi run -e bakta -- printenv CONDA_PREFIX)/bin"; }
submit() {  # sbatch, then print the resource line the global rules require (read back from SLURM)
    local id job; id=$(sbatch --parsable "$@"); id=${id%%;*}
    job=$(scontrol show job "$id")
    echo "JOB $id $(grep -m1 -oE 'JobName=[^ ]+' <<<"$job") | CPU | $(grep -m1 -oE 'ReqTRES=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'TimeLimit=[^ ]+' <<<"$job") | $(grep -m1 -oE 'Partition=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'Account=[^ ]+' <<<"$job")"
}
need_cohort() {
    [[ -n $COHORT && -s $COHORT ]] || {
        echo "no cohort at $COHORT_ARG (collate the genome stage first; or, before QC, COHORT=$PLAN/built_genomes.csv)"
        exit 1; }
}
pinned() {  # the checkout a job runs: ARRAY_REPO, which must be a checkout with the shared env linked in
    local ar=${ARRAY_REPO:?set ARRAY_REPO to a detached worktree at the SHA to run (see the header)}
    git -C "$ar" rev-parse HEAD >/dev/null 2>&1 || { echo "ARRAY_REPO=$ar is not a git checkout" >&2; return 1; }
    [[ -d $ar/src/bacpredict && -x $ar/.venv/bin/python ]] || {
        echo "ARRAY_REPO=$ar: no src/bacpredict or no .venv (symlink the shared one)" >&2; return 1; }
    echo "$ar"
}
# The start of a --wrap job run from checkout $1: refuse unless it is still at the SHA submitted with no tracked file
# modified (HEAD alone does not pin the code), and put its code first on PYTHONPATH.
pin_guard() {
    local sha; sha=$(git -C "$1" rev-parse HEAD)
    echo "EXPECT_SHA=$sha; h=\$(git -C $1 rev-parse HEAD); [ \"\$h\" = \"\$EXPECT_SHA\" ] ||" \
         "{ echo \"$1 moved: \$h, submitted at \$EXPECT_SHA\"; exit 3; };" \
         "[ -z \"\$(git -C $1 status --porcelain --untracked-files=no)\" ] ||" \
         "{ echo \"$1 has modified tracked files: what would run is at no SHA\"; exit 3; };" \
         "echo repo=$1@\$h; export PYTHONPATH=$1/src;"
}
refuse_if_live() {  # <job names, comma-separated> <why>: stop if any of mine is queued or running
    local ids
    ids=$(squeue -u "$USER" -h -n "$1" -o '%i') || { echo "squeue failed: cannot tell whether $1 is live"; exit 1; }
    [[ -z $ids ]] || { echo "queued or running ($1): $(echo $ids) — $2"; exit 1; }
}
OWN_RUNS=$JOB-bakta-all,$JOB-bakta-planned,$JOB-bakta-refused
# the fingerprint of §3.6's runs: the run record of the first genome of REF_RUN_ROSTER
ref_fingerprint() {
    local ref
    read -r ref _ < "$REF_RUN_ROSTER"
    uv run --no-sync python -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["fingerprint"])' \
        "$ANN/$ref.bakta.run.json"
}
plan() {
    [[ -s $BAKREP_EXPORT ]] || { echo "no BakRep export at $BAKREP_EXPORT"; exit 1; }
    reuse bakta-plan --manifest "$GEN/genome_manifest.csv" --shard-root "$GEN/shards" \
        --reused "$GEN/reused_genomes.csv" --bakrep-export "$BAKREP_EXPORT" --out-dir "$PLAN"
}

case "${1:-}" in
plan)
    plan
    ;;
annotated)
    # built_genomes.csv as the shards are now: a genome re-planned to another source drops out until it is rebuilt,
    # and once rebuilt its contigs are newer than any record it had
    plan
    from=${ANNOTATED_FROM:-$PLAN/built_genomes.csv}
    reuse bakta-annotated --built "$from" --annotation-dir "$ANN" --fingerprint "$(ref_fingerprint)" \
        ${HOLD:+--hold "$HOLD"} --out "$GEN/preqc_annotated.csv"
    echo "next: COHORT=$GEN/preqc_annotated.csv LIVE_RUNS_OK=1 <wrapper> collate"
    ;;
ingest)
    [[ -s $PLAN/bakrep_roster.tsv ]] || { echo "no BakRep roster at $PLAN/bakrep_roster.tsv: run plan first"; exit 1; }
    ar=$(pinned) || exit 1
    refuse_if_live "$JOB-bakta-ingest,$JOB-bakta-all,$JOB-bakta-refused" \
        "an ingest racing our runs on the same genomes could leave two annotations"
    # network-bound: 8 BakRep streams (a 76-way pull was throttled). §3.6 took 2 h 4 min for 63,700 genomes, 4 min for
    # PsA's 1,792. Resumable (a current ingest record is kept), so a wall only costs a re-run
    run_dir=$LOGS/${JOB}_bakta_ingest_$(date +%Y%m%d_%H%M%S); mkdir -p "$run_dir"
    submit --account=$ACCOUNT --partition=icelake --cpus-per-task=8 --mem=16G --time=${INGEST_TIME:-04:00:00} \
        --job-name="$JOB-bakta-ingest" --chdir="$run_dir" --output="$run_dir/%x_%j.out" --error="$run_dir/%x_%j.out" \
        --wrap="$(pin_guard "$ar") PYTHONUNBUFFERED=1 uv run --project $ar --no-sync \
python -m bacpredict.engine.genomes.bakta ingest --roster $PLAN/bakrep_roster.tsv --out-dir $ANN --workers 8 \
--status-out $STATUS"
    echo "logs: $run_dir"
    ;;
submit)  # submit [all|planned|refused]: all = planned + refused (the default)
    mode=${2:-all}
    case $mode in
        all) clash=$OWN_RUNS ;;
        planned|refused) clash=$JOB-bakta-all,$JOB-bakta-$mode ;;
        *) echo "submit [all|planned|refused]"; exit 2 ;;
    esac
    ar=$(pinned) || exit 1
    need_cohort
    for f in "$PLAN/built_genomes.csv" "$GEN/reused_genomes.csv" "$DB/version.json" "$REF_RUN_ROSTER"; do
        [[ -s $f ]] || { echo "missing $f (built_genomes.csv: run plan)"; exit 1; }
    done
    [[ -e $PLAN/bakrep_roster.tsv ]] || { echo "no $PLAN/bakrep_roster.tsv: run plan"; exit 1; }
    max_run=${MAX_RUN:-6000}  # a guard on cost: a roster this large is more likely a wrong COHORT than the plan
    [[ $max_run =~ ^[0-9]+$ ]] || { echo "MAX_RUN=$max_run: not a whole number"; exit 2; }
    # checked before a roster is frozen: one array of 4-genome tasks holds at most 40,000 (CSD3 MaxArraySize 10,000)
    (( max_run <= 40000 )) || { echo "MAX_RUN=$max_run: one array holds at most 40,000 genomes; split the COHORT"; exit 2; }
    allow=()
    case ${ALLOW_ERRORS:-0} in 0) ;; 1) allow=(--allow-errors) ;; *) echo "ALLOW_ERRORS: 0 or 1"; exit 2 ;; esac
    refuse_if_live "$clash" "an overlapping run array would run the same genomes twice"
    # an owner's live Bakta job may be writing a reused genome's files: reused genomes then wait for a later submit
    skip=()
    if [[ -n $OWNER_BAKTA_JOBS ]]; then
        live=$(squeue -u "$USER" -h -n "$OWNER_BAKTA_JOBS" -o '%i') || { echo "squeue failed: cannot tell"; exit 1; }
        [[ -z $live ]] || { skip=(--skip-reused); echo "owner Bakta jobs live ($(echo $live)): reused genomes left for a later submit"; }
    fi
    status=()
    if [[ $mode != planned ]]; then
        [[ -s $STATUS ]] || {
            echo "no $STATUS: run ingest first, or re-run it (a killed ingest leaves only $STATUS.part)," \
                 "or: submit planned"
            exit 1; }
        refuse_if_live "$JOB-bakta-ingest" "its status file is not final yet"
        status=(--status "$STATUS")
    fi
    # one annotation of record across cohorts: our runs must use the tool env and database §3.6's runs used
    tb=$(tools_bin)
    fp=$(PYTHONPATH=$ar/src uv run --project "$ar" --no-sync python -m bacpredict.engine.genomes.bakta fingerprint \
        --tools-bin "$tb" --db "$DB")
    ref_fp=$(ref_fingerprint)
    [[ $fp == "$ref_fp" ]] || {
        echo "fingerprint of $tb + $DB is $fp; §3.6's runs ($REF_RUN_ROSTER) used $ref_fp: not submitting"
        exit 1; }
    roster=$PLAN/run_roster_$(date +%Y%m%d_%H%M%S).tsv
    reuse bakta-roster --cohort "$COHORT" --built "$PLAN/built_genomes.csv" --reused "$GEN/reused_genomes.csv" \
        --bakrep-roster "$PLAN/bakrep_roster.tsv" "${status[@]}" --annotation-dir "$ANN" --fingerprint "$fp" \
        --mode "$mode" --max-run "$max_run" "${allow[@]}" "${skip[@]}" --out "$roster"
    [[ -s $roster ]] || { echo "nothing to run"; exit 0; }
    chunk=4
    n=$(( ($(wc -l < "$roster") + chunk - 1) / chunk ))
    # as §3.6 (sacct): 33 min mean / 61 min max per 4-genome task, MaxRSS ≤ 42.6 GB per genome, so 8 cores / 52 GB
    # (under 8 × 6,760 MB) on icelake-himem and 3 h. On a congested himem queue, icelake (3,380 MB per core) needs
    # BAKTA_PARTITION=icelake BAKTA_CPUS=16 for the same 52G
    # TMPDIR: run_bakta_array.sh reads each roster row through a here-string, which bash 4.4 writes to $TMPDIR — /tmp
    # was full on several icelake nodes
    export_vars="ALL,REPO=$ar,ROSTER=$roster,OUT_DIR=$ANN,TOOLS_BIN=$tb,DB=$DB,CHUNK=$chunk,KEEP=gff3+faa+txt,TMPDIR=/local"
    export_vars+=",EXPECT_SHA=$(git -C "$ar" rev-parse HEAD)"
    # BAKTA_ACCOUNT: the run array's account (FLOTO-SL3-CPU — free, low priority — for a large array that can wait)
    submit --array=0-$((n - 1))%${BAKTA_CONCURRENT:-100} --account=${BAKTA_ACCOUNT:-$ACCOUNT} \
        --partition=${BAKTA_PARTITION:-icelake-himem} --cpus-per-task=${BAKTA_CPUS:-8} --mem=${BAKTA_MEM:-52G} \
        --time=${BAKTA_TIME:-03:00:00} --job-name="$JOB-bakta-$mode" --export="$export_vars" \
        "$ar/src/bacpredict/engine/genomes/scripts/run_bakta_array.sh"
    echo "roster: $roster"
    ;;
collate)
    # exit 1, and no table, unless every COHORT genome has exactly one current annotation (md5 of its contigs now).
    # Only embedding_input.csv's collate writes bakta_annotation.csv: any other cohort's must neither replace it nor
    # (failing) delete it
    need_cohort
    case ${LIVE_RUNS_OK:-0} in
        0) refuse_if_live "$JOB-bakta-ingest,$OWN_RUNS" "collate must see every annotation finished" ;;
        1) refuse_if_live "$JOB-bakta-ingest" "an ingest may be writing a cohort genome's GFF3" ;;
        *) echo "LIVE_RUNS_OK: 0 or 1"; exit 2 ;;
    esac
    canon=$(realpath -e -- "$GEN/embedding_input.csv" 2>/dev/null) || canon=
    if [[ $COHORT == "$canon" ]]; then
        out=$GEN/bakta_annotation.csv
    else
        stem=$(basename -- "$COHORT"); out=$GEN/bakta_annotation_${stem%.*}.csv
    fi
    ar=$(pinned) || exit 1
    run_dir=$LOGS/${JOB}_bakta_collate_$(date +%Y%m%d_%H%M%S); mkdir -p "$run_dir"
    # §3.6 collated 67,326 genomes in 13 min at 16 cores (sacct)
    submit --account=$ACCOUNT --partition=icelake --cpus-per-task=16 --mem=32G --time=${COLLATE_TIME:-01:00:00} \
        --job-name="$JOB-bakta-collate" --chdir="$run_dir" --output="$run_dir/%x_%j.out" --error="$run_dir/%x_%j.out" \
        --wrap="$(pin_guard "$ar") PYTHONUNBUFFERED=1 uv run --project $ar --no-sync \
python -m bacpredict.engine.genomes.bakta collate --cohort $COHORT --annotation-dir $ANN --tools-bin $(tools_bin) \
--db $DB --out $out"
    echo "cohort: $COHORT → $out"
    echo "logs: $run_dir"
    ;;
status)
    reuse bakta-status --plan-dir "$PLAN" --annotation-dir "$ANN" --reused "$GEN/reused_genomes.csv" \
        ${COHORT:+--cohort "$COHORT"}
    q=$(squeue -u "$USER" -h -o '%.12i %.24j %.9T %.11M') || { echo "squeue failed: cannot tell which jobs are live"; exit 1; }
    grep -E "$JOB-bakta" <<<"$q" || echo "no $JOB-bakta job queued or running"
    ;;
*)
    sed -n '2,18p' "$0"; exit 2 ;;
esac
