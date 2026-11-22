#!/usr/bin/env bash
# ESM-C embeddings of a second cohort's genomes (CABBAGE v2, PROJECT_STATE §3.7), leaving every genome an earlier
# cohort built to that cohort's own embedding. Run it through a cohort wrapper, from a CSD3 login node:
#
#   ARRAY_REPO=<worktree> <wrapper> proteins        CPU job: the cohort's genomes → protein parquets (shared store)
#   ARRAY_REPO=<worktree> <wrapper> reextract       CPU job: re-extract the genomes built here, rewriting a parquet
#                                                   only where its proteins differ (after an extractor setting changed)
#   ARRAY_REPO=<worktree> <wrapper> check-proteins  CPU job: every cohort parquet holds its GFF3's proteins
#   ARRAY_REPO=<worktree> <wrapper> link-kp         CPU job: Kp-store .pt linked in where they match (LINK_GENUS)
#   ARRAY_REPO=<worktree> <wrapper> submit          GPU array: the genomes built here with no .pt yet → esm/
#   ARRAY_REPO=<worktree> <wrapper> check           CPU job: every cohort genome, end to end
#
# §3.6's procedure (apps/entero/scripts/build_entero_embeddings.sh, after apps/pa/scripts/build_pa_embeddings.sh;
# mechanics there and in apps/pa/CLAUDE.md), with the same extractor, array, settings and checks: --max-proteins 9000,
# bf16 storage (§6), parquets and .pt in the flat per-BioSample stores $ROOT/processed/{protein_sequences,esm}, the
# array given an md5-pinned sample list. No smoke: the code and settings are §3.6's, which its smoke checked. What
# differs, and why:
# - The cohort is a Bakta collate table (ANNOTATION, under genomes/ or absolute; default bakta_annotation.csv). Before
#   QC exists it is bakta_annotation_preqc_annotated.csv, collated for genomes/preqc_annotated.csv (the annotate
#   wrapper's `annotated` stage): every genome built here whose annotation has finished. Re-run as more finish; the
#   array then takes only those still without a .pt.
# - A reused genome (genomes/reused_genomes.csv) is its owner's: never on this cohort's array or link list. check
#   still requires its .pt, so a reused genome its owner never embedded fails check rather than being embedded here.
# - Ambiguous codons (David, 2026-11-22, PROJECT_STATE §6): every codon holding a base other than A, C, G or T is
#   translated as X, as Bakta's .faa does (--ambiguous-codons-as-x). v2's NCBI GenBank assemblies carry IUPAC codes;
#   under §3.6's setting Biopython read some as stops and the CDS was dropped. reextract brings parquets written
#   before the setting up to it, touching only those that change; check-proteins must pass again after it.
# - Every job, CPU ones too, runs from ARRAY_REPO (a detached worktree at the SHA to run, .venv symlinked to the
#   shared env), refuses to start if its HEAD moved or a tracked file is modified, and takes TMPDIR on node-local
#   /local (/tmp was full on some icelake nodes). submit refuses unless ARRAY_REPO's array code is this checkout's.
#
# Configuration (exported by the wrapper):
#   ROOT, T       the AMR root; the cohort's folder (genomes/ under it; cohort.csv with species, for LINK_GENUS)
#   JOB           job-name and log prefix: $JOB-protein-sequences, $JOB-esm-{check-proteins,link-kp,embeddings,check}
#   LINK_GENUS    the genus whose genomes link-kp looks up in the Kp ESM-C store (empty: no link-kp stage)
# Overrides: ANNOTATION, QC_COHORT (the table the cohort must equal — by default the one its annotation was collated
# for; `none` for no such check), KP_ESM_DIR, ESM_PER_TASK, ESM_CONCURRENT, SKIP_LINK_KP=1, and the CPU jobs' shapes
# (PROT_*, CHECK_*, LINK_* — PARTITION, CPUS, MEM, TIME).
set -euo pipefail

: "${ROOT:?}" "${T:?}" "${JOB:?}"
LINK_GENUS=${LINK_GENUS:-}
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
GEN=$T/genomes
COHORT=${ANNOTATION:-bakta_annotation.csv}
[[ $COHORT == /* ]] || COHORT=$GEN/$COHORT
# the QC table the cohort must equal: the one its annotation was collated for (the annotate driver's collate writes
# bakta_annotation.csv for embedding_input.csv and bakta_annotation_<stem>.csv for <stem>.csv)
ann=$(basename -- "$COHORT" .csv)
case $ann in
    bakta_annotation) QC_DEFAULT=$GEN/embedding_input.csv ;;
    bakta_annotation_*) QC_DEFAULT=$GEN/${ann#bakta_annotation_}.csv ;;
    *) QC_DEFAULT= ;;
esac
QC_COHORT=${QC_COHORT:-$QC_DEFAULT}
[[ -z $QC_COHORT || $QC_COHORT == none || $QC_COHORT == /* ]] || QC_COHORT=$GEN/$QC_COHORT
REUSED=$GEN/reused_genomes.csv
SAMPLES=$GEN/esm_samples_to_embed.txt
PROT=$ROOT/processed/protein_sequences
ESM=$ROOT/processed/esm
# link-kp: the Klebsiella ESM-C store (fp32, 960-dim, keyed by BioSample; PROJECT_STATE §3.6 Caveats)
KP_ESM=${KP_ESM_DIR:-/rds/project/rds-4k08a2yyQLw/david/bacformer_processed/esm/klebsiella}
LINK_SAMPLES=$GEN/esm_samples_kp_link.txt
LINK_REPORT=$GEN/esm_linked_kp.tsv
MAX_PROTEINS=9000
SAVE_DTYPE=bfloat16
# genomes per 4 h GPU task: §3.6's 300 (~13 s an Enterobacterales genome, ~18 s a PsA one, at 2.8 ms per protein on
# one A100: 65-90 min a task, so the wall is 2.7-3.7× the estimate)
PER_TASK=${ESM_PER_TASK:-300}
LOGS=/rds/user/dca36/hpc-work/logs
ACCOUNT=FLOTO-PROJECT-K-SL2-CPU
ARRAY=src/bacpredict/engine/embedding/scripts/esm_embeddings_array.sh
# what the array runs in ARRAY_REPO: it must be this driver's own copy of each (same_array_code)
ARRAY_CODE=("$ARRAY" src/bacpredict/engine/embedding/generate_embeddings.py
            src/bacpredict/engine/embedding/sample_list.py)
cd "$REPO"
# this checkout's code even though the shared .venv's editable install points at the main checkout
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

submit() {  # sbatch, then print the resource line the global rules require (read back from SLURM)
    local id job kind; id=$(sbatch --parsable "$@"); id=${id%%;*}
    job=$(scontrol show job "$id")
    kind=CPU; grep -q 'gres/gpu' <<<"$job" && kind="GPU x$(grep -m1 -oE 'gres/gpu=[0-9]+' <<<"$job" | cut -d= -f2)"
    echo "JOB $id $(grep -m1 -oE 'JobName=[^ ]+' <<<"$job") | $kind | $(grep -m1 -oE 'ReqTRES=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'TimeLimit=[^ ]+' <<<"$job") | $(grep -m1 -oE 'Partition=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'Account=[^ ]+' <<<"$job")"
}
# live <ERE>: how many of my queued or running jobs have a name matching it. A failed squeue is not "none live", so
# it exits; call it as n=$(live …), an assignment, so that set -e stops the driver too
live() {
    local jobs
    jobs=$(squeue -u "$USER" -h -o '%j') || {
        echo "squeue failed: cannot tell whether a job matching $1 is live" >&2; exit 1; }
    grep -cE "$1" <<<"$jobs" || true
}
pinned() {  # the checkout a job runs: ARRAY_REPO, a clean checkout with the shared env linked in
    local ar=${ARRAY_REPO:?set ARRAY_REPO to a detached worktree at the SHA to run (see the header)}
    git -C "$ar" rev-parse HEAD >/dev/null 2>&1 || { echo "ARRAY_REPO=$ar is not a git checkout" >&2; return 1; }
    [[ -d $ar/src/bacpredict && -x $ar/.venv/bin/python ]] || {
        echo "ARRAY_REPO=$ar: no src/bacpredict or no .venv (symlink the shared one)" >&2; return 1; }
    [[ -z $(git -C "$ar" status --porcelain --untracked-files=no) ]] || {
        echo "ARRAY_REPO=$ar has modified tracked files: what would run is at no SHA" >&2; return 1; }
    echo "$ar"
}
# The start of a --wrap job run from checkout $1: refuse unless it is still at the SHA submitted with no tracked file
# modified, put its code first on PYTHONPATH, and take node-local scratch for TMPDIR
pin_guard() {
    local sha; sha=$(git -C "$1" rev-parse HEAD)
    echo "EXPECT_SHA=$sha; h=\$(git -C $1 rev-parse HEAD); [ \"\$h\" = \"\$EXPECT_SHA\" ] ||" \
         "{ echo \"$1 moved: \$h, submitted at \$EXPECT_SHA\"; exit 3; };" \
         "[ -z \"\$(git -C $1 status --porcelain --untracked-files=no)\" ] ||" \
         "{ echo \"$1 has modified tracked files: what would run is at no SHA\"; exit 3; };" \
         "if [ -d /local ] && [ -w /local ]; then export TMPDIR=/local; fi;" \
         "echo repo=$1@\$h; export PYTHONPATH=$1/src;"
}
# same_array_code <array repo>: refuse unless the array script and the modules it runs there are this checkout's
same_array_code() {
    local f
    for f in "${ARRAY_CODE[@]}"; do
        cmp -s "$REPO/$f" "$1/$f" || {
            echo "ARRAY_REPO's $1/$f is not this driver's $REPO/$f (differs or missing):"
            echo "run the driver from ARRAY_REPO"; exit 1; }
    done
}
need_cohort() { [[ -s $COHORT ]] || { echo "no $COHORT — collate the cohort's annotation first"; exit 1; }; }

# proteins_passed: refuse unless the latest check-proteins run passed on the cohort table as it is now. Its
# check_cohort_store writes problems.csv.pass (the md5 of the table it checked) only on PASS, so a later run that is
# pending, running or failed has none, and a table rewritten since has another md5
proteins_passed() {
    local runs last have want
    runs=("$LOGS"/${JOB}_esm_check-proteins_*)  # timestamped names: glob order is time order
    last=${runs[${#runs[@]}-1]}
    [[ -d $last ]] || { echo "no check-proteins run in $LOGS: run check-proteins first"; exit 1; }
    have=$(cat "$last/problems.csv.pass" 2>/dev/null) || {
        echo "the latest check-proteins ($last) has not passed: read its .out (or wait for it)"; exit 1; }
    want=$(md5sum < "$COHORT" | cut -d' ' -f1)
    [[ $have == "$want" ]] || {
        echo "$COHORT has changed since the latest check-proteins ($last) passed: run check-proteins again"; exit 1; }
    PROTEINS_PASS=$last/problems.csv.pass  # kp_linked dates the link-kp run against it
    # a reextract rewrites parquets in place, leaving the cohort table (so its md5) as it was
    local rx last_rx
    rx=("$LOGS"/${JOB}_protein_reextract_*)
    last_rx=${rx[${#rx[@]}-1]}
    [[ ! -d $last_rx || $PROTEINS_PASS -nt $last_rx ]] || {
        echo "parquets were re-extracted ($last_rx) after the latest check-proteins passed: run check-proteins again"
        exit 1; }
}

# kp_linked (after proteins_passed): with LINK_GENUS set, refuse unless the latest link-kp ran to completion after
# the check-proteins pass just verified — otherwise the list submit writes still holds every genome the Kp store has.
# SKIP_LINK_KP=1 submits without one (those genomes are then embedded)
kp_linked() {
    [[ -n $LINK_GENUS ]] || return 0
    if [[ ${SKIP_LINK_KP:-0} == 1 ]]; then echo "SKIP_LINK_KP=1: not requiring a link-kp run"; return 0; fi
    local runs last outs out
    runs=("$LOGS"/${JOB}_esm_link-kp_*)
    last=${runs[${#runs[@]}-1]}
    [[ -d $last ]] || {
        echo "no link-kp run in $LOGS: run link-kp first (or SKIP_LINK_KP=1 to embed the $LINK_GENUS genomes too)"; exit 1; }
    outs=("$last"/"$JOB"-esm-link-kp_*.out); out=${outs[${#outs[@]}-1]}
    [[ -f $out ]] && grep -qx done "$out" || {
        echo "the latest link-kp ($last) did not finish with 'done': read its .out, then run link-kp again"
        echo "(or SKIP_LINK_KP=1 to submit without it)"; exit 1; }
    [[ $out -nt $PROTEINS_PASS && -s $LINK_REPORT && $LINK_REPORT -nt $PROTEINS_PASS ]] || {
        echo "check-proteins passed again ($PROTEINS_PASS) after the latest link-kp ($last), or $LINK_REPORT is"
        echo "missing or older: run link-kp again"; exit 1; }
    echo "link-kp: $last ($LINK_REPORT)"
}

# write_list <out> <python> [args…]: (re)write <out> with what <python> prints — sorted, unique samples, from the
# cohort table (argv 1), cohort.csv (argv 2), reused_genomes.csv (argv 3) and args (argv 4…) — atomically and only if
# it changed; print its md5 (what the array's SAMPLES_MD5 pins). Called as md5=$(write_list …), where bash drops
# set -e: hence the explicit return, or a failed write would hand on the md5 of a stale list
write_list() {
    local out=$1 tmp; tmp=$1.$$.part
    uv run --no-sync python -I -c "$2" "$COHORT" "$T/cohort.csv" "$REUSED" "${@:3}" > "$tmp" || { rm -f "$tmp"; return 1; }
    if cmp -s "$tmp" "$out"; then rm -f "$tmp"; else mv "$tmp" "$out"; fi
    md5sum < "$out" | cut -d' ' -f1
}
# the cohort's own (non-reused) Samples, checked unique, non-empty and unpadded
OWN='
import os
import sys
import pandas as pd
s = pd.read_csv(sys.argv[1], dtype={"Sample": str})["Sample"]
if s.isna().any() or not s.is_unique or s.str.strip().ne(s).any() or s.eq("").any():
    sys.exit(f"{sys.argv[1]}: Sample must be unique, non-empty and unpadded")
reused = set(pd.read_csv(sys.argv[3], dtype=str, usecols=["Sample"])["Sample"])
own = s[~s.isin(reused)]
'
# the cohort genomes built here with no .pt in the ESM store (argv 4), from one directory read: a .pt counts whether
# it was embedded or linked, and whatever a link points at (check flags a broken one). Prints nothing if none
TO_EMBED=$OWN'
suffix = "_esm_embeddings.pt"
with os.scandir(sys.argv[4]) as it:
    have = {e.name[: -len(suffix)] for e in it if e.name.endswith(suffix)}
todo = sorted(set(own) - have)
lacking = len(set(s) - set(own) - have)
print(f"cohort: {len(s)} genomes, {len(s) - len(own)} reused ({lacking} of them without a .pt: their owner'"'"'s to "
      f"embed, and check fails until it does); of the {len(own)} built here, {len(own) - len(todo)} already have a "
      f".pt in {sys.argv[4]} (embedded or linked), {len(todo)} to embed", file=sys.stderr)
if todo:
    print("\n".join(todo))'
# the cohort genomes built here whose cohort.csv species is of genus argv 4 and with no .pt in the ESM store (argv 5)
# (link-kp). Two guards, as for the array, so neither alone keeps v2 out of another cohort's .pt: reused_genomes.csv,
# and the store's own listing (link_existing_embeddings also never replaces a name that exists). Prints nothing if
# none
GENUS_SAMPLES=$OWN'
genus = pd.read_csv(sys.argv[2], dtype={"Sample": str}).set_index("Sample")["species"].str.split().str[0]
suffix = "_esm_embeddings.pt"
with os.scandir(sys.argv[5]) as it:
    have = {e.name[: -len(suffix)] for e in it if e.name.endswith(suffix)}
labelled = own[own.map(genus).eq(sys.argv[4])]
pick = sorted(set(labelled) - have)
print(f"cohort: {len(s)} genomes, {len(own)} built here, {len(labelled)} of them labelled {sys.argv[4]} (cohort.csv "
      f"species), {len(pick)} of those without a .pt in {sys.argv[5]}", file=sys.stderr)
if pick:
    print("\n".join(pick))'

# check_job <stage> <args…>: check_cohort_store as a 32-core icelake job (CHECK_* override the shape)
check_job() {
    local stage=$1 ar run_dir qc=(); shift
    ar=$(pinned) || exit 1
    need_cohort
    if [[ $QC_COHORT != none ]]; then
        [[ -s $QC_COHORT ]] || { echo "no QC table at '$QC_COHORT' for $COHORT: set QC_COHORT (or QC_COHORT=none)"; exit 1; }
        qc=(--qc-cohort "$QC_COHORT")
    fi
    run_dir=$LOGS/${JOB}_esm_${stage}_$(date +%Y%m%d_%H%M%S); mkdir -p "$run_dir"
    # I/O-bound: per genome a contigs md5 (~1.5 MB), a parquet column (~1.5 MB) and a .pt zip check. §3.6's checks
    # of ~68k genomes took 4.5 and 7.5 min (sacct), so 30 min; ~20 GB for 32 workers, so 48G
    submit --account=$ACCOUNT --partition=${CHECK_PARTITION:-icelake} --cpus-per-task=32 \
        --mem=${CHECK_MEM:-48G} --time=${CHECK_TIME:-00:30:00} --job-name="$JOB-esm-$stage" --chdir="$run_dir" \
        --output="$run_dir/%x_%j.out" --error="$run_dir/%x_%j.out" \
        --wrap="$(pin_guard "$ar") PYTHONUNBUFFERED=1 uv run --project $ar --no-sync \
python -m bacpredict.engine.embedding.check_cohort_store --cohort $COHORT ${qc[*]} --protein-dir $PROT \
--max-proteins $MAX_PROTEINS --save-dtype $SAVE_DTYPE --workers 32 --report $run_dir/problems.csv $*"
    echo "cohort: $COHORT${qc:+ (= ${qc[1]})}; logs: $run_dir (the .out ends PASS or FAIL; every problem is in problems.csv)"
}

case "${1:-}" in
proteins)
    ar=$(pinned) || exit 1
    need_cohort
    n=$(live "^$JOB-protein-sequences\$")
    (( n == 0 )) || { echo "a $JOB-protein-sequences job is queued or running"; exit 1; }
    run_dir=$LOGS/${JOB}_protein_sequences_$(date +%Y%m%d_%H%M%S); mkdir -p "$run_dir"
    # one job: 32 cores (one genome per worker) on icelake; 96G is under icelake's 3,380 MB per core × 32. §3.6's ran
    # ~68k genomes in 26 min (sacct), so 1 h for up to that many; resumable (--skip-existing, atomic parquets: a reused
    # genome's parquet is kept), so a wall kill only means resubmitting
    cpus=${PROT_CPUS:-32}
    submit --account=$ACCOUNT --partition=${PROT_PARTITION:-icelake} --cpus-per-task=$cpus \
        --mem=${PROT_MEM:-96G} --time=${PROT_TIME:-01:00:00} --job-name="$JOB-protein-sequences" --chdir="$run_dir" \
        --output="$run_dir/%x_%j.out" --error="$run_dir/%x_%j.err" \
        --wrap="$(pin_guard "$ar") PYTHONUNBUFFERED=1 uv run --project $ar --no-sync \
python -m bacpredict.engine.embedding.preprocess_assemblies_to_protein_sequences --input-csv $COHORT \
--output-dir $PROT --workers $cpus --skip-existing --ambiguous-codons-as-x"
    echo "cohort: $COHORT, $(($(wc -l < "$COHORT") - 1)) genomes; logs: $run_dir"
    ;;
reextract)
    ar=$(pinned) || exit 1
    need_cohort
    # nothing else may write these parquets meanwhile; an array or link job reading them would race the rewrite
    n=$(live "^$JOB-(protein-sequences|protein-reextract|esm-embeddings|esm-link-kp)\$")
    (( n == 0 )) || { echo "a $JOB protein, array or link-kp job is queued or running"; exit 1; }
    run_dir=$LOGS/${JOB}_protein_reextract_$(date +%Y%m%d_%H%M%S); mkdir -p "$run_dir"
    # as proteins (every genome is extracted again, then compared): ~26 min per 68k genomes at 32 cores
    cpus=${PROT_CPUS:-32}
    submit --account=$ACCOUNT --partition=${PROT_PARTITION:-icelake} --cpus-per-task=$cpus \
        --mem=${PROT_MEM:-96G} --time=${PROT_TIME:-01:00:00} --job-name="$JOB-protein-reextract" --chdir="$run_dir" \
        --output="$run_dir/%x_%j.out" --error="$run_dir/%x_%j.out" \
        --wrap="$(pin_guard "$ar") PYTHONUNBUFFERED=1 uv run --project $ar --no-sync \
python -m bacpredict.engine.embedding.reextract_proteins --input-csv $COHORT --exclude $REUSED --output-dir $PROT \
--ambiguous-codons-as-x --workers $cpus --report $run_dir/reextract.csv"
    echo "cohort: $COHORT (reused genomes excluded); report: $run_dir/reextract.csv; then check-proteins"
    ;;
check-proteins)
    check_job check-proteins
    ;;
link-kp)
    [[ -n $LINK_GENUS ]] || { echo "LINK_GENUS is not set for this cohort: no link-kp stage"; exit 2; }
    ar=$(pinned) || exit 1
    need_cohort
    [[ -d $KP_ESM ]] || { echo "no Kp ESM-C store at $KP_ESM (set KP_ESM_DIR)"; exit 1; }
    # the comparison reads our parquets, so they must be the checked ones
    proteins_passed
    # links made while an array runs only race its writes; two link jobs would write one report
    n=$(live "^$JOB-esm-(embeddings|link-kp)\$")
    (( n == 0 )) || { echo "a $JOB-esm-embeddings array or a $JOB-esm-link-kp job is queued or running"; exit 1; }
    write_list "$LINK_SAMPLES" "$GENUS_SAMPLES" "$LINK_GENUS" "$ESM" > /dev/null
    [[ -s $LINK_SAMPLES ]] || {
        echo "every $LINK_GENUS genome built here has a .pt already: nothing to link; submit with SKIP_LINK_KP=1"; exit 0; }
    run_dir=$LOGS/${JOB}_esm_link-kp_$(date +%Y%m%d_%H%M%S); mkdir -p "$run_dir"
    # as §3.6's: per genome a parquet column and the Kp .pt opened memory-mapped; 16 workers, 16G (~2× 16 imports of
    # torch). Loader paths unset and faulthandler on, as for the ESM-C array: this job imports torch
    cpus=${LINK_CPUS:-16}
    submit --account=$ACCOUNT --partition=${LINK_PARTITION:-icelake} --cpus-per-task=$cpus \
        --mem=${LINK_MEM:-16G} --time=${LINK_TIME:-00:30:00} --job-name="$JOB-esm-link-kp" --chdir="$run_dir" \
        --output="$run_dir/%x_%j.out" --error="$run_dir/%x_%j.out" \
        --wrap="$(pin_guard "$ar") unset LD_LIBRARY_PATH LIBRARY_PATH LD_PRELOAD; PYTHONUNBUFFERED=1 uv run \
--project $ar --no-sync python -X faulthandler -m bacpredict.engine.embedding.link_existing_embeddings \
--samples-file $LINK_SAMPLES --protein-dir $PROT --src-dir $KP_ESM --esm-dir $ESM --max-proteins $MAX_PROTEINS \
--workers $cpus --report $LINK_REPORT"
    echo "genomes: $LINK_SAMPLES; report: $LINK_REPORT; logs: $run_dir"
    ;;
submit)
    ar=$(pinned) || exit 1
    same_array_code "$ar"
    # a second array of this cohort's would share each sample's temporary .pt name with the first; a link-kp job
    # still live has not yet linked the genomes this list would otherwise hand the array. Another cohort's arrays
    # need no check: each embeds only the genomes it built, and the build stage asserts those are disjoint
    n=$(live "^$JOB-esm-(embeddings|link-kp)\$")
    (( n == 0 )) || { echo "a $JOB-esm-embeddings array or a $JOB-esm-link-kp job is queued or running"; exit 1; }
    need_cohort
    proteins_passed
    kp_linked
    md5=$(write_list "$SAMPLES" "$TO_EMBED" "$ESM")
    total=$(awk '{ gsub(/[[:space:]]/, "") } length($0) { n++ } END { print n + 0 }' "$SAMPLES")
    (( total > 0 )) || { echo "every cohort genome built here has a .pt in $ESM: nothing to submit (run check)"; exit 0; }
    # every listed genome has a parquet (one directory read; the counts were check-proteins', required above)
    uv run --no-sync python -m bacpredict.engine.embedding.check_cohort_store --cohort "$COHORT" --protein-dir "$PROT" \
        --samples-file "$SAMPLES" --presence-only \
        || { echo "proteins incomplete: run proteins, then check-proteins"; exit 1; }
    ntasks=$(( (total + PER_TASK - 1) / PER_TASK ))
    # 8 cores / 64 GB / 1 A100 per task on FLOTO-SL2-GPU (the array's defaults; §6: all v2 GPU work on SL2); the
    # tasks pass --skip-existing, so a requeued task resumes within its slice
    sha=$(git -C "$ar" rev-parse HEAD)
    submit --array="0-$((ntasks - 1))%${ESM_CONCURRENT:-64}" --time=04:00:00 --job-name="$JOB-esm-embeddings" \
        --export="ALL,REPO=$ar,EXPECT_SHA=$sha,INPUT_DIR=$PROT,ESM_DIR=$ESM,SAMPLES=$SAMPLES,SAMPLES_MD5=$md5,MAX_PROTEINS=$MAX_PROTEINS,SAVE_DTYPE=$SAVE_DTYPE,NTASKS=$ntasks" \
        "$ar/$ARRAY"
    echo "samples: $SAMPLES ($total genomes, md5 $md5); $ntasks tasks of ≤$(( (total + ntasks - 1) / ntasks ))"
    ;;
check)
    check_job check --esm-dir "$ESM"
    ;;
*)
    sed -n '2,11p' "$0"; exit 2 ;;
esac
