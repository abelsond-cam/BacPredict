#!/usr/bin/env bash
# Build a cohort's genomes into the shared per-BioSample store, reusing every genome an earlier cohort's manifest names
# — the genome stage of a second experiment over the same BioSamples (CABBAGE v2, PROJECT_STATE §3.7). Run it through a
# cohort wrapper, which sets the configuration below, from a CSD3 login node:
#
#   <wrapper> plan              partition cohort.csv (reused_genomes.csv, cohort_to_plan.csv), plan_genome_sources on
#                               the rest with the NCBI fallback (genome_manifest.csv; ENA and NCBI queries — submit it
#                               as a job), then one read-only manifest per MLST scheme (by_scheme/). Refuses if
#                               by_scheme/ exists
#   <wrapper> submit [scheme|all] [atb|portal|ncbi|shovill|all] [indices]
#                               one build array per scheme × source, the scheme pinned (none for `auto`), of the tasks
#                               that have written no shard yet — so a re-run fills gaps and never duplicates a finished
#                               task. Skips (or, with indices, refuses) an array whose name is still queued. Indices
#                               (e.g. a task holding a failed genome, from status) need one scheme and one source
#   <wrapper> qc checkm2        freeze the next CheckM2 list (built genomes in no list yet) and submit its array;
#                               run again for genomes built later
#   <wrapper> qc checkm2-rerun <list number> <indices>
#   <wrapper> qc skani          one skani job over every list (the build complete, every built genome listed); rerun it
#                               after any later list — collate refuses a skani result that missed one
#   <wrapper> collate           collate_inputs/ (this cohort's builds + the reused genomes' rows from their owner),
#                               collate_genomes into a staging folder, the MLST check, then publish into genomes/
#   <wrapper> status
#
# Mechanics are §3.6's (engine/genomes/, apps/pa/CLAUDE.md); what is new is engine/genomes/reuse_built_genomes.py.
# A genome in an earlier cohort's manifest is never built here: build_genomes.build_one deletes and rebuilds a genome
# whose stored src= header differs from its manifest row's source, so naming it again could destroy the owner's copy.
# Disjointness is checked at plan, at every submit, by every array task before it builds (build_genomes_array_
# disjoint.sh, against the owners' manifests as they are when the task starts), and again at collate.
#
# Configuration (exported by the wrapper):
#   ROOT, T            the AMR root; the cohort's folder (cohort.csv, binary_ast.csv; genomes/ is written under it)
#   JOB                job-name prefix
#   OWNER_MANIFESTS    every other cohort's genome_manifest.csv that writes into the store (all must exist)
#   REUSE_SHARD_DIRS, REUSE_CHECKM2, REUSE_SKANI, REUSE_GENOME_QC
#                      the owning cohort's build shards, QC reports and genome_qc.csv, for the genomes this cohort reuses
#   SCHEME_BY_GENUS    Genus=mlst scheme,…; SCHEME_BY_SPECIES (optional) "Genus epithet=scheme,…" wins over it.
#                      Any other genome is built without --scheme (`auto`)
#   GENOME_SIZE        bp, any genus not in GENOME_SIZES (optional Genus=bp,…); GENOME_SIZES_BY_SPECIES (optional
#                      "Genus epithet=bp,…") wins over the genus; MIN_DEPTH (default 10)
#   REFS               species<TAB>reference path, one per line: the owner's skani panel (collate refuses if this
#                      cohort's skani references differ from the ones the owner's skani rows name)
#   SPECIES_MODE       target (with TARGET_SPECIES) or report — as the owner's collate
#   ATB_INDEX          the ATB S3 inventory folder
# Arrays and QC jobs run from ARRAY_REPO, a detached worktree at the SHA to run (.venv symlinked to the shared env),
# and refuse to start if its HEAD has moved (EXPECT_SHA).
set -euo pipefail

: "${ROOT:?}" "${T:?}" "${JOB:?}" "${OWNER_MANIFESTS:?}" "${REUSE_SHARD_DIRS:?}" "${REUSE_CHECKM2:?}" "${REUSE_SKANI:?}"
: "${REUSE_GENOME_QC:?}" "${SCHEME_BY_GENUS:?}" "${GENOME_SIZE:?}" "${REFS:?}" "${SPECIES_MODE:?}" "${ATB_INDEX:?}"
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$REPO"
# this checkout's code even though the shared .venv's editable install points at the main checkout
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
GEN=$T/genomes
BYS=$GEN/by_scheme       # genome_manifest_<scheme>.csv, frozen: array tasks take rows by position
SHARDS=$GEN/shards       # one directory per scheme (shard file names repeat across arrays)
QCD=$GEN/qc
STORE=$ROOT/raw          # build_genomes writes contigs/ and prodigal/ under it
OWNERS_FILE=$GEN/owner_manifests.txt  # what each array task checks itself against, one path per line
MIN_DEPTH=${MIN_DEPTH:-10}
CHECKM2_CHUNK=2000       # task k takes rows k*2000+1 … of its list: fixed, never overridden between qc and its reruns
CHECKM2_DB=${CHECKM2_DB:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/CheckM2/Zenodo_db/CheckM2_database/uniref100.KO.1.dmnd}
# the pixi project whose envs hold shovill/prodigal/mlst/skani and CheckM2: the shared checkout's, so a worktree never
# installs a second copy (its env is keyed by the project's path)
PIXI_DIR=${PIXI_DIR:-$HOME/workspace/BacPredict/src/bacpredict/engine/genomes}
ARRAY=src/bacpredict/engine/genomes/scripts/build_genomes_array_disjoint.sh
MAX_ARRAY=10000          # CSD3 MaxArraySize

reuse() { uv run --no-sync python -m bacpredict.engine.genomes.reuse_built_genomes "$@"; }
tools_bin() { echo "$(cd "$PIXI_DIR" && pixi run -- printenv CONDA_PREFIX)/bin"; }
checkm2_bin() { echo "$(cd "$PIXI_DIR" && pixi run -e checkm2 -- printenv CONDA_PREFIX)/bin"; }
submit() {  # sbatch, then print the resource line the global rules require (read back from SLURM)
    local id job; id=$(sbatch --parsable "$@"); id=${id%%;*}
    job=$(scontrol show job "$id")
    echo "JOB $id $(grep -m1 -oE 'JobName=[^ ]+' <<<"$job") | CPU | $(grep -m1 -oE 'ReqTRES=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'TimeLimit=[^ ]+' <<<"$job") | $(grep -m1 -oE 'Partition=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'Account=[^ ]+' <<<"$job")"
}
# how many of this user's jobs have a name starting with $1; exits the script if squeue fails, which must never read as
# "none queued". Call as n=$(n_queued …) || exit 1
n_queued() {
    local q
    q=$(squeue -u "$USER" -h -o '%j') || { echo "squeue failed: cannot tell whether any $1 job is queued" >&2; return 1; }
    awk -v p="$1" 'index($0, p) == 1 {n++} END {print n + 0}' <<<"$q"
}
refuse_if_queued() {  # <prefix> <why>
    local n; n=$(n_queued "$1") || exit 1
    (( n == 0 )) || { echo "$n $1 job(s) queued or running: $2"; exit 1; }
}
owners_exist() {
    local f
    for f in $OWNER_MANIFESTS; do [[ -s $f ]] || { echo "owner manifest missing: $f"; exit 1; }; done
}
pin() {  # REPO/EXPECT_SHA for --export, from ARRAY_REPO. Call as p=$(pin) || exit 1, never inside an argument
    local sha
    [[ -n ${ARRAY_REPO:-} ]] || { echo "set ARRAY_REPO to a detached worktree at the SHA to run (see the header)" >&2; return 1; }
    sha=$(git -C "$ARRAY_REPO" rev-parse HEAD) || { echo "ARRAY_REPO=$ARRAY_REPO is not a git checkout" >&2; return 1; }
    [[ -x $ARRAY_REPO/.venv/bin/python ]] || { echo "ARRAY_REPO=$ARRAY_REPO has no .venv (symlink the shared one)" >&2; return 1; }
    echo "REPO=$ARRAY_REPO,EXPECT_SHA=$sha"
}
lists() { compgen -G "$QCD/checkm2_list_*.tsv" | sort || true; }
n_tasks() { echo $(( ($(wc -l < "$1") + CHECKM2_CHUNK - 1) / CHECKM2_CHUNK )); }
submit_checkm2() {  # <list> <indices or empty> <pin> <checkm2 bin>
    local list=$1 k n
    k=$(basename "$list" .tsv); k=${k#checkm2_list_}
    n=$(n_tasks "$list")
    # 2,000 genomes at 32 cores: §3.6 measured a median 88 min and MaxRSS ≤ 67 GB, so 4 h and 100G (icelake's 32 × 3,380 MB)
    submit --array="${2:-0-$((n - 1))%40}" --partition=${QC_PARTITION:-icelake} --cpus-per-task=32 \
        --mem=${QC_MEM:-100G} --time=${QC_TIME:-04:00:00} --job-name="$JOB-checkm2-$k" \
        --export="ALL,$3,LIST=$list,QC_DIR=$QCD/list_$k,CHUNK=$CHECKM2_CHUNK,CHECKM2_BIN=$4,CHECKM2_DB=$CHECKM2_DB" \
        src/bacpredict/engine/genomes/scripts/run_checkm2_shard.sh
}
checkm2_ready() {  # resolve and check everything a CheckM2 submission needs, before anything is frozen
    [[ -s $CHECKM2_DB ]] || { echo "no CheckM2 database at $CHECKM2_DB"; exit 1; }
    cbin=$(checkm2_bin)
    "$cbin/checkm2" --version >/dev/null || { echo "no working checkm2 in $cbin"; exit 1; }
}

case "${1:-}" in
plan)
    refuse_if_queued "$JOB-build" "re-planning would shift its rows"
    owners_exist
    [[ ! -e $BYS ]] || { echo "$BYS exists: the plan is frozen (array tasks take its rows by position)"; exit 1; }
    mkdir -p "$GEN"
    reuse partition --cohort "$T/cohort.csv" --built $OWNER_MANIFESTS \
        --out-cohort "$GEN/cohort_to_plan.csv" --out-reused "$GEN/reused_genomes.csv"
    # fresh ENA and NCBI caches per cohort: a cache never expires a "no runs" / "not listed" answer
    uv run --no-sync python -m bacpredict.engine.genomes.plan_genome_sources \
        --cohort "$GEN/cohort_to_plan.csv" --atb-index "$ATB_INDEX"/*.csv.gz --genome-size "$GENOME_SIZE" \
        ${GENOME_SIZES:+--genome-size-by-genus "$GENOME_SIZES"} \
        ${GENOME_SIZES_BY_SPECIES:+--genome-size-by-species "$GENOME_SIZES_BY_SPECIES"} \
        --min-depth "$MIN_DEPTH" --prefer-reads --ena-cache "$GEN/ena_filereport_cache.tsv" \
        --ncbi-fallback --ncbi-cache "$GEN/ncbi_assembly_dirs.tsv" --out "$GEN/genome_manifest.csv"
    reuse check-disjoint --manifest "$GEN/genome_manifest.csv" --built $OWNER_MANIFESTS
    reuse split-by-scheme --manifest "$GEN/genome_manifest.csv" --cohort "$T/cohort.csv" \
        --scheme-by-genus "$SCHEME_BY_GENUS" ${SCHEME_BY_SPECIES:+--scheme-by-species "$SCHEME_BY_SPECIES"} --out-dir "$BYS"
    chmod a-w "$GEN/genome_manifest.csv"
    ;;
submit)  # submit [scheme|all] [atb|portal|ncbi|shovill|all] [indices]
    want_scheme=${2:-all}; want_src=${3:-all}; idx=${4:-}
    case $want_src in portal_gff) want_src=portal ;; ncbi_gca) want_src=ncbi ;; esac
    [[ $want_src =~ ^(all|atb|portal|ncbi|shovill)$ ]] || { echo "source must be atb, portal, ncbi, shovill or all"; exit 2; }
    if [[ $want_scheme != all && ! -e $BYS/genome_manifest_$want_scheme.csv ]]; then
        echo "no scheme $want_scheme: $(cd "$BYS" && ls genome_manifest_*.csv | sed 's/genome_manifest_//; s/\.csv//' | tr '\n' ' ')"; exit 2
    fi
    if [[ -n $idx && ( $want_scheme == all || $want_src == all ) ]]; then
        echo "indices need one scheme and one source"; exit 2
    fi
    owners_exist
    # an owner's manifest can grow after this plan: check again at every submission (each task checks once more)
    reuse check-disjoint --manifest "$GEN/genome_manifest.csv" "$BYS"/genome_manifest_*.csv --built $OWNER_MANIFESTS
    printf '%s\n' $OWNER_MANIFESTS > "$OWNERS_FILE.part" && mv "$OWNERS_FILE.part" "$OWNERS_FILE"
    p=$(pin) || exit 1
    tb=$(tools_bin)
    for t in shovill prodigal mlst; do [[ -x $tb/$t ]] || { echo "no $t in $tb"; exit 1; }; done
    # what is queued now, read once: an array submitted by this run must not make the next scheme's look queued
    queued=$(squeue -u "$USER" -h -o '%j') || { echo "squeue failed: cannot tell what is queued"; exit 1; }
    for f in "$BYS"/genome_manifest_*.csv; do
        scheme=${f##*/genome_manifest_}; scheme=${scheme%.csv}
        [[ $want_scheme == all || $want_scheme == "$scheme" ]] || continue
        mlst=""; [[ $scheme == auto ]] || mlst=",MLST_SCHEME=$scheme"
        common="$p,MANIFEST=$f,OWNERS_FILE=$OWNERS_FILE,OUT_ROOT=$STORE,SHARD_DIR=$SHARDS/$scheme,GENOME_SIZE=$GENOME_SIZE,TOOLS_BIN=$tb$mlst"
        for src in atb portal ncbi shovill; do
            [[ $want_src == all || $want_src == "$src" ]] || continue
            case $src in
            # 100 genomes per 8-core task: §3.6 measured 2.0–4.4 min and MaxRSS 1.6 GB per task
            atb) s=atb; chunk=100; conc=${ATB_CONCURRENT:-50}
                 res=(--partition=${DL_PARTITION:-icelake} --cpus-per-task=8 --mem=${DL_MEM:-6G} --time=${DL_TIME:-00:15:00}) ;;
            portal) s=portal_gff; chunk=40; conc=${PORTAL_CONCURRENT:-20}
                 res=(--partition=${DL_PARTITION:-icelake} --cpus-per-task=8 --mem=24G --time=01:30:00) ;;
            # 40 NCBI assemblies per 8-core task, as portal. Every scheme's ncbi array shares one job name and runs
            # --dependency=singleton, one array after another, so NCBI_CONCURRENT caps the whole cohort's load on NCBI's
            # FTP (4 tasks × 8 workers), not each scheme's
            ncbi) s=ncbi_gca; chunk=40; conc=${NCBI_CONCURRENT:-4}
                 res=(--partition=${DL_PARTITION:-icelake} --cpus-per-task=8 --mem=${DL_MEM:-6G} --time=${NCBI_TIME:-00:45:00}
                      --dependency=singleton) ;;
            # one genome per task on icelake-himem's 6,760 MB per core (shovill --ram 48 of 52G). Like ncbi, one job name
            # and singleton across schemes, so SHOVILL_CONCURRENT caps the cohort's load on ENA's fastq server (§3.6 lost
            # 45 of 214 shovill builds to EBI refusing connections at 100 concurrent)
            shovill) s=shovill; chunk=1; conc=${SHOVILL_CONCURRENT:-25}
                 res=(--partition=icelake-himem --cpus-per-task=8 --mem=52G --time=03:00:00 --dependency=singleton) ;;
            esac
            name="$JOB-build-$scheme-$src"
            [[ $src == ncbi || $src == shovill ]] && name="$JOB-build-$src"
            nq=$(awk -v p="$name" '$0 == p {n++} END {print n + 0}' <<<"$queued")
            if (( nq > 0 )); then
                [[ -z $idx ]] || { echo "$name is still queued or running: wait for it, or scancel it, before rerunning tasks"; exit 1; }
                echo "$name: still queued or running — skipped (submit again once it is done to fill any gaps)"; continue
            fi
            n=$(uv run --no-sync python -m bacpredict.engine.genomes.build_genomes --manifest "$f" --sources "$s" \
                    --chunk-size "$chunk" --count-chunks)
            (( n > 0 )) || continue
            (( n <= MAX_ARRAY )) || { echo "$scheme $src: $n tasks is over MaxArraySize $MAX_ARRAY"; exit 1; }
            todo=${idx:-$(reuse missing-tasks --manifest "$f" --source "$s" --shard-dir "$SHARDS/$scheme")}
            [[ -n $todo ]] || { echo "$name: all $n tasks have written a shard"; continue; }
            [[ -n $idx ]] && arr=$todo || arr="$todo%$conc"
            submit --array="$arr" "${res[@]}" --job-name="$name" --export="ALL,$common,SOURCES=$s,CHUNK=$chunk" "$ARRAY"
        done
    done
    ;;
qc)  # qc checkm2 | qc checkm2-rerun <list number> <indices> | qc skani
    mapfile -t L < <(lists)
    case ${2:-} in
    checkm2)
        p=$(pin) || exit 1
        checkm2_ready
        k=$(printf '%02d' $(( ${#L[@]} + 1 )))
        reuse qc-list --manifest "$GEN/genome_manifest.csv" --shard-root "$SHARDS" ${L[@]:+--listed "${L[@]}"} \
            --out "$QCD/checkm2_list_$k.tsv"
        submit_checkm2 "$QCD/checkm2_list_$k.tsv" "" "$p" "$cbin"
        [[ ! -e $QCD/skani/skani_dist.tsv ]] || echo "a skani result exists: rerun qc skani once the build is done"
        ;;
    checkm2-rerun)
        p=$(pin) || exit 1
        checkm2_ready
        list=$QCD/checkm2_list_$(printf '%02d' "${3:?list number}").tsv
        [[ -s $list ]] || { echo "no $list"; exit 1; }
        submit_checkm2 "$list" "${4:?indices}" "$p" "$cbin"
        ;;
    skani)
        refuse_if_queued "$JOB-build" "skani must see every built genome"
        refuse_if_queued "$JOB-skani" "wait for it, or scancel it, before submitting another"
        (( ${#L[@]} )) || { echo "no CheckM2 list yet: qc checkm2 first"; exit 1; }
        p=$(pin) || exit 1
        tb=$(tools_bin)
        [[ -x $tb/skani ]] || { echo "no skani in $tb"; exit 1; }
        reuse qc-list --manifest "$GEN/genome_manifest.csv" --shard-root "$SHARDS" --listed "${L[@]}" --require-complete
        mkdir -p "$QCD/skani"
        while IFS=$'\t' read -r sp path; do
            [[ -s $path ]] || { echo "reference $path ($sp) is not on disk"; exit 1; }
        done <<<"$REFS"
        cat "${L[@]}" > "$QCD/skani/skani_list.tsv"
        printf '%s\n' "$REFS" > "$QCD/skani/skani_refs.tsv"
        submit --partition=${QC_PARTITION:-icelake} --cpus-per-task=32 --mem=64G --time=01:00:00 --job-name="$JOB-skani" \
            --export="ALL,$p,LIST=$QCD/skani/skani_list.tsv,REFS_TSV=$QCD/skani/skani_refs.tsv,QC_DIR=$QCD/skani,TOOLS_BIN=$tb" \
            src/bacpredict/engine/genomes/scripts/run_skani.sh
        ;;
    *) echo "qc checkm2 | qc checkm2-rerun <list number> <indices> | qc skani"; exit 2 ;;
    esac
    ;;
collate)
    refuse_if_queued "$JOB-build" "collate must see every built genome"
    refuse_if_queued "$JOB-skani" "collate must read a finished skani result"
    owners_exist
    reuse check-disjoint --manifest "$GEN/genome_manifest.csv" --built $OWNER_MANIFESTS
    mapfile -t L < <(lists)
    (( ${#L[@]} )) || { echo "no CheckM2 list yet"; exit 1; }
    reuse qc-list --manifest "$GEN/genome_manifest.csv" --shard-root "$SHARDS" --listed "${L[@]}" --require-complete
    reports=(); missing=()
    for list in "${L[@]}"; do
        k=$(basename "$list" .tsv); k=${k#checkm2_list_}
        for ((i = 0; i < $(n_tasks "$list"); i++)); do
            f=$(printf '%s/list_%s/checkm2/checkm2_quality_report_%05d.tsv' "$QCD" "$k" "$i")
            if [[ -s $f ]]; then reports+=("$f"); else missing+=("list $k task $i"); fi
        done
    done
    (( ${#missing[@]} == 0 )) || { echo "CheckM2 reports missing: ${missing[*]} (qc checkm2-rerun)"; exit 1; }
    # the result and the two lists the skani job published beside it (run_skani.sh): never skani_list.tsv, which the
    # next qc skani rewrites before its job has produced anything
    for f in "$QCD/skani/skani_dist.tsv" "$QCD/skani/skani_dist.refs.tsv" "$QCD/skani/skani_dist.list.tsv"; do
        [[ -s $f ]] || { echo "no $f: qc skani first (or its job did not finish)"; exit 1; }
    done
    CI=$GEN/collate_inputs
    reuse collate-inputs --cohort "$T/cohort.csv" --manifest "$GEN/genome_manifest.csv" --shard-root "$SHARDS" \
        --reused "$GEN/reused_genomes.csv" --reuse-shard-dir $REUSE_SHARD_DIRS --checkm2 "${reports[@]}" $REUSE_CHECKM2 \
        --skani "$QCD/skani/skani_dist.tsv" --reuse-skani $REUSE_SKANI --refs-tsv "$QCD/skani/skani_dist.refs.tsv" \
        --skani-list "$QCD/skani/skani_dist.list.tsv" --lists "${L[@]}" --out-dir "$CI"
    # the references the skani job ran with (it publishes them beside its result)
    refs=()
    while IFS=$'\t' read -r sp path; do refs+=(--species-ref "$sp=$path"); done < "$QCD/skani/skani_dist.refs.tsv"
    mode=(--species-mode "$SPECIES_MODE")
    [[ $SPECIES_MODE == target ]] && mode+=(--target-species "${TARGET_SPECIES:?}")
    # into a staging folder: nothing is published unless the MLST check below passes
    stage=$GEN/collate_staging
    rm -rf "$stage"
    uv run --no-sync python -m bacpredict.engine.genomes.collate_genomes \
        --manifest "$CI/genome_manifest.csv" --shard-dir "$CI/shards" --checkm2 "$CI/checkm2_quality_report.tsv" \
        --skani "$CI/skani_dist.tsv" "${refs[@]}" "${mode[@]}" --cohort "$T/cohort.csv" --labels "$T/binary_ast.csv" \
        --out-dir "$stage"
    # built here: on their pin. Reused: exactly the owner's scheme and ST — the owner's re-typed STs win in its shard
    # directory only by file mtime (collate_genomes.read_shards), so a copy or restore that lost them would show here
    reuse check-schemes --genome-qc "$stage/genome_qc.csv" --reused "$GEN/reused_genomes.csv" \
        --owner-genome-qc $REUSE_GENOME_QC --scheme-by-genus "$SCHEME_BY_GENUS" \
        ${SCHEME_BY_SPECIES:+--scheme-by-species "$SCHEME_BY_SPECIES"}
    for f in "$stage"/*; do mv -f "$f" "$GEN/"; done
    rmdir "$stage"
    echo "published into $GEN: $(cd "$GEN" && ls genome_qc.csv embedding_input*.csv st_counts_by_drug.csv species_crosstab.csv 2>/dev/null | tr '\n' ' ')"
    ;;
status)
    reuse status --by-scheme "$BYS" --shard-root "$SHARDS" --qc-dir "$QCD"
    ;;
*)
    sed -n '2,27p' "$0"; exit 2 ;;
esac
