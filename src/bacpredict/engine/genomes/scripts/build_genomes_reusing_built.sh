#!/usr/bin/env bash
# Build a cohort's genomes into the shared per-BioSample store, reusing every genome an earlier cohort's manifest names
# — the genome stage of a second experiment over the same BioSamples (CABBAGE v2, PROJECT_STATE §3.7). Run it through a
# cohort wrapper, which sets the configuration below, from a CSD3 login node:
#
#   <wrapper> plan              partition cohort.csv (reused_genomes.csv, cohort_to_plan.csv), plan_genome_sources on
#                               the rest (genome_manifest.csv; ENA queries — submit it as a job for a large cohort),
#                               then one read-only manifest per MLST scheme (by_scheme/). Refuses if by_scheme/ exists
#   <wrapper> submit [scheme|all] [atb|portal|shovill|all] [indices]
#                               one build array per scheme × source, the scheme pinned (none for `auto`); checks
#                               disjointness from every OWNER_MANIFESTS first. Indices need one scheme and one source
#   <wrapper> qc checkm2        freeze the next CheckM2 list (built genomes in no list yet) and submit its array;
#                               run again for genomes built later
#   <wrapper> qc checkm2-rerun <list number> <indices>
#   <wrapper> qc skani          one skani job over every list (the build complete, every built genome listed)
#   <wrapper> collate           collate_inputs/ (this cohort's builds + the reused genomes' rows from their owner),
#                               then collate_genomes over it → genome_qc.csv, embedding_input(_unfiltered).csv, …
#   <wrapper> status
#
# Mechanics are §3.6's (engine/genomes/, apps/pa/CLAUDE.md); what is new is engine/genomes/reuse_built_genomes.py.
# A genome in an earlier cohort's manifest is never built here: build_genomes.build_one deletes and rebuilds a genome
# whose stored src= header differs from its manifest row's source, so naming it again could destroy the owner's copy.
#
# Configuration (exported by the wrapper):
#   ROOT, T            the AMR root; the cohort's folder (cohort.csv, binary_ast.csv; genomes/ is written under it)
#   JOB                job-name prefix
#   OWNER_MANIFESTS    every other cohort's genome_manifest.csv that writes into the store (all must exist)
#   REUSE_SHARD_DIRS, REUSE_CHECKM2, REUSE_SKANI
#                      the owning cohort's build shards and QC reports, for the genomes this cohort reuses
#   SCHEME_BY_GENUS    Genus=mlst scheme,…; any other genus is built without --scheme (`auto`)
#   GENOME_SIZE        bp, any genus not in GENOME_SIZES (optional Genus=bp,…); MIN_DEPTH (default 10)
#   REFS               species<TAB>reference path, one per line (the skani panel; must be the owner's, so the reused
#                      skani rows and this cohort's share one reference list)
#   SPECIES_MODE       target (with TARGET_SPECIES) or report — as the owner's collate
#   ATB_INDEX          the ATB S3 inventory folder
# Arrays and QC jobs run from ARRAY_REPO, a detached worktree at the SHA to run (.venv symlinked to the shared env),
# and refuse to start if its HEAD has moved (EXPECT_SHA).
set -euo pipefail

: "${ROOT:?}" "${T:?}" "${JOB:?}" "${OWNER_MANIFESTS:?}" "${REUSE_SHARD_DIRS:?}" "${REUSE_CHECKM2:?}" "${REUSE_SKANI:?}"
: "${SCHEME_BY_GENUS:?}" "${GENOME_SIZE:?}" "${REFS:?}" "${SPECIES_MODE:?}" "${ATB_INDEX:?}"
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$REPO"
# this checkout's code even though the shared .venv's editable install points at the main checkout
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
GEN=$T/genomes
BYS=$GEN/by_scheme       # genome_manifest_<scheme>.csv, frozen: array tasks take rows by position
SHARDS=$GEN/shards       # one directory per scheme (shard file names repeat across arrays)
QCD=$GEN/qc
STORE=$ROOT/raw          # build_genomes writes contigs/ and prodigal/ under it
MIN_DEPTH=${MIN_DEPTH:-10}
CHECKM2_CHUNK=2000       # task k takes rows k*2000+1 … of its list: fixed, never overridden between qc and its reruns
CHECKM2_DB=${CHECKM2_DB:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/CheckM2/Zenodo_db/CheckM2_database/uniref100.KO.1.dmnd}
# the pixi project whose envs hold shovill/prodigal/mlst/skani and CheckM2: the shared checkout's, so a worktree never
# installs a second copy (its env is keyed by the project's path)
PIXI_DIR=${PIXI_DIR:-$HOME/workspace/BacPredict/src/bacpredict/engine/genomes}
ARRAY=src/bacpredict/engine/genomes/scripts/build_genomes_array.sh
MAX_ARRAY=10000          # CSD3 MaxArraySize

reuse() { uv run --no-sync python -m bacpredict.engine.genomes.reuse_built_genomes "$@"; }
chunks() { uv run --no-sync python -m bacpredict.engine.genomes.build_genomes --manifest "$1" --sources "$2" \
               --chunk-size "$3" --count-chunks; }
tools_bin() { echo "$(cd "$PIXI_DIR" && pixi run -- printenv CONDA_PREFIX)/bin"; }
checkm2_bin() { echo "$(cd "$PIXI_DIR" && pixi run -e checkm2 -- printenv CONDA_PREFIX)/bin"; }
submit() {  # sbatch, then print the resource line the global rules require (read back from SLURM)
    local id job; id=$(sbatch --parsable "$@"); id=${id%%;*}
    job=$(scontrol show job "$id")
    echo "JOB $id $(grep -m1 -oE 'JobName=[^ ]+' <<<"$job") | CPU | $(grep -m1 -oE 'ReqTRES=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'TimeLimit=[^ ]+' <<<"$job") | $(grep -m1 -oE 'Partition=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'Account=[^ ]+' <<<"$job")"
}
# exit 1 if any of this user's jobs whose name starts with $1 is queued or running — or if squeue fails, which must
# never read as "none queued". Call directly, never in $(…), so its exit stops the script
refuse_if_queued() {
    local q n
    q=$(squeue -u "$USER" -h -o '%j') || { echo "squeue failed: cannot tell whether any $1 job is queued, so refusing"; exit 1; }
    n=$(awk -v p="$1" 'index($0, p) == 1 {n++} END {print n + 0}' <<<"$q") || exit 1
    (( n == 0 )) || { echo "$n $1 job(s) queued or running: $2"; exit 1; }
}
owners_exist() {
    local f
    for f in $OWNER_MANIFESTS; do [[ -s $f ]] || { echo "owner manifest missing: $f"; exit 1; }; done
}
pin() {  # REPO/EXPECT_SHA for --export, from ARRAY_REPO
    local r=${ARRAY_REPO:?set ARRAY_REPO to a detached worktree at the SHA to run (see the header)}
    echo "REPO=$r,EXPECT_SHA=$(git -C "$r" rev-parse HEAD)"
}
lists() { compgen -G "$QCD/checkm2_list_*.tsv" | sort || true; }
n_tasks() { echo $(( ($(wc -l < "$1") + CHECKM2_CHUNK - 1) / CHECKM2_CHUNK )); }
submit_checkm2() {  # <list> <indices or empty>
    local list=$1 k n cbin
    k=$(basename "$list" .tsv); k=${k#checkm2_list_}
    [[ -s $CHECKM2_DB ]] || { echo "no CheckM2 database at $CHECKM2_DB"; exit 1; }
    cbin=$(checkm2_bin); "$cbin/checkm2" --version
    n=$(n_tasks "$list")
    # 2,000 genomes at 32 cores: §3.6 measured a median 88 min and MaxRSS ≤ 67 GB, so 4 h and 100G (icelake's 32 × 3,380 MB)
    submit --array="${2:-0-$((n - 1))%40}" --partition=${QC_PARTITION:-icelake} --cpus-per-task=32 \
        --mem=${QC_MEM:-100G} --time=${QC_TIME:-04:00:00} --job-name="$JOB-checkm2-$k" \
        --export="ALL,$(pin),LIST=$list,QC_DIR=$QCD/list_$k,CHUNK=$CHECKM2_CHUNK,CHECKM2_BIN=$cbin,CHECKM2_DB=$CHECKM2_DB" \
        src/bacpredict/engine/genomes/scripts/run_checkm2_shard.sh
}

case "${1:-}" in
plan)
    refuse_if_queued "$JOB-build" "re-planning would shift its rows"
    owners_exist
    [[ ! -e $BYS ]] || { echo "$BYS exists: the plan is frozen (array tasks take its rows by position)"; exit 1; }
    mkdir -p "$GEN"
    reuse partition --cohort "$T/cohort.csv" --built $OWNER_MANIFESTS \
        --out-cohort "$GEN/cohort_to_plan.csv" --out-reused "$GEN/reused_genomes.csv"
    # a fresh ENA cache per cohort: the cache never expires a "no runs" answer
    uv run --no-sync python -m bacpredict.engine.genomes.plan_genome_sources \
        --cohort "$GEN/cohort_to_plan.csv" --atb-index "$ATB_INDEX"/*.csv.gz --genome-size "$GENOME_SIZE" \
        ${GENOME_SIZES:+--genome-size-by-genus "$GENOME_SIZES"} --min-depth "$MIN_DEPTH" --prefer-reads \
        --ena-cache "$GEN/ena_filereport_cache.tsv" --out "$GEN/genome_manifest.csv"
    reuse check-disjoint --manifest "$GEN/genome_manifest.csv" --built $OWNER_MANIFESTS
    reuse split-by-scheme --manifest "$GEN/genome_manifest.csv" --cohort "$T/cohort.csv" \
        --scheme-by-genus "$SCHEME_BY_GENUS" --out-dir "$BYS"
    chmod a-w "$GEN/genome_manifest.csv"
    ;;
submit)  # submit [scheme|all] [atb|portal|shovill|all] [indices]
    want_scheme=${2:-all}; want_src=${3:-all}; idx=${4:-}
    if [[ -n $idx && ( $want_scheme == all || $want_src == all ) ]]; then
        echo "indices need one scheme and one source"; exit 2
    fi
    owners_exist
    # an owner's manifest can grow after this plan: check again at every submission
    reuse check-disjoint --manifest "$GEN/genome_manifest.csv" "$BYS"/genome_manifest_*.csv --built $OWNER_MANIFESTS
    p=$(pin); tb=$(tools_bin)
    for t in shovill prodigal mlst; do [[ -x $tb/$t ]] || { echo "no $t in $tb"; exit 1; }; done
    for f in "$BYS"/genome_manifest_*.csv; do
        scheme=${f##*/genome_manifest_}; scheme=${scheme%.csv}
        [[ $want_scheme == all || $want_scheme == "$scheme" ]] || continue
        mlst=""; [[ $scheme == auto ]] || mlst=",MLST_SCHEME=$scheme"
        common="$p,MANIFEST=$f,OUT_ROOT=$STORE,SHARD_DIR=$SHARDS/$scheme,GENOME_SIZE=$GENOME_SIZE,TOOLS_BIN=$tb$mlst"
        for src in atb portal shovill; do
            [[ $want_src == all || $want_src == "$src" ]] || continue
            case $src in
            # 100 genomes per 8-core task: §3.6 measured 2.0–4.4 min and MaxRSS 1.6 GB per task
            atb) s=atb; chunk=100; conc=${ATB_CONCURRENT:-50}
                 res=(--partition=${DL_PARTITION:-icelake} --cpus-per-task=8 --mem=${DL_MEM:-6G} --time=${DL_TIME:-00:15:00}) ;;
            portal) s=portal_gff; chunk=40; conc=${PORTAL_CONCURRENT:-20}
                 res=(--partition=${DL_PARTITION:-icelake} --cpus-per-task=8 --mem=24G --time=01:30:00) ;;
            # one genome per task on icelake-himem's 6,760 MB per core (shovill --ram 48 of 52G)
            shovill) s=shovill; chunk=1; conc=${SHOVILL_CONCURRENT:-25}
                 res=(--partition=icelake-himem --cpus-per-task=8 --mem=52G --time=03:00:00) ;;
            esac
            n=$(chunks "$f" "$s" "$chunk")
            (( n > 0 )) || continue
            (( n <= MAX_ARRAY )) || { echo "$scheme $src: $n tasks is over MaxArraySize $MAX_ARRAY"; exit 1; }
            submit --array="${idx:-0-$((n - 1))%$conc}" "${res[@]}" --job-name="$JOB-build-$scheme-$src" \
                --export="ALL,$common,SOURCES=$s,CHUNK=$chunk" "$ARRAY"
        done
    done
    ;;
qc)  # qc checkm2 | qc checkm2-rerun <list number> <indices> | qc skani
    mapfile -t L < <(lists)
    case ${2:-} in
    checkm2)
        k=$(printf '%02d' $(( ${#L[@]} + 1 )))
        reuse qc-list --manifest "$GEN/genome_manifest.csv" --shard-root "$SHARDS" ${L[@]:+--listed "${L[@]}"} \
            --out "$QCD/checkm2_list_$k.tsv"
        submit_checkm2 "$QCD/checkm2_list_$k.tsv" ""
        ;;
    checkm2-rerun)
        list=$QCD/checkm2_list_$(printf '%02d' "${3:?list number}").tsv
        [[ -s $list ]] || { echo "no $list"; exit 1; }
        submit_checkm2 "$list" "${4:?indices}"
        ;;
    skani)
        refuse_if_queued "$JOB-build" "skani must see every built genome"
        refuse_if_queued "$JOB-skani" "wait for it, or scancel it, before submitting another"
        (( ${#L[@]} )) || { echo "no CheckM2 list yet: qc checkm2 first"; exit 1; }
        reuse qc-list --manifest "$GEN/genome_manifest.csv" --shard-root "$SHARDS" --listed "${L[@]}" --require-complete
        mkdir -p "$QCD/skani"
        cat "${L[@]}" > "$QCD/skani/skani_list.tsv"
        while IFS=$'\t' read -r sp path; do
            [[ -s $path ]] || { echo "reference $path ($sp) is not on disk"; exit 1; }
        done <<<"$REFS"
        printf '%s\n' "$REFS" > "$QCD/skani/skani_refs.tsv"
        submit --partition=${QC_PARTITION:-icelake} --cpus-per-task=32 --mem=64G --time=01:00:00 --job-name="$JOB-skani" \
            --export="ALL,$(pin),LIST=$QCD/skani/skani_list.tsv,REFS_TSV=$QCD/skani/skani_refs.tsv,QC_DIR=$QCD/skani,TOOLS_BIN=$(tools_bin)" \
            src/bacpredict/engine/genomes/scripts/run_skani.sh
        ;;
    *) echo "qc checkm2 | qc checkm2-rerun <list number> <indices> | qc skani"; exit 2 ;;
    esac
    ;;
collate)
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
    for f in "$QCD/skani/skani_dist.tsv" "$QCD/skani/skani_dist.refs.tsv"; do
        [[ -s $f ]] || { echo "no $f: qc skani first"; exit 1; }
    done
    CI=$GEN/collate_inputs
    reuse collate-inputs --cohort "$T/cohort.csv" --manifest "$GEN/genome_manifest.csv" --shard-root "$SHARDS" \
        --reused "$GEN/reused_genomes.csv" --reuse-shard-dir $REUSE_SHARD_DIRS \
        --checkm2 "${reports[@]}" $REUSE_CHECKM2 --skani "$QCD/skani/skani_dist.tsv" $REUSE_SKANI --out-dir "$CI"
    # the references the skani job ran with (it publishes them beside its result); the owner's skani rows must name
    # the same files, which collate_genomes checks
    refs=()
    while IFS=$'\t' read -r sp path; do refs+=(--species-ref "$sp=$path"); done < "$QCD/skani/skani_dist.refs.tsv"
    mode=(--species-mode "$SPECIES_MODE")
    [[ $SPECIES_MODE == target ]] && mode+=(--target-species "${TARGET_SPECIES:?}")
    uv run --no-sync python -m bacpredict.engine.genomes.collate_genomes \
        --manifest "$CI/genome_manifest.csv" --shard-dir "$CI/shards" --checkm2 "$CI/checkm2_quality_report.tsv" \
        --skani "$CI/skani_dist.tsv" "${refs[@]}" "${mode[@]}" --cohort "$T/cohort.csv" --labels "$T/binary_ast.csv" \
        --out-dir "$GEN"
    ;;
status)
    reuse status --by-scheme "$BYS" --shard-root "$SHARDS" --qc-dir "$QCD"
    ;;
*)
    sed -n '2,22p' "$0"; exit 2 ;;
esac
