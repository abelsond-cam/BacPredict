#!/usr/bin/env bash
# Build the Enterobacterales cohort's genomes (contigs + Prodigal + mlst), one stage at a time, from a CSD3 login node:
#
#   bash src/bacpredict/apps/entero/scripts/build_entero_genomes.sh plan     # genome_manifest.csv (ENA queries)
#   bash src/bacpredict/apps/entero/scripts/build_entero_genomes.sh smoke    # 1 ATB genome per genus + 2 per other
#                                                                             #  source → shards_smoke/ (sizes the arrays)
#   bash src/bacpredict/apps/entero/scripts/build_entero_genomes.sh submit   # the full arrays (or: submit atb|portal|shovill
#                                                                             #  [indices], e.g. submit atb 17,240)
#   bash src/bacpredict/apps/entero/scripts/build_entero_genomes.sh refs     # the skani panel's genomes (login node)
#   bash src/bacpredict/apps/entero/scripts/build_entero_genomes.sh qc       # freeze qc/checkm2_list.tsv; CheckM2 array
#                                                                             #  + skani job (or: qc checkm2 [indices] |
#                                                                             #  qc skani, reruns from the frozen list)
#   bash src/bacpredict/apps/entero/scripts/build_entero_genomes.sh collate  # genome_qc.csv, species_crosstab.csv,
#                                                                             #  embedding_input_unfiltered.csv, …
#   bash src/bacpredict/apps/entero/scripts/build_entero_genomes.sh status   # build, CheckM2 and skani progress
#
# The same engine stage as P. aeruginosa (engine/genomes/; mechanics in apps/pa/CLAUDE.md), with what differs for a
# multi-genus cohort: mlst picks its scheme per genome (no --scheme), and each genome's expected size comes from its
# genus (read depth for the reads-only genomes, and shovill's --gsize). QC is sharded: CheckM2 as an array over a
# frozen list (2,000 genomes per 32-core task), skani as one job against a multi-reference panel
# (apps/entero/species_reference_panel.tsv, a DRAFT), and collate runs in report mode: no species rule, a label ×
# called species crosstab to decide one from (apps/entero/CLAUDE.md).
#
# Contigs and Prodigal GFFs go to the flat per-BioSample stores shared with PsA ($ROOT/raw/{contigs,prodigal}); the
# manifest, shards and QC under processed/enterobacterales/genomes/. The ATB array reads a frozen, read-only snapshot
# of the manifest's atb rows (genome_manifest_atb.csv, written by plan), so re-planning the non-ATB genomes can never
# shift its task indices. Long arrays run from a pinned worktree
# (ARRAY_REPO, a detached worktree at the submitting SHA with .venv symlinked to the shared env), because array tasks
# otherwise run whatever the shared checkout's HEAD is when they start.
set -euo pipefail

# the multi-species AMR root on project_k (PROJECT_STATE §3.6 *Data layout*): per-sample stores flat, keyed by
# BioSample; per-species tables under processed/<species>/
ROOT=${BACPREDICT_AMR_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction}
[[ -d $ROOT ]] || { echo "no AMR root at $ROOT (set BACPREDICT_AMR_ROOT)"; exit 1; }
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
export REPO
T=$ROOT/processed/enterobacterales
GEN=$T/genomes
STORE=$ROOT/raw  # build_genomes writes contigs/ and prodigal/ under it
ATB_INDEX=$ROOT/raw/allthebacteria/assemblies_list_20261115
PSA_COHORT=$ROOT/processed/paeruginosa/genomes/embedding_input.csv
QCD=$GEN/qc
REFDIR=$ROOT/raw/reference_genomes
PANEL=src/bacpredict/apps/entero/species_reference_panel.tsv
# CheckM2 task k takes rows k*2000+1 … of the frozen list: fixed here, never overridden between qc and its reruns
CHECKM2_CHUNK=2000
CHECKM2_DB=${CHECKM2_DB:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/CheckM2/Zenodo_db/CheckM2_database/uniref100.KO.1.dmnd}
# expected genome size (bp) per genus: typical complete-genome lengths, chromosome plus usual plasmid load
GENOME_SIZES=Escherichia=5.1e6,Shigella=4.6e6,Salmonella=4.9e6,Klebsiella=5.6e6,Enterobacter=4.9e6,Serratia=5.2e6,Proteus=4.1e6,Morganella=3.9e6,Providencia=4.3e6
GENOME_SIZE=5.0e6  # any other genus
MIN_DEPTH=10
# download tasks (one genome per core, Prodigal + mlst) need little memory, so they go to icelake (3,380 MB per core),
# which had ~7,400 idle cores when icelake-himem had a 3 h queue (2026-11-20); shovill keeps himem's 52 GB
ARRAY=src/bacpredict/engine/genomes/scripts/build_genomes_array.sh
cd "$REPO"

submit() {  # sbatch, then print the resource line the global rules require (read back from SLURM)
    local id job; id=$(sbatch --parsable "$@"); id=${id%%;*}
    job=$(scontrol show job "$id")
    echo "JOB $id $(grep -m1 -oE 'JobName=[^ ]+' <<<"$job") | CPU | $(grep -m1 -oE 'ReqTRES=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'TimeLimit=[^ ]+' <<<"$job") | $(grep -m1 -oE 'Partition=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'Account=[^ ]+' <<<"$job")"
}
chunks() { uv run --no-sync python -m bacpredict.engine.genomes.build_genomes --manifest "$1" --sources "$2" \
               --chunk-size "$3" --count-chunks; }
tools_bin() { echo "$(cd "$REPO/src/bacpredict/engine/genomes" && pixi run -- printenv CONDA_PREFIX)/bin"; }
checkm2_bin() { echo "$(cd "$REPO/src/bacpredict/engine/genomes" && pixi run -e checkm2 -- printenv CONDA_PREFIX)/bin"; }
panel() { grep -v '^#' "$PANEL" | tail -n +2 | cut -f1,2; }  # species<TAB>accession, the DRAFT panel's rows
n_checkm2_tasks() { echo $(( ($(wc -l < "$QCD/checkm2_list.tsv") + CHECKM2_CHUNK - 1) / CHECKM2_CHUNK )); }
# refuse_if_queued <job-name prefix> <why>: exit 1 if any of this user's jobs whose name starts with the prefix is
# queued or running — or if squeue itself fails, which must never read as "none queued" (as `grep -c … || true` on a
# failed squeue's empty output would). Call it directly, never in $(…), so its exit stops the script.
refuse_if_queued() {
    local q n
    q=$(squeue -u "$USER" -h -o '%j') || { echo "squeue failed: cannot tell whether any $1 job is queued, so refusing"; exit 1; }
    n=$(awk -v p="$1" 'index($0, p) == 1 {n++} END {print n + 0}' <<<"$q") || exit 1
    (( n == 0 )) || { echo "$n $1 job(s) queued or running: $2"; exit 1; }
}

case "${1:-}" in
plan)
    # array tasks take manifest rows by position: never re-plan while any build array of this cohort is queued
    refuse_if_queued entero-build-genomes "re-planning would shift its rows"
    mkdir -p "$GEN"
    # a fresh ENA cache per cohort: the cache never expires a "no runs" answer
    uv run --no-sync python -m bacpredict.engine.genomes.plan_genome_sources \
        --cohort "$T/cohort.csv" --atb-index "$ATB_INDEX"/*.csv.gz \
        --genome-size "$GENOME_SIZE" --genome-size-by-genus "$GENOME_SIZES" --min-depth "$MIN_DEPTH" --prefer-reads \
        --ena-cache "$GEN/ena_filereport_cache.tsv" --out "$GEN/genome_manifest.csv"
    # the contig store is shared with PsA and keyed by BioSample: a genome in both cohorts would be rebuilt over
    uv run --no-sync python -I - "$GEN/genome_manifest.csv" "$PSA_COHORT" <<'EOF'
import sys
import pandas as pd
m = pd.read_csv(sys.argv[1], dtype=str)
both = set(m.loc[m["source"].ne("drop"), "Sample"]) & set(pd.read_csv(sys.argv[2], dtype=str)["Sample"])
if both:
    sys.exit(f"{len(both)} genomes are in both the Enterobacterales and the PsA cohort, e.g. {sorted(both)[:5]}")
print("no genome shared with the PsA cohort")
EOF
    # the ATB array's own manifest: the atb rows only, frozen (task k takes rows by position)
    snap=$GEN/genome_manifest_atb.csv
    [[ -e $snap ]] && chmod u+w "$snap"
    uv run --no-sync python -I -c "
import pandas as pd, sys
m = pd.read_csv(sys.argv[1], dtype=str)
m[m.source.eq('atb')].to_csv(sys.argv[2], index=False)" "$GEN/genome_manifest.csv" "$snap"
    chmod a-w "$snap"
    echo "ATB snapshot: $(($(wc -l < "$snap") - 1)) genomes → $snap"
    ;;
smoke)
    # one ATB genome of every genus (mlst's scheme choice, sizes) plus 2 of each other source; sizes the arrays
    uv run --no-sync python -I - "$GEN/genome_manifest.csv" "$T/cohort.csv" "$GEN/genome_manifest_smoke.csv" <<'EOF'
import sys
import pandas as pd
m = pd.read_csv(sys.argv[1], dtype=str)
genus = pd.read_csv(sys.argv[2], dtype=str).set_index("Sample")["species"].str.split().str[0]
m["genus"] = m["Sample"].map(genus)
atb = m[m["source"].eq("atb")].groupby("genus").head(1)
other = m[m["source"].isin(["portal_gff", "shovill"])].groupby("source").head(2)
pd.concat([atb, other]).drop(columns="genus").sort_values("Sample").to_csv(sys.argv[3], index=False)
print(pd.concat([atb, other])[["Sample", "source", "genus"]].to_string(index=False))
EOF
    common="REPO=$REPO,EXPECT_SHA=$(git rev-parse HEAD),MANIFEST=$GEN/genome_manifest_smoke.csv,OUT_ROOT=$STORE"
    common+=",SHARD_DIR=$GEN/shards_smoke,GENOME_SIZE=$GENOME_SIZE,TOOLS_BIN=$(tools_bin)"
    n=$(chunks "$GEN/genome_manifest_smoke.csv" atb,portal_gff 100)
    submit --array=0-$((n - 1)) --partition=${DL_PARTITION:-icelake} --cpus-per-task=8 --mem=${DL_MEM:-24G} \
        --time=00:45:00 --job-name=entero-build-genomes-smoke-download --export=ALL,$common,SOURCES=atb+portal_gff,CHUNK=100 "$ARRAY"
    n=$(chunks "$GEN/genome_manifest_smoke.csv" shovill 1)
    (( n == 0 )) || submit --array=0-$((n - 1)) --cpus-per-task=8 --mem=52G --time=03:00:00 \
        --job-name=entero-build-genomes-smoke-shovill --export=ALL,$common,SOURCES=shovill,CHUNK=1 "$ARRAY"
    ;;
submit)  # submit [atb|portal|shovill] [indices] — all three, all indices, by default
    M=$GEN/genome_manifest.csv
    array_repo=${ARRAY_REPO:?set ARRAY_REPO to a detached worktree at the SHA to run (see the header)}
    common="REPO=$array_repo,EXPECT_SHA=$(git -C "$array_repo" rev-parse HEAD),OUT_ROOT=$STORE"
    common+=",SHARD_DIR=$GEN/shards,GENOME_SIZE=$GENOME_SIZE,TOOLS_BIN=$(tools_bin)"
    if [[ "${2:-all}" == all || "${2:-}" == atb ]]; then
        # 100 genomes per 8-core task (one genome per core), from the frozen snapshot; wall and memory from the smoke
        snap=$GEN/genome_manifest_atb.csv
        n=$(chunks "$snap" atb 100)
        submit --array="${3:-0-$((n - 1))%${ATB_CONCURRENT:-200}}" --partition=${DL_PARTITION:-icelake} \
            --cpus-per-task=8 --mem=${DL_MEM:-24G} \
            --time=${DL_TIME:-01:30:00} --job-name=entero-build-genomes-atb \
            --export=ALL,$common,MANIFEST=$snap,SOURCES=atb,CHUNK=100 "$ARRAY"
    fi
    if [[ "${2:-all}" == all || "${2:-}" == portal ]]; then
        n=$(chunks "$M" portal_gff 40)
        submit --array="${3:-0-$((n - 1))%60}" --partition=${DL_PARTITION:-icelake} --cpus-per-task=8 \
            --mem=${DL_MEM:-24G} --time=01:30:00 \
            --job-name=entero-build-genomes-portal --export=ALL,$common,MANIFEST=$M,SOURCES=portal_gff,CHUNK=40 "$ARRAY"
    fi
    if [[ "${2:-all}" == all || "${2:-}" == shovill ]]; then
        n=$(chunks "$M" shovill 1)
        (( n == 0 )) || submit --array="${3:-0-$((n - 1))%100}" --cpus-per-task=8 --mem=52G --time=03:00:00 \
            --job-name=entero-build-genomes-shovill --export=ALL,$common,MANIFEST=$M,SOURCES=shovill,CHUNK=1 "$ARRAY"
    fi
    ;;
refs)
    # the panel's reference genomes, as build_pa_genomes.sh setup fetches PsA's; skipped where already on disk
    mkdir -p "$REFDIR"
    panel | cut -f2 | sort -u | while read -r acc; do
        [[ -s "$REFDIR/$acc.fna.gz" ]] && continue
        tmp=$(mktemp -d)
        curl -fsSL --retry 5 -o "$tmp/ds.zip" \
            "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/$acc/download?include_annotation_type=GENOME_FASTA"
        unzip -p "$tmp/ds.zip" '*.fna' | gzip > "$REFDIR/$acc.fna.gz.part"
        mv "$REFDIR/$acc.fna.gz.part" "$REFDIR/$acc.fna.gz"
        rm -rf "$tmp"
        echo "$acc → $REFDIR/$acc.fna.gz"
    done
    n=$(panel | wc -l); have=0
    while IFS=$'\t' read -r sp acc; do [[ -s "$REFDIR/$acc.fna.gz" ]] && have=$((have + 1)); done < <(panel)
    echo "panel: $have of $n references on disk in $REFDIR"
    (( have == n ))
    ;;
qc)  # qc — freeze the list, submit both; qc checkm2 [indices] | qc skani — rerun from the frozen list
    LIST=$QCD/checkm2_list.tsv
    what=${2:-all}
    [[ $what == all || $what == checkm2 || $what == skani ]] || { echo "qc [checkm2 [indices] | skani]"; exit 2; }
    # tasks run from a pinned worktree, exactly as for submit: the shared checkout's HEAD moves under queued tasks.
    # Checked first, so a missing ARRAY_REPO never leaves a frozen list with nothing submitted against it
    array_repo=${ARRAY_REPO:?set ARRAY_REPO to a detached worktree at the SHA to run (see the header)}
    pin="REPO=$array_repo,EXPECT_SHA=$(git -C "$array_repo" rev-parse HEAD)"
    # one skani job at a time: two would race each other to qc/skani_dist.tsv and its reference list (checked before
    # anything is frozen or submitted)
    [[ $what == checkm2 ]] || refuse_if_queued entero-skani "wait for it, or scancel it, before submitting another"
    if [[ ! -e $LIST ]]; then
        [[ $what == all ]] || { echo "no frozen list yet: run qc with no arguments first"; exit 1; }
        refuse_if_queued entero-build-genomes "its genomes would miss CheckM2"
        mkdir -p "$QCD"
        # every built genome (status ok), from the newest shard row per genome (as collate reads them), sorted; the
        # build must be complete, so no genome can become ok after the list is frozen
        uv run --no-sync python -I - "$GEN/genome_manifest.csv" "$GEN/shards" "$LIST" <<'EOF'
import sys
from pathlib import Path
import pandas as pd
from bacpredict.engine.genomes.collate_genomes import read_shards
m = pd.read_csv(sys.argv[1], dtype=str)
want = set(m.loc[m["source"].ne("drop"), "Sample"])
s = read_shards(Path(sys.argv[2]))
s = s[s["Sample"].isin(want)]
if missing := want - set(s["Sample"]):
    sys.exit(f"{len(missing)} of {len(want)} buildable genomes have no shard row yet, e.g. {sorted(missing)[:5]}: "
             "finish the build (status) before freezing the CheckM2 list")
ok = s[s["status"].eq("ok")].sort_values("Sample")
if not ok["contigs_path"].str.startswith("/").all():
    sys.exit("contigs paths must be absolute")
ok[["Sample", "contigs_path"]].to_csv(sys.argv[3] + ".part", sep="\t", header=False, index=False)
Path(sys.argv[3] + ".part").replace(sys.argv[3])
print(f"CheckM2 list: {len(ok)} built genomes of {len(want)} buildable ({len(s) - len(ok)} failed to build) → {sys.argv[3]}")
EOF
        chmod a-w "$LIST"
    elif [[ $what == all ]]; then
        echo "$LIST is frozen (task k takes rows by position): rerun with qc checkm2 <indices>, or qc skani"; exit 1
    fi
    if [[ $what == all || $what == checkm2 ]]; then
        [[ -s $CHECKM2_DB ]] || { echo "no CheckM2 database at $CHECKM2_DB"; exit 1; }
        cbin=$(checkm2_bin)
        "$cbin/checkm2" --version  # runs from its bin without pixi, as the tasks will
        n=$(n_checkm2_tasks)
        submit --array="${3:-0-$((n - 1))%40}" --partition=${QC_PARTITION:-icelake} --cpus-per-task=32 \
            --mem=${QC_MEM:-100G} --time=${QC_TIME:-04:00:00} --job-name=entero-checkm2 \
            --export=ALL,$pin,LIST=$LIST,QC_DIR=$QCD,CHUNK=$CHECKM2_CHUNK,CHECKM2_BIN=$cbin,CHECKM2_DB=$CHECKM2_DB \
            src/bacpredict/engine/genomes/scripts/run_checkm2_shard.sh
        echo "  CheckM2: $n tasks × $CHECKM2_CHUNK genomes; ~1.7 h each at PsA's 1.6 core-min per genome, so 4 h;" \
             "memory unmeasured, 100G fits icelake's 32 × 3,380 MB (QC_PARTITION=icelake-himem QC_MEM=200G if not)"
    fi
    if [[ $what == all || $what == skani ]]; then
        # skani's input only: the job copies it at its start and publishes that copy as qc/skani_dist.refs.tsv
        # beside qc/skani_dist.tsv, which is what collate reads, so a later rewrite of this file cannot drift from
        # a result
        panel | while IFS=$'\t' read -r sp acc; do
            [[ -s "$REFDIR/$acc.fna.gz" ]] || { echo "reference $acc is not on disk: run refs" >&2; exit 1; }
            printf '%s\t%s\n' "$sp" "$REFDIR/$acc.fna.gz"
        done > "$QCD/skani_refs.tsv.part"
        mv "$QCD/skani_refs.tsv.part" "$QCD/skani_refs.tsv"
        submit --partition=${QC_PARTITION:-icelake} --cpus-per-task=32 --mem=64G --time=01:00:00 \
            --job-name=entero-skani \
            --export="ALL,$pin,LIST=$LIST,REFS_TSV=$QCD/skani_refs.tsv,QC_DIR=$QCD,TOOLS_BIN=$(tools_bin)" \
            src/bacpredict/engine/genomes/scripts/run_skani.sh
        echo "  skani: $(wc -l < "$LIST") genomes × $(wc -l < "$QCD/skani_refs.tsv") references in slices of 5,000;" \
             "minutes expected (PsA: seconds for 3,773 × 2), so 1 h; 64G bounds 5,000 sketches"
    fi
    ;;
collate)
    LIST=$QCD/checkm2_list.tsv
    # the skani result and the reference list it ran with, both written by the skani job in one step
    SKANI_REFS=$QCD/skani_dist.refs.tsv
    for f in "$LIST" "$QCD/skani_dist.tsv" "$SKANI_REFS"; do
        [[ -s $f ]] || { echo "no $f: run qc (or qc skani for the skani pair) first (status)"; exit 1; }
    done
    reports=(); missing=(); n=$(n_checkm2_tasks)
    for ((k = 0; k < n; k++)); do
        f=$(printf '%s/checkm2/checkm2_quality_report_%05d.tsv' "$QCD" "$k")
        if [[ -s $f ]]; then reports+=("$f"); else missing+=("$k"); fi
    done
    (( ${#missing[@]} == 0 )) || { echo "CheckM2 tasks without a report: $(IFS=,; echo "${missing[*]}") (qc checkm2 <indices>)"; exit 1; }
    refs=()
    while IFS=$'\t' read -r sp path; do refs+=(--species-ref "$sp=$path"); done < "$SKANI_REFS"
    # report mode: no species rule yet — it is decided from species_crosstab.csv (label × call) with David. So the
    # QC pass set goes to embedding_input_unfiltered.csv; embedding_input.csv waits for a target-mode collate
    uv run --no-sync python -m bacpredict.engine.genomes.collate_genomes \
        --manifest "$GEN/genome_manifest.csv" --shard-dir "$GEN/shards" \
        --checkm2 "${reports[@]}" --skani "$QCD/skani_dist.tsv" "${refs[@]}" \
        --species-mode report --cohort "$T/cohort.csv" --labels "$T/binary_ast.csv" --out-dir "$GEN"
    ;;
status)
    uv run --no-sync python -I - "$GEN/genome_manifest.csv" "$GEN/shards" <<'EOF'
import sys
from pathlib import Path
import pandas as pd
m = pd.read_csv(sys.argv[1], dtype=str)
shards = sorted(Path(sys.argv[2]).glob("*.tsv"))
rows = pd.concat([pd.read_csv(f, sep="\t", dtype=str) for f in shards]) if shards else pd.DataFrame(columns=["Sample", "status"])
for src, chunk in (("atb", 100), ("portal_gff", 40), ("shovill", 1)):
    want = m[m["source"].isin(src.split("+"))].sort_values("Sample").reset_index(drop=True)
    have = {int(f.stem.rsplit("_", 1)[1]) for f in shards if f.stem.startswith(src + "_")}
    n = (len(want) + chunk - 1) // chunk
    missing = sorted(set(range(n)) - have)
    print(f"{src}: {len(have)}/{n} tasks written; missing tasks: {','.join(map(str, missing[:40]))}{' …' if len(missing) > 40 else ''}")
ok = rows["status"].eq("ok")
print(f"genomes: {int(ok.sum())} ok, {int((~ok).sum())} failed, of {int(m['source'].ne('drop').sum())} to build")
if (~ok).any():
    print(rows.loc[~ok, "status"].str.slice(0, 60).value_counts().head(10).to_string())
EOF
    uv run --no-sync python -I - "$QCD" "$CHECKM2_CHUNK" <<'EOF'
import sys
from pathlib import Path
import pandas as pd
qcd, chunk = Path(sys.argv[1]), int(sys.argv[2])
lst = qcd / "checkm2_list.tsv"
if not lst.exists():
    print("QC: no frozen CheckM2 list yet (qc)")
    sys.exit(0)
want = pd.read_csv(lst, sep="\t", header=None, names=["Sample", "contigs"], dtype=str)["Sample"]
n = -(-len(want) // chunk)
reports = {int(p.stem.rsplit("_", 1)[1]): p for p in (qcd / "checkm2").glob("checkm2_quality_report_*.tsv")}
missing = sorted(set(range(n)) - set(reports))
# Name normalised as collate_genomes.read_checkm2 does (a trailing .fna or .fna.gz dropped)
got = pd.concat([pd.read_csv(p, sep="\t", usecols=["Name"], dtype=str) for p in reports.values()])["Name"] \
    .str.replace(r"\.fna(\.gz)?$", "", regex=True) if reports else pd.Series(dtype=str)
print(f"CheckM2: {len(reports)}/{n} tasks written; missing tasks: {','.join(map(str, missing))}; "
      f"{int(want.isin(got).sum())} of {len(want)} listed genomes reported")
sk = qcd / "skani_dist.tsv"
if sk.exists():
    hit = pd.read_csv(sk, sep="\t", usecols=["Query_file"])["Query_file"].map(lambda p: Path(p).name.split(".")[0])
    print(f"skani: {hit.nunique()} of {len(want)} listed genomes within ~80 % ANI of a panel reference")
    if not (qcd / "skani_dist.refs.tsv").exists():
        print("skani: no skani_dist.refs.tsv beside it (the job's own reference list), so collate refuses: qc skani")
else:
    print("skani: skani_dist.tsv not written")
xt = qcd.parent / "species_crosstab.csv"
if xt.exists():
    print(f"collate: {xt} written")
EOF
    ;;
*)
    sed -n '2,16p' "$0"; exit 2 ;;
esac
