#!/usr/bin/env bash
# Build the Enterobacterales cohort's genomes (contigs + Prodigal + mlst), one stage at a time, from a CSD3 login node:
#
#   bash src/bacpredict/apps/entero/scripts/build_entero_genomes.sh plan     # genome_manifest.csv (ENA queries)
#   bash src/bacpredict/apps/entero/scripts/build_entero_genomes.sh smoke    # 1 ATB genome per genus + 2 per other
#                                                                             #  source → shards_smoke/ (sizes the arrays)
#   bash src/bacpredict/apps/entero/scripts/build_entero_genomes.sh submit   # the full arrays (or: submit atb|portal|shovill
#                                                                             #  [indices], e.g. submit atb 17,240)
#   bash src/bacpredict/apps/entero/scripts/build_entero_genomes.sh status   # shards written vs tasks, failures by cause
#
# The same engine stage as P. aeruginosa (engine/genomes/; mechanics in apps/pa/CLAUDE.md), with what differs for a
# multi-genus cohort: mlst picks its scheme per genome (no --scheme), and each genome's expected size comes from its
# genus (read depth for the reads-only genomes, and shovill's --gsize). QC (CheckM2 sharded, a multi-reference skani
# panel) and collate are the next stages and are not built yet (apps/entero/CLAUDE.md).
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
# expected genome size (bp) per genus: typical complete-genome lengths, chromosome plus usual plasmid load
GENOME_SIZES=Escherichia=5.1e6,Shigella=4.6e6,Salmonella=4.9e6,Klebsiella=5.6e6,Enterobacter=4.9e6,Serratia=5.2e6,Proteus=4.1e6,Morganella=3.9e6,Providencia=4.3e6
GENOME_SIZE=5.0e6  # any other genus
MIN_DEPTH=10
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
# array tasks take manifest rows by position: never re-plan while any build array of this cohort is queued
no_live_arrays() {
    local live; live=$(squeue -u "$USER" -h -o '%j' | grep -c '^entero-build-genomes' || true)
    (( live == 0 )) || { echo "an entero-build-genomes array is queued or running: re-planning would shift its rows"; exit 1; }
}

case "${1:-}" in
plan)
    no_live_arrays
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
    submit --array=0-$((n - 1)) --cpus-per-task=8 --mem=40G --time=00:45:00 \
        --job-name=entero-build-genomes-smoke-download --export=ALL,$common,SOURCES=atb+portal_gff,CHUNK=100 "$ARRAY"
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
        submit --array="${3:-0-$((n - 1))%${ATB_CONCURRENT:-200}}" --cpus-per-task=8 --mem=${DL_MEM:-40G} \
            --time=${DL_TIME:-01:30:00} --job-name=entero-build-genomes-atb \
            --export=ALL,$common,MANIFEST=$snap,SOURCES=atb,CHUNK=100 "$ARRAY"
    fi
    if [[ "${2:-all}" == all || "${2:-}" == portal ]]; then
        n=$(chunks "$M" portal_gff 40)
        submit --array="${3:-0-$((n - 1))%60}" --cpus-per-task=8 --mem=40G --time=01:30:00 \
            --job-name=entero-build-genomes-portal --export=ALL,$common,MANIFEST=$M,SOURCES=portal_gff,CHUNK=40 "$ARRAY"
    fi
    if [[ "${2:-all}" == all || "${2:-}" == shovill ]]; then
        n=$(chunks "$M" shovill 1)
        (( n == 0 )) || submit --array="${3:-0-$((n - 1))%100}" --cpus-per-task=8 --mem=52G --time=03:00:00 \
            --job-name=entero-build-genomes-shovill --export=ALL,$common,MANIFEST=$M,SOURCES=shovill,CHUNK=1 "$ARRAY"
    fi
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
    ;;
*)
    sed -n '2,10p' "$0"; exit 2 ;;
esac
