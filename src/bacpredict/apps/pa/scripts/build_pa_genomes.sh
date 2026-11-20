#!/usr/bin/env bash
# Build the P. aeruginosa cohort's genomes, one stage at a time, from a CSD3 login node:
#
#   bash src/bacpredict/apps/pa/scripts/build_pa_genomes.sh setup    # pixi envs + species reference genomes
#   bash src/bacpredict/apps/pa/scripts/build_pa_genomes.sh plan     # genome_manifest.csv (ENA queries, ~2 min)
#   bash src/bacpredict/apps/pa/scripts/build_pa_genomes.sh smoke    # 2 genomes per source → shards_smoke/
#   bash src/bacpredict/apps/pa/scripts/build_pa_genomes.sh submit   # the full arrays (or: submit download|shovill
#                                                                     #  [indices], e.g. submit shovill 55,98)
#   bash src/bacpredict/apps/pa/scripts/build_pa_genomes.sh qc       # CheckM2 + skani job
#   bash src/bacpredict/apps/pa/scripts/build_pa_genomes.sh collate  # genome_qc.csv, embedding_input.csv, ST counts
#
# The engine stage is organism-agnostic (src/bacpredict/engine/genomes/); what is P. aeruginosa here is the
# genome size (read depth), the PubMLST scheme and the species references: PAO1 for the target and
# PA7 for P. paraeruginosa, the neighbour the portal's own tables already confuse it with.
set -euo pipefail

# the multi-species AMR root on project_k (PROJECT_STATE §3.6 *Data layout*): per-sample stores flat, keyed by
# BioSample; per-species tables under processed/<species>/
ROOT=${BACPREDICT_AMR_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction}
[[ -d $ROOT ]] || { echo "no AMR root at $ROOT (set BACPREDICT_AMR_ROOT)"; exit 1; }
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
export REPO  # the arrays run this checkout's code, not whatever $HOME/workspace/BacPredict holds
T=$ROOT/processed/paeruginosa
GEN=$T/genomes
STORE=$ROOT/raw  # build_genomes writes contigs/ and prodigal/ under it
REFDIR=$ROOT/raw/reference_genomes
ATB_INDEX=$ROOT/raw/allthebacteria/assemblies_list_20261115
PAO1=$REFDIR/GCF_000006765.1.fna.gz
PA7=$REFDIR/GCF_000017205.1.fna.gz
GENOME_SIZE=6.6e6
MIN_DEPTH=10
SCHEME=paeruginosa
ARRAY=$REPO/src/bacpredict/engine/genomes/scripts/build_genomes_array.sh
cd "$REPO"

submit() {  # sbatch, then print the resource line the global rules require (read back from SLURM)
    local id job; id=$(sbatch --parsable "$@"); id=${id%%;*}
    job=$(scontrol show job "$id")
    echo "JOB $id $(grep -m1 -oE 'JobName=[^ ]+' <<<"$job") | CPU | $(grep -m1 -oE 'ReqTRES=[^ ]+' <<<"$job")" \
         "| $(grep -m1 -oE 'TimeLimit=[^ ]+' <<<"$job") | $(grep -m1 -oE 'Partition=[^ ]+' <<<"$job")"
}

chunks() { uv run python -m bacpredict.engine.genomes.build_genomes --manifest "$1" --sources "$2" --chunk-size "$3" --count-chunks; }
# resolve the tool env once, at submit time, so array tasks never run pixi themselves
tools_bin() { echo "$(cd "$REPO/src/bacpredict/engine/genomes" && pixi run -- printenv CONDA_PREFIX)/bin"; }

case "${1:-}" in
setup)
    (cd src/bacpredict/engine/genomes && pixi install --all \
        && pixi run -- spades.py --version && pixi run -- shovill --check && pixi run -- mlst --version \
        && pixi run -e checkm2 checkm2 --version)
    mkdir -p "$REFDIR"
    for acc in GCF_000006765.1 GCF_000017205.1; do
        [[ -s "$REFDIR/$acc.fna.gz" ]] && continue
        tmp=$(mktemp -d)
        curl -fsSL --retry 5 -o "$tmp/ds.zip" \
            "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/$acc/download?include_annotation_type=GENOME_FASTA"
        unzip -p "$tmp/ds.zip" '*.fna' | gzip > "$REFDIR/$acc.fna.gz"
        rm -rf "$tmp"
    done
    ls -la "$REFDIR" "$ATB_INDEX"
    ;;
plan)
    uv run python -m bacpredict.engine.genomes.plan_genome_sources \
        --cohort "$T/cohort.csv" --atb-index "$ATB_INDEX"/*.csv.gz \
        --genome-size "$GENOME_SIZE" --min-depth "$MIN_DEPTH" --prefer-reads \
        --ena-cache "$GEN/ena_filereport_cache.tsv" --out "$GEN/genome_manifest.csv"
    ;;
smoke)
    uv run python -I -c "
import pandas as pd, sys
m = pd.read_csv(sys.argv[1], dtype=str)
m[m.source.ne('drop')].groupby('source').head(2).to_csv(sys.argv[2], index=False)" \
        "$GEN/genome_manifest.csv" "$GEN/genome_manifest_smoke.csv"
    common="MANIFEST=$GEN/genome_manifest_smoke.csv,OUT_ROOT=$STORE,SHARD_DIR=$GEN/shards_smoke,MLST_SCHEME=$SCHEME,GENOME_SIZE=$GENOME_SIZE,TOOLS_BIN=$(tools_bin)"
    submit --array=0 --cpus-per-task=4 --time=00:30:00 --job-name=build-genomes-smoke-download \
        --export=ALL,$common,SOURCES=atb+portal_gff,CHUNK=4 "$ARRAY"
    submit --array=0-1 --cpus-per-task=8 --time=03:00:00 --job-name=build-genomes-smoke-shovill \
        --export=ALL,$common,SOURCES=shovill,CHUNK=1 "$ARRAY"
    ;;
submit)  # submit [download|shovill] — both by default
    M=$GEN/genome_manifest.csv
    common="MANIFEST=$M,OUT_ROOT=$STORE,SHARD_DIR=$GEN/shards,MLST_SCHEME=$SCHEME,GENOME_SIZE=$GENOME_SIZE,TOOLS_BIN=$(tools_bin)"
    if [[ "${2:-all}" == all || "${2:-}" == download ]]; then
        n=$(chunks "$M" atb,portal_gff 40)
        submit --array=0-$((n - 1))%40 --cpus-per-task=8 --time=01:00:00 --job-name=build-genomes-download \
            --export=ALL,$common,SOURCES=atb+portal_gff,CHUNK=40 "$ARRAY"
    fi
    if [[ "${2:-all}" == all || "${2:-}" == shovill ]]; then
        n=$(chunks "$M" shovill 1)
        # optional third argument: task indices to rerun (= shards/shovill_<index>.tsv), e.g. submit shovill 55,98
        submit --array="${3:-0-$((n - 1))%60}" --cpus-per-task=8 --time=03:00:00 --job-name=build-genomes-shovill \
            --export=ALL,$common,SOURCES=shovill,CHUNK=1 "$ARRAY"
    fi
    ;;
qc)
    submit --export=ALL,SHARD_DIR=$GEN/shards,QC_DIR=$GEN/qc,REFS="$PAO1 $PA7" \
        src/bacpredict/engine/genomes/scripts/run_genome_qc.sh
    ;;
collate)
    uv run python -m bacpredict.engine.genomes.collate_genomes \
        --manifest "$GEN/genome_manifest.csv" --shard-dir "$GEN/shards" \
        --checkm2 "$GEN/qc/checkm2_quality_report.tsv" --skani "$GEN/qc/skani_dist.tsv" \
        --species-ref "Pseudomonas aeruginosa=$PAO1" --species-ref "Pseudomonas paraeruginosa=$PA7" \
        --target-species "Pseudomonas aeruginosa" --labels "$T/binary_ast.csv" --out-dir "$GEN"
    ;;
*)
    sed -n '2,9p' "$0"; exit 2 ;;
esac
