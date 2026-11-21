#!/usr/bin/env bash
# CABBAGE v2 (PROJECT_STATE §3.7): the P. aeruginosa cohort's genomes, built as build_pa_genomes.sh built §3.6's —
# ATB → portal GFF → shovill (--prefer-reads, 10×, 6.6 Mbp), Prodigal + mlst (scheme paeruginosa), CheckM2 + skani
# against PAO1 and PA7, collate in target mode — for the genomes no §3.6 manifest names; every other genome is §3.6's
# own, reused with its QC. Stages and mechanics: engine/genomes/scripts/build_genomes_reusing_built.sh.
#
#   ARRAY_REPO=<detached worktree> bash src/bacpredict/apps/pa/scripts/build_pa_genomes_cabbage_v2.sh <stage> …
set -euo pipefail

ROOT=${BACPREDICT_AMR_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction}
[[ -d $ROOT ]] || { echo "no AMR root at $ROOT (set BACPREDICT_AMR_ROOT)"; exit 1; }
P=$ROOT/processed
REFDIR=$ROOT/raw/reference_genomes
export ROOT T=$P/paeruginosa_cabbage_v2 JOB=v2-pa
# every cohort writing into the shared genome store: the two §3.6 cohorts (v2's Enterobacterales cohort is planned
# after this one and checks against it)
export OWNER_MANIFESTS="$P/paeruginosa/genomes/genome_manifest.csv $P/enterobacterales/genomes/genome_manifest.csv"
export REUSE_SHARD_DIRS=$P/paeruginosa/genomes/shards
export REUSE_CHECKM2=$P/paeruginosa/genomes/qc/checkm2_quality_report.tsv
export REUSE_SKANI=$P/paeruginosa/genomes/qc/skani_dist.tsv
export SCHEME_BY_GENUS=Pseudomonas=paeruginosa
export GENOME_SIZE=6.6e6
export REFS="Pseudomonas aeruginosa	$REFDIR/GCF_000006765.1.fna.gz
Pseudomonas paraeruginosa	$REFDIR/GCF_000017205.1.fna.gz"
export SPECIES_MODE=target TARGET_SPECIES="Pseudomonas aeruginosa"
export ATB_INDEX=$ROOT/raw/allthebacteria/assemblies_list_20261115
exec bash "$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)/src/bacpredict/engine/genomes/scripts/build_genomes_reusing_built.sh" "$@"
