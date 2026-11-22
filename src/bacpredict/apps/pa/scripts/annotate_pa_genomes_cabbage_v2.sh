#!/usr/bin/env bash
# CABBAGE v2 (PROJECT_STATE §3.7): Bakta 1.9.4 on the P. aeruginosa cohort's QC-passing genomes, as
# annotate_pa_genomes.sh annotated §3.6's — BakRep's GFF3 where BakRep holds the genome on our ATB contigs, our run of
# the same release, flags, env and database otherwise — for the genomes v2 built; every genome §3.6 built keeps §3.6's
# annotation. Stages and mechanics: engine/genomes/scripts/annotate_genomes_reusing_built.sh.
#
#   ARRAY_REPO=<detached worktree> bash src/bacpredict/apps/pa/scripts/annotate_pa_genomes_cabbage_v2.sh <stage> …
set -euo pipefail

ROOT=${BACPREDICT_AMR_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction}
[[ -d $ROOT ]] || { echo "no AMR root at $ROOT (set BACPREDICT_AMR_ROOT)"; exit 1; }
P=$ROOT/processed
export ROOT T=$P/paeruginosa_cabbage_v2 JOB=v2-pa
export COHORT_DEFAULT=embedding_input.csv  # target mode, as §3.6's PsA
# the Bakta jobs of every cohort whose genomes this one reuses: §3.6's PsA and Enterobacterales
export OWNER_BAKTA_JOBS=pa-bakta-ingest,pa-bakta,entero-bakta-ingest,entero-bakta-all,entero-bakta-planned,entero-bakta-refused
export REF_RUN_ROSTER=$P/paeruginosa/genomes/bakta/run_roster_with_refused.tsv
export BAKREP_EXPORT=$ROOT/raw/bakrep/v2_export_20250612/bakrep-export.tsv.gz
exec bash "$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)/src/bacpredict/engine/genomes/scripts/annotate_genomes_reusing_built.sh" "$@"
