#!/usr/bin/env bash
# CABBAGE v2 (PROJECT_STATE §3.7): ESM-C embeddings of the P. aeruginosa genomes v2 built, as build_pa_embeddings.sh
# embedded §3.6's; every genome §3.6 built keeps its §3.6 embedding. The cohort is genomes/bakta_annotation.csv (the
# annotate wrapper's collate of embedding_input.csv). Stages and mechanics:
# engine/embedding/scripts/build_embeddings_reusing_built.sh.
#
#   ARRAY_REPO=<detached worktree> bash src/bacpredict/apps/pa/scripts/build_pa_embeddings_cabbage_v2.sh <stage>
set -euo pipefail

ROOT=${BACPREDICT_AMR_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction}
[[ -d $ROOT ]] || { echo "no AMR root at $ROOT (set BACPREDICT_AMR_ROOT)"; exit 1; }
export ROOT T=$ROOT/processed/paeruginosa_cabbage_v2 JOB=v2-pa
export LINK_GENUS=  # no existing store of P. aeruginosa ESM-C embeddings to reuse
exec bash "$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)/src/bacpredict/engine/embedding/scripts/build_embeddings_reusing_built.sh" "$@"
