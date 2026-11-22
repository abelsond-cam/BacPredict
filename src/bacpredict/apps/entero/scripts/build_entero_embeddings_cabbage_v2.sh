#!/usr/bin/env bash
# CABBAGE v2 (PROJECT_STATE §3.7): ESM-C embeddings of the Enterobacterales genomes v2 built, as
# build_entero_embeddings.sh embedded §3.6's — Klebsiella genomes linked from the Kp ESM-C store where it holds them
# protein for protein, the rest embedded; every genome §3.6 (or v2's P. aeruginosa cohort) built keeps that cohort's
# embedding. Stages and mechanics: engine/embedding/scripts/build_embeddings_reusing_built.sh.
#
# Before QC, the cohort is the genomes whose annotation has finished (David, 2026-11-22: start ESM-C on every genome
# Bakta has annotated):
#   annotate_entero_genomes_cabbage_v2.sh annotated                       → genomes/preqc_annotated.csv
#   COHORT=<that> LIVE_RUNS_OK=1 annotate_entero_genomes_cabbage_v2.sh collate → genomes/bakta_annotation_preqc_annotated.csv
#   ANNOTATION=bakta_annotation_preqc_annotated.csv <this> proteins | check-proteins | link-kp | submit | check
# After QC: ANNOTATION=bakta_annotation_embedding_input_unfiltered.csv (§3.6 embedded every genome passing the
# non-species QC, PROJECT_STATE §6, 2026-11-20); submit then embeds only the genomes still without a .pt.
#
#   ARRAY_REPO=<detached worktree> [ANNOTATION=…] bash src/bacpredict/apps/entero/scripts/build_entero_embeddings_cabbage_v2.sh <stage>
set -euo pipefail

ROOT=${BACPREDICT_AMR_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction}
[[ -d $ROOT ]] || { echo "no AMR root at $ROOT (set BACPREDICT_AMR_ROOT)"; exit 1; }
export ROOT T=$ROOT/processed/enterobacterales_cabbage_v2 JOB=v2-entero
export LINK_GENUS=Klebsiella  # the Kp ESM-C store (§3.6's link-kp)
export ANNOTATION=${ANNOTATION:-bakta_annotation_embedding_input_unfiltered.csv}
exec bash "$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)/src/bacpredict/engine/embedding/scripts/build_embeddings_reusing_built.sh" "$@"
