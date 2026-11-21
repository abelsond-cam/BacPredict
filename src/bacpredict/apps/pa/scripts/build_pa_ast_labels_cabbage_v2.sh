#!/usr/bin/env bash
# Build the P. aeruginosa AST labels for the CABBAGE v2 experiment (PROJECT_STATE §3.7): the same build as
# build_pa_ast_labels.sh — same drugs, same rule, same AMR-GNN supplement, same flags — with the portal's
# phenotype.parquet swapped for CABBAGE v2 written in the portal schema (engine/ast_labels/cabbage_v2_phenotype.py).
# The genotype table stays the 2026-07 portal release: it is only consulted for portal assembly accessions and the
# species cross-check.
#
# Light (one CSV read + pandas) — runs inline on a CSD3 login node:
#     bash src/bacpredict/apps/pa/scripts/build_pa_ast_labels_cabbage_v2.sh
# Outputs land in <root>/processed/paeruginosa_cabbage_v2/ — never in the 2026-07 folder.
set -euo pipefail

ROOT=${BACPREDICT_AMR_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction}
[[ -d $ROOT ]] || { echo "no AMR root at $ROOT (set BACPREDICT_AMR_ROOT)"; exit 1; }
PORTAL=$ROOT/raw/ebi_amr_portal/2026-07
V2CSV=$ROOT/raw/ebi_amr_portal/cabbage_v2_20261121/cabbage_v2.csv
V2=$ROOT/processed/ebi_amr_portal/cabbage_v2_20261121/phenotype.parquet
GNN=$ROOT/raw/paeruginosa/amr_gnn_nguyen2026
OUT=$ROOT/processed/paeruginosa_cabbage_v2
DRUGS=ciprofloxacin,ceftazidime,meropenem,tobramycin,amikacin,imipenem,piperacillin-tazobactam,aztreonam,levofloxacin,cefepime

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
mkdir -p "$OUT"

# shared with the Enterobacterales build; written once
[[ -s $V2 ]] || uv run python -m bacpredict.engine.ast_labels.cabbage_v2_phenotype --csv "$V2CSV" --out "$V2"

uv run python -m bacpredict.apps.pa.amr_gnn_tests \
    --long "$GNN/amr_gnn_long.csv" \
    --biosample-map "$GNN/amr_gnn_isolate_biosample.csv" \
    --out "$OUT/amr_gnn_tests.csv"

uv run python -m bacpredict.engine.ast_labels.build_binary_ast_labels \
    --phenotype "$V2" \
    --portal-source cabbage_v2 \
    --genotype "$PORTAL/genotype.parquet" \
    --species "Pseudomonas aeruginosa" \
    --drugs "$DRUGS" \
    --extra-tests "$OUT/amr_gnn_tests.csv" \
    --unitless-mic-as-mg-l \
    --output-dir "$OUT"
