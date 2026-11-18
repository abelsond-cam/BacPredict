#!/usr/bin/env bash
# Build the P. aeruginosa AST labels: AMR-GNN supplement → test table, then portal + AMR-GNN → binary_ast.csv.
#
# Light (one parquet filter + pandas on ~30k tests, ~1 min) — runs inline on a CSD3 login node:
#     bash src/bacpredict/apps/pa/scripts/build_pa_ast_labels.sh
# Inputs are mapped in src/bacpredict/docs/HPC_DATA.md; outputs land in <root>/processed/train_pa_ast/.
set -euo pipefail

ROOT=${BACPREDICT_DATA_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction}
PORTAL=$ROOT/raw/ebi_amr_portal/2026-07
GNN=$ROOT/raw/amr_gnn_nguyen2026
OUT=$ROOT/processed/train_pa_ast
DRUGS=ciprofloxacin,ceftazidime,meropenem,tobramycin,amikacin,imipenem,piperacillin-tazobactam,aztreonam,levofloxacin,cefepime

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
mkdir -p "$OUT"

uv run python -m bacpredict.apps.pa.amr_gnn_tests \
    --long "$GNN/amr_gnn_long.csv" \
    --biosample-map "$GNN/amr_gnn_isolate_biosample.csv" \
    --out "$OUT/amr_gnn_tests.csv"

uv run python -m bacpredict.engine.ast_labels.build_binary_ast_labels \
    --phenotype "$PORTAL/phenotype.parquet" \
    --genotype "$PORTAL/genotype.parquet" \
    --species "Pseudomonas aeruginosa" \
    --drugs "$DRUGS" \
    --extra-tests "$OUT/amr_gnn_tests.csv" \
    --output-dir "$OUT"
