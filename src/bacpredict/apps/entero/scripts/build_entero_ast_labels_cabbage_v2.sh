#!/usr/bin/env bash
# Build the Enterobacterales AST labels for the CABBAGE v2 experiment (PROJECT_STATE §3.7): the same build as
# build_entero_ast_labels.sh — same genera, drugs, rule and flags — with the portal's phenotype.parquet swapped for
# CABBAGE v2 written in the portal schema (engine/ast_labels/cabbage_v2_phenotype.py). The genotype table stays the
# 2026-07 portal release: it is only consulted for portal assembly accessions and the species cross-check.
#
# Light (one CSV read + pandas) — runs inline on a CSD3 login node:
#     bash src/bacpredict/apps/entero/scripts/build_entero_ast_labels_cabbage_v2.sh
# Outputs land in <root>/processed/enterobacterales_cabbage_v2/ — never in the 2026-07 folder.
set -euo pipefail

ROOT=${BACPREDICT_AMR_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction}
[[ -d $ROOT ]] || { echo "no AMR root at $ROOT (set BACPREDICT_AMR_ROOT)"; exit 1; }
PORTAL=$ROOT/raw/ebi_amr_portal/2026-07
V2CSV=$ROOT/raw/ebi_amr_portal/cabbage_v2_20261121/cabbage_v2.csv
V2=$ROOT/processed/ebi_amr_portal/cabbage_v2_20261121/phenotype.parquet
OUT=$ROOT/processed/enterobacterales_cabbage_v2
DRUGS=ciprofloxacin,ceftazidime,meropenem,tobramycin,amikacin,imipenem,piperacillin-tazobactam,aztreonam,levofloxacin,cefepime

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
mkdir -p "$OUT"

# shared with the P. aeruginosa build; written once
[[ -s $V2 ]] || uv run python -m bacpredict.engine.ast_labels.cabbage_v2_phenotype --csv "$V2CSV" --out "$V2"

# the order comes from the breakpoint module's own taxonomy, exactly as in build_entero_ast_labels.sh
mapfile -t GENERA < <(uv run python -c "
from bacpredict.engine.ast_labels.breakpoints import FAMILY_TO_ORDER, GENUS_TO_FAMILY
print('\n'.join(sorted(g for g, f in GENUS_TO_FAMILY.items() if FAMILY_TO_ORDER.get(f) == 'Enterobacterales')))")
echo "Enterobacterales genera (${#GENERA[@]}): ${GENERA[*]}"
GENUS_ARGS=()
for g in "${GENERA[@]}"; do GENUS_ARGS+=(--genus "$g"); done

uv run python -m bacpredict.engine.ast_labels.build_binary_ast_labels \
    --phenotype "$V2" \
    --portal-source cabbage_v2 \
    --genotype "$PORTAL/genotype.parquet" \
    "${GENUS_ARGS[@]}" \
    --drugs "$DRUGS" \
    --unitless-mic-as-mg-l \
    --output-dir "$OUT"
