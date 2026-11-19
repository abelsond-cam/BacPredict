#!/usr/bin/env bash
# Build the Enterobacterales AST labels from the EBI AMR Portal: every genus of the order, the same ten drugs
# and the same rule as P. aeruginosa (EUCAST v15 from the MIC, the source's own call where there is no MIC).
# Unitless portal MICs are called as mg/L (--unitless-mic-as-mg-l) — David, 2026-11-18: MIC first.
#
# Light (one parquet filter + pandas; the portal holds ~72k Enterobacterales BioSamples) — runs inline on a
# CSD3 login node:
#     bash src/bacpredict/apps/entero/scripts/build_entero_ast_labels.sh
# Outputs land in <root>/processed/train_entero_ast/ (files listed in engine/ast_labels/build_binary_ast_labels.py).
set -euo pipefail

ROOT=${BACPREDICT_DATA_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction}
PORTAL=$ROOT/raw/ebi_amr_portal/2026-07
OUT=$ROOT/processed/train_entero_ast
DRUGS=ciprofloxacin,ceftazidime,meropenem,tobramycin,amikacin,imipenem,piperacillin-tazobactam,aztreonam,levofloxacin,cefepime

cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
mkdir -p "$OUT"

# The order is defined by the breakpoint module's own taxonomy, so the label filter and the breakpoint
# resolution can never disagree about which genera are Enterobacterales.
mapfile -t GENERA < <(uv run python -c "
from bacpredict.engine.ast_labels.breakpoints import FAMILY_TO_ORDER, GENUS_TO_FAMILY
print('\n'.join(sorted(g for g, f in GENUS_TO_FAMILY.items() if FAMILY_TO_ORDER.get(f) == 'Enterobacterales')))")
echo "Enterobacterales genera (${#GENERA[@]}): ${GENERA[*]}"
GENUS_ARGS=()
for g in "${GENERA[@]}"; do GENUS_ARGS+=(--genus "$g"); done

uv run python -m bacpredict.engine.ast_labels.build_binary_ast_labels \
    --phenotype "$PORTAL/phenotype.parquet" \
    --genotype "$PORTAL/genotype.parquet" \
    "${GENUS_ARGS[@]}" \
    --drugs "$DRUGS" \
    --unitless-mic-as-mg-l \
    --output-dir "$OUT"
