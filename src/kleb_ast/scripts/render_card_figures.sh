#!/bin/bash
# Re-render the full per-antibiotic CARD figure set from the committed determinant / per-gene / ingredient
# CSVs — pure matplotlib, login-node / laptop safe (no GPU, seconds per drug). Run this after the
# card_determinant_lr / gene_ingredient_concat compute jobs land their CSVs, to refresh the figures:
#
#     bash src/kleb_ast/scripts/render_card_figures.sh
#
# Renders, per drug:
#   #1 per-gene ESM-vs-FT (family)            -> card_esm_vs_ft_per_gene_<drug>_family.png   (CARD ceiling line)
#   #2 CARD cause histogram (family + allele) -> <drug>_card_cause_histogram_<grain>.png
#   #3 CARD ladder (family)                   -> <drug>_card_ladder_family.png
# and once across all drugs:
#   #4 combined panel (family + allele)       -> kp_card_summary_panel_<grain>.png
#   #5 gene-ingredient concat summary         -> ingredient/gene_ingredient_concat_<mean>.png
set -euo pipefail
cd "$(dirname "$0")/../../.."        # repo root

VIS_SUB="amr_per_abx"                    # figure-folder name under docs/visualisations
VIS="src/kleb_ast/docs/visualisations/${VIS_SUB}"

DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
       gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
       levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)

for drug in "${DRUGS[@]}"; do
    dir="${VIS}/kp_${drug}"
    [ -f "${dir}/card_determinant_lr_${drug}_family.csv" ] || { echo "skip ${drug} (no determinant CSV)"; continue; }

    # #2 CARD cause histogram — both grains (mutation-aware: GyrA (mut)/(WT) bars + __ALL_CARD__ ceiling)
    for grain in family allele; do
        uv run python -m kleb_ast.plot_kleborate_cause_histogram \
            --drug "${drug}" --source-name CARD --all-key __ALL_CARD__ --grain "${grain}" \
            --csv "${dir}/card_determinant_lr_${drug}_${grain}.csv" \
            --out "${dir}/${drug}_card_cause_histogram_${grain}.png"
    done

    # #3 CARD ladder (family) — ceiling rung now picks the (mut) determinant
    uv run python -m kleb_ast.build_card_ladder --drug "${drug}" --grain family

    # #1 per-gene ESM-vs-FT (family) — bars unchanged; CARD ceiling reference line refreshed
    uv run python -m kleb_ast.per_gene_esm_vs_ft_card --drug "${drug}" --grain family || \
        echo "  (per-gene #1 skipped for ${drug})"
done

# #4 combined panel (both grains)
uv run python -m kleb_ast.build_card_panel --grains family allele

# #5 gene-ingredient concat summary (render from the committed per-drug CSVs under ${VIS}/ingredient)
uv run python -m kleb_ast.plot_gene_ingredient_concat \
    --concat-root "${VIS}/ingredient" --out-dir "${VIS}/ingredient"

echo "=== render_card_figures done -> ${VIS} ==="
