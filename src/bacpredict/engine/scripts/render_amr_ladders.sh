#!/usr/bin/env bash
# Render every drug's AMR ladder figure, reporting which arms each one actually got.
#
# The compute side (build_amr_ladder.sh) has always had a many-drug driver; the RENDER side never did,
# so the 22 Kp figures were produced by an uncommitted scratchpad script. This is that script, committed,
# with the unitig arm wired in.
#
# It is built to be run REPEATEDLY while a GWAS fan-out lands: a drug with no ladder table is skipped,
# a drug with no unitig results.json still renders (without the purple bar), and the per-drug line says
# which arms were found. That last part is the point -- a partial panel must never be mistaken for a
# complete one.
#
# Matplotlib over saved CSV/JSON: a legitimate login-node task on either cluster. Figures land in the
# CHECKOUT's visualisations tree (visualisations_dir is source-tree relative) and *.png is gitignored,
# so scp them down to look at them.
#
# Usage:
#   BACPREDICT_DATA_ROOT=<root> bash src/bacpredict/engine/scripts/render_amr_ladders.sh
#   SPECIES=tb  BACPREDICT_DATA_ROOT=<root> bash .../render_amr_ladders.sh
#   METRIC=auprc ...                                  # AUPRC instead of AUROC
#   DEDUP=1 ...                                       # add the pale-purple LD-controlled bar
#   DRUGS="gentamicin colistin" ...                   # just these
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
SPECIES=${SPECIES:-kp}
METRIC=${METRIC:-auroc}
DEDUP=${DEDUP:-0}
DATA_ROOT=${BACPREDICT_DATA_ROOT:?set BACPREDICT_DATA_ROOT (CSD3: <project>/david/bac_ast_prediction)}

case "$SPECIES" in
    kp) TASK=train_kleb_ast
        DEFAULT_DRUGS="cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin \
ceftazidime gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin" ;;
    tb) TASK=train_tb_ast
        DEFAULT_DRUGS="rifampin isoniazid ethambutol pyrazinamide moxifloxacin levofloxacin streptomycin \
ethionamide rifabutin kanamycin" ;;
    *)  echo "SPECIES must be kp or tb" >&2; exit 1 ;;
esac
DRUGS=${DRUGS:-$DEFAULT_DRUGS}

LADDER_ROOT=$DATA_ROOT/processed/$TASK/pangena_predict/amr_ladder
SPLIT_ROOT=$DATA_ROOT/processed/$TASK/splits
PYSEER_ROOT=${PYSEER_ROOT:-$DATA_ROOT/processed/pyseer_ast/$SPECIES}
# The full-cohort GWAS shares one OUT_DIR across drugs, so a drug's read-out is <root>/<drug>/lr.
# The train+validate-vocabulary rebuild gives every drug its OWN OUT_DIR, which makes run_drug.sh's
# DRUG_DIR <root>/<drug>/<drug> -- one level deeper. Default 0 keeps the existing layout exactly.
NESTED_DRUG_DIR=${NESTED_DRUG_DIR:-0}

cd "$REPO"
echo "species=$SPECIES  metric=$METRIC  dedup=$DEDUP"
echo "ladders: $LADDER_ROOT"
if [ "$NESTED_DRUG_DIR" = "1" ]; then
    echo "unitigs: $PYSEER_ROOT  (nested per-drug layout: <root>/<drug>/<drug>/lr)"
else
    echo "unitigs: $PYSEER_ROOT"
fi
echo
printf '%-32s %-8s %-11s %-8s %s\n' DRUG LADDER CATALOGUE UNITIG FIGURE
n_drawn=0; n_skipped=0; n_purple=0
skipped_drugs=()

for DRUG in $DRUGS; do
    TABLE=$LADDER_ROOT/$DRUG/${DRUG}_amr_ladder_table.csv
    if [ ! -s "$TABLE" ]; then
        printf '%-32s %-8s %-11s %-8s %s\n' "$DRUG" "-" "-" "-" "SKIPPED: no ladder table"
        n_skipped=$((n_skipped + 1)); skipped_drugs+=("$DRUG")
        continue
    fi

    ARGS=(--species "$SPECIES" --drug "$DRUG" --table-csv "$TABLE" --metric "$METRIC")

    # The fresh CARD ceiling in the data root wins over the committed one, matching build_amr_ladder.sh.
    CAT=$DATA_ROOT/processed/$TASK/card_ceiling/$DRUG/card_determinant_lr_${DRUG}_family.csv
    if [ -s "$CAT" ]; then ARGS+=(--catalogue-csv "$CAT"); cat_state=fresh; else cat_state=default; fi

    # --split-table is passed whenever it exists: it is what lets the plot refuse a unitig result that
    # scored a different set of genomes than the ladder rungs did.
    SPLIT=$SPLIT_ROOT/${DRUG}_split.csv
    [ -s "$SPLIT" ] && ARGS+=(--split-table "$SPLIT")

    # if/fi, not `[ ... ] && assign`: under `set -e` (line 23) that list returns non-zero whenever
    # the test is false, which aborts the script on every non-nested run -- i.e. every existing use.
    if [ "$NESTED_DRUG_DIR" = "1" ]; then
        UNITIG_DRUG_ROOT=$PYSEER_ROOT/$DRUG/$DRUG
    else
        UNITIG_DRUG_ROOT=$PYSEER_ROOT/$DRUG
    fi
    UNI=$UNITIG_DRUG_ROOT/lr/results.json
    if [ -s "$UNI" ]; then ARGS+=(--unitig-results "$UNI"); uni_state=yes; n_purple=$((n_purple + 1))
    else uni_state=-; fi
    UNID=$UNITIG_DRUG_ROOT/lr_dedup/results.json
    if [ "$DEDUP" = "1" ] && [ -s "$UNID" ]; then ARGS+=(--unitig-dedup-results "$UNID"); uni_state="$uni_state+LD"; fi

    OUT=$(uv run python -m bacpredict.engine.plots.plot_amr_ladder "${ARGS[@]}" | sed -n 's/^Wrote //p')
    printf '%-32s %-8s %-11s %-8s %s\n' "$DRUG" "yes" "$cat_state" "$uni_state" "${OUT##*/visualisations/}"
    n_drawn=$((n_drawn + 1))
done

echo
echo "drawn $n_drawn, of which $n_purple carry the unitig (purple) arm; skipped $n_skipped"
if [ "$n_skipped" -gt 0 ]; then echo "skipped: ${skipped_drugs[*]}"; fi
if [ "$n_purple" -lt "$n_drawn" ]; then
    echo "NOTE: this panel is PARTIAL — $((n_drawn - n_purple)) figure(s) have no unitig bar yet."
    echo "      Re-run this script unchanged once the remaining GWAS drugs land."
fi
