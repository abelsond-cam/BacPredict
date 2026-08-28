#!/usr/bin/env bash
# C6: turn the two completed arms into the comparison of record — gates first, then numbers.
#
# The order is deliberate and is the point of the script. Every stage that can invalidate a number
# runs BEFORE the number is computed, so a bad gate stops the run rather than being noticed after a
# figure has been drawn:
#
#   (1) re-run every drug's design audit      — uniform records regardless of completion order
#   (2) read-out gate table                    — shard completeness, scanner-vs-GGCAT, coverage
#   (3) STOP if any drug is not clean          — no comparison over an unverified read-out
#   (4) GWAS-level table                       — patterns / threshold / lambda, pheno_var control
#   (5) paired AUROC comparison + bootstrap CI — the headline, both lr and lr_dedup
#   (6) the two figures
#
# Step (1) matters because audit_design's recorded fields have changed during the run: drugs whose
# read-out finished before the shard-completeness gate existed have a design section without it, and
# a record that varies by when a drug happened to finish is not a record.
#
# Usage (CSD3):
#   ORGANISM=kp BACPREDICT_DATA_ROOT=<rds>/david/bac_ast_prediction \
#     bash src/bac_pyseer/ast_gwas/scripts/run_vocab_comparison.sh
#   SKIP_AUDIT=1 ...   # numbers only, when the audits are known current
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
ORGANISM=${ORGANISM:-kp}
DATA_ROOT=${BACPREDICT_DATA_ROOT:?set BACPREDICT_DATA_ROOT}
COHORT=${COHORT:-trainval_vocab}
P=$DATA_ROOT/processed/pyseer_ast
FULL_ROOT=${FULL_ROOT:-$P/$ORGANISM}
VOCAB_ROOT=${VOCAB_ROOT:-$P/${ORGANISM}_${COHORT}}
OUT_DIR=${OUT_DIR:-$VOCAB_ROOT/comparison}
N_BOOT=${N_BOOT:-2000}

cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
mkdir -p "$OUT_DIR"
echo "comparator: $FULL_ROOT"
echo "rebuild:    $VOCAB_ROOT"
echo "out:        $OUT_DIR"
echo

if [ -z "${SKIP_AUDIT:-}" ]; then
    echo "=== (1) refresh every drug's design audit ==="
    for d in "$VOCAB_ROOT"/*/; do
        drug=$(basename "$d")
        m=$d$drug/design_merged/merge_manifest.json
        [ -s "$m" ] || { echo "  $drug: no merge manifest yet — skipped"; continue; }
        uv run python -m bac_pyseer.ast_gwas.leakage_audit \
            --audit-json "$d/leakage_audit.json" design --merge-manifest "$m" > /dev/null
        echo "  $drug: design audit refreshed"
    done
    echo
fi

echo "=== (2) read-out gates ==="
# `|| gates_failed=1` rather than letting set -e kill us: the table is worth printing in full even
# when a drug fails, so the reader sees which one and why rather than just a non-zero exit.
gates_failed=0
uv run python -m bac_pyseer.ast_gwas.summarise_vocab_build --stage readout \
    --vocab-root "$VOCAB_ROOT" --out-tsv "$OUT_DIR/readout_gates.tsv" || gates_failed=1
echo
if [ "$gates_failed" != "0" ]; then
    echo "=== (3) STOPPING: a read-out gate did not pass ===" >&2
    echo "No comparison is computed over an unverified read-out. Fix the drug(s) marked FAIL or" >&2
    echo "'not yet run' above, then re-run. A number produced now would look exactly like a good" >&2
    echo "one -- that is the whole reason the gates run first." >&2
    exit 1
fi
echo "=== (3) all read-out gates pass ==="
echo

echo "=== (4) GWAS-level comparison ==="
uv run python -m bac_pyseer.ast_gwas.compare_vocab_arms --stage gwas \
    --full-root "$FULL_ROOT" --vocab-root "$VOCAB_ROOT" \
    --out-csv "$OUT_DIR/gwas_arm_comparison.csv"
echo

for arm in lr lr_dedup; do
    echo "=== (5) paired AUROC comparison — $arm ==="
    uv run python -m bac_pyseer.ast_gwas.compare_vocab_arms --stage readout --arm "$arm" \
        --full-root "$FULL_ROOT" --vocab-root "$VOCAB_ROOT" --n-boot "$N_BOOT" \
        --out-csv "$OUT_DIR/auroc_comparison_${arm}.csv"
    echo
done

echo "=== (6) figures ==="
uv run python -m bac_pyseer.ast_gwas.plot_vocab_comparison \
    --comparison-csv "$OUT_DIR/auroc_comparison_lr.csv" --out-dir "$OUT_DIR" --organism "$ORGANISM"

cat <<EOF

=== done ===
  $OUT_DIR/readout_gates.tsv
  $OUT_DIR/gwas_arm_comparison.csv
  $OUT_DIR/auroc_comparison_{lr,lr_dedup}.csv
  $OUT_DIR/vocab_{paired_scatter,delta_caterpillar}_${ORGANISM}.png

[look] delta = full_cohort - trainval_vocab, so positive means the OLD arm scored higher. It is not
all leakage: MIN_SAMP fell from a flat 71 to a per-drug 12-40, which is more correct and pushes the
other way. The table carries min_samples, n_unitigs and n_patterns for both arms so the three
effects -- representation advantage, out-of-vocabulary penalty, MAF floor -- can be separated
rather than collapsed into one number.

Re-render the ladders with the rebuild as the number of record:
  NESTED_DRUG_DIR=1 PYSEER_ROOT=$VOCAB_ROOT \\
    bash src/bacpredict/engine/scripts/render_amr_ladders.sh
EOF
