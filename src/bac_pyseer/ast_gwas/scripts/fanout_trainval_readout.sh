#!/usr/bin/env bash
# Submit run_trainval_readout.sh for every drug whose GWAS has landed and whose read-out has not.
#
# The GWAS chains finish at different times across ~22 drugs, so the read-outs are submitted in
# dribs and drabs. Doing that by hand is how one drug quietly never gets submitted and a "22-drug"
# comparison turns out to be 21. This is idempotent: run it as often as you like, and each run
# submits exactly the drugs that became ready since the last one.
#
# Readiness is the HIT TABLE, not the .assoc. The combine phase writes the assoc first and the
# annotated hits afterwards, so keying on the assoc would submit a read-out against a hit table that
# is still being written.
#
# Usage (CSD3):
#   ORGANISM=kp BACPREDICT_DATA_ROOT=<rds>/david/bac_ast_prediction \
#     bash src/bac_pyseer/ast_gwas/scripts/fanout_trainval_readout.sh [drug...]
#   DRY_RUN=1 ...   # report what is ready, submit nothing
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
ORGANISM=${ORGANISM:-kp}
DATA_ROOT=${BACPREDICT_DATA_ROOT:?set BACPREDICT_DATA_ROOT}
COHORT=${COHORT:-trainval_vocab}
VOCAB_ROOT=${VOCAB_ROOT:-$DATA_ROOT/processed/pyseer_ast/${ORGANISM}_${COHORT}}

if [ "$#" -gt 0 ]; then
    DRUGS=("$@")
else
    mapfile -t DRUGS < <(find "$VOCAB_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi
[ "${#DRUGS[@]}" -gt 0 ] || { echo "no drug directories under $VOCAB_ROOT" >&2; exit 1; }

ready=(); waiting=(); done_=(); queued=()
for DRUG in "${DRUGS[@]}"; do
    DRUG_DIR=$VOCAB_ROOT/$DRUG/$DRUG
    hits=$DRUG_DIR/${DRUG}_hits_annotated.tsv
    [ -s "$hits" ] || hits=$DRUG_DIR/gwas/${DRUG}_hits_annotated.tsv
    if [ -s "$DRUG_DIR/lr/results.json" ]; then done_+=("$DRUG"); continue; fi
    # squeue is the only way to see a chain that is submitted but has produced nothing yet;
    # without this check a second run resubmits every in-flight drug.
    if squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -qE "^(udesign|uscan|ulr)_${DRUG}$"; then
        queued+=("$DRUG"); continue
    fi
    if [ -s "$hits" ]; then ready+=("$DRUG"); else waiting+=("$DRUG"); fi
done

echo "vocab root: $VOCAB_ROOT"
echo "done ${#done_[@]}: ${done_[*]:-—}"
echo "in flight ${#queued[@]}: ${queued[*]:-—}"
echo "waiting on GWAS ${#waiting[@]}: ${waiting[*]:-—}"
echo "ready to submit ${#ready[@]}: ${ready[*]:-—}"
echo

[ "${#ready[@]}" -gt 0 ] || { echo "nothing to submit"; exit 0; }
if [ -n "${DRY_RUN:-}" ]; then echo "DRY_RUN: submitting nothing"; exit 0; fi

failed=()
for DRUG in "${ready[@]}"; do
    echo "=== $DRUG"
    if ORGANISM="$ORGANISM" DRUG="$DRUG" REPO="$REPO" COHORT="$COHORT" \
            BACPREDICT_DATA_ROOT="$DATA_ROOT" VOCAB_ROOT="$VOCAB_ROOT" \
            bash "$REPO/src/bac_pyseer/ast_gwas/scripts/run_trainval_readout.sh"; then
        echo "  $DRUG: read-out chain submitted"
    else
        echo "!! $DRUG: run_trainval_readout.sh failed" >&2
        failed+=("$DRUG")
    fi
    echo
done

if [ "${#failed[@]}" -gt 0 ]; then
    echo "FAILED: ${failed[*]}" >&2
    exit 1
fi
echo "submitted ${#ready[@]} read-out chain(s); re-run this script as more GWAS land"
