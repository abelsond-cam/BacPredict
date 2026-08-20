#!/usr/bin/env bash
# Fan the per-drug GWAS out over several drugs, one run_drug.sh chain each.
#
# run_drug.sh defaults to Isambard (brics.u6fp / workq). This wrapper carries the CSD3 invocation --
# account, partition, and the fact that the canonical data root is `bac_ast_prediction`, NOT the
# deprecated `processed/` tree beside it. Getting that root wrong makes the whole GWAS look absent.
#
# Steps 1-2 of each drug run inline here (phenotype, then kinship subset from the cohort triangle),
# so this takes a couple of minutes per drug on the login node before its SLURM chain is submitted.
# That is within the login-node budget; step 3 onwards is all sbatch.
#
# Usage:
#   bash src/bac_pyseer/ast_gwas/scripts/run_fanout.sh gentamicin ceftazidime meropenem
#   DRY_RUN=1 bash .../run_fanout.sh <drugs...>     # print what would run, submit nothing
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
ORGANISM=${ORGANISM:-kp}
DATA_ROOT=${BACPREDICT_DATA_ROOT:?set BACPREDICT_DATA_ROOT (CSD3: <project>/david/bac_ast_prediction)}

export BACPREDICT_DATA_ROOT="$DATA_ROOT"
export ACCT=${ACCT:-floto-project-k-sl2-cpu}
export PART=${PART:-icelake-himem}
# QOS= (explicitly empty) makes run_drug.sh omit --qos entirely. The FLOTO associations allow only
# cpu1/intr, so a --qos=normal it cannot drop is a rejected submission.
export QOS=${QOS-}
export LOGDIR=${LOGDIR:-$DATA_ROOT/logs}

# Mirror run_drug.sh's organism -> task mapping. Deriving it as train_${ORGANISM}_ast is wrong:
# kp's task dir is train_kleb_ast, so the pre-check silently rejected every drug.
case "$ORGANISM" in
    kp) TASK=train_kleb_ast ;;
    tb) TASK=train_tb_ast ;;
    *)  echo "ORGANISM must be kp or tb" >&2; exit 1 ;;
esac

[ "$#" -gt 0 ] || { echo "usage: $0 <drug> [drug...]" >&2; exit 1; }
mkdir -p "$LOGDIR"

echo "repo:      $REPO"
echo "data root: $DATA_ROOT"
echo "account:   $ACCT   partition: $PART   qos: ${QOS:-<none>}"
echo "drugs:     $*"
echo

failed=()
for drug in "$@"; do
    split_table=$DATA_ROOT/processed/$TASK/splits/${drug}_split.csv
    if [ ! -s "$split_table" ]; then
        echo "!! $drug: no split table at $split_table — skipping" >&2
        failed+=("$drug")
        continue
    fi
    echo "=================================================================="
    echo "=== $drug"
    echo "=================================================================="
    if [ -n "${DRY_RUN:-}" ]; then
        echo "  DRY_RUN: would run run_drug.sh for $drug"
        continue
    fi
    if ORGANISM="$ORGANISM" DRUG="$drug" REPO="$REPO" \
            bash "$REPO/src/bac_pyseer/ast_gwas/scripts/run_drug.sh"; then
        echo "  $drug: chain submitted"
    else
        echo "!! $drug: run_drug.sh failed" >&2
        failed+=("$drug")
    fi
    echo
done

if [ "${#failed[@]}" -gt 0 ]; then
    echo "FAILED / skipped: ${failed[*]}" >&2
    exit 1
fi
echo "all ${#} drug(s) submitted; watch with: squeue -u \$USER"
