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

# Walltime knobs, passed through to run_unitig_lmm_sharded.sh. They matter because SLURM checks
# AssocGrpCPUMinutesLimit against cores x walltime REQUESTED, not used: the old hardcoded 24 h meant
# each drug's 64 x 8-cpu array reserved 12,288 core-h, so 22 drugs reserved 270,336 against ~106,371
# available and NOTHING would ever have started. Measured shard max is 6.2 min, so 2 h is still ~19x.
export ARRAY_TIME=${ARRAY_TIME:-02:00:00}
export PREP_TIME=${PREP_TIME:-06:00:00}
export COMB_TIME=${COMB_TIME:-03:00:00}

# The sharded driver keys its scratch shard dir on $PAIR/$COHORT. Naming the cohort is what stops a
# re-run writing chunk_NN.assoc into a completed run's directory, where a shard that never starts
# leaves the old file behind, both the empty-check and the runt-check pass on it, and the combined
# .assoc silently mixes two vocabularies. Change it whenever the vocabulary changes.
export COHORT=${COHORT:-ast_$ORGANISM}

# VOCAB_ROOT switches to the per-drug layout used by the train+validate-vocabulary rebuild: each drug
# owns its own graph, structure and outputs, so OUT_DIR moves inside the loop. Unset (the default),
# every drug shares one OUT_DIR and one matrix, which is the full-cohort run.
VOCAB_ROOT=${VOCAB_ROOT:-}

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
echo "cohort:    $COHORT   array wall: $ARRAY_TIME"
echo "layout:    ${VOCAB_ROOT:+per-drug under $VOCAB_ROOT}${VOCAB_ROOT:-shared (full-cohort)}"
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
    # Empty OUT_DIR falls through run_drug.sh's ${OUT_DIR:-...} to the shared full-cohort default.
    drug_out=${VOCAB_ROOT:+$VOCAB_ROOT/$drug}
    if ORGANISM="$ORGANISM" DRUG="$drug" REPO="$REPO" OUT_DIR="$drug_out" \
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
