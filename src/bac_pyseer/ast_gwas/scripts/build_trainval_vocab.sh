#!/usr/bin/env bash
# Per drug: a unitig vocabulary built over train+validate ONLY, plus its structure and its audit.
#
# The full-cohort run built one GGCAT graph over all 7,080 Kp genomes and every drug reused it. No
# label ever leaked -- the GWAS phenotype is train+validate and the holdout is scored once -- but the
# *feature representation* was shaped by holdout sequence: a k-mer surviving `-s 2` only because a
# holdout genome carried it is a node, nodes are branch points, and a branch point splits a unitig
# that every genome then inherits. This script removes that last path, which costs one build per drug
# because holdout membership is drawn per drug (a genome labelled for m drugs escapes every holdout
# with probability ~0.8^m -- about 1% survive, so a union build is not an option).
#
# Layout, deliberately disjoint from the full-cohort run so nothing can be reused by accident:
#   <root>/processed/pyseer_ast/<organism>_trainval_vocab/<drug>/{unitigs,structure,<drug>}
# The full-cohort matrix, triangle, clusters and results stay untouched -- they are the comparator.
#
# Usage (CSD3):
#   ORGANISM=kp ACCT=FLOTO-PROJECT-K-SL2-CPU PART=icelake-himem QOS= \
#   BACPREDICT_DATA_ROOT=<rds>/david/bac_ast_prediction \
#   FILE_LIST=<rds>/david/raw/assemblies_file_list.tsv \
#     bash src/bac_pyseer/ast_gwas/scripts/build_trainval_vocab.sh ertapenem colistin ...
#   DRY_RUN=1 ... # resolve and audit the reflists, submit nothing
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
ORGANISM=${ORGANISM:-kp}
DATA_ROOT=${BACPREDICT_DATA_ROOT:?set BACPREDICT_DATA_ROOT}
COHORT=${COHORT:-trainval_vocab}

case "$ORGANISM" in
    kp) TASK=train_kleb_ast ;;
    tb) TASK=train_tb_ast ;;
    *)  echo "ORGANISM must be kp or tb" >&2; exit 1 ;;
esac

VOCAB_ROOT=${VOCAB_ROOT:-$DATA_ROOT/processed/pyseer_ast/${ORGANISM}_${COHORT}}
FULL_ROOT=${FULL_ROOT:-$DATA_ROOT/processed/pyseer_ast/$ORGANISM}   # the comparator; read-only here
SCRATCH_ROOT=${SCRATCH_ROOT:-/home/dca36/rds/hpc-work/ggcat_tmp/$COHORT}
LOGDIR=${LOGDIR:-$DATA_ROOT/logs}
METADATA_TSV=${METADATA_TSV:-$DATA_ROOT/../final/metadata_v2_all_samples_and_columns.tsv}
# The deployed clusters were built with these two extra LIN-typing tables merged in; reproducing the
# invocation is what keeps n_clusters and named coverage comparable between the two runs.
LIN_DIR=${LIN_DIR:-$DATA_ROOT/../lin_typing}
EXTRA_LIN=${EXTRA_LIN:-"$LIN_DIR/archived_lin.tsv $LIN_DIR/mist_lin_new.tsv"}
MIN_CLUSTER_SIZE=${MIN_CLUSTER_SIZE:-100}

ACCT=${ACCT:-FLOTO-PROJECT-K-SL2-CPU}
PART=${PART:-icelake-himem}
QOS=${QOS-}
# GGCAT on ~35% of the cohort should take ~25 min (54 min for all 7,080 at 38 cpus). The wall is ~3x
# rather than the usual 1.5x for a sub-hour job because 22 of these run at once and GGCAT is
# disk-bound -- they contend for hpc-work, and a mid-build kill costs the whole build plus a requeue.
GGCAT_CPUS=${GGCAT_CPUS:-32}; GGCAT_MEM=${GGCAT_MEM:-120G}; GGCAT_TIME=${GGCAT_TIME:-01:30:00}
GGCAT_MEMGB=${GGCAT_MEMGB:-90}    # GGCAT's in-RAM budget (-m); MUST sit below --mem or it over-allocates
MASH_CPUS=${MASH_CPUS:-16};   MASH_MEM=${MASH_MEM:-48G};   MASH_TIME=${MASH_TIME:-00:40:00}
AUDIT_CPUS=${AUDIT_CPUS:-2};  AUDIT_MEM=${AUDIT_MEM:-16G}; AUDIT_TIME=${AUDIT_TIME:-00:30:00}

[ "$#" -gt 0 ] || { echo "usage: $0 <drug> [drug...]" >&2; exit 1; }
mkdir -p "$LOGDIR"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

echo "repo:        $REPO"
echo "vocab root:  $VOCAB_ROOT"
echo "comparator:  $FULL_ROOT  (read-only)"
echo "account:     $ACCT   partition: $PART   qos: ${QOS:-<none>}"
echo "drugs:       $*"
echo

# Resolve through the full-cohort reflist when it exists. It is already a Sample<TAB>path TSV whose
# paths were stat-validated when it was built, so the subset is instant AND provably uses the same
# assemblies as the comparator run -- a genome the full-cohort arm could not resolve is excluded here
# too, rather than the two arms silently running on slightly different genome sets. Going back to the
# 95k file list instead costs a cold stat per sample: ~50 s per drug, ~20 min across 22, on a login
# node. Set REF_SOURCE= (empty) to force resolution against $FILE_LIST.
REF_SOURCE=${REF_SOURCE-$FULL_ROOT/unitigs/assembly_refs.txt}
if [ -n "$REF_SOURCE" ] && [ -s "$REF_SOURCE" ]; then
    RESOLVE_ARGS=(--file-list "$REF_SOURCE" --no-check-exists)
    echo "reflists resolve through the comparator's reflist: $REF_SOURCE ($(wc -l < "$REF_SOURCE") genomes)"
else
    RESOLVE_ARGS=(${FILE_LIST:+--file-list "$FILE_LIST"})
    echo "reflists resolve against ${FILE_LIST:-the flat assemblies dir} (stat per sample)"
fi
echo

sb() { sbatch --parsable --account="$ACCT" --partition="$PART" ${QOS:+--qos="$QOS"} \
              --nodes=1 --ntasks=1 "$@"; }

failed=()
for DRUG in "$@"; do
    SPLIT_TABLE=$DATA_ROOT/processed/$TASK/splits/${DRUG}_split.csv
    DRUG_ROOT=$VOCAB_ROOT/$DRUG
    UNITIG_DIR=$DRUG_ROOT/unitigs
    STRUCT_DIR=$DRUG_ROOT/structure
    REFLIST=$UNITIG_DIR/assembly_refs.txt      # run_ggcat_unitigs.sh reads exactly this path
    AUDIT=$DRUG_ROOT/leakage_audit.json
    TMP=$SCRATCH_ROOT/$DRUG

    echo "=================================================================="
    echo "=== $DRUG"
    if [ ! -s "$SPLIT_TABLE" ]; then
        echo "!! no split table at $SPLIT_TABLE — skipping" >&2; failed+=("$DRUG"); continue
    fi
    mkdir -p "$UNITIG_DIR" "$STRUCT_DIR"

    # (0) reflist -- train+validate only. Written to .new first and compared, because
    # run_ggcat_unitigs.sh reuses an existing reflist AND an existing graph without complaint: a
    # stale pair from another cohort would silently supply the full-cohort vocabulary.
    uv run python -m bac_pyseer.ast_gwas.resolve_ast_assemblies \
        --organism "$ORGANISM" --split-table "$SPLIT_TABLE" --splits train,validate \
        --out-tsv "$REFLIST.new" --data-root "$DATA_ROOT" \
        "${RESOLVE_ARGS[@]}" > /dev/null
    if [ -s "$REFLIST" ] && ! cmp -s "$REFLIST" "$REFLIST.new"; then
        echo "!! $REFLIST exists and differs from a fresh resolve — refusing to reuse a foreign" >&2
        echo "   cohort's graph. Delete $UNITIG_DIR to rebuild." >&2
        failed+=("$DRUG"); continue
    fi
    mv "$REFLIST.new" "$REFLIST"
    NSAMP=$(wc -l < "$REFLIST")

    # (0b) the assertion the whole rebuild rests on, checked before any compute is spent
    uv run python -m bac_pyseer.ast_gwas.leakage_audit --audit-json "$AUDIT" reflist \
        --reflist "$REFLIST" --split-table "$SPLIT_TABLE" > /dev/null
    echo "  reflist: $NSAMP train+validate genomes, 0 holdout — audited"

    if [ -n "${DRY_RUN:-}" ]; then echo "  DRY_RUN: submitting nothing"; echo; continue; fi

    GGCAT_JOB=$(REPO="$REPO" OUT_DIR="$UNITIG_DIR" TMP="$TMP" MEMGB="$GGCAT_MEMGB" \
        MAX_FRAC="${MAX_FRAC:-0.99}" \
        sb --cpus-per-task="$GGCAT_CPUS" --mem="$GGCAT_MEM" --time="$GGCAT_TIME" \
           --job-name="ggcat_${DRUG}" \
           --output="$LOGDIR/ggcat_${COHORT}_${DRUG}_%j.out" \
           --error="$LOGDIR/ggcat_${COHORT}_${DRUG}_%j.err" \
           --export=ALL,REPO,OUT_DIR,TMP,MEMGB,MAX_FRAC \
           "$REPO/src/bac_pyseer/kleb_iso_source/scripts/run_ggcat_unitigs.sh")
    echo "JOB $GGCAT_JOB ggcat_${DRUG} | CPU | mem=$GGCAT_MEM | cores=$GGCAT_CPUS | wall=$GGCAT_TIME | $PART   (GGCAT -m ${GGCAT_MEMGB}G, below --mem)"

    # mash + clusters: independent of the graph, so no dependency. The fresh similarity is asserted
    # against the old cohort triangle's subset -- zero difference, which is what turns "subsetting a
    # triangle is the same as re-sketching" from an argument into a line an auditor can read.
    MASH_JOB=$(sb --cpus-per-task="$MASH_CPUS" --mem="$MASH_MEM" --time="$MASH_TIME" \
        --job-name="mash_${DRUG}" \
        --output="$LOGDIR/mash_${COHORT}_${DRUG}_%j.out" \
        --error="$LOGDIR/mash_${COHORT}_${DRUG}_%j.err" \
        --wrap "set -euo pipefail
                cd '$REPO'; unset PYTHONPATH PYTHONHOME; export PYTHONPATH='$REPO/src'
                uv run python -m bac_pyseer.ast_gwas.mash_kinship sketch \
                    --reflist '$REFLIST' --out-dir '$STRUCT_DIR' --threads $MASH_CPUS
                uv run python -m bac_pyseer.ast_gwas.sublineage_from_metadata \
                    --reflist '$REFLIST' --metadata-tsv '$METADATA_TSV' \
                    --min-size $MIN_CLUSTER_SIZE \
                    $(for f in $EXTRA_LIN; do printf -- '--extra-sublineage-tsv %q ' "$f"; done) \
                    --out-tsv '$STRUCT_DIR/lineage_clusters.tsv'
                uv run python -m bac_pyseer.ast_gwas.leakage_audit --audit-json '$AUDIT' clusters \
                    --clusters-tsv '$STRUCT_DIR/lineage_clusters.tsv' --reflist '$REFLIST'")
    echo "JOB $MASH_JOB mash_${DRUG}  | CPU | mem=$MASH_MEM | cores=$MASH_CPUS | wall=$MASH_TIME | $PART   (3.7 min for all 7,080; this is ~35% of that)"

    # The vocabulary assertion, against GGCAT's own colour record rather than the reflist we passed.
    AUDIT_JOB=$(sb --cpus-per-task="$AUDIT_CPUS" --mem="$AUDIT_MEM" --time="$AUDIT_TIME" \
        --dependency=afterok:"$GGCAT_JOB" --job-name="vocab_${DRUG}" \
        --output="$LOGDIR/vocab_${COHORT}_${DRUG}_%j.out" \
        --error="$LOGDIR/vocab_${COHORT}_${DRUG}_%j.err" \
        --wrap "set -euo pipefail
                cd '$REPO'; unset PYTHONPATH PYTHONHOME; export PYTHONPATH='$REPO/src'
                uv run python -m bac_pyseer.ast_gwas.leakage_audit --audit-json '$AUDIT' vocabulary \
                    --color-names '$UNITIG_DIR/color_names.jsonl' --reflist '$REFLIST' \
                    --split-table '$SPLIT_TABLE' --matrix-gz '$UNITIG_DIR/unitigs.pyseer.gz'
                ls -l '$UNITIG_DIR'")
    echo "JOB $AUDIT_JOB vocab_${DRUG} | CPU | mem=$AUDIT_MEM | cores=$AUDIT_CPUS | wall=$AUDIT_TIME | $PART   (after $GGCAT_JOB)"
    echo
done

if [ "${#failed[@]}" -gt 0 ]; then
    echo "FAILED / skipped: ${failed[*]}" >&2; exit 1
fi
cat <<EOF
=== submitted ===
Watch:  squeue -u \$USER
Audit:  cat $VOCAB_ROOT/<drug>/leakage_audit.json
Sizes:  ls -lh $VOCAB_ROOT/<drug>/unitigs/unitigs.pyseer.gz

[look] Before running any GWAS, check per drug:
  - leakage_audit.json has BOTH a 'reflist' and a 'vocabulary' section (a missing one is unchecked,
    not passed), and n_holdout_coloured == 0
  - MIN_SAMP: it auto-rebases to 1% of the SMALLER reflist, so it is now per-drug (12 for colistin's
    1,128 genomes, 40 for gentamicin's 3,932) rather than a flat 71. Expressed over one drug's
    phenotyped genomes, 71 was an effective ~4% MAF floor while pyseer was told --min-af 0.01, so
    the old run was silently under-powered for rare unitigs on every drug. The rebase is more
    correct -- and it is a second, NON-LEAKAGE difference that will tend to RAISE the new AUROC,
    partially masking the leakage cost. leakage_audit.json records min_samples_floor per drug; the
    read-out controls for it by refitting on columns with >=71 train+validate carriers.
EOF
