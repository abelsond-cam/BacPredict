#!/usr/bin/env bash
# Per drug, AFTER the train+validate-vocabulary GWAS lands: design -> holdout scan -> merge -> LR.
#
# This differs from run_readout.sh in exactly one structural way, and it is the whole point. The
# unitig matrix here was built over train+validate only, so it holds NO holdout carriers -- asking it
# for holdout rows returns zeros, silently, and the LR would report a well-formed AUROC of ~0.5. The
# holdout is therefore scored from sequence instead, by k-mer containment, which is the same operator
# GGCAT's colouring encodes. The merge asserts that: it re-scores train+validate with the scanner and
# requires exact agreement with the matrix rows before it will emit a design.
#
# Chain (each stage --dependency on the last):
#   (A) design over train+validate      -- streams the per-drug matrix once, caches the sub-matrix
#   (B) scan array over ALL splits      -- train+validate rows exist only to verify the scanner
#   (C) merge + LR + LD-control LR      -- one scan serves both designs; columns align by sequence
#
# The fine-tune arm is deliberately NOT re-run: it is unchanged by any of this, so re-scoring 22
# checkpoints buys identical numbers for ~22 GPU-hours. Reuse each drug's existing eval_scores.npz.
#
# Usage:
#   ORGANISM=kp DRUG=ertapenem bash src/bac_pyseer/ast_gwas/scripts/run_trainval_readout.sh
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
ORGANISM=${ORGANISM:-kp}
DRUG=${DRUG:?set DRUG=<ast column name>}
DATA_ROOT=${BACPREDICT_DATA_ROOT:?set BACPREDICT_DATA_ROOT}
COHORT=${COHORT:-trainval_vocab}

case "$ORGANISM" in
    kp) TASK=train_kleb_ast ;;
    tb) TASK=train_tb_ast ;;
    *)  echo "ORGANISM must be kp or tb" >&2; exit 1 ;;
esac

VOCAB_ROOT=${VOCAB_ROOT:-$DATA_ROOT/processed/pyseer_ast/${ORGANISM}_${COHORT}}
FULL_ROOT=${FULL_ROOT:-$DATA_ROOT/processed/pyseer_ast/$ORGANISM}   # the comparator; read-only here
DRUG_ROOT=$VOCAB_ROOT/$DRUG
UNITIG_DIR=$DRUG_ROOT/unitigs
DRUG_DIR=$DRUG_ROOT/$DRUG                      # run_drug.sh's DRUG_DIR when OUT_DIR=$DRUG_ROOT
MATRIX=$UNITIG_DIR/unitigs.pyseer.gz
REFLIST=$UNITIG_DIR/assembly_refs.txt
SPLIT_TABLE=${SPLIT_TABLE:-$DATA_ROOT/processed/$TASK/splits/${DRUG}_split.csv}
HITS=$DRUG_DIR/${DRUG}_hits_annotated.tsv
AUDIT=$DRUG_ROOT/leakage_audit.json
SCAN_DIR=$DRUG_DIR/scan
LOGDIR=${LOGDIR:-$DATA_ROOT/logs}

# The scan needs every genome's assembly, INCLUDING the holdout's -- which is why it cannot use the
# vocabulary reflist. Scoring a holdout genome from its own sequence is not a leak; supplying it to
# GGCAT would have been.
SCAN_REFLIST=${SCAN_REFLIST:-$DRUG_DIR/scan_refs.txt}
# Same source as the vocabulary reflist, for the same reason: identical assemblies to the comparator,
# and no cold stat per sample. See build_trainval_vocab.sh.
REF_SOURCE=${REF_SOURCE-$FULL_ROOT/unitigs/assembly_refs.txt}

ACCT=${ACCT:-FLOTO-PROJECT-K-SL2-CPU}
PART=${PART:-icelake-himem}
QOS=${QOS-}
DESIGN_CPUS=${DESIGN_CPUS:-16}; DESIGN_MEM=${DESIGN_MEM:-100G}; DESIGN_WALL=${DESIGN_WALL:-04:00:00}
# The scan is CPU-bound and small in memory: the feature k-mer table is ~35 MB even for ceftazidime's
# 450,950 features, and one genome's index is ~44 MB. Sharding is for wall time, not for RAM.
SCAN_SHARDS=${SCAN_SHARDS:-8}; SCAN_CPUS=${SCAN_CPUS:-2}; SCAN_MEM=${SCAN_MEM:-24G}
SCAN_WALL=${SCAN_WALL:-04:00:00}
LR_CPUS=${LR_CPUS:-16}; LR_MEM=${LR_MEM:-100G}; LR_WALL=${LR_WALL:-04:00:00}

[ -s "$HITS" ] || HITS=$DRUG_DIR/gwas/${DRUG}_hits_annotated.tsv
for required in "$MATRIX" "$SPLIT_TABLE" "$HITS" "$REFLIST"; do
    [ -s "$required" ] || { echo "ERROR: missing $required — has the GWAS chain finished?" >&2; exit 1; }
done
mkdir -p "$LOGDIR" "$SCAN_DIR"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

# The scan reflist spans all three splits and is resolved inline (seconds, no compute).
if [ -n "$REF_SOURCE" ] && [ -s "$REF_SOURCE" ]; then
    RESOLVE_ARGS=(--file-list "$REF_SOURCE" --no-check-exists)
else
    RESOLVE_ARGS=(${FILE_LIST:+--file-list "$FILE_LIST"})
fi
uv run python -m bac_pyseer.ast_gwas.resolve_ast_assemblies \
    --organism "$ORGANISM" --split-table "$SPLIT_TABLE" --splits train,validate,holdout \
    --out-tsv "$SCAN_REFLIST" --data-root "$DATA_ROOT" "${RESOLVE_ARGS[@]}" > /dev/null
echo "scan reflist: $(wc -l < "$SCAN_REFLIST") genomes (all splits) -> $SCAN_REFLIST"

sb() { sbatch --parsable --account="$ACCT" --partition="$PART" ${QOS:+--qos="$QOS"} \
              --nodes=1 --ntasks=1 "$@"; }
P='.venv/bin/python'

echo "=== (A) design over train+validate ==="
DESIGN_JOB=$(sb --cpus-per-task="$DESIGN_CPUS" --mem="$DESIGN_MEM" --time="$DESIGN_WALL" \
    --job-name="udesign_${DRUG}" \
    --output="$LOGDIR/udesign_${COHORT}_${DRUG}_%j.out" \
    --error="$LOGDIR/udesign_${COHORT}_${DRUG}_%j.err" \
    --wrap "set -euo pipefail
        cd '$REPO'
        $P -m bac_pyseer.ast_gwas.unitig_design_matrix \
            --hits-tsv '$HITS' --matrix-gz '$MATRIX' --split-table '$SPLIT_TABLE' \
            --splits train,validate --out-dir '$DRUG_DIR/design' --decomp-threads $DESIGN_CPUS
        $P -m bac_pyseer.ast_gwas.unitig_design_matrix \
            --hits-tsv '$HITS' --matrix-gz '$MATRIX' --split-table '$SPLIT_TABLE' \
            --splits train,validate --out-dir '$DRUG_DIR/design_dedup' --dedupe-patterns \
            --decomp-threads $DESIGN_CPUS")
echo "JOB $DESIGN_JOB udesign_${DRUG} | CPU | mem=$DESIGN_MEM | cores=$DESIGN_CPUS | wall=$DESIGN_WALL | $PART"

echo "=== (B) holdout scan (array) ==="
SCAN_JOB=$(sb --array=0-$((SCAN_SHARDS - 1)) --cpus-per-task="$SCAN_CPUS" --mem="$SCAN_MEM" \
    --time="$SCAN_WALL" --dependency=afterok:"$DESIGN_JOB" --job-name="uscan_${DRUG}" \
    --output="$LOGDIR/uscan_${COHORT}_${DRUG}_%A_%a.out" \
    --error="$LOGDIR/uscan_${COHORT}_${DRUG}_%A_%a.err" \
    --wrap "set -euo pipefail
        cd '$REPO'
        $P -m bac_pyseer.ast_gwas.unitig_kmer_presence score \
            --id-map '$DRUG_DIR/design/id_map.tsv' --split-table '$SPLIT_TABLE' \
            --reflist '$SCAN_REFLIST' --splits train,validate,holdout \
            --shard-index \$SLURM_ARRAY_TASK_ID --n-shards $SCAN_SHARDS \
            --out '$SCAN_DIR/scan_\$(printf %02d \$SLURM_ARRAY_TASK_ID).npz'")
echo "JOB $SCAN_JOB uscan_${DRUG} | CPU | mem=$SCAN_MEM | cores=$SCAN_CPUS | wall=$SCAN_WALL | $PART   --array=0-$((SCAN_SHARDS - 1))  (CPU-bound; ~44 MB/genome index)"

echo "=== (C) merge + LR + LD control ==="
LR_JOB=$(sb --cpus-per-task="$LR_CPUS" --mem="$LR_MEM" --time="$LR_WALL" \
    --dependency=afterok:"$SCAN_JOB" --job-name="ulr_${DRUG}" \
    --output="$LOGDIR/ulr_${COHORT}_${DRUG}_%j.out" \
    --error="$LOGDIR/ulr_${COHORT}_${DRUG}_%j.err" \
    --wrap "set -euo pipefail
        cd '$REPO'
        for d in design design_dedup; do
            $P -m bac_pyseer.ast_gwas.unitig_kmer_presence merge \
                --design-dir '$DRUG_DIR/'\$d --shard-dir '$SCAN_DIR' \
                --split-table '$SPLIT_TABLE' --out-dir '$DRUG_DIR/'\${d}_merged
        done
        $P -m bac_pyseer.ast_gwas.unitig_lr \
            --design-dir '$DRUG_DIR/design_merged' --split-table '$SPLIT_TABLE' \
            --drug '$DRUG' --organism '$ORGANISM' --out-dir '$DRUG_DIR/lr' \
            --gwas-summary '$DRUG_DIR/${DRUG}_gwas_summary.json'
        $P -m bac_pyseer.ast_gwas.unitig_lr \
            --design-dir '$DRUG_DIR/design_dedup_merged' --split-table '$SPLIT_TABLE' \
            --drug '$DRUG' --organism '$ORGANISM' --out-dir '$DRUG_DIR/lr_dedup' \
            --gwas-summary '$DRUG_DIR/${DRUG}_gwas_summary.json'
        $P -m bac_pyseer.ast_gwas.leakage_audit --audit-json '$AUDIT' design \
            --merge-manifest '$DRUG_DIR/design_merged/merge_manifest.json'")
echo "JOB $LR_JOB ulr_${DRUG} | CPU | mem=$LR_MEM | cores=$LR_CPUS | wall=$LR_WALL | $PART   (after $SCAN_JOB)"

cat <<EOF

=== chain submitted: $DESIGN_JOB -> $SCAN_JOB -> $LR_JOB ===

[look] When it lands, read $AUDIT before the result:
  - design.verification.n_mismatch_cells == 0 AND n_shared == the train+validate count. Zero
    mismatches out of zero comparisons is not a passed gate.
  - design.holdout_coverage.checked == true, and the ratio. A depressed ratio is the genuine
    out-of-vocabulary penalty and is part of the finding; a ratio near zero is a broken scan.

Then the paired comparison against the full-cohort run, which is untouched at
$DATA_ROOT/processed/pyseer_ast/$ORGANISM/$DRUG/lr/results.json
EOF
