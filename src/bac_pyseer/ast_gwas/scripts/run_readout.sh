#!/usr/bin/env bash
# Per drug, AFTER run_drug.sh's GWAS chain lands: the unitig read-out plus the fine-tune's holdout
# re-score, which together give one comparison row under one convention.
#
# Two jobs on two partitions, submitted together and independent of each other:
#
#   (A) CPU  — hits -> sparse design matrix -> LR (fit train, sweep C on validate, score holdout
#              once), then the same again on one unitig per perfect-LD block as the LD control.
#              Streams the whole unitig matrix (27 GB for Kp), so it is a batch job, never login.
#   (B) GPU  — engine.finetune.evaluate over the SAME holdout, writing eval_scores.npz.
#
# (B) is not optional if a head-to-head claim is wanted. Fine-tune training saves only aggregate
# metrics at threshold 0.5 -- no per-sample scores -- so without this pass there is no way to
# compute the paired bootstrap CI, and a point-estimate gap cannot be told from a tie. On Kp
# ertapenem it cost 14 min and turned "FT looks better" into delta -0.0103, CI [-0.0187, -0.0031].
# Bacformer does NOT need flash-attn, which is absent on CSD3; that wall is baclm-only.
#
# Usage:
#   ORGANISM=kp DRUG=colistin bash src/bac_pyseer/ast_gwas/scripts/run_readout.sh
#   ORGANISM=tb DRUG=ethionamide SKIP_FT=1 bash .../run_readout.sh   # unitig arm only
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
ORGANISM=${ORGANISM:?set ORGANISM=kp|tb}
DRUG=${DRUG:?set DRUG=<ast column name>}
DATA_ROOT=${BACPREDICT_DATA_ROOT:-${SCRATCHDIR:?set BACPREDICT_DATA_ROOT or SCRATCHDIR}}

case "$ORGANISM" in
    # The two organisms disagree on both the checkpoint subpath and the species prefix.
    kp) TASK=train_kleb_ast; FT_SUBDIR=models/finetune; SPECIES=klebsiella_pneumoniae ;;
    tb) TASK=train_tb_ast;   FT_SUBDIR=checkpoints;     SPECIES=mycobacterium_tuberculosis ;;
    *)  echo "ORGANISM must be kp or tb" >&2; exit 1 ;;
esac

OUT_DIR=${OUT_DIR:-$DATA_ROOT/processed/pyseer_ast/$ORGANISM}
DRUG_DIR=$OUT_DIR/$DRUG
MATRIX=$OUT_DIR/unitigs/unitigs.pyseer.gz
SPLIT_TABLE=${SPLIT_TABLE:-$DATA_ROOT/processed/$TASK/splits/${DRUG}_split.csv}
HITS=$DRUG_DIR/${DRUG}_hits_annotated.tsv
LOGDIR=${LOGDIR:-$DATA_ROOT/logs}

ACCT=${ACCT:-FLOTO-PROJECT-K-SL2-CPU}
PART=${PART:-icelake-himem}
GPU_ACCT=${GPU_ACCT:-FLOTO-SL2-GPU}
GPU_PART=${GPU_PART:-ampere}
CPUS=${CPUS:-16}; MEM=${MEM:-100G}; WALL=${WALL:-04:00:00}

# The GWAS chain's combine phase writes its own hit table under gwas/, but with pheno_var left at the
# 0.249 default on any run predating the combine-phase fix. Prefer $DRUG_DIR's, which is regenerated
# with --phenotype-tsv; fall back only if it is genuinely absent.
[ -s "$HITS" ] || HITS=$DRUG_DIR/gwas/${DRUG}_hits_annotated.tsv
for required in "$MATRIX" "$SPLIT_TABLE" "$HITS"; do
    [ -s "$required" ] || { echo "ERROR: missing $required — has the GWAS chain finished?" >&2; exit 1; }
done
mkdir -p "$LOGDIR"

echo "=== (A) unitig read-out (CPU) ==="
READOUT=$(sbatch --parsable --account="$ACCT" --partition="$PART" --nodes=1 --ntasks=1 \
    --cpus-per-task="$CPUS" --mem="$MEM" --time="$WALL" --job-name="uread_${DRUG}" \
    --output="$LOGDIR/uread_${DRUG}_%j.out" --error="$LOGDIR/uread_${DRUG}_%j.err" \
    --wrap "set -euo pipefail
        cd '$REPO'
        P=.venv/bin/python
        \$P -m bac_pyseer.ast_gwas.unitig_design_matrix \
            --hits-tsv '$HITS' --matrix-gz '$MATRIX' --split-table '$SPLIT_TABLE' \
            --out-dir '$DRUG_DIR/design' --decomp-threads $CPUS
        \$P -m bac_pyseer.ast_gwas.unitig_lr \
            --design-dir '$DRUG_DIR/design' --split-table '$SPLIT_TABLE' \
            --drug '$DRUG' --organism '$ORGANISM' --out-dir '$DRUG_DIR/lr' \
            --gwas-summary '$DRUG_DIR/${DRUG}_gwas_summary.json'
        \$P -m bac_pyseer.ast_gwas.unitig_design_matrix \
            --hits-tsv '$HITS' --matrix-gz '$MATRIX' --split-table '$SPLIT_TABLE' \
            --out-dir '$DRUG_DIR/design_dedup' --dedupe-patterns --decomp-threads $CPUS
        \$P -m bac_pyseer.ast_gwas.unitig_lr \
            --design-dir '$DRUG_DIR/design_dedup' --split-table '$SPLIT_TABLE' \
            --drug '$DRUG' --organism '$ORGANISM' --out-dir '$DRUG_DIR/lr_dedup' \
            --gwas-summary '$DRUG_DIR/${DRUG}_gwas_summary.json'")
echo "JOB $READOUT uread_${DRUG} | CPU | mem=$MEM | cores=$CPUS | wall=$WALL | $PART"

if [ "${SKIP_FT:-0}" = "1" ]; then
    echo "=== (B) skipped (SKIP_FT=1) — no fine-tune arm, so no paired CI ==="
    exit 0
fi

# Checkpoint step counts differ per drug (750, 2500, 7750, 31000 …), so glob rather than assume.
FT_RUN=$(ls -d "$DATA_ROOT/processed/$TASK/$FT_SUBDIR/${SPECIES}_${DRUG}_"* 2>/dev/null | head -1 || true)
CKPT=$(ls -d "$FT_RUN"/checkpoint-* 2>/dev/null | head -1 || true)
if [ -z "$CKPT" ]; then
    echo "=== (B) no fine-tune checkpoint under $TASK/$FT_SUBDIR for $DRUG — unitig arm only ==="
    exit 0
fi

echo "=== (B) fine-tune holdout re-score (GPU) ==="
# --n-folds/--fold/--seed/--evaluate-seed must match the training run or a DIFFERENT holdout is
# reconstructed and the two arms are silently compared on different genomes.
FOLDS=${FOLDS:-5}; FOLD=${FOLD:-0}; SEED=${SEED:-1}; EVAL_SEED=${EVAL_SEED:-1}
FTJOB=$(sbatch --parsable --account="$GPU_ACCT" --partition="$GPU_PART" --nodes=1 --ntasks=1 \
    --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00 --job-name="ftscore_${DRUG}" \
    --output="$LOGDIR/ftscore_${DRUG}_%j.out" --error="$LOGDIR/ftscore_${DRUG}_%j.err" \
    --wrap "set -euo pipefail
        cd '$REPO'
        .venv/bin/python -m bacpredict.engine.finetune.evaluate \
            --checkpoint '$CKPT' --drug '$DRUG' --task ${ORGANISM}_ast \
            --ast-sheet-path '$DATA_ROOT/processed/$TASK/binary_ast_with_split.csv' \
            --embeddings-dir '$DATA_ROOT/processed/$TASK/esm' \
            --n-folds $FOLDS --fold $FOLD --seed $SEED --evaluate-seed $EVAL_SEED \
            --batch-size 1 --num-workers 8 --out-dir '$FT_RUN'")
echo "JOB $FTJOB ftscore_${DRUG} | GPU x1 | mem=64G (ampere floors ~250G) | cores=8 | wall=01:00:00 | $GPU_PART"
echo "  checkpoint: $CKPT"

cat <<EOF

[look] When both land, build the comparison row — it recomputes the operating point for BOTH arms
from their own eval_scores.npz, so they cannot drift onto different threshold conventions:

  .venv/bin/python -m bac_pyseer.ast_gwas.collect_comparison \\
      --results-json '$DRUG_DIR/lr/results.json' \\
      --ft-scores ${DRUG}='$FT_RUN/eval_scores.npz' \\
      --out-csv '$OUT_DIR/comparison_${ORGANISM}.csv'

Report AUROC, AUPRC and balanced accuracy per arm, plus the paired delta and its CI. The sens/spec/
balanced accuracy are at Youden ON THE HOLDOUT — the best achievable operating point, optimistically
biased, never quoted as expected field sensitivity.
EOF
