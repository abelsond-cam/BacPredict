#!/usr/bin/env bash
# Parse the TB-Profiler call set once -> the four artifacts both consumers read.
#
# TB-Profiler already ran over the TB cohort (June 2026, for the concat comparator), so this parses
# the existing <Sample>.results.json rather than re-calling 36k assemblies. That is legitimate only
# because a results.json is a deterministic function of one assembly and one catalogue version --
# which parse_tbprofiler_calls now records, and refuses to mix.
#
# Two consumers, one parse:
#   tbprofiler_lineage.csv      -> bac_pyseer.ast_gwas.tb_lineage_from_tbprofiler (comparator
#                                  clusters + within-lineage permutation strata)
#   tbprofiler_variants.parquet -> bacpredict.apps.tb.tbprofiler_gene_lr (the WHO ceiling)
#
# The calls live in the DEPRECATED tree (processed/train_tb_ast/...); the output goes to the
# CANONICAL one (bac_ast_prediction/processed/train_tb_ast/...). Those really are two different
# trees with two different cohort sizes -- 36,684 vs 36,692 -- so both paths are explicit here
# rather than resolved from one root.
#
# Usage:
#   bash src/bacpredict/apps/tb/scripts/parse_tbprofiler_cohort.sh          # submit
#   SUBMIT=0 bash .../parse_tbprofiler_cohort.sh                            # print the sbatch only
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
RDS=${RDS:-/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david}

CALLS_DIR=${CALLS_DIR:-$RDS/processed/train_tb_ast/snp_embeddings/tbprofiler_calls/results}
OUT_DIR=${OUT_DIR:-$RDS/bac_ast_prediction/processed/train_tb_ast/tbprofiler}
COHORT_CSV=${COHORT_CSV:-$RDS/bac_ast_prediction/processed/train_tb_ast/binary_ast_with_split.csv}
LOGDIR=${LOGDIR:-$RDS/logs/tb_tbprofiler_parse}

ACCT=${ACCT:-FLOTO-PROJECT-K-SL2-CPU}
PART=${PART:-icelake-himem}
QOS=${QOS-}                      # CSD3 FLOTO associations allow only cpu1/intr; empty omits --qos
CPUS=${CPUS:-1}
MEM=${MEM:-8G}                   # MEASURED 4.0 GB peak (sacct MaxRSS, job 34597335) against a 4G request
                                 # -- it completed, but at the cap. The cost is the TRANSIENT per-file
                                 # object graph (gene_name2locus_tag, other_variants, the lineage support
                                 # arrays: 108 KB of JSON -> multi-MB of Python objects), which CPython
                                 # does not return to the OS, so RSS tracks the high-water mark. The
                                 # accumulators are ~85 MB and were never the cost.
TIME=${TIME:-00:40:00}           # MEASURED 17:51 wall for 36,684 files (job 34597335); ~2.2x cushion

[ -d "$CALLS_DIR" ] || { echo "ERROR: no calls dir at $CALLS_DIR" >&2; exit 1; }
[ -s "$COHORT_CSV" ] || { echo "ERROR: no cohort csv at $COHORT_CSV" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$LOGDIR"

JOB=$(sbatch --parsable --account="$ACCT" --partition="$PART" ${QOS:+--qos="$QOS"} \
    --job-name=tb-tbprofiler-parse --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
    --chdir="$REPO" --output="$LOGDIR/parse_%j.out" --error="$LOGDIR/parse_%j.err" \
    --wrap "set -euo pipefail; export PYTHONPATH=$REPO/src:\${PYTHONPATH:-}; \
        uv run python -m bacpredict.apps.tb.parse_tbprofiler_calls \
            --results-dir '$CALLS_DIR' --out-dir '$OUT_DIR' --cohort-csv '$COHORT_CSV'")

echo "JOB $JOB tb-tbprofiler-parse | CPU    | mem=$MEM  | cores=$CPUS  | wall=$TIME | $PART"
echo "  mem=\$MEM: measured 4.0 GB peak at 36,684 files (job 34597335); 2x that, since the driver is"
echo "            per-file transient objects and CPython's arena high-water mark, not the accumulators."
echo "  wall=$TIME: 36,684 x ~108 KB JSON = ~4 GB, cold-RDS metadata bound (~28 ms/file worst case)."
echo "  logs: $LOGDIR/parse_$JOB.{out,err}"
echo "  out:  $OUT_DIR"
