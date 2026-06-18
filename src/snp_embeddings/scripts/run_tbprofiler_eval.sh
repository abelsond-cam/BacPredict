#!/bin/bash
# TB-Profiler (WHO v2+ catalogue) over the evaluate-holdout assemblies — the AST comparator the concat
# models must beat. --fasta mode on raw/tb/assemblies/{Sample}.fa.gz (~12 s/genome, smoke-measured).
#
# Per genome: tb-profiler writes results/<Sample>.results.json with dr_variants (per-drug WHO-catalogue
# resistance calls + mutations), drtype, and lineage. The parser (parse_tbprofiler_calls.py) turns these
# into (a) a per-drug native R/S call and (b) a one-hot variant matrix → LR, both vs the phenotype.
#
# Restricted to the EVALUATE holdout (~7.3k genomes) — the same eval set the concat probes score on, so
# TB-Profiler's native call is a like-for-like comparator. (The one-hot-LR variant later needs train too;
# this eval set is the headline.)
#
# Runs as a SLURM array; each task profiles a strided slice of the eval list. A quick array smoke:
#     TBP_MAX_PER_TASK=3 sbatch --array=0 src/snp_embeddings/scripts/run_tbprofiler_eval.sh
# Full run:
#     sbatch src/snp_embeddings/scripts/run_tbprofiler_eval.sh
#
#SBATCH --job-name=tbprofiler_eval
#SBATCH --output=tbprofiler_eval_%A_%a.out
#SBATCH --error=tbprofiler_eval_%A_%a.err
#SBATCH --array=0-19
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --account=FLOTO-SL2-CPU
#SBATCH --open-mode=append
# CPU-only; ~12 s/genome × ~365 genomes/task ≈ 75 min/task at full 20-way. 12 h budget (over-request).

set -uo pipefail
cd /home/dca36/workspace/BacPredict/src/snp_embeddings/tbprofiler

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david
ASM=$RDS/raw/tb/assemblies
SHEET=$RDS/processed/train_tb_ast/binary_ast_with_split.csv
OUT=$RDS/processed/train_tb_ast/snp_embeddings/tbprofiler_eval
mkdir -p "$OUT"

NTASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASKID=${SLURM_ARRAY_TASK_ID:-0}
THREADS=${SLURM_CPUS_PER_TASK:-4}
MAX_PER_TASK=${TBP_MAX_PER_TASK:-0}   # >0 = smoke cap per task

# Eval-holdout Sample IDs (column-name-driven), strided across array tasks; deterministic order.
mapfile -t SAMPLES < <(awk -F, -v nt="$NTASKS" -v ti="$TASKID" '
  NR==1 { for (i=1;i<=NF;i++){ if($i=="Sample") sc=i; if($i=="train_val_eval") tc=i }; next }
  $tc=="evaluate" && $sc!="" { if (k % nt == ti) print $sc; k++ }' "$SHEET")

if [[ "$MAX_PER_TASK" -gt 0 ]]; then
    SAMPLES=("${SAMPLES[@]:0:$MAX_PER_TASK}")
fi

echo "=== tbprofiler_eval task $TASKID/$NTASKS: ${#SAMPLES[@]} genomes (threads=$THREADS) -> $OUT ==="
done_n=0; skip_n=0; fail_n=0
for S in "${SAMPLES[@]}"; do
    if [[ -s "$OUT/results/${S}.results.json" ]]; then skip_n=$((skip_n+1)); continue; fi   # idempotent resume
    GZ=$ASM/${S}.fa.gz
    if [[ ! -s "$GZ" ]]; then echo "MISSING assembly: $S" >&2; fail_n=$((fail_n+1)); continue; fi
    FA=${TMPDIR:-/tmp}/${S}.fa
    if ! zcat "$GZ" > "$FA"; then echo "DECOMPRESS FAIL: $S" >&2; fail_n=$((fail_n+1)); rm -f "$FA"; continue; fi
    if pixi run tb-profiler profile --fasta "$FA" --prefix "$S" --dir "$OUT" --threads "$THREADS" --txt >/dev/null 2>&1; then
        done_n=$((done_n+1))
    else
        echo "TBPROFILER FAIL: $S" >&2; fail_n=$((fail_n+1))
    fi
    rm -f "$FA"
done
echo "=== task $TASKID done: profiled=$done_n skipped(existing)=$skip_n failed=$fail_n ==="
