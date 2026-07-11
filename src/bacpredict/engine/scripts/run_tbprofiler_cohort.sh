#!/bin/bash
# TB-Profiler (WHO v2+ catalogue) over the FULL labelled TB cohort — the AST comparator for the concat
# models. --fasta mode on raw/tb/assemblies/{Sample}.fa.gz (~12 s/genome, smoke-measured).
#
# Per genome: results/<Sample>.results.json with dr_variants (per-drug WHO-catalogue calls + mutations,
# incl. the non-coding causes rrs/rrl and the inhA/fabG1 promoter), drtype, lineage. The parser turns
# these into (a) a per-drug native R/S call and (b) a one-hot variant matrix → LR — the one-hot captures
# the non-coding mechanisms a protein-only embedding (ESM-C / current Bacformer) structurally cannot, so
# (one-hot − concat) AUROC per drug measures the un-embeddable contribution.
#
# Full labelled cohort (train+validate+evaluate, ~36.7k) so the one-hot LR can fit on train and score on
# eval on the same universe as concat. Wide SLURM array — one strided slice per task — to finish quickly
# (icelake-himem has the cores). Idempotent resume (skips existing JSONs), TBP_SPLIT / TBP_MAX_PER_TASK
# knobs. Smoke:  TBP_MAX_PER_TASK=3 sbatch --array=0 ...   Full:  sbatch run_tbprofiler_cohort.sh
#
#SBATCH --job-name=tbprofiler_cohort
#SBATCH --output=tbprofiler_cohort_%A_%a.out
#SBATCH --error=tbprofiler_cohort_%A_%a.err
#SBATCH --array=0-199
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH --account=FLOTO-SL2-CPU
#SBATCH --open-mode=append
# CPU-only; ~12 s/genome × ~184 genomes/task ≈ 37 min/task at full 200-way. 6 h budget (over-request).

set -uo pipefail
# Run from node-local scratch (NOT the pixi dir): tb-profiler drops intermediate .paf/.vcf files in CWD,
# which would litter the repo and collide across 200 tasks. Locate the env via --manifest-path instead.
MANIFEST=/home/dca36/workspace/BacPredict/src/bacpredict/apps/tb/tbprofiler/pixi.toml
WORK=${TMPDIR:-/tmp}/tbp_work_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$WORK" && cd "$WORK"

RDS=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david
ASM=$RDS/raw/tb/assemblies
SHEET=$RDS/processed/train_tb_ast/binary_ast_with_split.csv
OUT=$RDS/processed/train_tb_ast/pangena_predict/tbprofiler_calls
mkdir -p "$OUT"

NTASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASKID=${SLURM_ARRAY_TASK_ID:-0}
THREADS=${SLURM_CPUS_PER_TASK:-4}
SPLIT=${TBP_SPLIT:-all}                # all | train | validate | evaluate
MAX_PER_TASK=${TBP_MAX_PER_TASK:-0}    # >0 = smoke cap per task

# Labelled Sample IDs for SPLIT (column-name-driven), strided across array tasks; deterministic order.
mapfile -t SAMPLES < <(awk -F, -v nt="$NTASKS" -v ti="$TASKID" -v sp="$SPLIT" '
  NR==1 { for (i=1;i<=NF;i++){ if($i=="Sample") sc=i; if($i=="train_val_eval") tc=i }; next }
  (sp=="all" || $tc==sp) && $sc!="" && $tc!="" { if (k % nt == ti) print $sc; k++ }' "$SHEET")

if [[ "$MAX_PER_TASK" -gt 0 ]]; then
    SAMPLES=("${SAMPLES[@]:0:$MAX_PER_TASK}")
fi

echo "=== tbprofiler_cohort task $TASKID/$NTASKS split=$SPLIT: ${#SAMPLES[@]} genomes (threads=$THREADS) -> $OUT ==="
done_n=0; skip_n=0; fail_n=0
for S in "${SAMPLES[@]}"; do
    if [[ -s "$OUT/results/${S}.results.json" ]]; then skip_n=$((skip_n+1)); continue; fi   # idempotent resume
    GZ=$ASM/${S}.fa.gz
    if [[ ! -s "$GZ" ]]; then echo "MISSING assembly: $S" >&2; fail_n=$((fail_n+1)); continue; fi
    FA=${TMPDIR:-/tmp}/${S}.fa
    if ! zcat "$GZ" > "$FA"; then echo "DECOMPRESS FAIL: $S" >&2; fail_n=$((fail_n+1)); rm -f "$FA"; continue; fi
    if pixi run --manifest-path "$MANIFEST" tb-profiler profile --fasta "$FA" --prefix "$S" --dir "$OUT" --threads "$THREADS" --txt >/dev/null 2>&1; then
        done_n=$((done_n+1))
    else
        echo "TBPROFILER FAIL: $S" >&2; fail_n=$((fail_n+1))
    fi
    rm -f "$FA"
done
echo "=== task $TASKID done: profiled=$done_n skipped(existing)=$skip_n failed=$fail_n ==="
