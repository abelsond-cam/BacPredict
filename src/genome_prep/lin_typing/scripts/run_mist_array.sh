#!/usr/bin/env bash
# LIN-type a batch of assemblies with MiST: one SLURM array task handles CHUNK genomes.
#
# Each task decompresses its assemblies into $TMPDIR and calls them one at a time. The assemblies
# live in a SHARED directory as .fa.gz and MiST needs plain FASTA, so decompressing in place would
# mutate data other people depend on -- hence the copy. $TMPDIR is node-local and cleaned by SLURM.
#
# Already-present output JSONs are skipped, so a partially failed array is resumed by resubmitting
# it unchanged rather than by working out which tasks died.
#
# Usage:
#   WORK_LIST=... DB=... OUT_DIR=... sbatch --array=0-53%25 run_mist_array.sh
set -euo pipefail

#SBATCH --job-name=mist-lincodes
#SBATCH --nodes=1
#SBATCH --ntasks=1

WORK_LIST=${WORK_LIST:?set WORK_LIST (Sample<TAB>path)}
DB=${DB:?set DB (MiST index directory)}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
CHUNK=${CHUNK:-8}
THREADS=${THREADS:-${SLURM_CPUS_PER_TASK:-4}}
ENV_NAME=${ENV_NAME:-mist-lincodes}

mkdir -p "$OUT_DIR"
TASK=${SLURM_ARRAY_TASK_ID:-0}
START=$(( TASK * CHUNK + 1 ))
END=$(( START + CHUNK - 1 ))

echo "task $TASK -> work-list lines $START..$END (chunk=$CHUNK threads=$THREADS)"
echo "db=$DB"
echo "out=$OUT_DIR"

n_done=0 n_skip=0 n_fail=0
while IFS=$'\t' read -r sample path; do
    [ -z "${sample:-}" ] && continue
    out_json=$OUT_DIR/$sample.json
    if [ -s "$out_json" ]; then
        echo "  skip  $sample (already called)"
        n_skip=$(( n_skip + 1 ))
        continue
    fi
    fa=$TMPDIR/$sample.fa
    if ! zcat "$path" > "$fa" 2>/dev/null; then
        echo "  FAIL  $sample: cannot decompress $path"
        n_fail=$(( n_fail + 1 ))
        continue
    fi
    if micromamba run -n "$ENV_NAME" mist call \
            --db "$DB" --fasta "$fa" --out-json "$out_json" \
            --sample-id "$sample" --threads "$THREADS" >/dev/null 2>&1; then
        echo "  ok    $sample"
        n_done=$(( n_done + 1 ))
    else
        echo "  FAIL  $sample: mist call returned non-zero"
        rm -f "$out_json"
        n_fail=$(( n_fail + 1 ))
    fi
    rm -f "$fa"
done < <(sed -n "${START},${END}p" "$WORK_LIST")

echo "task $TASK done: $n_done called, $n_skip skipped, $n_fail failed"
[ "$n_fail" -eq 0 ]
