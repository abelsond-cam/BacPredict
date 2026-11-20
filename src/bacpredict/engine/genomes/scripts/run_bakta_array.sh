#!/usr/bin/env bash
# SLURM array: Bakta 1.9.4 (engine/genomes/bakta.py) on CHUNK genomes of a roster per task.
#
#   ROSTER     TSV, no header: Sample<TAB>contigs path. Task k takes rows k*CHUNK+1 … (k+1)*CHUNK.
#   OUT_DIR    <Sample>.bakta.<ext>.gz (KEEP, default gff3,faa,json,txt), the log, and <Sample>.bakta.run.json
#   TOOLS_BIN  bin/ of the engine/genomes `bakta` pixi env — resolve at submit time, never in a task
#   DB         Bakta database v5.1 full ($ROOT/raw/bakta_db/v5.1_full)
#   CHUNK      genomes per task (default 1); EXPECT_SHA: refuse to run if the shared checkout has moved
#
# A genome whose finished run was made from the same contigs (md5 in its run.json) is skipped, so a rerun of
# the same indices only redoes what is missing. Sizing (icelake-himem, 6,760 MB per core): 10 cores and 64 GB per
# task; nuna measured 543 s per Klebsiella genome at 8 threads on the same release and database.
#SBATCH --job-name=bakta
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x_%A_%a.out
set -euo pipefail

: "${ROSTER:?}" "${OUT_DIR:?}" "${TOOLS_BIN:?}" "${DB:?}"
REPO=${REPO:-$HOME/workspace/BacPredict}
CHUNK=${CHUNK:-1}
CPUS=${SLURM_CPUS_PER_TASK:-10}
K=${SLURM_ARRAY_TASK_ID:-0}
cd "$REPO"
head=$(git rev-parse HEAD)
if [[ -n "${EXPECT_SHA:-}" && "$head" != "$EXPECT_SHA"* ]]; then
    echo "the shared checkout moved: HEAD ${head:0:7}, this array was submitted at $EXPECT_SHA"; exit 3
fi

# Node-local disk for Bakta's temporary files; RDS scratch if /local is unavailable.
if [[ -d /local && -w /local ]]; then BASE=/local; else BASE=$HOME/rds/hpc-work/tmp; fi
WORK="$BASE/bakta_${SLURM_ARRAY_JOB_ID:-local}_${K}"
mkdir -p "$WORK" "$OUT_DIR"
trap 'rm -rf "$WORK"' EXIT

echo "[$(date -Is)] host=$(hostname) task=$K chunk=$CHUNK cpus=$CPUS repo=$REPO@${head:0:7} db=$DB"
fail=0
while IFS=$'\t' read -r sample contigs; do
    # --no-sync: concurrent tasks must not race to re-sync the shared venv
    if ! uv run --no-sync python -m bacpredict.engine.genomes.bakta run --contigs "$contigs" --sample "$sample" \
            --out-dir "$OUT_DIR" --tools-bin "$TOOLS_BIN" --db "$DB" --threads "$CPUS" --tmp-dir "$WORK" \
            --keep "${KEEP:-gff3,faa,json,txt}" --skip-existing; then
        echo "FAIL $sample"; fail=$((fail + 1))
    fi
done < <(sed -n "$((K * CHUNK + 1)),$(((K + 1) * CHUNK))p" "$ROSTER")
echo "[$(date -Is)] done, $fail failed"
[[ $fail -eq 0 ]]
