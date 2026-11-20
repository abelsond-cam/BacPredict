#!/usr/bin/env bash
# SLURM array: build one chunk of an AST cohort's genomes (fetch/assemble → Prodigal 2.6.3 → mlst).
# One task = one chunk of `genome_manifest.csv` rows whose source is in $SOURCES (see build_genomes.py).
#
# Everything is passed by --export; size the array with build_genomes --count-chunks. Two shapes:
#   download sources (atb, portal_gff): CHUNK≈40, 8 cores (one genome per core), ~15 min
#   shovill: CHUNK=1, 8 cores / 54 GB (icelake-himem 6,760 MB per core), ~45 min per genome; needs GENOME_SIZE (bp)
# e.g.
#   N=$(uv run python -m bacpredict.engine.genomes.build_genomes --manifest $M --sources shovill \
#         --chunk-size 1 --count-chunks)
#   sbatch --array=0-$((N-1)) --cpus-per-task=8 --time=03:00:00 --job-name=build-genomes-shovill \
#     --export=ALL,MANIFEST=$M,SOURCES=shovill,CHUNK=1,OUT_ROOT=$G,SHARD_DIR=$S,MLST_SCHEME=paeruginosa,GENOME_SIZE=6.6e6 \
#     src/bacpredict/engine/genomes/scripts/build_genomes_array.sh
#
# REPO (default ~/workspace/BacPredict): the checkout to run. For a long array, a detached worktree at the submitting
# SHA with .venv symlinked to the shared env, so other agents' pulls of the shared checkout cannot change the code
# mid-array. EXPECT_SHA: refuse to run if REPO's HEAD is not it.
#SBATCH --job-name=build-genomes
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x_%A_%a.err
set -euo pipefail

: "${MANIFEST:?}" "${SOURCES:?}" "${CHUNK:?}" "${OUT_ROOT:?}" "${SHARD_DIR:?}"
REPO=${REPO:-$HOME/workspace/BacPredict}
CPUS=${SLURM_CPUS_PER_TASK:-8}
RAM_GB=$(( CPUS * 6 ))   # shovill --ram, under the 6.76 GB/core the partition grants

cd "$REPO"
# $REPO's own code, even when $REPO is a worktree whose .venv is the shared env (its editable install points at the
# main checkout): PYTHONPATH precedes site-packages
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
head=$(git rev-parse HEAD)
if [[ -n "${EXPECT_SHA:-}" && "$head" != "$EXPECT_SHA"* ]]; then
    echo "the checkout moved: HEAD ${head:0:7}, this array was submitted at $EXPECT_SHA"; exit 3
fi
# Normally resolved once at submit time (build_pa_genomes.sh exports it); fall back to resolving here.
TOOLS_BIN=${TOOLS_BIN:-"$(cd src/bacpredict/engine/genomes && pixi run -- printenv CONDA_PREFIX)/bin"}

# Node-local disk for SPAdes' random I/O and the fastq; RDS scratch if /local is unavailable.
if [[ -d /local && -w /local ]]; then BASE=/local; else BASE=$HOME/rds/hpc-work/tmp; fi
WORK="$BASE/build_genomes_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

echo "[$(date -Is)] host=$(hostname) task=${SLURM_ARRAY_TASK_ID:-0} sources=$SOURCES chunk=$CHUNK cpus=$CPUS work=$WORK repo=$REPO@$(git -C "$REPO" rev-parse --short HEAD)"
# --no-sync: hundreds of concurrent tasks must not race to re-sync the shared venv
uv run --no-sync python -m bacpredict.engine.genomes.build_genomes \
    --manifest "$MANIFEST" --sources "$SOURCES" --chunk-size "$CHUNK" --task-index "${SLURM_ARRAY_TASK_ID:-0}" \
    --out-root "$OUT_ROOT" --shard-dir "$SHARD_DIR" --tools-bin "$TOOLS_BIN" \
    --threads "$CPUS" --ram-gb "$RAM_GB" --tmp-dir "$WORK" ${MLST_SCHEME:+--mlst-scheme "$MLST_SCHEME"} \
    ${GENOME_SIZE:+--genome-size "$GENOME_SIZE"}
echo "[$(date -Is)] done"
