#!/usr/bin/env bash
# SLURM array: build_genomes_array.sh, after checking this task's manifest against every other cohort's — the array a
# second cohort over the shared genome store runs (build_genomes_reusing_built.sh submits it). build_genomes deletes and
# rebuilds a genome whose stored src= header differs from its manifest row's source, so if an owner's manifest gained
# one of this manifest's genomes after submission, building it would destroy the owner's copy: the task refuses (exit 7)
# instead, before touching the store. Everything else is build_genomes_array.sh's, by --export, plus
#   OWNERS_FILE   one owner genome_manifest.csv path per line
#SBATCH --job-name=build-genomes-disjoint
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x_%A_%a.err
set -euo pipefail

: "${MANIFEST:?}" "${OWNERS_FILE:?}"
REPO=${REPO:-$HOME/workspace/BacPredict}
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
head=$(git rev-parse HEAD)
if [[ -n "${EXPECT_SHA:-}" && "$head" != "$EXPECT_SHA"* ]]; then
    echo "the checkout moved: HEAD ${head:0:7}, this array was submitted at $EXPECT_SHA"; exit 3
fi
mapfile -t owners < "$OWNERS_FILE"
(( ${#owners[@]} )) || { echo "no owner manifests in $OWNERS_FILE"; exit 7; }
uv run --no-sync python -m bacpredict.engine.genomes.reuse_built_genomes check-disjoint \
    --manifest "$MANIFEST" --built "${owners[@]}" || { echo "refusing to build: see above"; exit 7; }
exec bash "$REPO/src/bacpredict/engine/genomes/scripts/build_genomes_array.sh"
