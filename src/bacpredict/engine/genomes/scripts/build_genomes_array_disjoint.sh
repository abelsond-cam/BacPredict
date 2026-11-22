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
# HEAD alone does not pin the code: an edited file in the pinned worktree leaves HEAD unchanged, so the task would run
# code at no SHA while reporting the submitted one (found by the §3.6 agent in finetune_amr_array.sh, d75e0d7). Tracked
# files only: the worktree carries an untracked .venv symlink to the shared env.
if [[ -n "${EXPECT_SHA:-}" ]]; then
    dirty=$(git status --porcelain --untracked-files=no)
    [[ -z $dirty ]] || { echo "$REPO is at $EXPECT_SHA but its tracked files are modified:"; echo "$dirty" | head -20; exit 3; }
fi
mapfile -t owners < "$OWNERS_FILE"
(( ${#owners[@]} )) || { echo "no owner manifests in $OWNERS_FILE"; exit 7; }
uv run --no-sync python -m bacpredict.engine.genomes.reuse_built_genomes check-disjoint \
    --manifest "$MANIFEST" --built "${owners[@]}" || { echo "refusing to build: see above"; exit 7; }
# mlst (any2fasta) writes its temporary FASTA to $TMPDIR, by default /tmp, which was full on six icelake nodes and
# failed 374 NCBI genomes ("No space left on device"): give every tool node-local disk, as the build's own work dir is
if [[ -d /local && -w /local ]]; then
    export TMPDIR=/local/tmp_build_${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}
    mkdir -p "$TMPDIR"
    trap 'rm -rf "$TMPDIR"' EXIT
fi
bash "$REPO/src/bacpredict/engine/genomes/scripts/build_genomes_array.sh"
