#!/usr/bin/env bash
# One job: skani dist of every genome in a list against a reference panel → $QC_DIR/skani_dist.tsv, the species
# evidence collate_genomes reads (run_genome_qc.sh does the same for a two-reference cohort inside its CheckM2 job).
#
#   LIST       TSV, no header, absolute path: Sample<TAB>contigs path (absolute) — the CheckM2 list. Read once, at the
#              start, and published beside the result as $QC_DIR/skani_dist.list.tsv, like the references: a later
#              submit may rewrite LIST before this job's result exists, or after a job that never produced one
#   REFS_TSV   TSV, no header, absolute path: species<TAB>reference genome path (absolute); a species may repeat.
#              Read once, at the start: the copy the job runs with is the one it publishes
#   QC_DIR     writes $QC_DIR/skani_dist.tsv and, in the same step, $QC_DIR/skani_dist.refs.tsv (the REFS_TSV it
#              ran with): the result carries its own reference list, which collate reads (never the panel, never
#              REFS_TSV, which a later submit may have rewritten). The old list is removed before the new result
#              moves in, so a job killed between the two moves leaves a result with no list, which collate refuses
#   TOOLS_BIN  bin/ of the engine/genomes default pixi env (skani) — resolve at submit time, never pixi in a job
#   QCHUNK     genomes per skani call (default 5000): memory stays bounded however long the list; each (genome,
#              reference) pair is computed independently, so the calls concatenate to one run
#   REPO       the checkout (default ~/workspace/BacPredict); EXPECT_SHA: refuse to run if its HEAD is not that SHA
#
# Genomes are linked as <Sample>.fna.gz (collate names results by query file). skani is silent below ~80 % ANI, so
# a genome close to no reference has no row. Cost: 3,773 P. aeruginosa genomes against 2 references took seconds
# inside run_genome_qc.sh; ~68k genomes against ~40 references, reading ~100 GB of contigs off RDS, is minutes.
#SBATCH --job-name=skani
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x_%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x_%j.out
set -euo pipefail

: "${LIST:?}" "${REFS_TSV:?}" "${QC_DIR:?}" "${TOOLS_BIN:?}"
for f in "$LIST" "$REFS_TSV"; do
    [[ $f == /* && -r $f ]] || { echo "LIST and REFS_TSV must be readable absolute paths: $f"; exit 4; }
done
[[ -x $TOOLS_BIN/skani ]] || { echo "no skani in TOOLS_BIN=$TOOLS_BIN"; exit 4; }
REPO=${REPO:-$HOME/workspace/BacPredict}
QCHUNK=${QCHUNK:-5000}
CPUS=${SLURM_CPUS_PER_TASK:-32}
cd "$REPO"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
head=$(git rev-parse HEAD)
if [[ -n "${EXPECT_SHA:-}" && "$head" != "$EXPECT_SHA"* ]]; then
    echo "the checkout moved: HEAD ${head:0:7}, this job was submitted at $EXPECT_SHA"; exit 3
fi
# HEAD alone does not pin the code: a modified tracked file leaves HEAD unchanged, so the task would run code at no SHA
# while reporting the submitted one (d75e0d7). Tracked files only: a pinned worktree carries an untracked .venv symlink.
if [[ -n "${EXPECT_SHA:-}" && -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "$PWD is at ${head:0:7} but its tracked files are modified: what would run is at no SHA"; exit 3
fi

if [[ -d /local && -w /local ]]; then BASE=/local; else BASE=$HOME/rds/hpc-work/tmp; fi
WORK="$BASE/skani_${SLURM_JOB_ID:-local}"
mkdir -p "$WORK/links" "$QC_DIR"
trap 'rm -rf "$WORK"' EXIT
echo "[$(date -Is)] host=$(hostname) cpus=$CPUS qchunk=$QCHUNK repo=$REPO@${head:0:7} skani=$("$TOOLS_BIN/skani" --version)"

cp "$REFS_TSV" "$WORK/refs.tsv"  # the list this job runs with, and publishes
cp "$LIST" "$WORK/list.tsv"      # likewise the genomes
cut -f2 "$WORK/refs.tsv" > "$WORK/refs.txt"
while read -r ref; do
    [[ $ref == /* && -r $ref ]] || { echo "reference not readable: $ref"; exit 4; }
done < "$WORK/refs.txt"
awk -F'\t' -v d="$WORK/links" '{print $1 "\t" $2 "\t" d "/" $1 ".fna.gz"}' "$WORK/list.tsv" > "$WORK/plan.tsv"
while IFS=$'\t' read -r _ contigs link; do ln -s "$contigs" "$link"; done < "$WORK/plan.tsv"
cut -f3 "$WORK/plan.tsv" > "$WORK/queries.txt"
N=$(wc -l < "$WORK/queries.txt")
echo "[$(date -Is)] $N genomes linked, $(wc -l < "$WORK/refs.txt") references"

split -l "$QCHUNK" -d -a 4 "$WORK/queries.txt" "$WORK/q_"
for q in "$WORK"/q_*; do
    "$TOOLS_BIN/skani" dist -t "$CPUS" --ql "$q" --rl "$WORK/refs.txt" -o "$q.tsv"
    echo "[$(date -Is)] $(basename "$q"): $(wc -l < "$q") genomes, $(( $(wc -l < "$q.tsv") - 1 )) hits"
done
# one header: the first file's, then every file without its own
awk 'FNR == 1 && NR != 1 {next} {print}' "$WORK"/q_*.tsv > "$QC_DIR/skani_dist.tsv.part"
cp "$WORK/refs.tsv" "$QC_DIR/skani_dist.refs.tsv.part"
cp "$WORK/list.tsv" "$QC_DIR/skani_dist.list.tsv.part"
# result and its two lists move in together; the old lists go first, so no moment pairs a result with a list it was
# not run with
rm -f "$QC_DIR/skani_dist.refs.tsv" "$QC_DIR/skani_dist.list.tsv"
mv "$QC_DIR/skani_dist.tsv.part" "$QC_DIR/skani_dist.tsv"
mv "$QC_DIR/skani_dist.refs.tsv.part" "$QC_DIR/skani_dist.refs.tsv"
mv "$QC_DIR/skani_dist.list.tsv.part" "$QC_DIR/skani_dist.list.tsv"
hit=$(tail -n +2 "$QC_DIR/skani_dist.tsv" | cut -f2 | sort -u | wc -l)
echo "[$(date -Is)] skani done → $QC_DIR/skani_dist.tsv (references: $QC_DIR/skani_dist.refs.tsv):" \
     "$hit of $N genomes within ~80 % ANI of a reference"
