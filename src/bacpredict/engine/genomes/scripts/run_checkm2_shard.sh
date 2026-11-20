#!/usr/bin/env bash
# SLURM array: CheckM2 predict on CHUNK genomes of a frozen list per task — the sharded CheckM2 of a large cohort
# (run_genome_qc.sh runs CheckM2 + skani as one job, which at ~68k genomes would take ~2 days).
#
#   LIST         TSV, no header, absolute path, sorted and frozen (chmod a-w): Sample<TAB>contigs path (absolute).
#                Task k takes rows k*CHUNK+1 … (k+1)*CHUNK; a task whose slice is empty fails (exit 4: the --array
#                range does not match the list), and one with an unreadable contigs file fails (exit 5).
#   QC_DIR       writes $QC_DIR/checkm2/checkm2_quality_report_<k>.tsv (k zero-padded to 5 digits, as the build
#                shards), atomically; collate_genomes --checkm2 takes them all
#   CHECKM2_BIN  bin/ of the engine/genomes `checkm2` pixi env — resolve at submit time, never pixi in a task
#   CHECKM2_DB   CheckM2's uniref100.KO.1.dmnd (default: the copy on project_k shared with the Kp work)
#   CHUNK        genomes per task (default 2000)
#   REPO         the checkout to run (default ~/workspace/BacPredict); EXPECT_SHA: refuse to run if its HEAD is not
#                that SHA. For a long array, a detached worktree at the submitting SHA (see run_bakta_array.sh).
#
# CheckM2's DIAMOND works in 250-genome chunks and each genome is predicted independently, so the concatenated
# reports equal one run over the whole list. A task exits 6 if CheckM2 reports fewer genomes than it was given
# (the report is still written; collate names every built genome with no CheckM2 row).
# Cost: 3,773 P. aeruginosa genomes took 3 h 6 min at 32 cores, DIAMOND dominant (~1.6 core-min per genome), so
# 2,000 genomes ≈ 1.7 h at 32 cores: 4 h wall. Memory was never measured (the PsA job had icelake-himem's full
# 32 × 6,760 MB); 100 GB fits icelake's 32 × 3,380 MB. Read sacct MaxRSS on the first tasks.
#SBATCH --job-name=checkm2-shard
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --partition=icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x_%A_%a.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x_%A_%a.out
set -euo pipefail

: "${LIST:?}" "${QC_DIR:?}" "${CHECKM2_BIN:?}"
[[ $LIST == /* && -r $LIST ]] || { echo "LIST must be a readable absolute path: $LIST"; exit 4; }
[[ -x $CHECKM2_BIN/checkm2 ]] || { echo "no checkm2 in CHECKM2_BIN=$CHECKM2_BIN"; exit 4; }
REPO=${REPO:-$HOME/workspace/BacPredict}
CHUNK=${CHUNK:-2000}
CPUS=${SLURM_CPUS_PER_TASK:-32}
K=${SLURM_ARRAY_TASK_ID:-0}
DB=${CHECKM2_DB:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/CheckM2/Zenodo_db/CheckM2_database/uniref100.KO.1.dmnd}
export OMP_NUM_THREADS=$CPUS  # CSD3 exports 1 to every job; OpenMP tools would otherwise run on one core
cd "$REPO"
# $REPO's own code, even when $REPO is a worktree whose .venv is the shared env (its editable install points at the
# main checkout): PYTHONPATH precedes site-packages
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
head=$(git rev-parse HEAD)
if [[ -n "${EXPECT_SHA:-}" && "$head" != "$EXPECT_SHA"* ]]; then
    echo "the checkout moved: HEAD ${head:0:7}, this array was submitted at $EXPECT_SHA"; exit 3
fi
# CheckM2 runs prodigal and diamond from its own env, by name
export PATH="$CHECKM2_BIN:$PATH"

# Node-local disk for the links, Prodigal's proteins and DIAMOND's temporary files; RDS scratch if /local is unavailable.
if [[ -d /local && -w /local ]]; then BASE=/local; else BASE=$HOME/rds/hpc-work/tmp; fi
WORK="$BASE/checkm2_${SLURM_ARRAY_JOB_ID:-local}_${K}"
mkdir -p "$WORK/links" "$QC_DIR/checkm2"
trap 'rm -rf "$WORK"' EXIT
OUT=$(printf '%s/checkm2/checkm2_quality_report_%05d.tsv' "$QC_DIR" "$K")

echo "[$(date -Is)] host=$(hostname) task=$K chunk=$CHUNK cpus=$CPUS repo=$REPO@${head:0:7} db=$DB"
# mapfile, not `while read < <(sed …)`: a process substitution's failure is invisible to set -e, and read drops a
# final line with no newline
mapfile -t rows < <(sed -n "$((K * CHUNK + 1)),$(((K + 1) * CHUNK))p" "$LIST")
if (( ${#rows[@]} == 0 )); then
    echo "task $K: no list rows at CHUNK=$CHUNK — the array range does not match the list"; exit 4
fi
unreadable=()
for row in "${rows[@]}"; do
    IFS=$'\t' read -r sample contigs <<<"$row"
    if [[ $contigs == /* && -r $contigs ]]; then
        ln -s "$contigs" "$WORK/links/$sample.fna.gz"
    else
        unreadable+=("$sample:$contigs")
    fi
done
if (( ${#unreadable[@]} )); then
    echo "task $K: ${#unreadable[@]} contigs files are not readable absolute paths: ${unreadable[*]:0:10}"; exit 5
fi
echo "[$(date -Is)] ${#rows[@]} genomes linked"

"$CHECKM2_BIN/checkm2" predict --threads "$CPUS" --input "$WORK/links" -x fna.gz \
    --output-directory "$WORK/checkm2" --database_path "$DB" --tmpdir "$WORK" --force
cp "$WORK/checkm2/quality_report.tsv" "$OUT.part"
mv "$OUT.part" "$OUT"
n_rows=$(( $(wc -l < "$OUT") - 1 ))
echo "[$(date -Is)] checkm2 done: $n_rows of ${#rows[@]} genomes → $OUT"
if (( n_rows != ${#rows[@]} )); then
    # Name normalised as collate_genomes.read_checkm2 does (a trailing .fna or .fna.gz dropped), so a reported genome
    # is never listed as missing; one collation order for both sides of comm
    missing=$(LC_ALL=C comm -23 <(printf '%s\n' "${rows[@]}" | cut -f1 | LC_ALL=C sort) \
                       <(tail -n +2 "$OUT" | cut -f1 | sed -E 's/\.fna(\.gz)?$//' | LC_ALL=C sort) | head -20)
    echo "task $K: CheckM2 reported $n_rows of ${#rows[@]} genomes; missing (first 20): $(tr '\n' ' ' <<<"$missing")"; exit 6
fi
