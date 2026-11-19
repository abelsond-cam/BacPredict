#!/usr/bin/env bash
# CheckM2 + skani over every successfully built genome of a cohort (the inputs to collate_genomes).
#
#   sbatch --export=ALL,SHARD_DIR=$S,QC_DIR=$T/genomes/qc,REFS="$R/raw/reference_genomes/GCF_000006765.1.fna.gz \
#            $R/raw/reference_genomes/GCF_000017205.1.fna.gz" src/bacpredict/engine/genomes/scripts/run_genome_qc.sh
#
# CheckM2's DIAMOND step is the cost: 3,773 P. aeruginosa genomes took 3 h 6 min at 32 cores (DIAMOND 2 h 12 min),
# so keep the 6 h wall for a cohort this size and scale it with genome count. skani is seconds.
#SBATCH --job-name=genome-qc
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x_%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x_%j.err
set -euo pipefail

: "${SHARD_DIR:?}" "${QC_DIR:?}" "${REFS:?}"
REPO=${REPO:-$HOME/workspace/BacPredict}
CPUS=${SLURM_CPUS_PER_TASK:-32}
export OMP_NUM_THREADS=$CPUS  # CSD3 exports 1 to every job; OpenMP tools would otherwise run on one core
DB=${CHECKM2_DB:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/CheckM2/Zenodo_db/CheckM2_database/uniref100.KO.1.dmnd}

cd "$REPO/src/bacpredict/engine/genomes"
if [[ -d /local && -w /local ]]; then BASE=/local; else BASE=$HOME/rds/hpc-work/tmp; fi
WORK="$BASE/genome_qc_${SLURM_JOB_ID:-local}"
mkdir -p "$WORK/links" "$QC_DIR"
trap 'rm -rf "$WORK"' EXIT

# Link every genome whose build row is ok, as <Sample>.fna.gz (CheckM2 and skani name results by file).
awk -F'\t' 'FNR == 1 {for (i = 1; i <= NF; i++) col[$i] = i; next}
           $col["status"] == "ok" {print $col["Sample"] "\t" $col["contigs_path"]}' "$SHARD_DIR"/*.tsv | sort -u -k1,1 \
  | while IFS=$'\t' read -r sample path; do ln -sf "$path" "$WORK/links/$sample.fna.gz"; done
N=$(find "$WORK/links" -name '*.fna.gz' | wc -l)
echo "[$(date -Is)] $N genomes linked"

find "$WORK/links" -name '*.fna.gz' | sort > "$WORK/queries.txt"
printf '%s\n' $REFS > "$WORK/refs.txt"
pixi run -- skani dist -t "$CPUS" --ql "$WORK/queries.txt" --rl "$WORK/refs.txt" -o "$QC_DIR/skani_dist.tsv"
echo "[$(date -Is)] skani done → $QC_DIR/skani_dist.tsv"

pixi run -e checkm2 checkm2 predict --threads "$CPUS" --input "$WORK/links" -x fna.gz \
    --output-directory "$WORK/checkm2" --database_path "$DB" --tmpdir "$WORK" --force
cp "$WORK/checkm2/quality_report.tsv" "$QC_DIR/checkm2_quality_report.tsv"
echo "[$(date -Is)] checkm2 done → $QC_DIR/checkm2_quality_report.tsv"
