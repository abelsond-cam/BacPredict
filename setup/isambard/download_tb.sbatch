#!/usr/bin/env bash
# Get the full TB AST cohort on Isambard-AI: filter the EBI all-species sheet to
# M. tuberculosis, then download ATB assemblies + BakRep Bakta GFF3s. Network-bound and
# large (~38k assemblies + GFFs), so runs on workq CPU via Slurm (must survive an SSH
# drop). Idempotent (all stages skip-existing) — re-run until it converges.
#SBATCH --job-name=download-tb-ast
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/projects/u6fp/david/logs/%x-%j.out
#SBATCH --error=/projects/u6fp/david/logs/%x-%j.out
set -uo pipefail
: "${PROJECTDIR:?}"
export PATH="$HOME/.pixi/bin:$PATH"

EBI_ALL="$PROJECTDIR/david/raw/ebi_amr_records_all_species_20260630.csv"
export BASE="$PROJECTDIR/david/raw/tb"
export METADATA="$BASE/ebi_tb_amr_records.csv"
export NCORES=32 BATCH=100
SETUP="$HOME/BacPredict/setup/isambard"
PIXI="pixi run --manifest-path $HOME/nuna/setup/isambard/tools/pixi.toml"
mkdir -p "$BASE"

echo "=== [$(date -u +%H:%M:%S)] select TB cohort from EBI all-species sheet ==="
$PIXI python "$HOME/BacPredict/src/tl/genome_download/scripts/select_ast_cohort.py" \
  --ebi-csv "$EBI_ALL" --species "Mycobacterium tuberculosis" --out "$METADATA"

echo "=== [$(date -u +%H:%M:%S)] assemblies (ATB) into $BASE ==="
bash "$SETUP/download_ast_assemblies.sh"
echo "=== [$(date -u +%H:%M:%S)] GFFs (BakRep) into $BASE ==="
bash "$SETUP/download_ast_gffs.sh"

A=$(find "$BASE/assemblies" -maxdepth 1 -name '*.fa.gz' -type f -size +0 2>/dev/null | wc -l)
G=$(find "$BASE/gff" -name '*.bakta.gff3.gz' -type f -size +0 2>/dev/null | wc -l)
echo "=== [$(date -u +%H:%M:%S)] DONE TB: assemblies=$A  gffs=$G  (target ~38k) ==="
