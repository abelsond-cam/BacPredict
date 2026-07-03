#!/usr/bin/env bash
# Get the full Klebsiella pneumoniae AST cohort on Isambard-AI: filter the EBI all-species
# sheet to K. pneumoniae, then download ATB assemblies + BakRep Bakta GFF3s. Network-bound
# (~6.5k assemblies + GFFs), runs on workq CPU via Slurm. Idempotent — re-run to converge.
#SBATCH --job-name=download-kleb-ast
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/projects/u6fp/david/logs/%x-%j.out
#SBATCH --error=/projects/u6fp/david/logs/%x-%j.out
set -uo pipefail
: "${PROJECTDIR:?}"
export PATH="$HOME/.pixi/bin:$PATH"

EBI_ALL="$PROJECTDIR/david/raw/ebi_amr_records_all_species_20260630.csv"
export BASE="$PROJECTDIR/david/raw/kleb_ast"
export METADATA="$BASE/ebi_kleb_amr_records.csv"
export NCORES=32 BATCH=100
SETUP="$HOME/BacPredict/setup/isambard"
PIXI="pixi run --manifest-path $HOME/nuna/setup/isambard/tools/pixi.toml"
mkdir -p "$BASE"

echo "=== [$(date -u +%H:%M:%S)] select Kp cohort from EBI all-species sheet ==="
$PIXI python "$HOME/BacPredict/src/tl/genome_download/scripts/select_ast_cohort.py" \
  --ebi-csv "$EBI_ALL" --species "Klebsiella pneumoniae" --out "$METADATA"

echo "=== [$(date -u +%H:%M:%S)] assemblies (ATB) into $BASE ==="
bash "$SETUP/download_ast_assemblies.sh"
echo "=== [$(date -u +%H:%M:%S)] GFFs (BakRep) into $BASE ==="
bash "$SETUP/download_ast_gffs.sh"

A=$(find "$BASE/assemblies" -maxdepth 1 -name '*.fa.gz' -type f -size +0 2>/dev/null | wc -l)
G=$(find "$BASE/gff" -name '*.bakta.gff3.gz' -type f -size +0 2>/dev/null | wc -l)
echo "=== [$(date -u +%H:%M:%S)] DONE Kp: assemblies=$A  gffs=$G  (target ~6.5k) ==="
