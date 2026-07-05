#!/usr/bin/env bash
# GFF-only recovery pass for the TB cohort: run the BakRep GFF3 convergence loop over the
# BioSamples that have an assembly but not yet a GFF (retries crashed batches until no new
# GFFs come down). Assumes assemblies are already on disk. Network-bound → workq CPU.
#SBATCH --job-name=download-tb-gffs
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%j.out
set -uo pipefail
: "${SCRATCHDIR:?}"
export PATH="$HOME/.pixi/bin:$PATH"
export BASE="$SCRATCHDIR/raw/tb"
export BATCH=100 MAX_PASSES=15
bash "$HOME/BacPredict/setup/isambard/download_ast_gffs.sh"
