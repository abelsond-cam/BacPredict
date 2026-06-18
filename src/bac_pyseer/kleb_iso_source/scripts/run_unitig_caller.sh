#!/bin/bash
#SBATCH --job-name=unitig_caller
#SBATCH --output=unitig_caller_%j.out
#SBATCH --error=unitig_caller_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --mem=480G
#SBATCH --time=36:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU

# Build the unitig presence/absence matrix across the cohort assemblies — the accessory/HGT
# feature space the reference-anchored variant GWAS can't see. One compressed de Bruijn graph
# over the UNION of the invasion cohorts (blood/faeces + faeces/respiratory) so the same
# unitig ids appear in both GWASes (directly comparable hits, as for the variant analysis);
# each GWAS then runs pyseer with its own cohort phenotype (pyseer intersects samples).
#
# Generous himem (large DBG over ~17-18k genomes). Idempotent on the ref-list build.
# Usage: sbatch src/bac_pyseer/kleb_iso_source/scripts/run_unitig_caller.sh

set -euo pipefail
export PYTHONUNBUFFERED=1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
unset PYTHONPATH PYTHONHOME
REPO=/home/dca36/workspace/BacPredict
PIXI_MANIFEST=$REPO/src/bac_pyseer/pixi.toml
cd "$REPO"

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
TRAIN=$DATA/david/processed/train_iso_source
PYSEER=$DATA/david/processed/pyseer_iso_source
OUT_DIR=$PYSEER/unitigs/blood_faeces_resp_union
REFLIST=$OUT_DIR/assembly_refs.txt
mkdir -p "$OUT_DIR"

BF_CSV=$TRAIN/blood_faeces/sampled_country_2_1_all/kpsc_human/binary_blood_vs_faeces_with_split.csv
RF_CSV=$TRAIN/faeces_respiratory/sampled_country_2_1_all/kpsc_human/binary_respiratory_vs_faeces_labels.csv

echo "Job $SLURM_JOB_ID  Node $SLURMD_NODENAME  $(date)"

echo "=== (1) resolve union Sample -> assembly FASTA ==="
uv run python src/bac_pyseer/kleb_iso_source/resolve_assembly_paths.py \
    --sample-csv "$BF_CSV" "$RF_CSV" \
    --check-exists \
    --out-tsv "$REFLIST"

echo "=== (2) unitig-caller: compressed DBG over the union assemblies -> pyseer unitig matrix ==="
pixi run --manifest-path "$PIXI_MANIFEST" unitig-caller \
    --call --refs "$REFLIST" \
    --pyseer --out "$OUT_DIR/unitigs" \
    --threads "$SLURM_CPUS_PER_TASK"

echo "Done  $(date)"
ls -lh "$OUT_DIR"
