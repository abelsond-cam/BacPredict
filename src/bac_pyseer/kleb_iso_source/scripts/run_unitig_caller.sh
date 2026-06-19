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
BF_CSV=$TRAIN/blood_faeces/sampled_country_2_1_all/kpsc_human/binary_blood_vs_faeces_with_split.csv
RF_CSV=$TRAIN/faeces_respiratory/sampled_country_2_1_all/kpsc_human/binary_respiratory_vs_faeces_labels.csv
# Default = union of both invasion cohorts. Bifrost OOMs at the ~18k union on a 502 GB himem node,
# so for a lower-memory PER-COHORT build override OUT_NAME + COHORT_CSVS, e.g.:
#   OUT_NAME=faeces_respiratory COHORT_CSVS="$RF_CSV" sbatch … run_unitig_caller.sh
#   OUT_NAME=blood_faeces        COHORT_CSVS="$BF_CSV" sbatch … run_unitig_caller.sh
# (compare hits across per-cohort builds by unitig SEQUENCE — identical sequence = same unitig.)
OUT_NAME=${OUT_NAME:-blood_faeces_resp_union}
COHORT_CSVS=${COHORT_CSVS:-$BF_CSV $RF_CSV}
OUT_DIR=$PYSEER/unitigs/$OUT_NAME
REFLIST=$OUT_DIR/assembly_refs.txt        # 2-col record: Sample<TAB>assembly_path
PATHS=$OUT_DIR/assembly_paths.txt         # 1-col: just the paths (what unitig-caller --refs wants)
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/unitigs.pyseer" "$OUT_DIR/unitigs_raw.pyseer" "$OUT_DIR/unitigs.pyseer.gz"  # clear stale

echo "Job $SLURM_JOB_ID  Node $SLURMD_NODENAME  $(date)"

echo "=== (1) resolve union Sample -> assembly FASTA ==="
uv run python src/bac_pyseer/kleb_iso_source/resolve_assembly_paths.py \
    --sample-csv $COHORT_CSVS \
    --check-exists \
    --out-tsv "$REFLIST"

# unitig-caller --refs wants ONE fasta path per line (the 2-col name<TAB>path form silently
# builds an EMPTY graph — verified). Derive the 1-col paths from the resolution record.
cut -f2 "$REFLIST" > "$PATHS"
echo "unitig-caller refs: $(wc -l < "$PATHS") fasta paths"

echo "=== (2) unitig-caller: compressed DBG over the union assemblies -> pyseer unitig matrix ==="
pixi run --manifest-path "$PIXI_MANIFEST" unitig-caller \
    --call --refs "$PATHS" \
    --pyseer --out "$OUT_DIR/unitigs_raw" \
    --threads "$SLURM_CPUS_PER_TASK"

# fail loudly on an empty matrix — unitig-caller can exit 0 having built nothing
[ -s "$OUT_DIR/unitigs_raw.pyseer" ] || { echo "ERROR: unitig-caller produced an empty matrix"; exit 1; }

echo "=== (3) normalise column names to cohort Sample IDs (strip .fa/.fna/.fasta) + gzip ==="
# unitig-caller names columns by the fasta basename (e.g. SAMN0….fa); strip the extension so they
# equal the metadata_v2 Sample IDs the pyseer phenotype is keyed on, then gzip for pyseer --kmers.
sed -E 's/\.(fa|fna|fasta):/:/g' "$OUT_DIR/unitigs_raw.pyseer" | gzip > "$OUT_DIR/unitigs.pyseer.gz"
rm -f "$OUT_DIR/unitigs_raw.pyseer"

echo "Done  $(date)"
ls -lh "$OUT_DIR"
echo "unitigs: $(zcat "$OUT_DIR/unitigs.pyseer.gz" | wc -l) lines; first-line sample names:"
zcat "$OUT_DIR/unitigs.pyseer.gz" | head -1 | sed 's/^.*| //' | tr ' ' '\n' | head -3
