#!/bin/bash
#SBATCH --job-name=ggcat_unitigs
#SBATCH --output=ggcat_unitigs_%j.out
#SBATCH --error=ggcat_unitigs_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --mem=350G
#SBATCH --time=36:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU

# Build the unitig presence/absence matrix across ONE pyseer cohort's assemblies with GGCAT — the
# accessory/HGT feature space the reference-anchored variant GWAS can't see. GGCAT is disk-based
# and MEMORY-CAPPED (`-m`), so unlike unitig-caller's Bifrost backend (which OOM'd at >503 GB on a
# 502 GB himem node over even 9.2k genomes) it builds without OOMing.
#
# ONE BUILD PER COHORT (not a shared union). Each GWAS owns its inputs end-to-end (its assemblies ->
# its unitig matrix): repeatable, scales with the cohort rather than with "every contrast unioned",
# and avoids carrying unitigs that exist only in another cohort's samples (a union matrix generates
# those and each GWAS then discards them via the MAF filter). Cross-contrast comparison is done at
# the GENE level (align hit unitigs to MGH 78578 in Phase 2), for which separate builds are fully
# comparable — a union would add only exact-sequence comparability, which is marginal and not even
# guaranteed (unitig boundaries shift when the sample set changes).
#
# Coloured build via `-d <COLOR_NAME><TAB><FILE_PATH>` — the colour NAME is the Sample ID directly
# (our assembly_refs.txt is exactly Sample<TAB>path), so no basename/extension stripping is needed.
# GGCAT emits a unitig FASTA + a binary colormap; ggcat_to_pyseer.py joins
#   FASTA (unitig segment -> colour-subset id, HEX)  +  dump-colors (colour id -> Sample)  +
#   dump-colormap (subset id, DEC -> colour ids)  ->  the pyseer --kmers matrix `<seq> | <Sample>:1 …`.
# (GGCAT unitigs are sequence-maximal so colour can change along one; ggcat_to_pyseer.py splits the
#  segments into monochromatic features.)
#
# Node-fraction (memory-capped GGCAT → a fraction is safe, no OOM) so it backfills sooner.
# Idempotent: reuses an existing build + colour-names, only regenerates the colormap + matrix.
# Usage (run once per cohort):
#   OUT_NAME=blood_faeces       COHORT_CSVS="$BF_CSV" sbatch src/bac_pyseer/kleb_iso_source/scripts/run_ggcat_unitigs.sh
#   OUT_NAME=faeces_respiratory COHORT_CSVS="$RF_CSV" sbatch src/bac_pyseer/kleb_iso_source/scripts/run_ggcat_unitigs.sh

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
# Default = the blood/faeces cohort; run again with OUT_NAME=faeces_respiratory COHORT_CSVS="$RF_CSV"
# for the replication contrast (one build per cohort — see header).
OUT_NAME=${OUT_NAME:-blood_faeces}
COHORT_CSVS=${COHORT_CSVS:-$BF_CSV}
OUT_DIR=$PYSEER/unitigs/$OUT_NAME
REFLIST=$OUT_DIR/assembly_refs.txt        # COLOR_NAME(Sample)<TAB>assembly_path — GGCAT -d input
mkdir -p "$OUT_DIR"

K=${K:-31}
SVAL=${SVAL:-2}                           # min k-mer multiplicity: drop af≈1/N singletons (« pyseer MAF)
MEMGB=${MEMGB:-200}                        # GGCAT in-RAM temp budget (-m); rest spills to disk (-t)
THREADS=$SLURM_CPUS_PER_TASK
TMP=/home/dca36/rds/hpc-work/ggcat_tmp/$SLURM_JOB_ID    # big scratch (1 TB) for spill temp files
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

OUT_FASTA=$OUT_DIR/unitigs_ggcat.fa.gz
COLORS_DAT=${OUT_FASTA%.gz}.colors.dat     # GGCAT writes <output-sans-.gz>.colors.dat alongside
NAMES_JSONL=$OUT_DIR/color_names.jsonl
COLORMAP_CSV=$OUT_DIR/colormap_ranges.csv
MATRIX=$OUT_DIR/unitigs.pyseer.gz
rm -f "$MATRIX"   # regenerate the matrix; build, colour-names AND colormap are reused if present

echo "Job $SLURM_JOB_ID  Node $SLURMD_NODENAME  $(date)"
echo "OUT_NAME=$OUT_NAME  k=$K  s=$SVAL  -m ${MEMGB}G  threads=$THREADS  tmp=$TMP"

echo "=== (1) resolve cohort Sample -> assembly FASTA (reuse if already built) ==="
if [ -s "$REFLIST" ]; then
    echo "reusing existing reflist: $REFLIST ($(wc -l < "$REFLIST") samples)"
else
    uv run python src/bac_pyseer/kleb_iso_source/resolve_assembly_paths.py \
        --sample-csv $COHORT_CSVS --check-exists --out-tsv "$REFLIST"
fi
echo "GGCAT -d colours (samples): $(wc -l < "$REFLIST")"
# pyseer drops unitigs below ~1% MAF, so pre-filter the converter at 1% of the cohort: untestable
# rarer unitigs are skipped before any expansion (bounds output size; the disk sort-merge bounds RAM).
NSAMP=$(wc -l < "$REFLIST")
MIN_SAMP=$(awk -v n="$NSAMP" 'BEGIN{m=n*0.01; c=int(m); if(c<m)c++; if(c<1)c=1; print c}')
echo "converter --min-samples (1% MAF floor): $MIN_SAMP of $NSAMP"

echo "=== (2) ggcat build: coloured compacted DBG over the cohort assemblies (reuse if present) ==="
if [ -s "$OUT_FASTA" ] && [ -s "$COLORS_DAT" ]; then
    echo "reusing existing build: $OUT_FASTA + $COLORS_DAT (delete to force rebuild)"
else
    rm -f "$OUT_FASTA" "$COLORS_DAT"
    pixi run --manifest-path "$PIXI_MANIFEST" ggcat build \
        -c -k "$K" -s "$SVAL" -j "$THREADS" -m "$MEMGB" -t "$TMP" \
        -d "$REFLIST" -o "$OUT_FASTA"
    [ -s "$OUT_FASTA" ] || { echo "ERROR: ggcat produced no unitig FASTA"; exit 1; }
    [ -s "$COLORS_DAT" ] || { echo "ERROR: ggcat produced no colormap (.colors.dat)"; exit 1; }
fi

echo "=== (3) dump-colors: colour-id -> Sample ID (reuse if present) ==="
if [ ! -s "$NAMES_JSONL" ]; then
    pixi run --manifest-path "$PIXI_MANIFEST" ggcat dump-colors "$COLORS_DAT" "$NAMES_JSONL"
fi
[ -s "$NAMES_JSONL" ] || { echo "ERROR: dump-colors produced no colour names"; exit 1; }

echo "=== (4) dump-colormap (ranges-csv): subset-id -> colour ids (reuse if present) ==="
if [ -s "$COLORMAP_CSV" ]; then
    echo "reusing existing colormap: $COLORMAP_CSV ($(wc -l < "$COLORMAP_CSV") rows; delete to regenerate)"
else
    # dump-colormap expands only the subset ids it is given, so enumerate the distinct ids that
    # appear in the FASTA and feed them in ARG_MAX-safe chunks. NB the FASTA writes subset ids in HEX
    # and a unitig may carry several C: segments — extract all of them and convert hex->decimal (gawk
    # strtonum), since dump-colormap takes/emits DECIMAL subset ids.
    SUBSETS=$TMP/subset_ids.txt
    zcat "$OUT_FASTA" | awk '/^>/ { for (i=1;i<=NF;i++) if ($i ~ /^C:/) { split($i,p,":"); print strtonum("0x" p[2]) } }' \
        | sort -un > "$SUBSETS"
    echo "distinct colour subsets: $(wc -l < "$SUBSETS")"
    : > "$COLORMAP_CSV"
    split -l 100000 -d "$SUBSETS" "$TMP/subchunk_"
    for c in "$TMP"/subchunk_*; do
        pixi run --manifest-path "$PIXI_MANIFEST" ggcat dump-colormap \
            -k "$K" --format ranges-csv "$COLORS_DAT" "$TMP/cm_part.csv" $(cat "$c")
        cat "$TMP/cm_part.csv" >> "$COLORMAP_CSV"
    done
    [ -s "$COLORMAP_CSV" ] || { echo "ERROR: dump-colormap produced no colormap rows"; exit 1; }
    echo "colormap rows: $(wc -l < "$COLORMAP_CSV")"
fi

echo "=== (5) disk sort-merge join -> pyseer --kmers matrix (<seq> | <Sample>:1 …) ==="
uv run python src/bac_pyseer/kleb_iso_source/ggcat_to_pyseer.py \
    --fasta "$OUT_FASTA" --color-names "$NAMES_JSONL" --colormap "$COLORMAP_CSV" \
    --kmer-length "$K" --min-samples "$MIN_SAMP" --tmp-dir "$TMP" --out "$MATRIX"
[ -s "$MATRIX" ] || { echo "ERROR: empty pyseer matrix"; exit 1; }

echo "=== done  $(date) ==="
ls -lh "$OUT_DIR"
echo "unitigs (matrix lines): $(zcat "$MATRIX" | wc -l)"
echo "first-line sample tokens:"; zcat "$MATRIX" | head -1 | sed 's/^.*| //' | tr ' ' '\n' | head -3
