#!/bin/bash
#SBATCH --job-name=unitig_lmm_shard
#SBATCH --output=/home/dca36/rds/hpc-work/pyseer_scratch/unitig_lmm_shard_%A_%a.out
#SBATCH --error=/home/dca36/rds/hpc-work/pyseer_scratch/unitig_lmm_shard_%A_%a.err
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
# Resource directives (partition/cpus/mem/time/array) are set PER PHASE by the orchestrator
# (run_unitig_lmm_sharded.sh) via sbatch CLI overrides — the defaults here are placeholders.
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00

# Worker for the SHARDED unitig LMM GWAS. One script, three PHASEs (set via $PHASE):
#
#   prep     — build the sublineage-clusters file + PRIME the LMM cache (eigendecomp of the n×n
#              core-SNP kinship + null h^2, saved once with --save-lmm) if missing, then SPLIT the
#              unitig matrix into $NSHARDS gzipped chunks (single zcat pass, round-robin by line —
#              unitigs are independent tests so any partition is exact).
#   task     — array body: run pyseer on chunk_${SLURM_ARRAY_TASK_ID}, REUSING the one cache via
#              --load-lmm (identical U/S/h^2 for every shard ⇒ per-unitig β/p identical to a single
#              run; sharding is exact, not an approximation). Emits chunk_i.assoc + patterns_i.
#   combine  — concatenate the shard .assoc (one header) + UNION the patterns (the only cross-unitig
#              quantity is the Bonferroni pattern count, a set union), then pyseer_postprocess
#              --feature-mode unitigs. Removes the scratch chunks on success.
#
# Why sharded (not just lower --cpu): pyseer ships each worker a copy of the n×n LMM rotation matrix,
# so peak RAM ≈ cpu × n² — that OOM'd the single 32-cpu run at 134 GB. Sharding bounds per-job cpu×n²
# AND parallelises across nodes. See PROGRESS.md / the kleb_iso_source CLAUDE.md.

set -euo pipefail
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
unset PYTHONPATH PYTHONHOME
REPO=/home/dca36/workspace/BacPredict
PIXI_MANIFEST=$REPO/src/bac_pyseer/pixi.toml
cd "$REPO"

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
TRAIN=$DATA/david/processed/train_iso_source
P=$DATA/david/processed/pyseer_iso_source
# Cohort knobs (defaults = blood/faeces); override via env for faeces/respiratory.
PAIR=${PAIR:-blood_faeces}
COHORT=${COHORT:-sampled_country_2_1_all}
LABEL_COL=${LABEL_COL:-blood_vs_faeces_label}
OUT_STEM=${OUT_STEM:-blood_vs_faeces_unitig}
POS_LABEL="${POS_LABEL:-blood (invasion)}"
PAIR_TITLE="${PAIR_TITLE:-blood vs faeces (unitigs)}"
COHORT_CSV=${COHORT_CSV:-$TRAIN/$PAIR/$COHORT/kpsc_human/binary_blood_vs_faeces_with_split.csv}
NSHARDS=${NSHARDS:-16}
CPU=${CPU:-${SLURM_CPUS_PER_TASK:-8}}
MIN_AF=${MIN_AF:-0.01}; MAX_AF=${MAX_AF:-0.99}
MIN_SL_SIZE=${MIN_SL_SIZE:-100}

CO=$P/$PAIR/$COHORT
GD=$CO/gwas_unitig_lmm
M=$P/unitigs/$PAIR/unitigs.pyseer.gz
CACHE=$GD/lmm_cache.npz
DIST=$CO/jaccard_distances.tsv
SIM=$CO/similarity.tsv
CLUST=$GD/sublineage_clusters.tsv
PHENO=$CO/phenotype.tsv
GFF=$DATA/david/raw/related_lr/gff/GCF_000016305.1.gff
# Keyed by PAIR *and* COHORT: the task phase writes chunk_$i.assoc / patterns_$i.txt in here, so two
# cohorts of the same pair (e.g. the full cohort and its train+validate-only counterpart) sharing one
# directory would silently overwrite each other's per-shard results. The gzipped chunks themselves are
# cohort-independent, so the only cost of separating them is one extra split pass. Overridable.
SHARD_DIR=${SHARD_DIR:-/home/dca36/rds/hpc-work/unitig_shards/$PAIR/$COHORT}
mkdir -p "$GD" "$SHARD_DIR"

PHASE=${PHASE:?set PHASE=prep|task|combine}
echo "PHASE=$PHASE  PAIR=$PAIR  NSHARDS=$NSHARDS  CPU=$CPU  job=$SLURM_JOB_ID  $(date)"
pyseer_run () { pixi run --manifest-path "$PIXI_MANIFEST" pyseer "$@"; }

# ---------------------------------------------------------------------------------------------------
case "$PHASE" in
prep)
    echo "=== (1) sublineage-clusters file (reuse if present) ==="
    if [ ! -s "$CLUST" ]; then
        uv run python - "$COHORT_CSV" "$PHENO" "$CLUST" "$MIN_SL_SIZE" <<'PY'
import sys, pandas as pd
cohort_csv, pheno_tsv, out, min_sl = sys.argv[1:5]; min_sl = int(min_sl)
samples = set(pd.read_csv(pheno_tsv, sep="\t")["samples"].astype(str))
meta = pd.read_csv(cohort_csv, usecols=["Sample", "Sublineage"], low_memory=False)
meta["Sample"] = meta["Sample"].astype(str)
meta = meta.drop_duplicates("Sample"); meta = meta[meta["Sample"].isin(samples)]
meta["Sublineage"] = meta["Sublineage"].fillna("unknown").astype(str).replace({"": "unknown", "nan": "unknown"})
if min_sl > 0:
    big = meta["Sublineage"].value_counts(); big = big.index[big >= min_sl]
    meta["Sublineage"] = meta["Sublineage"].where(meta["Sublineage"].isin(big), "other")
meta[["Sample", "Sublineage"]].to_csv(out, sep="\t", header=False, index=False)
print(f"wrote {out}: {len(meta)} samples, {meta['Sublineage'].nunique()} clusters")
PY
    else echo "reusing $CLUST"; fi

    echo "=== (2) prime LMM cache (eigendecomp + null h^2) if missing ==="
    if [ -s "$CACHE" ]; then
        echo "reusing existing cache $CACHE ($(ls -lh "$CACHE" | awk '{print $5}'))"
    else
        [ -s "$SIM" ] || { echo "ERROR: kinship $SIM missing — run the variant LMM first"; exit 1; }
        PRIME=$SHARD_DIR/_prime.kmers.gz
        zcat "$M" | head -500 | gzip > "$PRIME"
        echo "priming cache from $SIM (tiny kmer set; cache depends on samples+kinship, not kmers)"
        pyseer_run --kmers "$PRIME" --phenotypes "$PHENO" --phenotype-column "$LABEL_COL" \
            --lmm --similarity "$SIM" --save-lmm "$CACHE" --no-distances \
            --min-af "$MIN_AF" --max-af "$MAX_AF" --cpu "$CPU" > /dev/null
        rm -f "$PRIME"
        [ -s "$CACHE" ] || { echo "ERROR: cache not produced"; exit 1; }
        echo "wrote cache $CACHE ($(ls -lh "$CACHE" | awk '{print $5}'))"
    fi

    echo "=== (3) split matrix -> $NSHARDS gzipped chunks (single pass, round-robin) ==="
    rm -f "$SHARD_DIR"/chunk_*.gz
    zcat "$M" | awk -v n="$NSHARDS" -v d="$SHARD_DIR" \
        '{ f = sprintf("%s/chunk_%02d.gz", d, (NR-1)%n); print | ("gzip > " f) }'
    echo "chunk line counts:"; for c in "$SHARD_DIR"/chunk_*.gz; do echo "  $(basename "$c"): $(zcat "$c" | wc -l)"; done
    echo "=== prep done  $(date) ==="
    ;;

task)
    i=$(printf "%02d" "${SLURM_ARRAY_TASK_ID:?task phase needs --array}")
    CHUNK=$SHARD_DIR/chunk_$i.gz
    [ -s "$CHUNK" ] || { echo "ERROR: missing $CHUNK"; exit 1; }
    [ -s "$CACHE" ] || { echo "ERROR: missing cache $CACHE (run prep first)"; exit 1; }
    WD=$SHARD_DIR/work_$i; mkdir -p "$WD"; cd "$WD"   # own cwd so lineage_effects.txt doesn't collide
    echo "=== shard $i: pyseer --kmers $CHUNK --load-lmm (cpu=$CPU) ==="
    pyseer_run --kmers "$CHUNK" --phenotypes "$PHENO" --phenotype-column "$LABEL_COL" \
        --lmm --load-lmm "$CACHE" --distances "$DIST" --lineage --lineage-clusters "$CLUST" \
        --min-af "$MIN_AF" --max-af "$MAX_AF" --output-patterns "$SHARD_DIR/patterns_$i.txt" \
        --cpu "$CPU" > "$SHARD_DIR/chunk_$i.assoc"
    [ -s "$SHARD_DIR/chunk_$i.assoc" ] || { echo "ERROR: empty assoc for shard $i"; exit 1; }
    echo "shard $i done: $(wc -l < "$SHARD_DIR/chunk_$i.assoc") assoc lines  $(date)"
    ;;

combine)
    cd "$REPO"
    ASSOC=$GD/${OUT_STEM}.assoc
    PATTERNS=$GD/patterns.txt
    echo "=== concatenate $NSHARDS shard .assoc (one header) + union patterns ==="
    H=$(printf "%02d" 0)
    head -1 "$SHARD_DIR/chunk_$H.assoc" > "$ASSOC"
    for j in $(seq 0 $((NSHARDS-1))); do
        i=$(printf "%02d" "$j"); f=$SHARD_DIR/chunk_$i.assoc
        [ -s "$f" ] || { echo "ERROR: missing shard assoc $f"; exit 1; }
        tail -n +2 "$f" >> "$ASSOC"
    done
    cat "$SHARD_DIR"/patterns_*.txt > "$PATTERNS"
    echo "combined assoc lines: $(wc -l < "$ASSOC")   patterns (pre-dedup): $(wc -l < "$PATTERNS")"

    echo "=== postprocess (--feature-mode unitigs: lambda/QQ/threshold + VE-ranked hit table) ==="
    uv run python src/bac_pyseer/kleb_iso_source/pyseer_postprocess.py \
        --assoc "$ASSOC" --patterns "$PATTERNS" --gff "$GFF" --feature-mode unitigs \
        --out-fig-dir "$GD" --out-table "$GD/${OUT_STEM}_hits_annotated.tsv" \
        --summary-json "$GD/${OUT_STEM}_gwas_summary.json" \
        --contig NC_009648 --pos-label "$POS_LABEL" --neg-label faeces --pair-title "$PAIR_TITLE"
    echo "=== cleanup scratch chunks (keep $GD outputs) ==="
    rm -rf "$SHARD_DIR"/chunk_*.gz "$SHARD_DIR"/work_*
    echo "=== combine done  $(date) ==="; ls -lh "$GD"
    ;;
*) echo "unknown PHASE=$PHASE"; exit 1 ;;
esac
