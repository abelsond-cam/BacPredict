#!/bin/bash
#SBATCH --job-name=pyseer_blood_faeces
#SBATCH --output=pyseer_blood_faeces_%j.out
#SBATCH --error=pyseer_blood_faeces_%j.err
#SBATCH --partition=icelake-himem  # himem: far less oversubscribed than icelake (see ~/check_slurm_nodes.sh) and its ~6.7GB/core fits 128G on 32 cores
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU

# Blood-vs-faeces GWAS with pyseer (fixed-effects + MDS population-structure correction),
# then our own diagnostics + gene mapping. Four steps:
#   0) build the sublineage-clusters file (sample -> Sublineage) from the cohort split CSV.
#   1) scree_plot_pyseer over the Jaccard distances -> scree_plot.png (informs the K choice).
#   2) pyseer: per-variant logistic regression with K MDS axes as covariates, --save-m so a
#      K-sensitivity rerun is cheap (--load-m). lrt-pvalue is the structure-adjusted p.
#   3) pyseer_postprocess.py (uv env): Bonferroni-on-patterns threshold, genomic-inflation λ
#      + QQ, Manhattan, and the GFF-annotated significant-hit table (+ virulence cross-ref).
#
# pyseer + scree_plot_pyseer come from the bac_pyseer pixi env (isolated from the uv env so
# their numpy/scipy never perturbs the Bacformer pytorch stack); our post-processing runs
# back under `uv run python`. Default K=10 (pyseer-tutorial ballpark); override as $1 and
# rerun against --load-m if λ says the correction is mis-calibrated.
#
# Usage: sbatch [--time=H:MM:SS] src/bac_pyseer/kleb_iso_source/scripts/run_pyseer.sh [K] [output_subdir] [min_sl_size]
#   output_subdir (default "gwas") isolates outputs — pass a fresh name (e.g. gwas_36h) to run a
#   second, independent GWAS alongside one already in flight without the two clobbering each other.
#   min_sl_size (default 0 = keep all) collapses Sublineages with fewer samples into 'other', so
#   --lineage attributes hits only to the big SLs (e.g. 100) instead of ~1300 tiny n=1-5 clusters.
#   Env overrides: MIN_AF / MAX_AF (default 0.01/0.99) set the allele-freq window; USE_LINEAGE=0
#   omits --lineage entirely; USE_LMM=1 uses a linear mixed model (random effects via a kinship
#   matrix built from the Rtab by similarity_pyseer) instead of fixed-effects MDS — the better
#   structure correction for clonal data (no K to truncate). e.g. USE_LMM=1 sbatch ... 10 gwas_lmm 100
#   e.g. USE_LINEAGE=0 MIN_AF=0.05 MAX_AF=0.95 sbatch ... 10 gwas_nolin_af5 0

set -euo pipefail
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg  # headless node: scree_plot_pyseer (+ any mpl use) must not pick TkAgg
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
unset PYTHONPATH PYTHONHOME

REPO=/home/dca36/workspace/BacPredict
PIXI_MANIFEST=$REPO/src/bac_pyseer/pixi.toml
cd "$REPO"

K=${1:-10}

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
# Toggle these (env) to retarget a pair/cohort; defaults = blood/faeces sampled_country_2_1_all.
# e.g.  PAIR=faeces_respiratory LABEL_COL=respiratory_vs_faeces_label OUT_STEM=respiratory_vs_faeces \
#       POS_LABEL='respiratory (invasion)' PAIR_TITLE='faeces vs respiratory' \
#       COHORT_CSV=…/faeces_respiratory/$COHORT/kpsc_human/binary_respiratory_vs_faeces_labels.csv \
#       USE_LMM=1 sbatch … run_pyseer.sh 10 gwas_lmm 100
PAIR=${PAIR:-blood_faeces}
COHORT=${COHORT:-sampled_country_2_1_all}
LABEL_COL=${LABEL_COL:-blood_vs_faeces_label}
OUT_STEM=${OUT_STEM:-blood_vs_faeces}            # filename stem for the .assoc / hit table / summary
POS_LABEL="${POS_LABEL:-blood (invasion)}"       # postprocess direction label for β>0 (phenotype==1)
NEG_LABEL="${NEG_LABEL:-faeces}"                 # ... and for β<0 (phenotype==0)
PAIR_TITLE="${PAIR_TITLE:-blood vs faeces}"      # Manhattan title contrast name
COHORT_CSV=${COHORT_CSV:-$DATA/david/processed/train_iso_source/$PAIR/$COHORT/kpsc_human/binary_blood_vs_faeces_with_split.csv}
IN_DIR=$DATA/david/processed/pyseer_iso_source/$PAIR/$COHORT
GWAS_SUBDIR=${2:-gwas}   # 2nd arg = output subdir; use a fresh name to run an isolated GWAS alongside another in flight
MIN_SL_SIZE=${3:-0}      # 3rd arg = min samples/Sublineage to keep as its own --lineage cluster; smaller SLs collapse to 'other' (0 = keep all)
MIN_AF=${MIN_AF:-0.01}          # env: allele-frequency window; raise to 0.05/0.95 to drop the rare, separating variants that force slow Firth fits
MAX_AF=${MAX_AF:-0.99}
USE_LINEAGE=${USE_LINEAGE:-1}   # env: 0 = omit --lineage entirely (no per-variant SL attribution); 1 = keep it
USE_LMM=${USE_LMM:-0}           # env: 1 = LMM (FaST-LMM random effects via a kinship matrix) instead of fixed-effects MDS — better structure control for clonal data
FEATURES=${FEATURES:-variants}  # env: variants (--pres core-SNP Rtab) | unitigs (--kmers GGCAT matrix = the accessory/HGT axis)
GWAS_DIR=$IN_DIR/$GWAS_SUBDIR
GFF=$DATA/david/raw/related_lr/gff/GCF_000016305.1.gff
EFFECT_MAP=$DATA/david/processed/pyseer_iso_source/source_hotspot/locus_effect_map.tsv.gz  # per-hit SNP consequence (variant mode)
mkdir -p "$GWAS_DIR"

RTAB=$IN_DIR/variant_by_loci_presence.Rtab
# Per-cohort GGCAT unitig matrix (Phase 1 output). For FEATURES=unitigs we swap --pres RTAB -> --kmers
# this, but DELIBERATELY reuse the variant phenotype + core-SNP kinship (similarity.tsv) below: a
# unitig-derived kinship would let the LMM random effect absorb the accessory/HGT structure we are
# trying to detect, whereas the core-SNP kinship corrects clonal/phylogenetic structure and leaves
# the unitig fixed-effects free to find HGT-linked invasion features. pyseer intersects to the common
# samples (blood/faeces 13,171; resp/faeces 8,979) — same set as the variant GWAS, so the two are
# directly comparable.
UNITIG_MATRIX=${UNITIG_MATRIX:-$DATA/david/processed/pyseer_iso_source/unitigs/$PAIR/unitigs.pyseer.gz}
PHENO=$IN_DIR/phenotype.tsv
DIST=$IN_DIR/jaccard_distances.tsv
SAMPLES=$IN_DIR/samples.txt          # sample-id list (phenotype/Rtab order) — input to similarity_pyseer
SIMILARITY=$IN_DIR/similarity.tsv    # LMM kinship built from the Rtab; built once, shared across LMM runs
CLUSTERS=$GWAS_DIR/sublineage_clusters.tsv
PATTERNS=$GWAS_DIR/patterns.txt
ASSOC=$GWAS_DIR/${OUT_STEM}.assoc

echo "Job $SLURM_JOB_ID  Node $SLURMD_NODENAME  cohort=$COHORT  K=$K  features=$FEATURES  $(date)"

# Feature input: variants (--pres core-SNP Rtab) or unitigs (--kmers GGCAT matrix).
if [ "$FEATURES" = "unitigs" ]; then
    [ -s "$UNITIG_MATRIX" ] || { echo "ERROR: unitig matrix $UNITIG_MATRIX missing/empty"; exit 1; }
    FEATURE_ARGS=(--kmers "$UNITIG_MATRIX")
    echo "FEATURES=unitigs -> --kmers $UNITIG_MATRIX ($(ls -lh "$UNITIG_MATRIX" | awk '{print $5}'))"
else
    FEATURE_ARGS=(--pres "$RTAB")
fi

# 0) sublineage-clusters file (tab-sep: sample <TAB> Sublineage; NaN -> 'unknown'), for
#    pyseer --lineage (reports the lineage each hit is most associated with). Aligned to
#    the phenotype's samples. With MIN_SL_SIZE>0, Sublineages smaller than that collapse to
#    a single 'other' bucket — keeps the big SLs (SL258, SL147, ...) interpretable while
#    sparing pyseer the per-variant attribution over ~1300 tiny n=1-5 clusters.
if [ "$USE_LINEAGE" = "1" ]; then
uv run python - "$COHORT_CSV" "$PHENO" "$CLUSTERS" "$MIN_SL_SIZE" <<'PY'
import sys
import pandas as pd

cohort_csv, pheno_tsv, out, min_sl = sys.argv[1:5]
min_sl = int(min_sl)
samples = set(pd.read_csv(pheno_tsv, sep="\t")["samples"].astype(str))
meta = pd.read_csv(cohort_csv, usecols=["Sample", "Sublineage"], low_memory=False)
meta["Sample"] = meta["Sample"].astype(str)
meta = meta.drop_duplicates(subset=["Sample"])
meta = meta[meta["Sample"].isin(samples)]
meta["Sublineage"] = meta["Sublineage"].fillna("unknown").astype(str).replace({"": "unknown", "nan": "unknown"})
if min_sl > 0:
    counts = meta["Sublineage"].value_counts()
    big = counts.index[counts >= min_sl]
    meta["Sublineage"] = meta["Sublineage"].where(meta["Sublineage"].isin(big), "other")
    print(f"collapsed Sublineages with <{min_sl} samples into 'other'; kept {len(big)} big SLs")
meta[["Sample", "Sublineage"]].to_csv(out, sep="\t", header=False, index=False)
print(f"wrote {out}: {len(meta)} samples, {meta['Sublineage'].nunique()} clusters")
PY
else
  echo "USE_LINEAGE=0 -> no --lineage; skipping sublineage-clusters build"
fi

# 1) structure inputs (method-dependent).
if [ "$USE_LMM" = "1" ]; then
    # LMM: build the kinship/similarity matrix once from the variant Rtab (shared across LMM runs).
    #      similarity_pyseer wants a sample-id list (phenotype order, == Rtab columns).
    if [ ! -s "$SIMILARITY" ]; then
        # unitig runs must REUSE the core-SNP kinship (see UNITIG_MATRIX note); never build it from
        # the k-mers, which would absorb the HGT signal into the random effect.
        [ "$FEATURES" = "unitigs" ] && { echo "ERROR: FEATURES=unitigs needs the variant core-SNP kinship at $SIMILARITY — run the variant LMM first"; exit 1; }
        echo "building LMM similarity (kinship) from Rtab via similarity_pyseer  $(date)"
        tail -n +2 "$PHENO" | cut -f1 > "$SAMPLES"
        pixi run --manifest-path "$PIXI_MANIFEST" similarity_pyseer \
            --pres "$RTAB" --min-af "$MIN_AF" --max-af "$MAX_AF" "$SAMPLES" > "$SIMILARITY"
        echo "wrote $SIMILARITY ($(wc -l < "$SIMILARITY") rows)  $(date)"
    else
        echo "reusing existing similarity matrix $SIMILARITY (core-SNP kinship)"
    fi
else
    # fixed-effects: scree plot of the MDS eigenvalues to eyeball/justify K (non-fatal, MDS-only).
    ( cd "$GWAS_DIR" && pixi run --manifest-path "$PIXI_MANIFEST" \
        scree_plot_pyseer "$DIST" --max-dimensions 30 ) \
        || echo "WARN: scree_plot_pyseer failed (non-fatal) — continuing to pyseer"
fi

# 2) the GWAS itself. Structure correction is fixed-effects MDS (--distances + K) by default,
#    or LMM random effects (--lmm + kinship) when USE_LMM=1. --lineage is optional (USE_LINEAGE);
#    the af window + the SL-collapse threshold are the levers for the speed-vs-rare-variant tradeoff.
LINEAGE_ARGS=()
if [ "$USE_LINEAGE" = "1" ]; then
    LINEAGE_ARGS=(--lineage --lineage-clusters "$CLUSTERS")
fi
if [ "$USE_LMM" = "1" ]; then
    STRUCT_ARGS=(--lmm --similarity "$SIMILARITY" --save-lmm "$GWAS_DIR/lmm_cache")
    # --lmm corrects via the kinship, but pyseer still needs a distance matrix to compute the
    # --lineage effects report — so pass --distances too whenever lineage attribution is on.
    [ "$USE_LINEAGE" = "1" ] && STRUCT_ARGS+=(--distances "$DIST")
    echo "pyseer config: LMM (random effects)  min-af=$MIN_AF  max-af=$MAX_AF  use-lineage=$USE_LINEAGE  min-sl-size=$MIN_SL_SIZE"
else
    STRUCT_ARGS=(--distances "$DIST" --max-dimensions "$K" --save-m "$GWAS_DIR/mds_cache")
    echo "pyseer config: fixed-effects K=$K  min-af=$MIN_AF  max-af=$MAX_AF  use-lineage=$USE_LINEAGE  min-sl-size=$MIN_SL_SIZE"
fi
pixi run --manifest-path "$PIXI_MANIFEST" pyseer \
    "${FEATURE_ARGS[@]}" \
    --phenotypes "$PHENO" --phenotype-column "$LABEL_COL" \
    "${STRUCT_ARGS[@]}" \
    ${LINEAGE_ARGS[@]+"${LINEAGE_ARGS[@]}"} \
    --min-af "$MIN_AF" --max-af "$MAX_AF" \
    --output-patterns "$PATTERNS" \
    --cpu "$SLURM_CPUS_PER_TASK" \
    > "$ASSOC"

echo "pyseer done: $(wc -l < "$ASSOC") assoc lines  $(date)"

# 3) diagnostics + gene mapping (uv env). Figures land in GWAS_DIR; the small PNGs + hit
#    table are scp'd to the repo docs/figures from the laptop afterwards (commit from local).
# variant mode: annotate each hit's SNP consequence (synonymous/missense/LoF/noncoding) from the effect map
EMAP_ARG=()
[ "$FEATURES" = "variants" ] && [ -s "$EFFECT_MAP" ] && EMAP_ARG=(--effect-map "$EFFECT_MAP")
uv run python src/bac_pyseer/kleb_iso_source/pyseer_postprocess.py \
    --assoc "$ASSOC" --patterns "$PATTERNS" --gff "$GFF" \
    --feature-mode "$FEATURES" ${EMAP_ARG[@]+"${EMAP_ARG[@]}"} \
    --out-fig-dir "$GWAS_DIR" \
    --out-table "$GWAS_DIR/${OUT_STEM}_hits_annotated.tsv" \
    --summary-json "$GWAS_DIR/${OUT_STEM}_gwas_summary.json" \
    --contig NC_009648 --max-dimensions "$K" \
    --pos-label "$POS_LABEL" --neg-label "$NEG_LABEL" --pair-title "$PAIR_TITLE"

echo "GWAS outputs in $GWAS_DIR"
ls -lh "$GWAS_DIR"
echo "Done  $(date)"
