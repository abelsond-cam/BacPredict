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
COHORT=sampled_country_2_1_all
COHORT_CSV=$DATA/david/processed/train_iso_source/blood_faeces/$COHORT/kpsc_human/binary_blood_vs_faeces_with_split.csv
IN_DIR=$DATA/david/processed/pyseer_iso_source/blood_faeces/$COHORT
GWAS_SUBDIR=${2:-gwas}   # 2nd arg = output subdir; use a fresh name to run an isolated GWAS alongside another in flight
MIN_SL_SIZE=${3:-0}      # 3rd arg = min samples/Sublineage to keep as its own --lineage cluster; smaller SLs collapse to 'other' (0 = keep all)
GWAS_DIR=$IN_DIR/$GWAS_SUBDIR
GFF=$DATA/david/raw/related_lr/gff/GCF_000016305.1.gff
mkdir -p "$GWAS_DIR"

RTAB=$IN_DIR/variant_by_loci_presence.Rtab
PHENO=$IN_DIR/phenotype.tsv
DIST=$IN_DIR/jaccard_distances.tsv
CLUSTERS=$GWAS_DIR/sublineage_clusters.tsv
PATTERNS=$GWAS_DIR/patterns.txt
ASSOC=$GWAS_DIR/blood_vs_faeces.assoc

echo "Job $SLURM_JOB_ID  Node $SLURMD_NODENAME  cohort=$COHORT  K=$K  $(date)"

# 0) sublineage-clusters file (tab-sep: sample <TAB> Sublineage; NaN -> 'unknown'), for
#    pyseer --lineage (reports the lineage each hit is most associated with). Aligned to
#    the phenotype's samples. With MIN_SL_SIZE>0, Sublineages smaller than that collapse to
#    a single 'other' bucket — keeps the big SLs (SL258, SL147, ...) interpretable while
#    sparing pyseer the per-variant attribution over ~1300 tiny n=1-5 clusters.
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

# 1) scree plot of the MDS eigenvalues — eyeball the elbow to pick/justify K (output to GWAS_DIR).
#    Non-fatal: it is purely informational (K is validated empirically by lambda below), so a
#    plotting hiccup must never abort the GWAS. MPLBACKEND=Agg (above) keeps it headless-safe.
( cd "$GWAS_DIR" && pixi run --manifest-path "$PIXI_MANIFEST" \
    scree_plot_pyseer "$DIST" --max-dimensions 30 ) \
    || echo "WARN: scree_plot_pyseer failed (non-fatal) — continuing to pyseer"

# 2) the GWAS itself (fixed-effects + K MDS covariates).
pixi run --manifest-path "$PIXI_MANIFEST" pyseer \
    --pres "$RTAB" \
    --phenotypes "$PHENO" --phenotype-column blood_vs_faeces_label \
    --distances "$DIST" --max-dimensions "$K" \
    --lineage --lineage-clusters "$CLUSTERS" \
    --min-af 0.01 --max-af 0.99 \
    --output-patterns "$PATTERNS" \
    --save-m "$GWAS_DIR/mds_cache" \
    --cpu "$SLURM_CPUS_PER_TASK" \
    > "$ASSOC"

echo "pyseer done: $(wc -l < "$ASSOC") assoc lines  $(date)"

# 3) diagnostics + gene mapping (uv env). Figures land in GWAS_DIR; the small PNGs + hit
#    table are scp'd to the repo docs/figures from the laptop afterwards (commit from local).
uv run python src/bac_pyseer/kleb_iso_source/pyseer_postprocess.py \
    --assoc "$ASSOC" --patterns "$PATTERNS" --gff "$GFF" \
    --out-fig-dir "$GWAS_DIR" \
    --out-table "$GWAS_DIR/blood_vs_faeces_hits_annotated.tsv" \
    --summary-json "$GWAS_DIR/blood_vs_faeces_gwas_summary.json" \
    --contig NC_009648 --max-dimensions "$K"

echo "GWAS outputs in $GWAS_DIR"
ls -lh "$GWAS_DIR"
echo "Done  $(date)"
