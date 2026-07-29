#!/bin/bash
#SBATCH --job-name=pyseer_permnull_blood_faeces
#SBATCH --output=/home/dca36/rds/hpc-work/pyseer_scratch/permnull_blood_faeces_%j.out
#SBATCH --error=/home/dca36/rds/hpc-work/pyseer_scratch/permnull_blood_faeces_%j.err
#SBATCH --partition=icelake-himem  # himem: less oversubscribed; ~6.7 GB/core fits 128G on 32 cores
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU

# Within-lineage permutation NULL for the VARIANT GWAS — the rigour backstop on the reported
# genomic-inflation λ. Shuffling case/control WITHIN each lineage cluster preserves the
# phenotype↔lineage (between-lineage structure) correlation while destroying genuine within-lineage
# genotype↔phenotype signal; re-running the SAME correction model on the permuted phenotype and
# recomputing λ isolates the cause of the real-run calibration:
#   λ_perm not inflated (≲1.1)  ⇒ the correction adequately controls structure; the hits are NOT
#                                 residual-structure artifacts (the desired backstop).
#   λ_perm ≫ 1                  ⇒ residual between-lineage confounding the correction fails to absorb.
#
# This wrapper spans the 2×2 matrix via two env toggles (suffix _${MODEL}_${LEVEL} on every output
# so no cell clobbers another; default = the original SL-LMM behaviour):
#   MODEL=lmm|mds   lmm  = FaST-LMM random effects on the core-SNP kinship (real λ=0.562);
#                   mds  = fixed-effects with K=10 MDS axes of the Jaccard distances (real λ=4.34).
#                   ⚠ For MDS, --distances IS the correction — it must NEVER be omitted (unlike the
#                     LMM null, where distances are only needed for the separate --lineage report).
#   LEVEL=sl|cg     sl = Sublineage (finest, strictest null);  cg = "Clonal group" (coarser).
#                   The permuted phenotype depends on LEVEL only, so LMM & MDS at a level share the
#                   SAME null realization and are directly comparable.
#
# Fresh refit EVERY run — a NEW --save-lmm / --save-m (NEVER --load-lmm / --load-m): pyseer's cache
# bakes in h^2 (LMM) / the MDS axes, which must be re-estimated for the permuted phenotype.
#
# Usage:  MODEL=mds LEVEL=cg SEED=1 sbatch permute_variant_lambda.sh   (run a few seeds for a stable null)
# Reads the cluster builder from the repo checkout ($REPO) and the staged helpers from RDS scratch
# ($LB: permute_phenotype_within_lineage.py, genomic_inflation_by_af.py) — NEVER home (quota-limited,
# code only). All bulk outputs go to project_k ($IN/gwas_lmm_permnull); logs + caches stay on RDS.
set -euo pipefail
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
unset PYTHONPATH PYTHONHOME

REPO=/home/dca36/workspace/BacPredict
PIXI=$REPO/src/bac_pyseer/pixi.toml
BUILDER=$REPO/src/bac_pyseer/kleb_iso_source/build_lineage_clusters.py
LB=/home/dca36/rds/hpc-work/pyseer_scratch   # RDS scratch for staged helpers + SLURM logs (NOT home)
DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
IN=$DATA/david/processed/pyseer_iso_source/blood_faeces/sampled_country_2_1_all
CSV=$DATA/david/processed/train_iso_source/blood_faeces/sampled_country_2_1_all/kpsc_human/binary_blood_vs_faeces_with_split.csv
METADATA=$DATA/david/final/metadata_v2_all_samples_and_columns.tsv   # CG "Clonal group" join fallback
PHENO=$IN/phenotype.tsv
SIM=$IN/similarity.tsv
DIST=$IN/jaccard_distances.tsv
RTAB=$IN/variant_by_loci_presence.Rtab
LABEL=blood_vs_faeces_label

MODEL=${MODEL:-lmm}     # lmm | mds
LEVEL=${LEVEL:-sl}      # sl  | cg
SEED=${SEED:-1}
MAX_DIM=${MAX_DIM:-10}  # MDS axes (K); K=10 matches the real MDS run (λ=4.34)
case "$MODEL" in lmm|mds) ;; *) echo "ERROR: MODEL must be lmm|mds, got '$MODEL'"; exit 2;; esac
case "$LEVEL" in sl|cg)   ;; *) echo "ERROR: LEVEL must be sl|cg, got '$LEVEL'";   exit 2;; esac

OUT=$IN/gwas_lmm_permnull
mkdir -p "$OUT"
if [ "$LEVEL" = "cg" ]; then
    CLUST=$OUT/clonal_group_clusters_full.tsv;  COLUMN="Clonal group"
else
    CLUST=$OUT/sublineage_clusters_full.tsv;     COLUMN="Sublineage"
fi
PERM=$OUT/phenotype_perm_${LEVEL}_seed${SEED}.tsv           # depends on LEVEL only (shared by lmm/mds)
STEM=${MODEL}_${LEVEL}_seed${SEED}
ASSOC=$OUT/blood_vs_faeces_permnull_${STEM}.assoc
PYERR=$OUT/pyseer_permnull_${STEM}.err
PATTERNS=$OUT/patterns_perm_${STEM}.txt
LAMBDA=$OUT/permnull_af_lambda_${STEM}.tsv

echo "Job ${SLURM_JOB_ID:-?}  node ${SLURMD_NODENAME:-?}  model=$MODEL  level=$LEVEL  seed=$SEED  $(date)"

# 0) lineage clusters at the chosen LEVEL (finest resolution, no collapse — the strictest null),
#    aligned to the phenotype; guard-reuse if already built for this level.
if [ -s "$CLUST" ]; then
    echo "reusing existing cluster file $CLUST ($(wc -l < "$CLUST") samples)"
else
    uv run --project "$REPO" python "$BUILDER" \
        --split-csv "$CSV" --phenotype "$PHENO" --column "$COLUMN" --metadata "$METADATA" --out "$CLUST"
fi

# 1) within-lineage permuted phenotype (per-cluster case count preserved). Reuse across models at
#    this level so LMM and MDS score the same null realization.
if [ -s "$PERM" ]; then
    echo "reusing existing permuted phenotype $PERM (LEVEL=$LEVEL, SEED=$SEED)"
else
    uv run --project "$REPO" python "$LB/permute_phenotype_within_lineage.py" \
        --phenotype "$PHENO" --clusters "$CLUST" --label-col "$LABEL" --seed "$SEED" --out "$PERM"
fi

# 2) the SAME variant model on the permuted phenotype, FRESH fit (NEW --save-lmm/--save-m).
if [ "$MODEL" = "lmm" ]; then
    STRUCT_ARGS=(--lmm --similarity "$SIM" --save-lmm "$OUT/lmm_cache_perm_${STEM}")
else
    # MDS: --distances IS the correction (never omit) + K axes; --save-m re-estimated for this null.
    STRUCT_ARGS=(--distances "$DIST" --max-dimensions "$MAX_DIM" --save-m "$OUT/mds_cache_perm_${STEM}")
fi
pixi run --manifest-path "$PIXI" pyseer \
    --pres "$RTAB" \
    --phenotypes "$PERM" --phenotype-column "$LABEL" \
    "${STRUCT_ARGS[@]}" \
    --min-af 0.01 --max-af 0.99 \
    --output-patterns "$PATTERNS" \
    --cpu "$SLURM_CPUS_PER_TASK" \
    > "$ASSOC" 2> "$PYERR"
echo "pyseer permnull done ($MODEL/$LEVEL): $(wc -l < "$ASSOC") assoc lines  $(date)"
grep -iE "h\^2|found in both|patterns" "$PYERR" | head || true

# 3) genomic-inflation λ of the permutation null (overall + af bins) vs the real run
#    (LMM 0.562 / MDS 4.34). lrt-pvalue is present in both LMM and fixed-effects pyseer output.
uv run --project "$REPO" python "$LB/genomic_inflation_by_af.py" \
    --assoc "variant_permnull_${MODEL}_${LEVEL}=$ASSOC" \
    --bins 0.01,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.70,1.0 \
    --out "$LAMBDA"
echo "PERMNULL_DONE  model=$MODEL  level=$LEVEL  seed=$SEED  $(date)"
