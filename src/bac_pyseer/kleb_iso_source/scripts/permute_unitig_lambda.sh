#!/bin/bash
#SBATCH --job-name=pyseer_unitig_permnull
#SBATCH --output=/home/dca36/rds/hpc-work/pyseer_scratch/unitig_permnull_%j.out
#SBATCH --error=/home/dca36/rds/hpc-work/pyseer_scratch/unitig_permnull_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=12:00:00
# cpu=8 + CHUNKED scan (step 3): a single pyseer --kmers process accumulates memory over the scan and OOM'd
# even at 200G (~184k unitigs in; cpu=32 OOM'd immediately). The production fix is to prime the LMM cache
# once and scan ~85k-unitig chunks as SEPARATE processes (~26 GB peak each, like the production shards), so
# peak memory is bounded by one chunk. 200G is then generous headroom.
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU

# Within-lineage permutation NULL for the UNITIG LMM — the DIRECT test of whether the unitig common-af
# inflation (real af-λ: 0.05–0.10 = 1.09 … 0.70–1.0 = 23.8) is residual population structure the core-SNP
# kinship cannot absorb, vs real signal. Shuffling case/control WITHIN each sublineage preserves the
# between-lineage structure but destroys within-lineage signal; re-running the SAME unitig LMM (core-SNP
# kinship) on the permuted phenotype and recomputing af-stratified λ isolates the STRUCTURE-only inflation
# per af bucket, giving a NON-ARBITRARY reliability ceiling:
#   λ_perm ≈ 1 in a bucket  ⇒ test calibrated there ⇒ unitig calls at that af are reliable;
#   λ_perm ≫ 1 in a bucket  ⇒ inflation is structure the kinship misses ⇒ unreliable.
# Highest af bucket with λ_perm ≈ 1 = the ceiling. The common-af buckets (0.50–1.0) double as a POSITIVE
# CONTROL: if λ_perm there reproduces the real λ≈7/24, that confirms the common-af inflation is structure
# (correctly disregarded); if it were ≈1 there, the common-af signal would be real.
#
# Mirrors permute_variant_lambda.sh but feeds --kmers (a stride-subsample of the 6.28M-unitig matrix; a
# representative ~420k is λ-sufficient and runs in one job) instead of --pres. pyseer's --save-lmm cache
# bakes in h^2, so we RE-FIT fresh (a NEW --save-lmm) — NOT --load-lmm the real h²=0.83 cache. Runs pyseer
# AND the python helpers via PIXI (numpy/scipy/pandas present; sidesteps the current uv.lock breakage).
#
# Usage:  SEED=1 sbatch permute_unitig_lambda.sh   (seed 1 reuses the variant run's permuted phenotype)
# Staged helpers (permute_phenotype_within_lineage.py, genomic_inflation_by_af.py) live in $LB on RDS.
set -euo pipefail
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
unset PYTHONPATH PYTHONHOME

REPO=/home/dca36/workspace/BacPredict
PIXI=$REPO/src/bac_pyseer/pixi.toml
LB=/home/dca36/rds/hpc-work/pyseer_scratch          # RDS scratch: staged helpers, SLURM logs, unitig subset (NOT home)
DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
IN=$DATA/david/processed/pyseer_iso_source/blood_faeces/sampled_country_2_1_all
CSV=$DATA/david/processed/train_iso_source/blood_faeces/sampled_country_2_1_all/kpsc_human/binary_blood_vs_faeces_with_split.csv
UNITIGS=$DATA/david/processed/pyseer_iso_source/unitigs/blood_faeces/unitigs.pyseer.gz
SIM=$IN/similarity.tsv
LABEL=blood_vs_faeces_label
SEED=${SEED:-1}
STRIDE=${STRIDE:-15}
OUT=$IN/gwas_lmm_permnull
mkdir -p "$OUT" "$LB"
PHENO=$IN/phenotype.tsv
CLUST=$OUT/sublineage_clusters_full.tsv
PERM=$OUT/phenotype_perm_seed${SEED}.tsv
SUB=$LB/unitig_subset_stride${STRIDE}.gz
ASSOC=$OUT/blood_vs_faeces_unitig_permnull_seed${SEED}.assoc
PYERR=$OUT/pyseer_unitig_permnull_seed${SEED}.err

echo "Job ${SLURM_JOB_ID:-?}  node ${SLURMD_NODENAME:-?}  seed=$SEED  stride=$STRIDE  $(date)"

# 0+1) within-lineage permuted phenotype — REUSE if present (seed 1 = the variant run's, same null
#      realization → directly comparable), else build fine clusters + permute (via pixi: pandas/numpy).
if [ -s "$PERM" ]; then
    echo "reusing existing permuted phenotype $PERM"
else
    if [ ! -s "$CLUST" ]; then
        pixi run --manifest-path "$PIXI" python - "$CSV" "$PHENO" "$CLUST" <<'PY'
import sys
import pandas as pd
csv, pheno, out = sys.argv[1:4]
samples = set(pd.read_csv(pheno, sep="\t")["samples"].astype(str))
meta = pd.read_csv(csv, usecols=["Sample", "Sublineage"], low_memory=False)
meta["Sample"] = meta["Sample"].astype(str)
meta = meta.drop_duplicates("Sample")
meta = meta[meta["Sample"].isin(samples)]
meta["Sublineage"] = meta["Sublineage"].fillna("unknown").astype(str).replace({"": "unknown", "nan": "unknown"})
meta[["Sample", "Sublineage"]].to_csv(out, sep="\t", header=False, index=False)
print(f"wrote {out}: {len(meta)} samples, {meta['Sublineage'].nunique()} clusters", file=sys.stderr)
PY
    fi
    pixi run --manifest-path "$PIXI" python "$LB/permute_phenotype_within_lineage.py" \
        --phenotype "$PHENO" --clusters "$CLUST" --label-col "$LABEL" --seed "$SEED" --out "$PERM"
fi

# 2) stride-subsample the 6.28M-unitig matrix -> ~420k representative unitigs (guard-reuse; the zcat of the
#    77 GB matrix is the slow step ~40 min). awk NR%STRIDE==1 is unbiased w.r.t. af (matrix is not af-sorted).
if [ -s "$SUB" ]; then
    echo "reusing existing unitig subset $SUB ($(zcat "$SUB" | wc -l) unitigs)"
else
    echo "building 1/$STRIDE stride subsample of $UNITIGS  $(date)"
    zcat "$UNITIGS" | awk -v s="$STRIDE" 'NR % s == 1' | gzip > "$SUB"
    echo "subset unitigs: $(zcat "$SUB" | wc -l)  $(date)"
fi

# 3) UNITIG LMM on the permuted phenotype — CHUNKED (mirrors the production sharded run). A single pyseer
#    --kmers process accumulates memory over the scan and OOM'd even at 200G (~184k unitigs in), so we prime
#    the kinship eigendecomposition + h^2 ONCE (--save-lmm, tiny kmer set), then scan each ~85k-unitig chunk
#    as a FRESH process with --load-lmm (memory freed per chunk; same permuted phenotype → n & h^2 consistent;
#    the combined per-unitig p-values equal a single full run). Chunk dir lives on RDS scratch, cleaned after.
CHUNKDIR=$LB/unitig_chunks_seed${SEED}
CACHE=$OUT/unitig_lmm_cache_perm_seed${SEED}.npz   # explicit .npz so --save-lmm/--load-lmm agree on the path
rm -rf "$CHUNKDIR"; mkdir -p "$CHUNKDIR"
: > "$PYERR"
# NB: pyseer --kmers REQUIRES gzipped input (it gzip-reads the file), so the prime set and every chunk must
# be gzipped — plain-text files crash with gzip.BadGzipFile.
# 3a) prime the LMM cache (decomposition + h^2) on a tiny GZIPPED kmer set — isolates the one-time cost.
#     guard the zcat|head SIGPIPE so it doesn't trip `set -o pipefail` when head closes zcat early.
set +o pipefail; zcat "$SUB" | head -300 | gzip > "$CHUNKDIR/prime.kmers.gz"; set -o pipefail
pixi run --manifest-path "$PIXI" pyseer --kmers "$CHUNKDIR/prime.kmers.gz" \
    --phenotypes "$PERM" --phenotype-column "$LABEL" \
    --lmm --similarity "$SIM" --save-lmm "$CACHE" \
    --min-af 0.01 --max-af 0.99 --cpu "$SLURM_CPUS_PER_TASK" > /dev/null 2>> "$PYERR"
echo "primed LMM cache  $(date)"; grep -iE "h\^2|found in both" "$PYERR" | head || true
# 3b) split the subset into ~85k-unitig chunks, gzip each (pyseer --kmers needs gzip), scan with --load-lmm.
zcat "$SUB" | split -l 85000 -d - "$CHUNKDIR/chunk_"
gzip "$CHUNKDIR"/chunk_*
echo "split subset into $(ls "$CHUNKDIR"/chunk_*.gz | wc -l) gzipped chunks  $(date)"
: > "$ASSOC"
first=1
for ck in "$CHUNKDIR"/chunk_*.gz; do
    echo "  pyseer chunk $(basename "$ck") ($(zcat "$ck" | wc -l) unitigs)  $(date)"
    pixi run --manifest-path "$PIXI" pyseer --kmers "$ck" \
        --phenotypes "$PERM" --phenotype-column "$LABEL" \
        --lmm --load-lmm "$CACHE" \
        --min-af 0.01 --max-af 0.99 --cpu "$SLURM_CPUS_PER_TASK" \
        > "$CHUNKDIR/out.tmp" 2>> "$PYERR"
    if [ "$first" = 1 ]; then cat "$CHUNKDIR/out.tmp" >> "$ASSOC"; first=0; else tail -n +2 "$CHUNKDIR/out.tmp" >> "$ASSOC"; fi
done
rm -rf "$CHUNKDIR"
echo "pyseer unitig permnull done: $(wc -l < "$ASSOC") assoc lines  $(date)"

# 4) af-stratified λ of the unitig permutation null — full af range:
#    rare end = calibration sanity, 0.10–0.50 = ceiling search, 0.50–1.0 = structure positive control.
pixi run --manifest-path "$PIXI" python "$LB/genomic_inflation_by_af.py" \
    --assoc "unitig_permnull=$ASSOC" \
    --bins 0.01,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.70,1.0 \
    --out "$OUT/unitig_permnull_af_lambda_seed${SEED}.tsv"
echo "UNITIG_PERMNULL_DONE  seed=$SEED  $(date)"
