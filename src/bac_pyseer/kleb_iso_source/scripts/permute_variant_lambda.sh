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

# Within-lineage permutation NULL for the variant LMM — the rigour backstop on the reported
# genomic-inflation λ (real blood/faeces variant LMM λ=0.562). Shuffling case/control WITHIN each
# sublineage preserves the phenotype↔lineage (between-lineage structure) correlation while destroying
# genuine within-lineage genotype↔phenotype signal. Re-running the SAME LMM (core-SNP kinship) on the
# permuted phenotype and recomputing λ isolates the cause of the real-run calibration:
#   λ_perm not inflated (≲1.1)  ⇒ the kinship adequately controls structure; the variant hits are NOT
#                                 residual-structure artifacts (the desired backstop).
#   λ_perm ≫ 1                  ⇒ residual between-lineage confounding the kinship fails to absorb.
# (Our variant λ is already <1, so we expect λ_perm ≤ ~1 — i.e. no uncontrolled structure.)
#
# pyseer's --save-lmm cache bakes in h^2  (verified in lmm.py: np.savez(lmm.U, lmm.S, np.array([h2]));
# load reads h2=data['arr_2'][0]), so we MUST re-fit fresh (a NEW --save-lmm) — NOT --load-lmm the real
# cache — so h^2 is re-estimated for the permuted phenotype. --lineage/--distances are omitted: λ uses
# only the LMM lrt-pvalue, which the (separate) lineage-attribution report does not affect.
#
# Usage:  SEED=1 sbatch permute_variant_lambda.sh      (run a few seeds for a stable null)
# Reads staged helper scripts from RDS scratch $LB (permute_phenotype_within_lineage.py,
# genomic_inflation_by_af.py) — NEVER home: /home is quota-limited and reserved for code, not data/logs.
# All bulk outputs go to project_k ($IN/gwas_lmm_permnull); SLURM logs + staged helpers + caches stay on RDS.
set -euo pipefail
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
unset PYTHONPATH PYTHONHOME

REPO=/home/dca36/workspace/BacPredict
PIXI=$REPO/src/bac_pyseer/pixi.toml
LB=/home/dca36/rds/hpc-work/pyseer_scratch   # RDS scratch for staged helpers + SLURM logs (NOT home)
DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw
IN=$DATA/david/processed/pyseer_iso_source/blood_faeces/sampled_country_2_1_all
CSV=$DATA/david/processed/train_iso_source/blood_faeces/sampled_country_2_1_all/kpsc_human/binary_blood_vs_faeces_with_split.csv
PHENO=$IN/phenotype.tsv
SIM=$IN/similarity.tsv
RTAB=$IN/variant_by_loci_presence.Rtab
LABEL=blood_vs_faeces_label
SEED=${SEED:-1}
OUT=$IN/gwas_lmm_permnull
mkdir -p "$OUT"
CLUST=$OUT/sublineage_clusters_full.tsv          # FINE resolution (no 'other' collapse) — strictest null
PERM=$OUT/phenotype_perm_seed${SEED}.tsv
ASSOC=$OUT/blood_vs_faeces_permnull_seed${SEED}.assoc
PYERR=$OUT/pyseer_permnull_seed${SEED}.err

echo "Job ${SLURM_JOB_ID:-?}  node ${SLURMD_NODENAME:-?}  seed=$SEED  $(date)"

# 0) FINE sublineage clusters (every SL its own group; no MIN_SL_SIZE collapse) aligned to phenotype.
uv run --project "$REPO" python - "$CSV" "$PHENO" "$CLUST" <<'PY'
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

# 1) within-lineage permuted phenotype (per-cluster case count preserved).
uv run --project "$REPO" python "$LB/permute_phenotype_within_lineage.py" \
    --phenotype "$PHENO" --clusters "$CLUST" --label-col "$LABEL" --seed "$SEED" --out "$PERM"

# 2) the SAME variant LMM on the permuted phenotype, FRESH fit (NEW --save-lmm; NOT --load-lmm).
pixi run --manifest-path "$PIXI" pyseer \
    --pres "$RTAB" \
    --phenotypes "$PERM" --phenotype-column "$LABEL" \
    --lmm --similarity "$SIM" --save-lmm "$OUT/lmm_cache_perm_seed${SEED}" \
    --min-af 0.01 --max-af 0.99 \
    --output-patterns "$OUT/patterns_perm_seed${SEED}.txt" \
    --cpu "$SLURM_CPUS_PER_TASK" \
    > "$ASSOC" 2> "$PYERR"
echo "pyseer permnull done: $(wc -l < "$ASSOC") assoc lines  $(date)"
grep -iE "h\^2|found in both|patterns" "$PYERR" | head || true

# 3) genomic-inflation λ of the permutation null (overall + af bins) vs the real run's 0.562.
uv run --project "$REPO" python "$LB/genomic_inflation_by_af.py" \
    --assoc "variant_permnull=$ASSOC" \
    --bins 0.01,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.70,1.0 \
    --out "$OUT/permnull_af_lambda_seed${SEED}.tsv"
echo "PERMNULL_DONE  seed=$SEED  $(date)"
