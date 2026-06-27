#!/bin/bash
# Test whether a MASH (whole-genome k-mer Jaccard) kinship — which captures the deep soft-core /
# cross-species + accessory structure a single-reference core-SNP kinship misses — fixes the common-af
# genomic-inflation in the unitig GWAS (core-SNP kinship gave af>0.5 λ=21).
#
# Chain (one icelake-himem job; mash triangle already built in $S=~/rds/hpc-work/mash_kinship on RDS):
#   1. mash triangle -> pyseer similarity (mash_dist_to_kinship.py)
#   2. stride-subsample ~419k unitigs (representative across af) from the 6.28M matrix
#   3. prime an LMM cache from the MASH kinship (--save-lmm) — reports the new h^2
#   4. run pyseer --lmm --load-lmm on the subset -> mash_subset.assoc
# Then (locally) genomic_inflation_by_af.py on mash_subset.assoc, compared to the core-SNP-kinship λ.
#
# Submitted by hand with CLI sbatch overrides (icelake-himem, 38c/128G/8h, project_k). Run from RDS
# scratch ($S=~/rds/hpc-work/mash_kinship) — never home (/home is quota-limited, code-only).
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
unset PYTHONPATH PYTHONHOME
REPO=/home/dca36/workspace/BacPredict
PIXI=$REPO/src/bac_pyseer/pixi.toml
P=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/pyseer_iso_source
CO=$P/blood_faeces/sampled_country_2_1_all
M=$P/unitigs/blood_faeces/unitigs.pyseer.gz
PHENO=$CO/phenotype.tsv
S=/home/dca36/rds/hpc-work/mash_kinship   # RDS scratch (home is too small for the ~5 GB subset)
cd "$S"

echo "=== 0. phenotype restricted+ordered to the mash-kinship samples  $(date) ==="
# The mash kinship covers only the 13,171 assembly (SAMN) samples; the full phenotype has 13,602
# (incl. 431 ncbi/GCF not sketched). pyseer --load-lmm checks phenotype length == cache, so prime and
# run must both use a phenotype trimmed to (and ordered like) the kinship's samples.
if [ ! -s "$S/pheno_mash.tsv" ]; then
    uv run --project "$REPO" python - "$S/mash_kinship.tsv" "$PHENO" "$S/pheno_mash.tsv" <<'PY'
import sys, pandas as pd
kin, pheno, out = sys.argv[1:4]
samples = pd.read_csv(kin, sep="\t", index_col=0, nrows=0).columns.tolist()  # kinship sample order
ph = pd.read_csv(pheno, sep="\t"); sc, lc = ph.columns[0], ph.columns[1]
ph[sc] = ph[sc].astype(str); lab = dict(zip(ph[sc], ph[lc]))
keep = [s for s in samples if s in lab]
pd.DataFrame({"samples": keep, lc: [int(lab[s]) for s in keep]}).to_csv(out, sep="\t", index=False)
print(f"wrote {out}: {len(keep)}/{len(samples)} kinship samples carry a phenotype", file=sys.stderr)
PY
    rm -f "$S/mash_cache.npz"   # stale cache (primed against the full phenotype) — force a clean re-prime
fi
PHENO="$S/pheno_mash.tsv"   # prime + run both use the SAME, cache-length-matched phenotype

echo "=== 1. mash triangle -> kinship  $(date) ==="
if [ -s "$S/mash_kinship.tsv" ]; then
    echo "reusing existing kinship $S/mash_kinship.tsv"
else
    uv run --project "$REPO" python "$S/mash_dist_to_kinship.py" --triangle "$S/mash_triangle.txt" --out "$S/mash_kinship.tsv"
fi

echo "=== 2. stride-subsample (~1/15) of the unitig matrix  $(date) ==="
if [ -s "$S/usub.gz" ]; then
    echo "reusing existing subset $S/usub.gz"
else
    zcat "$M" | awk 'NR%15==1' | gzip > "$S/usub.gz"
fi
echo "subset unitigs: $(zcat "$S/usub.gz" | wc -l)"

echo "=== 3. prime LMM cache from the MASH kinship  $(date) ==="
if [ -s "$S/mash_cache.npz" ]; then
    echo "reusing existing cache $S/mash_cache.npz"
else
    # tiny priming kmer set (cache depends on samples+kinship+phenotype, not the kmers). The
    # zcat|head pipe must not trip `set -o pipefail` when head closes zcat early (SIGPIPE).
    [ -s "$S/uprime.gz" ] || { set +o pipefail; zcat "$S/usub.gz" | head -300 | gzip > "$S/uprime.gz"; set -o pipefail; }
    pixi run --manifest-path "$PIXI" pyseer --kmers "$S/uprime.gz" --phenotypes "$PHENO" \
        --phenotype-column blood_vs_faeces_label --lmm --similarity "$S/mash_kinship.tsv" \
        --save-lmm "$S/mash_cache.npz" --min-af 0.01 --max-af 0.99 > "$S/prime.out" 2> "$S/prime.err"
    grep -iE "h\^2|found in both" "$S/prime.err" "$S/prime.out" | head
fi

echo "=== 4. LMM on subset with the MASH cache  $(date) ==="
pixi run --manifest-path "$PIXI" pyseer --kmers "$S/usub.gz" --phenotypes "$PHENO" \
    --phenotype-column blood_vs_faeces_label --lmm --load-lmm "$S/mash_cache.npz" \
    --cpu 8 --min-af 0.01 --max-af 0.99 > "$S/mash_subset.assoc" 2> "$S/run.err"
echo "MASHLMM_DONE  assoc lines: $(wc -l < "$S/mash_subset.assoc")  $(date)"
grep -iE "h\^2|found in both" "$S/run.err" | head
