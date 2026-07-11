#!/bin/bash
# Kp concat probe — ESM-C top-gene vector (auto-picked from the per-gene ranking) (+) Bacformer
# genome-mean -> logistic regression, scored on the canonical eval fold + a k-fold x m-seed harness.
#
# The Kp port of src/bacpredict/engine/scripts/run_concat_kfold_frozen.sh (same module,
# pangena_predict.concatenate_bacformer_genome_esm_protein_emb). CPU-only: the Bacformer genome-mean is
# loaded from the cached frozen NPZ (bacformer_frozen_genome_mean.npz, 6838 x 960) via --bacformer-vectors,
# and --gene-from-ranking reads the top out-of-fold-AUROC gene from each drug's per_gene_lr_<drug>.csv.
# One array task per drug, the same four as the ranking. Writes concat_frozen_<drug>_<jobid>.json (with a
# "kfold" block of per-frame mean +/- sd + paired deltas) — the substrate for the ladder.
#
# Prereqs: the per-gene ranking job (build_per_gene_lr_ranking.sh) must have written per_gene_lr_<drug>.csv,
# and the genome-mean NPZ must exist. Usage:  sbatch src/kleb_ast/scripts/run_concat_kleb.sh
#
#SBATCH --job-name=kleb_concat
#SBATCH --output=kleb_concat_%A_%a.out
#SBATCH --error=kleb_concat_%A_%a.err
#SBATCH --array=0-21
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --open-mode=append
# CPU-only (LR over precomputed vectors; the Bacformer mean is loaded from the NPZ, not recomputed). The
# only real I/O is the pooled ESM-C gene reads over the cohort (--pool-workers) — a sub-hour job. 16 cores
# / 96 GB is ample; 4 h is a generous ceiling. Uses the project_k SL2-CPU account (personal FLOTO-SL2-CPU
# is nearly exhausted; project_k has ample budget).

cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# UNSUPERVISED gene selection: --gene-from-ranking picks the single highest-out-of-fold-AUROC gene from
# the per-gene ranking (whatever it is — a lineage marker, a chromosomal SNP gene, or an acquired gene).
# We do NOT pre-choose the canonical gene: the ESM/hotspot plots test whether the unsupervised top gene
# IS canonical, and the ladder tests using that top gene's embedding. The injected gene's prevalence may
# be low (e.g. a lineage marker present in a minority); the ladder fades the ESM-gene bar by prevalence to
# flag it. Source ranking = the zero-imputed ranking (the corrected read-out). Next step: inject the top-k
# genes (e.g. all with AUROC > 0.6), not just one.
DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
       gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
       levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then
    echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

D=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed
SHEET=$D/train_kleb_ast/binary_ast_with_split.csv
PARQUET=$D/klebsiella_protein_sequences
EMB=$D/klebsiella_esm_embeddings
NPZ=$D/train_kleb_ast/bacformer_frozen_genome_mean.npz
RANK=$D/train_kleb_ast/pangena_predict/per_gene_lr_ranking_imputed/$DRUG/per_gene_lr_${DRUG}.csv
OUT=$D/train_kleb_ast/pangena_predict/concat/$DRUG
mkdir -p "$OUT"

if [[ ! -f "$RANK" ]]; then
    echo "ERROR: ranking CSV missing: $RANK (run build_per_gene_lr_ranking.sh first)" >&2
    exit 1
fi

echo "========================================================================"
echo "Kp concat probe — drug=$DRUG (array task $SLURM_ARRAY_TASK_ID)"
echo "Ranking: $RANK   NPZ: $NPZ"
echo "Out:     $OUT/concat_frozen_${DRUG}_${SLURM_ARRAY_JOB_ID}.json"
echo "========================================================================"

uv run python src/bacpredict/engine/concat/concatenate_bacformer_genome_esm_protein_emb.py \
    --ast-sheet-path "$SHEET" \
    --parquet-dir "$PARQUET" \
    --esm-store-dir "$EMB" \
    --output-json "$OUT/concat_frozen_${DRUG}_${SLURM_ARRAY_JOB_ID}.json" \
    --drug "$DRUG" \
    --gene-from-ranking "$RANK" \
    --bacformer-vectors "$NPZ" \
    --qc-log "$OUT/gene_presence_qc_${DRUG}.log" \
    --pool-workers "${SLURM_CPUS_PER_TASK:-16}" \
    --kfold 5 --seeds 1 2 3 --evaluate-seed 1 --evaluate-fraction 0.20

echo "Kp concat probe ($DRUG) finished — JSON in $OUT"
