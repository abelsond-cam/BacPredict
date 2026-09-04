#!/bin/bash
#SBATCH --job-name=unitig_presence_model
#SBATCH --output=/rds/user/dca36/hpc-work/logs/unitig_presence_model_%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/unitig_presence_model_%j.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU

# Build the genome x hit-unitig presence matrix and fit the invasion comparator model, then
# re-score Bacformer on the identical genome subset for a like-for-like head-to-head.
#
# Resources. The build parses the 1.66 GB cached hits_submatrix.tsv (33,039 unitigs x 13,171
# carriers, ~10^8 non-zeros): CSC indices int32 + data float32 is ~0.9 GB, and the one CSC->CSR
# conversion transiently doubles it, so ~5 GB peak with headroom for the six L2 fits over a
# sparse 13.6k x 33k design. 96G is deliberately generous because this is a >2h job where an OOM
# near the end wastes the whole run and another queue wait. 16 cores for the BLAS in lbfgs.
#
# Usage:
#   sbatch src/bac_pyseer/kleb_iso_source/scripts/run_unitig_presence_model.sh
#   ALSO_L1=1 sbatch .../run_unitig_presence_model.sh            # + the L1 locus shortlist
#   # the honest re-run: hit set selected by an LMM that never saw a holdout genome. Note the
#   # submatrix must still cover the WHOLE cohort (the holdout genomes need presence values to be
#   # scored) — only the unitig SELECTION is restricted, never the rows.
#   SELECTION_SCOPE=trainval_only SCORE_ALL_SPLITS=1 HITS_SUBMATRIX=<trainval hit submatrix> sbatch ...
#   # + one-hot sublineage, as a FLOOR on what lineage information adds (separate OUT_DIR, enforced)
#   WITH_SUBLINEAGE=1 SELECTION_SCOPE=trainval_only SCORE_ALL_SPLITS=1 HITS_SUBMATRIX=<...> sbatch ...
#   # the k-fold sweep's one unitig model: selection AND fit on the invariant train+validate 80%,
#   # so C must be cross-validated inside it rather than tuned on validate.
#   C_SELECTION=inner-cv SELECTION_SCOPE=trainval_only SCORE_ALL_SPLITS=1 COHORT=..._kfold_trainval sbatch ...

set -euo pipefail
export PYTHONUNBUFFERED=1
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/home/dca36/rds/hpc-work/.uv_cache
# Clear PYTHONPATH/HOME so a stray spack/module leak cannot shadow uv's numpy (see cluster_uohpc.md).
unset PYTHONPATH PYTHONHOME
# Keep BLAS single-threaded per process; sklearn's own threading handles the parallelism.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
# Overridable so this can run from a detached worktree when the shared checkout is mid-edit by
# another agent (pulling it would change files under a live job).
cd "${REPO_DIR:-/home/dca36/workspace/BacPredict}"

DATA=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david
PAIR=${PAIR:-blood_faeces}
COHORT=${COHORT:-sampled_country_2_1_all}
LABEL_COL=${LABEL_COL:-blood_vs_faeces_label}

PYSEER_COHORT=$DATA/processed/pyseer_iso_source/$PAIR/$COHORT
GWAS_DIR=$PYSEER_COHORT/gwas_unitig_lmm
FT_COHORT=$DATA/processed/train_iso_source/$PAIR/$COHORT/kpsc_human

HITS_SUBMATRIX=${HITS_SUBMATRIX:-$GWAS_DIR/mge_mapping/hits_submatrix.tsv}
# The GWAS phenotype file is the full modelled sample universe. Passing it is what keeps genomes
# that carry NONE of the hit unitigs in the matrix as all-zero rows — dropping them would quietly
# remove exactly the genomes the model should be calling faecal, and bias the comparison.
SAMPLE_UNIVERSE=${SAMPLE_UNIVERSE:-$PYSEER_COHORT/phenotype.tsv}
MATRIX_DIR=${MATRIX_DIR:-$GWAS_DIR/presence_matrix}
SPLIT_CSV=${SPLIT_CSV:-$FT_COHORT/binary_${LABEL_COL%_label}_with_split.csv}
BAC_SCORES=${BAC_SCORES:-$FT_COHORT/models/eval_scores.npz}
BAC_CKPT=${BAC_CKPT:-$FT_COHORT/models}
SELECTION_SCOPE=${SELECTION_SCOPE:-full_cohort}
WITH_SUBLINEAGE=${WITH_SUBLINEAGE:-0}
# How C is chosen. `validate` (default) tunes on the validate split and fits on train — the protocol
# behind the deployed 0.7655, kept as the default so that number stays reproducible. `inner-cv` fits
# on train+validate and cross-validates C inside it, which is what the k-fold sweep needs: there,
# selection ALSO used train+validate, so tuning on validate would tune on rows that helped choose the
# features. C is worth ~5 pp (validate 0.775 at C=0.01 down to 0.728 at C=10), so it cannot just be
# pinned instead.
C_SELECTION=${C_SELECTION:-validate}

# The sublineage block makes this a DIFFERENT model, so it defaults to a different output dir.
# presence_model/unitig_cohort_scores.npz is the artifact the lab-collection comparison is gated on;
# writing a sublineage model over it would replace a number of record with a silently different one.
if [ "$WITH_SUBLINEAGE" = "1" ]; then
  OUT_DIR=${OUT_DIR:-$GWAS_DIR/presence_model_sublineage}
else
  OUT_DIR=${OUT_DIR:-$GWAS_DIR/presence_model}
fi

echo "=== inputs ==="
for f in "$HITS_SUBMATRIX" "$SAMPLE_UNIVERSE" "$SPLIT_CSV" "$BAC_SCORES"; do
  [ -f "$f" ] || { echo "MISSING: $f"; exit 1; }
  echo "  ok  $f"
done

if [ "$WITH_SUBLINEAGE" = "1" ]; then
  # Refuse to overwrite the plain model even if OUT_DIR was passed explicitly.
  if [ "$(basename "$OUT_DIR")" = "presence_model" ]; then
    echo "REFUSING: --with-sublineage writing into $OUT_DIR would overwrite the plain unitig model" >&2
    echo "  and its gated unitig_cohort_scores.npz. Pick a different OUT_DIR." >&2
    exit 1
  fi
  # Fail here, not 24 minutes into the fit, if the split CSV cannot supply the block.
  # Bash string match, NOT `... | grep -qx`: under `set -o pipefail` grep -q exits on first match,
  # tr then dies of SIGPIPE (141), and pipefail promotes that to the pipeline status — so the grep
  # form fires exactly when the column IS present. Same family as the `[ … ] &&` note below.
  HEADER=",$(head -1 "$SPLIT_CSV" | tr -d '\r'),"
  if [[ "$HEADER" != *",Sublineage,"* ]]; then
    echo "MISSING: WITH_SUBLINEAGE=1 but $SPLIT_CSV has no 'Sublineage' column" >&2
    exit 1
  fi
  echo "  ok  Sublineage column present; writing the sublineage model to $OUT_DIR"
fi

MOD=bac_pyseer.kleb_iso_source.unitig_presence_model

# Build is cached by its output dir; skip when the matrix is already there.
if [ -f "$MATRIX_DIR/X.npz" ]; then
  echo "=== reusing cached presence matrix at $MATRIX_DIR ==="
else
  echo "=== building presence matrix ==="
  uv run python -m $MOD build \
    --submatrix "$HITS_SUBMATRIX" \
    --sample-universe "$SAMPLE_UNIVERSE" \
    --matrix-dir "$MATRIX_DIR"
fi

echo "=== fitting (C by $C_SELECTION) + head-to-head vs Bacformer ==="
EXTRA=()
# `if`, not `[ … ] && …`. Measured, because the folk rule is wrong in both directions: a false
# `[ … ] && cmd` does NOT abort here — set -e exempts non-final commands of an AND-OR list, so
# mid-script at top level it is safe. It IS fatal in two places: as a script's last command (the
# non-zero status becomes the exit status, and SLURM reports FAILED) and as a function's last
# command (the function returns 1 and set -e kills the caller). `if` is immune to both, so it costs
# nothing to be the form that survives someone later moving the line.
if [ "${ALSO_L1:-0}" = "1" ]; then EXTRA+=(--also-l1); fi
# SCORE_ALL_SPLITS writes unitig_cohort_scores.npz over every genome in the matrix (with its split
# label), not just the holdout — the artifact that lets this model be compared genome-for-genome
# against Bacformer's cohort_scores.npz, and the input to the agreement scatter.
if [ "${SCORE_ALL_SPLITS:-0}" = "1" ]; then EXTRA+=(--score-all-splits); fi
# WITH_SUBLINEAGE stacks one-hot Sublineage columns onto the unitig design. It measures a FLOOR on
# what lineage adds — L2 penalises those columns too, so their coefficients are shrunk and the lift
# is a lower bound. It does NOT decompose "what unitigs add beyond lineage".
if [ "$WITH_SUBLINEAGE" = "1" ]; then EXTRA+=(--with-sublineage); fi

uv run python -m $MOD fit \
  --matrix-dir "$MATRIX_DIR" \
  --split-csv "$SPLIT_CSV" \
  --label-column "$LABEL_COL" \
  --bacformer-scores "$BAC_SCORES" \
  --bacformer-checkpoint-dir "$BAC_CKPT" \
  --selection-scope "$SELECTION_SCOPE" \
  --c-selection "$C_SELECTION" \
  --out-dir "$OUT_DIR" \
  "${EXTRA[@]}"

echo "=== done — results in $OUT_DIR ==="
