#!/usr/bin/env bash
# Per-drug DRIVER PANEL (one-hot | baclm [| ESM | Bacformer]) — the UNGATED per-determinant embedding LR
# (no 10% prevalence gate), so low-prevalence acquired determinants (CTX-M/KPC/AAC/MphA) each get a real LR
# bar showing what the model reads for them. Reads the per-drug driver CSVs (Kp: the fresh CARD determinant
# lists under the data-root card_ceiling/<drug>/; TB: the committed WHO tbprofiler CSVs), scores the coding
# drivers' baclm vectors vs the drug label, leaving non-coding / rRNA rows one-hot only. CPU-only (NO --gres).
#
#   TASK=kleb sbatch src/bacpredict/engine/scripts/plot_catalogue_vs_embeddings.sh   # Kp CARD, baclm-B, skip ESM
#   TASK=tb   sbatch src/bacpredict/engine/scripts/plot_catalogue_vs_embeddings.sh   # TB WHO
#   # optional: BACFORMER_NPZ=/path/to/panel_tokens.npz to fill the Bacformer column.
#SBATCH --job-name=driver_panel
#SBATCH --partition=icelake-himem
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --open-mode=append
set -uo pipefail
# Data root + env — cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david). Matches the other launchers.
: "${BACPREDICT_DATA_ROOT:="$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction"}"
D="$BACPREDICT_DATA_ROOT"
TASK="${TASK:-kleb}"
PY="$HOME/workspace/BacPredict/.venv/bin/python"
REPO="${REPO:-$HOME/workspace/BacPredict}"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

EXTRA=()
case "$TASK" in
  tb)   SPECIES=tb; CSVPREFIX=tbprofiler_gene_lr; CSVSUFFIX="";
        CSVDIR="$REPO/src/bacpredict/visualisations/tb" ;;
  kleb) SPECIES=kp; CSVPREFIX=card_determinant_lr; CSVSUFFIX="_family";
        CSVDIR="$D/processed/train_kleb_ast/card_ceiling";           # fresh per-<drug> CARD determinant lists
        # CARD sidecars locate acquired genes Bakta misses; baclm indexes the B-parquet, so score baclm-only
        # against protein_sequences_B (the A-ordered ESM store would be mis-indexed → --skip-esm).
        EXTRA=(--amr-sidecar-dir "$D/processed/train_kleb_ast/amr_annotation" \
               --parquet-dir "$D/processed/train_kleb_ast/protein_sequences_B" --skip-esm) ;;
  *) echo "unknown TASK=$TASK (want tb|kleb)"; exit 1 ;;
esac
OUT="$D/processed/train_${TASK}_ast/pangena_predict/driver_panel"
NPZ_ARG=(); [ -n "${BACFORMER_NPZ:-}" ] && NPZ_ARG=(--bacformer-npz "${BACFORMER_NPZ}")

echo "=== driver panel: species=$SPECIES csv=$CSVDIR/<drug> out=$OUT npz=${BACFORMER_NPZ:-none} ==="
"$PY" -m bacpredict.engine.plots.plot_catalogue_vs_embeddings \
  --species "$SPECIES" \
  --csv-dir "$CSVDIR" --csv-prefix "$CSVPREFIX" --csv-suffix "$CSVSUFFIX" \
  --n-folds 5 --seeds 1,2,3 --pool-workers "${SLURM_CPUS_PER_TASK:-16}" \
  "${EXTRA[@]}" "${NPZ_ARG[@]}" \
  --output "$OUT"
echo "driver panel -> $OUT"
