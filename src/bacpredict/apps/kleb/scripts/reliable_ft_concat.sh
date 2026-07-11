#!/bin/bash
# Phase 2b (CPU): reliable-label ESM-vs-FT per-gene head-to-head + FT-mean ⊕ best-gene concat, one drug
# per array task. Reads the Phase-2b FT token cache (ft_amr_cache/<drug>/) + the ESM store; no forward
# pass. Run AFTER the matching cache_ft_amr_proteins GPU task for the drug has finished.
#
#     sbatch --array=11 src/kleb_ast/scripts/reliable_ft_concat.sh                 # smoke (TMP-SMX)
#     sbatch --dependency=afterok:<gpu_jid> --array=0-21 src/kleb_ast/scripts/reliable_ft_concat.sh
#
#SBATCH --job-name=kleb_reliable_ft_concat
#SBATCH --output=kleb_reliable_ft_concat_%A_%a.out
#SBATCH --error=kleb_reliable_ft_concat_%A_%a.err
#SBATCH --partition=icelake-himem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --open-mode=append

set -euo pipefail
cd /home/dca36/workspace/BacPredict
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
       gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
       levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
if [[ -z "$DRUG" ]]; then echo "ERROR: no drug for array index $SLURM_ARRAY_TASK_ID" >&2; exit 1; fi

D=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed
FTC=$D/train_kleb_ast/pangena_predict/ft_amr_cache/$DRUG
FRC=$D/train_kleb_ast/pangena_predict/frozen_amr_cache/$DRUG
OUT=$D/train_kleb_ast/pangena_predict/reliable_ft_concat/$DRUG
mkdir -p "$OUT"
if [[ ! -f "$FTC/ft_genome_mean_${DRUG}.npz" ]]; then echo "ERROR: FT cache missing: $FTC" >&2; exit 1; fi
[[ -d "$FRC/frozen_amr_emb" ]] || echo "WARN: frozen cache missing ($FRC) — frozen per-gene LR skipped"
echo "=== Kp reliable ESM-vs-FT(+frozen) + concat — drug=$DRUG (task $SLURM_ARRAY_TASK_ID) ==="

uv run python -m kleb_ast.reliable_ft_concat \
    --drug "$DRUG" \
    --ast-sheet-path "$D/train_kleb_ast/binary_ast_with_split.csv" \
    --ft-cache-dir "$FTC" \
    --frozen-cache-dir "$FRC" \
    --esm-store-dir "$D/klebsiella_esm_embeddings" \
    --parquet-dir "$D/klebsiella_protein_sequences" \
    --sidecar-dir "$D/train_kleb_ast/amr_annotation" \
    --out-dir "$OUT" --grain family --n-folds 5 --seed 1

echo "Kp reliable ESM-vs-FT(+frozen) + concat ($DRUG) finished — $OUT"
