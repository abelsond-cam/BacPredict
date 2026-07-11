#!/bin/bash
# Plot #5 (CPU): the gene-ingredient concat — frozen-ESM vs frozen-Bacformer vs FT-Bacformer gene block,
# each ⊕ {frozen mean, FT mean}, one drug per array task (kleb_ast.gene_ingredient_concat). Reads the FT
# cache (ft_amr_cache/) + the frozen cache (frozen_amr_cache/) + the ESM store; no forward pass. Run AFTER
# the matching cache_frozen_amr_proteins GPU task for the drug.
#
#     sbatch --array=11 src/kleb_ast/scripts/gene_ingredient_concat.sh                  # smoke (TMP-SMX)
#     sbatch --dependency=afterok:<gpu_jid> --array=0-21 src/kleb_ast/scripts/gene_ingredient_concat.sh
#
#SBATCH --job-name=kleb_gene_ingredient_concat
#SBATCH --output=kleb_gene_ingredient_concat_%A_%a.out
#SBATCH --error=kleb_gene_ingredient_concat_%A_%a.err
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
GRAIN=${GRAIN:-family}

D=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed
FTC=$D/train_kleb_ast/pangena_predict/ft_amr_cache/$DRUG
FRC=$D/train_kleb_ast/pangena_predict/frozen_amr_cache/$DRUG
OUT=$D/train_kleb_ast/pangena_predict/gene_ingredient_concat/$DRUG
mkdir -p "$OUT"
if [[ ! -f "$FTC/ft_genome_mean_${DRUG}.npz" ]]; then echo "ERROR: FT cache missing: $FTC" >&2; exit 1; fi
if [[ ! -f "$FRC/frozen_genome_mean_${DRUG}.npz" ]]; then echo "ERROR: frozen cache missing: $FRC" >&2; exit 1; fi
echo "=== Kp gene-ingredient concat — drug=$DRUG grain=$GRAIN (task $SLURM_ARRAY_TASK_ID) ==="

uv run python -m kleb_ast.gene_ingredient_concat \
    --drug "$DRUG" \
    --ast-sheet-path "$D/train_kleb_ast/binary_ast_with_split.csv" \
    --ft-cache-dir "$FTC" --frozen-cache-dir "$FRC" \
    --esm-store-dir "$D/klebsiella_esm_embeddings" \
    --parquet-dir "$D/klebsiella_protein_sequences" \
    --sidecar-dir "$D/train_kleb_ast/amr_annotation" \
    --out-dir "$OUT" --grain "$GRAIN" --n-folds 5 --seed 1

echo "Kp gene-ingredient concat ($DRUG) finished — $OUT"
