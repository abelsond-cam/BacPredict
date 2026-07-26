#!/bin/bash
# AMR concat LADDER — per drug, the four score configs vs the catalogue one-hot ceiling:
#   ft_mean  |  + best baclm coding gene  |  + best baclm non-coding  |  + both
# The non-coding block is the top-imputed-AUROC region (NO prevalence gate) across three IMPUTED full-band
# rankings on the baclm re-embed store — upstream promoter (upstream_lr_ranking_imputed_full), per-unit
# named body (per_unit_lr_ranking_imputed), and per-IGR flank pair incl. merged convergent regions
# (per_igr_lr_ranking_imputed_full); the coding block comes from per_gene_lr_ranking_imputed_baclm.
# Each config is re-scored by the same zero-imputed OOF k-fold LR over the FT eval-holdout universe.
# Raw AUROC recovery — NO lineage netting (rif/cipro are coding-determinant controls). CPU-only.
#
# Diagnostics-first (David 2026-07-19): rif/cipro coding controls + eth/strep/kan (TB) + azithro (Kp).
#   sbatch --export=ALL,REPO=$SCRATCHDIR/worktrees/concat --array=0,6,7,9 \
#       src/bacpredict/engine/scripts/build_amr_ladder.sh                                   # TB diag
#   SPECIES=kp sbatch --export=ALL,SPECIES=kp,REPO=$SCRATCHDIR/worktrees/concat --array=5,20 \
#       src/bacpredict/engine/scripts/build_amr_ladder.sh                                   # Kp diag (cipro,azithro)
# Full panel: --array=0-9 (TB) / --array=0-21 (Kp). Needs the imputed full-band non-coding rankings
# (build_upstream_region_lr_ranking.sh / build_per_igr_lr_ranking.sh FEATURE=imputed_full + per_unit imputed).
#
#SBATCH --job-name=amr_ladder
#SBATCH --output=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --error=/scratch/u6fp/dca36.u6fp/logs/%x-%A_%a.out
#SBATCH --array=0-9
#SBATCH --partition=workq
#SBATCH --account=brics.u6fp
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --open-mode=append
# CPU-only (.pt reads + GFF parse + sklearn LR over the FT holdout universe; ~7k genomes for TB, ~865 for
# Kp). No --gres=gpu; a GPU-less workq job schedules normally, but MUST set --mem (DefMemPerGPU otherwise).

set -uo pipefail
: "${BACPREDICT_DATA_ROOT:="$SCRATCHDIR"}"
D="$BACPREDICT_DATA_ROOT"
# Code checkout: the non-coding rung (per-unit loader, core selection) needs the concat worktree at 5c1d523+
# until the consolidate worktree is advanced; override REPO=... otherwise.
REPO="${REPO:-$SCRATCHDIR/worktrees/consolidate}"
PY="$SCRATCHDIR/envs/bacpredict-gpu-venv/bin/python"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SPECIES=${SPECIES:-tb}
if [[ "$SPECIES" == "kp" ]]; then
    TASK=kleb_ast
    DRUGS=(cefotaxime ertapenem ampicillin-sulbactam ceftriaxone cefuroxime ciprofloxacin ceftazidime \
           gentamicin cefazolin imipenem meropenem trimethoprim-sulfamethoxazole tobramycin amikacin \
           levofloxacin piperacillin-tazobactam cefoxitin tetracycline aztreonam cefepime azithromycin colistin)
else
    TASK=tb_ast
    DRUGS=(rifampin isoniazid ethambutol pyrazinamide moxifloxacin levofloxacin streptomycin ethionamide rifabutin kanamycin)
fi
DRUG=${DRUGS[$SLURM_ARRAY_TASK_ID]}
[[ -z "$DRUG" ]] && { echo "ERROR: no drug for index $SLURM_ARRAY_TASK_ID (species=$SPECIES)" >&2; exit 1; }

# FT genome-mean dir. Defaults to the standard cache; override FT_CACHE for a drug whose npz sits elsewhere
# (e.g. Kp ciprofloxacin's CP-0 cache lives under ft_amr_cache/, not ft_bacformer_cache/).
FT_CACHE="${FT_CACHE:-$D/processed/train_${TASK}/pangena_predict/ft_bacformer_cache/$DRUG}"
OUT=$D/processed/train_${TASK}/pangena_predict/amr_ladder/$DRUG

echo "========================================================================"
echo "AMR ladder — species=$SPECIES drug=$DRUG (array task $SLURM_ARRAY_TASK_ID)"
echo "FT cache: $FT_CACHE"
echo "Out dir:  $OUT   (non-coding rung: top imputed AUROC, no gate, over upstream/per_unit/per_igr)"
echo "Job ID:   ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "========================================================================"
if [[ ! -f "$FT_CACHE/ft_genome_mean_${DRUG}.npz" ]]; then
    echo "ERROR: FT genome-mean missing: $FT_CACHE/ft_genome_mean_${DRUG}.npz" >&2
    exit 1
fi

"$PY" -m bacpredict.engine.segment_amr_lr.concat.build_amr_ladder \
    --species "$SPECIES" \
    --drug "$DRUG" \
    --ft-cache-dir "$FT_CACHE" \
    --out-dir "$OUT"

echo "=== ladder table ($DRUG) ==="
cat "$OUT/${DRUG}_amr_ladder_table.csv"
echo "AMR ladder ($SPECIES $DRUG) finished — $OUT/${DRUG}_amr_ladder_table.csv"
