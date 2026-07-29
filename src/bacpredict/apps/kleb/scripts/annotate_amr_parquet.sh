#!/bin/bash
# Kp AST cohort: reliable Kleborate-style AMR labels → per-sample {Sample}_amr.parquet sidecars.
#
# CPU array, one CHUNK of genomes per array task. Each task minimap2-aligns the vendored CARD +
# chromosomal AMR refs against each genome's assembly and writes a flat-index-keyed AMR sidecar
# (no protein parquet rewrite, no re-embedding). Python runs under the venv $PY; the minimap2 binary
# comes from this task's pixi env (never `module load` — see csd3-spack-module-breaks-uv).
#
# One-time setup (login node, needs internet):
#     cd "$HOME/BacPredict/src/bacpredict/apps/kleb" && pixi install
#
# Isambard has no metadata_v2, so build the assembly/GFF path sheet once from the raw layout
# (login node, seconds — one listdir per gff shard):
#     "$PY" -m bacpredict.apps.kleb.build_isambard_amr_path_sheet \
#         --ast-sheet   "$SCRATCHDIR/processed/train_kleb_ast/binary_ast_with_split.csv" \
#         --assembly-dir "$SCRATCHDIR/raw/kleb_ast/assemblies" \
#         --gff-root     "$SCRATCHDIR/raw/kleb_ast/gff" \
#         --out          "$SCRATCHDIR/processed/train_kleb_ast/amr_path_sheet.tsv"
#   Then pass it below via META (default points there). On CSD3 use metadata_v2 instead (unset META).
#
# Size the array first (login node, seconds):
#     MM2=$(cd "$HOME/BacPredict/src/bacpredict/apps/kleb" && pixi run -- which minimap2)
#     "$PY" -m bacpredict.apps.kleb.annotate_amr_sidecar --dry-run --minimap2-bin "$MM2" \
#         --metadata "$SCRATCHDIR/processed/train_kleb_ast/amr_path_sheet.tsv" \
#         --amr-ref-dir "$HOME/BacHGT/src/bac_kleborate/refs/kleb_amr/inputs"
#   -> prints "worklist size = N". With CHUNK=200, set --array=0-$(( (N+199)/200 - 1 )).
#
# Smoke (5 known carbapenemase carriers; login node or a short interactive — minutes):
#     "$PY" -m bacpredict.apps.kleb.annotate_amr_sidecar --minimap2-bin "$MM2" \
#         --samples SAMPLE_A SAMPLE_B ... --workers 5
#
# Full run:
#     sbatch --array=0-NN src/bacpredict/apps/kleb/scripts/annotate_amr_parquet.sh
#
#SBATCH --job-name=kleb_amr_annotate
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%A_%a.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%A_%a.out
#SBATCH --partition=icelake-himem
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=icelake-himem --account=FLOTO-PROJECT-K-SL2-CPU,
#   logs → a project-tier logs dir (e.g. ~/rds/hpc-work/logs/%x-%A_%a.out).

set -euo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction"}"
D="$BACPREDICT_DATA_ROOT"
PY="$HOME/workspace/BacPredict/.venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

CHUNK=${CHUNK:-200}
START=$(( SLURM_ARRAY_TASK_ID * CHUNK ))

# Isambard: no metadata_v2 → the raw-layout path sheet (build_isambard_amr_path_sheet). The CARD/
# chromosomal refs live in the sibling BacHGT checkout ($HOME/BacHGT), not the stale CSD3 default.
# On CSD3, unset META (metadata_v2 default) and AMR_REF (module default resolves the workspace path).
META=${META:-$D/processed/train_kleb_ast/amr_path_sheet.tsv}
AMR_REF=${AMR_REF:-$HOME/BacHGT/src/bac_kleborate/refs/kleb_amr/inputs}
META_ARGS=()
[[ -n "$META" ]] && META_ARGS+=(--metadata "$META")
[[ -n "$AMR_REF" ]] && META_ARGS+=(--amr-ref-dir "$AMR_REF")
# The Isambard protein/embedding store was built keep_internal_stop=False; the sidecar flat_index must
# match it or every sample is 'misaligned'. Default off here; set KEEP_INTERNAL_STOP=1 on CSD3.
if [[ "${KEEP_INTERNAL_STOP:-0}" == "1" ]]; then META_ARGS+=(--keep-internal-stop); else META_ARGS+=(--no-keep-internal-stop); fi

# minimap2 from the kleb pixi env (absolute path; no PATH/module fiddling). Prefer a
# pre-resolved $MM2 (passed via --export) so array tasks don't each invoke pixi on a compute node.
MM2=${MM2:-$(cd "$HOME/BacPredict/src/bacpredict/apps/kleb" && pixi run -- which minimap2)}
echo "=== Kp AMR annotate — array task $SLURM_ARRAY_TASK_ID, chunk [$START:$((START+CHUNK))], minimap2=$MM2 ==="

# 32 cores: 8 genome workers × 4 minimap2 threads each.
"$PY" -m bacpredict.apps.kleb.annotate_amr_sidecar \
    --minimap2-bin "$MM2" \
    "${META_ARGS[@]}" \
    --workers 8 --threads 4 \
    --start "$START" --count "$CHUNK" \
    --skip-existing

echo "Kp AMR annotate (task $SLURM_ARRAY_TASK_ID) finished."
