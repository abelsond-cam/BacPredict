#!/bin/bash
# Rebuild the invasion-GWAS report's tables + figures (docs/PROGRESS.md) from the saved artifacts.
# LOCAL or HPC login node: pure pandas + matplotlib over the committed hit tables, concordance-union,
# lineage-breadth, and the Poisson file — no SLURM, seconds. Safe to re-run anytime.
#
# The two HPC-data-producing steps run ONCE (re-run only if the .assoc / Rtab change):
#   (1) §2 concordance-union — needs both full .assoc on RDS; run on the HPC login node:
#         uv run python src/bac_pyseer/kleb_iso_source/build_blood_resp_concordance.py \
#           --out src/bac_pyseer/docs/visualise/faeces_resp_lmm_model/blood_resp_concordance_union.tsv
#   (2) §3 lineage breadth — needs the ~10 GB --pres Rtab; submit an icelake-himem job (~70 s):
#         sbatch --account=FLOTO-PROJECT-K-SL2-CPU --partition=icelake-himem --cpus-per-task=8 \
#           --mem=64G --time=4:00:00 --wrap="cd ~/workspace/BacPredict && uv run python \
#           src/bac_pyseer/kleb_iso_source/build_lineage_breadth.py --hits <union.tsv> --out <breadth.tsv>"
#       (see build_lineage_breadth.py header for the exact Rtab/labels defaults).
#
# Then THIS script rebuilds the cross-axis table + every figure/table from those TSVs.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO"
K=src/bac_pyseer/kleb_iso_source
FR=src/bac_pyseer/docs/visualise/faeces_resp_lmm_model

uv run python "$K/build_cross_axis_table.py" --lineage-breadth "$FR/lineage_breadth.tsv"
MPLBACKEND=Agg uv run python "$K/make_progress_figures.py"
echo "rebuilt: cross_axis_candidates.tsv + progress_figures/*.png + orientation/regulator tables"
