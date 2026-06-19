#!/bin/bash
# Build the Kp Kleborate determinant "ceiling" (the WHO-catalogue analogue) for one or more drugs.
#
# Reads metadata_v2 (Kleborate determinant columns) + binary_ast_with_split.csv, joins on Sample, and
# scores each drug's determinant one-hot through the shared k-fold harness — emitting per-mechanism bars
# (HGT vs chromosomal) + the __ALL_Kleborate__ ceiling to docs/visualisations/kp_<drug>/.
#
# Light, CPU-only, no GPU — runs on the login node in a few minutes (one TSV + one CSV read, sklearn LR).
# Watch the join-coverage log line per drug: if >50% of labelled samples don't match a metadata_v2 row,
# the Sample key (assembly accession vs BioSample) needs reconciling before the numbers mean anything.
#
# Ensure the HPC checkout is on the intended branch and up to date FIRST (do not let this script pull the
# shared tree). Usage (from anywhere on HPC):
#     bash /home/dca36/workspace/BacPredict/src/kleb_ast/scripts/run_kleborate_ceiling.sh
#     bash .../run_kleborate_ceiling.sh colistin azithromycin        # explicit drug subset
#     bash .../run_kleborate_ceiling.sh ALL                          # the full 22-drug panel

set -euo pipefail
cd /home/dca36/workspace/BacPredict
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1

# Weakest-first default (Bacformer held-out AUROC ascending) — where concat is most likely to help.
WEAK="colistin azithromycin cefepime aztreonam cefoxitin tetracycline"
FULL="$WEAK piperacillin-tazobactam levofloxacin amikacin tobramycin trimethoprim-sulfamethoxazole \
imipenem cefazolin meropenem ceftazidime ciprofloxacin cefuroxime ceftriaxone ampicillin-sulbactam \
ertapenem cefotaxime gentamicin"

if [ "${1:-}" = "ALL" ]; then
  DRUGS="$FULL"
elif [ "$#" -gt 0 ]; then
  DRUGS="$*"
else
  DRUGS="$WEAK"
fi

echo "Building Kleborate ceiling for: $DRUGS"
uv run python src/kleb_ast/kleborate_determinant_lr.py --drugs $DRUGS
echo "KLEBORATE_CEILING_DONE"
