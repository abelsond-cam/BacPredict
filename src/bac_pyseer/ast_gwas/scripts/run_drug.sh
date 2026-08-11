#!/usr/bin/env bash
# Per drug: phenotype -> kinship subset -> sharded pyseer LMM -> postprocess -> design -> LR.
#
# Everything expensive (unitig matrix, mash triangle, lineage clusters) was built once per organism
# by build_cohort_once.sh; this only pays for one pyseer run plus a cached sub-matrix extraction.
#
# The phenotype carries train+validate ONLY, so unitig selection never sees a holdout label -- that
# is what makes the resulting AUROC comparable to the Bacformer fine-tune's. The read-out then
# scores the untouched holdout once. Set SPLITS=train,validate,holdout to reproduce the leaky
# classic-GWAS framing for comparison; the manifest records which was used.
#
# Usage:
#   ORGANISM=kp DRUG=ertapenem   bash src/bac_pyseer/ast_gwas/scripts/run_drug.sh
#   ORGANISM=kp DRUG=colistin    bash src/bac_pyseer/ast_gwas/scripts/run_drug.sh
#   ORGANISM=tb DRUG=rifampin    bash src/bac_pyseer/ast_gwas/scripts/run_drug.sh   # NOT rifampicin
#   ORGANISM=tb DRUG=ethionamide bash src/bac_pyseer/ast_gwas/scripts/run_drug.sh
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
ORGANISM=${ORGANISM:?set ORGANISM=kp|tb}
DRUG=${DRUG:?set DRUG=<ast column name>}
DATA_ROOT=${BACPREDICT_DATA_ROOT:-${SCRATCHDIR:?set BACPREDICT_DATA_ROOT or SCRATCHDIR}}

case "$ORGANISM" in
    kp) TASK=train_kleb_ast ;;
    tb) TASK=train_tb_ast ;;
    *)  echo "ORGANISM must be kp or tb" >&2; exit 1 ;;
esac

OUT_DIR=${OUT_DIR:-$DATA_ROOT/processed/pyseer_ast/$ORGANISM}
UNITIG_DIR=$OUT_DIR/unitigs
STRUCT_DIR=$OUT_DIR/structure
DRUG_DIR=$OUT_DIR/$DRUG
GWAS_DIR=$DRUG_DIR/gwas
SPLIT_TABLE=${SPLIT_TABLE:-$DATA_ROOT/processed/$TASK/splits/${DRUG}_split.csv}
MATRIX=$UNITIG_DIR/unitigs.pyseer.gz
TRIANGLE=$STRUCT_DIR/mash_triangle.txt
CLUSTERS=$STRUCT_DIR/lineage_clusters.tsv
PHENO=$DRUG_DIR/phenotype.tsv
SIMILARITY=$DRUG_DIR/similarity.tsv
DISTANCES=$DRUG_DIR/distances.tsv              # pyseer --lineage needs a --distances file too
SPLITS=${SPLITS:-train,validate}
LOGDIR=${LOGDIR:-$DATA_ROOT/logs}

ACCT=${ACCT:-brics.u6fp}
PART=${PART:-workq}
# ${QOS-normal}, not ${QOS:-normal}: QOS= (explicitly empty) must omit --qos entirely, as CSD3
# requires — see the same note in build_cohort_once.sh.
QOS=${QOS-normal}
NSHARDS=${NSHARDS:-64}
CPU=${CPU:-8}

for required in "$MATRIX" "$TRIANGLE" "$CLUSTERS" "$SPLIT_TABLE"; do
    [ -s "$required" ] || { echo "ERROR: missing $required — run build_cohort_once.sh first" >&2; exit 1; }
done
mkdir -p "$DRUG_DIR" "$GWAS_DIR" "$LOGDIR"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

echo "=== (1) phenotype (${SPLITS}) -> $PHENO ==="
uv run python -m bac_pyseer.ast_gwas.build_ast_phenotype \
    --split-table "$SPLIT_TABLE" --drug "$DRUG" --out-tsv "$PHENO" --splits "$SPLITS"

echo "=== (2) kinship + distances for exactly this drug's samples ==="
# Cutting the cohort-wide triangle down per drug is what keeps n^2 tractable: most drugs have far
# fewer labelled genomes than the cohort (Kp ertapenem ~2.1k, colistin ~1.4k vs a 9.7k cohort).
uv run python -m bac_pyseer.ast_gwas.mash_kinship kinship \
    --triangle "$TRIANGLE" --out-tsv "$SIMILARITY" --phenotype-tsv "$PHENO" \
    --distances-tsv "$DISTANCES"

echo "=== (3) sharded pyseer LMM ==="
# Reuses the calibrated three-phase prep -> array -> combine chain. Peak RAM is ~cpu x n^2, so for a
# large phenotype (TB rifampin, n~29k) either drop CPU and raise NSHARDS or ask for a whole node.
GWAS_JOB=$(PAIR="$DRUG" LABEL_COL="${DRUG}_label" OUT_STEM="$DRUG" \
    PHENO="$PHENO" SIM="$SIMILARITY" DIST="$DISTANCES" CLUSTERS_TSV="$CLUSTERS" \
    MATRIX="$MATRIX" GWAS_DIR="$GWAS_DIR" NSHARDS="$NSHARDS" CPU="$CPU" \
    ACCT="$ACCT" PART="$PART" QOS="$QOS" LOGDIR="$LOGDIR" \
    bash "$REPO/src/bac_pyseer/kleb_iso_source/scripts/run_unitig_lmm_sharded.sh" | tail -1)
echo "  gwas chain submitted ($GWAS_JOB)"

cat <<EOF

=== (3) is running on SLURM; when it finishes, run the read-out ===

  # threshold + lambda + hit table (pheno_var computed from THIS cohort, not assumed 50:50)
  uv run python -m bac_pyseer.kleb_iso_source.pyseer_postprocess \\
      --assoc '$GWAS_DIR/$DRUG.assoc' --patterns '$GWAS_DIR/patterns.txt' \\
      --feature-mode unitigs \\
      --phenotype-tsv '$PHENO' --phenotype-column '${DRUG}_label' \\
      --out-fig-dir '$DRUG_DIR' --out-table '$DRUG_DIR/${DRUG}_hits_annotated.tsv' \\
      --summary-json '$DRUG_DIR/${DRUG}_gwas_summary.json'

  # significant unitigs -> sparse presence over ALL split genomes (one cached pass over the matrix)
  uv run python -m bac_pyseer.ast_gwas.unitig_design_matrix \\
      --hits-tsv '$DRUG_DIR/${DRUG}_hits_annotated.tsv' --matrix-gz '$MATRIX' \\
      --split-table '$SPLIT_TABLE' --out-dir '$DRUG_DIR/design'

  # fit on train, threshold on validate, score the holdout once
  uv run python -m bac_pyseer.ast_gwas.unitig_lr \\
      --design-dir '$DRUG_DIR/design' --split-table '$SPLIT_TABLE' \\
      --drug '$DRUG' --organism '$ORGANISM' --out-dir '$DRUG_DIR/lr' \\
      --gwas-summary '$DRUG_DIR/${DRUG}_gwas_summary.json'

  # LD control: refit on one unitig per perfect-LD block
  uv run python -m bac_pyseer.ast_gwas.unitig_design_matrix \\
      --hits-tsv '$DRUG_DIR/${DRUG}_hits_annotated.tsv' --matrix-gz '$MATRIX' \\
      --split-table '$SPLIT_TABLE' --out-dir '$DRUG_DIR/design_dedup' --dedupe-patterns

[look] Before trusting the hits, run the calibration protocol -- lambda by allele frequency and the
within-lineage permutation null (genomic_inflation_by_af.py / permute_phenotype_within_lineage.py
with '$CLUSTERS'). If the permutation null is inflated, the mash kinship is under-correcting and
the hits are structure, not signal.
EOF
