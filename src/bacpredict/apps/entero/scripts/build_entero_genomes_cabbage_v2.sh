#!/usr/bin/env bash
# CABBAGE v2 (PROJECT_STATE §3.7): the Enterobacterales cohort's genomes, built as build_entero_genomes.sh built §3.6's —
# ATB → portal GFF → shovill (--prefer-reads, 10×, §3.6's genome size per genus), Prodigal + mlst, CheckM2 + skani
# against the same DRAFT panel, collate in report mode — for the genomes no §3.6 (or v2 P. aeruginosa) manifest names;
# every other genome is §3.6's own, reused with its QC. Stages and mechanics:
# engine/genomes/scripts/build_genomes_reusing_built.sh.
#
# The one difference from §3.6's first build is the fix §3.6 then applied: the MLST scheme is pinned per genus at
# submission (one array per scheme), not auto-detected (PROJECT_STATE §6, 2026-11-21). The map is the scheme each
# genus ended on in §3.6's re-typed genome_qc.csv, plus the v2 genera §3.6 did not have (Citrobacter, Yersinia,
# Cronobacter); a genus with no scheme (Morganella, Kluyvera, Hafnia, …) is built without one, as in §3.6. One species
# is pinned apart from its genus: aerogenes, which §3.6 labels Enterobacter and v2 Klebsiella, takes `kaerogenes`
# under either name — §3.6's are on it (the §3.6 agent, 2026-11-21), and a genus pin would split it across two.
#
#   ARRAY_REPO=<detached worktree> bash src/bacpredict/apps/entero/scripts/build_entero_genomes_cabbage_v2.sh <stage> …
set -euo pipefail

ROOT=${BACPREDICT_AMR_ROOT:-$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/amr_prediction}
[[ -d $ROOT ]] || { echo "no AMR root at $ROOT (set BACPREDICT_AMR_ROOT)"; exit 1; }
P=$ROOT/processed
E=$P/enterobacterales/genomes
REPO=$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)
export ROOT T=$P/enterobacterales_cabbage_v2 JOB=v2-entero
export OWNER_MANIFESTS="$P/paeruginosa/genomes/genome_manifest.csv $E/genome_manifest.csv $P/paeruginosa_cabbage_v2/genomes/genome_manifest.csv"
export REUSE_SHARD_DIRS=$E/shards  # holds the MLST re-type's shards too: collate keeps each genome's newest row
REUSE_CHECKM2=$(compgen -G "$E/qc/checkm2/checkm2_quality_report_*.tsv"; compgen -G "$E/qc/topup/checkm2/checkm2_quality_report_*.tsv")
export REUSE_CHECKM2=${REUSE_CHECKM2//$'\n'/ }
export REUSE_SKANI=$E/qc/skani_dist.tsv
export SCHEME_BY_SPECIES="Klebsiella aerogenes=kaerogenes,Enterobacter aerogenes=kaerogenes"
export SCHEME_BY_GENUS=Escherichia=ecoli_achtman_4,Shigella=ecoli_achtman_4,Salmonella=salmonella,Klebsiella=klebsiella,Enterobacter=ecloacae,Serratia=serratia,Proteus=proteus,Providencia=providencia,Citrobacter=cfreundii,Yersinia=yersinia,Cronobacter=cronobacter
# §3.6's: typical complete-genome lengths per genus; any other genus 5.0 Mbp
export GENOME_SIZE=5.0e6
export GENOME_SIZES=Escherichia=5.1e6,Shigella=4.6e6,Salmonella=4.9e6,Klebsiella=5.6e6,Enterobacter=4.9e6,Serratia=5.2e6,Proteus=4.1e6,Morganella=3.9e6,Providencia=4.3e6
REFS=$(grep -v '^#' "$REPO/src/bacpredict/apps/entero/species_reference_panel.tsv" | tail -n +2 \
       | awk -F'\t' -v d="$ROOT/raw/reference_genomes" '{print $1 "\t" d "/" $2 ".fna.gz"}')
export REFS
export SPECIES_MODE=report
export ATB_INDEX=$ROOT/raw/allthebacteria/assemblies_list_20261115
exec bash "$REPO/src/bacpredict/engine/genomes/scripts/build_genomes_reusing_built.sh" "$@"
