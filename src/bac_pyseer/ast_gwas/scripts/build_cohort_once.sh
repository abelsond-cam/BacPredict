#!/usr/bin/env bash
# Once per organism: reflist -> GGCAT unitig matrix -> mash sketch/triangle -> lineage clusters.
#
# Everything here is drug-independent and is the expensive part, so it runs ONCE per organism and
# every drug's GWAS then reuses it. That is what makes the fan-out to all 22 Kp + 10 TB drugs
# affordable: per drug we pay one pyseer run plus a cached sub-matrix extraction, not a rebuild.
#
# Login-node orchestrator (no compute here) -- it resolves the reflist inline, then submits the
# GGCAT build and the mash chain with --dependency so the clusters wait for the triangle.
#
# Usage:
#   ORGANISM=kp bash src/bac_pyseer/ast_gwas/scripts/build_cohort_once.sh
#   ORGANISM=tb MAX_FRAC=0.99 bash src/bac_pyseer/ast_gwas/scripts/build_cohort_once.sh
#
# Cluster knobs (defaults = Isambard; override for CSD3):
#   ACCT/PART/QOS/LOGDIR, and GGCAT_CPUS/GGCAT_MEM/GGCAT_TIME, MASH_CPUS/MASH_MEM/MASH_TIME.
#
# FILE_LIST resolves the cohort through a Sample<TAB>path TSV instead of scanning a flat directory.
# Needed for Kp on CSD3, which has no flat BioSample-keyed dir -- see resolve_ast_assemblies.py.
#   ORGANISM=kp FILE_LIST=$DATA/raw/assemblies_file_list.tsv bash .../build_cohort_once.sh
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
ORGANISM=${ORGANISM:?set ORGANISM=kp|tb}
DATA_ROOT=${BACPREDICT_DATA_ROOT:-${SCRATCHDIR:?set BACPREDICT_DATA_ROOT or SCRATCHDIR}}

OUT_DIR=${OUT_DIR:-$DATA_ROOT/processed/pyseer_ast/$ORGANISM}
COHORT_DIR=$OUT_DIR/cohort
UNITIG_DIR=$OUT_DIR/unitigs
STRUCT_DIR=$OUT_DIR/structure
REFLIST=$UNITIG_DIR/assembly_refs.txt          # run_ggcat_unitigs.sh reads exactly this path
SCRATCH=${SCRATCH:-$DATA_ROOT/scratch/ggcat/$ORGANISM}
LOGDIR=${LOGDIR:-$DATA_ROOT/logs}

# Isambard defaults; CSD3 would be ACCT=FLOTO-PROJECT-K-SL2-CPU PART=icelake-himem QOS=
ACCT=${ACCT:-brics.u6fp}
PART=${PART:-workq}
# ${QOS-normal}, not ${QOS:-normal}: unset gives Isambard's default, but QOS= (explicitly empty)
# must omit --qos entirely. On CSD3 the FLOTO associations allow only cpu1/intr, so a --qos=normal
# it cannot be told to drop is a rejected submission.
QOS=${QOS-normal}
GGCAT_CPUS=${GGCAT_CPUS:-64}; GGCAT_MEM=${GGCAT_MEM:-350G}; GGCAT_TIME=${GGCAT_TIME:-36:00:00}
MASH_CPUS=${MASH_CPUS:-32};   MASH_MEM=${MASH_MEM:-128G};   MASH_TIME=${MASH_TIME:-12:00:00}

mkdir -p "$COHORT_DIR" "$UNITIG_DIR" "$STRUCT_DIR" "$SCRATCH" "$LOGDIR"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

echo "=== (0) resolve the AST cohort -> $REFLIST ==="
if [ -s "$REFLIST" ]; then
    echo "reusing existing reflist ($(wc -l < "$REFLIST") samples)"
else
    uv run python -m bac_pyseer.ast_gwas.resolve_ast_assemblies \
        --organism "$ORGANISM" --out-tsv "$REFLIST" --data-root "$DATA_ROOT" \
        ${FILE_LIST:+--file-list "$FILE_LIST"}
fi
NSAMP=$(wc -l < "$REFLIST")
echo "cohort: $NSAMP genomes"

sb() { sbatch --parsable --account="$ACCT" --partition="$PART" ${QOS:+--qos="$QOS"} \
              --nodes=1 --ntasks=1 "$@"; }

echo "=== (1) GGCAT unitig build (once per organism) ==="
# Reuses the proven kleb_iso_source build; its cluster roots are env-overridable and its #SBATCH
# header is overridden on the command line (the pattern run_unitig_lmm_sharded.sh already uses).
GGCAT_JOB=$(REPO="$REPO" OUT_DIR="$UNITIG_DIR" TMP="$SCRATCH" MAX_FRAC="${MAX_FRAC:-0.99}" \
    sb --cpus-per-task="$GGCAT_CPUS" --mem="$GGCAT_MEM" --time="$GGCAT_TIME" \
       --job-name="ggcat_${ORGANISM}" \
       --output="$LOGDIR/ggcat_${ORGANISM}_%j.out" --error="$LOGDIR/ggcat_${ORGANISM}_%j.err" \
       --export=ALL,REPO,OUT_DIR,TMP,MAX_FRAC \
       "$REPO/src/bac_pyseer/kleb_iso_source/scripts/run_ggcat_unitigs.sh")
echo "  ggcat job: $GGCAT_JOB -> $UNITIG_DIR/unitigs.pyseer.gz"

echo "=== (2) mash sketch + triangle + lineage clusters (once per organism) ==="
MASH_JOB=$(sb --cpus-per-task="$MASH_CPUS" --mem="$MASH_MEM" --time="$MASH_TIME" \
    --job-name="mash_${ORGANISM}" \
    --output="$LOGDIR/mash_${ORGANISM}_%j.out" --error="$LOGDIR/mash_${ORGANISM}_%j.err" \
    --wrap "set -euo pipefail
            cd '$REPO'
            unset PYTHONPATH PYTHONHOME
            export PYTHONPATH='$REPO/src'
            # mash_kinship shells out to \`mash\`, which is in the bac_pyseer pixi env, not on PATH.
            # Without this the job dies in ~9 s with \"'mash' not on PATH\" — measured 2026-08-28.
            export PATH='${PIXI_BIN:-$REPO/src/bac_pyseer/.pixi/envs/default/bin}':\$PATH
            uv run python -m bac_pyseer.ast_gwas.mash_kinship sketch \
                --reflist '$REFLIST' --out-dir '$STRUCT_DIR' --threads $MASH_CPUS
            uv run python -m bac_pyseer.ast_gwas.lineage_from_distances \
                --triangle '$STRUCT_DIR/mash_triangle.txt' \
                --out-tsv '$STRUCT_DIR/lineage_clusters.tsv'")
echo "  mash job:  $MASH_JOB -> $STRUCT_DIR/{mash_triangle.txt,lineage_clusters.tsv}"

cat <<EOF

=== submitted ===
  ggcat  $GGCAT_JOB
  mash   $MASH_JOB
Watch with: squeue -u \$USER   (never pgrep; login processes die on disconnect)

[look] Before running any drug, check:
  - unitig count + matrix size:  ls -lh $UNITIG_DIR
  - how many near-universal unitigs the --max-samples cap dropped (ggcat log, stderr)
  - cluster size distribution:   $STRUCT_DIR/lineage_clusters.manifest.json

Then per drug: src/bac_pyseer/ast_gwas/scripts/run_drug.sh
EOF
