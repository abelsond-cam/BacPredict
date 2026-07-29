#!/bin/bash
#SBATCH --job-name=kleb_protein_parquets
#SBATCH --output=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --error=/rds/user/dca36/hpc-work/logs/%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=icelake-himem
#SBATCH --account=FLOTO-PROJECT-K-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --mem=128G
#SBATCH --open-mode=append
# CSD3/UoHPC variant (when it returns): --partition=icelake-himem --account=FLOTO-SL2-CPU,
#   logs → relative or ~/rds/hpc-work/logs/.
#
# Regenerate the Kp {Sample}_protein_sequences.parquet store (the gene->embedding-index map for the
# ESM ranking + concat). These were transient and never captured by the RCS backup, so rebuild from the
# SR Bakta GFF + assembly FASTA that the metadata references — the SAME source the embeddings were made
# from, so the parquet's flat protein order aligns with each {Sample}_esm_embeddings.pt (the downstream
# n_proteins guard enforces this and skips any mismatch).
#
# Node fraction (38 cores) for faster allocation; --skip-existing makes it resumable. After it completes,
# remember the archive-then-delete discipline: CONFIRM the parquets are backed up before deleting them.
#
# Submit: sbatch src/bacpredict/apps/kleb/scripts/regenerate_protein_parquets.sh

set -uo pipefail

# Data root — one env var, cluster-agnostic (Isambard: $SCRATCHDIR; CSD3: project_k/david).
: "${BACPREDICT_DATA_ROOT:="$HOME/rds/rds-floto-bacterial-4k08a2yyQLw/david/bac_ast_prediction"}"
D="$BACPREDICT_DATA_ROOT"
PY="$HOME/workspace/BacPredict/.venv/bin/python"
export PYTHONPATH="$HOME/BacPredict/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1

# 1) Build the (Sample, sr_assembly_file, sr_gff_file) input CSV = AST cohort ∩ metadata_v2 with SR paths.
"$PY" - "$D" <<'PY'
import sys
import pandas as pd
from pathlib import Path
base = Path(sys.argv[1])
ast = pd.read_csv(base / "processed/train_kleb_ast/binary_ast_with_split.csv", low_memory=False)
scol = "Sample" if "Sample" in ast.columns else "phenotype-BioSample_ID"
samples = set(ast[scol].astype(str))
meta = pd.read_csv(base / "final/metadata_v2_all_samples_and_columns.tsv", sep="\t",
                   usecols=["Sample", "sr_assembly_file", "sr_gff_file"], low_memory=False)
meta["Sample"] = meta["Sample"].astype(str)
sub = meta[meta["Sample"].isin(samples)].dropna(subset=["sr_assembly_file", "sr_gff_file"]).copy()
sub = sub[(sub["sr_assembly_file"].astype(str).str.strip() != "")
          & (sub["sr_gff_file"].astype(str).str.strip() != "")]
# metadata stores paths relative to the data-root parent (CSD3: david/raw/...) — resolve to absolute.
rds = base.parent
sub["sr_assembly_file"] = sub["sr_assembly_file"].apply(lambda p: str(rds / str(p)))
sub["sr_gff_file"] = sub["sr_gff_file"].apply(lambda p: str(rds / str(p)))
out = base / "processed/protein_parquet_regen_input.csv"
sub.to_csv(out, index=False)
print(f"wrote {out} with {len(sub)} samples (of {len(samples)} AST samples)")
PY

# 2) Regenerate the parquets. --keep-internal-stop reproduces the original protein order (the embeddings
#    were made before the internal-stop skip was added; without this the parquet is off by ~3 proteins
#    and the n_proteins guard would skip every genome). --skip-existing makes it resumable.
"$PY" -m bacpredict.engine.embedding.preprocess_assemblies_to_protein_sequences \
    --input-csv "$D/processed/protein_parquet_regen_input.csv" \
    --output-dir "$D/processed/train_kleb_ast/protein_sequences" \
    --keep-internal-stop --skip-existing --workers 38
echo "PROTEIN_PARQUET_REGEN_DONE"
