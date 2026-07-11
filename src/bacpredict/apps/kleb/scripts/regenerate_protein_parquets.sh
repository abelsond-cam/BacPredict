#!/bin/bash
#SBATCH --job-name=kleb_protein_parquets
#SBATCH --output=kleb_protein_parquets_%j.out
#SBATCH --error=kleb_protein_parquets_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=icelake-himem
#SBATCH --account=FLOTO-SL2-CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=38
#SBATCH --mem=128G
#SBATCH --open-mode=append
#
# Regenerate the Kp {Sample}_protein_sequences.parquet store (the gene->embedding-index map for the
# ESM ranking + concat). These were transient and never captured by the RCS backup, so rebuild from the
# SR Bakta GFF + assembly FASTA that the metadata references — the SAME source the embeddings were made
# from, so the parquet's flat protein order aligns with each {Sample}_esm_embeddings.pt (the downstream
# n_proteins guard enforces this and skips any mismatch).
#
# Node fraction (38 cores) for faster allocation; --skip-existing makes it resumable. After it completes,
# remember the archive-then-delete discipline: run backup_rds_to_rcs.sh and CONFIRM the parquets are on
# RCS before deleting them from RDS (the auto-login backup is additive but only captures what is present
# at login time).
#
# Submit: sbatch src/kleb_ast/scripts/regenerate_protein_parquets.sh

cd /home/dca36/workspace/BacPredict
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
D=/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david

# 1) Build the (Sample, sr_assembly_file, sr_gff_file) input CSV = AST cohort ∩ metadata_v2 with SR paths.
uv run python - <<'PY'
import pandas as pd
from pathlib import Path
base = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david")
ast = pd.read_csv(base / "processed/train_kleb_ast/binary_ast_with_split.csv", low_memory=False)
scol = "Sample" if "Sample" in ast.columns else "phenotype-BioSample_ID"
samples = set(ast[scol].astype(str))
meta = pd.read_csv(base / "final/metadata_v2_all_samples_and_columns.tsv", sep="\t",
                   usecols=["Sample", "sr_assembly_file", "sr_gff_file"], low_memory=False)
meta["Sample"] = meta["Sample"].astype(str)
sub = meta[meta["Sample"].isin(samples)].dropna(subset=["sr_assembly_file", "sr_gff_file"]).copy()
sub = sub[(sub["sr_assembly_file"].astype(str).str.strip() != "")
          & (sub["sr_gff_file"].astype(str).str.strip() != "")]
# metadata stores paths relative to the RDS root (david/raw/...) — resolve to absolute.
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
uv run python src/tl/embed/preprocess_assemblies_to_protein_sequences.py \
    --input-csv "$D/processed/protein_parquet_regen_input.csv" \
    --output-dir "$D/processed/klebsiella_protein_sequences" \
    --keep-internal-stop --skip-existing --workers 38
echo "PROTEIN_PARQUET_REGEN_DONE"
