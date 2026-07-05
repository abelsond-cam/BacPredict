"""Pair downloaded ATB assemblies with their BakRep GFF3s into an embedding input CSV.

The Isambard download layout for a cohort dir `<base>` is:
  <base>/assemblies/<BIOSAMPLE>.fa.gz                              (flat)
  <base>/gff/<datasetID>/<BIOSAMPLE>/<BIOSAMPLE>.bakta.gff3.gz     (BakRep-nested)

This scans both trees once and writes the `(Sample, sr_assembly_file, sr_gff_file)` CSV that
`preprocess_assemblies_to_protein_sequences.py` consumes — one row per BioSample that has
**both** an assembly and a GFF (protein extraction needs the pair). BioSamples with only one
are reported (counts) and skipped.

Example (on Isambard):
  python build_ast_input_csv.py --base /scratch/u6fp/dca36.u6fp/raw/tb \
    --out /scratch/u6fp/dca36.u6fp/processed/train_tb_ast/embedding_input.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def _biosample_from_gff(p: Path) -> str:
    """`<BIOSAMPLE>.bakta.gff3.gz` -> `<BIOSAMPLE>`."""
    return p.name[: -len(".bakta.gff3.gz")]


def main() -> None:
    """Scan a cohort dir's assemblies + gff trees and write the paired input CSV."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, type=Path, help="cohort dir with assemblies/ and gff/ subdirs")
    ap.add_argument("--out", required=True, type=Path, help="output embedding input CSV")
    args = ap.parse_args()

    asm_dir = args.base / "assemblies"
    gff_dir = args.base / "gff"

    # BioSample -> assembly path (flat layout).
    asm = {p.name[: -len(".fa.gz")]: p for p in asm_dir.glob("*.fa.gz") if p.stat().st_size > 0}
    # BioSample -> gff path (nested; first match wins if BakRep wrote duplicates).
    gff: dict[str, Path] = {}
    for p in gff_dir.rglob("*.bakta.gff3.gz"):
        if p.stat().st_size > 0:
            gff.setdefault(_biosample_from_gff(p), p)

    paired = sorted(set(asm) & set(gff))
    rows = [{"Sample": s, "sr_assembly_file": str(asm[s]), "sr_gff_file": str(gff[s])} for s in paired]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["Sample", "sr_assembly_file", "sr_gff_file"]).to_csv(args.out, index=False)

    print(f"assemblies on disk : {len(asm)}")
    print(f"gffs on disk       : {len(gff)}")
    print(f"paired (both)      : {len(paired)}  -> {args.out}")
    print(f"assembly-only      : {len(set(asm) - set(gff))} (no GFF; skipped)")
    print(f"gff-only           : {len(set(gff) - set(asm))} (no assembly; skipped)")


if __name__ == "__main__":
    main()
