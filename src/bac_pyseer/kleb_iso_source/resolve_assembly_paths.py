"""Resolve cohort ``Sample`` IDs to their assembly FASTA on RDS → a unitig-caller input list.

Unitigs need the *full* genome sequence (core + accessory), so — unlike the variant GWAS,
which uses snippy's reference-anchored calls — the unitig caller is fed the per-sample
assembly FASTAs. metadata_v2 already carries these paths (added by the BacHGT
``add_paths_gff_fna_to_metadata.py``):

- ``lr_assembly_file`` — long-read / NCBI assembly (used for ``GCF_/GCA_`` Samples).
- ``sr_assembly_file`` — short-read assembly (used for biosample Samples).

This unions one or more cohort/split CSVs (by their ``Sample`` column), picks the right
assembly path per Sample (long-read column for ``GCF_/GCA_*``, short-read otherwise, with a
fallback to the other column), and writes the tab-separated ``sample<TAB>assembly_path``
list that ``unitig-caller --call --refs`` consumes. Pure metadata lookup — no per-file stat.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# metadata_v2 stores assembly paths RELATIVE to the project_k RDS root (e.g.
# ``seb/assemblies_2/…`` or ``david/raw/related_lr/assemblies/…``), so prefix this.
PROJECT_K_ROOT = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw")
METADATA_V2_DEFAULT = PROJECT_K_ROOT / "david/final/metadata_v2_all_samples_and_columns.tsv"
_META_COLS = ["Sample", "sr_assembly_file", "lr_assembly_file", "kpsc_final_list"]


def resolve(
    sample_csvs: list[Path], all_kpsc: bool, metadata_path: Path, out_tsv: Path,
    path_root: Path = PROJECT_K_ROOT, check_exists: bool = False,
) -> pd.DataFrame:
    """Resolve every (unioned) cohort Sample to an assembly FASTA; write the unitig-caller list."""
    print(f"Loading metadata (usecols): {metadata_path}")
    meta = pd.read_csv(metadata_path, sep="\t", usecols=_META_COLS, low_memory=False)
    meta["Sample"] = meta["Sample"].astype(str)

    if all_kpsc:
        samples = meta.loc[meta["kpsc_final_list"].fillna(False).astype(bool), "Sample"]
    else:
        parts = [pd.read_csv(c, usecols=["Sample"])["Sample"].astype(str) for c in sample_csvs]
        samples = pd.concat(parts, ignore_index=True)
    samples = samples.drop_duplicates().reset_index(drop=True)
    print(f"Union work-list: {len(samples)} unique Samples from {len(sample_csvs)} cohort CSV(s)")

    df = pd.DataFrame({"Sample": samples})
    sr = meta.set_index("Sample")["sr_assembly_file"]
    lr = meta.set_index("Sample")["lr_assembly_file"]
    is_lr = df["Sample"].str.startswith(("GCF_", "GCA_"))
    lr_path = df["Sample"].map(lr)
    sr_path = df["Sample"].map(sr)
    # long-read Samples → lr column (fallback sr); biosample Samples → sr column (fallback lr)
    df["assembly_path"] = lr_path.where(is_lr, sr_path)
    df["assembly_path"] = df["assembly_path"].fillna(sr_path.where(is_lr, lr_path))
    df["source"] = is_lr.map({True: "lr_assembly", False: "sr_assembly"})

    # metadata paths are relative to project_k → make absolute (leave any already-absolute path)
    root = str(path_root).rstrip("/")

    def _abs(p: object) -> object:
        if pd.isna(p) or str(p) == "":
            return p
        p = str(p)
        return p if p.startswith("/") else f"{root}/{p}"

    df["assembly_path"] = df["assembly_path"].map(_abs)

    resolved = df[df["assembly_path"].notna() & (df["assembly_path"].astype(str).str.len() > 0)].copy()
    if check_exists:  # drop stale paths so a missing file can't abort the (long) DBG build
        present = resolved["assembly_path"].map(os.path.exists)
        n_missing = int((~present).sum())
        if n_missing:
            print(f"  dropping {n_missing} resolved Samples whose assembly file is missing on disk")
            resolved = resolved[present]
    unresolved = df[~df["Sample"].isin(resolved["Sample"])].copy()

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    resolved[["Sample", "assembly_path"]].to_csv(out_tsv, sep="\t", index=False, header=False)
    unresolved["Sample"].to_csv(out_tsv.with_suffix(".unresolved.txt"), index=False, header=False)

    print("\n=== resolution summary ===")
    print(f"  requested : {len(df)}")
    print(f"  resolved  : {len(resolved)}")
    for src, n in resolved["source"].value_counts().items():
        print(f"    {src:>12} : {n}")
    print(f"  unresolved: {len(unresolved)}  -> {out_tsv.with_suffix('.unresolved.txt')}")
    print(f"  wrote     : {out_tsv}  (sample<TAB>assembly_path; unitig-caller --refs input)")
    return df


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sample-csv", type=Path, nargs="+", help="One or more cohort/split CSVs (unioned on 'Sample').")
    src.add_argument("--all-kpsc", action="store_true", help="Use every kpsc_final_list==True Sample (Tier-2).")
    p.add_argument("--metadata", type=Path, default=METADATA_V2_DEFAULT)
    p.add_argument("--path-root", type=Path, default=PROJECT_K_ROOT,
                   help="Prefix for relative assembly paths (project_k RDS root).")
    p.add_argument("--check-exists", action="store_true",
                   help="Drop Samples whose assembly file is missing on disk (stats each; run in a job).")
    p.add_argument("--out-tsv", type=Path, required=True, help="Output sample<TAB>assembly_path list.")
    args = p.parse_args(argv)
    resolve(args.sample_csv or [], args.all_kpsc, args.metadata, args.out_tsv,
            path_root=args.path_root, check_exists=args.check_exists)


if __name__ == "__main__":
    main()
