"""
Add sr_assembly_file and sr_gff_file columns to metadata TSV.

Reads the three .txt path lists (assemblies, ncbi_gff, klebsiella_gff), parses
Sample from each path with GC normalization, writes TSVs, then loads metadata
and adds sr_assembly_file / sr_gff_file via dict lookup. Run after building the .txt
files with: bash slurm_scripts/build_assemblies_and_gff_file_list.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/")

METADATA_F = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_v2_all_samples_and_columns.tsv"
)

ASSEMBLY_LIST_F = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/assemblies_file_list.txt"
)
NCBI_GFF_LIST_F = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/ncbi_gff.txt"
)
KLEBSIELLA_GFF_LIST_F = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/klebsiella_gff.txt"
)

ASSEMBLY_TSV_F = ASSEMBLY_LIST_F.with_suffix(".tsv")
NCBI_GFF_TSV_F = NCBI_GFF_LIST_F.with_suffix(".tsv")
KLEBSIELLA_GFF_TSV_F = KLEBSIELLA_GFF_LIST_F.with_suffix(".tsv")


def _normalize_sample_for_lookup(s: str) -> str:
    """If Sample starts with GC, return first two underscore-separated parts; else return as-is."""
    if not isinstance(s, str):
        s = str(s)
    if s.startswith("GC"):
        parts = s.split("_")
        return "_".join(parts[:2]) if len(parts) >= 2 else s
    return s


def _gc_normalize_series(s: pd.Series) -> pd.Series:
    """Vectorised GC normalization: if starts with GC, take first two parts."""
    mask = s.astype(str).str.startswith("GC")
    parts = s.astype(str).str.split("_")
    first_two = parts.apply(lambda x: "_".join(x[:2]) if len(x) >= 2 else (x[0] if x else ""))
    return s.where(~mask, first_two)


def _parse_assemblies(path_series: pd.Series) -> pd.Series:
    """
    Sample = basename with common FASTA+compression suffixes removed.

    Assemblies are now stored as flat .fa.gz / .fna.gz (always two extensions),
    so we strip these composite suffixes first; if that doesn't match, fall
    back to stripping only the last extension.
    """
    basename = path_series.str.rsplit("/", n=1).str[-1]
    sample = basename.str.replace(".fa.gz", "", regex=False).str.replace(
        ".fna.gz", "", regex=False
    )
    # Return all extensions, but normalise by GC starting names while doing so
    return _gc_normalize_series(sample)


def _parse_klebsiella_gff(path_series: pd.Series) -> pd.Series:
    """Sample = basename with .bakta.gff3.gz removed."""
    basename = path_series.str.rsplit("/", n=1).str[-1]
    sample = basename.str.removesuffix(".bakta.gff3.gz")
    # Return all extensions, but normalise by GC starting names while doing so
    return _gc_normalize_series(sample)


def _parse_ncbi_gff(path_series: pd.Series) -> pd.Series:
    """Sample = basename, everything before first .gff."""
    basename = path_series.str.rsplit("/", n=1).str[-1]
    sample = basename.str.split(".gff").str[0]
    # Return all extensions, but normalise by GC starting names while doing so
    return _gc_normalize_series(sample)


def _load_and_parse_txt(
    path: Path,
    parse_fn,
    label: str,
    out_tsv: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load .txt (one path per line), parse Sample, report lines/duplicates, write TSV, return (df, sample->path dict)."""
    if not path.exists():
        raise FileNotFoundError(f"{label} list file not found: {path}")
    df = pd.read_csv(path, header=None, names=["path"])
    df["path"] = df["path"].astype(str).str.strip()
    df = df[df["path"].str.len() > 0]
    n_lines = len(df)
    if n_lines == 0:
        raise ValueError(f"{label} list file is empty: {path}")
    df["Sample"] = parse_fn(df["path"])
    before = len(df)
    df = df.drop_duplicates(subset=["Sample"], keep="first")  # type: ignore[arg-type]
    n_unique = len(df)
    n_dropped = before - n_unique
    print(f"\n{label}:")
    print(f"  Lines read: {n_lines}")
    print(f"  Duplicates dropped: {n_dropped}")
    print(f"  Unique Samples: {n_unique}")
    out_df = df[["Sample", "path"]].copy()
    out_df.to_csv(out_tsv, sep="\t", index=False)
    print(f"  Wrote {out_tsv}")
    d: dict[str, str] = out_df.set_index("Sample")["path"].astype(str).to_dict()
    return (pd.DataFrame(out_df), d)


def _summarise_matches(total_files: int, used_count: int, label: str) -> None:
    """Print summary of how many files in the list were matched to at least one sample."""
    if total_files == 0:
        all_matched_str = "N/A (directory empty)"
    else:
        all_matched_str = str(used_count == total_files)
    print(
        f"{label}: total_files={total_files}, "
        f"matched_at_least_once={used_count}, "
        f"all_files_matched={all_matched_str}"
    )


def run(metadata_path: Path | None = None) -> None:
    """Load .txt path lists, parse Sample, write TSVs, add paths to metadata, overwrite file."""
    meta_path = Path(metadata_path) if metadata_path is not None else METADATA_F

    print(f"Metadata file: {meta_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {meta_path}")
    print("  File exists: Yes")

    print("\nParsing path lists and writing TSVs...")
    _, assembly_dict = _load_and_parse_txt(
        ASSEMBLY_LIST_F,
        _parse_assemblies,
        "Assemblies",
        ASSEMBLY_TSV_F,
    )
    _, ncbi_dict = _load_and_parse_txt(
        NCBI_GFF_LIST_F,
        _parse_ncbi_gff,
        "NCBI GFF",
        NCBI_GFF_TSV_F,
    )
    _, kleb_dict = _load_and_parse_txt(
        KLEBSIELLA_GFF_LIST_F,
        _parse_klebsiella_gff,
        "Klebsiella GFF",
        KLEBSIELLA_GFF_TSV_F,
    )

    print("\n" + "=" * 80)
    print("Loading metadata...")
    df = pd.read_csv(meta_path, sep="\t", low_memory=False)
    print(f"Loaded {len(df)} samples from metadata\n")

    df["sample_key"] = df["Sample"].apply(_normalize_sample_for_lookup)
    df["sr_assembly_file"] = df["sample_key"].map(assembly_dict)  # type: ignore[arg-type]

    total_samples = len(df)
    n_assembly_found = df["sr_assembly_file"].notna().sum()
    n_assembly_not_found = total_samples - n_assembly_found
    used_assembly = int(df["sr_assembly_file"].dropna().nunique())

    print("Assembly files:")
    print(f"  Samples in metadata: {total_samples}")
    print(f"  Samples with sr_assembly_file: {n_assembly_found}")
    print(f"  Samples without sr_assembly_file: {n_assembly_not_found}")
    if "kpsc_final_list" in df.columns:
        kpsc_mask = df["kpsc_final_list"]
        n_kpsc_no_asm = int(df.loc[kpsc_mask, "sr_assembly_file"].isna().sum())
        print(
            f"    Metadata samples in kpsc_final_list without assemblies: {n_kpsc_no_asm}"
        )
    _summarise_matches(len(assembly_dict), used_assembly, "  Assembly list coverage")

    search_ncbi = df["is_refseq"].astype(bool) | df["is_nctc"].astype(bool)
    df["sr_gff_file"] = np.where(
        search_ncbi,
        df["sample_key"].map(ncbi_dict),  # type: ignore[arg-type]
        df["sample_key"].map(kleb_dict),  # type: ignore[arg-type]
    )

    n_gff_found = df["sr_gff_file"].notna().sum()
    n_gff_not_found = total_samples - n_gff_found
    used_ncbi = int(df.loc[search_ncbi, "sr_gff_file"].dropna().nunique())
    used_kleb = int(df.loc[~search_ncbi, "sr_gff_file"].dropna().nunique())

    print("\nGFF files:")
    print(f"  Samples in metadata: {total_samples}")
    print(f"  Samples with sr_gff_file: {n_gff_found}")
    print(f"  Samples without sr_gff_file: {n_gff_not_found}")
    _summarise_matches(len(ncbi_dict), used_ncbi, "  ncbi_gff3 list coverage")
    _summarise_matches(len(kleb_dict), used_kleb, "  klebsiella_gff3 list coverage")

    df = df.drop(columns=["sample_key"])
    df.to_csv(meta_path, sep="\t", index=False)

    print("\n" + "=" * 80)
    print(f"Writing updated metadata to: {meta_path}")
    print("Done!")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point. Usage: python -m kleb_ast.add_paths_gff_fna_to_metadata [optional_metadata_tsv]."""
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) > 1:
        raise SystemExit(
            "Usage: python -m kleb_ast.add_paths_gff_fna_to_metadata [optional_metadata_tsv]"
        )
    metadata_path = Path(argv[0]) if argv else None
    run(metadata_path)


if __name__ == "__main__":
    main()
