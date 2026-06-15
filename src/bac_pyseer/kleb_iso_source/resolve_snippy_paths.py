"""Resolve each cohort ``Sample`` to its per-sample snippy VCF on RDS.

The blood-vs-faeces GWAS cohort is keyed by metadata_v2 ``Sample`` (a biosample
``SAMN*/SAME*/SAMD*`` for short-read rows, or a versioned ``GCF_/GCA_`` accession
for long-read rows). The variant calls, however, live in two trees keyed
differently:

- ``klebsiella/phylogeny/snippy/<RUN>_snippy/snps.raw.vcf.gz`` — keyed by SRA **run**
  accession (84,549 folders). A biosample reaches its run via metadata_v2's
  ``sr_run_accession`` (falling back to ``run_accession_used`` / ``run_accession``).
- ``klebsiella/phylogeny/snippy_ncbi/<GCF_/GCA_acc>/snps.raw.vcf`` — keyed by the same
  versioned assembly accession used as the long-read ``Sample`` (3,620 folders).

This module mirrors the vectorised, single-filesystem-pass pattern of
``BacHGT/src/bac_metadata/pp/add_paths_gff_fna_to_metadata.py``: it reads the pre-made
``all_snippy_dirs.txt`` listing (one traversal, already on disk) and one ``scandir`` of
``snippy_ncbi/`` to build ``key -> raw-vcf-path`` dicts, then resolves every ``Sample``
with a single vectorised ``map`` — **never** an ``ls``/``stat`` per sample.

Both sources point at the **raw** VCF on purpose: the user chose a single, uniform
re-filter from raw across every sample (see this folder's CLAUDE.md), so the native
filtered ``snps.vcf`` retained by ``snippy_ncbi`` is intentionally bypassed here (it is
used only as filter-fidelity ground truth elsewhere).

Output: a resolution TSV (``Sample``, ``source``, ``run_accession``, ``vcf_path``) for
the rows that resolved, a sidecar list of unresolved samples, and per-source coverage
counts. The resolution TSV is the work-list the extraction array chunks over.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PHYLOGENY_ROOT_DEFAULT = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/klebsiella/phylogeny")
METADATA_V2_DEFAULT = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_v2_all_samples_and_columns.tsv"
)

# Metadata_v2 columns needed for resolution (read with usecols to keep memory modest).
_META_COLS = [
    "Sample",
    "sr_run_accession",
    "run_accession_used",
    "run_accession",
    "is_variant_called",
    "kpsc_final_list",
]
# Preference order for the short-read run accession that keys snippy/<RUN>_snippy.
_RUN_COLS = ["sr_run_accession", "run_accession_used", "run_accession"]


def _build_sr_run_to_vcf(snippy_dirs_list: Path, phylogeny_root: Path) -> dict[str, str]:
    """Map SRA run accession -> ``snps.raw.vcf.gz`` path from the pre-made dir listing.

    Parameters
    ----------
    snippy_dirs_list
        Path to ``all_snippy_dirs.txt`` (one ``./snippy/<RUN>_snippy`` path per line).
    phylogeny_root
        Absolute root the relative listing paths hang off.

    Returns
    -------
    dict
        ``run_accession -> absolute snps.raw.vcf.gz path``.
    """
    df = pd.read_csv(snippy_dirs_list, header=None, names=["relpath"])
    rel = df["relpath"].astype(str).str.strip()
    rel = rel[rel.str.len() > 0].str.removeprefix("./")
    base = rel.str.rsplit("/", n=1).str[-1]
    run = base.str.removesuffix("_snippy")
    vcf = str(phylogeny_root) + "/" + rel + "/snps.raw.vcf.gz"
    return dict(zip(run, vcf, strict=True))


def _build_ncbi_acc_to_vcf(phylogeny_root: Path) -> dict[str, str]:
    """Map versioned ``GCF_/GCA_`` accession -> ``snps.raw.vcf`` path via one scandir.

    The ``snippy_ncbi`` tree holds only ~3,620 folders, so a single ``scandir`` is cheap
    (unlike the 84k ``snippy/`` tree, which is read from the pre-made listing instead).
    """
    ncbi_root = phylogeny_root / "snippy_ncbi"
    out: dict[str, str] = {}
    with os.scandir(ncbi_root) as it:
        for entry in it:
            if entry.is_dir():
                out[entry.name] = str(Path(entry.path) / "snps.raw.vcf")
    return out


def _load_sample_list(sample_csv: Path | None, metadata: pd.DataFrame, all_kpsc: bool) -> pd.Series:
    """Return the ordered, de-duplicated ``Sample`` work-list.

    Either the ``Sample`` column of a cohort/split CSV (Tier-1 blood/faeces union), or —
    with ``all_kpsc`` — every ``kpsc_final_list == True`` row of metadata_v2 (Tier-2).
    """
    if all_kpsc:
        mask = metadata["kpsc_final_list"].fillna(False).astype(bool)
        samples = metadata.loc[mask, "Sample"]
    else:
        if sample_csv is None:
            raise ValueError("Provide --sample-csv or pass --all-kpsc.")
        samples = pd.read_csv(sample_csv, usecols=["Sample"])["Sample"]
    return samples.astype(str).drop_duplicates().reset_index(drop=True)


def resolve(
    *,
    sample_csv: Path | None,
    all_kpsc: bool,
    metadata_path: Path,
    phylogeny_root: Path,
    snippy_dirs_list: Path,
    out_tsv: Path,
) -> pd.DataFrame:
    """Resolve every cohort ``Sample`` to a raw-VCF path; write the resolution TSV.

    Returns the full resolution frame (incl. unresolved rows) for programmatic use.
    """
    print(f"Loading metadata (usecols): {metadata_path}")
    metadata = pd.read_csv(metadata_path, sep="\t", usecols=_META_COLS, low_memory=False)
    metadata["Sample"] = metadata["Sample"].astype(str)

    samples = _load_sample_list(sample_csv, metadata, all_kpsc)
    df = pd.DataFrame({"Sample": samples})
    print(f"Work-list: {len(df)} unique Samples ({'all-kpsc' if all_kpsc else sample_csv})")

    print("Building key -> raw-VCF dicts (one pass each)...")
    sr_run_to_vcf = _build_sr_run_to_vcf(snippy_dirs_list, phylogeny_root)
    ncbi_acc_to_vcf = _build_ncbi_acc_to_vcf(phylogeny_root)
    print(f"  snippy/ runs indexed: {len(sr_run_to_vcf)}")
    print(f"  snippy_ncbi/ accessions indexed: {len(ncbi_acc_to_vcf)}")

    # Classify by Sample prefix: GCF_/GCA_ are long-read rows -> snippy_ncbi.
    is_lr = df["Sample"].str.startswith(("GCF_", "GCA_"))

    # Pick the short-read run accession (preference order) for biosample rows.
    meta_runs = metadata.set_index("Sample")[_RUN_COLS]
    run_pick = meta_runs.bfill(axis=1).iloc[:, 0]  # first non-null across the preference order
    df["run_accession"] = df["Sample"].map(run_pick).where(~is_lr, other=pd.NA)

    # Resolve VCF paths: LR via accession dict, SR via run dict.
    vcf_lr = df["Sample"].map(ncbi_acc_to_vcf)
    vcf_sr = df["run_accession"].map(sr_run_to_vcf)
    df["vcf_path"] = vcf_lr.where(is_lr, vcf_sr)

    df["source"] = "unresolved"
    df.loc[is_lr & df["vcf_path"].notna(), "source"] = "snippy_ncbi"
    df.loc[~is_lr & df["vcf_path"].notna(), "source"] = "snippy_sr"

    resolved = df[df["source"] != "unresolved"].copy()
    unresolved = df[df["source"] == "unresolved"].copy()

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    resolved[["Sample", "source", "run_accession", "vcf_path"]].to_csv(out_tsv, sep="\t", index=False)
    unresolved_path = out_tsv.with_suffix(".unresolved.txt")
    unresolved["Sample"].to_csv(unresolved_path, index=False, header=False)

    print("\n=== resolution summary ===")
    print(f"  requested : {len(df)}")
    print(f"  resolved  : {len(resolved)}")
    for src, n in resolved["source"].value_counts().items():
        print(f"    {src:>12} : {n}")
    print(f"  unresolved: {len(unresolved)}  -> {unresolved_path}")
    print(f"  wrote     : {out_tsv}")
    return df


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--sample-csv", type=Path, help="Cohort/split CSV with a 'Sample' column (Tier-1 union).")
    src.add_argument("--all-kpsc", action="store_true", help="Use every kpsc_final_list==True Sample (Tier-2).")
    parser.add_argument("--metadata", type=Path, default=METADATA_V2_DEFAULT)
    parser.add_argument("--phylogeny-root", type=Path, default=PHYLOGENY_ROOT_DEFAULT)
    parser.add_argument(
        "--snippy-dirs-list",
        type=Path,
        default=None,
        help="all_snippy_dirs.txt (default: <phylogeny-root>/all_snippy_dirs.txt).",
    )
    parser.add_argument("--out-tsv", type=Path, required=True, help="Output resolution TSV path.")
    args = parser.parse_args(argv)

    snippy_dirs_list = args.snippy_dirs_list or (args.phylogeny_root / "all_snippy_dirs.txt")
    resolve(
        sample_csv=args.sample_csv,
        all_kpsc=args.all_kpsc,
        metadata_path=args.metadata,
        phylogeny_root=args.phylogeny_root,
        snippy_dirs_list=snippy_dirs_list,
        out_tsv=args.out_tsv,
    )


if __name__ == "__main__":
    main()
