#!/usr/bin/env python3
"""
Flatten NCBI GFF3 batch-run directories by moving result files up one level.

Starting from the top directory
    /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/ncbi_gff3
this script will:

- Look at each immediate child directory (sample) under the top directory.
- For each sample directory S, look for:
    ncbi_gff_dir/S/ncbi_dataset/data/S/genomic.gff
- If that file exists:
    - Move it to:
        ncbi_gff_dir/S.gff
    - Then remove the entire sample directory:
        ncbi_gff_dir/S
- If that file does not exist:
    - Leave the sample directory (including empty ones) as-is, to mark failures.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tqdm import tqdm


NCBI_GFF_DIR = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/ncbi_gff3"
)

NCBI_DATASET_DIR = "ncbi_dataset"
NCBI_DATA_SUBDIR = "data"
GFF_NAME = "genomic.gff"


def flatten_ncbi_gff3(ncbi_gff_dir: Path) -> None:
    """
    Move per-sample genomic.gff files to top level and remove their sample dirs.

    For each immediate child directory S under `ncbi_gff_dir`, if
    ncbi_gff_dir/S/ncbi_dataset/data/S/genomic.gff exists, move it to
    ncbi_gff_dir/S.gff and then remove the entire directory ncbi_gff_dir/S.

    If that expected genomic.gff path does not exist, the sample directory
    is left untouched (even if empty) as a marker of a failed or incomplete
    download.
    """
    if not ncbi_gff_dir.is_dir():
        raise NotADirectoryError(
            f"Top directory does not exist or is not a directory: {ncbi_gff_dir}"
        )

    sample_dirs = [p for p in ncbi_gff_dir.iterdir() if p.is_dir()]

    moved_count = 0
    missing_count = 0

    for sample_dir in tqdm(
        sample_dirs, desc="Processing NCBI sample dirs", unit="sample"
    ):
        sample_name = sample_dir.name

        gff_path = (
            sample_dir
            / NCBI_DATASET_DIR
            / NCBI_DATA_SUBDIR
            / sample_name
            / GFF_NAME
        )

        if gff_path.is_file():
            dest = ncbi_gff_dir / f"{sample_name}.gff"

            if dest.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing file: {dest}\n"
                    f"Source that triggered this conflict: {gff_path}"
                )

            shutil.move(str(gff_path), str(dest))
            shutil.rmtree(sample_dir)
            moved_count += 1
        else:
            missing_count += 1

    print(
        f"Moved {moved_count} genomic.gff files to top level as <sample>.gff; "
        f"left {missing_count} sample directories without genomic.gff "
        f"(likely failed or incomplete downloads)."
    )


def main() -> None:
    flatten_ncbi_gff3(NCBI_GFF_DIR)


if __name__ == "__main__":
    main()

