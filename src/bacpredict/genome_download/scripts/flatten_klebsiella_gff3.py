#!/usr/bin/env python3
"""
Flatten bakta GFF3 batch-run directories by moving result files up one level.

Starting from the top directory
    /home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/klebsiella_gff3
this script will:

- Recurse through all nested subdirectories.
- Find all files ending with `.bakta.gff3.gz`.
- Move each such file into the top directory, preserving only the basename.
- Track which directories had at least one file moved out.
- After all moves, delete only those directories that:
  - had at least one file moved from them, and
  - are now empty.

Directories that never contained a matching file are left as-is, even if empty.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tqdm import tqdm


TOP_DIR = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/raw/klebsiella_gff3"
)
PATTERN = "*.bakta.gff3.gz"


def flatten_bakta_gff3(top: Path) -> None:
    """
    Move all PATTERN-matching files from nested subdirectories into `top`.

    Only delete subdirectories that had at least one file moved and are
    now empty. The top directory itself is never removed.
    """
    if not top.is_dir():
        raise NotADirectoryError(f"Top directory does not exist or is not a directory: {top}")

    moved_from: set[Path] = set()

    # Find all matching files at any depth under `top`, then iterate with a progress bar.
    files = list(top.rglob(PATTERN))

    for file_path in tqdm(files, desc="Moving .bakta.gff3.gz files", unit="file"):
        # Skip files already in the top directory.
        if file_path.parent == top:
            continue

        subdir = file_path.parent
        dest = top / file_path.name

        if dest.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file: {dest}\n"
                f"Source that triggered this conflict: {file_path}"
            )

        shutil.move(str(file_path), str(dest))
        moved_from.add(subdir)

    # Remove only those directories we actually moved files from and that
    # are now empty. Do not touch directories that never had matching files.
    # Sort by descending path length so deeper directories are removed first.
    for subdir in sorted(moved_from, key=lambda p: len(p.parts), reverse=True):
        # Never attempt to remove the top directory itself.
        if subdir == top:
            continue

        try:
            is_empty = not any(subdir.iterdir())
        except FileNotFoundError:
            # Directory may already have been removed as a parent of another.
            continue

        if is_empty:
            subdir.rmdir()

    # Finally, remove any now-empty immediate subdirectories directly under `top`.
    for child in top.iterdir():
        if not child.is_dir():
            continue
        try:
            is_empty = not any(child.iterdir())
        except FileNotFoundError:
            continue
        if is_empty:
            child.rmdir()


def main() -> None:
    flatten_bakta_gff3(TOP_DIR)


if __name__ == "__main__":
    main()

