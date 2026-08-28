r"""Convert a ``mash triangle`` distance matrix into a pyseer LMM kinship (similarity matrix).

mash distance D is an alignment-free MinHash estimate of ``1 − ANI`` over each genome's whole k-mer
content (core **and** accessory), so a kinship built from it captures the deep soft-core / cross-species
+ accessory population structure that a single-reference core-SNP kinship misses — the candidate fix for
the common-unitig genomic-inflation (λ=21 at af>0.5) in the unitig GWAS.

Reads ``mash triangle`` output (lower-triangular PHYLIP: a count line, then one genome per line with its
distances to all previous genomes), rebuilds the square symmetric distance matrix, relabels genomes to
their sample IDs (filename stem, e.g. ``…/SAMN02138595.fa.gz`` → ``SAMN02138595``), and writes a square
similarity TSV in the format ``pyseer --lmm --similarity`` expects. Default similarity is ``S = 1 − D``
(diagonal 1); ``--double-center`` instead emits the classical-MDS Gram ``−½·J·D²·J`` (a PSD kernel) if a
strictly positive-semidefinite kinship is preferred.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SUFFIXES = (".fa.gz", ".fasta.gz", ".fna.gz", ".fa", ".fasta", ".fna", ".gz")


def _sample_id(name: str) -> str:
    """Genome path/name → sample id (basename minus a FASTA suffix)."""
    b = Path(name.strip()).name
    for suf in _SUFFIXES:
        if b.endswith(suf):
            return b[: -len(suf)]
    return b


def parse_triangle(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse a ``mash triangle`` lower-triangular matrix → (sample_ids, square symmetric D).

    Each row is converted to a compact ``float64`` array as it is read, and the square matrix is
    filled one **row** at a time by slice assignment.

    The obvious implementation — accumulate a list-of-lists of Python floats, then assign ``d[i, j]``
    and ``d[j, i]`` cell by cell — is quadratic in the cohort twice over, and both costs are
    interpreter-level. A boxed float plus its list slot is ~32 B, and each cell takes two scalar
    stores. At *Klebsiella* scale (7,080 genomes, 25 M cells) that is ~0.8 GB and survivable; at TB
    scale (36,389 genomes, **662 M cells**) it is **~21 GB of boxed floats held all at once**, on top
    of the 10.6 GB matrix, and 1.3 billion scalar stores. Holding rows as arrays instead costs 8 B a
    cell with no per-cell Python object, and the fill becomes one vectorised store per row.

    Behaviour is unchanged, including the header-count warning. One difference is an improvement: a
    row whose length disagrees with its position now raises ``ValueError`` from the slice assignment
    instead of silently writing to the wrong cell or raising ``IndexError`` further along.
    """
    names: list[str] = []
    rows: list[np.ndarray] = []
    with open(path) as fh:
        first = fh.readline().split()
        n = int(first[0])
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts or parts[0] == "":
                continue
            names.append(_sample_id(parts[0]))
            rows.append(np.asarray(parts[1:], dtype=np.float64))
    if len(names) != n:
        print(f"WARNING: header said {n} genomes, parsed {len(names)}", file=sys.stderr)
    m = len(names)
    d = np.zeros((m, m), dtype=np.float64)
    for i, row in enumerate(rows):  # row i holds distances to genomes 0..i-1 (lower triangle)
        d[i, :i] = row
        d[:i, i] = row
    return names, d


def to_similarity(d: np.ndarray, double_center: bool) -> np.ndarray:
    """Distance → similarity: ``1 − D`` (default) or the classical-MDS Gram (PSD) if requested."""
    if not double_center:
        return 1.0 - d
    n = d.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * j @ (d ** 2) @ j


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--triangle", type=Path, required=True, help="mash triangle output.")
    p.add_argument("--out", type=Path, required=True, help="Square similarity TSV for pyseer --similarity.")
    p.add_argument("--double-center", action="store_true", help="Emit MDS Gram (PSD) instead of 1-D.")
    args = p.parse_args(argv)

    names, d = parse_triangle(args.triangle)
    s = to_similarity(d, args.double_center)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(s, index=names, columns=names).to_csv(args.out, sep="\t")
    print(f"wrote {args.out}: {len(names)} samples; D in [{d.min():.4f},{d.max():.4f}]; "
          f"S in [{s.min():.4f},{s.max():.4f}]", file=sys.stderr)


if __name__ == "__main__":
    main()
