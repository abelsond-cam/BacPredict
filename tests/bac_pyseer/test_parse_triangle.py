"""The vectorised ``parse_triangle`` must equal the one it replaced, bit for bit.

``parse_triangle`` is shared engine: the AMR GWAS, the invasion GWAS and every per-drug kinship
subset go through it. It was rewritten because its original form held ~32 B of boxed float per
triangle cell and did two interpreter-level stores per cell — survivable at *Klebsiella* scale
(25 M cells) and roughly 21 GB of boxed floats at TB scale (662 M cells).

A faster parser that returns *different numbers* would silently move every published λ, β and p.
So the reference implementation is kept here, in the test, and the two are compared with
``array_equal`` — exact equality, not ``allclose``. Distances are parsed, never computed, so there is
no floating-point reordering to excuse a tolerance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from bac_pyseer.kleb_iso_source.mash_dist_to_kinship import parse_triangle


def _reference_parse(path: Path) -> tuple[list[str], np.ndarray]:
    """The pre-2026-08-28 implementation, preserved verbatim as the oracle."""
    from bac_pyseer.kleb_iso_source.mash_dist_to_kinship import _sample_id

    names: list[str] = []
    dists: list[list[float]] = []
    with open(path) as fh:
        first = fh.readline().split()
        n = int(first[0])
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if not parts or parts[0] == "":
                continue
            names.append(_sample_id(parts[0]))
            dists.append([float(x) for x in parts[1:]])
    if len(names) != n:
        print(f"WARNING: header said {n} genomes, parsed {len(names)}", file=sys.stderr)
    m = len(names)
    d = np.zeros((m, m), dtype=np.float64)
    for i, row in enumerate(dists):
        for j, v in enumerate(row):
            d[i, j] = v
            d[j, i] = v
    return names, d


def _write_triangle(path: Path, names: list[str], d: np.ndarray, *, header: int | None = None) -> Path:
    """Write lower-triangular PHYLIP exactly as ``mash triangle`` does."""
    lines = [f"\t{len(names) if header is None else header}"]
    for i, name in enumerate(names):
        lines.append("\t".join([name, *(f"{d[i, j]:.6f}" for j in range(i))]))
    path.write_text("\n".join(lines) + "\n")
    return path


def _random_triangle(path: Path, m: int, seed: int, *, suffix: str = "") -> Path:
    rng = np.random.default_rng(seed)
    a = rng.random((m, m))
    d = np.tril(a, -1)
    d = d + d.T
    names = [f"SAMEA{seed}{i:05d}{suffix}" for i in range(m)]
    return _write_triangle(path, names, d)


@pytest.mark.parametrize("m", [1, 2, 3, 7, 40])
def test_matches_the_reference_implementation_exactly(tmp_path: Path, m: int) -> None:
    """Across sizes, including the degenerate 1-genome triangle whose only row has no values."""
    path = _random_triangle(tmp_path / "t.txt", m, seed=m)
    got_names, got_d = parse_triangle(path)
    ref_names, ref_d = _reference_parse(path)
    assert got_names == ref_names
    assert np.array_equal(got_d, ref_d), "vectorised parser diverged from the reference"
    assert got_d.dtype == ref_d.dtype == np.float64


def test_the_matrix_is_symmetric_with_a_zero_diagonal(tmp_path: Path) -> None:
    """Mirroring is what the row-wise fill replaced; assert the property, not just the equality."""
    _, d = parse_triangle(_random_triangle(tmp_path / "t.txt", 12, seed=1))
    assert np.array_equal(d, d.T)
    assert np.array_equal(np.diag(d), np.zeros(12))


def test_fasta_suffixes_are_stripped_from_names(tmp_path: Path) -> None:
    """The sample id is the basename minus a FASTA suffix — how every downstream join keys."""
    d = np.array([[0.0, 0.5], [0.5, 0.0]])
    path = _write_triangle(tmp_path / "t.txt", ["/data/SAMEA1.fa.gz", "/data/SAMEA2.fna"], d)
    names, _ = parse_triangle(path)
    assert names == ["SAMEA1", "SAMEA2"]


def test_a_header_count_mismatch_warns_but_still_parses(tmp_path: Path, capsys) -> None:
    """A truncated triangle must not be silently accepted as a smaller cohort."""
    d = np.tril(np.random.default_rng(3).random((4, 4)), -1)
    d = d + d.T
    path = _write_triangle(tmp_path / "t.txt", [f"S{i}" for i in range(4)], d, header=9)
    names, got = parse_triangle(path)
    ref_names, ref = _reference_parse(path)
    assert len(names) == 4
    assert np.array_equal(got, ref)
    assert "header said 9 genomes, parsed 4" in capsys.readouterr().err


def test_blank_trailing_lines_are_skipped(tmp_path: Path) -> None:
    d = np.array([[0.0, 0.25], [0.25, 0.0]])
    path = _write_triangle(tmp_path / "t.txt", ["A", "B"], d)
    path.write_text(path.read_text() + "\n\n")
    names, got = parse_triangle(path)
    assert names == ["A", "B"]
    assert np.array_equal(got, _reference_parse(path)[1])


def test_a_row_of_the_wrong_length_is_rejected(tmp_path: Path) -> None:
    """The reference wrote such a row to the wrong cells; the slice assignment refuses it.

    This is the one deliberate behaviour change, and it turns silent corruption into a crash.
    """
    path = tmp_path / "t.txt"
    path.write_text("\t3\nA\nB\t0.1\nC\t0.2\t0.3\t0.4\n")
    with pytest.raises(ValueError):
        parse_triangle(path)
