"""The duplicate-assembly screen: the fingerprint finds candidates, the sequence hash settles them."""

from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import pytest

from bacpredict.engine.genomes.find_duplicate_assemblies import (
    candidate_groups,
    confirm_groups,
    contig_hashes,
)

A = "ACGTACGTTTGCA" * 7
B = "GGCATTACGATCC" * 5
C = "TTTACGGCATACG" * 9


def _fasta(path: Path, contigs: list[str], sample: str, wrap: int = 60) -> Path:
    """A gzipped FASTA whose headers embed the sample, as the real store's do."""
    with gzip.open(path, "wt") as fh:
        for i, seq in enumerate(contigs, 1):
            fh.write(f">{sample}.contig{i:05d} orig={sample}.contig{i:05d} src=atb\n")
            for j in range(0, len(seq), wrap):
                fh.write(seq[j:j + wrap] + "\n")
    return path


def _revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _qc(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_contig_hashes_ignore_header_wrapping_and_case(tmp_path: Path) -> None:
    """The point of hashing sequence-only: two deposits differ in header and layout, never in sequence."""
    one = _fasta(tmp_path / "one.fna.gz", [A, B], "SAMN00000001", wrap=60)
    two = _fasta(tmp_path / "two.fna.gz", [A.lower(), B.lower()], "SAMN00000002", wrap=17)
    assert contig_hashes(one) == contig_hashes(two)


def test_contig_hashes_are_orientation_insensitive(tmp_path: Path) -> None:
    one = _fasta(tmp_path / "one.fna.gz", [A], "SAMN00000001")
    flipped = _fasta(tmp_path / "flipped.fna.gz", [_revcomp(A)], "SAMN00000002")
    assert contig_hashes(one) == contig_hashes(flipped)


def test_contig_hashes_differ_on_different_sequence(tmp_path: Path) -> None:
    one = _fasta(tmp_path / "one.fna.gz", [A], "SAMN00000001")
    other = _fasta(tmp_path / "other.fna.gz", [C], "SAMN00000002")
    assert contig_hashes(one) != contig_hashes(other)


def test_candidate_groups_needs_every_fingerprint_column() -> None:
    """⛔ The regression this screen exists for: n_cds separates two genomes of equal length and contig count."""
    qc = _qc([
        {"Sample": "S1", "contigs_path": "/a", "qc_pass": "True", "total_length": 100, "n_contigs": 5, "n_cds": 90},
        {"Sample": "S2", "contigs_path": "/b", "qc_pass": "True", "total_length": 100, "n_contigs": 5, "n_cds": 88},
    ])
    assert candidate_groups(qc).empty


def test_candidate_groups_finds_an_exact_triple() -> None:
    qc = _qc([
        {"Sample": "S2", "contigs_path": "/b", "qc_pass": "True", "total_length": 100, "n_contigs": 5, "n_cds": 90},
        {"Sample": "S1", "contigs_path": "/a", "qc_pass": "True", "total_length": 100, "n_contigs": 5, "n_cds": 90},
        {"Sample": "S3", "contigs_path": "/c", "qc_pass": "True", "total_length": 101, "n_contigs": 5, "n_cds": 90},
    ])
    got = candidate_groups(qc)
    assert list(got["Sample"]) == ["S1", "S2"]
    assert set(got["group"]) == {"dup01"}


def test_candidate_groups_skips_qc_failures_and_honours_a_sample_list() -> None:
    qc = _qc([
        {"Sample": "S1", "contigs_path": "/a", "qc_pass": "True", "total_length": 100, "n_contigs": 5, "n_cds": 90},
        {"Sample": "S2", "contigs_path": "/b", "qc_pass": "False", "total_length": 100, "n_contigs": 5, "n_cds": 90},
        {"Sample": "S3", "contigs_path": "/c", "qc_pass": "True", "total_length": 100, "n_contigs": 5, "n_cds": 90},
    ])
    assert list(candidate_groups(qc)["Sample"]) == ["S1", "S3"]  # S2 fails QC, so S1/S3 are the only pair
    assert candidate_groups(qc, samples=["S1"]).empty  # a unit holding only one of them has no pair


def test_candidate_groups_refuses_a_sample_outside_the_qc_table() -> None:
    qc = _qc([{"Sample": "S1", "contigs_path": "/a", "qc_pass": "True",
               "total_length": 100, "n_contigs": 5, "n_cds": 90}])
    with pytest.raises(ValueError, match="not QC-passing rows"):
        candidate_groups(qc, samples=["S1", "S404"])


def test_confirm_groups_separates_identical_from_merely_similar(tmp_path: Path) -> None:
    """A fingerprint collision is a candidate, not a finding: one pair is the same assembly, one is not."""
    same_a = _fasta(tmp_path / "same_a.fna.gz", [A, B], "SAMN00000001")
    same_b = _fasta(tmp_path / "same_b.fna.gz", [_revcomp(B), A], "SAMN00000002", wrap=23)
    diff_a = _fasta(tmp_path / "diff_a.fna.gz", [A, B], "SAMN00000003")
    diff_b = _fasta(tmp_path / "diff_b.fna.gz", [A, C], "SAMN00000004")
    qc = _qc([
        {"Sample": "S1", "contigs_path": str(same_a), "qc_pass": "True",
         "total_length": 100, "n_contigs": 2, "n_cds": 90},
        {"Sample": "S2", "contigs_path": str(same_b), "qc_pass": "True",
         "total_length": 100, "n_contigs": 2, "n_cds": 90},
        {"Sample": "S3", "contigs_path": str(diff_a), "qc_pass": "True",
         "total_length": 200, "n_contigs": 2, "n_cds": 95},
        {"Sample": "S4", "contigs_path": str(diff_b), "qc_pass": "True",
         "total_length": 200, "n_contigs": 2, "n_cds": 95},
    ])
    report = confirm_groups(candidate_groups(qc))
    by_sample = report.set_index("Sample")
    assert bool(by_sample.loc["S1", "identical"]) and bool(by_sample.loc["S2", "identical"])
    assert not bool(by_sample.loc["S3", "identical"]) and not bool(by_sample.loc["S4", "identical"])
    assert int(by_sample.loc["S3", "n_shared"]) == 1  # they share contig A only


def test_confirm_groups_handles_no_candidates() -> None:
    qc = _qc([{"Sample": "S1", "contigs_path": "/a", "qc_pass": "True",
               "total_length": 100, "n_contigs": 5, "n_cds": 90}])
    report = confirm_groups(candidate_groups(qc))
    assert report.empty
    assert "identical" in report.columns
