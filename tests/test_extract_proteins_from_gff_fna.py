"""Tests for the GFF + FNA protein extractor."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from predict_kleb_by_bacformer.pp.extract_proteins_from_gff_fna import (
    extract_proteins_from_gff_fna,
    is_gbff_path,
    is_gff_path,
)

# A minimal two-contig synthetic genome:
# contig_1: forward CDS encoding "M*" + a "+" strand CDS encoding "MK*"
# contig_2: a "-" strand CDS whose reverse-complement encodes "MR*"
#
# Codon table 11 maps:
#   ATG -> M, AAA -> K, CGT -> R, TAA -> *
#
# contig_1 sequence: ATG AAA TAA | ATG TAA  (positions 1-9 + 10-15)
#   - CDS1 at 1..9 (+) -> M K * -> "MK"
#   - CDS2 at 10..15 (+) -> M * -> "M"
# contig_2 sequence: TTA ACG CAT  (positions 1-9)
#   - CDS3 at 1..9 (-) on the reverse complement: ATG CGT TAA -> M R * -> "MR"

CONTIG_1 = "ATGAAATAAATGTAA"
CONTIG_2 = "TTAACGCAT"

GFF_BODY = (
    "##gff-version 3\n"
    "contig_1\tprodigal\tCDS\t1\t9\t.\t+\t0\tID=cds-1;locus_tag=GENE_1;gene=lacZ\n"
    "contig_1\tprodigal\tCDS\t10\t15\t.\t+\t0\tID=cds-2;locus_tag=GENE_2;pseudo=true\n"
    "contig_2\tprodigal\tCDS\t1\t9\t.\t-\t0\tID=cds-3;locus_tag=GENE_3;protein_id=WP_1\n"
)

FNA_BODY = f">contig_1\n{CONTIG_1}\n>contig_2\n{CONTIG_2}\n"


@pytest.fixture
def gff_fna_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Write a plain-text GFF + FNA pair to a temp dir."""
    gff = tmp_path / "sample.gff"
    fna = tmp_path / "sample.fna"
    gff.write_text(GFF_BODY)
    fna.write_text(FNA_BODY)
    return gff, fna


@pytest.fixture
def gff_fna_pair_gz(tmp_path: Path) -> tuple[Path, Path]:
    """Write a gzipped GFF3 + FNA pair to a temp dir."""
    gff = tmp_path / "sample.gff3.gz"
    fna = tmp_path / "sample.fna.gz"
    with gzip.open(gff, "wt") as fh:
        fh.write(GFF_BODY)
    with gzip.open(fna, "wt") as fh:
        fh.write(FNA_BODY)
    return gff, fna


def test_extract_proteins_basic(gff_fna_pair: tuple[Path, Path]) -> None:
    """Forward + reverse-strand CDSs are translated and pseudogene is skipped."""
    gff, fna = gff_fna_pair
    out = extract_proteins_from_gff_fna(gff, fna)

    # Two contigs (both have at least one non-pseudo CDS)
    assert out["contig_idx"] == [0, 1]
    # Protein sequences are list-of-lists, one inner list per contig.
    assert out["protein_sequence"] == [["MK"], ["MR"]]
    # Pseudogene on contig_1 was excluded — gene_name list reflects that.
    assert out["gene_name"] == [["lacZ"], ["GENE_3"]]
    assert out["protein_id"] == [[None], ["WP_1"]]


def test_extract_proteins_gzipped(gff_fna_pair_gz: tuple[Path, Path]) -> None:
    """Gzipped inputs produce the same output as plain text."""
    gff, fna = gff_fna_pair_gz
    out = extract_proteins_from_gff_fna(gff, fna)
    assert out["protein_sequence"] == [["MK"], ["MR"]]


def test_extract_proteins_missing_contig(tmp_path: Path) -> None:
    """A CDS pointing at an absent contig is skipped, not crashed on."""
    gff = tmp_path / "sample.gff"
    fna = tmp_path / "sample.fna"
    gff.write_text(
        "##gff-version 3\n"
        "contig_1\tprodigal\tCDS\t1\t9\t.\t+\t0\tID=cds-1;locus_tag=G1\n"
        "missing_contig\tprodigal\tCDS\t1\t9\t.\t+\t0\tID=cds-x;locus_tag=GX\n"
    )
    fna.write_text(f">contig_1\n{CONTIG_1}\n")
    out = extract_proteins_from_gff_fna(gff, fna)
    assert out["protein_sequence"] == [["MK"]]


def test_path_extension_helpers() -> None:
    """Extension detectors handle the relevant suffix variants."""
    assert is_gff_path("a/b.gff")
    assert is_gff_path("a/b.gff3")
    assert is_gff_path("a/b.gff.gz")
    assert is_gff_path("a/b.gff3.gz")
    assert not is_gff_path("a/b.gbff.gz")
    assert is_gbff_path("a/b.gbff")
    assert is_gbff_path("a/b.gbff.gz")
    assert not is_gbff_path("a/b.gff")
