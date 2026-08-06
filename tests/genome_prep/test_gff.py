"""Tests for the genome_prep GFF/FASTA IO primitives."""

from __future__ import annotations

import gzip

from genome_prep import is_gbff_path, is_gff_path, load_fna, open_text, parse_gff_attributes


def test_path_type_detectors():
    assert is_gff_path("x.gff") and is_gff_path("x.gff3.gz") and is_gff_path("X.GFF")
    assert not is_gff_path("x.gbff")
    assert is_gbff_path("x.gbff") and is_gbff_path("x.gbff.gz")
    assert not is_gbff_path("x.gff")


def test_open_text_plain_and_gzip(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello\n")
    with open_text(p) as fh:
        assert fh.read() == "hello\n"
    g = tmp_path / "a.txt.gz"
    with gzip.open(g, "wt") as fh:
        fh.write("gz\n")
    with open_text(g) as fh:
        assert fh.read() == "gz\n"


def test_load_fna(tmp_path):
    fna = tmp_path / "g.fna"
    fna.write_text(">c1 desc\nACGT\n>c2\nTTTT\n")
    contigs = load_fna(fna)
    assert set(contigs) == {"c1", "c2"}
    assert str(contigs["c1"]) == "ACGT"


def test_parse_gff_attributes_last_value_wins():
    a = parse_gff_attributes("ID=x;gene=abc; product=some thing ;gene=def")
    assert a["ID"] == "x" and a["gene"] == "def" and a["product"] == "some thing"
