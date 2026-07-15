"""Unit smoke for the 2d non-coding extraction: only ``CDS`` occupies, RNA is merged into runs + indexed.

Fabricates a tiny GFF + FASTA where an rRNA sits between two CDS. Asserts the rRNA is (a) swallowed
into the maximal non-CDS run rather than splitting it, and (b) emitted on its own in the named-RNA
index — the two properties the re-embed depends on (``rrs`` locatable; IGR+RNA embedded together).
"""

from __future__ import annotations

from bacpredict.engine.embedding.extract_intergenic_from_gff_fna import extract_intergenic_from_gff_fna

# contig c1, 1-based inclusive:
#   CDS   100-200 (+)
#   rRNA  250-350 (gene=rrs)      <- sits inside the non-coding gap between the two CDS
#   CDS   400-500 (+)
# only-CDS-occupying -> non-coding runs = [1-99], [201-399] (contains the rRNA), [501-1000]
GFF_LINES = [
    "c1\tProdigal\tCDS\t100\t200\t.\t+\t0\tID=a;gene=featA",
    "c1\tBarrnap\trRNA\t250\t350\t.\t+\t0\tID=r;gene=rrs;product=16S ribosomal RNA",
    "c1\tProdigal\tCDS\t400\t500\t.\t+\t0\tID=b;gene=featB",
]


def _write(tmp_path):
    gff = tmp_path / "g.gff3"
    gff.write_text("##gff-version 3\n" + "\n".join(GFF_LINES) + "\n##FASTA\n")
    fna = tmp_path / "g.fna"
    fna.write_text(">c1\n" + ("ACGT" * 250) + "\n")  # 1000 bp
    return gff, fna


def test_only_cds_occupies_merges_rna_into_run(tmp_path):
    gff, fna = _write(tmp_path)
    out = extract_intergenic_from_gff_fna(gff, fna, min_len=30)
    runs = list(zip(out["noncoding_start"], out["noncoding_end"], strict=True))
    # Three maximal non-CDS runs; the middle one spans the whole CDS-CDS gap and contains the rRNA.
    assert runs == [(1, 99), (201, 399), (501, 1000)]
    # The rRNA is NOT a standalone run — it lives inside 201-399 (IGR+RNA embedded together).
    assert (250, 350) not in runs


def test_rna_body_indexed_by_name(tmp_path):
    gff, fna = _write(tmp_path)
    out = extract_intergenic_from_gff_fna(gff, fna, min_len=30)
    assert out["rna_gene_name"] == ["rrs"]
    assert out["rna_type"] == ["rrna"]
    assert out["rna_seqid"] == ["c1"]
    assert (out["rna_start"][0], out["rna_end"][0]) == (250, 350)
    # Its own sequence is the 101 bp body (250..350 inclusive), lowercase.
    assert len(out["rna_sequence"][0]) == 101
    assert out["rna_sequence"][0] == out["rna_sequence"][0].lower()
