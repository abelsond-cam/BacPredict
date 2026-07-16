"""Unit smoke for the three-view non-coding extraction: only ``CDS`` occupies.

Fabricates tiny GFF + FASTA fixtures and asserts the three channels the re-embed depends on:
``noncoding_*`` (whole CDS-to-CDS runs, RNA swallowed inside), ``fragment_*`` (runs split at every
named-feature boundary), and ``feature_*`` (each named non-CDS body indexed by type + name — so ``rrs``
is locatable, and CRISPR / regulatory_region / oriC are first-class feature types).
"""

from __future__ import annotations

from bacpredict.engine.embedding.extract_intergenic_from_gff_fna import extract_intergenic_from_gff_fna

# contig c1, 1000 bp, 1-based inclusive:
#   CDS   100-200 (+)
#   rRNA  250-350 (gene=rrs)      <- inside the non-coding gap between the two CDS
#   CDS   400-500 (+)
# only-CDS-occupying -> whole runs = [1-99], [201-399] (contains the rRNA), [501-1000]
GFF_LINES = [
    "c1\tProdigal\tCDS\t100\t200\t.\t+\t0\tID=a;gene=featA",
    "c1\tBarrnap\trRNA\t250\t350\t.\t+\t0\tID=r;gene=rrs;product=16S ribosomal RNA",
    "c1\tProdigal\tCDS\t400\t500\t.\t+\t0\tID=b;gene=featB",
]


def _write(tmp_path, lines=GFF_LINES):
    gff = tmp_path / "g.gff3"
    gff.write_text("##gff-version 3\n" + "\n".join(lines) + "\n##FASTA\n")
    fna = tmp_path / "g.fna"
    fna.write_text(">c1\n" + ("ACGT" * 250) + "\n")  # 1000 bp
    return gff, fna


def test_whole_runs_only_cds_occupies_rna_inside(tmp_path):
    gff, fna = _write(tmp_path)
    out = extract_intergenic_from_gff_fna(gff, fna, min_len=30)
    runs = list(zip(out["noncoding_start"], out["noncoding_end"], strict=True))
    # Three maximal non-CDS runs; the middle spans the whole CDS-CDS gap and contains the rRNA.
    assert runs == [(1, 99), (201, 399), (501, 1000)]
    assert (250, 350) not in runs  # rRNA is NOT a standalone run — it lives inside 201-399


def test_fragments_split_run_at_feature_boundaries(tmp_path):
    gff, fna = _write(tmp_path)
    out = extract_intergenic_from_gff_fna(gff, fna, min_len=30)
    frags = list(zip(out["fragment_start"], out["fragment_end"], strict=True))
    # The middle run 201-399 is split by the rRNA 250-350 -> [201-249] + [351-399]; the others are whole.
    assert frags == [(1, 99), (201, 249), (351, 399), (501, 1000)]


def test_feature_body_indexed_by_type_and_name(tmp_path):
    gff, fna = _write(tmp_path)
    out = extract_intergenic_from_gff_fna(gff, fna, min_len=30)
    assert out["feature_name"] == ["rrs"]
    assert out["feature_type"] == ["rrna"]
    assert out["feature_seqid"] == ["c1"]
    assert (out["feature_start"][0], out["feature_end"][0]) == (250, 350)
    assert len(out["feature_sequence"][0]) == 101  # 250..350 inclusive body, lowercase
    assert out["feature_sequence"][0] == out["feature_sequence"][0].lower()


def test_crispr_regulatory_oric_are_feature_types(tmp_path):
    # A run [501-1000] carrying a CRISPR array, a regulatory_region, and an oriC — all named feature types.
    lines = GFF_LINES + [
        "c1\tPILER-CR\tCRISPR\t600\t700\t.\t+\t.\tID=cr;Name=CRISPR-1",
        "c1\tPromoter\tregulatory_region\t760\t810\t.\t+\t.\tID=reg;Name=promoterX",
        "c1\tSkew\toriC\t880\t930\t.\t+\t.\tID=o;Name=oriC",
    ]
    out = extract_intergenic_from_gff_fna(*_write(tmp_path, lines), min_len=30)
    got = dict(zip(out["feature_type"], out["feature_name"], strict=True))
    assert got["crispr"] == "CRISPR-1"
    assert got["regulatory_region"] == "promoterX"
    assert got["oric"] == "oriC"
    # The CRISPR/regulatory/oriC bodies fragment the [501-1000] run, so it's no longer a single fragment.
    frags = set(zip(out["fragment_start"], out["fragment_end"], strict=True))
    assert (501, 1000) not in frags
    assert (501, 599) in frags  # fragment upstream of the CRISPR array (600-700)
