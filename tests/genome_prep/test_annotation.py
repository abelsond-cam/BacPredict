"""Tests for genome_prep: the moved 3-view IGR extractor, the CodingIndex classifier, coding_fraction.

The extraction cases mirror the engine's original ``test_extract_intergenic_from_gff_fna`` (behaviour
must be preserved through the move); the classifier + baseline cases are new.
"""

from __future__ import annotations

from genome_prep import CodingIndex, coding_fraction, extract_intergenic, parse_gff_features

# contig c1, 1000 bp, 1-based inclusive:
#   CDS   100-200 (+)
#   rRNA  250-350 (gene=rrs)   <- inside the non-coding gap between the two CDS
#   CDS   400-500 (+)
# only-CDS-occupies -> whole runs = [1-99], [201-399] (contains the rRNA), [501-1000]
GFF_LINES = [
    "c1\tProdigal\tCDS\t100\t200\t.\t+\t0\tID=a;gene=featA",
    "c1\tBarrnap\trRNA\t250\t350\t.\t+\t0\tID=r;gene=rrs;product=16S ribosomal RNA",
    "c1\tProdigal\tCDS\t400\t500\t.\t+\t0\tID=b;gene=featB",
]


def _write(tmp_path, lines=GFF_LINES, with_region=False):
    body = "##gff-version 3\n"
    if with_region:
        body += "##sequence-region c1 1 1000\n"
    gff = tmp_path / "g.gff3"
    gff.write_text(body + "\n".join(lines) + "\n##FASTA\n")
    fna = tmp_path / "g.fna"
    fna.write_text(">c1\n" + ("ACGT" * 250) + "\n")  # 1000 bp
    return gff, fna


# --------------------------------------------------------------------------- #
# extract_intergenic — behaviour preserved through the move                     #
# --------------------------------------------------------------------------- #
def test_whole_runs_only_cds_occupies_rna_inside(tmp_path):
    out = extract_intergenic(*_write(tmp_path), min_len=30)
    runs = list(zip(out["noncoding_start"], out["noncoding_end"], strict=True))
    assert runs == [(1, 99), (201, 399), (501, 1000)]
    assert (250, 350) not in runs  # rRNA lives inside 201-399, not a standalone run


def test_fragments_split_run_at_feature_boundaries(tmp_path):
    out = extract_intergenic(*_write(tmp_path), min_len=30)
    frags = list(zip(out["fragment_start"], out["fragment_end"], strict=True))
    assert frags == [(1, 99), (201, 249), (351, 399), (501, 1000)]


def test_feature_body_indexed_by_type_and_name(tmp_path):
    out = extract_intergenic(*_write(tmp_path), min_len=30)
    assert out["feature_name"] == ["rrs"]
    assert out["feature_type"] == ["rrna"]
    assert (out["feature_start"][0], out["feature_end"][0]) == (250, 350)
    assert len(out["feature_sequence"][0]) == 101
    assert out["feature_sequence"][0] == out["feature_sequence"][0].lower()


def test_crispr_regulatory_oric_are_feature_types(tmp_path):
    lines = GFF_LINES + [
        "c1\tPILER-CR\tCRISPR\t600\t700\t.\t+\t.\tID=cr;Name=CRISPR-1",
        "c1\tPromoter\tregulatory_region\t760\t810\t.\t+\t.\tID=reg;Name=promoterX",
        "c1\tSkew\toriC\t880\t930\t.\t+\t.\tID=o;Name=oriC",
    ]
    out = extract_intergenic(*_write(tmp_path, lines), min_len=30)
    got = dict(zip(out["feature_type"], out["feature_name"], strict=True))
    assert got["crispr"] == "CRISPR-1"
    assert got["regulatory_region"] == "promoterX"
    assert got["oric"] == "oriC"
    frags = set(zip(out["fragment_start"], out["fragment_end"], strict=True))
    assert (501, 1000) not in frags
    assert (501, 599) in frags


# --------------------------------------------------------------------------- #
# parse_gff_features                                                            #
# --------------------------------------------------------------------------- #
def test_parse_gff_features_keeps_cds_and_named_only(tmp_path):
    gff, _ = _write(tmp_path)
    feats = parse_gff_features(gff)["c1"]
    kinds = sorted((f.ftype, f.is_cds) for f in feats)
    assert kinds == [("CDS", True), ("CDS", True), ("rRNA", False)]
    rna = next(f for f in feats if not f.is_cds)
    assert rna.igr_type == "rrna" and rna.name == "rrs"


# --------------------------------------------------------------------------- #
# CodingIndex.classify_span                                                     #
# --------------------------------------------------------------------------- #
def test_classify_span_cds_igr_unclassified_and_priority(tmp_path):
    gff, _ = _write(tmp_path)
    idx = CodingIndex.from_gff(gff)
    assert idx.classify_span("c1", 150, 160) == ("CDS", None)          # inside CDS 100-200
    assert idx.classify_span("c1", 300, 320) == ("IGR", "rrna")        # inside rRNA 250-350
    assert idx.classify_span("c1", 210, 240) == ("IGR", "unclassified")  # gap, no named feature
    assert idx.classify_span("c1", 195, 260) == ("CDS", None)          # straddles CDS -> CDS wins
    assert idx.classify_span("c1", 200, 205) == ("CDS", None)          # boundary: touches CDS end 200
    assert idx.classify_span("cX", 10, 20) == ("IGR", "unclassified")  # unknown contig


def test_classify_span_boundary_of_named_feature(tmp_path):
    idx = CodingIndex.from_gff(_write(tmp_path)[0])
    assert idx.classify_span("c1", 350, 360) == ("IGR", "rrna")   # touches rRNA end 350
    assert idx.classify_span("c1", 351, 360) == ("IGR", "unclassified")  # just past it


def test_cds_overlap_bp(tmp_path):
    idx = CodingIndex.from_gff(_write(tmp_path)[0])
    assert idx.cds_overlap_bp("c1", 150, 160) == 11    # 150-160 fully inside CDS 100-200
    assert idx.cds_overlap_bp("c1", 195, 205) == 6     # 195-200 in CDS (6 bp), 201-205 in IGR
    assert idx.cds_overlap_bp("c1", 210, 240) == 0     # intergenic gap
    assert idx.cds_overlap_bp("c1", 300, 320) == 0     # inside rRNA, not CDS
    assert idx.cds_overlap_bp("c1", 100, 200) == 101   # whole CDS
    assert idx.cds_overlap_bp("cX", 10, 20) == 0       # unknown contig


def test_cds_overlap_bp_spanning_two_cds(tmp_path):
    # Two touching CDS 100-200 and 201-300 → a span across the join is fully coding (no gap).
    lines = ["c1\tX\tCDS\t100\t200\t.\t+\t0\tID=a", "c1\tX\tCDS\t201\t300\t.\t+\t0\tID=b"]
    idx = CodingIndex.from_gff(_write(tmp_path, lines)[0])
    assert idx.cds_overlap_bp("c1", 150, 250) == 101   # 150..250 all coding across the 200/201 join


# --------------------------------------------------------------------------- #
# coding_fraction                                                               #
# --------------------------------------------------------------------------- #
def test_coding_fraction_from_sequence_region(tmp_path):
    gff, _ = _write(tmp_path, with_region=True)
    cf = coding_fraction(gff)  # lengths from ##sequence-region
    assert cf["total_bp"] == 1000
    assert cf["cds_bp"] == 202          # 101 + 101
    assert cf["igr_bp"] == 798
    assert cf["named_igr_bp"] == 101    # rRNA 250-350, wholly in IGR
    assert cf["unclassified_igr_bp"] == 697
    assert cf["per_type_bp"] == {"rrna": 101}


def test_coding_fraction_from_fasta_lengths(tmp_path):
    gff, fna = _write(tmp_path)  # no ##sequence-region
    cf = coding_fraction(gff, fna)
    assert cf["total_bp"] == 1000 and cf["cds_bp"] == 202
