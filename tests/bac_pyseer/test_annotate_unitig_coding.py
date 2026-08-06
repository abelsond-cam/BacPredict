"""Tests for the unitig CDS-vs-IGR classifier job: pair-level reduction + the combine pivot."""

from __future__ import annotations

import pandas as pd

from bac_pyseer.kleb_iso_source.annotate_unitig_coding import _overall_with_igr_total, _per_unitig, classify_pair
from genome_prep import CodingIndex

# c1: CDS 100-200, rRNA 250-350, CDS 400-500 (1-based inclusive)
GFF_LINES = [
    "c1\tProdigal\tCDS\t100\t200\t.\t+\t0\tID=a",
    "c1\tBarrnap\trRNA\t250\t350\t.\t+\t0\tID=r;gene=rrs",
    "c1\tProdigal\tCDS\t400\t500\t.\t+\t0\tID=b",
]


def _cidx(tmp_path):
    gff = tmp_path / "g.gff3"
    gff.write_text("##gff-version 3\n" + "\n".join(GFF_LINES) + "\n##FASTA\n")
    return CodingIndex.from_gff(gff)


def _endoff(one_based_end: int) -> int:
    """0-based end offset for a placement ending at ``one_based_end`` (1-based)."""
    return one_based_end - 1


def test_classify_pair_single_occurrence(tmp_path):
    idx = _cidx(tmp_path)
    # span 150-160 (len 11) fully inside CDS 100-200
    assert classify_pair([("c1", _endoff(160))], 11, idx) == "CDS"
    # span 310-320 inside rRNA 250-350
    assert classify_pair([("c1", _endoff(320))], 11, idx) == "IGR_rrna"
    # span 220-230 in the unannotated gap
    assert classify_pair([("c1", _endoff(230))], 11, idx) == "IGR_unclassified"


def test_classify_pair_tie_favours_cds(tmp_path):
    idx = _cidx(tmp_path)
    occ = [("c1", _endoff(160)), ("c1", _endoff(230))]  # one CDS, one IGR -> tie -> CDS
    assert classify_pair(occ, 11, idx) == "CDS"


def test_classify_pair_igr_majority(tmp_path):
    idx = _cidx(tmp_path)
    occ = [("c1", _endoff(230)), ("c1", _endoff(240)), ("c1", _endoff(160))]  # 2 IGR (unclassified), 1 CDS
    assert classify_pair(occ, 11, idx) == "IGR_unclassified"


def test_classify_pair_no_bakta_and_no_hit(tmp_path):
    idx = _cidx(tmp_path)
    assert classify_pair([("c1", _endoff(160))], 11, None) == "unknown_no_bakta"
    assert classify_pair([], 11, idx) == "no_asm_hit"


def test_per_unitig_pivot_and_fracs():
    cls = pd.DataFrame(
        {
            "unitig_idx": [0, 0, 1, 1],
            "rclass": ["CDS", "IGR_unclassified", "IGR_trna", "IGR_unclassified"],
            "n": [8, 2, 3, 1],
        }
    )
    id_map = pd.DataFrame(
        {"unitig_idx": [0, 1], "variant": ["A", "C"], "pattern_group": [10, 11],
         "direction": ["blood", "faeces"], "af": [0.9, 0.03]}
    )
    per, classes = _per_unitig(cls, id_map)
    assert classes == ["CDS", "IGR_trna", "IGR_unclassified"]
    u0 = per[per["unitig_idx"] == 0].iloc[0]
    assert u0["n_carriers"] == 10 and u0["frac_CDS"] == 0.8 and u0["frac_IGR"] == 0.2
    assert u0["dominant_igr_type"] == "unclassified"
    u1 = per[per["unitig_idx"] == 1].iloc[0]
    assert u1["frac_CDS"] == 0.0 and u1["frac_IGR"] == 1.0
    assert u1["frac_IGR_unclassified"] == 0.25 and u1["dominant_igr_type"] == "trna"
    # overall rollup runs and carries a summed frac_IGR
    ov = _overall_with_igr_total(per, classes)
    assert "frac_IGR" in ov.columns and (ov["stratum"] == "ALL").any()
