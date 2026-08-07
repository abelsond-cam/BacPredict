"""Tests for the unitig IGR-coverage job: per-placement igr_frac + the overall rollup."""

from __future__ import annotations

import pandas as pd

from bac_pyseer.kleb_iso_source.annotate_unitig_coding import _rollup, pair_igr_frac
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
    return one_based_end - 1


def test_pair_igr_frac_single_and_partial(tmp_path):
    idx = _cidx(tmp_path)
    assert pair_igr_frac([("c1", _endoff(160))], 11, idx) == 0.0            # 150-160 fully CDS
    assert pair_igr_frac([("c1", _endoff(220))], 11, idx) == 1.0            # 210-220 fully IGR (gap)
    # 195-205: 6 bp CDS (195-200), 5 bp IGR -> 5/11
    assert round(pair_igr_frac([("c1", _endoff(205))], 11, idx), 4) == round(5 / 11, 4)


def test_pair_igr_frac_multi_occurrence_mean(tmp_path):
    idx = _cidx(tmp_path)
    occ = [("c1", _endoff(160)), ("c1", _endoff(220))]  # fully CDS + fully IGR -> mean 0.5
    assert pair_igr_frac(occ, 11, idx) == 0.5


def _per_row(idx, direction, af, np_, mean, p, n):
    keys = ["entirely_cds", "touch", "significant", "predominant", "entirely_igr"]
    row = {"unitig_idx": idx, "direction": direction, "af": af, "n_pairs": np_, "mean_igr_frac": mean}
    row.update({f"p_{k}": p[i] for i, k in enumerate(keys)})
    row.update({f"n_{k}": n[i] for i, k in enumerate(keys)})
    return row


def test_rollup_unitig_and_placement_fractions():
    per = pd.DataFrame([
        _per_row(0, "blood", 0.9, 100, 0.05, [0.9, 0.1, 0.05, 0.02, 0.0], [90, 10, 5, 2, 0]),
        _per_row(1, "faeces", 0.03, 100, 0.60, [0.1, 0.9, 0.8, 0.7, 0.5], [10, 90, 80, 70, 50]),
    ])
    ov = _rollup(per)
    allrow = ov[ov["stratum"] == "ALL"].iloc[0]
    assert allrow["n_unitigs"] == 2 and allrow["n_pairs"] == 200
    # unitig-level: one of two unitigs has majority-carrier entirely_cds; one predominant_igr
    assert allrow["unitig_frac_entirely_cds"] == 0.5
    assert allrow["unitig_frac_predominant_igr"] == 0.5
    # placement-weighted
    assert allrow["placement_frac_entirely_cds"] == 0.5            # (90+10)/200
    assert allrow["placement_frac_predominant_igr"] == 0.36        # (2+70)/200
    # direction split present
    assert set(ov["stratum"]) >= {"ALL", "direction=blood", "direction=faeces"}
    blood = ov[ov["stratum"] == "direction=blood"].iloc[0]
    assert blood["placement_frac_entirely_cds"] == 0.9            # 90/100
