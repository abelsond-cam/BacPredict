"""Test the uniform-placement null slide against a hand-computable synthetic genome."""

from __future__ import annotations

import numpy as np

from bac_pyseer.kleb_iso_source.coding_null_model import genome_threshold_counts


def _gff(tmp_path):
    # contig c1, 20 bp; one CDS 1-10 (1-based inclusive) -> coding[0:10], IGR = 10..19
    g = tmp_path / "c.gff3"
    g.write_text(
        "##gff-version 3\n"
        "##sequence-region c1 1 20\n"
        "c1\tX\tCDS\t1\t10\t.\t+\t0\tID=a\n"
        "##FASTA\n"
    )
    return g


def test_slide_counts_match_analytic(tmp_path):
    acc = genome_threshold_counts(str(_gff(tmp_path)), np.array([5], dtype=np.int64))[0]
    n_starts, ent_cds, touch, _sig, _pred, ent_igr = acc
    # 20 - 5 + 1 = 16 window start positions
    assert n_starts == 16
    # entirely-CDS windows = L-mers fully inside CDS [0,10): starts 0..5 -> 6  (== analytic max(0,10-5+1))
    assert ent_cds == 6
    # entirely-IGR windows = L-mers fully inside IGR [10,20): starts 10..15 -> 6
    assert ent_igr == 6
    assert touch == 16 - 6  # everything not entirely-CDS touches IGR
