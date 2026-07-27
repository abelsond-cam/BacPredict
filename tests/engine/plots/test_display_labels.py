"""Unit tests for the shared plot label helpers (drug display + non-coding region labels)."""
from __future__ import annotations

from bacpredict.engine.plots.display_labels import display_name, region_label


def test_display_name_maps_rifampin_else_passthrough():
    assert display_name("rifampin") == "rifampicin"
    assert display_name("ethionamide") == "ethionamide"


def test_region_label_promoter_maps_anchor_to_determinant():
    # the mabA(fabG1)-inhA operon promoter is anchored 5′ of fabG1 → reads as the inhA promoter
    assert region_label("upstream:fabg1") == "inhA promoter"
    # an anchor with no determinant mapping keeps its own name
    assert region_label("upstream:eis") == "eis promoter"


def test_region_label_rna_body_and_other_units():
    assert region_label("rrna:rrs") == "rrs rRNA"
    assert region_label("rrna:rrl") == "rrl rRNA"
    # a non-RNA per-unit key (oric:description) collapses to its key
    assert region_label("oric:origin of replication") == "oric"


def test_region_label_convergent_and_bare_passthrough():
    assert region_label("between:mura→ogt") == "mura→ogt"  # convergent flank-pair fallback
    assert region_label("mura→ogt") == "mura→ogt"          # already-merged pair
    assert region_label("rpoB") == "rpoB"                  # a bare coding gene
