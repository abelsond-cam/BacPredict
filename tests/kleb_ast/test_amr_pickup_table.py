"""Tests for the CARD/Bakta per-class pickup grouping (:mod:`bacpredict.apps.kleb.amr_pickup_table`).

Exercises the pure pandas logic on a synthetic sidecar frame — no HPC sidecars or metadata needed.
"""

from __future__ import annotations

import pandas as pd

from bacpredict.apps.kleb.amr_pickup_table import _bakta_family_match, card_bakta_by_class, to_markdown


def _sidecar(rows: list[dict]) -> pd.DataFrame:
    cols = ["Sample", "flat_index", "amr_source", "amr_class", "amr_gene_family", "amr_allele",
            "bakta_gene_name"]
    return pd.DataFrame(rows, columns=cols)


def test_bakta_family_match() -> None:
    assert _bakta_family_match("blaKPC", "KPC", "kpc-2")
    assert _bakta_family_match("KPC-2", "KPC", "kpc-2")
    assert not _bakta_family_match(None, "KPC", "kpc-2")
    assert not _bakta_family_match("dnaA", "KPC", "kpc-2")


def test_card_bakta_by_class_counts_and_pickup() -> None:
    calls = _sidecar([
        # Bla_Carb: 3 calls — 2 named by Bakta, 1 orphan (no CDS) unnamed
        {"Sample": "g1", "flat_index": 10, "amr_source": "acquired", "amr_class": "Bla_Carb",
         "amr_gene_family": "KPC", "amr_allele": "kpc-2", "bakta_gene_name": "blaKPC"},
        {"Sample": "g2", "flat_index": 11, "amr_source": "acquired", "amr_class": "Bla_Carb",
         "amr_gene_family": "NDM", "amr_allele": "ndm-1", "bakta_gene_name": "blaNDM"},
        {"Sample": "g3", "flat_index": -1, "amr_source": "acquired", "amr_class": "Bla_Carb",
         "amr_gene_family": "KPC", "amr_allele": "kpc-3", "bakta_gene_name": None},
        # AGly: 1 call, Bakta did not name it
        {"Sample": "g1", "flat_index": 5, "amr_source": "acquired", "amr_class": "AGly",
         "amr_gene_family": "AAC(6')", "amr_allele": "aac(6')-Ib", "bakta_gene_name": "hypothetical"},
        # a chromosomal call must be ignored by the acquired grouping
        {"Sample": "g1", "flat_index": 7, "amr_source": "chromosomal", "amr_class": "Flq",
         "amr_gene_family": "GyrA", "amr_allele": "GyrA", "bakta_gene_name": "gyrA"},
    ])
    df = card_bakta_by_class(calls).set_index("class")

    carb = df.loc["Bla_Carb"]
    assert carb["n_card_calls"] == 3
    assert carb["n_card_on_cds"] == 2
    assert carb["n_card_orphan_no_cds"] == 1
    assert carb["n_gene_families"] == 2  # KPC, NDM
    assert carb["n_bakta_named"] == 2
    assert carb["bakta_pickup_pct"] == round(100 * 2 / 3, 1)

    agly = df.loc["AGly"]
    assert agly["n_card_calls"] == 1
    assert agly["n_bakta_named"] == 0
    assert agly["bakta_pickup_pct"] == 0.0

    assert "Flq" not in df.index  # chromosomal source excluded


def test_to_markdown_has_header_and_rows() -> None:
    calls = _sidecar([
        {"Sample": "g1", "flat_index": 1, "amr_source": "acquired", "amr_class": "Tet",
         "amr_gene_family": "TetA", "amr_allele": "tet(A)", "bakta_gene_name": "tetA"},
    ])
    df = card_bakta_by_class(calls)
    df["n_kleborate_carriers"] = None
    df["kleborate_agree_pct"] = None
    md = to_markdown(df)
    assert md.startswith("| class |")
    assert "Tet" in md
