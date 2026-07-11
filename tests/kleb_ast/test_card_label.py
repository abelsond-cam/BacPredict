"""Tests for the CARD label-migration core (:mod:`kleb_ast.card_label`).

The causal-lookup cases pin the plan's verification contract (ciprofloxacin / meropenem / colistin /
trimethoprim-sulfamethoxazole). They read the vendored ``CARD_AMR_clustered.csv``; the whole module is
skipped where those refs are not checked out (e.g. a clean CI box without the sibling BacHGT repo).
"""

from __future__ import annotations

import pandas as pd
import pytest

from kleb_ast.card_label import (
    _DRUG_CAUSAL,
    _DRUG_DETERMINANT,
    DEFAULT_CARD_CSV,
    causal_genes_for_drug,
    determinant_genes_for_drug,
    merged_label,
)

pytestmark = pytest.mark.skipif(
    not DEFAULT_CARD_CSV.exists(), reason=f"vendored CARD table not present at {DEFAULT_CARD_CSV}"
)


# --------------------------------------------------------------------------- #
# causal_genes_for_drug — the verification contract                            #
# --------------------------------------------------------------------------- #
def test_causal_ciprofloxacin_family() -> None:
    """Cipro (family): GyrA/ParC chromosomal + at least one acquired Qnr family; not the AGly family."""
    causal = causal_genes_for_drug("ciprofloxacin", grain="family")
    assert {"GyrA", "ParC"} <= causal
    assert any(g.lower().startswith("qnr") for g in causal)
    # aac(6')-Ib-cr is precise only at allele grain — its AGly family must NOT be hatched as FQ-causal.
    assert "AAC(6')" not in causal


def test_causal_ciprofloxacin_allele_adds_cr_variant() -> None:
    """Cipro (allele): the aac(6')-Ib-cr cross-resistance allele(s) appear alongside GyrA/ParC."""
    causal = causal_genes_for_drug("ciprofloxacin", grain="allele")
    assert {"GyrA", "ParC"} <= causal
    assert any("ib-cr" in a.lower() for a in causal)


def test_causal_meropenem_family() -> None:
    """Meropenem (family): carbapenemases + porins — and NOT pulled down to every β-lactamase."""
    causal = causal_genes_for_drug("meropenem", grain="family")
    assert {"KPC", "NDM", "OXA", "VIM", "IMP", "OmpK35", "OmpK36"} <= causal


def test_causal_colistin_family() -> None:
    """Colistin (family): mcr acquired genes + MgrB/PmrB chromosomal."""
    causal = causal_genes_for_drug("colistin", grain="family")
    assert {"MgrB", "PmrB"} <= causal
    assert any(g.lower().startswith("mcr") for g in causal)


def test_causal_tmp_smx_family() -> None:
    """TMP-SMX (family): trimethoprim (Dfr) + sulfonamide (Sul) acquired genes."""
    causal = causal_genes_for_drug("trimethoprim-sulfamethoxazole", grain="family")
    assert {"Dfr", "Sul"} <= causal


def test_causal_unknown_drug_raises() -> None:
    with pytest.raises(ValueError, match="no causal-mechanism spec"):
        causal_genes_for_drug("not-a-drug")


def test_causal_bad_grain_raises() -> None:
    with pytest.raises(ValueError, match="grain must be"):
        causal_genes_for_drug("meropenem", grain="subfamily")


def test_drug_causal_keys_match_drug_columns() -> None:
    """The causal + determinant specs cover exactly the drugs the ceiling map covers (no drift)."""
    pytest.importorskip("bacpredict.engine.gene_lr.kfold_probe")  # DRUG_COLUMNS import pulls the probe harness
    from kleb_ast.kleborate_determinant_lr import DRUG_COLUMNS

    assert set(_DRUG_CAUSAL) == set(DRUG_COLUMNS)
    assert set(_DRUG_DETERMINANT) == set(DRUG_COLUMNS)


def test_determinant_superset_of_causal_meropenem() -> None:
    """Inclusive determinant scope ⊇ the narrow causal set, and adds the non-carbapenem β-lactamases."""
    causal = causal_genes_for_drug("meropenem", grain="family")
    determ = determinant_genes_for_drug("meropenem", grain="family")
    assert causal <= determ
    # SHV-OKP-LEN (intrinsic β-lactamase, Bla_chr) belongs in the ceiling but is not carbapenem-causal
    assert "SHV-OKP-LEN" in determ
    assert "SHV-OKP-LEN" not in causal


def test_determinant_flq_matches_causal() -> None:
    """Where the causal map is already class-wide (fluoroquinolones), determinant == causal."""
    assert determinant_genes_for_drug("ciprofloxacin", grain="family") == causal_genes_for_drug(
        "ciprofloxacin", grain="family")


# --------------------------------------------------------------------------- #
# merged_label — CARD overrides Bakta; unnamed stays None                      #
# --------------------------------------------------------------------------- #
def _calls(rows: list[dict]) -> pd.DataFrame:
    cols = ["flat_index", "amr_source", "amr_gene_family", "amr_allele"]
    return pd.DataFrame(rows, columns=cols)


def test_merged_label_card_overrides_bakta_family() -> None:
    bakta = ["dnaA", None, "ompA", "rpoB"]  # idx1 unnamed
    calls = _calls([
        {"flat_index": 0, "amr_source": "acquired", "amr_gene_family": "KPC", "amr_allele": "kpc-2"},
        {"flat_index": 2, "amr_source": "chromosomal", "amr_gene_family": "OmpK36", "amr_allele": "OmpK36"},
    ])
    out = merged_label(bakta, calls, grain="family")
    assert list(out["label"]) == ["KPC", None, "OmpK36", "rpoB"]
    assert list(out["source"]) == ["card", None, "card", "bakta"]


def test_merged_label_allele_grain_picks_allele_column() -> None:
    bakta = ["x", "y"]
    calls = _calls([
        {"flat_index": 0, "amr_source": "acquired", "amr_gene_family": "KPC", "amr_allele": "kpc-3"},
    ])
    fam = merged_label(bakta, calls, grain="family")
    al = merged_label(bakta, calls, grain="allele")
    assert fam.loc[0, "label"] == "KPC"
    assert al.loc[0, "label"] == "kpc-3"


def test_merged_label_ignores_orphan_and_unmatched_source() -> None:
    bakta = ["geneA", "geneB"]
    calls = _calls([
        {"flat_index": -1, "amr_source": "acquired", "amr_gene_family": "CTX-M", "amr_allele": "ctx-m-15"},
        {"flat_index": 0, "amr_source": "partial_other", "amr_gene_family": "SHV", "amr_allele": "shv-1"},
    ])
    out = merged_label(bakta, calls, grain="family")
    # orphan (flat_index -1) dropped; non-acquired/chromosomal source ignored → Bakta names survive
    assert list(out["label"]) == ["geneA", "geneB"]
    assert list(out["source"]) == ["bakta", "bakta"]


def test_merged_label_no_calls_returns_bakta_only() -> None:
    bakta = ["a", None, "c"]
    out = merged_label(bakta, None, grain="family")
    assert list(out["label"]) == ["a", None, "c"]
    assert list(out["source"]) == ["bakta", None, "bakta"]
    assert list(out["flat_index"]) == [0, 1, 2]
