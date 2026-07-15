"""Tests for the chromosomal mut/WT one-hot split (:mod:`bacpredict.apps.kleb.card_determinant_lr`).

The split fixes the degenerate gene-presence one-hot for intrinsic point-mutation genes (gyrA/parC/…):
the gene is present in ~every genome, so only the **mutant** (sourced from Kleborate's mutation columns)
discriminates. These tests pin the parsing + the mut/WT column construction on synthetic inputs.
"""

from __future__ import annotations

import pandas as pd

from bacpredict.apps.kleb.card_determinant_lr import (
    _base_gene,
    _category,
    _is_causal,
    build_card_onehot,
    mutation_carriers,
)


def test_base_gene_category_causal_splits() -> None:
    assert _base_gene("GyrA (mut)") == "GyrA"
    assert _base_gene("GyrA (WT)") == "GyrA"
    assert _base_gene("KPC") == "KPC"
    # mut keeps the base gene's chromosomal category + causality; WT is intrinsic-coding + never causal
    assert _category("GyrA (mut)") == "chromosomal_mutation"
    assert _category("GyrA (WT)") == "chromosomal_coding"
    assert _category("KPC") == "acquired_hgt"
    causal = {"GyrA", "KPC"}
    assert _is_causal("GyrA (mut)", causal)
    assert not _is_causal("GyrA (WT)", causal)
    assert _is_causal("KPC", causal)


def test_mutation_carriers_parses_by_prefix(tmp_path) -> None:
    meta = tmp_path / "meta.tsv"
    pd.DataFrame({
        "Sample": ["g1", "g2", "g3"],
        "Flq_mutations": ["GyrA-83L;ParC-80I", "GyrA-87N", "-"],
        "Col_mutations": ["-", "MgrB-truncated", "-"],
    }).to_csv(meta, sep="\t", index=False)
    out = mutation_carriers(meta, {"GyrA", "ParC", "MgrB"}, {"g1", "g2", "g3"})
    assert out["GyrA"] == {"g1", "g2"}
    assert out["ParC"] == {"g1"}
    assert out["MgrB"] == {"g2"}


def test_mutation_carriers_empty_when_no_chrom_genes(tmp_path) -> None:
    # acquired-only gene set → no metadata read needed
    assert mutation_carriers(tmp_path / "nonexistent.tsv", {"KPC", "CTX-M"}, {"g1"}) == {}


def _calls(rows: list[dict]) -> pd.DataFrame:
    cols = ["Sample", "amr_source", "amr_gene_family", "amr_allele"]
    return pd.DataFrame(rows, columns=cols)


def test_build_card_onehot_splits_chromosomal_mut_wt() -> None:
    # 4 genomes all carry gyrA (intrinsic) + one carries KPC; g1/g2 are gyrA-mutant per chrom_mut.
    calls = _calls([
        {"Sample": "g1", "amr_source": "chromosomal", "amr_gene_family": "GyrA", "amr_allele": "GyrA"},
        {"Sample": "g2", "amr_source": "chromosomal", "amr_gene_family": "GyrA", "amr_allele": "GyrA"},
        {"Sample": "g3", "amr_source": "chromosomal", "amr_gene_family": "GyrA", "amr_allele": "GyrA"},
        {"Sample": "g4", "amr_source": "chromosomal", "amr_gene_family": "GyrA", "amr_allele": "GyrA"},
        {"Sample": "g1", "amr_source": "acquired", "amr_gene_family": "KPC", "amr_allele": "kpc-2"},
    ])
    universe = ["g1", "g2", "g3", "g4"]
    oh = build_card_onehot(calls, {"GyrA", "KPC"}, universe, "family", chrom_mut={"GyrA": {"g1", "g2"}})

    assert "GyrA (mut)" in oh.columns and "GyrA (WT)" in oh.columns and "GyrA" not in oh.columns
    assert list(oh["GyrA (mut)"]) == [1, 1, 0, 0]   # discriminative
    assert list(oh["GyrA (WT)"]) == [0, 0, 1, 1]    # the non-mutant carriers
    assert list(oh["KPC"]) == [1, 0, 0, 0]          # acquired stays simple presence


def test_build_card_onehot_no_chrom_mut_is_plain_presence() -> None:
    calls = _calls([
        {"Sample": "g1", "amr_source": "acquired", "amr_gene_family": "CTX-M", "amr_allele": "ctx-m-15"},
    ])
    oh = build_card_onehot(calls, {"CTX-M"}, ["g1", "g2"], "family")
    assert list(oh["CTX-M"]) == [1, 0]
