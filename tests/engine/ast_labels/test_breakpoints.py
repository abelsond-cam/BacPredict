"""Breakpoint resolution, MIC parsing and the binary-from-MIC rule (EUCAST v15)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bacpredict.engine.ast_labels.breakpoints import (
    binary_call_from_mic,
    load_breakpoints,
    parse_mic,
    resolve_r_gt,
    taxon_lineage,
)

TEN = ["ciprofloxacin", "ceftazidime", "meropenem", "tobramycin", "amikacin", "imipenem",
       "piperacillin-tazobactam", "aztreonam", "levofloxacin", "cefepime"]


@pytest.fixture(scope="module")
def bp() -> pd.DataFrame:
    return load_breakpoints()


def _r(bp, species, drug) -> float:
    return float(resolve_r_gt(pd.Series([species]), pd.Series([drug]), bp)[0])


def test_table_covers_the_ten_drugs_for_both_groups(bp):
    """Every agreed drug has a breakpoint for P. aeruginosa and for a generic Enterobacterales species."""
    for drug in TEN:
        assert not np.isnan(_r(bp, "Pseudomonas aeruginosa", drug)), drug
        assert not np.isnan(_r(bp, "Escherichia coli", drug)), drug


@pytest.mark.parametrize(("species", "drug", "r_gt"), [
    # P. aeruginosa — EUCAST v15 Pseudomonas sheet (also what AMR-GNN called with)
    ("Pseudomonas aeruginosa", "meropenem", 8), ("Pseudomonas aeruginosa", "piperacillin-tazobactam", 16),
    ("Pseudomonas aeruginosa", "ciprofloxacin", 0.5), ("Pseudomonas aeruginosa", "amikacin", 16),
    ("Pseudomonas aeruginosa", "aztreonam", 16), ("Pseudomonas aeruginosa", "levofloxacin", 2),
    # Enterobacterales order row, family and genus overrides
    ("Klebsiella pneumoniae", "ciprofloxacin", 0.5), ("Salmonella enterica", "ciprofloxacin", 0.06),
    ("Salmonella enterica", "meropenem", 8), ("Proteus mirabilis", "imipenem", 4),
    ("Escherichia coli", "piperacillin-tazobactam", 8), ("Escherichia coli", "levofloxacin", 1),
])
def test_resolution_picks_the_most_specific_row(bp, species, drug, r_gt):
    assert _r(bp, species, drug) == r_gt


def test_unknown_taxon_or_drug_has_no_breakpoint(bp):
    assert np.isnan(_r(bp, "Mycobacterium tuberculosis", "ciprofloxacin"))
    assert np.isnan(_r(bp, "Pseudomonas aeruginosa", "gentamicin"))
    assert np.isnan(_r(bp, "Escherichia coli", "colistin"))


def test_lineage_is_most_specific_first():
    assert [r for r, _ in taxon_lineage("Salmonella enterica")] == ["species", "genus", "family", "order"]
    assert taxon_lineage("Morganella morganii")[2] == ("family", "Morganellaceae")


def test_duplicate_breakpoint_keys_are_rejected(tmp_path):
    p = tmp_path / "bp.csv"
    pd.DataFrame({"taxon_rank": ["genus", "genus"], "taxon": ["X", "X"], "drug": ["d", "d"],
                  "r_gt_mg_l": [1, 2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_breakpoints(p)


def test_parse_mic_with_separate_sign_column():
    m = parse_mic(pd.Series(["8", "16/4", "1,2", "0.5", "1,5", None]),
                  pd.Series(["==", "<=", ">,<", ">", "=", "="]))
    assert m.mic_sign.tolist()[:4] == ["=", "<=", ">,<", ">"]
    assert m.mic_value.tolist()[:4] == [8.0, 16.0, 1.0, 0.5]
    assert m.mic_upper.iloc[2] == 2.0 and np.isnan(m.mic_upper.iloc[0])
    assert np.isnan(m.mic_value.iloc[4]) and pd.isna(m.mic_sign.iloc[4])  # non-bracketed comma refused
    assert np.isnan(m.mic_value.iloc[5]) and pd.isna(m.mic_sign.iloc[5])


def test_parse_mic_with_embedded_operator():
    m = parse_mic(pd.Series(["<=0.5", ">16", "8", ">=128", "<2", "0.125", "junk"]))
    assert m.mic_sign.tolist()[:6] == ["<=", ">", "=", ">=", "<", "="]
    assert m.mic_value.tolist()[:6] == [0.5, 16.0, 8.0, 128.0, 2.0, 0.125]
    assert pd.isna(m.mic_sign.iloc[6]) and np.isnan(m.mic_value.iloc[6])


@pytest.mark.parametrize(("sign", "value", "upper", "expected"), [
    ("=", 8, np.nan, 0.0),      # MIC 8 meropenem = S under R > 8 (AMR-GNN's call)
    ("=", 16, np.nan, 1.0),
    ("<=", 8, np.nan, 0.0),
    ("<=", 16, np.nan, np.nan),  # could be 8 or 16 — straddles
    ("<", 16, np.nan, np.nan),
    ("<", 8, np.nan, 0.0),
    (">", 8, np.nan, 1.0),       # MIC > 8 is > R
    (">", 4, np.nan, np.nan),    # could be 8 (S) or 16 (R)
    (">=", 8, np.nan, np.nan),   # could be exactly 8 (S)
    (">=", 16, np.nan, 1.0),
    (">,<", 8, 16, 1.0),         # 8 < MIC < 16 → above R=8
    (">,<", 2, 8, 0.0),          # 2 < MIC < 8 → at most R
    (">,<", 4, 16, np.nan),
])
def test_binary_rule_and_censoring(sign, value, upper, expected):
    got = binary_call_from_mic(pd.Series([sign]), pd.Series([value], dtype=float),
                               pd.Series([upper], dtype=float), np.array([8.0]))[0]
    assert (np.isnan(got) and np.isnan(expected)) or got == expected


def test_no_breakpoint_or_no_value_gives_nan():
    got = binary_call_from_mic(pd.Series(["=", "="]), pd.Series([4.0, np.nan]), pd.Series([np.nan, np.nan]),
                               np.array([np.nan, 8.0]))
    assert np.isnan(got).all()
