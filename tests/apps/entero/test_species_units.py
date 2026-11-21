"""The Enterobacterales species-unit rule: it names species, and never prefix-matches a genus."""

from __future__ import annotations

import pandas as pd
import pytest

from bacpredict.apps.entero.species_units import UNIT_GENERA, UNITS, assign_units, unit_samples


def _qc(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["Sample", "species", "qc_pass"])


def test_kpsc_names_its_species() -> None:
    """Every KpSC species lands in ``kpsc``, and nothing else does."""
    s = pd.Series(["Klebsiella pneumoniae", "Klebsiella variicola", "Klebsiella quasipneumoniae",
                   "Klebsiella africana", "Klebsiella tropica"])
    assert list(assign_units(s)) == ["kpsc"] * 5


@pytest.mark.parametrize("label", ["Klebsiella aerogenes", "Enterobacter aerogenes"])
def test_aerogenes_is_enterobacter_under_both_names(label: str) -> None:
    """⛔ The regression this rule exists for: K. aerogenes is not KpSC, and the two label sheets spell it
    differently, so a genus prefix match would split one organism across two units."""
    assert assign_units(pd.Series([label])).iloc[0] == "enterobacter"


def test_klebsiella_prefix_alone_is_not_enough() -> None:
    """A Klebsiella outside the complex is not silently absorbed into ``kpsc``."""
    assert assign_units(pd.Series(["Klebsiella oxytoca"])).iloc[0] != "kpsc"


def test_shigella_subgroups_are_separate_from_ecoli() -> None:
    """David's rule: E. coli and Shigella get subgroups."""
    s = pd.Series(["Escherichia coli", "Shigella sonnei", "Shigella flexneri"])
    assert list(assign_units(s)) == ["ecoli", "ssonnei", "sflexneri"]


def test_minor_shigella_join_ecoli() -> None:
    """The species too small for a unit of their own go to ``ecoli``, not to no unit at all."""
    s = pd.Series(["Shigella boydii", "Shigella dysenteriae", "Shigella"])
    assert list(assign_units(s)) == ["ecoli"] * 3


def test_salmonella_takes_serovar_labels() -> None:
    """Salmonella is matched by genus: no explicit list would catch its serovar-style labels."""
    s = pd.Series(["Salmonella enterica", "Salmonella Infantis", "Salmonella sp."])
    assert list(assign_units(s)) == ["salmonella"] * 3


def test_enterobacter_species_take_the_genus() -> None:
    s = pd.Series(["Enterobacter cloacae", "Enterobacter hormaechei", "Enterobacter sp."])
    assert list(assign_units(s)) == ["enterobacter"] * 3


def test_undecided_genera_get_no_unit() -> None:
    """Genera with no unit yet are NA, never quietly folded into a neighbour."""
    s = pd.Series(["Serratia marcescens", "Proteus mirabilis", "Morganella morganii", "Providencia stuartii"])
    assert assign_units(s).isna().all()


def test_blank_and_missing_species_get_no_unit() -> None:
    assert assign_units(pd.Series([None, "", "   "])).isna().all()


def test_units_are_disjoint() -> None:
    """No species may be claimed by two units; overlapping units are a builder refusal."""
    named = [n for names in UNITS.values() for n in names]
    assert len(named) == len(set(named))
    for unit, names in UNITS.items():
        for other, genera in UNIT_GENERA.items():
            if other != unit:
                assert not any(n.split()[0] in genera for n in names), f"{unit} overlaps {other}"


def test_unit_samples_filters_on_qc_pass_and_sorts() -> None:
    qc = _qc([
        ("S3", "Klebsiella pneumoniae", "True"),
        ("S1", "Klebsiella quasipneumoniae", "true"),
        ("S2", "Klebsiella pneumoniae", "False"),
        ("S4", "Escherichia coli", "True"),
        ("S5", "Enterobacter aerogenes", "True"),
    ])
    assert unit_samples(qc, "kpsc") == ["S1", "S3"]
    assert unit_samples(qc, "ecoli") == ["S4"]
    assert unit_samples(qc, "enterobacter") == ["S5"]


def test_unit_samples_refuses_an_unknown_unit() -> None:
    with pytest.raises(ValueError, match="unknown unit"):
        unit_samples(_qc([("S1", "Escherichia coli", "True")]), "klebsiella")
