"""Unit tests for the attribution cohort selector.

The two things that must not break: the cohort comes from the **holdout** of the materialised split
table and nowhere else, and the stratified draw keeps the holdout's phenotype/lineage composition
while staying reproducible from its seed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
import pandas as pd

from bacpredict.engine.explain.select_attribution_cohort import (
    UNKNOWN_LINEAGE,
    allocate_by_largest_remainder,
    attach_lineage,
    select_cohort,
)


def _split_table(tmp_path: Path, n: int = 300) -> Path:
    """A split table with train/validate/holdout and a mix of labels, plus one ambiguous row."""
    rows = []
    for i in range(n):
        split = "holdout" if i % 3 == 0 else ("validate" if i % 3 == 1 else "train")
        rows.append({"Sample": f"S{i:03d}", "ast_label": i % 2, "split": split})
    rows[0]["ast_label"] = 0.5  # ambiguous: holds a split slot but is never scored
    path = tmp_path / "rifampin_split.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _lineage_csv(tmp_path: Path, samples: list[str]) -> Path:
    """A TB-Profiler-shaped lineage table covering all but the last sample."""
    frame = pd.DataFrame({
        "Sample": samples[:-1],
        "main_lineage": [f"lineage{(i % 3) + 1}" for i in range(len(samples) - 1)],
    })
    path = tmp_path / "tbprofiler_lineage.csv"
    frame.to_csv(path, index=False)
    return path


def test_cohort_is_drawn_only_from_the_holdout(tmp_path: Path) -> None:
    """Every selected genome must carry split == holdout in the split table."""
    table = _split_table(tmp_path)
    cohort = select_cohort(table, n_genomes=None)

    holdout = pd.read_csv(table)
    holdout = set(holdout.loc[holdout["split"] == "holdout", "Sample"])
    assert set(cohort["Sample"]) <= holdout
    # The ambiguous row holds a holdout slot but has no clean label, so it must not be attributed.
    assert "S000" not in set(cohort["Sample"])


def test_subsample_is_reproducible_and_sized(tmp_path: Path) -> None:
    """The same seed gives the same genomes; a different seed generally does not."""
    table = _split_table(tmp_path)
    first = select_cohort(table, n_genomes=40, seed=7)
    again = select_cohort(table, n_genomes=40, seed=7)
    other = select_cohort(table, n_genomes=40, seed=8)

    assert len(first) == 40
    assert list(first["Sample"]) == list(again["Sample"])
    assert list(first["Sample"]) != list(other["Sample"])


def test_subsample_preserves_label_balance(tmp_path: Path) -> None:
    """Proportional allocation must keep the resistant share close to the holdout's."""
    table = _split_table(tmp_path)
    whole = select_cohort(table, n_genomes=None)
    drawn = select_cohort(table, n_genomes=40, seed=0)

    whole_share = (whole["ast_label"] == 1).mean()
    drawn_share = (drawn["ast_label"] == 1).mean()
    assert abs(whole_share - drawn_share) < 0.1


def test_requesting_more_than_available_returns_the_whole_holdout(tmp_path: Path) -> None:
    """A subsample larger than the holdout is not an error — it is the whole holdout."""
    table = _split_table(tmp_path)
    whole = select_cohort(table, n_genomes=None)
    assert len(select_cohort(table, n_genomes=10_000)) == len(whole)


def test_stratification_uses_lineage_when_given(tmp_path: Path) -> None:
    """With a lineage table the stratum is label|lineage, and unmatched genomes fall back to unknown."""
    table = _split_table(tmp_path)
    whole = select_cohort(table, n_genomes=None)
    lineage_csv = _lineage_csv(tmp_path, sorted(whole["Sample"]))

    cohort = select_cohort(table, n_genomes=None, lineage_csv=lineage_csv)
    assert cohort["stratum"].str.contains(r"^\d\|lineage\d$").any()
    assert (cohort["lineage"] == UNKNOWN_LINEAGE).sum() == 1


def test_attach_lineage_rejects_a_table_missing_its_column(tmp_path: Path) -> None:
    """A lineage table without the requested column is a hard error, not a silent unknown."""
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Sample": ["S001"], "something_else": ["x"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing column"):
        attach_lineage(pd.DataFrame({"Sample": ["S001"]}), path, "main_lineage")


def test_allocation_sums_to_total_and_respects_stratum_sizes() -> None:
    """Largest remainder must hit the total exactly and never over-draw a small stratum."""
    sizes = pd.Series({"a": 100, "b": 10, "c": 3, "d": 1})
    draws = allocate_by_largest_remainder(sizes, 40)
    assert draws.sum() == 40
    assert (draws <= sizes).all()
    assert (draws >= 0).all()


def test_allocation_clamps_when_asked_for_more_than_exists() -> None:
    """Asking for more than the strata hold yields everything, not an over-draw."""
    sizes = pd.Series({"a": 5, "b": 2})
    draws = allocate_by_largest_remainder(sizes, 99)
    assert draws.sum() == 7
    assert (draws == sizes).all()
