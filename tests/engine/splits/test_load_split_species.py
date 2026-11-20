"""``load_split_species`` — the species column of a split table, over the same clean rows as ``load_splits``."""

from __future__ import annotations

import pandas as pd

from bacpredict.engine.splits.load_splits import load_split_species, load_splits


def test_species_over_clean_rows_only(tmp_path):
    path = tmp_path / "ciprofloxacin_split.csv"
    pd.DataFrame({
        "Sample": ["a", "b", "c", "c", "d"],
        "species": ["paeruginosa", "enterobacterales", "paeruginosa", "enterobacterales", "paeruginosa"],
        "ast_label": [1, 0, 1, 0, 0.5],  # c twice: the first row wins; d fractional: never scored
        "split": ["train", "holdout", "holdout_st", "train", "train"],
    }).to_csv(path, index=False)
    species = load_split_species(path)
    assert species == {"a": "paeruginosa", "b": "enterobacterales", "c": "paeruginosa"}
    assert set(species) == set(load_splits(path)[0])


def test_a_table_without_species_gives_an_empty_map(tmp_path):
    path = tmp_path / "rifampin_split.csv"
    pd.DataFrame({"Sample": ["a"], "ast_label": [1], "split": ["train"]}).to_csv(path, index=False)
    assert load_split_species(path) == {}
