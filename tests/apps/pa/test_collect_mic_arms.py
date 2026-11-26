"""Collecting the arms: every drug gets a row from every arm, and the comparison is made on a common subset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bacpredict.apps.pa.collect_mic_arms import collect

INF = float("inf")


def _write_arm(root: Path, drug: str, ids, y_true, y_prob, *, mic: bool, sigma=1.5, sigma_train=0.9,
               validate: bool = False):
    run = root / drug
    run.mkdir(parents=True)
    arrays = {"y_true": np.asarray(y_true, dtype=int), "y_prob": np.asarray(y_prob, dtype=float),
              "sample_ids": np.asarray(ids, dtype=np.str_), "drug": np.array(drug),
              "split": np.asarray(["holdout"] * len(ids), dtype=np.str_),
              "species": np.asarray([""] * len(ids), dtype=np.str_),
              "operating_threshold": np.array(np.nan)}
    block = {"n_samples": len(ids), "scores": "eval_scores.npz", "metrics": {}}
    if mic:
        mu = np.log(np.asarray(y_prob) / (1 - np.asarray(y_prob))) * sigma
        arrays |= {"mu": mu, "lo": np.where(np.asarray(y_true) > 0, 0.0, -INF),
                   "hi": np.where(np.asarray(y_true) > 0, INF, 0.0), "sigma": np.array(sigma)}
        block["mic"] = {"sigma": sigma, "sigma_train": sigma_train}
    logit = np.log(np.asarray(y_prob) / (1 - np.asarray(y_prob)))
    arrays.setdefault("score", logit)
    np.savez(run / "eval_scores.npz", **arrays)
    if validate:
        # the same samples stand in for a validate split; only its scores and labels matter to the fit
        np.savez(run / "eval_scores_validate.npz", y_true=np.asarray(y_true, dtype=int),
                 y_prob=np.asarray(y_prob, dtype=float), score=logit,
                 sample_ids=np.asarray(ids, dtype=np.str_), drug=np.array(drug),
                 split=np.asarray(["validate"] * len(ids), dtype=np.str_),
                 species=np.asarray([""] * len(ids), dtype=np.str_),
                 operating_threshold=np.array(np.nan))
    (run / "results.json").write_text(json.dumps({
        "split": {"source": "split_table", "table_md5": "abc"}, "model": {"git_sha": "deadbeef"},
        "holdouts": {"holdout": block},
        **({"mic": {"objective": "censored", "cells": "all"}} if mic else {}),
    }))
    return run


@pytest.fixture
def two_arms(tmp_path):
    """A binary arm over six genomes and a MIC arm over four of them — nested holdouts, as the real ones are."""
    ids6 = [f"G{i}" for i in range(6)]
    y6 = [1, 0, 1, 0, 1, 0]
    p6 = [0.9, 0.2, 0.8, 0.1, 0.6, 0.3]
    _write_arm(tmp_path / "bce", "meropenem", ids6, y6, p6, mic=False)
    _write_arm(tmp_path / "cens", "meropenem", ids6[:4], y6[:4], p6[:4], mic=True)
    return {"bce": tmp_path / "bce", "cens": tmp_path / "cens"}


def test_every_arm_is_scored_on_its_own_holdout_and_on_the_common_one(two_arms):
    df = collect(two_arms, ["meropenem"], ("holdout",))
    own = df[df["subset"] == "own"].set_index("arm")["n_samples"].to_dict()
    common = df[df["subset"] == "common"].set_index("arm")["n_samples"].to_dict()
    assert own == {"bce": 6, "cens": 4}
    assert common == {"bce": 4, "cens": 4}   # the intersection, for both arms


def test_the_common_subset_is_the_intersection_not_the_smaller_arm(tmp_path):
    """Two arms that each drop a different genome must be compared on the four they share, not on five."""
    ids = [f"G{i}" for i in range(6)]
    y, p = [1, 0, 1, 0, 1, 0], [0.9, 0.2, 0.8, 0.1, 0.6, 0.3]
    a = {"a": tmp_path / "a", "b": tmp_path / "b"}
    _write_arm(a["a"], "m", ids[:5], y[:5], p[:5], mic=False)
    _write_arm(a["b"], "m", ids[1:], y[1:], p[1:], mic=False)
    df = collect(a, ["m"], ("holdout",))
    assert set(df[df["subset"] == "common"]["n_samples"]) == {4}


def test_a_missing_arm_is_a_row_not_an_absence(tmp_path):
    """A half-finished array must not read as a complete comparison."""
    ids, y, p = ["G0", "G1"], [1, 0], [0.8, 0.2]
    arms = {"done": tmp_path / "done", "pending": tmp_path / "pending"}
    _write_arm(arms["done"], "m", ids, y, p, mic=False)
    (tmp_path / "pending").mkdir()
    df = collect(arms, ["m"], ("holdout",))
    assert (df["status"] == "missing").sum() == 1
    assert df.loc[df["status"] == "missing", "arm"].iloc[0] == "pending"


def test_calibration_is_recomputed_for_every_arm_by_one_code_path(two_arms):
    """The incumbent arm predates all of this, so its Brier and ECE must come from its scores, not its JSON."""
    df = collect(two_arms, ["meropenem"], ("holdout",))
    assert df.loc[df["arm"] == "bce", "brier_raw"].notna().all()
    assert df.loc[df["arm"] == "bce", "ece_raw"].notna().all()


def test_a_binary_arm_with_validate_scores_is_tempered_like_the_mic_arms(tmp_path):
    """Both arms then carry a one-parameter scale fitted on validate, so their calibration is comparable."""
    ids = [f"G{i}" for i in range(12)]
    rng = np.random.default_rng(0)
    y = (rng.random(12) < 0.5).astype(int)
    p = np.clip(np.where(y > 0, rng.uniform(0.5, 0.99, 12), rng.uniform(0.01, 0.5, 12)), 0.01, 0.99)
    arms = {"bce": tmp_path / "bce"}
    _write_arm(arms["bce"], "m", ids, y, p, mic=False, validate=True)
    row = collect(arms, ["m"], ("holdout",)).set_index("subset").loc["own"]
    assert np.isfinite(row["temperature"]) and row["temperature"] > 0
    assert np.isfinite(row["ece"]) and np.isfinite(row["ece_raw"])


def test_the_binary_arm_has_no_mic_columns_and_says_so_with_nan(two_arms):
    """A binary model has no MIC scale; reporting 0 for its agreement would be a fabricated number."""
    df = collect(two_arms, ["meropenem"], ("holdout",)).set_index(["arm", "subset"])
    assert np.isnan(df.loc[("bce", "own"), "sigma"])
    assert np.isnan(df.loc[("bce", "own"), "essential_agreement"])
    assert np.isfinite(df.loc[("cens", "own"), "sigma"])


def test_the_temperature_correction_is_visible(two_arms):
    """ECE at the validate sigma and at the training one, so the size of the re-fit is on the record."""
    df = collect(two_arms, ["meropenem"], ("holdout",)).set_index(["arm", "subset"])
    row = df.loc[("cens", "own")]
    assert np.isfinite(row["ece"]) and np.isfinite(row["ece_raw"])
    assert row["sigma"] != row["sigma_train"]


def test_agreement_rates_need_the_interval_table_and_are_nan_without_it(two_arms, tmp_path):
    """Guessing a censoring class from the interval alone would silently mislabel ``> X, < U`` pairs."""
    df = collect(two_arms, ["meropenem"], ("holdout",))
    assert df.loc[df["arm"] == "cens", "essential_agreement"].isna().all()
    intervals = tmp_path / "intervals"
    intervals.mkdir()
    pd.DataFrame([{"Sample": f"G{i}", "censor": "exact"} for i in range(4)]).to_csv(
        intervals / "meropenem_mic_intervals.csv", index=False)
    df2 = collect(two_arms, ["meropenem"], ("holdout",), intervals_dir=intervals)
    assert df2.loc[df2["arm"] == "cens", "essential_agreement"].notna().all()
