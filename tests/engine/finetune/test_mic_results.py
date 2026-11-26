"""End to end over the seam a GPU smoke would otherwise be the first to test: write a MIC arm, then collect it.

``write_mic_split_table_results`` produces ``results.json`` and ``eval_scores*.npz``; ``collect_mic_arms``
and the driver's ``check-smoke`` both read them. Nothing else checks that the two halves agree on the shape,
and finding out that they do not costs an hour of GPU and a queue wait. A stub trainer stands in for
Bacformer: the only thing these tests need from it is predictions in the dataset's order.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from bacpredict.apps.pa.collect_mic_arms import collect
from bacpredict.engine.finetune.finetune_amr import SplitTableSplits, write_mic_split_table_results

INF = float("inf")


class _StubTrainer:
    """Stands in for the trained model: returns a fixed ``mu`` per sample, in the dataset's order."""

    def __init__(self, mu_by_sample: dict[str, float], sigma: float = 0.8):
        self.mu_by_sample = mu_by_sample
        self.sigma = sigma

    def predict(self, dataset):
        ids = list(dataset)
        mu = np.array([[self.mu_by_sample[s]] for s in ids], dtype=float)
        labels = np.array([_INTERVALS[s] for s in ids], dtype=float)
        return type("P", (), {"predictions": mu, "label_ids": labels})()


# twelve genomes: resistant ones right-censored or exact above the breakpoint, susceptible ones below
_INTERVALS: dict[str, tuple[float, float, float, float]] = {}
_CENSOR: dict[str, str] = {}
for _i in range(12):
    _r = _i % 2
    if _i % 4 == 0:        # exact
        _INTERVALS[f"G{_i}"] = (float(_r), 1.0, 2.0, 2.0) if _r else (0.0, -3.0, -2.0, -2.0)
        _CENSOR[f"G{_i}"] = "exact"
    elif _i % 4 == 1:      # right-censored
        _INTERVALS[f"G{_i}"] = (1.0, 3.0, INF, 4.0)
        _CENSOR[f"G{_i}"] = "right"
    elif _i % 4 == 2:      # left-censored
        _INTERVALS[f"G{_i}"] = (0.0, -INF, -2.0, -2.0)
        _CENSOR[f"G{_i}"] = "left"
    else:                  # a bare R/S call
        _INTERVALS[f"G{_i}"] = (float(_r), 0.0, INF, float("nan")) if _r else (0.0, -INF, 0.0, float("nan"))
        _CENSOR[f"G{_i}"] = "bare"
# make the labels consistent with the intervals
for _s, (_, _lo, _hi, _p) in list(_INTERVALS.items()):
    _INTERVALS[_s] = (1.0 if _lo >= 0 else 0.0, _lo, _hi, _p)


@pytest.fixture
def written(tmp_path):
    """Run the real results writer over a stub trainer, and hand back where it put everything."""
    ids = list(_INTERVALS)
    splits = SplitTableSplits(
        path=str(tmp_path / "meropenem_split.csv"), md5="abc",
        label_map={s: int(v[0]) for s, v in _INTERVALS.items()},
        # four genomes per split, one of each censoring class, so every split can exercise every read-out:
        # a holdout with no uncensored cell reports NaN agreement by design, which would hide a real bug here
        train=ids[0:4], validate=ids[4:8], holdout=ids[8:12], extra={"holdout_st": ids[0:2]},
        species=dict.fromkeys(ids, "paeruginosa"))
    rng = np.random.default_rng(0)
    mu = {s: float(v[0] * 2.5 - 1.2 + rng.normal(0, 0.6)) for s, v in _INTERVALS.items()}
    out = tmp_path / "models" / "censored_interval" / "meropenem"
    payload = write_mic_split_table_results(
        _StubTrainer(mu), lambda sample_ids: sample_ids, splits,
        intervals=_INTERVALS, censor=_CENSOR,
        cell_report={"table": "t.csv", "cells": "all", "objective": "censored", "by_split": {}},
        output_dir=str(out), task="pa_mic_ast", drug="meropenem",
        model_name_or_path="bacformer", model_revision=None, split_source="split_table",
        run_config={"seed": 1})
    return out, payload


def test_results_json_validates_and_carries_every_holdout(written):
    """``write_results_json`` enforces the v1 schema, so a missing metric key fails here, not on the cluster."""
    out, payload = written
    on_disk = json.loads((out / "results.json").read_text())
    assert on_disk["task"] == "pa_mic_ast"
    assert list(on_disk["holdouts"]) == ["holdout", "holdout_st"]
    assert on_disk["split"]["table_md5"] == "abc"
    assert payload["mic"]["objective"] == "censored"


def test_sigma_is_fitted_on_validate_and_both_values_are_recorded(written):
    """The decision of record is only auditable if the run says which sigma it used and what it trained."""
    _, payload = written
    mic = payload["mic"]
    assert mic["sigma_validate_fit"]["n"] == 4        # the validate split, not the holdout
    assert mic["sigma_train"] == 0.8                  # the stub trainer's
    assert mic["sigma"] != mic["sigma_train"]
    for block in payload["holdouts"].values():
        assert block["mic"]["sigma"] == mic["sigma"]  # one sigma, used for every holdout


def test_validate_scores_are_written_so_the_arm_can_be_calibrated(written):
    """Every arm must be temperature-comparable, which needs the split the scale is fitted on, saved."""
    out, payload = written
    path = out / "eval_scores_validate.npz"
    assert path.exists()
    with np.load(path) as z:
        assert len(z["sample_ids"]) == payload["mic"]["sigma_validate_fit"]["n"]
        assert {"score", "mu", "lo", "hi", "sigma", "y_true", "y_prob"} <= set(z)
        assert np.array_equal(z["score"], z["mu"])      # one key, whatever the arm's link
        assert set(z["split"]) == {"validate"}


def test_the_npz_keeps_the_old_layout_and_adds_the_mic_scale(written):
    """Every existing reader of eval_scores.npz must keep working on a MIC arm's output."""
    out, _ = written
    with np.load(out / "eval_scores.npz") as z:
        assert {"y_true", "y_prob", "sample_ids", "drug", "split", "species", "operating_threshold"} <= set(z)
        assert {"mu", "lo", "hi", "sigma"} <= set(z)
        assert len(z["sample_ids"]) == len(z["mu"]) == len(z["y_prob"])
        assert np.isinf(z["hi"]).any()                # the infinities survive the round trip


def test_the_probability_is_phi_of_mu_over_sigma(written):
    """It must be the censored model's own probability, not a sigmoid of a logit it never produced."""
    from scipy.special import ndtr
    out, _ = written
    with np.load(out / "eval_scores.npz") as z:
        assert np.allclose(z["y_prob"], ndtr(z["mu"] / float(z["sigma"])))
        assert np.array_equal((z["y_prob"] >= 0.5).astype(int), (z["mu"] >= 0).astype(int))


def test_the_temperature_correction_is_recorded_not_just_applied(written):
    """Calibration at the training sigma is kept beside the headline, so the size of the re-fit is visible."""
    _, payload = written
    mic = payload["holdouts"]["holdout"]["mic"]
    assert {"calibration", "calibration_at_sigma_train", "censored_nll",
            "censored_nll_at_sigma_train"} <= set(mic)
    assert mic["calibration"]["ece"] != mic["calibration_at_sigma_train"]["ece"]


def test_the_collector_reads_what_the_writer_wrote(written, tmp_path):
    """The seam itself: collect() over the real output, with no intermediate hand-editing."""
    out, payload = written
    import pandas as pd
    intervals = tmp_path / "intervals"
    intervals.mkdir()
    pd.DataFrame([{"Sample": s, "censor": c} for s, c in _CENSOR.items()]).to_csv(
        intervals / "meropenem_mic_intervals.csv", index=False)
    df = collect({"censored_interval": out.parent}, ["meropenem"], ("holdout",), intervals_dir=intervals)
    assert set(df["status"]) == {"ok"}
    row = df[(df["subset"] == "own")].iloc[0]
    assert row["n_samples"] == payload["holdouts"]["holdout"]["n_samples"]
    assert row["sigma"] == pytest.approx(payload["mic"]["sigma"])
    assert row["auroc"] == pytest.approx(payload["holdouts"]["holdout"]["metrics"]["auroc"])
    assert np.isfinite(row["censored_nll"]) and np.isfinite(row["essential_agreement"])


def test_an_empty_extra_holdout_is_recorded_with_n_zero_not_dropped(tmp_path):
    """A drug no held-out-ST genome is labelled for must still appear, or a gap reads as a result."""
    ids = list(_INTERVALS)
    splits = SplitTableSplits(
        path="s.csv", md5="abc", label_map={s: int(v[0]) for s, v in _INTERVALS.items()},
        train=ids[0:4], validate=ids[4:8], holdout=ids[8:12], extra={"holdout_st": []},
        species=dict.fromkeys(ids, "paeruginosa"))
    mu = {s: float(v[0] * 2 - 1) for s, v in _INTERVALS.items()}
    out = tmp_path / "m"
    payload = write_mic_split_table_results(
        _StubTrainer(mu), lambda s: s, splits, intervals=_INTERVALS, censor=_CENSOR,
        cell_report={"by_split": {}}, output_dir=str(out), task="t", drug="meropenem",
        model_name_or_path="b", model_revision=None, split_source="split_table", run_config={})
    assert payload["holdouts"]["holdout_st"]["n_samples"] == 0
    assert (out / "eval_scores_holdout_st.npz").exists()
