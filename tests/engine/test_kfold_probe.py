"""Unit tests for the k-fold × m-seed harness (``bacpredict.engine.gene_lr.kfold_probe``).

Cover the harness contract on synthetic feature frames (no HPC data, no ``.pt``): the run count, the
fixed evaluate holdout (pinned by ``evaluate_seed``, identical across seeds), the universe = frame
intersection rule, the paired-delta sign test, and the small-universe guard. Skipped where
sklearn/torch are unavailable (``fit_score_step`` imports torch at module load).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("torch")

from bacpredict.engine.gene_lr.kfold_probe import FeatureSpec, run_kfold_probe

N = 80
DIM = 4
IDS = [f"S{i:03d}" for i in range(N)]
LABEL_MAP = {s: i % 2 for i, s in enumerate(IDS)}  # balanced 40/40, alternating


def _frame(ids: list[str], *, sep: float, seed: int) -> pd.DataFrame:
    """A frame whose class means in feature 0 are ``±sep`` apart (separable when ``sep`` is large)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(len(ids), DIM))
    if sep:
        x[:, 0] += np.array([sep if LABEL_MAP[s] == 1 else -sep for s in ids])
    return pd.DataFrame(x, index=pd.Index(ids, name="Sample"))


def test_run_count_and_constant_evaluate() -> None:
    """3 folds × 2 seeds = 6 runs/frame, and the fixed holdout size is constant across runs."""
    specs = {"sep": FeatureSpec(_frame(IDS, sep=4.0, seed=0)), "noise": FeatureSpec(_frame(IDS, sep=0.0, seed=1))}
    res = run_kfold_probe(specs, LABEL_MAP, n_folds=3, seeds=(1, 2), evaluate_fraction=0.25)

    assert res["config"]["n_runs_per_frame"] == 6
    assert res["config"]["n_evaluate"] == max(1, int(0.25 * N))
    for frame in res["frames"].values():
        assert len(frame["per_run"]) == 6
        # n_evaluate is identical on every run (the holdout never moves).
        assert {r["n_evaluate"] for r in frame["per_run"]} == {res["config"]["n_evaluate"]}


def test_evaluate_fixed_by_evaluate_seed_not_seed() -> None:
    """Changing only ``seed`` leaves the evaluate holdout untouched; the harness asserts this internally."""
    specs = {"sep": FeatureSpec(_frame(IDS, sep=4.0, seed=0))}
    ev_seed1 = run_kfold_probe(specs, LABEL_MAP, n_folds=3, seeds=(1,), evaluate_fraction=0.25)["config"]["evaluate_ids"]
    ev_seed2 = run_kfold_probe(specs, LABEL_MAP, n_folds=3, seeds=(2,), evaluate_fraction=0.25)["config"]["evaluate_ids"]
    assert ev_seed1 == ev_seed2  # evaluate_seed (default 1) pins the holdout regardless of fold seed

    # And a multi-seed run does not raise the drift guard.
    multi = run_kfold_probe(specs, LABEL_MAP, n_folds=3, seeds=(1, 2, 3), evaluate_fraction=0.25)
    assert multi["config"]["evaluate_ids"] == ev_seed1


def test_separable_frame_beats_noise_paired() -> None:
    """A separable frame scores ~1.0 and wins the paired AUROC delta against pure noise on every run."""
    specs = {"sep": FeatureSpec(_frame(IDS, sep=5.0, seed=0)), "noise": FeatureSpec(_frame(IDS, sep=0.0, seed=1))}
    res = run_kfold_probe(specs, LABEL_MAP, n_folds=4, seeds=(1, 2), evaluate_fraction=0.25)

    assert res["frames"]["sep"]["aggregate"]["auroc"]["mean"] > 0.95
    delta = res["paired_auroc_deltas"]["sep__minus__noise"]
    assert delta["mean_delta"] > 0
    assert delta["win_fraction"] == 1.0
    assert delta["n_runs"] == 8


def test_aggregate_reports_all_fields() -> None:
    """Every aggregated metric carries mean/sd/min/max/n."""
    specs = {"sep": FeatureSpec(_frame(IDS, sep=4.0, seed=0))}
    agg = run_kfold_probe(specs, LABEL_MAP, n_folds=3, seeds=(1, 2), evaluate_fraction=0.25)["frames"]["sep"][
        "aggregate"
    ]
    for metric in ("auroc", "auprc", "sensitivity", "specificity", "balanced_accuracy"):
        assert set(agg[metric]) == {"mean", "sd", "min", "max", "n"}
        assert agg[metric]["n"] == 6


def test_universe_is_frame_intersection() -> None:
    """The fold universe is the intersection of every frame's samples ∩ the labelled set."""
    full = _frame(IDS, sep=4.0, seed=0)
    partial = _frame(IDS[:60], sep=4.0, seed=1)  # only the first 60 samples
    res = run_kfold_probe(
        {"full": FeatureSpec(full), "partial": FeatureSpec(partial)},
        LABEL_MAP, n_folds=3, seeds=(1,), evaluate_fraction=0.25,
    )
    assert res["config"]["n_universe"] == 60


def test_small_universe_raises() -> None:
    """Too few samples for the requested folds is a hard error, not a silent degenerate run."""
    tiny_ids = IDS[:4]
    specs = {"sep": FeatureSpec(_frame(tiny_ids, sep=4.0, seed=0))}
    with pytest.raises(ValueError, match="universe too small"):
        run_kfold_probe(specs, {s: LABEL_MAP[s] for s in tiny_ids}, n_folds=5, seeds=(1,))
