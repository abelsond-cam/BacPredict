"""Tests for per-stratum holdout scoring (:mod:`bacpredict.engine.finetune.stratified_metrics`)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bacpredict.engine.finetune.stratified_metrics import (
    NA_LABEL,
    OTHER_LABEL,
    bootstrap_auroc_ci,
    join_groups,
    load_eval_scores,
    resolve_sample_ids,
    stratified_metrics,
)


def _scored(groups: list[str], y: list[int], p: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"group": groups, "y_true": y, "y_prob": p})


def _separable(n: int, group: str, sep: float, seed: int = 0) -> pd.DataFrame:
    """A group of ``n`` genomes whose scores separate the classes by ``sep`` (0 = chance)."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    logit = rng.normal(0, 1, n) + y * sep
    return _scored([group] * n, y.tolist(), (1 / (1 + np.exp(-logit))).tolist())


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_point_estimate_for_a_separable_group():
    df = _separable(400, "SL258", sep=2.0)
    from sklearn.metrics import roc_auc_score

    auroc = roc_auc_score(df["y_true"], df["y_prob"])
    lo, hi, n_valid = bootstrap_auroc_ci(df["y_true"].to_numpy(), df["y_prob"].to_numpy(), n_boot=300)
    assert lo < auroc < hi
    assert n_valid > 250
    assert lo > 0.5  # a genuinely separable group's interval clears chance


def test_bootstrap_ci_is_wider_for_a_smaller_group():
    """The whole point of reporting CIs: a small group's interval must visibly widen."""
    big = _separable(800, "big", sep=1.2, seed=1)
    small = _separable(80, "small", sep=1.2, seed=1)
    lo_b, hi_b, _ = bootstrap_auroc_ci(big["y_true"].to_numpy(), big["y_prob"].to_numpy(), n_boot=300)
    lo_s, hi_s, _ = bootstrap_auroc_ci(small["y_true"].to_numpy(), small["y_prob"].to_numpy(), n_boot=300)
    assert (hi_s - lo_s) > (hi_b - lo_b)


def test_bootstrap_ci_is_nan_for_a_single_class_group():
    lo, hi, n_valid = bootstrap_auroc_ci(np.ones(50, dtype=int), np.linspace(0, 1, 50), n_boot=50)
    assert np.isnan(lo) and np.isnan(hi)
    assert n_valid == 0


# ---------------------------------------------------------------------------
# The per-group table
# ---------------------------------------------------------------------------


def test_group_counts_sum_to_the_holdout_and_pooled_row_is_first():
    df = pd.concat([_separable(300, "SL258", 1.5), _separable(50, "SLrare", 0.5)], ignore_index=True)
    out = stratified_metrics(df, min_group_n=100, n_boot=50)
    assert out.iloc[0]["group"] == "__pooled__"
    assert int(out.iloc[0]["n"]) == len(df)
    # Non-pooled rows must account for every holdout genome exactly once.
    assert int(out.loc[out["group"] != "__pooled__", "n"].sum()) == len(df)


def test_small_groups_are_pooled_into_other_and_never_dropped():
    parts = [_separable(200, "SL258", 1.5)]
    parts += [_separable(10, f"SLrare{i}", 0.0, seed=i) for i in range(12)]
    df = pd.concat(parts, ignore_index=True)
    out = stratified_metrics(df, min_group_n=100, n_boot=50)

    assert OTHER_LABEL in set(out["group"])
    other = out[out["group"] == OTHER_LABEL].iloc[0]
    assert int(other["n"]) == 120  # 12 groups x 10
    assert int(other["n_groups"]) == 12
    assert "SLrare0" not in set(out["group"])  # rolled up, not listed separately


def test_single_class_group_is_reported_with_nan_auroc_not_dropped():
    df = pd.concat(
        [_separable(200, "SL258", 1.5), _scored(["SLmono"] * 150, [1] * 150, list(np.linspace(0, 1, 150)))],
        ignore_index=True,
    )
    out = stratified_metrics(df, min_group_n=100, n_boot=50)
    mono = out[out["group"] == "SLmono"]
    assert len(mono) == 1
    assert bool(mono.iloc[0]["single_class"]) is True
    assert np.isnan(mono.iloc[0]["auroc"])
    assert int(mono.iloc[0]["n"]) == 150  # still counted


def test_recovers_a_planted_within_group_difference():
    """A strongly-separable group must score above a chance group by more than noise."""
    df = pd.concat([_separable(400, "strong", 2.5, seed=2), _separable(400, "chance", 0.0, seed=3)],
                   ignore_index=True)
    out = stratified_metrics(df, min_group_n=100, n_boot=200).set_index("group")
    assert out.loc["strong", "auroc"] > 0.8
    assert 0.4 < out.loc["chance", "auroc"] < 0.6
    # The intervals must not overlap — that is the claim the plot makes.
    assert out.loc["strong", "auroc_ci_lo"] > out.loc["chance", "auroc_ci_hi"]


def test_missing_required_column_raises():
    with pytest.raises(ValueError, match="missing required column"):
        stratified_metrics(pd.DataFrame({"group": ["a"], "y_true": [1]}), n_boot=10)


# ---------------------------------------------------------------------------
# Loading + joining
# ---------------------------------------------------------------------------


def test_load_and_join_round_trip(tmp_path):
    ids = [f"S{i}" for i in range(6)]
    npz = tmp_path / "eval_scores.npz"
    np.savez(
        npz,
        y_true=np.array([0, 1, 0, 1, 0, 1]),
        y_prob=np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
        sample_ids=np.asarray(ids, dtype=np.str_),
        drug=np.array("blood_vs_faeces_label"),
        operating_threshold=np.array(0.5),
    )
    meta = tmp_path / "split.csv"
    pd.DataFrame({"Sample": ids[:5], "Sublineage": ["SL258"] * 3 + ["SL147"] * 2}).to_csv(meta, index=False)

    scores = load_eval_scores(npz)
    assert scores["sample_ids"] == ids
    assert scores["drug"] == "blood_vs_faeces_label"

    got = resolve_sample_ids(scores, None, meta, "blood_vs_faeces_label")
    assert got == ids

    scored = join_groups(got, scores["y_true"], scores["y_prob"], meta, "Sublineage")
    assert len(scored) == 6
    # S5 has no metadata row -> explicit no-call group, not a dropped row.
    assert scored.loc[scored["Sample"] == "S5", "group"].iloc[0] == NA_LABEL


def test_resolve_sample_ids_rejects_a_length_mismatch(tmp_path):
    npz = tmp_path / "eval_scores.npz"
    np.savez(
        npz,
        y_true=np.array([0, 1, 0]),
        y_prob=np.array([0.1, 0.9, 0.2]),
        sample_ids=np.asarray(["A", "B"], dtype=np.str_),  # deliberately short
        drug=np.array("d"),
    )
    scores = load_eval_scores(npz)
    with pytest.raises(ValueError, match="refusing to join"):
        resolve_sample_ids(scores, None, tmp_path / "unused.csv", "d")


def _cohort_npz(path, ids, splits):
    """A whole-cohort npz as score_cohort.py writes it (carries the per-genome split array)."""
    n = len(ids)
    np.savez(
        path,
        y_true=np.array([i % 2 for i in range(n)]),
        y_prob=np.linspace(0.05, 0.95, n),
        sample_ids=np.asarray(ids, dtype=np.str_),
        split=np.asarray(splits, dtype=np.str_),
        drug=np.array("blood_vs_faeces_label"),
        operating_threshold=np.array(0.5),
    )


@pytest.mark.parametrize(
    ("scope", "expected_n"),
    [("evaluate", 4), ("validate", 3), ("heldout", 7), ("all", 12), ("train", 5)],
)
def test_restrict_split_selects_the_right_rows_through_the_cli(tmp_path, monkeypatch, scope, expected_n):
    """CLI-level: the parser and the restriction logic must agree.

    ``heldout`` is the only multi-split scope (validate + evaluate) — it exists because the whole-set
    scope is dominated by fitted-on train rows. Driven through ``_main_cli`` deliberately: a bug in
    the parser wiring (a choice the body never handles, or vice versa) is invisible to tests that
    call the functions directly, which is how an earlier missing ``--seed`` reached the cluster.
    """
    from bacpredict.engine.finetune.stratified_metrics import _main_cli

    ids = [f"S{i}" for i in range(12)]
    splits = ["train"] * 5 + ["validate"] * 3 + ["evaluate"] * 4
    npz = tmp_path / "cohort_scores.npz"
    _cohort_npz(npz, ids, splits)
    meta = tmp_path / "split.csv"
    pd.DataFrame({"Sample": ids, "Sublineage": ["SL258"] * 12}).to_csv(meta, index=False)
    out = tmp_path / f"per_sl_{scope}.csv"

    monkeypatch.setattr(
        "sys.argv",
        ["stratified_metrics", "--eval-scores", str(npz), "--metadata", str(meta),
         "--out", str(out), "--restrict-split", scope, "--min-group-n", "1", "--n-boot", "20"],
    )
    _main_cli()

    table = pd.read_csv(out)
    assert (table["split_scope"] == scope).all()
    # The pooled row carries the total n for the scope.
    assert int(table.iloc[0]["n"]) == expected_n


def test_missing_sample_ids_without_checkpoint_dir_raises_a_helpful_error(tmp_path):
    npz = tmp_path / "eval_scores.npz"
    np.savez(npz, y_true=np.array([0, 1]), y_prob=np.array([0.2, 0.8]), drug=np.array("d"))
    scores = load_eval_scores(npz)
    assert scores["sample_ids"] is None
    with pytest.raises(ValueError, match="--checkpoint-dir"):
        resolve_sample_ids(scores, None, tmp_path / "unused.csv", "d")


def test_join_groups_reports_a_missing_group_column(tmp_path):
    meta = tmp_path / "m.csv"
    pd.DataFrame({"Sample": ["A"], "Sublineage": ["SL1"]}).to_csv(meta, index=False)
    with pytest.raises(ValueError, match="Clonal group"):
        join_groups(["A"], np.array([1]), np.array([0.5]), meta, "Clonal group")
