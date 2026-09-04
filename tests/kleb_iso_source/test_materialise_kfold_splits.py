"""Tests for the k-fold sweep's split table.

The point of the module is that two arms hours and machines apart agree about who is held out, so the
tests are about *identity* of genome sets, not about counts. A count test would have passed for the
partition that caused the 2026-07 read-out leak.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bacpredict.engine.splits.generate_kfold_splits import generate_kfold_splits
from kleb_iso_source.materialise_kfold_splits import (
    FITTING,
    HOLDOUT,
    SPLIT_COL,
    _main,
    materialise,
    trainer_universe,
    verify_deployed_holdout,
)

LABEL = "blood_vs_faeces_label"


def _sheet(n: int = 200, n_unlabelled: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    label = rng.integers(0, 2, n).astype(float)
    label[:n_unlabelled] = np.nan
    return pd.DataFrame({
        "Sample": [f"SAM{i:05d}" for i in range(n)],
        LABEL: label,
        "Sublineage": rng.choice(["SL258", "SL307", "SL15"], n),
    })


@pytest.fixture
def labeled() -> pd.DataFrame:
    return _sheet()[lambda d: d[LABEL].notna()].copy()


def test_universe_reproduces_the_trainer_rule(tmp_path):
    """label.notna() and nothing else — an extra filter here silently moves the holdout."""
    sheet = tmp_path / "sheet.csv"
    _sheet(n=50, n_unlabelled=7).to_csv(sheet, index=False)
    got = trainer_universe(sheet, LABEL)
    assert len(got) == 43
    assert got["Sample"].dtype == object and got[LABEL].notna().all()


def test_universe_accepts_the_sample_accession_alias(tmp_path):
    sheet = tmp_path / "sheet.csv"
    df = _sheet(n=20, n_unlabelled=0).rename(columns={"Sample": "sample_accession"})
    df.to_csv(sheet, index=False)
    assert trainer_universe(sheet, LABEL)["Sample"].tolist()[:2] == ["SAM00000", "SAM00001"]


def test_universe_rejects_a_missing_label_column(tmp_path):
    sheet = tmp_path / "sheet.csv"
    _sheet(n=10, n_unlabelled=0).to_csv(sheet, index=False)
    with pytest.raises(ValueError, match="no label column"):
        trainer_universe(sheet, "not_a_column")


def test_holdout_is_identical_across_every_seed(labeled):
    """The invariant the whole sweep rests on — asserted by genome identity, never by n."""
    selection, _, manifest = materialise(labeled, label_column=LABEL)
    held = set(selection.loc[selection[SPLIT_COL] == HOLDOUT, "Sample"])
    for seed in manifest["params"]["seeds"]:
        direct, _ = generate_kfold_splits(labeled, n_folds=5, seed=seed, evaluate_seed=1)
        assert direct == held


def test_fitting_set_is_identical_across_every_run(labeled):
    """train ∪ validate does not move — which is why the unitig arm is one model, not fifteen."""
    selection, _, _ = materialise(labeled, label_column=LABEL)
    fitting = set(selection.loc[selection[SPLIT_COL] == FITTING, "Sample"])
    for seed in (1, 2, 3):
        _, folds = generate_kfold_splits(labeled, n_folds=5, seed=seed, evaluate_seed=1)
        for tr, va in folds:
            assert tr | va == fitting


def test_selection_partitions_the_universe_exactly(labeled):
    selection, _, manifest = materialise(labeled, label_column=LABEL)
    assert set(selection["Sample"]) == set(labeled["Sample"])
    assert len(selection) == selection["Sample"].nunique() == manifest["n_universe"]
    assert set(selection[SPLIT_COL]) == {FITTING, HOLDOUT}
    assert manifest["n_fitting"] + manifest["n_holdout"] == manifest["n_universe"]


def test_fold_assignments_cover_each_seed_and_flag_the_holdout(labeled):
    selection, assignments, _ = materialise(labeled, label_column=LABEL)
    held = selection[SPLIT_COL] == HOLDOUT
    for seed in (1, 2, 3):
        col = assignments[f"validate_fold_seed{seed}"]
        assert (col[held.to_numpy()] == -1).all()
        assert sorted(col[~held.to_numpy()].unique()) == [0, 1, 2, 3, 4]


def test_run_grid_is_the_slurm_array_mapping(labeled):
    """task_id must decode as FOLD = id % 5, SEED = id // 5 + 1, or the array trains the wrong grid."""
    _, _, manifest = materialise(labeled, label_column=LABEL)
    runs = manifest["runs"]
    assert len(runs) == 15
    assert sorted(r["task_id"] for r in runs) == list(range(15))
    for r in runs:
        assert r["fold"] == r["task_id"] % 5
        assert r["seed"] == r["task_id"] // 5 + 1
        assert r["n_train"] + r["n_validate"] == manifest["n_fitting"]


def test_a_moving_holdout_is_refused(labeled, monkeypatch):
    """If evaluate_seed ever stopped pinning the holdout, the sweep would be silently incomparable."""
    import kleb_iso_source.materialise_kfold_splits as mod

    real = mod.generate_kfold_splits
    calls = {"n": 0}

    def drifting(df, **kw):
        held, folds = real(df, **kw)
        calls["n"] += 1
        if calls["n"] > 1:  # second seed loses one holdout genome
            held = set(sorted(held)[1:])
        return held, folds

    monkeypatch.setattr(mod, "generate_kfold_splits", drifting)
    with pytest.raises(ValueError, match="holdout moved"):
        mod.materialise(labeled, label_column=LABEL)


def test_single_seed_is_allowed_and_no_seeds_is_not(labeled):
    _, _, manifest = materialise(labeled, label_column=LABEL, seeds=(1,))
    assert len(manifest["runs"]) == 5
    with pytest.raises(ValueError, match="at least one seed"):
        materialise(labeled, label_column=LABEL, seeds=())


def test_digest_is_order_independent_and_set_sensitive(labeled):
    _, _, manifest = materialise(labeled, label_column=LABEL)
    held = sorted(set(labeled["Sample"]))[:10]
    a = verify_deployed_holdout({"n_holdout": 10, "holdout_digest": ""}, held)
    b = verify_deployed_holdout({"n_holdout": 10, "holdout_digest": ""}, list(reversed(held)))
    c = verify_deployed_holdout({"n_holdout": 10, "holdout_digest": ""}, held[:-1] + ["SAM99999"])
    assert a["scored_digest"] == b["scored_digest"] != c["scored_digest"]


def test_verify_matches_the_real_holdout(labeled):
    selection, _, manifest = materialise(labeled, label_column=LABEL)
    held = selection.loc[selection[SPLIT_COL] == HOLDOUT, "Sample"].tolist()
    assert verify_deployed_holdout(manifest, held)["matches"] is True
    swapped = held[:-1] + [selection.loc[selection[SPLIT_COL] == FITTING, "Sample"].iloc[0]]
    bad = verify_deployed_holdout(manifest, swapped)
    assert bad["matches"] is False and bad["n_scored"] == bad["n_holdout"]  # n agrees, identity does not


def test_cli_build_then_verify_round_trip(tmp_path):
    sheet = tmp_path / "sheet.csv"
    _sheet().to_csv(sheet, index=False)
    out = tmp_path / "sweep"
    _main(["--sheet-path", str(sheet), "--out-dir", str(out), "--label-column", LABEL])

    manifest = json.loads((out / "kfold_splits_manifest.json").read_text())
    selection = pd.read_csv(out / "kfold_selection_split.csv")
    assert set(selection.columns) == {"Sample", LABEL, SPLIT_COL}
    assert len(pd.read_csv(out / "kfold_fold_assignments.csv")) == manifest["n_universe"]

    npz = tmp_path / "eval_scores.npz"
    held = selection.loc[selection[SPLIT_COL] == HOLDOUT, "Sample"].to_numpy().astype(np.str_)
    np.savez(npz, sample_ids=held, y_prob=np.zeros(len(held)))
    _main(["--out-dir", str(out), "--verify-eval-scores", str(npz)])


def test_cli_verify_refuses_an_npz_without_sample_ids(tmp_path):
    """The deployed models/eval_scores.npz predates the field; a count check would prove nothing."""
    sheet = tmp_path / "sheet.csv"
    _sheet().to_csv(sheet, index=False)
    out = tmp_path / "sweep"
    _main(["--sheet-path", str(sheet), "--out-dir", str(out), "--label-column", LABEL])
    npz = tmp_path / "old.npz"
    np.savez(npz, y_true=np.zeros(3), y_prob=np.zeros(3))
    with pytest.raises(SystemExit, match="no 'sample_ids'"):
        _main(["--out-dir", str(out), "--verify-eval-scores", str(npz)])


def test_cli_verify_exits_on_a_mismatch(tmp_path):
    sheet = tmp_path / "sheet.csv"
    _sheet().to_csv(sheet, index=False)
    out = tmp_path / "sweep"
    _main(["--sheet-path", str(sheet), "--out-dir", str(out), "--label-column", LABEL])
    npz = tmp_path / "wrong.npz"
    np.savez(npz, sample_ids=np.array(["SAM00000", "SAM00001"], dtype=np.str_))
    with pytest.raises(SystemExit, match="MISMATCH"):
        _main(["--out-dir", str(out), "--verify-eval-scores", str(npz)])
