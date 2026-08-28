"""The paired comparison must refuse two arms that did not score the same genomes."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bac_pyseer.ast_gwas.compare_vocab_arms import compare_drug, load_arm, run

RNG = np.random.default_rng(0)


def _scores(path, ids, y, p):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, sample_ids=np.asarray(ids, dtype=np.str_),
             y_true=np.asarray(y, dtype=int), y_prob=np.asarray(p, dtype=float))
    return path


def _cohort(n=120, seed=0):
    """A holdout with real signal, so AUROC is well away from 0.5 and the CI is meaningful."""
    rng = np.random.default_rng(seed)
    ids = [f"S{i:04d}" for i in range(n)]
    y = (rng.random(n) < 0.4).astype(int)
    strong = np.clip(y * 0.45 + rng.normal(0.3, 0.12, n), 0, 1)   # the flattered arm
    weak = np.clip(y * 0.30 + rng.normal(0.3, 0.16, n), 0, 1)     # the honest arm
    return ids, y, strong, weak


def test_pairs_by_id_regardless_of_row_order(tmp_path):
    """The two arms write rows in whatever order load_splits gave; pairing must not assume a match."""
    ids, y, a, b = _cohort()
    order = RNG.permutation(len(ids))
    _scores(tmp_path / "a.npz", ids, y, a)
    _scores(tmp_path / "b.npz", [ids[i] for i in order], y[order], b[order])

    row = compare_drug("colistin", tmp_path / "a.npz", tmp_path / "b.npz", n_boot=200)
    # Same comparison with both arms written in the same order must give an identical delta.
    _scores(tmp_path / "b2.npz", ids, y, b)
    same = compare_drug("colistin", tmp_path / "a.npz", tmp_path / "b2.npz", n_boot=200)
    assert row["delta"] == pytest.approx(same["delta"])
    assert row["full_cohort_auroc"] > row["trainval_vocab_auroc"]
    assert row["delta"] > 0          # positive == the old arm scored higher
    assert row["n_holdout"] == len(ids)


def test_load_arm_sorts_by_sample_id(tmp_path):
    _scores(tmp_path / "s.npz", ["S3", "S1", "S2"], [1, 0, 1], [0.9, 0.1, 0.8])
    ids, y, p = load_arm(tmp_path / "s.npz")
    assert list(ids) == ["S1", "S2", "S3"]
    assert list(y) == [0, 1, 1]
    assert list(p) == [0.1, 0.8, 0.9]


def test_different_holdout_sets_of_equal_size_are_refused(tmp_path):
    """The failure counts alone cannot see: same n, different genomes."""
    ids, y, a, b = _cohort(n=100)
    other = [f"T{i:04d}" for i in range(100)]
    _scores(tmp_path / "a.npz", ids, y, a)
    _scores(tmp_path / "b.npz", other, y, b)
    with pytest.raises(SystemExit) as e:
        compare_drug("colistin", tmp_path / "a.npz", tmp_path / "b.npz", n_boot=50)
    assert "different holdout genomes" in str(e.value)
    assert "100 vs 100" in str(e.value)


def test_disagreeing_labels_are_refused(tmp_path):
    ids, y, a, b = _cohort(n=100)
    flipped = y.copy()
    flipped[0] = 1 - flipped[0]
    _scores(tmp_path / "a.npz", ids, y, a)
    _scores(tmp_path / "b.npz", ids, flipped, b)
    with pytest.raises(SystemExit) as e:
        compare_drug("colistin", tmp_path / "a.npz", tmp_path / "b.npz", n_boot=50)
    assert "different label" in str(e.value)


def test_identical_arms_give_zero_delta_that_does_not_separate(tmp_path):
    ids, y, a, _ = _cohort()
    _scores(tmp_path / "a.npz", ids, y, a)
    _scores(tmp_path / "b.npz", ids, y, a)
    row = compare_drug("colistin", tmp_path / "a.npz", tmp_path / "b.npz", n_boot=300)
    assert row["delta"] == pytest.approx(0.0)
    assert row["separates_from_zero"] is False


def test_confound_columns_are_carried_from_both_arms(tmp_path):
    ids, y, a, b = _cohort()
    _scores(tmp_path / "a.npz", ids, y, a)
    _scores(tmp_path / "b.npz", ids, y, b)
    (tmp_path / "fr.json").write_text(json.dumps(
        {"extra": {"n_unitigs": 40000, "C": 0.01, "gwas_summary": {"n_patterns": 900, "threshold": 5.5e-8}}}))
    (tmp_path / "tr.json").write_text(json.dumps(
        {"extra": {"n_unitigs": 51000, "C": 0.1, "gwas_summary": {"n_patterns": 1200, "threshold": 4.1e-8}}}))
    (tmp_path / "audit.json").write_text(json.dumps({"reflist": {"min_samples_floor": 12, "n_reflist": 1128}}))

    row = compare_drug(
        "colistin", tmp_path / "a.npz", tmp_path / "b.npz",
        full_results=tmp_path / "fr.json", trainval_results=tmp_path / "tr.json",
        trainval_audit=tmp_path / "audit.json", n_boot=100,
    )
    assert row["full_n_unitigs"] == 40000
    assert row["trainval_n_unitigs"] == 51000
    assert row["full_n_patterns"] == 900
    assert row["trainval_n_patterns"] == 1200
    # The MAF-floor confound must be visible in the row, not only in the docstring.
    assert row["trainval_min_samples"] == 12
    assert row["trainval_n_reflist"] == 1128


def test_run_lays_out_the_rebuild_directory_shape(tmp_path, capsys):
    """vocab arm lives at <vocab>/<drug>/<drug>/lr, full arm at <full>/<drug>/lr."""
    full, vocab = tmp_path / "kp", tmp_path / "kp_trainval_vocab"
    for drug in ("colistin", "ertapenem"):
        ids, y, a, b = _cohort(seed=hash(drug) % 100)
        _scores(full / drug / "lr" / "eval_scores.npz", ids, y, a)
        _scores(vocab / drug / drug / "lr" / "eval_scores.npz", ids, y, b)
    out = tmp_path / "cmp.csv"
    assert run(full, vocab, out, n_boot=100) == 0
    text = capsys.readouterr().out
    assert "2 drug(s) compared" in text
    assert "NOT all leakage" in text
    assert out.exists()
    assert len(out.read_text().strip().splitlines()) == 3


def test_run_reports_a_drug_missing_one_arm_rather_than_dropping_it(tmp_path, capsys):
    full, vocab = tmp_path / "kp", tmp_path / "kp_trainval_vocab"
    ids, y, a, b = _cohort()
    _scores(full / "colistin" / "lr" / "eval_scores.npz", ids, y, a)
    _scores(vocab / "colistin" / "colistin" / "lr" / "eval_scores.npz", ids, y, b)
    (vocab / "ertapenem").mkdir(parents=True)          # built, but no read-out yet
    assert run(full, vocab, None, n_boot=100) == 0
    text = capsys.readouterr().out
    assert "skipped: ertapenem" in text
    assert "1 drug(s) compared" in text


def test_run_with_nothing_to_compare_is_an_error(tmp_path):
    (tmp_path / "kp").mkdir()
    (tmp_path / "kp_trainval_vocab" / "colistin").mkdir(parents=True)
    with pytest.raises(SystemExit):
        run(tmp_path / "kp", tmp_path / "kp_trainval_vocab", None, n_boot=10)
