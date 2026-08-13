"""Tests for the all_samples_2 split.

The entire comparison rests on one property: no genome may be in both the frozen test set and the
training pool. If that leaks, the new model's within-lineage AUROC is meaningless and would look
*better*, which is the direction that would fool us.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kleb_iso_source.build_all_samples_2_split import build_split


def _write(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _cohort_rows(ids, labels, splits):
    return [{"Sample": s, "blood_vs_faeces_label": lab, "train_val_eval": sp}
            for s, lab, sp in zip(ids, labels, splits, strict=True)]


def test_frozen_test_genomes_never_appear_in_training(tmp_path):
    frozen = _write(tmp_path / "frozen.csv", _cohort_rows(
        ["A", "B", "C", "D"], [1, 0, 1, 0], ["evaluate", "evaluate", "train", "validate"]))
    source = _write(tmp_path / "source.csv", _cohort_rows(
        [f"S{i}" for i in range(20)] + ["A", "B", "C", "D"],
        [i % 2 for i in range(20)] + [1, 0, 1, 0],
        ["train"] * 24))

    out, manifest = build_split(frozen, source, validate_frac=0.2, seed=0)
    test_ids = set(out.loc[out.train_val_eval == "evaluate", "Sample"])
    train_ids = set(out.loc[out.train_val_eval != "evaluate", "Sample"])

    assert test_ids == {"A", "B"}, "only the frozen cohort's evaluate rows become the test set"
    assert not (test_ids & train_ids), "a genome in both splits would invalidate the comparison"
    # C and D were train/validate in the frozen cohort, so they are legitimately trainable here.
    assert {"C", "D"} <= train_ids


def test_every_source_genome_is_accounted_for(tmp_path):
    frozen = _write(tmp_path / "frozen.csv", _cohort_rows(["A"], [1], ["evaluate"]))
    source = _write(tmp_path / "source.csv", _cohort_rows(
        [f"S{i}" for i in range(10)] + ["A"], [i % 2 for i in range(10)] + [1], ["train"] * 11))

    out, manifest = build_split(frozen, source, validate_frac=0.1, seed=0)
    assert len(out) == 11
    assert manifest["n_total"] == 11
    assert sum(manifest["split_counts"].values()) == 11


def test_frozen_genomes_absent_from_the_source_are_reported(tmp_path):
    """The new model cannot score a genome its cohort lacks; the count must surface, not vanish."""
    frozen = _write(tmp_path / "frozen.csv", _cohort_rows(
        ["A", "GHOST"], [1, 0], ["evaluate", "evaluate"]))
    source = _write(tmp_path / "source.csv", _cohort_rows(
        ["A"] + [f"S{i}" for i in range(9)], [1] + [i % 2 for i in range(9)], ["train"] * 10))

    _out, manifest = build_split(frozen, source, validate_frac=0.1, seed=0)
    assert manifest["n_frozen_test_requested"] == 2
    assert manifest["n_frozen_test_present"] == 1
    assert manifest["n_frozen_test_missing_from_source"] == 1


def test_ambiguous_labels_are_excluded(tmp_path):
    frozen = _write(tmp_path / "frozen.csv", _cohort_rows(["A"], [1], ["evaluate"]))
    source = _write(tmp_path / "source.csv", [
        {"Sample": "A", "blood_vs_faeces_label": 1, "train_val_eval": "train"},
        {"Sample": "S1", "blood_vs_faeces_label": 0, "train_val_eval": "train"},
        {"Sample": "S2", "blood_vs_faeces_label": 0.5, "train_val_eval": "train"},
    ])
    out, _ = build_split(frozen, source, validate_frac=0.5, seed=0)
    assert "S2" not in set(out["Sample"])


def test_missing_evaluate_rows_fail_loudly(tmp_path):
    frozen = _write(tmp_path / "frozen.csv", _cohort_rows(["A"], [1], ["train"]))
    source = _write(tmp_path / "source.csv", _cohort_rows(["A"], [1], ["train"]))
    with pytest.raises(SystemExit, match="no evaluate rows"):
        build_split(frozen, source)
