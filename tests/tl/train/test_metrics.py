"""Tests for bacpredict.engine.finetune.metrics (compute_full_metrics, HF wrapper, JSON write)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import EvalPrediction

from bacpredict.engine.finetune.metrics import (
    REQUIRED_METRICS_KEYS,
    REQUIRED_TOP_LEVEL_KEYS,
    build_results_payload,
    compute_full_metrics,
    compute_metrics_binary_genome_pred,
    write_results_json,
    youden_threshold,
)


def _separable_arrays(n: int = 40):
    """Two well-separated probability streams that should give AUROC≈1.0."""
    rng = np.random.default_rng(0)
    y_true = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])
    y_prob = np.concatenate([rng.uniform(0.0, 0.3, n // 2), rng.uniform(0.7, 1.0, n // 2)])
    return y_true, y_prob


def test_compute_full_metrics_returns_all_required_keys():
    y_true, y_prob = _separable_arrays()
    result = compute_full_metrics(y_true, y_prob)
    assert REQUIRED_METRICS_KEYS.issubset(result), (
        f"Missing keys: {REQUIRED_METRICS_KEYS - set(result)}"
    )


def test_compute_full_metrics_confusion_matrix_shape():
    y_true, y_prob = _separable_arrays()
    cm = compute_full_metrics(y_true, y_prob)["confusion_matrix"]
    assert isinstance(cm, list)
    assert len(cm) == 2 and all(len(row) == 2 for row in cm)
    assert all(isinstance(v, int) for row in cm for v in row)


def test_compute_full_metrics_perfectly_separable_high_auroc():
    y_true, y_prob = _separable_arrays()
    metrics = compute_full_metrics(y_true, y_prob)
    assert metrics["auroc"] >= 0.95
    assert metrics["n_samples"] == len(y_true)
    assert metrics["prevalence"] == pytest.approx(0.5)
    assert metrics["sensitivity"] >= 0.9
    assert metrics["specificity"] >= 0.9


def test_compute_full_metrics_single_class_returns_nan_auroc_but_full_payload():
    """All-negative input: AUROC is undefined; payload still has every required key."""
    y_true = np.zeros(10, dtype=int)
    y_prob = np.full(10, 0.1)
    metrics = compute_full_metrics(y_true, y_prob)
    assert np.isnan(metrics["auroc"])
    assert REQUIRED_METRICS_KEYS.issubset(metrics)
    assert metrics["confusion_matrix"][1] == [0, 0]


def test_compute_full_metrics_empty_arrays_safe():
    metrics = compute_full_metrics([], [])
    assert metrics["n_samples"] == 0
    assert metrics["confusion_matrix"] == [[0, 0], [0, 0]]
    assert np.isnan(metrics["prevalence"])


def test_hf_wrapper_emits_eval_auroc():
    y_true, y_prob = _separable_arrays()
    logits = np.log(np.clip(y_prob, 1e-6, 1 - 1e-6) / (1 - np.clip(y_prob, 1e-6, 1 - 1e-6)))
    preds = EvalPrediction(predictions=logits.astype(np.float32), label_ids=y_true.astype(np.int64))
    result = compute_metrics_binary_genome_pred(preds)
    assert "eval_auroc" in result
    assert result["eval_auroc"] >= 0.95
    assert result["eval_nr_samples"] == len(y_true)


def test_hf_wrapper_respects_ignore_index():
    logits = torch.tensor([2.0, -2.0, 2.0, -2.0]).numpy()
    labels = np.array([1, 0, -100, -100], dtype=np.int64)
    preds = EvalPrediction(predictions=logits, label_ids=labels)
    result = compute_metrics_binary_genome_pred(preds)
    assert result["eval_nr_samples"] == 2


def test_write_results_json_roundtrips(tmp_path: Path):
    y_true, y_prob = _separable_arrays()
    metrics = compute_full_metrics(y_true, y_prob)
    payload = build_results_payload(
        task="kleb_ast",
        drug="ceftriaxone",
        model_name_or_path="macwiatrak/bacformer-large-masked-complete-genomes",
        checkpoint_dir=str(tmp_path / "ckpt"),
        split_source="csv",
        metrics=metrics,
        n_evaluate=len(y_true),
    )
    out = tmp_path / "subdir" / "results.json"
    write_results_json(out, payload)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert REQUIRED_TOP_LEVEL_KEYS.issubset(loaded)
    assert loaded["drug"] == "ceftriaxone"
    assert loaded["metrics"]["auroc"] == pytest.approx(metrics["auroc"])


def test_write_results_json_rejects_missing_top_level_key(tmp_path: Path):
    bad_payload = {
        "schema_version": "1.0",
        "task": "kleb_ast",
        "drug": "ceftriaxone",
        "model": {},
        "split": {},
        "metrics": dict.fromkeys(REQUIRED_METRICS_KEYS, 0),
        # missing: timestamp, host
    }
    with pytest.raises(ValueError, match="missing required top-level keys"):
        write_results_json(tmp_path / "bad.json", bad_payload)


def test_write_results_json_rejects_missing_metrics_key(tmp_path: Path):
    payload = build_results_payload(
        task="kleb_ast",
        drug="ceftriaxone",
        model_name_or_path="x",
        checkpoint_dir="x",
        split_source="csv",
        metrics={"auroc": 0.9},  # incomplete
    )
    with pytest.raises(ValueError, match="missing required metric keys"):
        write_results_json(tmp_path / "bad.json", payload)


def test_youden_threshold_between_clusters():
    # Negatives near 0.2, positives near 0.8 → optimal cut should land between them.
    y_true, y_prob = _separable_arrays()
    thr = youden_threshold(y_true, y_prob)
    assert 0.3 < thr < 0.75


def test_youden_threshold_single_class_returns_half():
    assert youden_threshold(np.zeros(8, dtype=int), np.full(8, 0.1)) == 0.5
    assert youden_threshold([], []) == 0.5


def test_operating_point_roundtrips_and_validates(tmp_path: Path):
    y_true, y_prob = _separable_arrays()
    metrics = compute_full_metrics(y_true, y_prob)
    op = compute_full_metrics(y_true, y_prob, threshold=0.4)
    operating_point = {
        "objective": "youden_j",
        "selected_on": "validation",
        "threshold": 0.4,
        "sensitivity": op["sensitivity"],
        "specificity": op["specificity"],
        "balanced_accuracy": op["balanced_accuracy"],
        "f1": op["f1"],
        "confusion_matrix": op["confusion_matrix"],
    }
    payload = build_results_payload(
        task="kleb_ast",
        drug="ceftriaxone",
        model_name_or_path="m",
        checkpoint_dir="c",
        split_source="kfold",
        metrics=metrics,
        operating_point=operating_point,
    )
    out = tmp_path / "results.json"
    write_results_json(out, payload)  # operating_point is optional → still validates
    loaded = json.loads(out.read_text())
    assert loaded["operating_point"]["objective"] == "youden_j"
    assert loaded["operating_point"]["threshold"] == 0.4
