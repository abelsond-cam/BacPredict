"""Shared metrics + results-JSON helpers for AMR / phenotype tasks.

Provides the canonical §0.4 metrics block (AUROC, AUPRC, sensitivity,
specificity, balanced accuracy, F1, confusion matrix, calibration) plus a
Hugging Face Trainer-shaped wrapper and a JSON-write helper that enforces
the v1 results schema. See ``docs/results_schema.md``.
"""

from __future__ import annotations

import json
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from transformers import EvalPrediction

SCHEMA_VERSION = "1.2"
REQUIRED_TOP_LEVEL_KEYS = {
    "schema_version", "task", "drug", "model", "split", "metrics", "timestamp", "host",
}
REQUIRED_METRICS_KEYS = {
    "auroc", "auprc", "sensitivity", "specificity", "balanced_accuracy", "f1",
    "prevalence", "n_samples", "confusion_matrix", "threshold", "calibration",
}


def compute_full_metrics(
    y_true: np.ndarray | list,
    y_prob: np.ndarray | list,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute the §0.4 metrics block on a binary classification task.

    Parameters
    ----------
    y_true
        Ground-truth binary labels (0/1), shape (N,).
    y_prob
        Predicted probabilities for the positive class, shape (N,).
    threshold
        Decision threshold for hard labels. Default 0.5.
    n_bins
        Calibration bin count. Default 10.

    Returns
    -------
    dict
        Keys: auroc, auprc, sensitivity, specificity, balanced_accuracy, f1,
        prevalence, n_samples, confusion_matrix ([[tn, fp], [fn, tp]]),
        threshold, calibration ({prob_true, prob_pred, n_bins}).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    n = int(y_true.size)

    nan = float("nan")
    if n == 0:
        return {
            "auroc": nan, "auprc": nan, "sensitivity": nan, "specificity": nan,
            "balanced_accuracy": nan, "f1": nan, "prevalence": nan, "n_samples": 0,
            "confusion_matrix": [[0, 0], [0, 0]], "threshold": float(threshold),
            "calibration": {"prob_true": [], "prob_pred": [], "n_bins": int(n_bins)},
        }

    def _safe(fn):
        try:
            return float(fn())
        except ValueError:
            return nan

    auroc_val = _safe(lambda: roc_auc_score(y_true, y_prob))
    auprc_val = _safe(lambda: average_precision_score(y_true, y_prob))
    f1_val = _safe(lambda: f1_score(y_true, y_pred, average="binary", zero_division=0))
    bal_acc = _safe(lambda: balanced_accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn_count, tp = (int(v) for v in cm.ravel())
    sens = float(tp / (tp + fn_count)) if (tp + fn_count) > 0 else nan
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else nan
    prevalence = float(y_true.mean())

    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        calibration = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "n_bins": int(n_bins),
        }
    except ValueError:
        calibration = {"prob_true": [], "prob_pred": [], "n_bins": int(n_bins)}

    return {
        "auroc": auroc_val,
        "auprc": auprc_val,
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": bal_acc,
        "f1": f1_val,
        "prevalence": prevalence,
        "n_samples": n,
        "confusion_matrix": [[tn, fp], [fn_count, tp]],
        "threshold": float(threshold),
        "calibration": calibration,
    }


def compute_metrics_binary_genome_pred(
    preds: EvalPrediction,
    ignore_index: int = -100,
    prefix: str = "eval",
) -> dict[str, float]:
    """HF-Trainer-shaped wrapper around :func:`compute_full_metrics`.

    Re-keys the metrics dict with the ``eval_`` prefix expected by
    ``metric_for_best_model="eval_auroc"``. Calibration is omitted because the
    HF Trainer expects flat scalar metrics.
    """
    logits = torch.as_tensor(preds.predictions).flatten()
    labels = torch.as_tensor(preds.label_ids).flatten().long()
    if (labels == ignore_index).any():
        keep = labels != ignore_index
        logits, labels = logits[keep], labels[keep]
    prob = torch.sigmoid(logits.float()).cpu().numpy()
    y_true = labels.cpu().numpy()

    full = compute_full_metrics(y_true, prob)
    y_pred = (prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred)) if y_true.size > 0 else float("nan")

    return {
        f"{prefix}_accuracy": acc,
        f"{prefix}_auroc": full["auroc"],
        f"{prefix}_auprc": full["auprc"],
        f"{prefix}_f1": full["f1"],
        f"{prefix}_balanced_accuracy": full["balanced_accuracy"],
        f"{prefix}_nr_samples": full["n_samples"],
    }


def youden_threshold(y_true: np.ndarray | list, y_prob: np.ndarray | list) -> float:
    """Probability threshold maximizing Youden's J statistic (``tpr - fpr``).

    Derived from the ROC curve — the operating point with the best balance of
    sensitivity and specificity. Returns 0.5 when no informative point exists
    (empty input or a single class present).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best = int(np.argmax(tpr - fpr))
    thr = float(thresholds[best])
    # roc_curve prepends an +inf threshold (the "classify nothing positive" point).
    return thr if np.isfinite(thr) else 1.0


def _git_sha() -> str | None:
    # Resolve HEAD in THIS module's directory (inside the repo/worktree), not the process cwd. An FT
    # job launched from an arbitrary cwd with PYTHONPATH pointing at a worktree otherwise records null
    # (exactly what happened on the Isambard runs).
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=Path(__file__).resolve().parent,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def build_results_payload(
    *,
    task: str,
    drug: str,
    model_name_or_path: str,
    checkpoint_dir: str,
    split_source: str,
    metrics: dict[str, Any],
    evaluate_seed: int | None = None,
    n_folds: int | None = None,
    fold: int | None = None,
    n_evaluate: int | None = None,
    model_revision: str | None = None,
    run_config: dict[str, Any] | None = None,
    versions: dict[str, Any] | None = None,
    operating_point: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a results payload.

    ``metrics`` is the §0.4 block at the default 0.5 threshold. ``model_revision`` is the base-model HF
    commit hash; ``run_config`` records precision + the training hyperparameters and ``versions`` the
    torch/transformers versions — so a bf16-vs-fp32 or cross-cluster run is self-documenting (schema v1.2).
    ``operating_point`` (optional, schema v1.1) holds tuned-threshold metrics — see ``docs/results_schema.md``.
    """
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "drug": drug,
        "model": {
            "name_or_path": model_name_or_path,
            "revision": model_revision,
            "git_sha": _git_sha(),
            "checkpoint_dir": checkpoint_dir,
        },
        "split": {
            "source": split_source,
            "evaluate_seed": evaluate_seed,
            "n_folds": n_folds,
            "fold": fold,
            "n_evaluate": n_evaluate,
        },
        "metrics": metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
    }
    if run_config:
        payload["run"] = run_config
    if versions:
        payload["versions"] = versions
    if operating_point is not None:
        payload["operating_point"] = operating_point
    if extra:
        payload["extra"] = extra
    return payload


def write_results_json(path: Path | str, payload: dict[str, Any]) -> None:
    """Validate the payload against the v1 schema and write it to JSON."""
    missing_top = REQUIRED_TOP_LEVEL_KEYS - set(payload)
    if missing_top:
        raise ValueError(f"Results payload missing required top-level keys: {sorted(missing_top)}")
    missing_metrics = REQUIRED_METRICS_KEYS - set(payload["metrics"])
    if missing_metrics:
        raise ValueError(f"Results payload missing required metric keys: {sorted(missing_metrics)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
