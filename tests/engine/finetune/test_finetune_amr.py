"""Tests for bacpredict.engine.finetune.finetune_amr argument validation.

The precision/pooling/panel-mode guards fire before any file or model access, so they can be exercised
with dummy paths — no embeddings store or GPU required.
"""

import pytest

from bacpredict.engine.finetune.finetune_amr import run


def test_run_rejects_invalid_precision():
    with pytest.raises(ValueError, match="precision"):
        run("x", "x", "x", "x", precision="fp16")


def test_run_rejects_invalid_pooling():
    with pytest.raises(ValueError, match="pooling"):
        run("x", "x", "x", "x", pooling="max")


def test_run_accepts_bf16_and_fp32_precision(monkeypatch):
    """Both valid precisions pass validation; we stop the run right after by pointing at a missing sheet."""
    for precision in ("bf16", "fp32"):
        # A non-existent ast_sheet raises FileNotFoundError *after* the precision guard, proving the
        # guard accepted the value (a ValueError about precision would mean it rejected it).
        with pytest.raises((FileNotFoundError, ValueError)) as exc:
            run("x", "x", "x", "/no/such/sheet.csv", precision=precision, n_samples=10)
        assert "precision" not in str(exc.value)
