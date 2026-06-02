"""Tests for tl.train.predict (predict_proba).

The heavy inference path is exercised end-to-end by the evaluator tests + the
SLURM job; these tests cover the surface contract (empty input, type/shape,
delegation to run_inference preserves order).
"""

import numpy as np
import pytest

from tl.train import predict as predict_module


def test_predict_proba_empty_returns_empty_array():
    out = predict_module.predict_proba(
        checkpoint="ignored",  # never touched when sample_ids is empty
        sample_ids=[],
        embeddings_dir="/tmp",
        device="cpu",
    )
    assert isinstance(out, np.ndarray)
    assert out.dtype == float
    assert out.size == 0


def test_predict_proba_preserves_sample_order(monkeypatch, tmp_path):
    """predict_proba must return probabilities in the same order as sample_ids."""

    # Monkey-patch the model load and inference to avoid loading bacformer.
    sample_ids = ["S001", "S002", "S003", "S004"]
    expected_probs = np.array([0.10, 0.85, 0.30, 0.95])

    class _DummyModel:
        def to(self, device):
            return self

        def float(self):
            return self

        def parameters(self):
            import torch
            yield torch.tensor([0.0])

    monkeypatch.setattr(
        predict_module, "resolve_checkpoint_dir", lambda p: tmp_path
    )
    monkeypatch.setattr(
        predict_module.AutoModelForSequenceClassification,
        "from_pretrained",
        classmethod(lambda cls, *a, **kw: _DummyModel()),
    )

    def _fake_run_inference(model, loader, device):
        # Walk the loader (which yields batches in order) and return synthetic probs.
        order = []
        for batch in loader:
            for _ in range(batch["labels"].shape[0]):
                order.append(len(order))
        # Match input order — DataLoader with shuffle=False preserves it.
        return np.zeros(len(order), dtype=int), expected_probs[: len(order)]

    monkeypatch.setattr(predict_module, "run_inference", _fake_run_inference)

    # We still need real .pt files for LabelInjectingFileDataset to read.
    import torch

    for sid in sample_ids:
        torch.save(
            {"prot_embeddings": torch.randn(4, 8), "attention_mask": torch.ones(4)},
            tmp_path / f"{sid}_esm_embeddings.pt",
        )

    out = predict_module.predict_proba(
        checkpoint=tmp_path,
        sample_ids=sample_ids,
        embeddings_dir=tmp_path,
        device="cpu",
        num_workers=0,
    )
    assert out.shape == (4,)
    assert np.allclose(out, expected_probs)


def test_predict_proba_raises_on_missing_embedding(tmp_path):
    """Caller is responsible for pre-filtering; we surface a FileNotFoundError if not."""

    # No .pt files exist for these IDs.
    with pytest.raises((FileNotFoundError, RuntimeError)):
        predict_module.predict_proba(
            checkpoint=tmp_path,
            sample_ids=["MISSING_ID"],
            embeddings_dir=tmp_path,
            device="cpu",
            num_workers=0,
        )
