"""Unit tests for the full-eval attention-pool scorer (``pangena_predict.eval_attn_pool_on_full_split``).

Cover the leakage-exclusion of manifest-seen genomes, the manifest split CSV reader, and the
per-genome scoring loop (label alignment + missing-``.pt`` skip) against a fake model + synthetic
``.pt`` stores — no HPC data, no real checkpoint. Skipped where torch is unavailable.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

import torch

import pangena_predict.eval_attn_pool_on_full_split as eafs

DIM = 4


def test_select_eval_ids_excludes_seen() -> None:
    """Manifest train+validate genomes are dropped from the full eval, order preserved, count right."""
    eval_ids = ["e0", "s1", "e2", "s3", "e4"]  # s1, s3 were manifest-seen
    kept, n_excluded = eafs.select_eval_ids(eval_ids, {"s1", "s3"})
    assert kept == ["e0", "e2", "e4"]
    assert n_excluded == 2


def test_load_manifest_splits_reads_split_and_drops_ambiguous(tmp_path: Path) -> None:
    """The split CSV is partitioned by ``train_val_eval`` with only clean 0/1 labels kept."""
    csv = tmp_path / "manifest_split.csv"
    pd.DataFrame(
        {
            "Sample": ["t0", "t1", "v0", "e0", "e1", "amb0"],
            "rifampin": [1, 0, 1, 0, 1, 0.5],  # amb0 is ambiguous -> dropped
            "train_val_eval": ["train", "train", "validate", "evaluate", "evaluate", "evaluate"],
        }
    ).to_csv(csv, index=False)

    train, validate, evaluate, label_map = eafs.load_manifest_splits(csv, "rifampin")
    assert train == ["t0", "t1"]
    assert validate == ["v0"]
    assert evaluate == ["e0", "e1"]  # amb0 dropped (not 0/1)
    assert label_map == {"t0": 1, "t1": 0, "v0": 1, "e0": 0, "e1": 1}


class _FakeModel(torch.nn.Module):
    """Fake attention-pool model: logit = mean(protein_embeddings) so probs track the store value."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))  # float32 -> drives model dtype
        self.config = SimpleNamespace(panel_mode="none")

    def forward(self, protein_embeddings, attention_mask=None, contig_ids=None, **kwargs):
        return SimpleNamespace(logits=protein_embeddings.float().mean().view(1, 1))


def _write_store(path: Path, fill: float, n_proteins: int = 3) -> None:
    """A plain per-protein ``.pt`` store (the TB ESM-C layout) filled with a constant value."""
    torch.save(
        {
            "protein_embeddings": torch.full((1, n_proteins, DIM), fill),
            "attention_mask": torch.ones(1, n_proteins),
            "contig_ids": torch.zeros(1, n_proteins, dtype=torch.long),
        },
        path,
    )


def test_score_ids_aligns_labels_and_skips_missing(tmp_path: Path) -> None:
    """Probs follow the store fill (monotone), labels align by id, and a missing ``.pt`` is skipped."""
    store_dir = tmp_path / "emb"
    store_dir.mkdir()
    _write_store(store_dir / "a_esm_embeddings.pt", fill=-2.0)
    _write_store(store_dir / "b_esm_embeddings.pt", fill=2.0)
    # "c" has no .pt on disk -> must be skipped, not crash.
    label_map = {"a": 0, "b": 1, "c": 1}

    y_true, y_prob, kept, skips = eafs._score_ids(
        _FakeModel(), store_dir, ["a", "b", "c"], label_map, device="cpu", log_every=0
    )

    assert kept == ["a", "b"]
    assert list(y_true) == [0, 1]
    assert skips == {"missing_pt": 1}
    # logit = mean(fill): a -> sigmoid(-2) < 0.5 < sigmoid(2) -> b
    assert y_prob[0] < 0.5 < y_prob[1]
    assert np.all((y_prob >= 0.0) & (y_prob <= 1.0))
