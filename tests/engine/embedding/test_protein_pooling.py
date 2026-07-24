"""Unit tests for the shared protein/genome pooling primitives."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bacpredict.engine.embedding.protein_pooling import (
    genome_mean_pool,
    real_protein_indices,
    real_protein_rows,
)


def test_real_protein_indices_special_tokens_mask():
    """Bacformer-input bundle: real rows are flagged by ``special_tokens_mask == 4``."""
    store = {"special_tokens_mask": torch.tensor([[4, 0, 4, 4, 0]])}
    idx = real_protein_indices(store, n_rows=5)
    assert idx.tolist() == [0, 2, 3]


def test_real_protein_indices_attention_mask():
    """Plain per-protein store: real rows are where ``attention_mask == 1``."""
    store = {"attention_mask": torch.tensor([[1, 1, 0, 1]])}
    idx = real_protein_indices(store, n_rows=4)
    assert idx.tolist() == [0, 1, 3]


def test_real_protein_indices_neither_falls_back_to_arange():
    """No mask key → every row is treated as real (flat arange)."""
    idx = real_protein_indices({}, n_rows=3)
    assert idx.tolist() == [0, 1, 2]


def test_real_protein_rows_squeezes_and_selects():
    """A ``[1, T, dim]`` output is squeezed; the real rows are returned in flat order, as float."""
    lhs = torch.arange(4 * 2, dtype=torch.int64).reshape(1, 4, 2)  # [1, T=4, dim=2]
    real_idx = torch.tensor([0, 2, 3])
    rows = real_protein_rows(lhs, real_idx, input_len=4)
    assert rows.shape == (3, 2)
    assert rows.dtype == torch.float32
    assert rows.tolist() == [[0.0, 1.0], [4.0, 5.0], [6.0, 7.0]]


def test_real_protein_rows_guard_raises_on_length_mismatch():
    """Day-one guard: the output must align 1:1 with the input rows, else every flat index is off."""
    lhs = torch.zeros(3, 2)
    with pytest.raises(RuntimeError, match="misaligned"):
        real_protein_rows(lhs, torch.tensor([0, 1]), input_len=5)


def test_genome_mean_pool_torch_and_numpy_agree():
    """The genome mean is the row-axis mean; torch and numpy inputs give the same float32 vector."""
    rows_t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mean_t = genome_mean_pool(rows_t)
    mean_n = genome_mean_pool(rows_t.numpy())
    assert isinstance(mean_t, np.ndarray) and isinstance(mean_n, np.ndarray)
    np.testing.assert_allclose(mean_t, [3.0, 4.0])
    np.testing.assert_allclose(mean_t, mean_n)
