"""Unit smoke for the 2d long-region windowing in ``baclm_embed`` (no model load).

Covers ``_windowize`` tiling + the token-count-weighted aggregation in ``mean_pool_windowed`` (with a
stubbed per-window embedder), which is the only new numerical logic on the non-coding path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tl.embed import baclm_embed

CHUNK = baclm_embed._WINDOW_CHUNK  # 2046


def test_windowize_short_seq_is_single_window():
    seqs = ["acgt" * 10]  # 40 chars <= chunk
    windows, owner, weight = baclm_embed._windowize(seqs, CHUNK, overlap=0)
    assert windows == seqs
    assert owner == [0]
    assert weight == [40.0]


def test_windowize_tiles_long_seq_non_overlapping():
    seq = "a" * 5000
    windows, owner, weight = baclm_embed._windowize([seq], CHUNK, overlap=0)
    # 5000 / 2046 -> windows of 2046, 2046, 908
    assert [len(w) for w in windows] == [2046, 2046, 908]
    assert owner == [0, 0, 0]
    assert weight == [2046.0, 2046.0, 908.0]
    assert "".join(windows) == seq  # non-overlapping tiling reconstructs the region exactly


def test_windowize_overlap_shifts_start():
    seq = "a" * 3000
    windows, _, _ = baclm_embed._windowize([seq], chunk=2000, overlap=500)
    # step = 1500: starts at 0, 1500 -> [0:2000], [1500:3000]
    assert [len(w) for w in windows] == [2000, 1500]


def test_mean_pool_windowed_token_weighted_equals_full_region(monkeypatch):
    """With overlap=0 the weighted pool = mean-pool of the whole region; short seqs pass through."""
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=4))

    def fake_pool(windows, *a, **k):
        # Each window embeds to a constant vector equal to its own length (in every dim).
        return torch.tensor([[float(len(w))] * 4 for w in windows], dtype=torch.bfloat16)

    monkeypatch.setattr(baclm_embed, "mean_pool_embeddings", fake_pool)

    short = "a" * 99
    long = "a" * 5000  # windows 2046, 2046, 908
    out = baclm_embed.mean_pool_windowed([short, long], model, None, "cpu", "dna", batch_size=8)
    assert out.shape == (2, 4)
    # short: single window weight 99, value 99 -> 99
    assert out[0, 0].item() == pytest.approx(99.0, rel=0.01)
    # long: sum(w*value)/sum(w) with value==w -> sum(w^2)/sum(w)
    expected = (2046**2 + 2046**2 + 908**2) / 5000
    assert out[1, 0].item() == pytest.approx(expected, rel=0.02)


def test_mean_pool_windowed_empty():
    model = SimpleNamespace(config=SimpleNamespace(hidden_size=4))
    out = baclm_embed.mean_pool_windowed([], model, None, "cpu", "dna", batch_size=8)
    assert out.shape == (0, 4)
