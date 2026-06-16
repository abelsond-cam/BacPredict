"""Unit tests for the head-pool concentration stats (``_concentration_stats``).

These exercise the pure NumPy summary that turns one genome's pool-weight vector into
effective-gene-count + top-K-mass — the metric that says *how few* proteins the pool routes to.
No torch / GPU needed.
"""

from __future__ import annotations

import numpy as np

from snp_embeddings.head_pool_attention_probe import _concentration_stats


def test_uniform_pool_has_eff_n_equal_to_n():
    """A flat mean over n proteins → eff_n == n, perplexity == n, topK_mass == K/n."""
    n = 4000
    w = np.full(n, 1.0 / n)
    stats = _concentration_stats(w)
    assert stats["n_proteins"] == n
    assert np.isclose(stats["eff_n"], n)
    assert np.isclose(stats["perplexity"], n)
    assert np.isclose(stats["top50_mass"], 50 / n)
    assert np.isclose(stats["max_weight"], 1.0 / n)


def test_concentrated_pool_has_small_eff_n_and_large_topk_mass():
    """A pool that puts ~all mass on 50 of 4000 proteins → eff_n ≈ 50, top50_mass ≈ 1."""
    n = 4000
    w = np.full(n, 1e-9)
    w[:50] = 1.0 / 50  # 50 dominant proteins, equal weight
    stats = _concentration_stats(w)
    assert 45 < stats["eff_n"] < 55
    assert stats["top50_mass"] > 0.99
    assert stats["top10_mass"] < 0.25  # only 10 of the 50 dominant → ~0.2


def test_top_indices_are_the_heaviest():
    """top{k}_flat_idx lists the heaviest proteins, descending."""
    w = np.array([0.1, 0.5, 0.05, 0.3, 0.05])
    stats = _concentration_stats(w, top_k=3)
    assert stats["top3_flat_idx"] == [1, 3, 0]
    assert np.allclose(stats["top3_weight"], [0.5, 0.3, 0.1])
    assert np.isclose(stats["top1_mass"], 0.5)
