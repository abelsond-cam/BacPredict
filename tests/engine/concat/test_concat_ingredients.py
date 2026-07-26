"""Unit tests for the shared concat-ingredient guard — the FT-cache holdout-coverage check.

``assert_holdout_in_cache`` is the one guard every corrected FT read-out (the ladder + the four concat
scorers) shares: it refuses a cache that does not contain the deployed k-fold holdout (the pre-fix leak
signature — a cache built on the CSV single-split or eval-only holds ~none of it) or that has no FT-train
genomes to fit on.
"""

from __future__ import annotations

import pytest

from bacpredict.engine.concat.concat_ingredients import assert_holdout_in_cache


def test_full_coverage_returns_counts():
    """A cache holding the full holdout + a train side returns ``(n_holdout, n_train)``."""
    all_ids = [f"h{i}" for i in range(20)] + [f"t{i}" for i in range(40)]
    holdout_ids = [f"h{i}" for i in range(20)]
    assert assert_holdout_in_cache(all_ids, holdout_ids, "rifampin", "trainholdout") == (20, 40)


def test_partial_coverage_above_threshold_ok():
    """Missing a few holdout genomes (≥90% present) is tolerated — a slightly short forward still scores."""
    all_ids = [f"h{i}" for i in range(19)] + [f"t{i}" for i in range(30)]  # 19/20 holdout
    holdout_ids = [f"h{i}" for i in range(20)]
    n_holdout, n_train = assert_holdout_in_cache(all_ids, holdout_ids, "rifampin", "trainholdout")
    assert n_holdout == 19 and n_train == 30


def test_leaky_cache_raises():
    """A cache holding ~none of the deployed holdout (the CSV-vs-kfold leak signature) is refused."""
    all_ids = [f"c{i}" for i in range(60)]  # 0 of the 30 deployed holdout genomes are present
    holdout_ids = [f"h{i}" for i in range(30)]
    with pytest.raises(ValueError, match="leak signature"):
        assert_holdout_in_cache(all_ids, holdout_ids, "azithromycin", "eval")


def test_no_train_side_raises():
    """A cache that is holdout-only (``scope=eval``) has no train side to fit on → refuse for a fit-on-train read-out."""
    holdout_ids = [f"h{i}" for i in range(20)]
    with pytest.raises(ValueError, match="no FT-train genomes"):
        assert_holdout_in_cache(list(holdout_ids), holdout_ids, "rifampin", "eval")
