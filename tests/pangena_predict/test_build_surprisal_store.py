"""Unit tests for the surprisal-panel builder (``pangena_predict.build_surprisal_store``).

Verify the 9-panel order, by-construction consistency with ``protein_surprisal_stats``,
and the short-protein imputation that keeps the store NaN-free. Skipped where numpy/scipy
are unavailable (the local MacBook env).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")  # protein_surprisal_stats uses scipy.stats

from pangena_predict.build_surprisal_store import PANEL_DIM, PANEL_KEYS, panel_from_stats
from pangena_predict.llr_distribution_probe import protein_surprisal_stats


def test_panel_order_and_dim() -> None:
    """The panel is exactly [s1, s2, s3, s10, p95, p90, p50, participation_ratio, kurtosis]."""
    assert PANEL_DIM == 9
    assert PANEL_KEYS == (
        "max_surprisal", "top2_surprisal", "top3_surprisal", "top10_surprisal",
        "p95_surprisal", "p90_surprisal", "median_surprisal", "participation_ratio", "kurtosis_surprisal",
    )


def test_panel_matches_protein_stats_for_long_protein() -> None:
    """For a long protein every panel member equals the corresponding stat (no imputation)."""
    rng = np.random.default_rng(0)
    surprisal = np.abs(rng.normal(size=500))  # 500 residues, all stats defined
    stats = protein_surprisal_stats(-surprisal)  # builder passes -surprisal (stats expect log P)
    row = panel_from_stats(stats)
    for i, key in enumerate(PANEL_KEYS):
        assert row[i] == pytest.approx(float(stats[key]))


def test_panel_imputes_short_protein() -> None:
    """A 2-residue protein leaves top3/top10/kurtosis undefined → imputed, never NaN."""
    surprisal = np.array([3.0, 1.0])
    stats = protein_surprisal_stats(-surprisal)
    row = panel_from_stats(stats)
    s1 = float(stats["max_surprisal"])
    assert row[0] == pytest.approx(s1)              # max
    assert row[2] == pytest.approx(s1)              # top3 → max (only 2 residues)
    assert row[3] == pytest.approx(s1)              # top10 → max
    assert np.isfinite(row).all()                   # NaN-free guarantee
    assert row[8] == pytest.approx(0.0)             # kurtosis undefined (n<4) → 0.0


def test_panel_all_finite_single_residue() -> None:
    """A degenerate 1-residue protein still yields a finite 9-vector."""
    row = panel_from_stats(protein_surprisal_stats(-np.array([2.5])))
    assert len(row) == PANEL_DIM
    assert np.isfinite(row).all()
