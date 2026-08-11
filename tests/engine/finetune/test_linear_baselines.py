"""Tests for the linear metadata baselines — focused on the numeric (Kleborate-score) block.

The numeric path is the new one and the easy one to get subtly wrong: test rows must be scaled with
*train* statistics (using their own would leak), and a count-scaled column must end up on the same
footing as the 0/1 columns it is stacked beside, or the shared L2 penalty silently understates the
baseline we are trying to beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bacpredict.engine.finetune.linear_baselines import (
    FEATURE_BLOCKS,
    FEATURE_SET_RECIPES,
    _build_design_matrix,
    _numeric_frame_factory,
)


@pytest.fixture
def idx() -> pd.Index:
    return pd.Index([f"S{i}" for i in range(100)])


def test_kleborate_blocks_and_recipes_are_registered():
    assert "virulence_score" in FEATURE_BLOCKS
    assert "amr_score" in FEATURE_BLOCKS
    for recipe in ("virulence_score", "virulence_bsc+amr_class", "kleborate_all",
                   "virulence_score+virulence_bsc", "amr_score"):
        assert recipe in FEATURE_SET_RECIPES, recipe


def test_kleborate_only_recipes_carry_no_population_terms():
    """These recipes exist to isolate virulence biology — country/Sublineage must not sneak in."""
    for recipe in ("virulence_score", "virulence_bsc", "virulence_bsc+amr_class", "kleborate_all"):
        blocks = FEATURE_SET_RECIPES[recipe]
        assert "country" not in blocks and "sublineage" not in blocks and "k_locus" not in blocks


def test_numeric_block_is_standardised_on_train_statistics(idx):
    """Test rows must be scaled with the TRAIN mean/std, never their own."""
    joined = pd.DataFrame({"virulence_score": np.arange(100, dtype=float)}, index=idx)
    frame = FEATURE_BLOCKS["virulence_score"].numeric_materialise(joined)
    train, test = idx[:80], idx[80:]

    X_tr, X_te, n_feat = _build_design_matrix([], [], train, test, numeric_frames=[frame])
    assert n_feat == 1

    a = X_tr.toarray().ravel()
    assert a.mean() == pytest.approx(0.0, abs=1e-9)
    assert a.std() == pytest.approx(1.0, abs=1e-9)

    # Test rows (values 80..99) sit entirely above the train mean (39.5) -> all positive, and are
    # NOT independently re-centred (their own mean would be 0 if they were).
    b = X_te.toarray().ravel()
    assert (b > 0).all()
    assert b.mean() > 1.0


def test_numeric_nan_is_imputed_to_the_train_mean(idx):
    joined = pd.DataFrame({"virulence_score": np.ones(100)}, index=idx)
    joined.loc[idx[:10], "virulence_score"] = np.nan
    joined.loc[idx[10:], "virulence_score"] = np.arange(90, dtype=float)
    frame = _numeric_frame_factory(["virulence_score"])(joined)

    X_tr, _X_te, _ = _build_design_matrix([], [], idx[:80], idx[80:], numeric_frames=[frame])
    a = X_tr.toarray().ravel()
    # NaN -> train mean -> exactly 0 after centring.
    np.testing.assert_allclose(a[:10], 0.0, atol=1e-12)


def test_constant_numeric_column_does_not_divide_by_zero(idx):
    joined = pd.DataFrame({"virulence_score": np.full(100, 3.0)}, index=idx)
    frame = _numeric_frame_factory(["virulence_score"])(joined)
    X_tr, X_te, _ = _build_design_matrix([], [], idx[:80], idx[80:], numeric_frames=[frame])
    assert np.isfinite(X_tr.toarray()).all()
    assert np.isfinite(X_te.toarray()).all()


def test_numeric_factory_coerces_strings_and_tolerates_absent_columns(idx):
    joined = pd.DataFrame(
        {"resistance_score": ["1", "2"] * 50, "num_resistance_classes": np.arange(100)}, index=idx
    )
    # num_resistance_genes is absent from this metadata; the block should use what is present.
    frame = _numeric_frame_factory(
        ["resistance_score", "num_resistance_classes", "num_resistance_genes"]
    )(joined)
    assert list(frame.columns) == ["resistance_score", "num_resistance_classes"]
    # Coerced out of object dtype; int or float is fine (_build_design_matrix casts to float).
    assert frame["resistance_score"].dtype.kind in "if"
    assert frame["resistance_score"].tolist()[:2] == [1, 2]


def test_numeric_factory_raises_when_no_column_is_present(idx):
    joined = pd.DataFrame({"unrelated": np.arange(100)}, index=idx)
    with pytest.raises(ValueError, match="None of the numeric columns"):
        _numeric_frame_factory(["virulence_score"])(joined)


def test_numeric_and_binary_blocks_stack_on_a_comparable_scale(idx):
    """A 0-5 count must not dwarf the 0/1 columns it is hstacked with once standardised."""
    joined = pd.DataFrame({
        "virulence_score": np.random.default_rng(0).integers(0, 6, 100).astype(float),
    }, index=idx)
    numeric = _numeric_frame_factory(["virulence_score"])(joined)
    binary = pd.DataFrame({"ybt": np.random.default_rng(1).integers(0, 2, 100)}, index=idx)

    X_tr, _, n_feat = _build_design_matrix([], [binary], idx[:80], idx[80:], numeric_frames=[numeric])
    assert n_feat == 2
    cols = X_tr.toarray()
    # Both columns end up within the same order of magnitude, unlike raw 0-5 vs 0-1.
    assert 0.3 < cols[:, 1].std() / cols[:, 0].std() < 3.0
