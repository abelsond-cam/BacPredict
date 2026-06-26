"""Hard test for the gene-group plumbing (Step B) — the highest-risk correctness point.

skglm's ``grp_converter`` returns ``(grp_indices, grp_ptr)``, the *opposite* order to the constructor
argument order ``(grp_ptr, grp_indices)``. An off-by-960 / transposed block layout is silent and produces
plausible-but-wrong fits, so we pin the exact mapping here: gene ``g`` must own **exactly** columns
``[g*block, (g+1)*block)``, contiguous, disjoint, and covering all features.
"""

import numpy as np
import pytest

from gene_array_lasso.group_spec import EMB_DIM, membership_layout, uniform_block_layout


@pytest.mark.parametrize("n_genes,block", [(4, 3), (5, EMB_DIM), (1, EMB_DIM), (12, EMB_DIM)])
def test_uniform_block_layout_exact_columns(n_genes, block):
    """Group g owns exactly [g*block, (g+1)*block); contiguous, disjoint, full cover."""
    lay = uniform_block_layout(n_genes, block=block)
    assert lay.n_groups == n_genes
    assert lay.n_features == n_genes * block
    covered = []
    for g in range(n_genes):
        cols = lay.columns(g)
        # exact expected block — catches off-by-960 / transposition
        np.testing.assert_array_equal(cols, np.arange(g * block, (g + 1) * block))
        covered.append(cols)
    allcols = np.concatenate(covered)
    # disjoint + full cover of every feature exactly once
    np.testing.assert_array_equal(np.sort(allcols), np.arange(n_genes * block))
    assert len(set(allcols.tolist())) == n_genes * block
    np.testing.assert_array_equal(lay.group_sizes(), np.full(n_genes, block))


def test_groups_are_distinct():
    """Adjacent groups must not share columns (a shift bug would alias them)."""
    lay = uniform_block_layout(3, block=EMB_DIM)
    assert set(lay.columns(0).tolist()).isdisjoint(lay.columns(1).tolist())
    assert lay.columns(1)[0] == EMB_DIM  # second gene starts at column 960, not 0 or 959


def test_wrong_block_size_would_fail():
    """Decoding with the wrong block size must NOT match — guards against silent off-by-960."""
    lay = uniform_block_layout(4, block=EMB_DIM)
    # gene 1's true columns are [960, 1920); an off-by-one-block read [1*959 ..] must differ
    assert not np.array_equal(lay.columns(1), np.arange(EMB_DIM - 1, 2 * EMB_DIM - 1))


def test_membership_layout_variable_groups():
    """Swappable non-uniform grouping (future embedding clusters): explicit column membership."""
    lay = membership_layout([[0, 1], [2, 3, 4], [5]])
    assert lay.n_groups == 3
    assert lay.n_features == 6
    np.testing.assert_array_equal(lay.columns(0), [0, 1])
    np.testing.assert_array_equal(lay.columns(1), [2, 3, 4])
    np.testing.assert_array_equal(lay.columns(2), [5])
    np.testing.assert_array_equal(lay.group_sizes(), [2, 3, 1])
