"""Step B — swappable group-membership plumbing for the skglm sparse-group lasso.

The estimator groups the design-matrix columns by *gene*: one 960-dim embedding block per gene. skglm wants
this as ``(grp_indices, grp_ptr)`` from :func:`skglm.utils.data.grp_converter`, fed to **both** the datafit
(``LogisticGroup``) and the penalty (``WeightedL1GroupL2``).

**This is the highest-risk correctness point.** ``grp_converter`` returns ``(grp_indices, grp_ptr)`` — the
*opposite* order to the constructor argument order ``(grp_ptr, grp_indices)`` — so a positional mix-up, or an
off-by-960 / transposed block layout, is *silent* and produces plausible-but-wrong fits. We therefore wrap the
two arrays in a named :class:`GroupLayout` (never bare tuples) and pin the mapping with a hard test
(``tests/gene_array_lasso/test_group_indexing.py``): group *g* must own **exactly** columns
``[g*960, (g+1)*960)``.

**Swappable by design.** The grouping is an *input*: today uniform 960-blocks over Panaroo orthogroups
(:func:`uniform_block_layout`); later, embedding-cluster groups where one functional group spans several genes'
blocks (:func:`membership_layout`, list-of-lists). Only the layout + column ordering change — never the
estimator assembly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skglm.utils.data import grp_converter

EMB_DIM = 960


@dataclass(frozen=True)
class GroupLayout:
    """A gene-group partition of design-matrix columns, with named (non-confusable) skglm arrays.

    Attributes
    ----------
    grp_indices : numpy.ndarray, shape (n_features,)
        Feature indices stacked contiguously per group (skglm's ``grp_indices``).
    grp_ptr : numpy.ndarray, shape (n_groups + 1,)
        Group pointers; group ``g`` spans ``grp_indices[grp_ptr[g]:grp_ptr[g+1]]`` (skglm's ``grp_ptr``).
    block : int
        Per-gene embedding width (960) in the uniform case; 0 when groups are non-uniform.
    """

    grp_indices: np.ndarray
    grp_ptr: np.ndarray
    block: int = EMB_DIM

    @property
    def n_groups(self) -> int:
        """Number of gene groups."""
        return len(self.grp_ptr) - 1

    @property
    def n_features(self) -> int:
        """Total number of design-matrix columns covered by the partition."""
        return int(self.grp_ptr[-1])

    def columns(self, g: int) -> np.ndarray:
        """Column indices owned by group ``g``."""
        return self.grp_indices[self.grp_ptr[g]:self.grp_ptr[g + 1]]

    def group_sizes(self) -> np.ndarray:
        """Size (number of columns) of each group."""
        return np.diff(self.grp_ptr)


def uniform_block_layout(n_genes: int, block: int = EMB_DIM) -> GroupLayout:
    """Contiguous uniform-``block`` groups: gene ``g`` owns columns ``[g*block, (g+1)*block)``."""
    grp_indices, grp_ptr = grp_converter(block, n_genes * block)
    return GroupLayout(grp_indices=grp_indices, grp_ptr=grp_ptr, block=block)


def membership_layout(groups: list[list[int]]) -> GroupLayout:
    """Arbitrary group membership (for embedding-cluster grouping): ``groups[g]`` = its column indices."""
    n_features = sum(len(g) for g in groups)
    grp_indices, grp_ptr = grp_converter(groups, n_features)
    return GroupLayout(grp_indices=grp_indices, grp_ptr=grp_ptr, block=0)
