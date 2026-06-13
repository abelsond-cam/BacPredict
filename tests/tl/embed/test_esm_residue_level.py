"""Model-free unit tests for the residue-level ESM-C helpers.

Covers the pure-tensor / pure-string operations (``production_mean_pool``,
``apply_point_mutation``) that need no ESM++ weights, so they run in CI. The
byte-for-byte-vs-a-stored-vector check for ``production_mean_pool`` runs on the
HPC (it needs the embedding store) — see ``geometry_probe.py``.
"""

from __future__ import annotations

import pytest
import torch

from tl.embed.esm_residue_level import apply_point_mutation, production_mean_pool


def test_production_mean_pool_is_exact_mean():
    """With no mask the pool is byte-for-byte the residue mean."""
    # Integer-valued so the mean is exactly representable in float32.
    residues = torch.tensor([[2.0, 4.0], [4.0, 8.0], [6.0, 12.0]])
    pooled = production_mean_pool(residues)
    assert torch.equal(pooled, torch.tensor([4.0, 8.0]))


def test_production_mean_pool_respects_mask():
    """A 0 in the mask drops that residue from the average."""
    residues = torch.tensor([[2.0, 4.0], [4.0, 8.0], [100.0, 100.0]])
    mask = torch.tensor([1.0, 1.0, 0.0])
    pooled = production_mean_pool(residues, mask)
    assert torch.equal(pooled, torch.tensor([3.0, 6.0]))


def test_production_mean_pool_truncates_to_max_residues():
    """max_residues pools only the leading rows (matches the 1024 production cap)."""
    residues = torch.tensor([[2.0, 4.0], [4.0, 8.0], [100.0, 100.0]])
    pooled = production_mean_pool(residues, max_residues=2)
    assert torch.equal(pooled, torch.tensor([3.0, 6.0]))


def test_production_mean_pool_empty_mask_raises():
    with pytest.raises(ValueError, match="empty mask"):
        production_mean_pool(torch.zeros(3, 2), torch.zeros(3))


def test_apply_point_mutation_basic():
    assert apply_point_mutation("MKLV", 2, "Y") == "MKYV"


def test_apply_point_mutation_checks_expected_wt():
    assert apply_point_mutation("MKLV", 2, "Y", expected_wt="L") == "MKYV"
    with pytest.raises(ValueError, match="expected wild-type"):
        apply_point_mutation("MKLV", 2, "Y", expected_wt="S")


def test_apply_point_mutation_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        apply_point_mutation("MKLV", 9, "Y")


def test_apply_point_mutation_single_residue_only():
    with pytest.raises(ValueError, match="single residue"):
        apply_point_mutation("MKLV", 1, "YY")
