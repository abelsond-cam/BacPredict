"""Unit tests for the per-contig embedding/annotation alignment check."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from bacpredict.engine.explain.verify_store_alignment import (
    compare_one,
    contig_sizes_from_parquet,
    contig_sizes_from_store,
)


def _write(tmp_path, sample, contig_sizes_pt, contig_sizes_pq):
    """Write a .pt whose contigs hold contig_sizes_pt proteins, and a parquet holding ..._pq."""
    ids = torch.tensor([c for c, n in enumerate(contig_sizes_pt) for _ in range(n)], dtype=torch.long)
    torch.save({"protein_embeddings": torch.zeros(1, len(ids), 4),
                "contig_ids": ids.unsqueeze(0),
                "attention_mask": torch.ones(1, len(ids))}, tmp_path / f"{sample}.pt")
    pd.DataFrame([{
        "contig_idx": list(range(len(contig_sizes_pq))),
        "gene_name": [[f"g{c}_{i}" for i in range(n)] for c, n in enumerate(contig_sizes_pq)],
        "protein_name": [["" for _ in range(n)] for n in contig_sizes_pq],
        "protein_sequence": [["M" * 5 for _ in range(n)] for n in contig_sizes_pq],
        "start": [list(range(n)) for n in contig_sizes_pq],
        "end": [list(range(n)) for n in contig_sizes_pq],
        "protein_id": [[f"p{c}_{i}" for i in range(n)] for c, n in enumerate(contig_sizes_pq)],
    }]).to_parquet(tmp_path / f"{sample}.parquet")
    return tmp_path / f"{sample}.pt", tmp_path / f"{sample}.parquet"


def test_reads_per_contig_sizes_from_both_sides(tmp_path) -> None:
    pt, parquet = _write(tmp_path, "S", [3, 2, 4], [3, 2, 4])
    np.testing.assert_array_equal(contig_sizes_from_store(pt), np.array([3, 2, 4]))
    np.testing.assert_array_equal(contig_sizes_from_parquet(parquet), np.array([3, 2, 4]))


def test_aligned_genome_reports_aligned(tmp_path) -> None:
    pt, parquet = _write(tmp_path, "S", [3, 2, 4], [3, 2, 4])
    row = compare_one("S", pt, parquet)
    assert row["status"] == "aligned"
    assert row["n_proteins_after_break"] == 0


def test_compensating_change_is_caught_though_totals_agree(tmp_path) -> None:
    """THE case a total-count check misses: +1 on one contig, -1 on another.

    Totals are identical, so `verify_protein_count` passes, while every protein between the two
    contigs is attributed to the wrong gene.
    """
    pt, parquet = _write(tmp_path, "S", [4, 2, 3], [3, 2, 4])
    row = compare_one("S", pt, parquet)
    assert row["status"] == "contig_mismatch"
    assert row["n_pt"] == row["n_parquet"] == 9, "the totals must agree for this test to mean anything"
    assert row["first_divergent_contig"] == 0
    assert row["first_divergent_flat_index"] == 0
    assert row["n_proteins_after_break"] == 9


def test_break_position_counts_only_proteins_after_it(tmp_path) -> None:
    """Proteins before the first divergence are correctly identified and must not be counted."""
    pt, parquet = _write(tmp_path, "S", [5, 5, 3, 2], [5, 5, 4, 2])
    row = compare_one("S", pt, parquet)
    assert row["first_divergent_contig"] == 2
    assert row["first_divergent_flat_index"] == 10, "the first 10 proteins align"
    assert row["n_proteins_after_break"] == 5


def test_truncation_is_distinguished_from_misalignment(tmp_path) -> None:
    """An embedding capped at the limit still aligns for the rows it has — a different finding."""
    pt, parquet = _write(tmp_path, "S", [6000], [6000, 20])
    row = compare_one("S", pt, parquet)
    assert row["status"] == "contig_count_mismatch"


def test_missing_files_are_reported_not_raised(tmp_path) -> None:
    row = compare_one("S", tmp_path / "absent.pt", tmp_path / "absent.parquet")
    assert row["status"] == "missing_pt"
