"""Unit tests for the single-gene wild-type substitution counterfactual."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from bacpredict.engine.explain.substitute_wt_protein import (
    donor_embedding,
    gene_rows,
    samples_carrying,
)


class _Paths:
    """Minimal stand-in for StorePaths."""

    def __init__(self, parquet_dir, esm_suffix="_esm.pt", parquet_suffix="_proteins.parquet"):
        self.parquet_dir = parquet_dir
        self.esm_suffix = esm_suffix
        self.parquet_suffix = parquet_suffix


def _write_genome(tmp_path, sample, genes, embeddings, paths):
    """Write one genome's protein parquet (nested, one contig) and its embedding store."""
    pd.DataFrame([{
        "contig_idx": [0],
        "gene_name": [list(genes)],
        "protein_name": [[g or "" for g in genes]],
        "protein_sequence": [["M" * 10] * len(genes)],
        "start": [list(range(len(genes)))],
        "end": [[i + 9 for i in range(len(genes))]],
        "protein_id": [[f"{sample}_{i}" for i in range(len(genes))]],
    }]).to_parquet(paths.parquet_dir / f"{sample}{paths.parquet_suffix}")
    torch.save({"prot_embeddings": embeddings}, tmp_path / f"{sample}{paths.esm_suffix}")


def test_gene_rows_returns_the_flat_index(tmp_path) -> None:
    """The returned index must be the embedding tensor's row, in flat contig order."""
    paths = _Paths(tmp_path)
    _write_genome(tmp_path, "S1", ["dnaA", "rpoB", "gyrA"], torch.zeros(3, 4), paths)
    assert gene_rows("S1", "rpoB", tmp_path, paths.parquet_suffix) == [1]
    assert gene_rows("S1", "RPOB", tmp_path, paths.parquet_suffix) == [1], "match must ignore case"
    assert gene_rows("S1", "katG", tmp_path, paths.parquet_suffix) == []


def test_gene_rows_reports_every_copy(tmp_path) -> None:
    """A multi-copy gene returns all its indices so the caller can refuse, not silently pick one."""
    paths = _Paths(tmp_path)
    _write_genome(tmp_path, "S2", ["rpoB", "dnaA", "rpoB"], torch.zeros(3, 4), paths)
    assert gene_rows("S2", "rpoB", tmp_path, paths.parquet_suffix) == [0, 2]


def test_samples_carrying_filters_by_gene_and_variant() -> None:
    """Wild-type donor selection depends entirely on this being right."""
    variants = pd.DataFrame({
        "Sample": ["A", "B", "C"],
        "gene_name": ["rpoB", "rpoB", "katG"],
        "variant_id": ["rpoB@p.Ser450Leu", "rpoB@p.His445Tyr", "katG@p.Ser315Thr"],
    })
    assert samples_carrying(variants, "rpoB", None) == {"A", "B"}
    assert samples_carrying(variants, "rpoB", "rpoB@p.Ser450Leu") == {"A"}
    assert samples_carrying(variants, "gyrA", None) == set()


def test_donor_embedding_averages_only_single_copy_genomes(tmp_path) -> None:
    """A paralogue must not be blended into the counterfactual.

    Averaging a two-copy genome's gene would mean the donor is partly a different protein, which
    blurs exactly the comparison the substitution exists to make.
    """
    paths = _Paths(tmp_path)
    _write_genome(tmp_path, "W1", ["rpoB", "dnaA"], torch.tensor([[1.0, 1.0], [9.0, 9.0]]), paths)
    _write_genome(tmp_path, "W2", ["dnaA", "rpoB"], torch.tensor([[9.0, 9.0], [3.0, 3.0]]), paths)
    _write_genome(tmp_path, "W3", ["rpoB", "rpoB"], torch.tensor([[99.0, 99.0], [99.0, 99.0]]), paths)

    donor, n_donors = donor_embedding(
        ["W1", "W2", "W3"], "rpoB", tmp_path, paths, "cpu", torch.float32)

    assert n_donors == 2, "the two-copy genome must be skipped"
    np.testing.assert_allclose(donor.numpy(), np.array([2.0, 2.0]), rtol=1e-6)


def test_donor_embedding_raises_when_no_donor_has_the_gene(tmp_path) -> None:
    """Silently returning a zero vector here would look like a valid no-op result."""
    paths = _Paths(tmp_path)
    _write_genome(tmp_path, "N1", ["dnaA"], torch.zeros(1, 2), paths)
    with pytest.raises(ValueError, match="no donor genome carried exactly one copy"):
        donor_embedding(["N1"], "rpoB", tmp_path, paths, "cpu", torch.float32)


def test_substitution_changes_only_the_target_row(tmp_path) -> None:
    """The splice must touch one row and leave the rest of the genome byte-identical.

    This is the whole validity of the counterfactual: any other change and the logit shift is not
    attributable to the gene.
    """
    paths = _Paths(tmp_path)
    original = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    _write_genome(tmp_path, "S1", ["dnaA", "rpoB", "gyrA"], original[0], paths)
    index = gene_rows("S1", "rpoB", tmp_path, paths.parquet_suffix)[0]

    donor = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    substituted = original.clone()
    substituted[0, index] = donor

    assert torch.equal(substituted[0, index], donor)
    untouched = [i for i in range(3) if i != index]
    assert torch.equal(substituted[0, untouched], original[0, untouched])
