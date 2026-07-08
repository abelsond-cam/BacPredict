"""Unit tests for the concat probe's small pure helpers.

Covers ``_top_gene_from_ranking`` (auto-pick the causal gene per drug from a per-gene LR ranking CSV).
The heavy ``run_concat_probe`` path needs HPC embeddings and is exercised by the Stage-A smoke, not here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
import pandas as pd

import pangena_predict.concatenate_bacformer_genome_esm_protein_emb as concat


def _write_ranking(path: Path, rows: list[tuple[str, float]], drug: str = "rifampin") -> None:
    """A minimal per-gene ranking CSV (gene_name + the one lr_auroc_<drug> column)."""
    pd.DataFrame(
        {"gene_name": [g for g, _ in rows], "annotation": ["" for _ in rows],
         f"lr_auroc_{drug}": [a for _, a in rows]}
    ).to_csv(path, index=False)


def test_top_gene_from_ranking_picks_highest_auroc(tmp_path: Path) -> None:
    """Returns the gene_name of the top-AUROC row, regardless of input row order."""
    csv = tmp_path / "per_gene_lr_rifampin.csv"
    _write_ranking(csv, [("embB", 0.864), ("rpoB", 0.962), ("katG", 0.849)])
    assert concat._top_gene_from_ranking(csv) == "rpoB"


def test_top_gene_from_ranking_finds_the_auroc_column_for_any_drug(tmp_path: Path) -> None:
    """The auroc column is discovered by its lr_auroc_ prefix, so any drug works."""
    csv = tmp_path / "per_gene_lr_isoniazid.csv"
    _write_ranking(csv, [("rpoB", 0.55), ("katG", 0.93)], drug="isoniazid")
    assert concat._top_gene_from_ranking(csv) == "katG"


def test_top_gene_from_ranking_rejects_a_non_ranking_csv(tmp_path: Path) -> None:
    """A CSV with no lr_auroc_<drug> column is rejected (guards against a wrong path)."""
    csv = tmp_path / "not_a_ranking.csv"
    pd.DataFrame({"gene_name": ["rpoB"], "value": [1.0]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="no lr_auroc"):
        concat._top_gene_from_ranking(csv)
