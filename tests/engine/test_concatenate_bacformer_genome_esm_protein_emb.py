"""Unit tests for the concat probe's small pure helpers.

Covers ``_top_gene_from_ranking`` (auto-pick the causal gene per drug from a per-gene LR ranking CSV) and
``_resolve_concat_splits`` (a FINE-TUNED mean must resolve the deployed k-fold holdout, never the CSV
single-split — the train/test leak). The heavy ``run_concat_probe`` path needs HPC embeddings and is
exercised by the Stage-A smoke, not here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
import pandas as pd

import bacpredict.engine.segment_amr_lr.concat.concatenate_bacformer_genome_esm_protein_emb as concat


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


def test_resolve_concat_splits_frozen_uses_csv_holdout(monkeypatch, tmp_path: Path) -> None:
    """A FROZEN mean is label-blind → resolve the CSV single-split (no checkpoint_dir)."""
    seen: dict = {}

    def fake_resolve(ast, drug, **kw):
        seen["kw"] = kw
        return ({"s1": 1}, ["s1"], [], ["s2"], {"source": "csv"})

    monkeypatch.setattr(concat, "resolve_clean_splits", fake_resolve)
    out = concat._resolve_concat_splits("sheet.csv", "rifampin", finetuned=False, holdout_run_dir=None)
    assert "checkpoint_dir" not in seen["kw"]  # CSV single-split, not the deployed holdout
    assert out[0] == {"s1": 1}


def test_resolve_concat_splits_finetuned_uses_deployed_holdout(monkeypatch, tmp_path: Path) -> None:
    """A FINE-TUNED mean routes through the deployed run's results.json (checkpoint_dir=run dir)."""
    seen: dict = {}

    def fake_resolve(ast, drug, **kw):
        seen["kw"] = kw
        return ({"s1": 1}, ["s1"], [], ["s2"], {"source": "kfold"})

    monkeypatch.setattr(concat, "resolve_clean_splits", fake_resolve)
    run_dir = tmp_path / "ft_run"
    concat._resolve_concat_splits("sheet.csv", "rifampin", finetuned=True, holdout_run_dir=run_dir)
    assert seen["kw"].get("checkpoint_dir") == run_dir  # deployed k-fold holdout, not CSV


def test_resolve_concat_splits_finetuned_without_run_dir_raises(monkeypatch) -> None:
    """A FINE-TUNED mean with no holdout run dir MUST fail loud — never silently fall back to CSV (the leak)."""
    called = {"n": 0}

    def fake_resolve(*a, **k):
        called["n"] += 1
        return ({}, [], [], [], {})

    monkeypatch.setattr(concat, "resolve_clean_splits", fake_resolve)
    with pytest.raises(ValueError, match="deployed model's k-fold holdout"):
        concat._resolve_concat_splits("sheet.csv", "rifampin", finetuned=True, holdout_run_dir=None)
    assert called["n"] == 0  # raised before touching the resolver — no silent CSV fallback
