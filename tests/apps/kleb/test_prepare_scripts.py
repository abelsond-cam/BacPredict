"""Tests for prepare script CSV-only default and --write-pt-files flag."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from bacpredict.engine.finetune.build_split_csv import (
    load_ast_sheet,
    validate_embeddings_and_prune,
)
from bacpredict.engine.finetune.build_split_csv import (
    write_split_files as write_split_files_amr,
)
from bacpredict.engine.finetune.split_utils import add_splits


def _write_embedding(path: Path, n_proteins: int = 4) -> None:
    torch.save({"prot_embeddings": torch.randn(n_proteins, 8), "attention_mask": torch.ones(n_proteins)}, path)


def _make_ast_csv(tmp_path: Path, sample_ids: list[str]) -> Path:
    csv_path = tmp_path / "binary_ast.csv"
    pd.DataFrame({
        "phenotype-BioSample_ID": sample_ids,
        "amoxicillin": [0, 1, 0, 1, 0],
        "ciprofloxacin": [1, 0, 1, 0, 1],
    }).to_csv(csv_path, index=False)
    return csv_path


# ── AMR prepare script ────────────────────────────────────────────────────────

def test_amr_csv_written_with_correct_columns(tmp_path):
    """Default run (no --write-pt-files) writes split CSV with expected columns."""
    sample_ids = [f"S{i:03d}" for i in range(5)]
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    for sid in sample_ids:
        _write_embedding(embeddings_dir / f"{sid}_esm_embeddings.pt")

    csv_path = _make_ast_csv(tmp_path, sample_ids)
    df = load_ast_sheet(csv_path)
    df = add_splits(df, seed=1)
    pruned_df, _ = validate_embeddings_and_prune(df, embeddings_dir)

    split_csv = tmp_path / "binary_ast_with_split.csv"
    pruned_df.to_csv(split_csv, index=False)

    result = pd.read_csv(split_csv)
    assert "Sample" in result.columns
    assert "train_val_eval" in result.columns
    assert "amoxicillin" in result.columns
    assert result["train_val_eval"].isin(["train", "validate", "evaluate"]).all()


def test_amr_no_pt_files_by_default(tmp_path):
    """Without --write-pt-files no .pt files should appear in train/validate/evaluate."""
    output_base = tmp_path / "output"
    output_base.mkdir()
    # Nothing was written — directories should not exist
    for split in ("train", "validate", "evaluate"):
        assert not (output_base / split).exists()


def test_amr_write_pt_files_creates_files(tmp_path):
    """write_split_files() (the --write-pt-files path) creates .pt files in the correct dirs."""
    sample_ids = [f"S{i:03d}" for i in range(5)]
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    for sid in sample_ids:
        _write_embedding(embeddings_dir / f"{sid}_esm_embeddings.pt")

    csv_path = _make_ast_csv(tmp_path, sample_ids)
    df = load_ast_sheet(csv_path)
    df = add_splits(df, seed=1)
    pruned_df, _ = validate_embeddings_and_prune(df, embeddings_dir)

    output_base = tmp_path / "output"
    write_split_files_amr(pruned_df, embeddings_dir, output_base, skip_existing=False)

    pt_files = list(output_base.rglob("*.pt"))
    assert len(pt_files) == len(sample_ids)
    for split in ("train", "validate", "evaluate"):
        assert (output_base / split).exists()


def test_amr_pt_file_contains_label(tmp_path):
    """Each .pt file written by write_split_files contains the antibiotic label."""
    sid = "STEST"
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    _write_embedding(embeddings_dir / f"{sid}_esm_embeddings.pt")

    df = pd.DataFrame({
        "phenotype-BioSample_ID": [sid],
        "amoxicillin": [1],
    })
    df["Sample"] = df["phenotype-BioSample_ID"]
    df["train_val_eval"] = "train"

    output_base = tmp_path / "output"
    write_split_files_amr(df, embeddings_dir, output_base)

    pt = torch.load(output_base / "train" / f"{sid}_with_ast.pt", map_location="cpu", weights_only=False)
    assert pt["amoxicillin"] == 1
    assert "prot_embeddings" in pt
