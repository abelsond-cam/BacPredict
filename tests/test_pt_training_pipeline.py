"""Tests for the .pt-based AMR training pipeline."""

from predict_kleb_by_bacformer.pp.prepare_esmc_embeddings_and_labels_to_finetune_amr import (
    get_antibiotic_columns,
)
from predict_kleb_by_bacformer.pp.split_utils import add_splits
from predict_kleb_by_bacformer.tl.train_amr import PyTorchFileDataset


def test_pt_pipeline_imports():
    """Verify all pipeline modules can be imported."""
    from predict_kleb_by_bacformer.pp import prepare_esmc_embeddings_and_labels_to_finetune_amr  # noqa: F401
    from predict_kleb_by_bacformer.tl import train_amr  # noqa: F401


def test_pt_file_dataset_empty():
    """PyTorchFileDataset with empty file list has length 0."""
    ds = PyTorchFileDataset(file_paths=[], drug="ceftriaxone")
    assert len(ds) == 0


def test_add_splits():
    """add_splits produces 70/10/20 split over unique samples."""
    import pandas as pd

    df = pd.DataFrame({
        "phenotype-BioSample_ID": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"] * 2,
        "drug1": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 2,
    })
    df["Sample"] = df["phenotype-BioSample_ID"]
    result = add_splits(df, seed=42)
    assert "train_val_eval" in result.columns
    splits = result.groupby("Sample")["train_val_eval"].first()
    n_train = (splits == "train").sum()
    n_val = (splits == "validate").sum()
    n_eval = (splits == "evaluate").sum()
    assert n_train + n_val + n_eval == 10
    # 70/10/20 → 7, 1, 2
    assert n_train == 7
    assert n_val == 1
    assert n_eval == 2


def test_get_antibiotic_columns():
    """get_antibiotic_columns excludes ID/split columns."""
    import pandas as pd

    df = pd.DataFrame({
        "phenotype-BioSample_ID": ["A"],
        "Sample": ["A"],
        "train_val_eval": ["train"],
        "ceftriaxone": [0],
        "ampicillin": [1],
    })
    cols = get_antibiotic_columns(df)
    assert "ceftriaxone" in cols
    assert "ampicillin" in cols
    assert "Sample" not in cols
    assert "train_val_eval" not in cols
