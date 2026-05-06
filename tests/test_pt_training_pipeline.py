"""Tests for the .pt-based AMR training pipeline."""

from pathlib import Path

import pandas as pd
import torch

from predict_kleb_by_bacformer.pp.prepare_esmc_embeddings_and_labels_to_finetune_amr import (
    get_antibiotic_columns,
)
from predict_kleb_by_bacformer.pp.split_utils import add_splits
from predict_kleb_by_bacformer.tl.datasets import LabelInjectingFileDataset
from predict_kleb_by_bacformer.tl.train_amr import PyTorchFileDataset


def _write_embedding(path: Path, n_proteins: int = 6) -> None:
    torch.save({"prot_embeddings": torch.randn(n_proteins, 8), "attention_mask": torch.ones(n_proteins)}, path)


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


def test_label_injecting_dataset_from_csv(tmp_path):
    """LabelInjectingFileDataset loads correct label when built from a split CSV."""
    sample_ids = ["S001", "S002", "S003"]
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    for sid in sample_ids:
        _write_embedding(embeddings_dir / f"{sid}_esm_embeddings.pt")

    df = pd.DataFrame({
        "Sample": sample_ids,
        "train_val_eval": ["train", "train", "validate"],
        "amikacin": [0, 1, 0],
    })
    label_map = {row["Sample"]: int(row["amikacin"]) for _, row in df.iterrows() if pd.notna(row["amikacin"])}

    train_ids = df[df["train_val_eval"] == "train"]["Sample"].tolist()
    ds = LabelInjectingFileDataset(train_ids, embeddings_dir, label_map, "amikacin")

    assert len(ds) == 2

    item0 = ds[0]
    assert item0["labels"].item() == label_map["S001"]
    assert "protein_embeddings" in item0
    assert "attention_mask" in item0
    assert "contig_ids" in item0


def test_label_injecting_dataset_label_values(tmp_path):
    """LabelInjectingFileDataset returns the correct label for each sample."""
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    labels = {"SA": 1, "SB": 0, "SC": 1}
    for sid in labels:
        _write_embedding(embeddings_dir / f"{sid}_esm_embeddings.pt")

    ds = LabelInjectingFileDataset(list(labels.keys()), embeddings_dir, labels, "drug")
    retrieved = {ds.sample_ids[i]: ds[i]["labels"].item() for i in range(len(ds))}
    assert retrieved == {k: float(v) for k, v in labels.items()}


# ── k-fold integration (Step 6) ───────────────────────────────────────────────


def test_kfold_validate_sets_non_overlapping():
    """Fold 0 and fold 4 validation sets are disjoint (each sample validates exactly once)."""
    from predict_kleb_by_bacformer.pp.split_utils import generate_kfold_splits

    df = pd.DataFrame({"Sample": [f"S{i:03d}" for i in range(50)], "drug": [i % 2 for i in range(50)]})
    _, folds = generate_kfold_splits(df, n_folds=5, seed=1)
    _, val0 = folds[0]
    _, val4 = folds[4]
    assert val0.isdisjoint(val4), "Fold 0 and fold 4 validation sets unexpectedly overlap"


def test_kfold_evaluate_stable_between_fold_and_seed():
    """Evaluate set is identical for fold=0/seed=1 and fold=3/seed=2 with same evaluate_seed."""
    from predict_kleb_by_bacformer.pp.split_utils import generate_kfold_splits

    df = pd.DataFrame({"Sample": [f"S{i:03d}" for i in range(50)], "drug": [i % 2 for i in range(50)]})
    eval1, _ = generate_kfold_splits(df, n_folds=5, seed=1, evaluate_seed=42)
    eval2, _ = generate_kfold_splits(df, n_folds=5, seed=2, evaluate_seed=42)
    assert eval1 == eval2


def test_kfold_legacy_csv_mode_preserved(tmp_path):
    """With n_folds=None, train_val_eval column from CSV is used (backward compat)."""
    from predict_kleb_by_bacformer.pp.split_utils import add_splits

    sample_ids = ["S001", "S002", "S003", "S004", "S005"]
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    for sid in sample_ids:
        _write_embedding(embeddings_dir / f"{sid}_esm_embeddings.pt")

    df = pd.DataFrame({"Sample": sample_ids, "amikacin": [0, 1, 0, 1, 0]})
    df = add_splits(df, seed=1)
    sheet = tmp_path / "sheet.csv"
    df.to_csv(sheet, index=False)

    loaded = pd.read_csv(sheet)
    assert "train_val_eval" in loaded.columns
    train_ids = loaded[loaded["train_val_eval"] == "train"]["Sample"].tolist()
    val_ids = loaded[loaded["train_val_eval"] == "validate"]["Sample"].tolist()
    assert set(train_ids).isdisjoint(set(val_ids))
