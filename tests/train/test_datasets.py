"""Tests for bacpredict.train.datasets.LabelInjectingFileDataset."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from bacpredict.train.datasets import LabelInjectingFileDataset


def _write_embedding(path: Path, n_proteins: int = 5, embed_dim: int = 8, with_contig: bool = True) -> None:
    """Write a minimal synthetic ESM embedding .pt file."""
    data = {
        "prot_embeddings": torch.randn(n_proteins, embed_dim),
        "attention_mask": torch.ones(n_proteins, dtype=torch.float32),
    }
    if with_contig:
        data["contig_idx"] = torch.zeros(n_proteins, dtype=torch.long)
    torch.save(data, path)


def _make_dataset(tmp_path: Path, n: int = 3) -> tuple[LabelInjectingFileDataset, dict]:
    """Create n synthetic samples and return the dataset + label_map."""
    sample_ids = [f"S{i:03d}" for i in range(n)]
    label_map = {sid: i % 2 for i, sid in enumerate(sample_ids)}
    for sid in sample_ids:
        _write_embedding(tmp_path / f"{sid}_esm_embeddings.pt")
    ds = LabelInjectingFileDataset(
        sample_ids=sample_ids,
        embeddings_dir=tmp_path,
        label_map=label_map,
        label_column="test_label",
    )
    return ds, label_map


def test_len(tmp_path):
    """__len__ matches the number of sample IDs provided."""
    ds, _ = _make_dataset(tmp_path, n=3)
    assert len(ds) == 3


def test_empty(tmp_path):
    """Empty sample list gives length 0."""
    ds = LabelInjectingFileDataset([], tmp_path, {}, "label")
    assert len(ds) == 0


def test_getitem_label(tmp_path):
    """__getitem__ injects the correct label from label_map."""
    ds, label_map = _make_dataset(tmp_path, n=4)
    for idx in range(len(ds)):
        sample = ds[idx]
        expected = label_map[ds.sample_ids[idx]]
        assert sample["labels"].item() == pytest.approx(expected)
        assert sample["labels"].dtype == torch.float32


def test_getitem_keys(tmp_path):
    """__getitem__ returns the expected keys."""
    ds, _ = _make_dataset(tmp_path, n=2)
    sample = ds[0]
    assert set(sample.keys()) == {"protein_embeddings", "labels", "attention_mask", "contig_ids"}


def test_getitem_shapes(tmp_path):
    """Embeddings are 3-D (1, seq_len, embed_dim) after unsqueeze."""
    ds, _ = _make_dataset(tmp_path, n=2)
    sample = ds[0]
    assert sample["protein_embeddings"].dim() == 3
    assert sample["protein_embeddings"].shape[0] == 1  # batch dim added
    assert sample["protein_embeddings"].shape[1] == 5  # n_proteins
    assert sample["protein_embeddings"].shape[2] == 8  # embed_dim


def test_missing_file_raises(tmp_path):
    """FileNotFoundError raised when an embedding file is absent."""
    ds = LabelInjectingFileDataset(["MISSING"], tmp_path, {"MISSING": 0}, "label")
    with pytest.raises(FileNotFoundError):
        ds[0]


def test_fallback_attention_mask(tmp_path):
    """A ones attention_mask is synthesised when absent from the .pt file."""
    sid = "S000"
    torch.save({"prot_embeddings": torch.randn(4, 8)}, tmp_path / f"{sid}_esm_embeddings.pt")
    ds = LabelInjectingFileDataset([sid], tmp_path, {sid: 1}, "label")
    sample = ds[0]
    assert sample["attention_mask"].shape == (1, 4)
    assert sample["attention_mask"].all()


def test_fallback_contig_ids(tmp_path):
    """Zero contig_ids are synthesised when absent from the .pt file."""
    sid = "S000"
    _write_embedding(tmp_path / f"{sid}_esm_embeddings.pt", with_contig=False)
    ds = LabelInjectingFileDataset([sid], tmp_path, {sid: 0}, "label")
    sample = ds[0]
    assert (sample["contig_ids"] == 0).all()


def _collate_labels_only(samples: list[dict]) -> dict:
    return {"labels": torch.stack([s["labels"] for s in samples])}


def test_dataloader_multiprocessing(tmp_path):
    """DataLoader with num_workers=2 iterates without error (multiprocessing safety)."""
    ds, _ = _make_dataset(tmp_path, n=4)
    loader = DataLoader(ds, batch_size=2, num_workers=2, collate_fn=_collate_labels_only)
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0]["labels"].shape == (2,)
