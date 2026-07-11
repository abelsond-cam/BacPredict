"""Unit tests for ``PanelInjectingFileDataset`` (surprisal-panel injection + count-guard).

Pure-torch + numpy with on-disk fixtures written to ``tmp_path`` — no Bacformer
download. Skipped where torch is unavailable (the local MacBook env).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bacpredict.engine.finetune.datasets import PanelInjectingFileDataset  # noqa: E402 - after importorskip

PANEL_COLUMNS = [
    "max_surprisal", "top2_surprisal", "top3_surprisal", "top10_surprisal",
    "p95_surprisal", "p90_surprisal", "median_surprisal", "participation_ratio", "kurtosis_surprisal",
]
DIM = 8


def _write_embedding(emb_dir: Path, sample: str, n: int) -> None:
    torch.save(
        {
            "protein_embeddings": torch.randn(1, n, DIM),
            "attention_mask": torch.ones(1, n),
            "contig_idx": torch.zeros(n, dtype=torch.long),
        },
        emb_dir / f"{sample}_esm_embeddings.pt",
    )


def _write_panel(panel_dir: Path, sample: str, n_panel: int) -> np.ndarray:
    panel = np.arange(n_panel * len(PANEL_COLUMNS), dtype=np.float32).reshape(n_panel, len(PANEL_COLUMNS))
    np.savez(
        panel_dir / f"{sample}_panel.npz",
        panel=panel,
        flat_index=np.arange(n_panel, dtype=np.int64),
        n_proteins=np.array(n_panel, dtype=np.int64),
        columns=np.array(PANEL_COLUMNS),
    )
    return panel


def _standardization(mean: float = 0.0, std: float = 1.0) -> dict:
    return {
        "columns": PANEL_COLUMNS,
        "mean": [mean] * len(PANEL_COLUMNS),
        "std": [std] * len(PANEL_COLUMNS),
    }


def test_panel_dataset_attaches_standardized_panel(tmp_path: Path) -> None:
    """``__getitem__`` returns the base sample plus a ``[1, n, 9]`` standardized panel."""
    emb_dir, panel_dir = tmp_path / "emb", tmp_path / "panel"
    emb_dir.mkdir()
    panel_dir.mkdir()
    n = 5
    _write_embedding(emb_dir, "S1", n)
    raw_panel = _write_panel(panel_dir, "S1", n)

    ds = PanelInjectingFileDataset(["S1"], emb_dir, {"S1": 1}, "drug", panel_dir, _standardization(mean=2.0, std=4.0))
    item = ds[0]

    assert item["panel"].shape == (1, n, len(PANEL_COLUMNS))
    expected = (raw_panel - 2.0) / 4.0
    assert np.allclose(item["panel"].squeeze(0).numpy(), expected, atol=1e-5)
    # The base sample keys are still present.
    assert item["protein_embeddings"].shape == (1, n, DIM)
    assert item["labels"].item() == 1


def test_panel_dataset_accepts_json_path(tmp_path: Path) -> None:
    """Standardization may be passed as a path to panel_standardization.json."""
    emb_dir, panel_dir = tmp_path / "emb", tmp_path / "panel"
    emb_dir.mkdir()
    panel_dir.mkdir()
    _write_embedding(emb_dir, "S1", 4)
    _write_panel(panel_dir, "S1", 4)
    std_path = tmp_path / "panel_standardization.json"
    std_path.write_text(json.dumps(_standardization()))

    ds = PanelInjectingFileDataset(["S1"], emb_dir, {"S1": 0}, "drug", panel_dir, std_path)
    assert ds[0]["panel"].shape == (1, 4, len(PANEL_COLUMNS))


def test_panel_dataset_panel_shorter_raises(tmp_path: Path) -> None:
    """A panel *shorter* than the embedding is a genuine misalignment and fails loudly."""
    emb_dir, panel_dir = tmp_path / "emb", tmp_path / "panel"
    emb_dir.mkdir()
    panel_dir.mkdir()
    _write_embedding(emb_dir, "S1", 6)
    _write_panel(panel_dir, "S1", 5)  # one fewer protein than the embedding

    ds = PanelInjectingFileDataset(["S1"], emb_dir, {"S1": 1}, "drug", panel_dir, _standardization())
    with pytest.raises(ValueError, match="mismatch"):
        _ = ds[0]


def test_panel_dataset_oversized_panel_truncates(tmp_path: Path) -> None:
    """An over-long panel (embedding capped at its first-N proteins) is truncated, not rejected.

    The embedding store caps each genome at ``max_n_proteins`` in flat order, while the panel
    build applies no cap — so an oversized genome's panel is longer than its embedding. The
    dataset keeps the panel's first ``n_proteins`` rows (same flat order).
    """
    emb_dir, panel_dir = tmp_path / "emb", tmp_path / "panel"
    emb_dir.mkdir()
    panel_dir.mkdir()
    n_emb = 6
    n_panel = 9  # genome had 9 proteins; the embedding store kept only the first 6
    _write_embedding(emb_dir, "S1", n_emb)
    raw_panel = _write_panel(panel_dir, "S1", n_panel)

    ds = PanelInjectingFileDataset(
        ["S1"], emb_dir, {"S1": 1}, "drug", panel_dir, _standardization(mean=2.0, std=4.0)
    )
    item = ds[0]

    assert item["panel"].shape == (1, n_emb, len(PANEL_COLUMNS))
    expected = (raw_panel[:n_emb] - 2.0) / 4.0  # first-N rows in flat order
    assert np.allclose(item["panel"].squeeze(0).numpy(), expected, atol=1e-5)


def test_panel_dataset_missing_panel_raises(tmp_path: Path) -> None:
    """A missing panel file raises FileNotFoundError."""
    emb_dir, panel_dir = tmp_path / "emb", tmp_path / "panel"
    emb_dir.mkdir()
    panel_dir.mkdir()
    _write_embedding(emb_dir, "S1", 3)  # no panel written

    ds = PanelInjectingFileDataset(["S1"], emb_dir, {"S1": 1}, "drug", panel_dir, _standardization())
    with pytest.raises(FileNotFoundError):
        _ = ds[0]


def test_panel_dataset_zero_std_no_nan(tmp_path: Path) -> None:
    """A zero std column is sanitised to finite values (nan_to_num)."""
    emb_dir, panel_dir = tmp_path / "emb", tmp_path / "panel"
    emb_dir.mkdir()
    panel_dir.mkdir()
    _write_embedding(emb_dir, "S1", 4)
    _write_panel(panel_dir, "S1", 4)
    std = _standardization()
    std["std"] = [0.0] * len(PANEL_COLUMNS)  # pathological: divide-by-zero before nan_to_num

    ds = PanelInjectingFileDataset(["S1"], emb_dir, {"S1": 1}, "drug", panel_dir, std)
    panel = ds[0]["panel"]
    assert torch.isfinite(panel).all()
