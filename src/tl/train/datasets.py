"""Shared PyTorch dataset classes for Bacformer fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class LabelInjectingFileDataset(torch.utils.data.Dataset):
    """Lazy dataset that loads original ESM embedding files and injects labels at read time.

    No pre-built labeled ``.pt`` copies are required. The ``label_map`` dict is small
    (sample_id → int) and is safely copied to DataLoader worker processes via pickling.
    The embedding tensors are loaded one file at a time in ``__getitem__`` — the full
    dataset is never resident in memory simultaneously.

    Parameters
    ----------
    sample_ids : list[str]
        Ordered list of sample IDs to serve. Controls ``__len__`` and index→sample mapping.
    embeddings_dir : Path
        Directory containing ``{sample_id}_esm_embeddings.pt`` files.
    label_map : dict[str, int]
        Mapping from sample_id to integer label (0 or 1 for binary tasks).
    label_column : str
        Name used to describe the label in log messages (e.g. ``"blood_vs_faeces_label"``).
    """

    def __init__(
        self,
        sample_ids: list[str],
        embeddings_dir: Path,
        label_map: dict[str, int],
        label_column: str,
    ) -> None:
        self.sample_ids = list(sample_ids)
        self.embeddings_dir = Path(embeddings_dir)
        self.label_map = label_map
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sample_id = self.sample_ids[idx]
        embed_path = self.embeddings_dir / f"{sample_id}_esm_embeddings.pt"

        if not embed_path.exists():
            raise FileNotFoundError(f"Embedding file not found for {sample_id}: {embed_path}")

        data = torch.load(embed_path, map_location="cpu", weights_only=False)

        prot_embeddings = data.get("prot_embeddings", data.get("protein_embeddings"))
        if prot_embeddings is None:
            raise KeyError(f"Sample {sample_id} is missing 'prot_embeddings'/'protein_embeddings'.")

        if prot_embeddings.dim() == 2:
            prot_embeddings = prot_embeddings.unsqueeze(0)

        seq_len = prot_embeddings.shape[1]
        label_val = self.label_map[sample_id]

        sample: dict = {
            "protein_embeddings": prot_embeddings,
            "labels": torch.tensor(label_val, dtype=torch.float32),
        }

        am = data.get("attention_mask")
        sample["attention_mask"] = am if am is not None else torch.ones(1, seq_len, dtype=torch.float32)

        contig_src = data.get("contig_idx", data.get("contig_ids", data.get("token_type_ids")))
        if contig_src is not None:
            sample["contig_ids"] = contig_src.unsqueeze(0) if contig_src.dim() == 1 else contig_src
        else:
            sample["contig_ids"] = torch.zeros(1, seq_len, dtype=torch.long)

        return sample


class PanelInjectingFileDataset(LabelInjectingFileDataset):
    """:class:`LabelInjectingFileDataset` that also injects the per-protein surprisal panel.

    Loads a sibling ``{sample_id}_panel.npz`` (``panel`` ``[n_proteins, panel_dim]`` in flat
    protein order, built by ``snp_embeddings.build_panel_store``), standardises it with a
    train-only mean/std, and attaches it as ``sample["panel"]`` of shape ``[1, n, panel_dim]``
    so it concatenates onto the backbone tokens in the attention pool. ``none``-mode runs keep
    using the plain :class:`LabelInjectingFileDataset` — this subclass is opt-in.

    The panel rows must align with the embedding's protein rows: ``__getitem__`` enforces
    ``panel.shape[0] == n_proteins`` (mirrors the count-guard in
    ``snp_embeddings.snp_vs_esm_prediction``) so a flat-order misalignment fails loudly rather
    than silently feeding the gate the wrong protein's anomaly score.

    Parameters
    ----------
    sample_ids, embeddings_dir, label_map, label_column
        As :class:`LabelInjectingFileDataset`.
    panel_dir : Path
        Directory of ``{sample_id}_panel.npz`` files.
    standardization : dict or Path
        ``panel_standardization.json`` (or its parsed dict) with ``columns``/``mean``/``std``.
    """

    def __init__(
        self,
        sample_ids: list[str],
        embeddings_dir: Path,
        label_map: dict[str, int],
        label_column: str,
        panel_dir: Path,
        standardization: dict | Path,
    ) -> None:
        super().__init__(sample_ids, embeddings_dir, label_map, label_column)
        self.panel_dir = Path(panel_dir)
        if isinstance(standardization, (str, Path)):
            standardization = json.loads(Path(standardization).read_text())
        self.panel_columns = list(standardization["columns"])
        self.panel_mean = np.asarray(standardization["mean"], dtype=np.float32)
        self.panel_std = np.asarray(standardization["std"], dtype=np.float32)
        self.panel_dim = len(self.panel_columns)

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        sample_id = self.sample_ids[idx]
        n_proteins = sample["protein_embeddings"].shape[1]

        panel_path = self.panel_dir / f"{sample_id}_panel.npz"
        if not panel_path.exists():
            raise FileNotFoundError(f"Panel file not found for {sample_id}: {panel_path}")
        with np.load(panel_path) as z:
            panel = z["panel"].astype(np.float32)

        if panel.shape[0] != n_proteins:
            raise ValueError(
                f"Panel/embedding flat-order mismatch for {sample_id}: panel has {panel.shape[0]} "
                f"proteins but the embedding has {n_proteins}."
            )

        panel = (panel - self.panel_mean) / self.panel_std
        panel = np.nan_to_num(panel, nan=0.0, posinf=0.0, neginf=0.0)
        sample["panel"] = torch.from_numpy(panel).unsqueeze(0)  # [1, n_proteins, panel_dim]
        return sample
