"""Shared PyTorch dataset classes for Bacformer fine-tuning."""

from __future__ import annotations

from pathlib import Path

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
