"""Shared concat-ingredient helpers — impute a per-gene block, read cached FT/frozen vectors.

The "ESM/Bacformer gene vector ⊕ genome mean" concat probes each need to (a) place a gene's vector at
the right genome rows and zero-fill the non-carriers, and (b) read the cached fine-tuned genome-mean /
per-gene token stores. These were copy-pasted across the kleb concat drivers (``reliable_ft_concat``,
``concat_gene_panel_kleb``, ``gene_ingredient_concat``); they live here once.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def impute_block(present_ids: list[str], present_vecs: np.ndarray, all_ids: list[str], dim: int) -> np.ndarray:
    """``[len(all_ids), dim]`` block: the gene's real vector where carried single-copy, else a 0-vector."""
    pos = {s: i for i, s in enumerate(all_ids)}
    block = np.zeros((len(all_ids), dim), dtype=np.float32)
    rows = [pos[s] for s in present_ids if s in pos]
    if rows:
        block[rows] = present_vecs[: len(rows)]
    return block


def load_ft_mean(ft_cache_dir: Path, drug: str, label_map: dict[str, int]) -> tuple[list[str], np.ndarray]:
    """FT genome-mean over the eval holdout → ``(all_ids, mean_block)`` restricted to labelled genomes."""
    npz = np.load(ft_cache_dir / f"ft_genome_mean_{drug}.npz", allow_pickle=True)
    ids = [str(s) for s in npz["sample_ids"]]
    vecs = npz["mean_vectors"]
    pos = {s: i for i, s in enumerate(ids)}
    all_ids = [s for s in ids if s in label_map]
    return all_ids, np.vstack([vecs[pos[s]] for s in all_ids]).astype(np.float32)


def load_ft_gene(ft_cache_dir: Path, sanitized: str) -> tuple[list[str], np.ndarray]:
    """One family's FT tokens from the cache → ``(carrier_ids, vectors)``."""
    z = np.load(ft_cache_dir / "ft_amr_emb" / f"{sanitized}.npz", allow_pickle=True)
    return [str(s) for s in z["sample_ids"]], z["vectors"]


def load_frozen_gene(frozen_cache_dir: Path, sanitized: str) -> tuple[list[str], np.ndarray]:
    """One family's *frozen* Bacformer tokens from the cache → ``(carrier_ids, vectors)``."""
    z = np.load(frozen_cache_dir / "frozen_amr_emb" / f"{sanitized}.npz", allow_pickle=True)
    return [str(s) for s in z["sample_ids"]], z["vectors"]
