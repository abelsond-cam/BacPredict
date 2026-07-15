"""Shared concat-ingredient helpers — impute a per-gene block, read cached FT/frozen vectors.

The "ESM/Bacformer gene vector ⊕ genome mean" concat probes each need to (a) place a gene's vector at
the right genome rows and zero-fill the non-carriers, and (b) read the cached fine-tuned genome-mean /
per-gene token stores. These were copy-pasted across the concat drivers (``reliable_concat``, ``concat_gene_panel``,
``gene_ingredient_concat``, all now under :mod:`bacpredict.engine.concat`); they live here once.
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


def load_genome_mean(
    cache_dir: Path, drug: str, label_map: dict[str, int], *, prefix: str = "ft"
) -> tuple[list[str], np.ndarray]:
    """Cached genome-mean over the eval holdout → ``(all_ids, mean_block)`` restricted to labelled genomes.

    ``prefix`` selects the backbone whose mean was cached — ``ft`` (fine-tuned) reads
    ``ft_genome_mean_<drug>.npz``, ``frozen`` reads ``frozen_genome_mean_<drug>.npz``; the two stores are
    identical in layout ({sample_ids, mean_vectors}), differing only in the file prefix.
    """
    npz = np.load(cache_dir / f"{prefix}_genome_mean_{drug}.npz", allow_pickle=True)
    ids = [str(s) for s in npz["sample_ids"]]
    vecs = npz["mean_vectors"]
    pos = {s: i for i, s in enumerate(ids)}
    all_ids = [s for s in ids if s in label_map]
    return all_ids, np.vstack([vecs[pos[s]] for s in all_ids]).astype(np.float32)


def load_ft_mean(ft_cache_dir: Path, drug: str, label_map: dict[str, int]) -> tuple[list[str], np.ndarray]:
    """FT genome-mean over the eval holdout — :func:`load_genome_mean` with ``prefix="ft"``."""
    return load_genome_mean(ft_cache_dir, drug, label_map, prefix="ft")


def load_frozen_mean(frozen_cache_dir: Path, drug: str, label_map: dict[str, int]) -> tuple[list[str], np.ndarray]:
    """Frozen-Bacformer genome-mean over the eval holdout — :func:`load_genome_mean` with ``prefix="frozen"``."""
    return load_genome_mean(frozen_cache_dir, drug, label_map, prefix="frozen")


def load_ft_gene(ft_cache_dir: Path, sanitized: str) -> tuple[list[str], np.ndarray]:
    """One family's FT tokens from the cache → ``(carrier_ids, vectors)``."""
    z = np.load(ft_cache_dir / "ft_amr_emb" / f"{sanitized}.npz", allow_pickle=True)
    return [str(s) for s in z["sample_ids"]], z["vectors"]


def load_frozen_gene(frozen_cache_dir: Path, sanitized: str) -> tuple[list[str], np.ndarray]:
    """One family's *frozen* Bacformer tokens from the cache → ``(carrier_ids, vectors)``."""
    z = np.load(frozen_cache_dir / "frozen_amr_emb" / f"{sanitized}.npz", allow_pickle=True)
    return [str(s) for s in z["sample_ids"]], z["vectors"]
