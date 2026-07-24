"""Pull one segment's pooled 960-vector out of the per-protein ``.pt`` embedding store, in flat order.

Lazy by construction — each ``.pt`` is mmap'd and only the single flat-index row is materialised (loading
the whole ``[1, n_proteins, dim]`` tensor OOMs over tens of thousands of genomes). Generic over the
segment: the caller supplies the column holding each genome's flat protein index (a CDS's index from
:func:`bacpredict.engine.gene_lr.locate_gene.build_gene_presence_table`, or an rpoB index from a genotype
table), so the same reader serves the coding read-out steps.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bacpredict.engine.embedding.protein_pooling import real_protein_indices

logger = logging.getLogger(__name__)


def _read_pooled_one(
    sample_id: str,
    flat_index: int,
    n_expected: int | None,
    pt_path: Path,
) -> tuple[str, np.ndarray | None, str | None]:
    """Read one sample's pooled segment vector from its mmap'd ``.pt`` (worker-safe).

    Returns ``(sample_id, vector_or_None, skip_reason)``. ``skip_reason`` is one of ``"missing_pt"`` /
    ``"count_mismatch"`` / ``"out_of_range"`` or ``None``.
    """
    if not pt_path.exists():
        return sample_id, None, "missing_pt"
    # mmap so a single row is read out of the file instead of loading the whole [1, n_proteins, dim]
    # tensor into RAM (otherwise ~15 MB × ~38k OOMs).
    store = torch.load(pt_path, map_location="cpu", mmap=True)
    prot_emb = store["protein_embeddings"][0]
    real_idx = real_protein_indices(store, prot_emb.shape[0])
    # Guard against silent flat-order misalignment: the real-protein row count must match the parquet's
    # flat protein count, or the flat index is meaningless.
    if n_expected is not None and real_idx.numel() != n_expected:
        return sample_id, None, "count_mismatch"
    if flat_index >= real_idx.numel():
        return sample_id, None, "out_of_range"
    raw = int(real_idx[flat_index])
    return sample_id, prot_emb[raw].float().clone().numpy(), None


def load_pooled_gene_vectors(
    gene_table: pd.DataFrame,
    esm_store_dir: Path,
    *,
    flat_index_col: str = "gene_flat_index",
    pt_suffix: str = "_esm_embeddings.pt",
    pool_workers: int = 1,
) -> pd.DataFrame:
    """Pull each sample's pooled ESM-C **segment** 960-vector out of the embedding store.

    Generic over the segment: ``flat_index_col`` names the column holding its flat protein index
    (``"gene_flat_index"`` from :func:`bacpredict.engine.gene_lr.locate_gene.build_gene_presence_table`, or
    ``"rpob_flat_index"`` from the rpoB genotype table). Lazy by construction — each ``.pt`` is mmap'd and
    only the single row is materialised. Returns a DataFrame of the recovered vectors indexed by Sample
    (samples whose ``.pt`` is missing or whose index fails the guards are dropped). ``pool_workers > 1``
    reads in parallel with a ``multiprocessing.Pool`` (not a DataLoader — that exhausts file descriptors
    over tens of thousands of single-row reads).
    """
    tasks = [
        (
            str(sample_id),
            int(row[flat_index_col]),
            int(row["n_proteins"]) if "n_proteins" in row and not pd.isna(row["n_proteins"]) else None,
            esm_store_dir / f"{sample_id}{pt_suffix}",
        )
        for sample_id, row in gene_table.iterrows()
    ]

    results: list[tuple[str, np.ndarray | None, str | None]]
    if pool_workers > 1:
        import multiprocessing as mp

        with mp.Pool(pool_workers) as pool:
            results = pool.starmap(_read_pooled_one, tasks)
    else:
        results = [_read_pooled_one(*t) for t in tasks]

    skips: dict[str, int] = {}
    vectors: list[np.ndarray] = []
    kept: list[str] = []
    for sample_id, vec, reason in results:
        if reason is not None:
            skips[reason] = skips.get(reason, 0) + 1
            continue
        vectors.append(vec)
        kept.append(sample_id)
    if skips:
        logger.warning("pooled segment vectors: skipped %s", skips)
    if not vectors:
        return pd.DataFrame()
    return pd.DataFrame(np.vstack(vectors), index=pd.Index(kept, name="Sample"))
