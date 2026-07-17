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


# ---------------------------------------------------------------------------
# baclm block loaders — the best-gene / best-IGR rungs of the concat ladder.
# Unlike the FT/frozen token caches above (one NPZ per family), the baclm blocks are read live from the
# per-sample ``{sample}_baclm_embeddings.pt`` store for a SINGLE selected gene / IGR pair, reusing the
# same present-carrier readers the ranking stores fit on. The caller ``impute_block``s the returned
# carriers onto the FT-mean universe (non-carriers → 0-vector), exactly like a cached gene block.
# ---------------------------------------------------------------------------


def load_baclm_gene_block(
    sample_ids: list[str], gene: str, *, baclm_dir: Path, parquet_dir: Path,
) -> tuple[list[str], np.ndarray]:
    """Carrier ``(ids, [n, dim])`` for ONE ``gene``'s single-copy baclm coding vector across ``sample_ids``.

    Reuses :func:`build_per_gene_lr_store.read_genome` (``store_kind="baclm"``) — the parquet gene list
    aligned to the baclm ``[n_cds, dim]`` matrix — and keeps the row where ``gene`` occurs exactly once
    (single-copy), the same carrier universe the per-gene ranking fits. Non-carriers are dropped here;
    the caller zero-imputes them onto the full universe.
    """
    from bacpredict.engine.gene_lr.build_per_gene_lr_store import read_genome

    ids: list[str] = []
    vecs: list[np.ndarray] = []
    for sid in sample_ids:
        read = read_genome(str(sid), Path(baclm_dir), Path(parquet_dir), store_kind="baclm")
        if read is None:
            continue
        gene_names, emb = read
        idxs = [i for i, g in enumerate(gene_names) if g == gene]
        if len(idxs) == 1:  # single-copy occurrence in this genome
            ids.append(str(sid))
            vecs.append(emb[idxs[0]])
    return ids, (np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, 0), np.float32))


def load_baclm_igr_block(
    sample_ids: list[str], igr_pair: str, *, baclm_dir: Path, sample_gff: dict[str, str],
    boundary_tol: int = 3, baclm_suffix: str = "_baclm_embeddings.pt",
) -> tuple[list[str], np.ndarray]:
    """Carrier ``(ids, [n, dim])`` for ONE ``left→right`` IGR pair's baclm non-coding vector.

    Reuses :func:`build_per_igr_lr_store._genome_igr_records` (GFF flank-join over the ``noncoding_*``
    store rows) and keeps the genome iff the pair occurs exactly once. ``igr_pair`` is the ``left→right``
    key exactly as written in ``per_igr_lr_<drug>.csv``.
    """
    from bacpredict.engine.gene_lr.build_per_igr_lr_store import _genome_igr_records

    ids: list[str] = []
    vecs: list[np.ndarray] = []
    for sid in sample_ids:
        gff = sample_gff.get(str(sid))
        if not gff:
            continue
        res = _genome_igr_records(str(sid), gff, str(Path(baclm_dir) / f"{sid}{baclm_suffix}"), boundary_tol)
        if res is None:
            continue
        matches = [emb for pair, emb in res[1] if pair == igr_pair]
        if len(matches) == 1:  # single-copy pair in this genome
            ids.append(str(sid))
            vecs.append(matches[0])
    return ids, (np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, 0), np.float32))


def load_baclm_upstream_block(
    sample_ids: list[str], gene: str, *, baclm_dir: Path, sample_gff: dict[str, str],
    boundary_tol: int = 3, baclm_suffix: str = "_baclm_embeddings.pt",
) -> tuple[list[str], np.ndarray]:
    """Carrier ``(ids, [n, dim])`` for the region 5′ of ONE ``gene`` (the ``upstream:<gene>`` anchor).

    The promoter-anchored sibling of :func:`load_baclm_igr_block`, reusing
    :func:`build_upstream_region_lr_store._genome_upstream_records`. This is the loader for the recovered
    operon promoters (e.g. ``upstream:fabg1`` = the mabA-inhA promoter for ethionamide) the flank-pair
    scheme drops. ``gene`` is the bare downstream gene (the ranking's ``gene`` column, e.g. ``fabg1``).
    """
    from bacpredict.engine.gene_lr.build_upstream_region_lr_store import _genome_upstream_records

    key = f"upstream:{gene}"
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    for sid in sample_ids:
        gff = sample_gff.get(str(sid))
        if not gff:
            continue
        res = _genome_upstream_records(str(sid), gff, str(Path(baclm_dir) / f"{sid}{baclm_suffix}"), boundary_tol)
        if res is None:
            continue
        matches = [emb for k, emb in res[1] if k == key]
        if len(matches) == 1:
            ids.append(str(sid))
            vecs.append(matches[0])
    return ids, (np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, 0), np.float32))
