"""Class-balanced subsampling of a split's sample ids — the expedient train-set reducer.

Per-segment LR fitting (and FT-mean caching) over the full ~24k train cohort is I/O-heavy; for a first
pass we fit on a random, class-balanced subsample of the ``train`` ids returned by
:func:`bacpredict.engine.splits.load_splits.load_splits`. This lives in ``splits/`` — the bottom layer —
so both the ranking screens (:mod:`bacpredict.engine.segment_amr_lr.per_segment_lr`) and the embedding
cache (:mod:`bacpredict.engine.concat.cache_bacformer_gene_embeddings`) draw the same reducer.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def subsample_balanced(
    ids: list[str], label_map: dict[str, int], *, max_n: int | None, seed: int
) -> list[str]:
    """Random, **class-balanced** subsample of ``ids`` to ~``max_n`` (or all if ``max_n`` is None/larger).

    Balancing keeps both resistance classes represented so a segment's AUROC stays estimable. Returns at
    most ``max_n`` ids (≈half per class; the smaller class caps its half, the larger backfills the rest).
    ``None`` / a too-large ``max_n`` returns ``ids`` unchanged. Deterministic in ``seed``.
    """
    if max_n is None or max_n >= len(ids):
        return ids
    rng = np.random.default_rng(seed)
    pos = [s for s in ids if label_map.get(s) == 1]
    neg = [s for s in ids if label_map.get(s) == 0]
    half = max_n // 2
    n_pos = min(len(pos), half)
    n_neg = min(len(neg), max_n - n_pos)
    n_pos = min(len(pos), max_n - n_neg)  # backfill from the larger class if one is short
    picked = [pos[i] for i in rng.choice(len(pos), size=n_pos, replace=False)] if n_pos else []
    picked += [neg[i] for i in rng.choice(len(neg), size=n_neg, replace=False)] if n_neg else []
    rng.shuffle(picked)
    logger.info("subsampled train: %d of %d (pos=%d neg=%d, seed=%d)", len(picked), len(ids), n_pos, n_neg, seed)
    return picked
