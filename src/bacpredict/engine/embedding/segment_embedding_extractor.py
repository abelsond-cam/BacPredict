"""Sweep a genome cohort → per-segment design matrices: the type-agnostic core of the ranking screen.

One :class:`bacpredict.engine.embedding.segment_locator.SegmentLocator` + a two-pass sweep replaces the
four copy-forked collectors — ``collect_igr_matrices``, ``collect_upstream_matrices``,
``collect_unit_matrices``, and the coding ``discover_core_genes`` + ``assemble_segment_matrices`` pair.
The locator supplies the per-type keying / GFF / dedup; everything below is type-agnostic.

**Two-pass, streamed** (one genome at a time — never hold every genome's records at once):

* **Pass 1** (fit genomes only, via ``locator.discover_ids``) tallies each segment's single-copy
  prevalence over the genomes it could read and keeps those in the ``(min_prevalence, max_prevalence]``
  band — the *core* set. This is the light pass: ``protein`` reads only the parquet gene list.
* **Pass 2** (fit **and** eval, via ``locator.records``) materialises the single-copy vectors for core
  segments only. The eval genomes are swept so the fitted LR can be scored on them, but they never entered
  pass-1 selection, so the evaluate split cannot influence which segments are screened.

Two invariants are load-bearing and reproduce each original collector with **no per-type flag**:

* **Prevalence denominator** = the fit-only read count from pass 1. For ``protein`` that is every fit id
  (its ``discover_ids`` returns ``[]``, never ``None``, for an unreadable genome — matching
  ``discover_core_genes``'s ``n = len(train_ids)``); for the non-coding types it is only the fit genomes
  read (their ``discover_ids`` returns ``None`` when unreadable). The ``None`` vs ``[]`` distinction lives
  in the locator, so this module stays uniform.
* **``read_ids`` = the impute universe** = the fit ∪ eval genomes successfully read in pass 2 (decoupled
  from the prevalence denominator, exactly as the coding screen always was). Including the eval genomes is
  what lets a *zero-imputed* fit compute a held-out ``eval_auroc`` too — the correctness spine that every
  LR, present-conditioned or imputed, evaluates on the holdout. (The four screens diverged here: coding
  included eval, the non-coding collectors did not, silently dropping the held-out eval of an imputed
  non-coding ranking. Unifying restores the invariant.)

The serial read is deliberate: ``torch.load(mmap=True)`` in the record extractors cannot be forked before
the process-parallel fit (``fit_per_segment``, ``n_jobs``) on aarch64 (Grace) without segfaulting, so the
sweep stays single-process and the fit fans out afterwards.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

from bacpredict.engine.embedding.segment_locator import SegmentLocator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def collect_segment_matrices(
    locator: SegmentLocator,
    sweep_ids: list[str],
    *,
    eval_ids: set[str] | None = None,
    min_prevalence: float = 0.0,
    max_prevalence: float = 1.0,
    store_dtype: str = "float32",
    id_column: str = "segment",
) -> tuple[dict[str, tuple[list[str], np.ndarray]], pd.DataFrame, list[str]]:
    """Two-pass sweep → ``({segment_id: (carrier_ids, X[m, dim])}, prevalence, read_ids)``.

    Parameters
    ----------
    locator
        The per-type :class:`SegmentLocator` (protein / igr / upstream / unit), bound to its stores/GFF.
    sweep_ids
        The genomes to sweep — the fit genomes, plus the evaluate genomes when a held-out eval is wanted
        (``sweep_ids = fit_train_ids + evaluate_ids``); ``eval_ids`` marks which of these are held out.
    eval_ids
        The evaluate-split ids to **exclude from pass-1 selection** but include in pass-2 (scored, never
        selected). ``None`` / empty → an OOF-only screen over the fit genomes.
    min_prevalence, max_prevalence
        The single-copy prevalence band ``(min, max]`` over the fit genomes — the core segment set. The
        default ``(0.0, 1.0]`` keeps every single-copy segment (band applied later, if at all).
    store_dtype
        Design-matrix storage precision (``float32`` default; ``float16`` halves the full-cohort footprint
        — the LR upcasts to float32 to fit).
    id_column
        The prevalence table's segment-id column name (``gene`` / ``igr_pair`` / ``upstream_gene`` /
        ``unit`` for the four types; generic ``segment`` by default).

    Returns
    -------
    matrices
        ``{segment_id: (carrier_ids, X[m, dim])}`` — the single-copy design matrix per core segment, over
        the genomes carrying it single-copy (present-only; the fit zero-imputes the rest onto ``read_ids``).
    prevalence
        ``[id_column, n_single_copy, prevalence]`` over every single-copy segment seen in pass 1, sorted
        by prevalence descending. Non-core segments appear here (for provenance) but not in ``matrices``.
    read_ids
        The impute universe — the fit ∪ eval genomes read in pass 2.
    """
    core, prevalence = sweep_core_prevalence(
        locator, sweep_ids, eval_ids=eval_ids, min_prevalence=min_prevalence,
        max_prevalence=max_prevalence, id_column=id_column,
    )
    matrices, read_ids = collect_core_subset(locator, sweep_ids, core, store_dtype=store_dtype)
    logger.info("segment sweep: materialised %d core matrices over %d read genomes (fit ∪ eval)",
                len(matrices), len(read_ids))
    return matrices, prevalence, read_ids


def sweep_core_prevalence(
    locator: SegmentLocator,
    sweep_ids: list[str],
    *,
    eval_ids: set[str] | None = None,
    min_prevalence: float = 0.0,
    max_prevalence: float = 1.0,
    id_column: str = "segment",
) -> tuple[set[str], pd.DataFrame]:
    """Pass 1 of the sweep: single-copy prevalence over the fit genomes → ``(core, prevalence)``.

    The cheap, vector-free half of :func:`collect_segment_matrices`: reads only ``discover_ids`` per fit
    genome (never the embedding vectors), so it can run standalone to size a batched pass 2. ``core`` is the
    set of single-copy segments whose fit prevalence falls in ``(min_prevalence, max_prevalence]``; the
    ``[id_column, n_single_copy, prevalence]`` table covers every single-copy segment seen.
    """
    eval_ids = eval_ids or set()
    sweep_ids = [str(s) for s in sweep_ids]
    fit_ids = [s for s in sweep_ids if s not in eval_ids]

    single_copy: Counter[str] = Counter()
    n_fit_read = 0
    n_skipped = 0
    for sid in fit_ids:
        ids = locator.discover_ids(sid)
        if ids is None:  # unreadable (non-coding) — skip, do not count toward the denominator
            n_skipped += 1
            continue
        n_fit_read += 1  # readable (protein: [] still counts, matching discover_core_genes' denominator)
        counts = Counter(ids)
        single_copy.update(k for k, c in counts.items() if c == 1)
    if n_skipped:
        logger.warning("segment sweep pass 1: skipped %d fit genomes (unreadable)", n_skipped)

    denom = max(n_fit_read, 1)
    prevalence = pd.DataFrame(
        [{id_column: k, "n_single_copy": c, "prevalence": c / denom} for k, c in single_copy.items()],
        columns=[id_column, "n_single_copy", "prevalence"],  # survive an empty sweep
    ).sort_values("prevalence", ascending=False).reset_index(drop=True)
    core = {k for k, c in single_copy.items() if min_prevalence < (c / denom) <= max_prevalence}
    logger.info("segment sweep: %d core segments in band (%.3f, %.3f] of %d single-copy over %d fit genomes",
                len(core), min_prevalence, max_prevalence, len(single_copy), n_fit_read)
    return core, prevalence


def collect_core_subset(
    locator: SegmentLocator,
    sweep_ids: list[str],
    core_subset: set[str],
    *,
    store_dtype: str = "float32",
) -> tuple[dict[str, tuple[list[str], np.ndarray]], list[str]]:
    """Pass 2 of the sweep, restricted to ``core_subset``: → ``({seg: (carrier_ids, X[m, dim])}, read_ids)``.

    Materialises the single-copy design matrices for **only** the segments in ``core_subset`` (a slice of the
    full ``core`` set from :func:`sweep_core_prevalence`). A caller processes the core set in memory-bounded
    batches — one full genome scan per batch, holding only that batch's matrices at once — which bounds peak
    RAM for clonal cohorts where thousands of core segments are each near-ubiquitous (dense ``m ≈ n_read``).
    ``read_ids`` (the impute universe: every readable swept genome, in sweep order) is identical across
    batches, so a caller captures it once.
    """
    sweep_ids = [str(s) for s in sweep_ids]
    ids_by_key: dict[str, list[str]] = {}
    vecs_by_key: dict[str, list[np.ndarray]] = {}
    read_ids: list[str] = []
    for sid in sweep_ids:
        recs = locator.records(sid)
        if recs is None:
            continue
        _sid, records = recs
        read_ids.append(sid)
        counts = Counter(k for k, _ in records)
        for key, vec in records:
            if counts[key] == 1 and key in core_subset:  # single-copy AND in this batch's slice of core
                ids_by_key.setdefault(key, []).append(sid)
                vecs_by_key.setdefault(key, []).append(np.asarray(vec).astype(store_dtype, copy=False))
    matrices = {k: (ids_by_key[k], np.vstack(vecs_by_key[k])) for k in ids_by_key}
    return matrices, read_ids
