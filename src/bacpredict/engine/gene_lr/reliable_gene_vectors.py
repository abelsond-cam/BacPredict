"""Generic per-label carrier ESM/embedding-vector collector — the sidecar-agnostic concat seam.

The organism-agnostic core cut out of ``apps.kleb.per_gene_lr_from_annotation.collect_reliable_amr``:
given, per genome, a set of gene *calls* (each a ``(label, flat_index, source, tag_match)``), pull the
carrier's embedding row ``emb[flat_index]`` out of the shared ESM store and accumulate, per label, its
carrier ids + vectors (+ the tagged subset). It is deliberately blind to **how** the calls are produced —
Bakta gene names, a CARD/Kleborate AMR sidecar, or anything else — so both organisms' per-gene-LR and
reliable-concat drivers share one collector, and the annotation that yields the calls stays in the app.

A ``calls_fn(sample_id, n_real) -> Iterable[ProteinCall]`` supplies the per-genome calls; the caller is
responsible for any label single-copy / source filtering *before* yielding (the collector just carries
what it is given). ``tag_match`` flags a call for the optional tagged subset (e.g. "Bakta also named this
family") — opaque here; the app decides its meaning.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple

from bacpredict.engine.embedding.segment_locator import read_genome

logger = logging.getLogger(__name__)

# A gene-label needs at least this many single-copy carriers in the holdout to be scored (below this the
# per-gene AUROC is too noisy). Generic threshold — the app's reliability rule is upstream, in calls_fn.
MIN_CARRIERS = 8


class ProteinCall(NamedTuple):
    """One per-genome gene call the collector carries into a per-label vector set.

    Attributes
    ----------
    label : str
        The gene/label key vectors accumulate under (CARD family/allele, Bakta symbol, …).
    flat_index : int
        Row of this call's protein in the genome's flat ESM order — the vector pulled is ``emb[flat_index]``.
    source : str
        A per-label provenance tag stored once on the label entry (e.g. ``acquired`` / ``chromosomal``).
    tag_match : bool
        Whether this carrier also belongs to the optional tagged subset (app-defined, e.g. Bakta-named).
    """

    label: str
    flat_index: int
    source: str
    tag_match: bool


CallsFn = Callable[[str, int], Iterable[ProteinCall]]


def collect_reliable_gene_vectors(
    eval_ids: list[str], esm_dir: Path, parquet_dir: Path, calls_fn: CallsFn
) -> tuple[list[str], dict[str, dict]]:
    """One pass over ``eval_ids`` → per label, its carriers + ESM vectors (+ the tagged subset).

    For each genome, :func:`bacpredict.engine.embedding.segment_locator.read_genome` gives the flat
    ESM matrix; ``calls_fn(sample_id, n_real)`` yields that genome's gene calls, and each call's
    ``emb[flat_index]`` is appended under its label. Returns ``(read_ids, by_label)`` where ``read_ids``
    is the genomes successfully read (the zero-impute universe) and
    ``by_label[label] = {"source", "ids", "vecs", "tag_ids"}`` — carrier ids, their vectors, and the
    ``tag_match`` subset. Callers that need a different subset key (e.g. ``bakta_ids``) rename ``tag_ids``.
    """
    by_label: dict[str, dict] = {}
    read_ids: list[str] = []
    n_skip_read = 0
    for k, sid in enumerate(eval_ids, 1):
        read = read_genome(sid, esm_dir, parquet_dir)
        if read is None:
            n_skip_read += 1
            continue
        read_ids.append(sid)
        _gene_names, emb = read
        n_real = emb.shape[0]
        for call in calls_fn(sid, n_real):
            ent = by_label.setdefault(
                call.label, {"source": call.source, "ids": [], "vecs": [], "tag_ids": set()}
            )
            ent["ids"].append(sid)
            ent["vecs"].append(emb[call.flat_index])
            if call.tag_match:
                ent["tag_ids"].add(sid)
        if k % 300 == 0:
            logger.info("  reliable gene-vector extract: %d/%d genomes", k, len(eval_ids))
    if n_skip_read:
        logger.warning("reliable gene-vector extract: skipped %d genomes (unread/misaligned)", n_skip_read)
    return read_ids, by_label
