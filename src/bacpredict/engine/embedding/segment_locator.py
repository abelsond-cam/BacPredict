"""Locate each AMR segment type in one genome → its per-segment embedding vector(s).

A *segment* is the unit the per-segment LR screen ranks, and there are four types:

- **protein** — a coding gene's ESM-C (``esm``) or baclm coding vector, keyed by ``gene_name``;
- **igr** — an intergenic region, keyed by its sorted flanking-gene pair ``a→b`` (baclm non-coding);
- **upstream** — the region 5′ of a gene, keyed ``upstream:<gene>`` (baclm non-coding);
- **unit** — a named non-CDS body, keyed ``<feature_type>:<feature_name>`` (baclm re-embed).

Each per-type locator binds its stores/GFF once and exposes the uniform seam the extractor sweeps::

    records(sample_id)      -> (sample_id, [(segment_id, vector), ...]) | None   # None = unreadable → skip
    discover_ids(sample_id) -> list[str] | None                                  # the cheap prevalence pass

``records`` returns **every** occurrence, so the sweep applies the single-copy gate uniformly; ``unit``
already mean-pools its several copies to one row per unit (its relaxed gate). ``discover_ids`` is the
light prevalence pass: ``protein`` reads only the parquet gene list (no embedding load), the non-coding
types fall back to ``records`` because their identity needs the ``.pt``.

The ``None`` vs ``[]`` return of ``discover_ids`` on an unreadable genome is load-bearing — it encodes
the per-type prevalence **denominator** with no extra flag. ``protein`` returns ``[]`` for a
missing-parquet genome (so it still counts toward the denominator, reproducing ``discover_core_genes``'s
``n = len(train_ids)``); the non-coding types return ``None`` (skipped, not counted — reproducing the
collectors' ``n = len(read_ids)``). A single uniform two-pass sweep therefore reproduces both denominator
rules exactly.

The per-type keying / GFF / dedup lives here; the two-pass sweep, prevalence banding, fit and ranking are
type-agnostic (see :mod:`bacpredict.engine.embedding.segment_embedding_extractor` and
:mod:`bacpredict.engine.segment_amr_lr.per_segment_lr`). For now this is a thin *facade* over the record
extractors still housed in the ``gene_lr.build_*`` modules; those bodies move here when the screens are
folded into ``per_segment_lr`` and deleted (migration step 2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

# (sample_id, [(segment_id, vector), ...]) — one genome's located segments, all occurrences.
GenomeRecords = tuple[str, list[tuple[str, np.ndarray]]]


@runtime_checkable
class SegmentLocator(Protocol):
    """One genome → its located segments. A per-type locator, bound to its stores/GFF at construction."""

    def records(self, sample_id: str) -> GenomeRecords | None:
        """``(sample_id, [(segment_id, vector), ...])`` for one genome, or ``None`` if unreadable (skip it)."""
        ...

    def discover_ids(self, sample_id: str) -> list[str] | None:
        """The genome's segment ids for the prevalence pass, or ``None`` if unreadable (skip, uncounted)."""
        ...


def _ids_from_records(recs: GenomeRecords | None) -> list[str] | None:
    """Default ``discover_ids``: the ids of ``records`` (drop the vectors), or ``None`` if unreadable.

    ``None`` propagates the "genome unreadable → skip, do not count toward prevalence" signal, so a
    non-coding locator's denominator is exactly the genomes it could read.
    """
    if recs is None:
        return None
    return [seg_id for seg_id, _ in recs[1]]


@dataclass
class ProteinLocator:
    """Coding segment — a ``gene_name``'s ESM-C (``esm``) or baclm coding vector, in flat parquet order.

    ``discover_ids`` reads only the parquet gene list (no ``.pt`` load) — the cheap prevalence pass the
    coding screen has always used (``discover_core_genes``), so a full cohort is scanned without touching
    the embedding store twice. A missing parquet yields ``[]`` (the genome still counts toward the
    denominator), never ``None``. Bound once; not mutated after construction.
    """

    embed_dir: Path
    parquet_dir: Path
    store_kind: str = "esm"  # esm | baclm

    def records(self, sample_id: str) -> GenomeRecords | None:
        """Per-CDS ``(gene_name, vector)`` in flat order for every named gene, or ``None`` if unreadable.

        Delegates to :func:`build_per_gene_lr_store.read_genome` (the store-kind reader + flat-order
        guard); unnamed CDS rows (no ``gene_name``) are dropped. Every occurrence is emitted — the
        single-copy gate is the sweep's job, not the locator's.
        """
        from bacpredict.engine.gene_lr.build_per_gene_lr_store import read_genome

        read = read_genome(str(sample_id), Path(self.embed_dir), Path(self.parquet_dir), store_kind=self.store_kind)
        if read is None:
            return None
        gene_names, emb = read
        return str(sample_id), [(g, emb[i]) for i, g in enumerate(gene_names) if g]

    def discover_ids(self, sample_id: str) -> list[str]:
        """The genome's named ``gene_name`` list from the parquet only (no embedding load)."""
        from bacpredict.engine.gene_lr.build_per_gene_lr_store import _genome_segment_records

        return [r["gene_name"] for r in _genome_segment_records(str(sample_id), Path(self.parquet_dir)) if r["gene_name"]]


@dataclass
class IgrLocator:
    """Intergenic region — baclm non-coding vector keyed by its sorted flanking-gene pair ``a→b``.

    Needs the sample's Bakta GFF (``sample_gff[sample_id]``) to name each region by its abutting flanks;
    a genome absent from ``sample_gff`` (or with a missing GFF/``.pt``) is unreadable → ``None``.
    """

    baclm_dir: Path
    sample_gff: dict[str, str]
    boundary_tol: int = 3
    baclm_suffix: str = "_baclm_embeddings.pt"

    def records(self, sample_id: str) -> GenomeRecords | None:
        """Every named CDS-flanked region as ``(a→b, vector)`` for one genome (delegates to the extractor)."""
        from bacpredict.engine.gene_lr.build_per_igr_lr_store import _genome_igr_records

        gff = self.sample_gff.get(str(sample_id))
        if not gff:
            return None
        pt = str(Path(self.baclm_dir) / f"{sample_id}{self.baclm_suffix}")
        return _genome_igr_records(str(sample_id), gff, pt, self.boundary_tol)

    def discover_ids(self, sample_id: str) -> list[str] | None:
        """The genome's flank-pair keys (via :meth:`records`), or ``None`` if unreadable (uncounted)."""
        return _ids_from_records(self.records(sample_id))


@dataclass
class UpstreamLocator:
    """Upstream region — the baclm non-coding vector 5′ of each named gene, keyed ``upstream:<gene>``.

    Needs the sample's Bakta GFF. ``include_convergent`` additionally emits ``between:<a>→<b>`` for the
    convergent regions no 5′ anchor claims (the diagnostic whole-region view). Unreadable → ``None``.
    """

    baclm_dir: Path
    sample_gff: dict[str, str]
    boundary_tol: int = 3
    include_convergent: bool = False
    baclm_suffix: str = "_baclm_embeddings.pt"

    def records(self, sample_id: str) -> GenomeRecords | None:
        """Each named gene's 5′-abutting region as ``(upstream:<gene>, vector)`` (delegates to the extractor)."""
        from bacpredict.engine.gene_lr.build_upstream_region_lr_store import _genome_upstream_records

        gff = self.sample_gff.get(str(sample_id))
        if not gff:
            return None
        pt = str(Path(self.baclm_dir) / f"{sample_id}{self.baclm_suffix}")
        return _genome_upstream_records(str(sample_id), gff, pt, self.boundary_tol, self.include_convergent)

    def discover_ids(self, sample_id: str) -> list[str] | None:
        """The genome's ``upstream:<gene>`` keys (via :meth:`records`), or ``None`` if unreadable."""
        return _ids_from_records(self.records(sample_id))


@dataclass
class UnitLocator:
    """Named non-CDS body — a baclm re-embed vector keyed ``<feature_type>:<feature_name>``.

    No GFF: units self-identify from the store's ``feature_*`` keys. The extractor mean-pools a unit's
    several copies (the multiple *rrn* operons of ``rrna:rrs``) into one per-genome row, so the sweep's
    single-copy gate is a no-op here. ``unit_types`` restricts to a subset of the feature vocabulary.
    """

    baclm_dir: Path
    unit_types: frozenset[str] | None = None
    baclm_suffix: str = "_baclm_embeddings.pt"

    def records(self, sample_id: str) -> GenomeRecords | None:
        """Each named body as ``(<type>:<name>, mean_pooled_vector)`` for one genome (delegates to the extractor)."""
        from bacpredict.engine.gene_lr.build_per_unit_lr_store import _genome_unit_records

        pt = str(Path(self.baclm_dir) / f"{sample_id}{self.baclm_suffix}")
        type_filter = set(self.unit_types) if self.unit_types is not None else None
        return _genome_unit_records(str(sample_id), pt, type_filter)

    def discover_ids(self, sample_id: str) -> list[str] | None:
        """The genome's ``<type>:<name>`` unit keys (via :meth:`records`), or ``None`` if unreadable."""
        return _ids_from_records(self.records(sample_id))
