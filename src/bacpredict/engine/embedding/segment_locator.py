"""Locate each AMR segment type in one genome → its per-segment embedding vector(s).

A *segment* is the unit the per-segment LR screen ranks, and there are four types:

- **protein** — a coding gene's ESM-C (``esm``) or baclm coding vector, keyed by ``gene_name``;
- **igr** — an intergenic region, keyed by its sorted flanking-gene pair ``a→b`` (baclm non-coding);
- **upstream** — the region 5′ of a gene, keyed ``upstream:<gene>`` (baclm non-coding);
- **unit** — a named non-CDS body, keyed ``<feature_type>:<feature_name>`` (baclm re-embed).

This module owns both halves of "locate a segment": the low-level per-genome **record extractors** (read a
store / GFF, name each occurrence, dedup) and the per-type **locator classes** that bind an extractor to its
stores/GFF and expose the uniform seam the sweep consumes::

    records(sample_id)      -> (sample_id, [(segment_id, vector), ...]) | None   # None = unreadable → skip
    discover_ids(sample_id) -> list[str] | None                                  # the cheap prevalence pass

``records`` returns **every** occurrence, so the sweep applies the single-copy gate uniformly; ``unit``
already mean-pools its several copies to one row per unit (its relaxed gate). ``discover_ids`` is the
light prevalence pass: ``protein`` reads only the parquet gene list (no embedding load), the non-coding
types fall back to ``records`` because their identity needs the ``.pt``.

The ``None`` vs ``[]`` return of ``discover_ids`` on an unreadable genome is load-bearing — it encodes
the per-type prevalence **denominator** with no extra flag. ``protein`` returns ``[]`` for a
missing-parquet genome (so it still counts toward the denominator, reproducing the coding screen's
``n = len(train_ids)``); the non-coding types return ``None`` (skipped, not counted — the collectors'
``n = len(read_ids)``). A single uniform two-pass sweep therefore reproduces both denominator rules
exactly.

The per-type keying / GFF / dedup lives here; the two-pass sweep, prevalence banding, fit and ranking are
type-agnostic (see :mod:`bacpredict.engine.embedding.segment_embedding_extractor` and
:mod:`bacpredict.engine.segment_amr_lr.per_segment_lr`).

**Lazy heavy imports.** The record extractors read ``.pt`` (torch), parquet (pandas), and GFF, but those
imports are deferred *into* the functions so the module — and the Protocol / locator seam the light sweep
tests exercise — stays importable in a torch-free Stage-A environment. The serial read is also deliberate:
``torch.load(mmap=True)`` cannot be forked before the process-parallel fit (``fit_per_segment``, ``n_jobs``)
on aarch64 (Grace) without segfaulting, so the caller sweeps single-process and fans the fit out afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

# (sample_id, [(segment_id, vector), ...]) — one genome's located segments, all occurrences.
GenomeRecords = tuple[str, list[tuple[str, np.ndarray]]]

# The coding embedding store this locator reads. Both stores share the parquet flat CDS order so the same
# discovery + ranking machinery applies; only the reader differs (see :func:`_embedding_rows`).
EMBEDDING_STORES: dict[str, str] = {
    "esm": "_esm_embeddings.pt",      # ESM-C per-protein store: [1, T, dim] padded/interleaved + mask
    "baclm": "_baclm_embeddings.pt",  # baclm coding store: plain [n_cds, dim] in flat CDS order
}


# ---------------------------------------------------------------------------
# Coding (protein) record extractor — parquet gene list + embedding rows, flat-order aligned.
# ---------------------------------------------------------------------------


def _genome_segment_records(sample_id: str, parquet_dir: Path) -> list[dict]:
    """Flat per-protein records (``gene_name`` + ``protein_name`` …) for one genome (parquet only)."""
    import pandas as pd

    from bacpredict.engine.gene_lr.locate_gene import flatten_proteins

    pq = parquet_dir / f"{sample_id}_protein_sequences.parquet"
    if not pq.exists():
        return []
    return flatten_proteins(pd.read_parquet(pq))


def _embedding_rows(store: dict, store_kind: str, n_genes: int) -> np.ndarray | None:
    """Real-protein embedding rows ``[n_real, dim]`` in flat order, or ``None`` on misalignment.

    ESM-C / Bacformer-input stores pad and interleave CLS/SEP rows, so the real proteins are
    recovered via :func:`real_protein_indices`; the store caps at ``max_n_proteins`` so
    ``n_real <= n_genes``. baclm's ``protein_embeddings`` is a plain ``[n_cds, dim]`` matrix (one
    row per CDS in flat order, no batch dim / mask), so its row count must equal the parquet's CDS
    count exactly — a mismatch is a flat-order break and the genome is skipped.
    """
    from bacpredict.engine.embedding.protein_pooling import real_protein_indices

    if store_kind == "baclm":
        prot = store["protein_embeddings"]
        if prot.dim() == 3:  # defensive: a future writer might keep a leading batch dim
            prot = prot[0]
        if int(prot.shape[0]) != n_genes:
            return None
        return prot.float().numpy()
    prot_emb = store["protein_embeddings"][0]
    real_idx = real_protein_indices(store, prot_emb.shape[0])
    n_real = int(real_idx.numel())
    if n_real > n_genes:
        return None
    return prot_emb[real_idx].float().numpy()


def read_genome(
    sample_id: str, embed_dir: Path, parquet_dir: Path, *, store_kind: str = "esm"
) -> tuple[list[str | None], np.ndarray] | None:
    """Return ``(gene_names[:n_real], embedding[n_real, dim])`` aligned in flat order, or ``None``.

    ``store_kind`` selects the embedding store (``esm`` or ``baclm``); both index the same parquet
    flat protein order, so the gene list is truncated to the embedding's real-protein count. Returns
    ``None`` (skip) when a file is missing or the embedding fails its flat-order guard (ESM: more real
    proteins than the parquet annotates; baclm: a CDS-count mismatch).
    """
    import pandas as pd
    import torch

    from bacpredict.engine.gene_lr.locate_gene import flatten_proteins

    pq = parquet_dir / f"{sample_id}_protein_sequences.parquet"
    pt = embed_dir / f"{sample_id}{EMBEDDING_STORES[store_kind]}"
    if not pq.exists() or not pt.exists():
        return None
    gene_names = [r["gene_name"] for r in flatten_proteins(pd.read_parquet(pq))]

    store = torch.load(pt, map_location="cpu", mmap=True)
    emb = _embedding_rows(store, store_kind, len(gene_names))
    if emb is None:
        return None
    return gene_names[: emb.shape[0]], emb


# ---------------------------------------------------------------------------
# IGR record extractor — baclm intergenic rows named by their sorted flanking-gene pair.
# ---------------------------------------------------------------------------


def _genes_by_seqid(gff_path: Path) -> dict[str, list[tuple[int, int, str]]]:
    """``seqid → sorted [(start, end, gene_name_lower)]`` for every ``gene=`` CDS in the GFF.

    Reuses :func:`igr_amr_lr._parse_gff` (which keys named genes by symbol) and inverts it to a
    per-contig coordinate-sorted list — the substrate for locating each IGR's abutting flanks.
    """
    from bacpredict.engine.gene_lr.igr_amr_lr import _parse_gff

    _feats, genes = _parse_gff(gff_path)
    by_seqid: dict[str, list[tuple[int, int, str]]] = {}
    for gname, hits in genes.items():
        for seqid, start, end, _strand in hits:
            by_seqid.setdefault(seqid, []).append((int(start), int(end), gname))
    for seqid in by_seqid:
        by_seqid[seqid].sort()
    return by_seqid


def _read_intergenic(pt_path: Path) -> tuple[np.ndarray, list[str], list[int], list[int]] | None:
    """Load one baclm store's intergenic rows: ``(emb[n, dim], seqids, starts, ends)`` or ``None``.

    Prefers the 2d-re-embed ``noncoding_*`` keys, falls back to the legacy ``intergenic_*`` keys, so
    this reads both the old and re-embedded stores (same fallback as :mod:`igr_amr_lr`).
    """
    import torch

    if not pt_path.exists():
        return None
    store = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=True)
    if "noncoding_embeddings" in store:
        emb, seqid, start, end = (
            store["noncoding_embeddings"], store["noncoding_seqid"], store["noncoding_start"], store["noncoding_end"],
        )
    elif "intergenic_embeddings" in store:
        emb, seqid, start, end = (
            store["intergenic_embeddings"], store["intergenic_seqid"], store["intergenic_start"], store["intergenic_end"],
        )
    else:
        return None
    return emb.float().numpy(), list(seqid), [int(s) for s in start], [int(e) for e in end]


def _flank_pair(
    genes_here: list[tuple[int, int, str]], igr_start: int, igr_end: int, *, boundary_tol: int
) -> tuple[str, str] | None:
    """Canonical (orientation-invariant) flanking-gene pair for one IGR, or ``None`` if a flank is unnamed.

    Finds the two named genes directly abutting the region (``end`` closest below ``igr_start`` and ``start``
    closest above ``igr_end``, each within ``boundary_tol``) and returns them **sorted** — ``(min, max)`` by
    name. Abutment within a few bp enforces the "consistently-named flank" rule: if the immediately-adjacent
    CDS is unnamed (locus-tag only) the nearest *named* gene lies beyond it, its gap exceeds the tolerance,
    and the region is dropped.

    **Why sorted, not 5′→3′:** a contig is stored in an arbitrary orientation (≡ its reverse complement), so
    the *same physical* region flanked by two convergent genes appears as ``a→b`` in some genomes and ``b→a``
    in others — which split one region into two half-prevalence keys (the *rrn*/``rrs`` operon between ``murA``
    and ``ogt`` was ``ogt→mura`` + ``mura→ogt`` at ~50% each). Sorting collapses those into one key. The
    single-copy gate bounds the mis-merge risk (a pair flanking two distinct regions in one genome is dropped);
    the ``→`` now denotes adjacency, not 5′→3′ direction (finer strand/operon naming is the sister project's).
    """
    left, left_gap = None, boundary_tol + 1
    right, right_gap = None, boundary_tol + 1
    for start, end, gname in genes_here:
        if end < igr_start:
            gap = igr_start - 1 - end
            if gap <= boundary_tol and gap < left_gap:
                left, left_gap = gname, gap
        if start > igr_end:
            gap = start - (igr_end + 1)
            if gap <= boundary_tol and gap < right_gap:
                right, right_gap = gname, gap
    if left is None or right is None:
        return None
    lo, hi = sorted((left, right))
    return lo, hi


def _genome_igr_records(
    sid: str, gff_path: str, pt_path: str, boundary_tol: int = 3
) -> tuple[str, list[tuple[str, np.ndarray]]] | None:
    """One genome's ``[(igr_pair, embedding)]`` — the named, CDS-flanked baclm intergenic regions.

    ``igr_pair`` is ``"left→right"`` (ascending genome coord). Returns ``None`` when the GFF or ``.pt``
    is missing/unreadable (the genome is skipped, not imputed).
    """
    gpath, ppath = Path(gff_path), Path(pt_path)
    if not gpath.exists() or not ppath.exists():
        return None
    try:
        by_seqid = _genes_by_seqid(gpath)
    except (OSError, ValueError):
        return None
    read = _read_intergenic(ppath)
    if read is None:
        return None
    emb, seqids, starts, ends = read
    records: list[tuple[str, np.ndarray]] = []
    for i, (sq, s, e) in enumerate(zip(seqids, starts, ends, strict=True)):
        pair = _flank_pair(by_seqid.get(sq, []), s, e, boundary_tol=boundary_tol)
        if pair is None:
            continue
        records.append((f"{pair[0]}→{pair[1]}", emb[i]))
    return sid, records


# ---------------------------------------------------------------------------
# Upstream record extractor — the baclm region 5′ of each named gene (reuses the IGR readers).
# ---------------------------------------------------------------------------


def _upstream_region_index(
    gstart: int, gend: int, strand: str, rows: list[tuple[int, int, int]], *, boundary_tol: int
) -> int | None:
    """Index of the non-coding region abutting a gene's 5′ end, or ``None`` if none within tolerance.

    ``rows`` is the genome's ``[(start, end, row_idx)]`` on the gene's contig. On ``-`` strand the 5′ end
    is the high coordinate (``gend``) and the upstream region's ``start`` abuts ``gend+1``; on ``+`` strand
    the 5′ end is ``gstart`` and the region's ``end`` abuts ``gstart-1``. Ties break to the nearest region.
    """
    best_idx, best_gap = None, boundary_tol + 1
    if strand == "-":
        for s, _e, i in rows:
            gap = s - (gend + 1)  # region sits above the gene
            if -boundary_tol <= gap <= boundary_tol and abs(gap) < best_gap:
                best_idx, best_gap = i, abs(gap)
    else:  # '+' (and default) — upstream is below the gene start
        for _s, e, i in rows:
            gap = (gstart - 1) - e
            if -boundary_tol <= gap <= boundary_tol and abs(gap) < best_gap:
                best_idx, best_gap = i, abs(gap)
    return best_idx


def _genome_upstream_records(
    sid: str, gff_path: str, pt_path: str, boundary_tol: int = 3, include_convergent: bool = False
) -> tuple[str, list[tuple[str, np.ndarray]]] | None:
    """One genome's ``[(key, embedding)]`` — each named gene's 5′-abutting baclm region as ``upstream:<gene>``.

    With ``include_convergent`` the whole screen is completed by a **flank-pair fallback**: every non-coding
    region NOT claimed by any ``upstream:<gene>`` key — a region between two *convergent* genes (both abutting
    it with their 3′ ends) has no 5′ anchor — but with both flanks consistently named is emitted as
    ``between:<left>→<right>`` (the flank-pair key of :func:`_genome_igr_records`). This is what surfaces
    convergent regions like the *rrn*/``rrs`` operon (flanked by ``murA`` and ``ogt``) that ``upstream:<gene>``
    structurally omits; the ladder's input dir is generated without it, so the fallback only feeds the
    diagnostic ``per_igr_whole`` plot.
    """
    from bacpredict.engine.gene_lr.igr_amr_lr import _parse_gff

    gpath, ppath = Path(gff_path), Path(pt_path)
    if not gpath.exists() or not ppath.exists():
        return None
    try:
        _feats, genes = _parse_gff(gpath)
    except (OSError, ValueError):
        return None
    read = _read_intergenic(ppath)
    if read is None:
        return None
    emb, seqids, starts, ends = read
    rows_by_seqid: dict[str, list[tuple[int, int, int]]] = {}
    for i, (sq, s, e) in enumerate(zip(seqids, starts, ends, strict=True)):
        rows_by_seqid.setdefault(sq, []).append((s, e, i))

    records: list[tuple[str, np.ndarray]] = []
    claimed: set[int] = set()
    for gname, hits in genes.items():
        for seqid, gstart, gend, strand in hits:
            rows = rows_by_seqid.get(seqid)
            if not rows:
                continue
            idx = _upstream_region_index(gstart, gend, strand, rows, boundary_tol=boundary_tol)
            if idx is not None:
                records.append((f"upstream:{gname}", emb[idx]))
                claimed.add(idx)

    if include_convergent:
        # Invert the parsed genes to a per-contig coordinate-sorted list, then name each region with no 5′
        # anchor (unclaimed above) by its abutting flank pair — reusing :func:`_flank_pair`.
        by_seqid: dict[str, list[tuple[int, int, str]]] = {}
        for gname, hits in genes.items():
            for seqid, gstart, gend, _strand in hits:
                by_seqid.setdefault(seqid, []).append((int(gstart), int(gend), gname))
        for regions in by_seqid.values():
            regions.sort()
        for i, (sq, s, e) in enumerate(zip(seqids, starts, ends, strict=True)):
            if i in claimed:
                continue
            pair = _flank_pair(by_seqid.get(sq, []), s, e, boundary_tol=boundary_tol)
            if pair is not None:
                records.append((f"between:{pair[0]}→{pair[1]}", emb[i]))
    return sid, records


# ---------------------------------------------------------------------------
# Unit record extractor — named non-CDS bodies from the baclm re-embed ``feature_*`` channel.
# ---------------------------------------------------------------------------


def _read_features(pt_path: Path) -> tuple[np.ndarray, list[str], list[str]] | None:
    """Load one re-embed store's named-body rows: ``(emb[n, dim], feature_types, feature_names)`` or ``None``.

    ``None`` means the genome is unreadable *as a feature source* — the ``.pt`` is missing, has no
    ``feature_embeddings`` key (the legacy ``baclm/`` store), or its parallel type/name lists are
    length-mismatched (a schema break). A readable store with **zero** feature rows returns empty arrays,
    not ``None``: the genome is a valid, feature-less member of the read universe (a genuine absence for
    the zero-impute fit), distinct from a genome we could not read at all.
    """
    import torch

    if not pt_path.exists():
        return None
    store = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=True)
    if "feature_embeddings" not in store:
        return None
    emb = store["feature_embeddings"]
    n = int(emb.shape[0]) if emb is not None else 0
    ftypes = [str(t) for t in store.get("feature_type", [])]
    fnames = [str(nm) for nm in store.get("feature_name", [])]
    if len(ftypes) != n or len(fnames) != n:
        return None
    if n == 0:
        return np.zeros((0, 1), dtype=np.float32), [], []
    return emb.float().numpy(), ftypes, fnames


def _unit_key(ftype: str, fname: str) -> str:
    """``<feature_type>:<feature_name>`` — type lower-cased, name stripped (``unnamed`` if blank)."""
    return f"{ftype.strip().lower()}:{fname.strip() or 'unnamed'}"


def _genome_unit_records(
    sid: str, pt_path: str, type_filter: set[str] | None = None
) -> tuple[str, list[tuple[str, np.ndarray]]] | None:
    """One genome's ``[(unit_key, mean_pooled_embedding)]`` — one row per unit, copies mean-pooled.

    Multi-copy bodies (the several *rrn* copies of ``rrna:rrs``) are averaged into a single per-genome
    vector, so a genome contributes at most one row per unit. ``type_filter`` (lower-cased feature types)
    restricts to a subset of the vocabulary (e.g. ``{"rrna"}``); ``None`` keeps every named body.
    """
    read = _read_features(Path(pt_path))
    if read is None:
        return None
    emb, ftypes, fnames = read
    by_key: dict[str, list[np.ndarray]] = {}
    for i, (ftype, fname) in enumerate(zip(ftypes, fnames, strict=True)):
        if type_filter is not None and ftype.strip().lower() not in type_filter:
            continue
        by_key.setdefault(_unit_key(ftype, fname), []).append(emb[i])
    records = [(k, np.mean(np.vstack(v), axis=0).astype(np.float32)) for k, v in by_key.items()]
    return sid, records


# ---------------------------------------------------------------------------
# The uniform locator seam — one per-type class binding an extractor to its stores/GFF.
# ---------------------------------------------------------------------------


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
    coding screen has always used, so a full cohort is scanned without touching the embedding store
    twice. A missing parquet yields ``[]`` (the genome still counts toward the denominator), never
    ``None``. Bound once; not mutated after construction.
    """

    embed_dir: Path
    parquet_dir: Path
    store_kind: str = "esm"  # esm | baclm

    def records(self, sample_id: str) -> GenomeRecords | None:
        """Per-CDS ``(gene_name, vector)`` in flat order for every named gene, or ``None`` if unreadable.

        Delegates to :func:`read_genome` (the store-kind reader + flat-order guard); unnamed CDS rows
        (no ``gene_name``) are dropped. Every occurrence is emitted — the single-copy gate is the sweep's
        job, not the locator's.
        """
        read = read_genome(str(sample_id), Path(self.embed_dir), Path(self.parquet_dir), store_kind=self.store_kind)
        if read is None:
            return None
        gene_names, emb = read
        return str(sample_id), [(g, emb[i]) for i, g in enumerate(gene_names) if g]

    def discover_ids(self, sample_id: str) -> list[str]:
        """The genome's named ``gene_name`` list from the parquet only (no embedding load)."""
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
        """Every named CDS-flanked region as ``(a→b, vector)`` for one genome (via :func:`_genome_igr_records`)."""
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
        """Each named gene's 5′-abutting region as ``(upstream:<gene>, vector)`` (via :func:`_genome_upstream_records`)."""
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
        """Each named body as ``(<type>:<name>, mean_pooled_vector)`` for one genome (via :func:`_genome_unit_records`)."""
        pt = str(Path(self.baclm_dir) / f"{sample_id}{self.baclm_suffix}")
        type_filter = set(self.unit_types) if self.unit_types is not None else None
        return _genome_unit_records(str(sample_id), pt, type_filter)

    def discover_ids(self, sample_id: str) -> list[str] | None:
        """The genome's ``<type>:<name>`` unit keys (via :meth:`records`), or ``None`` if unreadable."""
        return _ids_from_records(self.records(sample_id))
