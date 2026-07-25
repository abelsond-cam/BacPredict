"""Unit tests for the uniform segment-locator seam (``…embedding.segment_locator``).

Each per-type locator is a thin facade over its record extractor; these assert it delegates with the
right arguments, drops unnamed coding rows, and — critically — returns the ``None`` vs ``[]``
``discover_ids`` signal that encodes the per-type prevalence denominator (``protein`` counts a
missing-parquet genome; the non-coding types skip an unreadable one). Synthetic delegates via
monkeypatch — no HPC data / ``.pt`` / GFF on disk. Skipped where numpy is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import bacpredict.engine.gene_lr.build_per_gene_lr_store as bpg
import bacpredict.engine.gene_lr.build_per_igr_lr_store as bigr
import bacpredict.engine.gene_lr.build_per_unit_lr_store as bpu
import bacpredict.engine.gene_lr.build_upstream_region_lr_store as bur
from bacpredict.engine.embedding.segment_locator import (
    IgrLocator,
    ProteinLocator,
    SegmentLocator,
    UnitLocator,
    UpstreamLocator,
    _ids_from_records,
)

DIM = 4


# ---------------------------------------------------------------------------
# _ids_from_records — the None-propagating default discover_ids
# ---------------------------------------------------------------------------


def test_ids_from_records_none_propagates() -> None:
    """An unreadable genome (``records`` → ``None``) yields ``None`` (skip, uncounted), not ``[]``."""
    assert _ids_from_records(None) is None
    assert _ids_from_records(("G", [("a→b", np.ones(DIM)), ("c→d", np.ones(DIM))])) == ["a→b", "c→d"]


# ---------------------------------------------------------------------------
# ProteinLocator — reads the store-kind reader; parquet-only discovery
# ---------------------------------------------------------------------------


def test_protein_locator_records_drops_unnamed_and_keeps_flat_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """``records`` emits ``(gene, vector)`` for every named CDS in flat order; unnamed rows are dropped."""
    emb = np.arange(3 * DIM, dtype=float).reshape(3, DIM)
    monkeypatch.setattr(bpg, "read_genome", lambda sid, e, p, *, store_kind="esm": (["rpoB", None, "katG"], emb))

    loc = ProteinLocator(embed_dir=Path("emb"), parquet_dir=Path("pq"), store_kind="baclm")
    assert isinstance(loc, SegmentLocator)  # satisfies the runtime-checkable Protocol
    sid, recs = loc.records("G")
    assert sid == "G"
    assert [g for g, _ in recs] == ["rpoB", "katG"]  # the None (unnamed) row is dropped
    np.testing.assert_array_equal(recs[0][1], emb[0])
    np.testing.assert_array_equal(recs[1][1], emb[2])  # flat index preserved past the dropped row


def test_protein_locator_records_none_when_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing/misaligned store (``read_genome`` → ``None``) makes ``records`` return ``None`` (skip)."""
    monkeypatch.setattr(bpg, "read_genome", lambda *a, **k: None)
    assert ProteinLocator(embed_dir=Path("e"), parquet_dir=Path("p")).records("G") is None


def test_protein_locator_discover_ids_is_parquet_only_and_never_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """``discover_ids`` reads the parquet gene list (no ``.pt``); a missing parquet yields ``[]``, not ``None``.

    The ``[]`` (never ``None``) is the load-bearing bit: a missing-parquet genome still counts toward the
    coding prevalence denominator (``discover_core_genes``'s ``n = len(train_ids)``).
    """
    monkeypatch.setattr(bpg, "_genome_segment_records",
                        lambda sid, pq: [{"gene_name": "rpoB"}, {"gene_name": None}, {"gene_name": "katG"}])
    loc = ProteinLocator(embed_dir=Path("e"), parquet_dir=Path("p"))
    assert loc.discover_ids("G") == ["rpoB", "katG"]  # unnamed dropped, order kept

    monkeypatch.setattr(bpg, "_genome_segment_records", lambda sid, pq: [])  # missing parquet
    assert loc.discover_ids("G") == []  # counts toward the denominator, never None


# ---------------------------------------------------------------------------
# IgrLocator — flank-pair extractor, GFF-gated
# ---------------------------------------------------------------------------


def test_igr_locator_delegates_with_pt_path_and_tol(monkeypatch: pytest.MonkeyPatch) -> None:
    """``records`` calls ``_genome_igr_records`` with the assembled ``.pt`` path + boundary tolerance."""
    seen: dict[str, object] = {}

    def fake(sid, gff, pt, boundary_tol):
        seen.update(sid=sid, gff=gff, pt=pt, tol=boundary_tol)
        return sid, [("gyra→gyrb", np.ones(DIM))]

    monkeypatch.setattr(bigr, "_genome_igr_records", fake)
    loc = IgrLocator(baclm_dir=Path("/b"), sample_gff={"G": "/gff/G.gff"}, boundary_tol=5)
    sid, recs = loc.records("G")
    assert sid == "G" and [k for k, _ in recs] == ["gyra→gyrb"]
    assert seen == {"sid": "G", "gff": "/gff/G.gff", "pt": str(Path("/b/G_baclm_embeddings.pt")), "tol": 5}
    assert loc.discover_ids("G") == ["gyra→gyrb"]


def test_igr_locator_none_when_gff_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genome absent from ``sample_gff`` is unreadable → ``records``/``discover_ids`` both ``None`` (skip)."""
    monkeypatch.setattr(bigr, "_genome_igr_records", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called")))
    loc = IgrLocator(baclm_dir=Path("/b"), sample_gff={})
    assert loc.records("G") is None
    assert loc.discover_ids("G") is None  # None (uncounted), not [] — the extractor is never called


# ---------------------------------------------------------------------------
# UpstreamLocator — 5′-anchor extractor, include_convergent passthrough
# ---------------------------------------------------------------------------


def test_upstream_locator_passes_include_convergent(monkeypatch: pytest.MonkeyPatch) -> None:
    """``include_convergent`` is threaded through to ``_genome_upstream_records``."""
    seen: dict[str, object] = {}

    def fake(sid, gff, pt, boundary_tol, include_convergent):
        seen.update(tol=boundary_tol, conv=include_convergent)
        return sid, [("upstream:katg", np.ones(DIM))]

    monkeypatch.setattr(bur, "_genome_upstream_records", fake)
    loc = UpstreamLocator(baclm_dir=Path("/b"), sample_gff={"G": "/g.gff"}, boundary_tol=2, include_convergent=True)
    _sid, recs = loc.records("G")
    assert [k for k, _ in recs] == ["upstream:katg"]
    assert seen == {"tol": 2, "conv": True}


def test_upstream_locator_none_when_gff_missing() -> None:
    """No GFF entry → ``None`` (skip)."""
    loc = UpstreamLocator(baclm_dir=Path("/b"), sample_gff={})
    assert loc.records("G") is None and loc.discover_ids("G") is None


# ---------------------------------------------------------------------------
# UnitLocator — named-body extractor, no GFF, type filter
# ---------------------------------------------------------------------------


def test_unit_locator_delegates_and_passes_type_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    """``records`` calls ``_genome_unit_records`` with the ``.pt`` path + a materialised type filter set."""
    seen: dict[str, object] = {}

    def fake(sid, pt, type_filter):
        seen.update(pt=pt, tf=type_filter)
        return sid, [("rrna:rrs", np.ones(DIM))]

    monkeypatch.setattr(bpu, "_genome_unit_records", fake)
    loc = UnitLocator(baclm_dir=Path("/b"), unit_types=frozenset({"rrna"}))
    _sid, recs = loc.records("G")
    assert [k for k, _ in recs] == ["rrna:rrs"]
    assert seen["pt"] == str(Path("/b/G_baclm_embeddings.pt"))
    assert seen["tf"] == {"rrna"} and isinstance(seen["tf"], set)


def test_unit_locator_none_type_filter_stays_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``unit_types`` → ``type_filter=None`` (every named body kept); unreadable store → ``None``."""
    monkeypatch.setattr(bpu, "_genome_unit_records", lambda sid, pt, tf: None if tf is None else (sid, []))
    loc = UnitLocator(baclm_dir=Path("/b"))
    assert loc.records("G") is None  # our fake returns None for tf=None — confirms None is forwarded
    assert loc.discover_ids("G") is None
