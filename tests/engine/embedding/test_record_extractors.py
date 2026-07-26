"""Unit tests for the per-genome record extractors in ``…embedding.segment_locator``.

These are the low-level readers each locator delegates to — the ordered flank-pair naming + abutment
tolerance (``_flank_pair``), the 5′-anchor index on both strands (``_upstream_region_index``), the
convergent flank-pair fallback (``_genome_upstream_records`` with ``include_convergent``), the re-embed
feature reader (``_read_features``), the multi-copy mean-pool + type filter (``_genome_unit_records``),
and the coding store-kind reader's flat-order guard (``_embedding_rows``). Relocated here from the four
retired ``gene_lr.build_*`` stores; the locator *classes* wrapping these are tested in
:mod:`tests.engine.embedding.test_segment_locator`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import bacpredict.engine.embedding.segment_locator as sl

DIM = 6


# ---------------------------------------------------------------------------
# _flank_pair — ordered pair naming + abutment tolerance
# ---------------------------------------------------------------------------


def test_flank_pair_names_abutting_neighbours_in_canonical_order() -> None:
    """Both flanks directly abut the region → the two names sorted (min, max)."""
    genes = [(1, 100, "gyra"), (151, 250, "gyrb")]  # sorted (start, end, name)
    assert sl._flank_pair(genes, igr_start=101, igr_end=150, boundary_tol=3) == ("gyra", "gyrb")


def test_flank_pair_canonicalizes_reversed_orientation() -> None:
    """The SAME convergent region on the opposite contig orientation collapses to ONE key, not two.

    ``ogt`` and ``mura`` flank the *rrn*/``rrs`` operon; whichever sits at the low coordinate depends on the
    arbitrary contig orientation. Sorting the pair merges the two half-prevalence keys into ``mura→ogt``.
    """
    fwd = [(1, 100, "ogt"), (151, 250, "mura")]   # ogt at the low coordinate in this genome
    rev = [(1, 100, "mura"), (151, 250, "ogt")]   # opposite orientation — mura at the low coordinate
    assert sl._flank_pair(fwd, 101, 150, boundary_tol=3) == ("mura", "ogt")
    assert sl._flank_pair(rev, 101, 150, boundary_tol=3) == ("mura", "ogt")


def test_flank_pair_drops_region_with_far_flank() -> None:
    """A flank beyond the tolerance (unnamed CDS / RNA in between) leaves that side unnamed → None."""
    genes = [(1, 100, "gyra"), (200, 300, "gyrb")]  # right flank starts 49 bp away
    assert sl._flank_pair(genes, igr_start=101, igr_end=150, boundary_tol=3) is None


def test_flank_pair_picks_nearest_named_flank() -> None:
    """When several named genes sit on one side, the directly-abutting (nearest) one is chosen."""
    genes = [(1, 50, "far"), (80, 100, "near"), (151, 250, "right")]
    assert sl._flank_pair(genes, igr_start=101, igr_end=150, boundary_tol=3) == ("near", "right")


def test_flank_pair_none_when_a_side_is_empty() -> None:
    """A contig-end region (no gene on one side) is unnamed and dropped."""
    genes = [(1, 100, "gyra")]  # nothing to the right
    assert sl._flank_pair(genes, igr_start=101, igr_end=150, boundary_tol=3) is None


# ---------------------------------------------------------------------------
# _upstream_region_index — 5′-anchoring on both strands + abutment tolerance
# ---------------------------------------------------------------------------


def test_upstream_region_index_minus_strand_picks_region_above_gene_end() -> None:
    """On ``-`` strand the 5′ end is the high coordinate; the region abutting ``gend+1`` is chosen."""
    rows = [(201, 260, 0), (400, 460, 1)]  # (start, end, row_idx); first abuts gend=200
    assert sl._upstream_region_index(100, 200, "-", rows, boundary_tol=3) == 0


def test_upstream_region_index_plus_strand_picks_region_below_gene_start() -> None:
    """On ``+`` strand the 5′ end is ``gstart``; the region abutting ``gstart-1`` is chosen."""
    rows = [(40, 99, 0), (200, 260, 1)]  # first ends at gstart-1 = 99
    assert sl._upstream_region_index(100, 180, "+", rows, boundary_tol=3) == 0


def test_upstream_region_index_none_beyond_tolerance() -> None:
    """A region whose gap exceeds the tolerance (an unnamed CDS/RNA in between) is not anchored."""
    rows = [(210, 260, 0)]  # 9 bp above gend+1 (=201), beyond tol 3
    assert sl._upstream_region_index(100, 200, "-", rows, boundary_tol=3) is None


# ---------------------------------------------------------------------------
# _genome_upstream_records — convergent flank-pair fallback (include_convergent)
# ---------------------------------------------------------------------------


def test_genome_upstream_records_convergent_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A region between two convergent genes has no 5′ anchor; the fallback names it by its flank pair.

    Synthetic contig ``c1``: ``mura`` (+, 100-200) and ``ogt`` (−, 300-400) both abut the region 201-299
    with their **3′** ends (convergent) — like the rrn/rrs operon — so no ``upstream:<gene>`` key claims it.
    ``rpob`` (+, 500-600) has a clean 5′ region 450-499. With the flag off only the two 5′-anchored keys
    appear; with it on the convergent region is emitted as ``between:mura→ogt``. (``_parse_gff`` lives in the
    heavy ``igr_amr_lr`` module for now — imported lazily; step 4 moves it into a light GFF util.)
    """
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    gff, pt = tmp_path / "g.gff", tmp_path / "g.pt"
    gff.write_text("x")
    pt.write_text("x")
    genes = {"mura": [("c1", 100, 200, "+")], "ogt": [("c1", 300, 400, "-")], "rpob": [("c1", 500, 600, "+")]}
    seqids, starts, ends = ["c1", "c1", "c1"], [1, 201, 450], [99, 299, 499]
    emb = np.arange(3 * DIM, dtype=float).reshape(3, DIM)
    monkeypatch.setattr("bacpredict.engine.gene_lr.igr_amr_lr._parse_gff", lambda p: ({}, genes))
    monkeypatch.setattr(sl, "_read_intergenic", lambda p: (emb, seqids, starts, ends))

    _sid, recs = sl._genome_upstream_records("G", str(gff), str(pt))
    assert {k for k, _ in recs} == {"upstream:mura", "upstream:rpob"}  # default: 5′ anchors only

    _sid, recs2 = sl._genome_upstream_records("G", str(gff), str(pt), include_convergent=True)
    by_key = dict(recs2)
    assert set(by_key) == {"upstream:mura", "upstream:rpob", "between:mura→ogt"}
    np.testing.assert_array_equal(by_key["between:mura→ogt"], emb[1])  # the 201-299 region's embedding


# ---------------------------------------------------------------------------
# _read_features / _genome_unit_records — the re-embed feature_* reader + mean-pool
# ---------------------------------------------------------------------------


def _save_genome(baclm_dir: Path, sid: str, rows: list[tuple[str, str, np.ndarray]]) -> None:
    """Write one ``{sid}_baclm_embeddings.pt`` with the re-embed ``feature_*`` schema from ``(type,name,vec)`` rows."""
    import torch

    baclm_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        emb = torch.tensor(np.vstack([v for _, _, v in rows]), dtype=torch.float32)
        ftypes = [t for t, _, _ in rows]
        fnames = [nm for _, nm, _ in rows]
    else:
        emb = torch.zeros((0, DIM), dtype=torch.float32)
        ftypes, fnames = [], []
    torch.save(
        {"feature_embeddings": emb, "feature_type": ftypes, "feature_name": fnames,
         "feature_seqid": ["c1"] * len(rows), "feature_start": [1] * len(rows), "feature_end": [2] * len(rows)},
        baclm_dir / f"{sid}_baclm_embeddings.pt",
    )


def test_read_features_reads_bodies(tmp_path: Path) -> None:
    """A store with named bodies reads back as ``(emb[n,dim], types, names)`` in row order."""
    pytest.importorskip("torch")
    _save_genome(tmp_path, "G", [("rrna", "rrs", np.ones(DIM)), ("trna", "ala", np.zeros(DIM))])
    read = sl._read_features(tmp_path / "G_baclm_embeddings.pt")
    assert read is not None
    emb, ftypes, fnames = read
    assert emb.shape == (2, DIM) and ftypes == ["rrna", "trna"] and fnames == ["rrs", "ala"]


def test_read_features_none_for_missing_or_legacy_or_mismatch(tmp_path: Path) -> None:
    """Missing file, a legacy store without ``feature_embeddings``, or a length mismatch all read as ``None``."""
    torch = pytest.importorskip("torch")
    assert sl._read_features(tmp_path / "absent_baclm_embeddings.pt") is None
    torch.save({"protein_embeddings": torch.zeros((3, DIM))}, tmp_path / "legacy_baclm_embeddings.pt")
    assert sl._read_features(tmp_path / "legacy_baclm_embeddings.pt") is None
    torch.save({"feature_embeddings": torch.zeros((2, DIM)), "feature_type": ["rrna"], "feature_name": ["rrs"]},
               tmp_path / "bad_baclm_embeddings.pt")
    assert sl._read_features(tmp_path / "bad_baclm_embeddings.pt") is None


def test_read_features_empty_is_not_none(tmp_path: Path) -> None:
    """A readable but feature-less genome returns empty arrays (a valid absence), not ``None`` (unreadable)."""
    pytest.importorskip("torch")
    _save_genome(tmp_path, "E", [])
    read = sl._read_features(tmp_path / "E_baclm_embeddings.pt")
    assert read is not None and read[0].shape[0] == 0 and read[1] == [] and read[2] == []


def test_genome_unit_records_mean_pools_copies(tmp_path: Path) -> None:
    """Multiple copies of one unit (the multi-copy rrn operons) collapse to one mean-pooled per-genome row."""
    pytest.importorskip("torch")
    a, b = np.full(DIM, 2.0), np.full(DIM, 4.0)  # two rrs copies → mean 3.0
    _save_genome(tmp_path, "G", [("rrna", "rrs", a), ("rrna", "rrs", b), ("trna", "ala", np.ones(DIM))])
    sid, records = sl._genome_unit_records("G", str(tmp_path / "G_baclm_embeddings.pt"))
    recs = dict(records)
    assert sid == "G" and set(recs) == {"rrna:rrs", "trna:ala"}
    assert np.allclose(recs["rrna:rrs"], 3.0)  # (2+4)/2


def test_genome_unit_records_type_filter(tmp_path: Path) -> None:
    """``type_filter`` keeps only the requested feature types (e.g. rRNA), dropping the rest."""
    pytest.importorskip("torch")
    _save_genome(tmp_path, "G", [("rrna", "rrs", np.ones(DIM)), ("crispr", "x", np.ones(DIM))])
    _sid, records = sl._genome_unit_records("G", str(tmp_path / "G_baclm_embeddings.pt"), type_filter={"rrna"})
    assert [k for k, _ in records] == ["rrna:rrs"]


# ---------------------------------------------------------------------------
# _embedding_rows — the coding store-kind reader's flat-order guard
# ---------------------------------------------------------------------------


def test_embedding_rows_baclm_direct_and_count_guard() -> None:
    """baclm store: plain [n_cds, dim] read directly; a CDS-count mismatch is skipped (None)."""
    torch = pytest.importorskip("torch")
    prot = torch.arange(12, dtype=torch.float32).reshape(4, 3)  # [n_cds=4, dim=3]
    store = {"protein_embeddings": prot}

    rows = sl._embedding_rows(store, "baclm", n_genes=4)
    assert rows is not None and rows.shape == (4, 3)
    np.testing.assert_allclose(rows, prot.numpy())

    assert sl._embedding_rows(store, "baclm", n_genes=5) is None  # count mismatch -> skip
    # A defensive leading batch dim is squeezed, not misread as n_cds=1.
    batched = {"protein_embeddings": prot[None]}  # [1, 4, 3]
    rows_b = sl._embedding_rows(batched, "baclm", n_genes=4)
    assert rows_b is not None and rows_b.shape == (4, 3)


def test_embedding_rows_esm_selects_real_proteins() -> None:
    """ESM store: real-protein rows are selected by attention_mask; n_real may be < n_genes (capped)."""
    torch = pytest.importorskip("torch")
    # 3 real proteins + 1 padding row, plain per-protein layout (attention_mask marks real rows).
    prot = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    store = {"protein_embeddings": prot, "attention_mask": torch.tensor([[1, 1, 1, 0]])}
    rows = sl._embedding_rows(store, "esm", n_genes=5)
    assert rows is not None and rows.shape == (3, 3)  # only the 3 real proteins
    np.testing.assert_allclose(rows, prot[0, :3].numpy())
    assert sl._embedding_rows(store, "esm", n_genes=2) is None  # more real proteins than genes -> skip
