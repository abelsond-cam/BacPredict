"""Unit tests for the uniform two-pass segment sweep (``…embedding.segment_embedding_extractor``).

Exercise the behaviours the four copy-forked collectors encoded, now in one function: the single-copy
gate, the per-type prevalence DENOMINATOR (protein counts an unreadable-parquet genome via ``discover_ids
→ []``; the non-coding types skip it via ``→ None``), the ``(min, max]`` band, core-only pass-2 (memory
safety), and the held-out-eval semantics (excluded from pass-1 selection, scored in pass-2, present in the
``read_ids`` impute universe — the correctness-spine fix). Synthetic locators; no HPC data.
"""

from __future__ import annotations

import numpy as np
import pytest

from bacpredict.engine.embedding.segment_embedding_extractor import (
    collect_core_subset,
    collect_segment_matrices,
    sweep_core_prevalence,
)
from bacpredict.engine.embedding.segment_locator import SegmentLocator

DIM = 4
V = np.ones(DIM, dtype=float)


class FakeLocator:
    """Test double with decoupled ``records`` and ``discover_ids`` maps.

    ``records_map[sid]`` is a list of ``(segment_id, vector)`` (or ``None`` for an unreadable genome);
    ``discover_map[sid]`` is the pass-1 id list (``[]`` for the protein "counts but empty" case, ``None``
    for the non-coding "unreadable → skip" case). Missing keys default to ``[]`` — a read, segment-less
    genome.
    """

    def __init__(self, records_map: dict, discover_map: dict) -> None:
        self.records_map = records_map
        self.discover_map = discover_map

    def records(self, sid: str):
        r = self.records_map.get(sid, [])
        return None if r is None else (sid, list(r))

    def discover_ids(self, sid: str):
        return self.discover_map.get(sid, [])


def _disc_from_records(records_map: dict) -> dict:
    """Non-coding-style discover map: ``None`` where records is ``None``, else the record ids."""
    return {s: (None if r is None else [k for k, _ in r]) for s, r in records_map.items()}


def test_fake_locator_satisfies_protocol() -> None:
    """The test double is a structural :class:`SegmentLocator` (runtime-checkable Protocol)."""
    assert isinstance(FakeLocator({}, {}), SegmentLocator)


def test_single_copy_gate_and_prevalence() -> None:
    """A segment single-copy in a genome is kept; a multi-copy occurrence is dropped for that genome only."""
    recs = {
        "G0": [("A→B", V), ("C→D", V * 2), ("C→D", V * 3)],  # C→D multi-copy in G0 → dropped there
        "G1": [("A→B", V)],
        "G2": [("A→B", V), ("C→D", V * 4)],  # C→D single-copy in G2 → kept
    }
    loc = FakeLocator(recs, _disc_from_records(recs))
    m, prev, read = collect_segment_matrices(loc, ["G0", "G1", "G2"], id_column="igr_pair")

    assert set(read) == {"G0", "G1", "G2"}
    assert m["A→B"][1].shape == (3, DIM)  # single-copy in all three
    assert m["C→D"][0] == ["G2"] and m["C→D"][1].shape == (1, DIM)  # multi in G0, absent G1, single G2
    p = dict(zip(prev["igr_pair"], prev["prevalence"], strict=True))
    assert p["A→B"] == pytest.approx(1.0)
    assert p["C→D"] == pytest.approx(1 / 3)  # one single-copy occurrence over 3 read genomes


def test_protein_denominator_counts_unreadable_parquet_via_empty_discover() -> None:
    """protein denominator: a records-unreadable genome whose ``discover_ids`` is ``[]`` still counts.

    Reproduces the coding screen: ``discover_core_genes`` divides by ``len(train_ids)`` (a missing-parquet
    genome contributes ``[]`` but is still in the denominator), while ``assemble_segment_matrices`` skips
    it (not in ``read_ids``). So prevalence denom = 3, but the impute universe = 2.
    """
    recs = {"G0": [("rpoB", V)], "G1": [("rpoB", V)], "Gbad": None}
    disc = {"G0": ["rpoB"], "G1": ["rpoB"], "Gbad": []}  # protein: parquet read but empty, never None
    m, prev, read = collect_segment_matrices(FakeLocator(recs, disc), ["G0", "G1", "Gbad"], id_column="gene")

    p = dict(zip(prev["gene"], prev["prevalence"], strict=True))
    assert p["rpoB"] == pytest.approx(2 / 3)  # 2 single-copy over denominator 3 (Gbad counted)
    assert set(read) == {"G0", "G1"}  # Gbad excluded from the impute universe (records None)


def test_noncoding_denominator_skips_unreadable_via_none_discover() -> None:
    """non-coding denominator: a records-unreadable genome whose ``discover_ids`` is ``None`` is skipped."""
    recs = {"G0": [("a→b", V)], "G1": [("a→b", V)], "Gbad": None}
    m, prev, read = collect_segment_matrices(
        FakeLocator(recs, _disc_from_records(recs)), ["G0", "G1", "Gbad"], id_column="igr_pair"
    )
    p = dict(zip(prev["igr_pair"], prev["prevalence"], strict=True))
    assert p["a→b"] == pytest.approx(1.0)  # denom 2 (Gbad skipped), 2 carriers → 1.0
    assert set(read) == {"G0", "G1"}


def test_prevalence_band_excludes_ubiquitous_and_pass2_is_core_only() -> None:
    """A prevalence-1.0 segment is band-excluded (not fitted) AND its vectors are never materialised."""
    ids = [f"G{i}" for i in range(6)]
    recs = {s: ([("ubiq", V)] + ([("mid", V * 2)] if i < 3 else [])) for i, s in enumerate(ids)}
    m, prev, read = collect_segment_matrices(
        FakeLocator(recs, _disc_from_records(recs)), ids, min_prevalence=0.01, max_prevalence=0.99, id_column="unit"
    )
    assert "ubiq" not in m and "mid" in m and m["mid"][1].shape == (3, DIM)  # ubiq (1.0) band-excluded
    p = dict(zip(prev["unit"], prev["prevalence"], strict=True))
    assert p["ubiq"] == pytest.approx(1.0) and p["mid"] == pytest.approx(0.5)  # both still in the table


def test_core_subset_partition_reproduces_single_shot() -> None:
    """Batching primitive: pass-1 core + unioning ``collect_core_subset`` over a partition == the single shot.

    Proves the memory fix is exact — ``sweep_core_prevalence`` recovers the same core set, and materialising
    that set in slices (``collect_core_subset`` per batch) reproduces the identical matrices + ``read_ids`` as
    the one-shot :func:`collect_segment_matrices`, so the downstream fits cannot differ.
    """
    ids = [f"G{i}" for i in range(8)]
    rng = np.random.default_rng(0)
    recs: dict = {}
    for i, s in enumerate(ids):
        row = [("segA", rng.normal(size=DIM)), ("segB", rng.normal(size=DIM))]
        if i % 2 == 0:  # segC present in half → an in-band (non-ubiquitous) core segment
            row.append(("segC", rng.normal(size=DIM)))
        recs[s] = row
    loc = FakeLocator(recs, _disc_from_records(recs))

    full_m, _prev, full_read = collect_segment_matrices(loc, ids, min_prevalence=0.0, max_prevalence=1.0, id_column="seg")
    core, _prev2 = sweep_core_prevalence(loc, ids, min_prevalence=0.0, max_prevalence=1.0, id_column="seg")
    assert set(core) == set(full_m)  # pass-1 core == what pass-2 materialised

    core_sorted = sorted(core)
    merged: dict = {}
    read2 = None
    for batch in (core_sorted[:1], core_sorted[1:]):  # arbitrary partition of the core set
        m, r = collect_core_subset(loc, ids, set(batch))
        read2 = r if read2 is None else read2
        merged.update(m)
    assert read2 == full_read  # impute universe identical across batches
    assert set(merged) == set(full_m)
    for k in full_m:
        assert merged[k][0] == full_m[k][0]
        np.testing.assert_array_equal(merged[k][1], full_m[k][1])


def test_eval_excluded_from_selection_scored_in_pass2_and_in_read_ids() -> None:
    """Held-out eval: never influences pass-1 selection, but is scored in pass-2 and joins the impute universe.

    G0–G4 are fit, G5 is held out. ``core`` (3/5 fit) is selected on the fit genomes; G5's ``core`` row is
    still collected (so ``eval_auroc`` can be computed). ``evalonly`` — carried only by G5 — is invisible to
    pass 1, so it is never screened. ``read_ids`` includes G5 (decision B: the impute universe spans
    fit ∪ eval, so a zero-imputed fit evaluates on the holdout too).
    """
    ids = [f"G{i}" for i in range(6)]
    recs = {
        "G0": [("core", V)], "G1": [("core", V)], "G2": [("core", V)],
        "G3": [("hk", V)], "G4": [("hk", V)],
        "G5": [("core", V), ("evalonly", V)],  # eval genome; evalonly is eval-exclusive
    }
    m, prev, read = collect_segment_matrices(
        FakeLocator(recs, _disc_from_records(recs)), ids, eval_ids={"G5"}, id_column="seg"
    )

    p = dict(zip(prev["seg"], prev["prevalence"], strict=True))
    assert p["core"] == pytest.approx(0.6)  # 3 of 5 FIT genomes (eval excluded from the denominator)
    assert "evalonly" not in m and "evalonly" not in set(prev["seg"])  # eval never influenced selection
    assert "G5" in m["core"][0]  # eval genome scored (swept into pass 2)
    assert set(read) == set(ids)  # decision B: eval genome in the impute universe


def test_store_dtype_float16_is_respected() -> None:
    """``store_dtype='float16'`` yields float16 design matrices (the whole-cohort memory path)."""
    recs = {"G0": [("s", V)], "G1": [("s", V)]}
    m, _prev, _read = collect_segment_matrices(
        FakeLocator(recs, _disc_from_records(recs)), ["G0", "G1"], store_dtype="float16"
    )
    assert m["s"][1].dtype == np.float16


def test_empty_sweep_yields_empty_matrices_and_typed_prevalence() -> None:
    """A sweep that reads nothing returns empty matrices + an empty, correctly-columned prevalence frame."""
    m, prev, read = collect_segment_matrices(FakeLocator({}, {}), [], id_column="gene")
    assert m == {} and read == []
    assert list(prev.columns) == ["gene", "n_single_copy", "prevalence"]
