"""Unit tests for the generic per-label carrier-vector collector (``…gene_lr.reliable_gene_vectors``).

Covers the sidecar-agnostic accumulation: read_genome → per-genome calls_fn → per-label ids/vecs/tag_ids,
with a monkeypatched reader (no ESM/parquet on disk). Skipped where numpy is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import bacpredict.engine.gene_lr.reliable_gene_vectors as rgv
from bacpredict.engine.gene_lr.reliable_gene_vectors import ProteinCall


def test_collector_accumulates_per_label_ids_vecs_and_tag_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each call's emb[flat_index] lands under its label; tag_match feeds the tag_ids subset."""
    emb_by_id = {
        "G0": np.arange(9, dtype=np.float32).reshape(3, 3),   # 3 proteins
        "G1": (np.arange(9, dtype=np.float32) + 100).reshape(3, 3),
    }
    monkeypatch.setattr(rgv, "read_genome", lambda sid, e, p: (None, emb_by_id[sid]) if sid in emb_by_id else None)

    # G0 carries blaKPC (flat 0, tagged) + gyrA (flat 2, untagged); G1 carries blaKPC (flat 1, untagged).
    calls = {
        "G0": [ProteinCall("blaKPC", 0, "acquired", True), ProteinCall("gyrA", 2, "chromosomal", False)],
        "G1": [ProteinCall("blaKPC", 1, "acquired", False)],
    }
    read_ids, by_label = rgv.collect_reliable_gene_vectors(
        ["G0", "G1"], Path("esm"), Path("pq"), lambda sid, n: calls[sid]
    )

    assert read_ids == ["G0", "G1"]
    assert set(by_label) == {"blaKPC", "gyrA"}
    kpc = by_label["blaKPC"]
    assert kpc["source"] == "acquired"
    assert kpc["ids"] == ["G0", "G1"]
    assert kpc["tag_ids"] == {"G0"}  # only G0's call was tag_match
    np.testing.assert_allclose(kpc["vecs"][0], emb_by_id["G0"][0])
    np.testing.assert_allclose(kpc["vecs"][1], emb_by_id["G1"][1])
    assert by_label["gyrA"]["ids"] == ["G0"] and by_label["gyrA"]["tag_ids"] == set()


def test_collector_skips_unread_genomes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genome whose reader returns None is not in read_ids and yields no calls."""
    monkeypatch.setattr(
        rgv, "read_genome",
        lambda sid, e, p: None if sid == "MISS" else (None, np.zeros((1, 2), dtype=np.float32)),
    )
    read_ids, by_label = rgv.collect_reliable_gene_vectors(
        ["OK", "MISS"], Path("e"), Path("p"), lambda sid, n: [ProteinCall("g", 0, "acquired", False)]
    )
    assert read_ids == ["OK"]
    assert by_label["g"]["ids"] == ["OK"]


def test_card_wrapper_preserves_bakta_ids_shape_and_single_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The kleb CARD wrapper yields the legacy by_label shape (bakta_ids), single-copy-filtered."""
    pytest.importorskip("pandas")
    import pandas as pd

    import bacpredict.apps.kleb.per_gene_lr_from_annotation as pgla

    # One genome, sidecar parquet: blaKPC single-copy (bakta match), gyrA duplicated (dropped), OXA out-of-range.
    side = tmp_path / "G0_amr.parquet"
    pd.DataFrame({
        "amr_gene_family": ["blaKPC", "gyrA", "gyrA", "blaOXA"],
        "amr_allele": ["blaKPC-2", "gyrA_S83L", "gyrA_D87N", "blaOXA-48"],
        "amr_source": ["acquired", "chromosomal", "chromosomal", "acquired"],
        "flat_index": [0, 1, 1, 99],  # gyrA appears twice (multi-copy); blaOXA index >= n_real
        "bakta_gene_name": ["blaKPC", "gyrA", "gyrA", None],
    }).to_parquet(side)

    monkeypatch.setattr(rgv, "read_genome", lambda sid, e, p: (None, np.zeros((3, 4), dtype=np.float32)))
    read_ids, by_label = pgla.collect_reliable_amr(["G0"], tmp_path, Path("esm"), Path("pq"), grain="family")

    assert read_ids == ["G0"]
    assert set(by_label) == {"blaKPC"}                 # gyrA multi-copy dropped; blaOXA out-of-range dropped
    assert "bakta_ids" in by_label["blaKPC"] and "tag_ids" not in by_label["blaKPC"]  # renamed for back-compat
    assert by_label["blaKPC"]["bakta_ids"] == {"G0"}   # Bakta named blaKPC -> tagged
