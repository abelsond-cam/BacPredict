"""Stage-A unit tests for the upstream-region LR ranking builder (``…gene_lr.build_upstream_region_lr_store``).

Cover the 5′-anchoring index (``_upstream_region_index`` on both strands + its tolerance), the
prevalence-band single-copy sweep in :func:`collect_upstream_matrices`, and the end-to-end ranking
driver across its three absence modes — carrier-only (default), ``impute_absent_zero`` (the
zero-imputed block the concat actually consumes), and ``feature="presence"`` (the one-hot control) —
asserting each writes its own file/column so nothing is overwritten. Synthetic records via monkeypatch;
no HPC data / ``.pt`` / GFF on disk. Skipped where sklearn is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

import bacpredict.engine.gene_lr.build_upstream_region_lr_store as bur

DIM = 6


# ---------------------------------------------------------------------------
# _upstream_region_index — 5′-anchoring on both strands + abutment tolerance
# ---------------------------------------------------------------------------


def test_upstream_region_index_minus_strand_picks_region_above_gene_end() -> None:
    """On ``-`` strand the 5′ end is the high coordinate; the region abutting ``gend+1`` is chosen."""
    rows = [(201, 260, 0), (400, 460, 1)]  # (start, end, row_idx); first abuts gend=200
    assert bur._upstream_region_index(100, 200, "-", rows, boundary_tol=3) == 0


def test_upstream_region_index_plus_strand_picks_region_below_gene_start() -> None:
    """On ``+`` strand the 5′ end is ``gstart``; the region abutting ``gstart-1`` is chosen."""
    rows = [(40, 99, 0), (200, 260, 1)]  # first ends at gstart-1 = 99
    assert bur._upstream_region_index(100, 180, "+", rows, boundary_tol=3) == 0


def test_upstream_region_index_none_beyond_tolerance() -> None:
    """A region whose gap exceeds the tolerance (an unnamed CDS/RNA in between) is not anchored."""
    rows = [(210, 260, 0)]  # 9 bp above gend+1 (=201), beyond tol 3
    assert bur._upstream_region_index(100, 200, "-", rows, boundary_tol=3) is None


# ---------------------------------------------------------------------------
# _genome_upstream_records — convergent flank-pair fallback (--include-convergent)
# ---------------------------------------------------------------------------


def test_genome_records_convergent_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A region between two convergent genes has no 5′ anchor; the fallback names it by its flank pair.

    Synthetic contig ``c1``: ``mura`` (+, 100-200) and ``ogt`` (−, 300-400) both abut the region 201-299
    with their **3′** ends (convergent) — like the rrn/rrs operon — so no ``upstream:<gene>`` key claims it.
    ``rpob`` (+, 500-600) has a clean 5′ region 450-499. With the flag off only the two 5′-anchored keys
    appear; with it on the convergent region is emitted as ``between:mura→ogt``.
    """
    gff, pt = tmp_path / "g.gff", tmp_path / "g.pt"
    gff.write_text("x")
    pt.write_text("x")
    genes = {"mura": [("c1", 100, 200, "+")], "ogt": [("c1", 300, 400, "-")], "rpob": [("c1", 500, 600, "+")]}
    seqids, starts, ends = ["c1", "c1", "c1"], [1, 201, 450], [99, 299, 499]
    emb = np.arange(3 * DIM, dtype=float).reshape(3, DIM)
    monkeypatch.setattr(bur, "_parse_gff", lambda p: ({}, genes))
    monkeypatch.setattr(bur, "_read_intergenic", lambda p: (emb, seqids, starts, ends))

    _sid, recs = bur._genome_upstream_records("G", str(gff), str(pt))
    assert {k for k, _ in recs} == {"upstream:mura", "upstream:rpob"}  # default: 5′ anchors only

    _sid, recs2 = bur._genome_upstream_records("G", str(gff), str(pt), include_convergent=True)
    by_key = dict(recs2)
    assert set(by_key) == {"upstream:mura", "upstream:rpob", "between:mura→ogt"}
    np.testing.assert_array_equal(by_key["between:mura→ogt"], emb[1])  # the 201-299 region's embedding


# ---------------------------------------------------------------------------
# collect_upstream_matrices — single-copy sweep + (min, max] prevalence band
# ---------------------------------------------------------------------------


def test_collect_upstream_matrices_applies_prevalence_band(monkeypatch: pytest.MonkeyPatch) -> None:
    """A near-ubiquitous anchor (prevalence 1.0) is dropped by the ceiling; a mid-band anchor is kept."""
    def fake_records(sid, gff, pt, boundary_tol=3, include_convergent=False):
        recs = [("upstream:ubiq", np.ones(DIM))]  # every genome carries it single-copy
        if sid in {"G0", "G1", "G2"}:
            recs.append(("upstream:mid", np.ones(DIM) * 2.0))  # half the genomes
        return sid, recs

    monkeypatch.setattr(bur, "_genome_upstream_records", fake_records)
    ids = [f"G{i}" for i in range(6)]
    sample_gff = {s: f"/gff/{s}.gff" for s in ids}

    matrices, prevalence, read_ids = bur.collect_upstream_matrices(
        ids, sample_gff, Path("baclm"), min_prevalence=0.01, max_prevalence=0.99
    )

    assert set(read_ids) == set(ids)
    prev = dict(zip(prevalence["upstream_gene"], prevalence["prevalence"], strict=False))
    assert prev["upstream:ubiq"] == pytest.approx(1.0)
    assert prev["upstream:mid"] == pytest.approx(0.5)
    assert "upstream:mid" in matrices and matrices["upstream:mid"][1].shape == (3, DIM)
    assert "upstream:ubiq" not in matrices  # prevalence 1.0 excluded by the 0.99 ceiling


# ---------------------------------------------------------------------------
# run — end-to-end ranking driver, three absence modes
# ---------------------------------------------------------------------------


def _write_split_and_input(tmp_path: Path, ids: list[str], label_map: dict[str, int]) -> tuple[Path, Path]:
    """Write the minimal split CSV (Sample,rifampin,train_val_eval) + input CSV (Sample,sr_gff_file)."""
    split_csv = tmp_path / "split.csv"
    split_csv.write_text("Sample,rifampin,train_val_eval\n" + "".join(f"{s},{label_map[s]},train\n" for s in ids))
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("Sample,sr_gff_file\n" + "".join(f"{s},/gff/{s}.gff\n" for s in ids))
    return split_csv, input_csv


def test_run_carrier_ranks_separable_anchor_top(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Carrier-only (default): a separable upstream anchor tops; the table carries the expected schema."""
    n = 24
    ids = [f"R{i}" for i in range(n // 2)] + [f"S{i}" for i in range(n // 2)]
    label_map = {s: (1 if s.startswith("R") else 0) for s in ids}

    shift = np.array([4.0 if label_map[s] == 1 else -4.0 for s in ids])
    x_sig = np.random.default_rng(1).normal(size=(n, DIM))
    x_sig[:, 0] += shift
    x_noise = np.random.default_rng(2).normal(size=(n, DIM))
    matrices = {"upstream:katg": (ids, x_sig), "upstream:hk": (ids, x_noise)}
    prevalence = bur.pd.DataFrame([
        {"upstream_gene": "upstream:katg", "n_single_copy": n, "prevalence": 1.0},
        {"upstream_gene": "upstream:hk", "n_single_copy": n, "prevalence": 1.0},
    ])
    monkeypatch.setattr(bur, "collect_upstream_matrices", lambda *a, **k: (matrices, prevalence, list(ids)))

    split_csv, input_csv = _write_split_and_input(tmp_path, ids, label_map)
    summary = bur.run(
        split_csv=split_csv, drug="rifampin", input_csv=input_csv, baclm_dir=tmp_path / "baclm",
        out_dir=tmp_path / "out", min_prevalence=0.5, auroc_filter=0.8, n_folds=5, seed=1, n_jobs=1,
    )

    assert summary["feature"] == "embedding" and summary["impute_absent_zero"] is False
    assert summary["include_convergent"] is False and summary["n_between"] == 0  # default: no fallback keys
    assert summary["best_anchor"] == "upstream:katg" and summary["best_auroc"] > 0.9
    table = bur.pd.read_csv(tmp_path / "out" / "per_upstream_lr_rifampin.csv")
    assert list(table.columns) == [
        "upstream_gene", "gene", "prevalence", "lr_auroc_rifampin", "eval_auroc_rifampin",
        "n_train", "n_pos", "n_eval", "n_eval_pos", "kept_filtered",
    ]
    top = table.iloc[0]
    assert top["upstream_gene"] == "upstream:katg" and top["gene"] == "katg"


def test_run_impute_absent_zero_scores_accessory_anchor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-imputed: an anchor carried only by resistant genomes separates once absent genomes enter as 0."""
    n = 24
    ids = [f"R{i}" for i in range(n // 2)] + [f"S{i}" for i in range(n // 2)]
    label_map = {s: (1 if s.startswith("R") else 0) for s in ids}
    r_ids = [s for s in ids if s.startswith("R")]

    # marker is single-copy ONLY in resistant genomes, with a linear +4 shift in feature 0; under
    # zero-imputation the S genomes become 0-vectors, so a linear boundary separates carriers from absent.
    x_marker = np.random.default_rng(3).normal(size=(len(r_ids), DIM))
    x_marker[:, 0] += 4.0
    matrices = {"upstream:marker": (r_ids, x_marker)}
    prevalence = bur.pd.DataFrame(
        [{"upstream_gene": "upstream:marker", "n_single_copy": len(r_ids), "prevalence": len(r_ids) / n}]
    )
    monkeypatch.setattr(bur, "collect_upstream_matrices", lambda *a, **k: (matrices, prevalence, list(ids)))

    split_csv, input_csv = _write_split_and_input(tmp_path, ids, label_map)
    summary = bur.run(
        split_csv=split_csv, drug="rifampin", input_csv=input_csv, baclm_dir=tmp_path / "baclm",
        out_dir=tmp_path / "out", min_prevalence=0.01, auroc_filter=0.8, n_folds=5, seed=1, n_jobs=1,
        impute_absent_zero=True,
    )

    assert summary["impute_absent_zero"] is True and summary["feature"] == "embedding"
    assert summary["best_anchor"] == "upstream:marker" and summary["best_auroc"] > 0.9
    table = bur.pd.read_csv(tmp_path / "out" / "per_upstream_lr_rifampin.csv")
    assert "lr_auroc_rifampin" in table.columns
    assert not (tmp_path / "out" / "per_upstream_presence_lr_rifampin.csv").exists()


def test_run_presence_scores_lineage_anchor_and_applies_band(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Presence one-hot: an anchor carried only by resistant genomes tops; a ubiquitous one is band-excluded."""
    n = 24
    ids = [f"R{i}" for i in range(n // 2)] + [f"S{i}" for i in range(n // 2)]
    label_map = {s: (1 if s.startswith("R") else 0) for s in ids}
    r_ids = [s for s in ids if s.startswith("R")]

    matrices = {
        "upstream:clone": (r_ids, np.random.default_rng(3).normal(size=(len(r_ids), DIM))),
        "upstream:ubiq": (ids, np.random.default_rng(4).normal(size=(n, DIM))),
    }
    prevalence = bur.pd.DataFrame([
        {"upstream_gene": "upstream:clone", "n_single_copy": len(r_ids), "prevalence": len(r_ids) / n},
        {"upstream_gene": "upstream:ubiq", "n_single_copy": n, "prevalence": 1.0},
    ])
    monkeypatch.setattr(bur, "collect_upstream_matrices", lambda *a, **k: (matrices, prevalence, list(ids)))

    split_csv, input_csv = _write_split_and_input(tmp_path, ids, label_map)
    summary = bur.run(
        split_csv=split_csv, drug="rifampin", input_csv=input_csv, baclm_dir=tmp_path / "baclm",
        out_dir=tmp_path / "out", min_prevalence=0.01, max_prevalence=0.99, auroc_filter=0.8,
        n_folds=5, seed=1, n_jobs=1, feature="presence",
    )

    assert summary["feature"] == "presence" and summary["impute_absent_zero"] is True
    assert summary["n_core"] == 1  # upstream:ubiq (prevalence 1.0) excluded by the 0.99 ceiling
    assert summary["best_anchor"] == "upstream:clone" and summary["best_auroc"] > 0.9
    table = bur.pd.read_csv(tmp_path / "out" / "per_upstream_presence_lr_rifampin.csv")
    assert "presence_lr_auroc_rifampin" in table.columns
    assert not (tmp_path / "out" / "per_upstream_lr_rifampin.csv").exists()  # presence writes its own file
