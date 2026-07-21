"""Stage-A unit tests for the per-IGR LR ranking builder (``…gene_lr.build_per_igr_lr_store``).

Cover the ordered 5′→3′ flank-pair naming (+ its abutment tolerance), the single-copy sweep in
:func:`collect_igr_matrices`, and the end-to-end ranking driver — all with synthetic records via
monkeypatch, no HPC data / ``.pt`` / GFF on disk. Skipped where sklearn/torch are unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

import bacpredict.engine.gene_lr.build_per_igr_lr_store as bigr

DIM = 6


# ---------------------------------------------------------------------------
# _flank_pair — ordered pair naming + abutment tolerance
# ---------------------------------------------------------------------------


def test_flank_pair_names_abutting_neighbours_in_coordinate_order() -> None:
    """Both flanks directly abut the region → ordered (low-coord, high-coord) pair."""
    genes = [(1, 100, "gyra"), (151, 250, "gyrb")]  # sorted (start, end, name)
    assert bigr._flank_pair(genes, igr_start=101, igr_end=150, boundary_tol=3) == ("gyra", "gyrb")


def test_flank_pair_drops_region_with_far_flank() -> None:
    """A flank beyond the tolerance (unnamed CDS / RNA in between) leaves that side unnamed → None."""
    genes = [(1, 100, "gyra"), (200, 300, "gyrb")]  # right flank starts 49 bp away
    assert bigr._flank_pair(genes, igr_start=101, igr_end=150, boundary_tol=3) is None


def test_flank_pair_picks_nearest_named_flank() -> None:
    """When several named genes sit on one side, the directly-abutting (nearest) one is chosen."""
    genes = [(1, 50, "far"), (80, 100, "near"), (151, 250, "right")]
    assert bigr._flank_pair(genes, igr_start=101, igr_end=150, boundary_tol=3) == ("near", "right")


def test_flank_pair_none_when_a_side_is_empty() -> None:
    """A contig-end region (no gene on one side) is unnamed and dropped."""
    genes = [(1, 100, "gyra")]  # nothing to the right
    assert bigr._flank_pair(genes, igr_start=101, igr_end=150, boundary_tol=3) is None


# ---------------------------------------------------------------------------
# collect_igr_matrices — single-copy sweep + prevalence
# ---------------------------------------------------------------------------


def test_collect_igr_matrices_keeps_single_copy_and_counts_prevalence(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pair present once per genome is collected; a multi-copy occurrence is dropped for that genome."""
    rng = np.random.default_rng(0)

    def fake_records(sid, gff, pt, boundary_tol=3):
        if sid == "G0":  # A→B single-copy; C→D appears twice (multi-copy -> dropped)
            return sid, [("A→B", rng.normal(size=DIM)), ("C→D", rng.normal(size=DIM)), ("C→D", rng.normal(size=DIM))]
        return sid, [("A→B", rng.normal(size=DIM))]

    monkeypatch.setattr(bigr, "_genome_igr_records", fake_records)
    ids = ["G0", "G1", "G2"]
    sample_gff = {s: f"/gff/{s}.gff" for s in ids}

    matrices, prevalence, read_ids = bigr.collect_igr_matrices(ids, sample_gff, Path("baclm"))

    assert set(read_ids) == set(ids)
    assert "A→B" in matrices and matrices["A→B"][1].shape == (3, DIM)  # single-copy in all 3
    assert "C→D" not in matrices  # its only occurrence was multi-copy -> never single-copy
    prev = dict(zip(prevalence["igr_pair"], prevalence["prevalence"], strict=False))
    assert prev["A→B"] == pytest.approx(1.0)


def test_collect_igr_matrices_skips_missing_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Genomes whose GFF/.pt is missing (reader returns None) are skipped, not counted as read."""
    def fake_records(sid, gff, pt, boundary_tol=3):
        return None if sid == "G1" else (sid, [("A→B", np.ones(DIM))])

    monkeypatch.setattr(bigr, "_genome_igr_records", fake_records)
    ids = ["G0", "G1", "G2"]
    matrices, prevalence, read_ids = bigr.collect_igr_matrices(ids, {s: f"{s}.gff" for s in ids}, Path("b"))
    assert set(read_ids) == {"G0", "G2"}
    assert dict(zip(prevalence["igr_pair"], prevalence["prevalence"], strict=False))["A→B"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# run — end-to-end ranking driver
# ---------------------------------------------------------------------------


def test_run_ranks_separable_pair_top(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A separable IGR pair tops the ranking; the wide table + summary carry the expected schema."""
    n = 24
    ids = [f"R{i}" for i in range(n // 2)] + [f"S{i}" for i in range(n // 2)]
    label_map = {s: (1 if s.startswith("R") else 0) for s in ids}

    # A separable pair (signal in feature 0) and a noise pair, both single-copy in every genome.
    shift = np.array([4.0 if label_map[s] == 1 else -4.0 for s in ids])
    x_sig = np.random.default_rng(1).normal(size=(n, DIM))
    x_sig[:, 0] += shift
    x_noise = np.random.default_rng(2).normal(size=(n, DIM))
    matrices = {"katg→furA": (ids, x_sig), "hkA→hkB": (ids, x_noise)}
    prevalence = bigr.pd.DataFrame([
        {"igr_pair": "katg→furA", "n_single_copy": n, "prevalence": 1.0},
        {"igr_pair": "hkA→hkB", "n_single_copy": n, "prevalence": 1.0},
    ])

    monkeypatch.setattr(bigr, "collect_igr_matrices", lambda *a, **k: (matrices, prevalence, list(ids)))

    split_csv = tmp_path / "split.csv"
    rows = "".join(f"{s},{label_map[s]},train\n" for s in ids)
    split_csv.write_text("Sample,rifampin,train_val_eval\n" + rows)
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("Sample,sr_gff_file\n" + "".join(f"{s},/gff/{s}.gff\n" for s in ids))

    summary = bigr.run(
        split_csv=split_csv, drug="rifampin", input_csv=input_csv, baclm_dir=tmp_path / "baclm",
        out_dir=tmp_path / "out", min_prevalence=0.5, auroc_filter=0.8, n_folds=5, seed=1, n_jobs=1,
    )

    assert summary["best_igr_pair"] == "katg→furA"
    assert summary["best_igr_auroc"] > 0.9
    table = bigr.pd.read_csv(tmp_path / "out" / "per_igr_lr_rifampin.csv")
    assert list(table.columns) == [
        "igr_pair", "left_gene", "right_gene", "prevalence", "lr_auroc_rifampin", "n_train", "n_pos",
        "kept_filtered", "impute_mode",
    ]
    assert (table["impute_mode"] == "carrier_only").all()  # embedding + no impute-absent → carrier-only
    top = table.iloc[0]
    assert top["igr_pair"] == "katg→furA" and top["left_gene"] == "katg" and top["right_gene"] == "furA"


def test_run_presence_feature_scores_lineage_pair_and_applies_band(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Presence one-hot: a pair carried only by resistant genomes tops; a ubiquitous pair is band-excluded."""
    n = 24
    ids = [f"R{i}" for i in range(n // 2)] + [f"S{i}" for i in range(n // 2)]
    label_map = {s: (1 if s.startswith("R") else 0) for s in ids}
    r_ids = [s for s in ids if s.startswith("R")]

    # clone→marker is single-copy ONLY in the resistant genomes → presence alone separates the label.
    # ubiq→ubiq is present in every genome (prevalence 1.0) → dropped by the 0.99 ceiling. The embedding
    # values are irrelevant here: presence mode overwrites them with a ones-column.
    matrices = {
        "clone→marker": (r_ids, np.random.default_rng(3).normal(size=(len(r_ids), DIM))),
        "ubiq→ubiq": (ids, np.random.default_rng(4).normal(size=(n, DIM))),
    }
    prevalence = bigr.pd.DataFrame([
        {"igr_pair": "clone→marker", "n_single_copy": len(r_ids), "prevalence": len(r_ids) / n},
        {"igr_pair": "ubiq→ubiq", "n_single_copy": n, "prevalence": 1.0},
    ])
    monkeypatch.setattr(bigr, "collect_igr_matrices", lambda *a, **k: (matrices, prevalence, list(ids)))

    split_csv = tmp_path / "split.csv"
    split_csv.write_text("Sample,rifampin,train_val_eval\n" + "".join(f"{s},{label_map[s]},train\n" for s in ids))
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("Sample,sr_gff_file\n" + "".join(f"{s},/gff/{s}.gff\n" for s in ids))

    summary = bigr.run(
        split_csv=split_csv, drug="rifampin", input_csv=input_csv, baclm_dir=tmp_path / "baclm",
        out_dir=tmp_path / "out", min_prevalence=0.01, max_prevalence=0.99, auroc_filter=0.8,
        n_folds=5, seed=1, n_jobs=1, feature="presence",
    )

    assert summary["feature"] == "presence"
    assert summary["n_core_pairs"] == 1  # ubiq→ubiq (prevalence 1.0) excluded by the 0.99 ceiling
    assert summary["best_igr_pair"] == "clone→marker" and summary["best_igr_auroc"] > 0.9
    table = bigr.pd.read_csv(tmp_path / "out" / "per_igr_presence_lr_rifampin.csv")
    assert "presence_lr_auroc_rifampin" in table.columns
    assert not (tmp_path / "out" / "per_igr_lr_rifampin.csv").exists()  # presence writes its own file
