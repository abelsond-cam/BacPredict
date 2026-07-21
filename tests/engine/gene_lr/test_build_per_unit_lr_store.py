"""Stage-A unit tests for the per-unit (named-body) LR ranking builder (``…gene_lr.build_per_unit_lr_store``).

Cover the re-embed feature reader (``_read_features`` — present/missing/no-key/mismatch/zero-row), the
multi-copy mean-pool + type filter (``_genome_unit_records``), the prevalence-band + eval-exclusion sweep
(``collect_unit_matrices``), and the end-to-end ranking driver across its three absence modes — carrier-only
(default), ``impute_absent_zero`` (the zero-imputed block the concat consumes) and ``feature="presence"``
(the one-hot control) — each asserted to write its own file/column. Real synthetic ``.pt`` stores on a
tmp path (the module reads them through its real collector — no GFF needed); no HPC data. Skipped where
sklearn / torch are unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("torch")

import torch

import bacpredict.engine.gene_lr.build_per_unit_lr_store as bpu

DIM = 6


def _save_genome(baclm_dir: Path, sid: str, rows: list[tuple[str, str, np.ndarray]]) -> None:
    """Write one ``{sid}_baclm_embeddings.pt`` with the re-embed ``feature_*`` schema from ``(type,name,vec)`` rows."""
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


# ---------------------------------------------------------------------------
# _read_features — the re-embed feature_* reader
# ---------------------------------------------------------------------------


def test_read_features_reads_bodies(tmp_path: Path) -> None:
    """A store with named bodies reads back as ``(emb[n,dim], types, names)`` in row order."""
    _save_genome(tmp_path, "G", [("rrna", "rrs", np.ones(DIM)), ("trna", "ala", np.zeros(DIM))])
    read = bpu._read_features(tmp_path / "G_baclm_embeddings.pt")
    assert read is not None
    emb, ftypes, fnames = read
    assert emb.shape == (2, DIM) and ftypes == ["rrna", "trna"] and fnames == ["rrs", "ala"]


def test_read_features_none_for_missing_or_legacy_or_mismatch(tmp_path: Path) -> None:
    """Missing file, a legacy store without ``feature_embeddings``, or a length mismatch all read as ``None``."""
    assert bpu._read_features(tmp_path / "absent_baclm_embeddings.pt") is None
    torch.save({"protein_embeddings": torch.zeros((3, DIM))}, tmp_path / "legacy_baclm_embeddings.pt")
    assert bpu._read_features(tmp_path / "legacy_baclm_embeddings.pt") is None
    torch.save({"feature_embeddings": torch.zeros((2, DIM)), "feature_type": ["rrna"], "feature_name": ["rrs"]},
               tmp_path / "bad_baclm_embeddings.pt")
    assert bpu._read_features(tmp_path / "bad_baclm_embeddings.pt") is None


def test_read_features_empty_is_not_none(tmp_path: Path) -> None:
    """A readable but feature-less genome returns empty arrays (a valid absence), not ``None`` (unreadable)."""
    _save_genome(tmp_path, "E", [])
    read = bpu._read_features(tmp_path / "E_baclm_embeddings.pt")
    assert read is not None and read[0].shape[0] == 0 and read[1] == [] and read[2] == []


# ---------------------------------------------------------------------------
# _genome_unit_records — multi-copy mean-pool + type filter
# ---------------------------------------------------------------------------


def test_genome_unit_records_mean_pools_copies(tmp_path: Path) -> None:
    """Multiple copies of one unit (the multi-copy rrn operons) collapse to one mean-pooled per-genome row."""
    a, b = np.full(DIM, 2.0), np.full(DIM, 4.0)  # two rrs copies → mean 3.0
    _save_genome(tmp_path, "G", [("rrna", "rrs", a), ("rrna", "rrs", b), ("trna", "ala", np.ones(DIM))])
    sid, records = bpu._genome_unit_records("G", str(tmp_path / "G_baclm_embeddings.pt"))
    recs = dict(records)
    assert sid == "G" and set(recs) == {"rrna:rrs", "trna:ala"}
    assert np.allclose(recs["rrna:rrs"], 3.0)  # (2+4)/2


def test_genome_unit_records_type_filter(tmp_path: Path) -> None:
    """``type_filter`` keeps only the requested feature types (e.g. rRNA), dropping the rest."""
    _save_genome(tmp_path, "G", [("rrna", "rrs", np.ones(DIM)), ("crispr", "x", np.ones(DIM))])
    _sid, records = bpu._genome_unit_records("G", str(tmp_path / "G_baclm_embeddings.pt"), type_filter={"rrna"})
    assert [k for k, _ in records] == ["rrna:rrs"]


# ---------------------------------------------------------------------------
# collect_unit_matrices — prevalence band, multi-copy-counts-once, eval exclusion
# ---------------------------------------------------------------------------


def test_collect_unit_matrices_band_and_multicopy_prevalence(tmp_path: Path) -> None:
    """A ubiquitous unit (prev 1.0, incl. a multi-copy genome) is band-excluded; a mid-band accessory is kept once."""
    ids = [f"G{i}" for i in range(6)]
    for i, s in enumerate(ids):
        rrs = [("rrna", "rrs", np.ones(DIM))]
        if i == 0:
            rrs.append(("rrna", "rrs", np.ones(DIM) * 5))  # G0 multi-copy — must still count once
        if i < 3:
            rrs.append(("crispr", "x", np.ones(DIM) * 2.0))  # 3/6 genomes
        _save_genome(tmp_path, s, rrs)

    matrices, prevalence, read_ids = bpu.collect_unit_matrices(
        ids, tmp_path, min_prevalence=0.01, max_prevalence=0.99
    )

    assert set(read_ids) == set(ids)
    prev = dict(zip(prevalence["unit"], prevalence["prevalence"], strict=False))
    assert prev["rrna:rrs"] == pytest.approx(1.0)  # multi-copy G0 counted once, not twice
    assert prev["crispr:x"] == pytest.approx(0.5)
    assert "crispr:x" in matrices and matrices["crispr:x"][1].shape == (3, DIM)
    assert "rrna:rrs" not in matrices  # prevalence 1.0 excluded by the 0.99 ceiling


def test_collect_unit_matrices_excludes_eval_from_prevalence(tmp_path: Path) -> None:
    """An ``eval_ids`` genome is scored (its rows enter the matrix) but never counts toward prevalence/universe."""
    ids = [f"G{i}" for i in range(6)]
    for s in ids:
        _save_genome(tmp_path, s, [("rrna", "rrs", np.ones(DIM))])

    matrices, _prev, read_ids = bpu.collect_unit_matrices(
        ids, tmp_path, eval_ids={"G5"}, min_prevalence=0.0, max_prevalence=1.0
    )
    assert "G5" not in read_ids and len(read_ids) == 5
    assert "G5" in matrices["rrna:rrs"][0]  # swept into pass 2 for scoring


# ---------------------------------------------------------------------------
# run — end-to-end ranking driver over real .pt, three absence modes
# ---------------------------------------------------------------------------

_DRUG = "streptomycin"


def _split_csv(tmp_path: Path, ids: list[str], label_map: dict[str, int]) -> Path:
    """Write the minimal split CSV (Sample,<drug>,train_val_eval) — every genome a train row."""
    path = tmp_path / "split.csv"
    path.write_text(f"Sample,{_DRUG},train_val_eval\n" + "".join(f"{s},{label_map[s]},train\n" for s in ids))
    return path


def _cohort_ids(n: int = 24) -> tuple[list[str], dict[str, int]]:
    ids = [f"R{i}" for i in range(n // 2)] + [f"S{i}" for i in range(n // 2)]
    return ids, {s: (1 if s.startswith("R") else 0) for s in ids}


def test_run_carrier_ranks_separable_body_top(tmp_path: Path) -> None:
    """Carrier-only: a separable multi-copy rRNA body tops the ranking; the table carries the expected schema."""
    ids, label_map = _cohort_ids()
    rng = np.random.default_rng(1)
    baclm = tmp_path / "baclm_reembed"
    for s in ids:
        shift = 4.0 if label_map[s] == 1 else -4.0
        c1, c2 = rng.normal(size=DIM), rng.normal(size=DIM)
        c1[0] += shift
        c2[0] += shift  # two rrs copies, both label-shifted → separable after mean-pool
        _save_genome(baclm, s, [("rrna", "rrs", c1), ("rrna", "rrs", c2), ("rrna", "rrl", rng.normal(size=DIM))])

    summary = bpu.run(split_csv=_split_csv(tmp_path, ids, label_map), drug=_DRUG, baclm_dir=baclm,
                      out_dir=tmp_path / "out", min_prevalence=0.0, auroc_filter=0.8, n_folds=5, seed=1, n_jobs=1)

    assert summary["feature"] == "embedding" and summary["impute_absent_zero"] is False
    assert summary["best_unit"] == "rrna:rrs" and summary["best_auroc"] > 0.9
    table = bpu.pd.read_csv(tmp_path / "out" / f"per_unit_lr_{_DRUG}.csv")
    assert list(table.columns) == [
        "unit", "feature_type", "feature_name", "prevalence", f"lr_auroc_{_DRUG}", f"eval_auroc_{_DRUG}",
        "n_train", "n_pos", "n_eval", "n_eval_pos", "kept_filtered", "impute_mode",
    ]
    assert (table["impute_mode"] == "carrier_only").all()  # embedding + no impute-absent → carrier-only
    top = table.iloc[0]
    assert top["unit"] == "rrna:rrs" and top["feature_type"] == "rrna" and top["feature_name"] == "rrs"


def test_run_impute_absent_zero_scores_accessory_body(tmp_path: Path) -> None:
    """Zero-imputed: a CRISPR body carried only by resistant genomes separates once absent genomes enter as 0."""
    ids, label_map = _cohort_ids()
    rng = np.random.default_rng(2)
    baclm = tmp_path / "baclm_reembed"
    for s in ids:
        rows = [("rrna", "rrs", rng.normal(size=DIM))]  # ubiquitous housekeeping (noise)
        if label_map[s] == 1:
            cx = rng.normal(size=DIM)
            cx[0] += 4.0  # accessory marker, R-only, linearly separable from the 0-imputed S genomes
            rows.append(("crispr", "x", cx))
        _save_genome(baclm, s, rows)

    summary = bpu.run(split_csv=_split_csv(tmp_path, ids, label_map), drug=_DRUG, baclm_dir=baclm,
                      out_dir=tmp_path / "out", min_prevalence=0.01, auroc_filter=0.8, n_folds=5, seed=1,
                      n_jobs=1, impute_absent_zero=True)

    assert summary["impute_absent_zero"] is True and summary["feature"] == "embedding"
    assert summary["best_unit"] == "crispr:x" and summary["best_auroc"] > 0.9
    assert (tmp_path / "out" / f"per_unit_lr_{_DRUG}.csv").exists()
    assert not (tmp_path / "out" / f"per_unit_presence_lr_{_DRUG}.csv").exists()


def test_run_presence_scores_lineage_body_and_applies_band(tmp_path: Path) -> None:
    """Presence one-hot: an R-only regulatory body tops; a ubiquitous rRNA (prev 1.0) is band-excluded."""
    ids, label_map = _cohort_ids()
    rng = np.random.default_rng(3)
    baclm = tmp_path / "baclm_reembed"
    for s in ids:
        rows = [("rrna", "rrs", rng.normal(size=DIM))]  # ubiquitous → excluded by the 0.99 ceiling
        if label_map[s] == 1:
            rows.append(("regulatory_region", "clone", rng.normal(size=DIM)))  # R-only → one-hot separates
        _save_genome(baclm, s, rows)

    summary = bpu.run(split_csv=_split_csv(tmp_path, ids, label_map), drug=_DRUG, baclm_dir=baclm,
                      out_dir=tmp_path / "out", min_prevalence=0.01, max_prevalence=0.99, auroc_filter=0.8,
                      n_folds=5, seed=1, n_jobs=1, feature="presence")

    assert summary["feature"] == "presence" and summary["impute_absent_zero"] is True
    assert summary["n_core"] == 1  # rrna:rrs (prevalence 1.0) excluded by the 0.99 ceiling
    assert summary["best_unit"] == "regulatory_region:clone" and summary["best_auroc"] > 0.9
    table = bpu.pd.read_csv(tmp_path / "out" / f"per_unit_presence_lr_{_DRUG}.csv")
    assert f"presence_lr_auroc_{_DRUG}" in table.columns
    assert not (tmp_path / "out" / f"per_unit_lr_{_DRUG}.csv").exists()  # presence writes its own file


def test_run_empty_when_pointed_at_legacy_store(tmp_path: Path) -> None:
    """A store with no ``feature_*`` bodies (the legacy baclm/) yields an empty table + a 0-unit summary, no crash."""
    ids, label_map = _cohort_ids(n=8)
    baclm = tmp_path / "baclm_legacy"
    baclm.mkdir()
    for s in ids:  # legacy-shaped stores: protein_embeddings only, no feature_* keys
        torch.save({"protein_embeddings": torch.zeros((3, DIM))}, baclm / f"{s}_baclm_embeddings.pt")

    summary = bpu.run(split_csv=_split_csv(tmp_path, ids, label_map), drug=_DRUG, baclm_dir=baclm,
                      out_dir=tmp_path / "out", n_folds=5, seed=1, n_jobs=1)
    assert summary["n_core"] == 0 and summary["best_unit"] is None
    assert (tmp_path / "out" / f"per_unit_lr_{_DRUG}.csv").exists()
