"""Stage-A parity tests for the unified per-segment LR screen (``…segment_amr_lr.per_segment_lr``).

These assert the folded-in driver reproduces each legacy ``gene_lr.build_*`` ranking's **output contract**
— the exact per-type ranking-table columns + the top segment + the prevalence-table schema — while running
through the one uniform sweep and the one split source. The sweep itself
(:func:`…segment_embedding_extractor.collect_segment_matrices`) is exercised in its own test module and is
**monkeypatched here**, so these isolate the driver: split resolution, the three absence modes, per-type
table/prevalence schemas, and the A1 fix (igr now reports a held-out ``eval_auroc``).

Split tables are written in the materialized ``<drug>_split.csv`` schema (``Sample, ast_label, split``) the
correctness spine reads. No HPC data / ``.pt`` / GFF on disk. Skipped where sklearn is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

import bacpredict.engine.segment_amr_lr.per_segment_lr as psl

DIM = 6
_DRUG = "rifampin"


# ---------------------------------------------------------------------------
# Fixtures — cohorts + the materialized <drug>_split.csv / GFF-map inputs
# ---------------------------------------------------------------------------


def _cohort(n: int = 24, prefix_pos: str = "R", prefix_neg: str = "S") -> tuple[list[str], dict[str, int]]:
    """A balanced 2-class cohort of ``n`` ids → ``(ids, label_map)`` (resistant ids first)."""
    ids = [f"{prefix_pos}{i}" for i in range(n // 2)] + [f"{prefix_neg}{i}" for i in range(n // 2)]
    return ids, {s: (1 if s.startswith(prefix_pos) else 0) for s in ids}


def _separable_matrix(ids: list[str], label_map: dict[str, int], seed: int) -> np.ndarray:
    """A label-separable design matrix — a +/-4 shift in feature 0 by class (easily > 0.9 AUROC)."""
    shift = np.array([4.0 if label_map[s] == 1 else -4.0 for s in ids])
    x = np.random.default_rng(seed).normal(size=(len(ids), DIM))
    x[:, 0] += shift
    return x


def _write_split_table(
    tmp_path: Path, label_map: dict[str, int], split_of: dict[str, str], name: str = f"{_DRUG}_split.csv"
) -> Path:
    """Write a materialized ``<drug>_split.csv`` (Sample, ast_label, split)."""
    path = tmp_path / name
    rows = "".join(f"{s},{label_map[s]},{split_of[s]}\n" for s in split_of)
    path.write_text("Sample,ast_label,split\n" + rows)
    return path


def _write_input_csv(tmp_path: Path, ids: list[str]) -> Path:
    """Write the Sample→sr_gff_file map igr/upstream ``run`` reads (the locator is monkeypatched away)."""
    path = tmp_path / "input.csv"
    path.write_text("Sample,sr_gff_file\n" + "".join(f"{s},/gff/{s}.gff\n" for s in ids))
    return path


def _patch_collect(monkeypatch: pytest.MonkeyPatch, matrices: dict, prevalence, read_ids: list[str]) -> None:
    """Monkeypatch the uniform sweep to return synthetic matrices/prevalence/read_ids (isolate the driver)."""
    monkeypatch.setattr(psl, "collect_segment_matrices", lambda *a, **k: (matrices, prevalence, list(read_ids)))


# ---------------------------------------------------------------------------
# igr — legacy schema + the A1 upgrade (gains held-out eval columns)
# ---------------------------------------------------------------------------

_IGR_COLUMNS = [
    "igr_pair", "left_gene", "right_gene", "prevalence", f"lr_auroc_{_DRUG}", f"eval_auroc_{_DRUG}",
    "n_train", "n_pos", "n_eval", "n_eval_pos", "kept_filtered", "impute_mode",
]


def test_run_igr_ranks_separable_pair_with_a1_eval_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A separable IGR pair tops; the table carries the A1-upgraded schema (igr now has the eval columns)."""
    ids, label_map = _cohort()
    matrices = {"katg→furA": (ids, _separable_matrix(ids, label_map, 1)),
                "hkA→hkB": (ids, np.random.default_rng(2).normal(size=(len(ids), DIM)))}
    prevalence = psl.pd.DataFrame([
        {"igr_pair": "katg→furA", "n_single_copy": len(ids), "prevalence": 1.0},
        {"igr_pair": "hkA→hkB", "n_single_copy": len(ids), "prevalence": 1.0},
    ])
    _patch_collect(monkeypatch, matrices, prevalence, ids)

    split_table = _write_split_table(tmp_path, label_map, dict.fromkeys(ids, "train"))
    summary = psl.run("igr", split_table=split_table, drug=_DRUG, input_csv=_write_input_csv(tmp_path, ids),
                      baclm_dir=tmp_path / "baclm", out_dir=tmp_path / "out", min_prevalence=0.5)

    assert summary["best_segment"] == "katg→furA" and summary["best_auroc"] > 0.9
    table = psl.pd.read_csv(tmp_path / "out" / f"per_igr_lr_{_DRUG}.csv")
    assert list(table.columns) == _IGR_COLUMNS  # A1: igr GAINS eval_auroc / n_eval / n_eval_pos
    assert (table["impute_mode"] == "carrier_only").all()
    top = table.iloc[0]
    assert top["igr_pair"] == "katg→furA" and top["left_gene"] == "katg" and top["right_gene"] == "furA"
    assert psl.pd.read_csv(tmp_path / "out" / "igr_prevalence.csv").columns.tolist() == [
        "igr_pair", "n_single_copy", "prevalence"]


def test_run_igr_holdout_populates_eval_auroc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """THE A1 proof: with a holdout split, a separable igr pair reports a real held-out ``eval_auroc``.

    The legacy per-IGR store had no holdout eval at all; the unified screen fits on ``train`` and scores the
    same LR on the deployed ``holdout`` genomes swept into the design matrix.
    """
    train_ids, train_labels = _cohort(24)
    hold_ids = [f"HR{i}" for i in range(4)] + [f"HS{i}" for i in range(4)]
    hold_labels = {s: (1 if s.startswith("HR") else 0) for s in hold_ids}
    all_ids = train_ids + hold_ids
    label_map = {**train_labels, **hold_labels}
    split_of = {**dict.fromkeys(train_ids, "train"), **dict.fromkeys(hold_ids, "holdout")}

    matrices = {"katg→furA": (all_ids, _separable_matrix(all_ids, label_map, 7))}  # carried by train + holdout
    prevalence = psl.pd.DataFrame([{"igr_pair": "katg→furA", "n_single_copy": len(train_ids), "prevalence": 1.0}])
    _patch_collect(monkeypatch, matrices, prevalence, all_ids)

    split_table = _write_split_table(tmp_path, label_map, split_of)
    psl.run("igr", split_table=split_table, drug=_DRUG, input_csv=_write_input_csv(tmp_path, all_ids),
            baclm_dir=tmp_path / "baclm", out_dir=tmp_path / "out", min_prevalence=0.5)

    row = psl.pd.read_csv(tmp_path / "out" / f"per_igr_lr_{_DRUG}.csv").iloc[0]
    assert row["n_eval"] == 8 and row["n_eval_pos"] == 4       # the 8 holdout genomes, 4 resistant
    assert row[f"eval_auroc_{_DRUG}"] > 0.8                     # a real held-out number, not NaN


# ---------------------------------------------------------------------------
# upstream — legacy schema + presence mode / band
# ---------------------------------------------------------------------------

_UPSTREAM_COLUMNS = [
    "upstream_gene", "gene", "prevalence", f"lr_auroc_{_DRUG}", f"eval_auroc_{_DRUG}",
    "n_train", "n_pos", "n_eval", "n_eval_pos", "kept_filtered", "impute_mode",
]


def test_run_upstream_carrier_schema_and_top(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Carrier-only: a separable upstream anchor tops; the table matches the legacy upstream schema exactly."""
    ids, label_map = _cohort()
    matrices = {"upstream:katg": (ids, _separable_matrix(ids, label_map, 1)),
                "upstream:hk": (ids, np.random.default_rng(2).normal(size=(len(ids), DIM)))}
    prevalence = psl.pd.DataFrame([
        {"upstream_gene": "upstream:katg", "n_single_copy": len(ids), "prevalence": 1.0},
        {"upstream_gene": "upstream:hk", "n_single_copy": len(ids), "prevalence": 1.0},
    ])
    _patch_collect(monkeypatch, matrices, prevalence, ids)

    split_table = _write_split_table(tmp_path, label_map, dict.fromkeys(ids, "train"))
    summary = psl.run("upstream", split_table=split_table, drug=_DRUG, input_csv=_write_input_csv(tmp_path, ids),
                      baclm_dir=tmp_path / "baclm", out_dir=tmp_path / "out", min_prevalence=0.5)

    assert summary["best_segment"] == "upstream:katg" and summary["best_auroc"] > 0.9
    table = psl.pd.read_csv(tmp_path / "out" / f"per_upstream_lr_{_DRUG}.csv")
    assert list(table.columns) == _UPSTREAM_COLUMNS
    assert (table["impute_mode"] == "carrier_only").all()
    top = table.iloc[0]
    assert top["upstream_gene"] == "upstream:katg" and top["gene"] == "katg"


def test_run_upstream_presence_writes_own_file_and_bands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Presence one-hot: writes its own file/column; a ubiquitous anchor (prev 1.0) is band-excluded."""
    ids, label_map = _cohort()
    r_ids = [s for s in ids if label_map[s] == 1]
    matrices = {"upstream:clone": (r_ids, np.random.default_rng(3).normal(size=(len(r_ids), DIM)))}
    prevalence = psl.pd.DataFrame([
        {"upstream_gene": "upstream:clone", "n_single_copy": len(r_ids), "prevalence": len(r_ids) / len(ids)},
        {"upstream_gene": "upstream:ubiq", "n_single_copy": len(ids), "prevalence": 1.0},
    ])
    # The band drops ubiq (1.0) BEFORE the sweep in the real collector; here the sweep is patched, so we hand
    # back only the in-band matrix (clone) — the driver must still route presence to its own file/column.
    _patch_collect(monkeypatch, matrices, prevalence, ids)

    split_table = _write_split_table(tmp_path, label_map, dict.fromkeys(ids, "train"))
    summary = psl.run("upstream", split_table=split_table, drug=_DRUG, input_csv=_write_input_csv(tmp_path, ids),
                      baclm_dir=tmp_path / "baclm", out_dir=tmp_path / "out", min_prevalence=0.01,
                      max_prevalence=0.99, feature="presence")

    assert summary["feature"] == "presence" and summary["impute_absent_zero"] is True
    assert summary["best_segment"] == "upstream:clone" and summary["best_auroc"] > 0.9
    table = psl.pd.read_csv(tmp_path / "out" / f"per_upstream_presence_lr_{_DRUG}.csv")
    assert f"presence_lr_auroc_{_DRUG}" in table.columns
    assert (table["impute_mode"] == "presence").all()
    assert not (tmp_path / "out" / f"per_upstream_lr_{_DRUG}.csv").exists()  # presence writes its own file


# ---------------------------------------------------------------------------
# unit — legacy schema + prevalence enrichment + impute mode
# ---------------------------------------------------------------------------

_UNIT_COLUMNS = [
    "unit", "feature_type", "feature_name", "prevalence", f"lr_auroc_{_DRUG}", f"eval_auroc_{_DRUG}",
    "n_train", "n_pos", "n_eval", "n_eval_pos", "kept_filtered", "impute_mode",
]


def test_run_unit_schema_top_and_prevalence_enrichment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit: ranking table matches the legacy schema; the prevalence CSV is enriched to feature_type/name/n_present.

    The uniform sweep returns a generic ``[unit, n_single_copy, prevalence]`` prevalence frame; the driver must
    re-derive ``feature_type`` / ``feature_name`` (from the ``type:name`` key) and rename ``n_single_copy`` →
    ``n_present`` to reproduce the legacy ``unit_prevalence.csv`` schema.
    """
    ids, label_map = _cohort()
    matrices = {"rrna:rrs": (ids, _separable_matrix(ids, label_map, 1)),
                "trna:ala": (ids, np.random.default_rng(2).normal(size=(len(ids), DIM)))}
    prevalence = psl.pd.DataFrame([  # the sweep's generic schema (id_column="unit")
        {"unit": "rrna:rrs", "n_single_copy": len(ids), "prevalence": 1.0},
        {"unit": "trna:ala", "n_single_copy": len(ids), "prevalence": 1.0},
    ])
    _patch_collect(monkeypatch, matrices, prevalence, ids)

    split_table = _write_split_table(tmp_path, label_map, dict.fromkeys(ids, "train"))
    summary = psl.run("unit", split_table=split_table, drug=_DRUG, baclm_dir=tmp_path / "baclm",
                      out_dir=tmp_path / "out", min_prevalence=0.0)

    assert summary["best_segment"] == "rrna:rrs" and summary["best_auroc"] > 0.9
    table = psl.pd.read_csv(tmp_path / "out" / f"per_unit_lr_{_DRUG}.csv")
    assert list(table.columns) == _UNIT_COLUMNS
    top = table.iloc[0]
    assert top["unit"] == "rrna:rrs" and top["feature_type"] == "rrna" and top["feature_name"] == "rrs"
    prev = psl.pd.read_csv(tmp_path / "out" / "unit_prevalence.csv")
    assert list(prev.columns) == ["unit", "feature_type", "feature_name", "n_present", "prevalence"]
    assert set(prev["feature_type"]) == {"rrna", "trna"}


def test_run_unit_impute_absent_zero_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero-imputed: an accessory body carried only by resistant genomes separates; impute_mode is stamped."""
    ids, label_map = _cohort()
    r_ids = [s for s in ids if label_map[s] == 1]
    x = np.random.default_rng(3).normal(size=(len(r_ids), DIM))
    x[:, 0] += 4.0  # R-only marker, linearly separable from the 0-imputed S genomes
    matrices = {"crispr:x": (r_ids, x)}
    prevalence = psl.pd.DataFrame([{"unit": "crispr:x", "n_single_copy": len(r_ids), "prevalence": len(r_ids) / len(ids)}])
    _patch_collect(monkeypatch, matrices, prevalence, ids)

    split_table = _write_split_table(tmp_path, label_map, dict.fromkeys(ids, "train"))
    summary = psl.run("unit", split_table=split_table, drug=_DRUG, baclm_dir=tmp_path / "baclm",
                      out_dir=tmp_path / "out", min_prevalence=0.01, impute_absent_zero=True)

    assert summary["impute_absent_zero"] is True and summary["impute_mode"] == "imputed_zero"
    assert summary["best_segment"] == "crispr:x" and summary["best_auroc"] > 0.9
    table = psl.pd.read_csv(tmp_path / "out" / f"per_unit_lr_{_DRUG}.csv")
    assert (table["impute_mode"] == "imputed_zero").all()
    assert not (tmp_path / "out" / f"per_unit_presence_lr_{_DRUG}.csv").exists()


# ---------------------------------------------------------------------------
# empty / core-count / rank_segments core
# ---------------------------------------------------------------------------


def test_run_empty_core_writes_typed_header_and_zero_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """0 in-band segments (e.g. a legacy store): a headered empty table + a 0-core summary, no crash."""
    ids, label_map = _cohort(n=8)
    empty_prev = psl.pd.DataFrame(columns=["unit", "n_single_copy", "prevalence"])
    _patch_collect(monkeypatch, {}, empty_prev, ids)

    split_table = _write_split_table(tmp_path, label_map, dict.fromkeys(ids, "train"))
    summary = psl.run("unit", split_table=split_table, drug=_DRUG, baclm_dir=tmp_path / "baclm",
                      out_dir=tmp_path / "out")

    assert summary["n_core"] == 0 and summary["n_fitted"] == 0 and summary["best_segment"] is None
    table = psl.pd.read_csv(tmp_path / "out" / f"per_unit_lr_{_DRUG}.csv")
    assert list(table.columns) == _UNIT_COLUMNS and len(table) == 0


def test_rank_segments_core_returns_mode_and_core_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """The type-agnostic core returns ``(fitted, prevalence, read_ids, n_core, impute_mode)`` with the right tags."""
    ids, label_map = _cohort()
    matrices = {"seg": (ids, _separable_matrix(ids, label_map, 1))}
    prevalence = psl.pd.DataFrame([{"seg_id": "seg", "n_single_copy": len(ids), "prevalence": 1.0}])
    _patch_collect(monkeypatch, matrices, prevalence, ids)

    fitted, prev, read_ids, n_core, impute_mode = psl.rank_segments(
        object(), ids, label_map, id_column="seg_id", eval_ids=None, impute_absent_zero=True,
    )
    assert n_core == 1 and impute_mode == "imputed_zero" and set(read_ids) == set(ids)
    assert "seg" in fitted and fitted["seg"]["auroc"] > 0.9
    assert prev is prevalence


# ---------------------------------------------------------------------------
# coding (protein) — gene_name + annotation table, no presence, panel side-output
# ---------------------------------------------------------------------------

_CODING_COLUMNS = [
    "gene_name", "annotation", "prevalence", f"lr_auroc_{_DRUG}", f"eval_auroc_{_DRUG}",
    "n_train", "n_pos", "n_eval", "n_eval_pos", "kept_filtered", "impute_mode",
]


def test_run_coding_table_schema_top_and_annotation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Coding: ``per_gene_lr_<drug>.csv`` matches write_gene_drug_table (gene_name + annotation); top gene wins.

    The sweep keys prevalence by ``gene``; the ranking table keys ``gene_name`` and joins a parquet
    ``annotation`` (product) column (here monkeypatched) — the one type whose table key ≠ prevalence key.
    """
    ids, label_map = _cohort()
    matrices = {"rpoB": (ids, _separable_matrix(ids, label_map, 1)),
                "hkA": (ids, np.random.default_rng(2).normal(size=(len(ids), DIM)))}
    prevalence = psl.pd.DataFrame([  # sweep schema, id_column="gene"
        {"gene": "rpoB", "n_single_copy": len(ids), "prevalence": 1.0},
        {"gene": "hkA", "n_single_copy": len(ids), "prevalence": 1.0},
    ])
    _patch_collect(monkeypatch, matrices, prevalence, ids)
    monkeypatch.setattr(psl, "_coding_annotation", lambda fit_ids, pq: {"rpoB": "DNA-directed RNA polymerase subunit beta"})

    split_table = _write_split_table(tmp_path, label_map, dict.fromkeys(ids, "train"))
    summary = psl.run("coding", split_table=split_table, drug=_DRUG, out_dir=tmp_path / "out",
                      embed_dir=tmp_path / "emb", parquet_dir=tmp_path / "pq", store_kind="esm", min_prevalence=0.5)

    assert summary["segment_type"] == "coding" and summary["embedding_store"] == "esm"
    assert summary["best_segment"] == "rpoB" and summary["best_auroc"] > 0.9
    table = psl.pd.read_csv(tmp_path / "out" / f"per_gene_lr_{_DRUG}.csv")
    assert list(table.columns) == _CODING_COLUMNS
    top = table.iloc[0]
    assert top["gene_name"] == "rpoB" and top["annotation"] == "DNA-directed RNA polymerase subunit beta"
    assert (table["impute_mode"] == "carrier_only").all()
    prev = psl.pd.read_csv(tmp_path / "out" / "gene_prevalence.csv")
    assert list(prev.columns) == ["gene", "n_single_copy", "prevalence"]  # sweep schema, no enrichment


def test_run_coding_rejects_presence(tmp_path: Path) -> None:
    """Coding has no presence one-hot mode → ``feature='presence'`` is a fail-fast ValueError."""
    ids, label_map = _cohort(n=8)
    split_table = _write_split_table(tmp_path, label_map, dict.fromkeys(ids, "train"))
    with pytest.raises(ValueError, match="presence"):
        psl.run("coding", split_table=split_table, drug=_DRUG, out_dir=tmp_path / "out",
                embed_dir=tmp_path / "emb", parquet_dir=tmp_path / "pq", feature="presence")


def test_run_coding_requires_embed_and_parquet(tmp_path: Path) -> None:
    """Coding needs the embedding-store + parquet dirs (fail-fast if omitted)."""
    ids, label_map = _cohort(n=8)
    split_table = _write_split_table(tmp_path, label_map, dict.fromkeys(ids, "train"))
    with pytest.raises(ValueError, match="embed_dir"):
        psl.run("coding", split_table=split_table, drug=_DRUG, out_dir=tmp_path / "out")


def test_run_coding_write_panels_invokes_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """write_panels wires build_panels over train+validate+holdout, standardising on the fit-train subsample."""
    train_ids, train_labels = _cohort(24)
    hold_ids = [f"H{i}" for i in range(4)]
    hold_labels = {s: (i % 2) for i, s in enumerate(hold_ids)}
    all_ids = train_ids + hold_ids
    label_map = {**train_labels, **hold_labels}
    split_of = {**dict.fromkeys(train_ids, "train"), **dict.fromkeys(hold_ids, "holdout")}

    matrices = {"rpoB": (all_ids, _separable_matrix(all_ids, label_map, 1))}
    prevalence = psl.pd.DataFrame([{"gene": "rpoB", "n_single_copy": len(train_ids), "prevalence": 1.0}])
    _patch_collect(monkeypatch, matrices, prevalence, all_ids)
    monkeypatch.setattr(psl, "_coding_annotation", lambda fit_ids, pq: {})

    captured: dict = {}

    def fake_build_panels(a_ids, fitted, filtered, embed_dir, parquet_dir, *, train_set, filtered_dir,
                          unfiltered_dir, store_kind):
        captured.update(all_ids=list(a_ids), train_set=set(train_set), store_kind=store_kind, genes=set(fitted))
        return len(a_ids)

    monkeypatch.setattr("bacpredict.engine.segment_amr_lr.panel_store.build_panels", fake_build_panels)

    split_table = _write_split_table(tmp_path, label_map, split_of)
    summary = psl.run("coding", split_table=split_table, drug=_DRUG, out_dir=tmp_path / "out",
                      embed_dir=tmp_path / "emb", parquet_dir=tmp_path / "pq", store_kind="baclm",
                      write_panels=True, min_prevalence=0.5)

    assert summary["wrote_panels"] is True and summary["n_genomes_written"] == len(all_ids)
    assert set(captured["all_ids"]) == set(all_ids)  # panels cover train + validate + holdout
    assert captured["store_kind"] == "baclm" and "rpoB" in captured["genes"]
