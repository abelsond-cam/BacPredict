"""Tests for the unitig presence/absence comparator model.

Covers the two places this pipeline can silently produce a wrong-but-plausible number: the
submatrix → sparse-matrix parse (carrier alignment, multi-copy de-duplication, all-zero rows for
non-carriers) and the alignment of the matrix to the deployed train/validate/evaluate split.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from bac_pyseer.kleb_iso_source.unitig_presence_model import (
    align_to_split,
    build_parser,
    build_presence_matrix,
    fit_l2_with_c_sweep,
    load_matrix,
    load_model,
    paired_delta_ci,
    predict_from_coefficients,
    read_sample_universe,
    save_model,
)


def _write_submatrix(path, rows: dict[str, list[str]]) -> None:
    """rows: {unitig_seq: [carrier sample ids]} in the pyseer ``<seq> | <S>:1 …`` format."""
    with path.open("w") as fh:
        for seq, carriers in rows.items():
            fh.write(f"{seq} | " + " ".join(f"{c}:1" for c in carriers) + "\n")


# ---------------------------------------------------------------------------
# Matrix build
# ---------------------------------------------------------------------------


def test_build_places_carriers_in_the_right_cells(tmp_path):
    sub = tmp_path / "hits.tsv"
    _write_submatrix(sub, {"AAA": ["S1", "S3"], "CCC": ["S2"]})
    universe = ["S1", "S2", "S3"]

    build_presence_matrix(sub, tmp_path / "mx", universe)
    X, samples, unitigs = load_matrix(tmp_path / "mx")

    assert samples == universe
    assert unitigs == ["AAA", "CCC"]
    np.testing.assert_array_equal(X.toarray(), np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32))


def test_non_carriers_become_all_zero_rows(tmp_path):
    """A genome carrying none of the hit unitigs is informative and must be kept, not dropped."""
    sub = tmp_path / "hits.tsv"
    _write_submatrix(sub, {"AAA": ["S1"]})
    manifest = build_presence_matrix(sub, tmp_path / "mx", ["S1", "S2", "S3"])

    X, samples, _ = load_matrix(tmp_path / "mx")
    assert X.shape == (3, 1)
    assert manifest["n_all_zero_rows"] == 2
    assert X[1].nnz == 0 and X[2].nnz == 0


def test_multi_copy_placements_are_deduplicated_to_presence(tmp_path):
    """The matrix is presence/absence — a unitig listed twice for a genome is still a single 1."""
    sub = tmp_path / "hits.tsv"
    _write_submatrix(sub, {"AAA": ["S1", "S1", "S2"]})
    manifest = build_presence_matrix(sub, tmp_path / "mx", ["S1", "S2"])

    X, _, _ = load_matrix(tmp_path / "mx")
    assert X.nnz == 2
    assert X.max() == 1.0
    assert manifest["n_placements_parsed"] == 3  # raw tokens still reported


def test_carriers_outside_the_universe_are_dropped_and_counted(tmp_path):
    sub = tmp_path / "hits.tsv"
    _write_submatrix(sub, {"AAA": ["S1", "GHOST"]})
    manifest = build_presence_matrix(sub, tmp_path / "mx", ["S1", "S2"])

    X, samples, _ = load_matrix(tmp_path / "mx")
    assert samples == ["S1", "S2"]
    assert manifest["n_carrier_tokens_outside_universe"] == 1
    assert X.nnz == 1


def test_build_without_a_universe_covers_carriers_only(tmp_path):
    sub = tmp_path / "hits.tsv"
    _write_submatrix(sub, {"AAA": ["S1"], "CCC": ["S2"]})
    manifest = build_presence_matrix(sub, tmp_path / "mx", None)
    assert manifest["sample_universe_given"] is False
    assert manifest["n_samples"] == 2  # S3-like non-carriers cannot be recovered


@pytest.mark.parametrize("header", [True, False])
def test_read_sample_universe_handles_both_table_shapes(tmp_path, header):
    p = tmp_path / "universe.tsv"
    body = "S1\t0\nS2\t1\n"
    p.write_text(("samples\tlabel\n" + body) if header else body)
    assert read_sample_universe(p) == ["S1", "S2"]


# ---------------------------------------------------------------------------
# Split alignment
# ---------------------------------------------------------------------------


def test_align_to_split_intersects_and_preserves_row_correspondence(tmp_path):
    sub = tmp_path / "hits.tsv"
    _write_submatrix(sub, {"AAA": ["S1", "S3"]})
    build_presence_matrix(sub, tmp_path / "mx", ["S1", "S2", "S3", "S4"])
    X, samples, _ = load_matrix(tmp_path / "mx")

    split_csv = tmp_path / "split.csv"
    pd.DataFrame({
        "Sample": ["S3", "S1", "S9"],  # S9 is absent from the matrix; order differs from the matrix
        "blood_vs_faeces_label": [1, 0, 1],
        "train_val_eval": ["train", "evaluate", "train"],
    }).to_csv(split_csv, index=False)

    Xa, df = align_to_split(X, samples, split_csv, "blood_vs_faeces_label")
    assert df["Sample"].tolist() == ["S3", "S1"]  # split order, matrix-absent row dropped
    assert Xa.shape == (2, 1)
    # Row i of Xa must be df.Sample[i] — S3 and S1 both carry AAA.
    np.testing.assert_array_equal(Xa.toarray().ravel(), np.array([1, 1], dtype=np.float32))


def test_align_drops_ambiguous_labels(tmp_path):
    sub = tmp_path / "hits.tsv"
    _write_submatrix(sub, {"AAA": ["S1"]})
    build_presence_matrix(sub, tmp_path / "mx", ["S1", "S2"])
    X, samples, _ = load_matrix(tmp_path / "mx")

    split_csv = tmp_path / "split.csv"
    pd.DataFrame({
        "Sample": ["S1", "S2"],
        "blood_vs_faeces_label": [1, 0.5],  # 0.5 = ambiguous, must not be modelled
        "train_val_eval": ["train", "train"],
    }).to_csv(split_csv, index=False)

    _Xa, df = align_to_split(X, samples, split_csv, "blood_vs_faeces_label")
    assert df["Sample"].tolist() == ["S1"]


def test_align_requires_the_split_columns(tmp_path):
    sub = tmp_path / "hits.tsv"
    _write_submatrix(sub, {"AAA": ["S1"]})
    build_presence_matrix(sub, tmp_path / "mx", ["S1"])
    X, samples, _ = load_matrix(tmp_path / "mx")

    bad = tmp_path / "bad.csv"
    pd.DataFrame({"Sample": ["S1"], "train_val_eval": ["train"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing"):
        align_to_split(X, samples, bad, "blood_vs_faeces_label")


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def _planted_signal(n=600, n_feat=40, n_signal=5, seed=0):
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.5).astype(int)
    cols = []
    for j in range(n_feat):
        p = np.where(y == 1, 0.7, 0.15) if j < n_signal else np.full(n, 0.3)
        cols.append((rng.random(n) < p).astype(np.float32))
    X = sp.csr_matrix(np.array(cols).T)
    split = np.array(["train"] * 400 + ["validate"] * 80 + ["evaluate"] * 120)
    return X, y, split


def test_fit_recovers_planted_signal_and_tunes_c_on_validate_only():
    X, y, split = _planted_signal()
    res = fit_l2_with_c_sweep(X, y, split, c_grid=(0.01, 0.1, 1.0))

    assert res["penalty"] == "l2"
    assert res["C"] in (0.01, 0.1, 1.0)
    # C is chosen by validate AUROC, so the reported best must be the sweep max.
    assert res["validate_auroc"] == max(s["validate_auroc"] for s in res["c_sweep"])
    assert res["n_train"] == 400 and res["n_validate"] == 80 and res["n_evaluate"] == 120
    assert len(res["y_prob"]) == 120

    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(res["y_true"], res["y_prob"]) > 0.75  # planted signal is recoverable


def test_fit_refuses_to_tune_without_a_validate_split():
    X, y, split = _planted_signal()
    split = np.where(split == "validate", "train", split)
    with pytest.raises(ValueError, match="no validate rows"):
        fit_l2_with_c_sweep(X, y, split, c_grid=(0.1,))


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_fit_parser_supplies_every_attribute_the_handler_reads():
    """Guards the class of bug where the handler reads an arg the subparser never declared.

    A missing `--seed` shipped once and only surfaced 24 minutes into a cluster job, after the
    expensive fit had already run — unit tests that call the fit functions directly cannot catch it.
    """
    args = build_parser().parse_args([
        "fit", "--matrix-dir", "m", "--split-csv", "s.csv", "--out-dir", "o",
    ])
    for attr in ("matrix_dir", "split_csv", "label_column", "out_dir", "bacformer_scores",
                 "bacformer_checkpoint_dir", "c_grid", "max_iter", "also_l1", "selection_scope",
                 "seed", "func"):
        assert hasattr(args, attr), f"fit subparser is missing {attr!r}"


def test_build_parser_supplies_every_attribute_the_build_handler_reads():
    args = build_parser().parse_args(["build", "--submatrix", "h.tsv", "--matrix-dir", "m"])
    for attr in ("submatrix", "sample_universe", "matrix_dir", "func"):
        assert hasattr(args, attr), f"build subparser is missing {attr!r}"


def test_paired_delta_ci_brackets_zero_for_identical_models():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.5).astype(int)
    p = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, 400), 0, 1)
    r = paired_delta_ci(y, p, p, n_boot=200)
    assert r["delta"] == pytest.approx(0.0, abs=1e-12)
    assert r["separates_from_zero"] is False


def test_paired_delta_ci_separates_when_one_model_is_clearly_better():
    rng = np.random.default_rng(1)
    y = (rng.random(600) < 0.5).astype(int)
    good = np.clip(0.5 + 0.45 * (y - 0.5) + rng.normal(0, 0.10, 600), 0, 1)
    poor = np.clip(0.5 + 0.05 * (y - 0.5) + rng.normal(0, 0.30, 600), 0, 1)
    r = paired_delta_ci(y, good, poor, n_boot=300)
    assert r["delta"] > 0
    assert r["ci_lo"] > 0
    assert r["separates_from_zero"] is True


# ---------------------------------------------------------------------------
# Model persistence — the coefficients have to survive a round trip, because the
# GWAS that selected them costs a 64-shard LMM and cannot be repeated per use.
# ---------------------------------------------------------------------------


def _toy_fit(n=300, p=40, seed=0):
    """A separable toy problem with a train/validate/evaluate split."""
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix((rng.random((n, p)) < 0.3).astype(float))
    beta = np.zeros(p)
    beta[:5] = 2.0
    y = (X @ beta + rng.normal(0, 0.5, n) > 1.5).astype(int)
    split = np.array(["train"] * (n // 2) + ["validate"] * (n // 4) + ["evaluate"] * (n - n // 2 - n // 4))
    return X, y, split, [f"UNITIG{i}" for i in range(p)]


def test_saved_coefficients_reproduce_the_fitted_probabilities(tmp_path):
    """save -> load -> re-score must be identical, or the persisted model is not the fitted one."""
    X, y, split, unitigs = _toy_fit()
    res = fit_l2_with_c_sweep(X, y, split, c_grid=(0.1, 1.0))
    save_model(tmp_path, res["model"], unitigs, C=res["C"], selection_scope="trainval_only",
               label_column="blood_vs_faeces_label")

    coef, intercept, meta = load_model(tmp_path)
    reloaded = predict_from_coefficients(X, coef, intercept, unitigs, meta)
    direct = res["model"].predict_proba(X)[:, 1]
    np.testing.assert_allclose(reloaded, direct, atol=1e-10)


def test_intercept_is_persisted(tmp_path):
    """Coefficients without an intercept can only rank, not predict — the ast_gwas sibling's gap."""
    X, y, split, unitigs = _toy_fit()
    res = fit_l2_with_c_sweep(X, y, split, c_grid=(1.0,))
    save_model(tmp_path, res["model"], unitigs, C=res["C"], selection_scope="trainval_only",
               label_column="lab")
    meta = json.loads((tmp_path / "unitig_model.json").read_text())
    assert "intercept" in meta and isinstance(meta["intercept"], float)
    assert meta["intercept"] == pytest.approx(float(res["model"].intercept_[0]))


def test_reordered_unitigs_are_refused_not_silently_scored(tmp_path):
    """Positional coefficients on a re-ordered matrix give a plausible, meaningless probability."""
    X, y, split, unitigs = _toy_fit()
    res = fit_l2_with_c_sweep(X, y, split, c_grid=(1.0,))
    save_model(tmp_path, res["model"], unitigs, C=res["C"], selection_scope="trainval_only",
               label_column="lab")
    coef, intercept, meta = load_model(tmp_path)

    shuffled = list(reversed(unitigs))
    with pytest.raises(ValueError, match="does not match the saved model"):
        predict_from_coefficients(X, coef, intercept, shuffled, meta)


def test_column_count_mismatch_is_refused(tmp_path):
    X, y, split, unitigs = _toy_fit()
    res = fit_l2_with_c_sweep(X, y, split, c_grid=(1.0,))
    save_model(tmp_path, res["model"], unitigs, C=res["C"], selection_scope="trainval_only",
               label_column="lab")
    coef, intercept, meta = load_model(tmp_path)
    with pytest.raises(ValueError, match="coefficients"):
        predict_from_coefficients(X[:, :10], coef, intercept, unitigs[:10], meta)


def test_fit_parser_exposes_score_all_splits():
    args = build_parser().parse_args(["fit", "--matrix-dir", "m", "--split-csv", "s.csv",
                                      "--out-dir", "o", "--score-all-splits"])
    assert args.score_all_splits is True


def test_predict_parser_supplies_every_attribute_the_handler_reads():
    args = build_parser().parse_args(["predict", "--matrix-dir", "m", "--model-dir", "d",
                                      "--out", "o.csv"])
    for attr in ("matrix_dir", "model_dir", "out", "func"):
        assert hasattr(args, attr), f"predict subparser is missing {attr!r}"
