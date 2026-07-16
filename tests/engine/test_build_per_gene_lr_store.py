"""Unit tests for the per-gene LR panel builder (``bacpredict.engine.gene_lr.build_per_gene_lr_store``).

Cover the leakage-safe out-of-fold fitting, the ``_prob_for`` train-vs-unseen routing, and the
one-pass filtered/unfiltered panel writer (synthetic genomes via a monkeypatched reader — no HPC
data, no ``.pt``/parquet on disk). Skipped where sklearn/torch are unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

import bacpredict.engine.gene_lr.build_per_gene_lr_store as bplr

DIM = 6


def _separable_matrix(ids: list[str], label_map: dict[str, int], *, sep: float, seed: int) -> np.ndarray:
    """A design matrix whose class means are ``±sep`` apart (separable when ``sep`` is large)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(len(ids), DIM))
    shift = np.array([sep if label_map[s] == 1 else -sep for s in ids])
    x[:, 0] += shift  # put the signal in feature 0
    return x


def _label_map(n_pos: int, n_neg: int) -> tuple[list[str], dict[str, int]]:
    ids = [f"R{i}" for i in range(n_pos)] + [f"S{i}" for i in range(n_neg)]
    return ids, {s: (1 if s.startswith("R") else 0) for s in ids}


def test_fit_per_gene_oof_is_leakage_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    """A separable gene yields high OOF AUROC; OOF probs cover every train id; full fit is stored."""
    ids, label_map = _label_map(12, 12)
    x = _separable_matrix(ids, label_map, sep=4.0, seed=0)

    fitted = bplr.fit_per_gene({"rpoB": (ids, x)}, label_map, n_folds=5, seed=1)

    assert "rpoB" in fitted
    f = fitted["rpoB"]
    assert f["auroc"] > 0.9  # separable -> strong, but a held-out estimate (not asserted ==1.0)
    assert set(f["oof_prob"]) == set(ids)  # every train genome got an out-of-fold value
    assert hasattr(f["clf"], "predict_proba") and hasattr(f["scaler"], "transform")


def test_fit_one_gene_eval_holdout_splits_fit_and_scores_evaluate() -> None:
    """With ``eval_ids`` the LR is fit on the non-eval genomes and scored on the held-out evaluate split.

    A separable gene → strong held-out AUROC; the eval genomes are excluded from ``oof_prob`` (they were
    never fit) and reported via ``n_eval``/``n_eval_pos``. This is the CP-R 'real numbers' path.
    """
    ids, label_map = _label_map(20, 20)
    x = _separable_matrix(ids, label_map, sep=4.0, seed=7)
    y = np.array([label_map[s] for s in ids], dtype=int)
    eval_ids = {ids[0], ids[1], ids[20], ids[21]}  # 2 pos (R0,R1) + 2 neg (S0,S1)

    f = bplr.fit_one_gene(ids, x, y, n_folds=5, seed=1, eval_ids=eval_ids)

    assert f is not None
    assert f["n_train"] == len(ids) - len(eval_ids)  # fit on the non-eval genomes only
    assert f["n_eval"] == 4 and f["n_eval_pos"] == 2
    assert set(f["oof_prob"]).isdisjoint(eval_ids)  # eval genomes never got an OOF (leakage-safe)
    assert len(f["oof_prob"]) == f["n_train"]
    assert f["eval_auroc"] > 0.9  # separable → strong on the held-out test too


def test_fit_one_gene_accepts_float16_storage() -> None:
    """float16 design-matrix storage still fits (upcast to float32) — the whole-cohort memory path."""
    ids, label_map = _label_map(12, 12)
    x = _separable_matrix(ids, label_map, sep=4.0, seed=1).astype(np.float16)
    y = np.array([label_map[s] for s in ids], dtype=int)

    f = bplr.fit_one_gene(ids, x, y, n_folds=5, seed=1)
    assert f is not None and f["auroc"] > 0.9


def test_fit_one_gene_no_eval_ids_is_backcompat() -> None:
    """``eval_ids=None`` reproduces the original OOF-only result: all ids fit, eval fields empty."""
    ids, label_map = _label_map(10, 10)
    x = _separable_matrix(ids, label_map, sep=4.0, seed=3)
    y = np.array([label_map[s] for s in ids], dtype=int)

    f = bplr.fit_one_gene(ids, x, y, n_folds=5, seed=1)

    assert set(f["oof_prob"]) == set(ids)  # every genome is a fit genome, as before
    assert f["n_train"] == len(ids) and f["n_eval"] == 0 and f["n_eval_pos"] == 0
    assert np.isnan(f["eval_auroc"])


def test_fit_per_gene_threads_eval_ids() -> None:
    """``fit_per_gene(..., eval_ids=...)`` yields the eval columns for every gene."""
    ids, label_map = _label_map(20, 20)
    x = _separable_matrix(ids, label_map, sep=4.0, seed=5)
    eval_ids = {ids[0], ids[20]}  # 1 pos + 1 neg

    fitted = bplr.fit_per_gene({"rpoB": (ids, x)}, label_map, n_folds=5, seed=1, eval_ids=eval_ids)

    f = fitted["rpoB"]
    assert f["n_eval"] == 2 and f["n_eval_pos"] == 1
    assert 0.0 <= f["eval_auroc"] <= 1.0


def test_fit_per_gene_drops_single_class() -> None:
    """A gene whose train labels are all one class has no resistance contrast and is dropped."""
    ids = [f"S{i}" for i in range(8)]
    label_map = dict.fromkeys(ids, 0)
    x = np.random.default_rng(0).normal(size=(8, DIM))
    assert bplr.fit_per_gene({"g": (ids, x)}, label_map, n_folds=5, seed=1) == {}


def test_fit_per_gene_impute_absent_recovers_presence_signal() -> None:
    """Zero-imputing absent genomes lets the LR use the presence/absence signal the drop-absent fit can't.

    The acquired-gene case: the gene is present *only* in the resistant genomes (presence == resistance).
    Drop-absent fits on present genomes only — all one class — so it is dropped (no contrast). Zero-impute
    over the full universe makes the absent (susceptible) genomes 0-vectors and the present (resistant)
    ones real, which is perfectly separable → high AUROC. This is the acquired-gene fix.
    """
    ids, label_map = _label_map(10, 10)
    present_ids = [s for s in ids if label_map[s] == 1]  # gene carried only by the resistant genomes
    # Real ESM embeddings sit well away from the origin, so a 0-vector is far from any real one — offset
    # the synthetic present vectors to reflect that (an origin-centred cloud would swallow the 0 rows).
    x_present = np.random.default_rng(0).normal(loc=5.0, size=(len(present_ids), DIM))
    gene_matrices = {"acq": (present_ids, x_present)}

    assert bplr.fit_per_gene(gene_matrices, label_map, n_folds=5, seed=1) == {}  # drop-absent: single-class

    fitted = bplr.fit_per_gene(
        gene_matrices, label_map, n_folds=5, seed=1, all_ids=ids, impute_absent_zero=True
    )
    assert "acq" in fitted
    assert fitted["acq"]["auroc"] > 0.9  # presence/absence now recovered
    assert fitted["acq"]["n_train"] == len(ids)  # fit over the full read universe, not just present


def test_prob_for_routes_train_to_oof_unseen_to_full_fit() -> None:
    """``_prob_for`` returns the stored OOF value for a fit train id, else a full-fit prediction."""
    ids, label_map = _label_map(10, 10)
    x = _separable_matrix(ids, label_map, sep=4.0, seed=2)
    fitted = bplr.fit_per_gene({"rpoB": (ids, x)}, label_map, n_folds=5, seed=1)

    train_id = ids[0]
    assert bplr._prob_for("rpoB", train_id, x[0], fitted) == fitted["rpoB"]["oof_prob"][train_id]

    unseen = bplr._prob_for("rpoB", "EVAL_GENOME", x[0], fitted)  # not in oof -> full-fit branch
    assert 0.0 <= unseen <= 1.0


def test_build_panels_filtered_zeroes_low_auroc_gene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Filtered store keeps only the high-AUROC gene; unfiltered keeps all core; non-core -> 0."""
    ids, label_map = _label_map(12, 12)
    x_good = _separable_matrix(ids, label_map, sep=4.0, seed=3)
    x_noise = np.random.default_rng(4).normal(size=(len(ids), DIM))  # label-independent -> ~0.5 AUROC
    fitted = bplr.fit_per_gene({"rpoB": (ids, x_good), "noise": (ids, x_noise)}, label_map, n_folds=5, seed=1)
    assert fitted["rpoB"]["auroc"] > fitted["noise"]["auroc"]
    filtered_genes = {"rpoB"}  # only the strong gene clears a realistic filter

    # Each genome has 3 proteins: rpoB (core+kept), noise (core, filtered out), hkA (non-core).
    gene_names = ["rpoB", "noise", "hkA"]
    emb_by_id = {s: np.stack([x_good[i], x_noise[i], np.zeros(DIM)]) for i, s in enumerate(ids)}
    emb_by_id["EVAL0"] = np.stack([x_good[0], x_noise[0], np.zeros(DIM)])  # an unseen genome

    def fake_read(sid, embed_dir, parquet_dir, *, store_kind="esm"):
        return (gene_names, emb_by_id[sid]) if sid in emb_by_id else None

    monkeypatch.setattr(bplr, "read_genome", fake_read)

    filtered_dir, unfiltered_dir = tmp_path / "filtered", tmp_path / "unfiltered"
    filtered_dir.mkdir()
    unfiltered_dir.mkdir()
    all_ids = [*ids, "EVAL0"]
    n = bplr.build_panels(
        all_ids, fitted, filtered_genes, Path("emb"), Path("pq"),
        train_set=set(ids), filtered_dir=filtered_dir, unfiltered_dir=unfiltered_dir,
    )
    assert n == len(all_ids)

    with np.load(unfiltered_dir / "R0_panel.npz") as z:
        unf = z["panel"]
        assert unf.shape == (3, 1)
        assert list(z["columns"]) == ["lr_resistance_prob"]
    with np.load(filtered_dir / "R0_panel.npz") as z:
        flt = z["panel"]

    assert unf[0, 0] != 0.0 and unf[1, 0] != 0.0 and unf[2, 0] == 0.0  # rpoB, noise nonzero; hkA zero
    assert flt[0, 0] != 0.0 and flt[1, 0] == 0.0 and flt[2, 0] == 0.0  # only rpoB survives the filter
    assert flt[0, 0] == unf[0, 0]  # the kept gene's value is identical in both stores

    # The unseen (eval) genome still gets a panel via the full-fit branch.
    assert (unfiltered_dir / "EVAL0_panel.npz").exists()

    std = json.loads((filtered_dir / "panel_standardization.json").read_text())
    assert std["columns"] == ["lr_resistance_prob"]
    assert len(std["mean"]) == 1 and len(std["std"]) == 1
    assert std["standardize_ids_restricted"] is True


def test_subsample_balanced_caps_balances_and_is_deterministic() -> None:
    """Subsample hits ~max_n with both classes represented, is a subset, and is seed-deterministic."""
    ids, label_map = _label_map(100, 100)
    picked = bplr.subsample_balanced(ids, label_map, max_n=40, seed=7)
    assert len(picked) == 40
    assert set(picked) <= set(ids)
    n_pos = sum(label_map[s] for s in picked)
    assert n_pos == 20 and (len(picked) - n_pos) == 20  # balanced halves
    assert picked == bplr.subsample_balanced(ids, label_map, max_n=40, seed=7)  # deterministic


def test_subsample_balanced_backfills_from_larger_class() -> None:
    """When one class is too small, the target is met by backfilling from the larger class."""
    ids, label_map = _label_map(5, 100)  # 5 positives only
    picked = bplr.subsample_balanced(ids, label_map, max_n=40, seed=1)
    assert len(picked) == 40
    assert sum(label_map[s] for s in picked) == 5  # all 5 positives kept, rest negatives


def test_subsample_balanced_none_or_large_returns_all() -> None:
    """``max_n`` None or ≥ len returns the input unchanged (no subsampling)."""
    ids, label_map = _label_map(10, 10)
    assert bplr.subsample_balanced(ids, label_map, max_n=None, seed=1) == ids
    assert bplr.subsample_balanced(ids, label_map, max_n=999, seed=1) == ids


def test_embedding_rows_baclm_direct_and_count_guard() -> None:
    """baclm store: plain [n_cds, dim] read directly; a CDS-count mismatch is skipped (None)."""
    torch = pytest.importorskip("torch")
    prot = torch.arange(12, dtype=torch.float32).reshape(4, 3)  # [n_cds=4, dim=3]
    store = {"protein_embeddings": prot}

    rows = bplr._embedding_rows(store, "baclm", n_genes=4)
    assert rows is not None and rows.shape == (4, 3)
    np.testing.assert_allclose(rows, prot.numpy())

    assert bplr._embedding_rows(store, "baclm", n_genes=5) is None  # count mismatch -> skip
    # A defensive leading batch dim is squeezed, not misread as n_cds=1.
    batched = {"protein_embeddings": prot[None]}  # [1, 4, 3]
    rows_b = bplr._embedding_rows(batched, "baclm", n_genes=4)
    assert rows_b is not None and rows_b.shape == (4, 3)


def test_embedding_rows_esm_selects_real_proteins() -> None:
    """ESM store: real-protein rows are selected by attention_mask; n_real may be < n_genes (capped)."""
    torch = pytest.importorskip("torch")
    # 3 real proteins + 1 padding row, plain per-protein layout (attention_mask marks real rows).
    prot = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    store = {"protein_embeddings": prot, "attention_mask": torch.tensor([[1, 1, 1, 0]])}
    rows = bplr._embedding_rows(store, "esm", n_genes=5)
    assert rows is not None and rows.shape == (3, 3)  # only the 3 real proteins
    np.testing.assert_allclose(rows, prot[0, :3].numpy())
    assert bplr._embedding_rows(store, "esm", n_genes=2) is None  # more real proteins than genes -> skip


def test_load_splits_drops_ambiguous_and_reads_split(tmp_path: Path) -> None:
    """``load_splits`` keeps only 0/1 labels and partitions by the train_val_eval column."""
    csv = tmp_path / "sheet.csv"
    csv.write_text(
        "Sample,rifampin,train_val_eval\n"
        "A,1,train\nB,0,train\nC,1,validate\nD,0,evaluate\nE,0.5,train\nF,,evaluate\n"
    )
    label_map, train, validate, evaluate = bplr.load_splits(csv, "rifampin")
    assert label_map == {"A": 1, "B": 0, "C": 1, "D": 0}  # E (0.5) and F (NaN) dropped
    assert set(train) == {"A", "B"}
    assert validate == ["C"]
    assert evaluate == ["D"]
