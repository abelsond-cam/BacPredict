"""Unit tests for the per-protein LR-probability panel store (``…segment_amr_lr.panel_store``).

Cover the leakage-safe ``_prob_for`` train-vs-unseen routing and the one-pass filtered/unfiltered panel
writer (synthetic genomes via a monkeypatched reader — no HPC data, no ``.pt``/parquet on disk). Relocated
from the retired ``gene_lr.build_per_gene_lr_store``. Skipped where sklearn is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

import bacpredict.engine.segment_amr_lr.panel_store as ps
from bacpredict.engine.segment_amr_lr.fit_lr import fit_per_segment

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


def test_prob_for_routes_train_to_oof_unseen_to_full_fit() -> None:
    """``_prob_for`` returns the stored OOF value for a fit train id, else a full-fit prediction."""
    ids, label_map = _label_map(10, 10)
    x = _separable_matrix(ids, label_map, sep=4.0, seed=2)
    fitted = fit_per_segment({"rpoB": (ids, x)}, label_map, n_folds=5, seed=1)

    train_id = ids[0]
    assert ps._prob_for("rpoB", train_id, x[0], fitted) == fitted["rpoB"]["oof_prob"][train_id]

    unseen = ps._prob_for("rpoB", "EVAL_GENOME", x[0], fitted)  # not in oof -> full-fit branch
    assert 0.0 <= unseen <= 1.0


def test_build_panels_filtered_zeroes_low_auroc_gene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Filtered store keeps only the high-AUROC gene; unfiltered keeps all core; non-core -> 0."""
    ids, label_map = _label_map(12, 12)
    x_good = _separable_matrix(ids, label_map, sep=4.0, seed=3)
    x_noise = np.random.default_rng(4).normal(size=(len(ids), DIM))  # label-independent -> ~0.5 AUROC
    fitted = fit_per_segment({"rpoB": (ids, x_good), "noise": (ids, x_noise)}, label_map, n_folds=5, seed=1)
    assert fitted["rpoB"]["auroc"] > fitted["noise"]["auroc"]
    filtered_genes = {"rpoB"}  # only the strong gene clears a realistic filter

    # Each genome has 3 proteins: rpoB (core+kept), noise (core, filtered out), hkA (non-core).
    gene_names = ["rpoB", "noise", "hkA"]
    emb_by_id = {s: np.stack([x_good[i], x_noise[i], np.zeros(DIM)]) for i, s in enumerate(ids)}
    emb_by_id["EVAL0"] = np.stack([x_good[0], x_noise[0], np.zeros(DIM)])  # an unseen genome

    def fake_read(sid, embed_dir, parquet_dir, *, store_kind="esm"):
        return (gene_names, emb_by_id[sid]) if sid in emb_by_id else None

    monkeypatch.setattr(ps, "read_genome", fake_read)

    filtered_dir, unfiltered_dir = tmp_path / "filtered", tmp_path / "unfiltered"
    filtered_dir.mkdir()
    unfiltered_dir.mkdir()
    all_ids = [*ids, "EVAL0"]
    n = ps.build_panels(
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
