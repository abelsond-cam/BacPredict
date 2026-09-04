"""Tests for the Stage 2c comparison.

The module's job is to decide whether a two-point AUROC gap survives refit variance, so the tests are
mostly about it refusing to answer on inputs it cannot trust: an archive that cannot be checked against
the holdout, a run scored on the wrong genomes, labels that disagree between the two arms.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from kleb_iso_source.compare_kfold_sweep import (
    _main,
    compare,
    discover_runs,
    load_scores,
    load_unitig_scores,
)

N_HOLDOUT = 200


def _ids(n: int = N_HOLDOUT) -> list[str]:
    return [f"SAM{i:05d}" for i in range(n)]


def _manifest(ids: list[str]) -> dict:
    from kleb_iso_source.materialise_kfold_splits import _digest

    return {"n_holdout": len(ids), "holdout_digest": _digest(set(ids)),
            "runs": [{"fold": f, "seed": s} for s in (1, 2, 3) for f in range(5)]}


def _write_run(root, fold, seed, ids, y, prob, prefix="models_kfold"):
    d = root / f"{prefix}_fold{fold:02d}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "eval_scores.npz", sample_ids=np.asarray(ids, dtype=np.str_),
             y_true=np.asarray(y, dtype=int), y_prob=np.asarray(prob, dtype=float))
    return d


def _signal(rng, y, strength):
    """Probabilities whose AUROC rises with ``strength``."""
    return np.clip(rng.normal(0.5 + strength * (y - 0.5), 0.2), 0.001, 0.999)


@pytest.fixture
def cohort():
    rng = np.random.default_rng(0)
    ids = _ids()
    y = rng.integers(0, 2, N_HOLDOUT)
    return ids, y, rng


def test_discover_runs_orders_by_task_id_and_skips_unfinished(tmp_path, cohort):
    ids, y, rng = cohort
    for seed in (1, 2):
        for fold in range(5):
            _write_run(tmp_path, fold, seed, ids, y, _signal(rng, y, 0.5))
    (tmp_path / "models_kfold_fold00_seed3").mkdir()  # started, never finished
    runs = discover_runs(tmp_path, "models_kfold")
    assert [(f, s) for f, s, _ in runs] == [(f, s) for s in (1, 2) for f in range(5)]


def test_discover_runs_ignores_a_different_prefix(tmp_path, cohort):
    ids, y, rng = cohort
    _write_run(tmp_path, 0, 1, ids, y, _signal(rng, y, 0.5), prefix="models_bf16")
    assert discover_runs(tmp_path, "models_kfold") == []


def test_load_scores_refuses_an_archive_without_sample_ids(tmp_path):
    npz = tmp_path / "old.npz"
    np.savez(npz, y_true=np.zeros(3), y_prob=np.zeros(3))
    with pytest.raises(ValueError, match="no 'sample_ids'"):
        load_scores(npz)


def test_load_unitig_scores_requires_its_keys(tmp_path):
    npz = tmp_path / "u.npz"
    np.savez(npz, sample_ids=np.array(["a"], dtype=np.str_))
    with pytest.raises(ValueError, match="y_prob"):
        load_unitig_scores(npz)


def _sweep(tmp_path, cohort, bac_strength=0.9, uni_strength=0.6, n_seeds=3):
    ids, y, rng = cohort
    runs = []
    for seed in range(1, n_seeds + 1):
        for fold in range(5):
            d = _write_run(tmp_path, fold, seed, ids, y, _signal(rng, y, bac_strength))
            runs.append((fold, seed, d / "eval_scores.npz"))
    unitig = pd.DataFrame({"Sample": ids, "unitig_true": y,
                           "unitig_prob": _signal(rng, y, uni_strength)})
    return runs, unitig, _manifest(ids)


def test_all_fifteen_runs_are_scored_and_verified(tmp_path, cohort):
    runs, unitig, manifest = _sweep(tmp_path, cohort)
    df, summary = compare(runs, unitig, manifest, n_boot=100)
    assert len(df) == summary["n_runs_found"] == 15 == summary["n_runs_expected"]
    assert summary["all_holdouts_match_split_table"] is True
    assert df["n_holdout"].eq(N_HOLDOUT).all() and df["n_shared"].eq(N_HOLDOUT).all()


def test_the_sd_is_over_refits_not_within_one(tmp_path, cohort):
    runs, unitig, manifest = _sweep(tmp_path, cohort)
    _, summary = compare(runs, unitig, manifest, n_boot=100)
    vals = np.asarray(summary["bacformer"]["values"])
    assert summary["bacformer"]["mean"] == pytest.approx(vals.mean())
    # ddof=1 — the 15 fits are a sample of the refit distribution, not the whole of it.
    assert summary["bacformer"]["sd"] == pytest.approx(vals.std(ddof=1))
    assert summary["unitig"]["sd_by_construction"] == 0.0


def test_a_clear_bacformer_win_is_reported_as_surviving_refit_variance(tmp_path, cohort):
    runs, unitig, manifest = _sweep(tmp_path, cohort, bac_strength=1.4, uni_strength=0.3)
    df, summary = compare(runs, unitig, manifest, n_boot=200)
    assert summary["n_bacformer_fits_above_unitig"] == 15
    assert "survives refit variance" in summary["verdict"]
    assert (df["delta"] > 0).all()


def test_a_tie_is_not_dressed_up_as_a_win(tmp_path, cohort):
    """Same underlying signal in both arms: the sign of the delta must wobble across refits."""
    runs, unitig, manifest = _sweep(tmp_path, cohort, bac_strength=0.6, uni_strength=0.6)
    _, summary = compare(runs, unitig, manifest, n_boot=200)
    assert 0 < summary["n_bacformer_fits_above_unitig"] < 15
    assert "not stable across refits" in summary["verdict"]


def test_a_run_scored_on_the_wrong_genomes_is_flagged_not_averaged_in(tmp_path, cohort):
    ids, y, rng = cohort
    runs, unitig, manifest = _sweep(tmp_path, cohort, n_seeds=1)
    wrong = ids[:-1] + ["SAM99999"]
    unitig = pd.concat([unitig, pd.DataFrame({"Sample": ["SAM99999"], "unitig_true": [1],
                                              "unitig_prob": [0.5]})], ignore_index=True)
    d = _write_run(tmp_path, 0, 9, wrong, y, _signal(rng, y, 0.9))
    runs.append((0, 9, d / "eval_scores.npz"))
    df, summary = compare(runs, unitig, manifest, n_boot=50)
    assert summary["all_holdouts_match_split_table"] is False
    # n matches for the bad run — identity is what catches it, exactly as it had to.
    bad = df[df["seed"] == 9].iloc[0]
    assert bad["n_holdout"] == N_HOLDOUT and bad["holdout_matches_split_table"] is np.False_


def test_disagreeing_labels_between_the_arms_are_refused(tmp_path, cohort):
    runs, unitig, manifest = _sweep(tmp_path, cohort, n_seeds=1)
    unitig = unitig.assign(unitig_true=1 - unitig["unitig_true"])
    with pytest.raises(ValueError, match="labels disagree"):
        compare(runs, unitig, manifest, n_boot=50)


def test_genomes_the_unitig_model_cannot_score_shrink_the_shared_set_visibly(tmp_path, cohort):
    runs, unitig, manifest = _sweep(tmp_path, cohort, n_seeds=1)
    df, _ = compare(runs, unitig.iloc[:150], manifest, n_boot=50)
    assert df["n_holdout"].eq(N_HOLDOUT).all() and df["n_shared"].eq(150).all()
    # The full-holdout AUROC is still reported, so the shrinkage cannot be mistaken for a score change.
    assert not np.allclose(df["bacformer_auroc"], df["bacformer_auroc_full_holdout"])


def test_no_finished_runs_is_an_error_not_an_empty_summary(cohort):
    ids, y, _ = cohort
    with pytest.raises(ValueError, match="no finished runs"):
        compare([], pd.DataFrame(columns=["Sample", "unitig_true", "unitig_prob"]), _manifest(ids))


def test_cli_writes_both_artifacts_and_exits_on_a_holdout_mismatch(tmp_path, cohort):
    ids, y, rng = cohort
    sweep, models, out = tmp_path / "sweep", tmp_path / "m", tmp_path / "out"
    sweep.mkdir()
    (sweep / "kfold_splits_manifest.json").write_text(json.dumps(_manifest(ids)))
    for fold in range(5):
        _write_run(models, fold, 1, ids, y, _signal(rng, y, 1.2))
    unitig = tmp_path / "u.npz"
    np.savez(unitig, sample_ids=np.asarray(ids, dtype=np.str_), y_true=y,
             y_prob=_signal(rng, y, 0.4))
    argv = ["--sweep-dir", str(sweep), "--models-root", str(models), "--unitig-scores", str(unitig),
            "--out-dir", str(out), "--n-boot", "100"]
    _main(argv)
    payload = json.loads((out / "kfold_sweep_comparison.json").read_text())
    assert payload["summary"]["n_runs_found"] == 5
    assert len(pd.read_csv(out / "kfold_sweep_per_run.csv")) == 5

    _write_run(models, 0, 2, ids[:-1] + ["SAM99999"], y, _signal(rng, y, 1.2))
    np.savez(unitig, sample_ids=np.asarray(ids + ["SAM99999"], dtype=np.str_),
             y_true=np.append(y, 1), y_prob=np.append(_signal(rng, y, 0.4), 0.5))
    with pytest.raises(SystemExit, match="NOT scored on the materialised holdout"):
        _main(argv)
