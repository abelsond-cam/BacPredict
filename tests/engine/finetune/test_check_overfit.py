"""Stage B reader (``engine/finetune/check_overfit``): judges the whole run's history, never a truncated one."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bacpredict.engine.finetune.check_overfit import main, stage_b


def _run(tmp_path: Path, *, losses: dict[int, float], global_step: int = 100, max_steps: int = 100,
         holdout_auroc: float = 1.0, logging_steps: int = 10) -> Path:
    """A smoke run directory as finetune_amr --split-table leaves it (top-level trainer_state.json + results.json)."""
    run = tmp_path / "run"
    run.mkdir()
    history = [{"step": s, "loss": v} for s, v in losses.items()]
    history += [{"step": s, "eval_auroc": 1.0} for s in range(2, global_step + 1, 2)]
    history.append({"step": global_step, "train_loss": 0.3, "train_runtime": 1.0})
    state = {"global_step": global_step, "max_steps": max_steps, "logging_steps": logging_steps,
             "log_history": history}
    (run / "trainer_state.json").write_text(json.dumps(state))
    (run / "results.json").write_text(json.dumps({"holdouts": {"holdout": {"metrics": {"auroc": holdout_auroc}}}}))
    return run


def test_a_whole_run_that_overfits_passes(tmp_path, capsys):
    run = _run(tmp_path, losses={1: 0.69, 10: 0.4, 50: 0.05, 100: 0.01})
    ok, lines = stage_b(run)
    assert ok and lines[-1].startswith("Stage B (PASS)")
    assert "last 0.0100 (step 100 of 100" in lines[0]
    assert main([str(run)]) == 0 and "PASS" in capsys.readouterr().out


def test_the_last_logged_loss_is_judged_not_the_minimum(tmp_path):
    ok, lines = stage_b(_run(tmp_path, losses={1: 0.69, 50: 0.01, 100: 0.5}))
    assert not ok and any("last train loss 0.5000" in line for line in lines)


def test_an_early_stopped_run_fails_even_with_a_low_loss(tmp_path):
    ok, lines = stage_b(_run(tmp_path, losses={1: 0.69, 10: 0.2, 20: 0.05}, global_step=22))
    assert not ok and any("stopped at step 22 of 100" in line for line in lines)


def test_a_history_that_stops_short_of_the_run_fails(tmp_path):
    # e.g. a best checkpoint's trainer_state.json copied up: its history ends at the best step, not the run's last
    ok, lines = stage_b(_run(tmp_path, losses={1: 0.69}))
    assert not ok and any("logged loss ends at step 1, the run at 100" in line for line in lines)


def test_low_holdout_auroc_fails(tmp_path):
    ok, lines = stage_b(_run(tmp_path, losses={1: 0.69, 100: 0.01}, holdout_auroc=0.8))
    assert not ok and any("holdout AUROC 0.800 < 0.99" in line for line in lines)


def test_a_run_without_a_top_level_trainer_state_is_refused(tmp_path, capsys):
    run = _run(tmp_path, losses={1: 0.69, 100: 0.01})
    ckpt = run / "checkpoint-4"
    ckpt.mkdir()
    (run / "trainer_state.json").rename(ckpt / "trainer_state.json")  # only the surviving best checkpoint's
    with pytest.raises(FileNotFoundError, match="trainer_state.json"):
        stage_b(run)
    assert main([str(run)]) == 1 and "never read" in capsys.readouterr().out
