"""Tests for resume-checkpoint selection.

The case that matters is the one HF's own get_last_checkpoint gets wrong: a directory left half
written by a wall-clock kill. On a chained run every later link would pick the same corpse.
"""

from __future__ import annotations

import logging

import pytest

from bacpredict.engine.finetune.checkpoints import (
    is_complete,
    list_checkpoints,
    pick_resume_checkpoint,
)


def _ckpt(root, step, complete=True):
    d = root / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / "model.safetensors").write_text("weights")
    if complete:
        (d / "trainer_state.json").write_text("{}")
    return d


def test_no_directory_and_no_checkpoints_both_mean_start_fresh(tmp_path):
    assert pick_resume_checkpoint(tmp_path / "never_created") is None
    (tmp_path / "empty").mkdir()
    assert pick_resume_checkpoint(tmp_path / "empty") is None


def test_the_newest_checkpoint_wins_by_step_not_by_name(tmp_path):
    """9000 sorts before 700 as a string; the step number is what orders them."""
    for step in (700, 1400, 9000):
        _ckpt(tmp_path, step)
    assert [p.name for p in list_checkpoints(tmp_path)] == [
        "checkpoint-9000", "checkpoint-1400", "checkpoint-700"]
    assert pick_resume_checkpoint(tmp_path).name == "checkpoint-9000"


def test_an_interrupted_save_is_stepped_over_not_resumed_from(tmp_path):
    _ckpt(tmp_path, 28000)
    partial = _ckpt(tmp_path, 30100, complete=False)
    assert is_complete(partial) is False
    assert pick_resume_checkpoint(tmp_path).name == "checkpoint-28000"


def test_stepping_over_an_interrupted_save_is_logged_by_name(tmp_path, caplog):
    _ckpt(tmp_path, 700)
    _ckpt(tmp_path, 1400, complete=False)
    with caplog.at_level(logging.WARNING):
        pick_resume_checkpoint(tmp_path)
    assert "checkpoint-1400" in caplog.text and "interrupted" in caplog.text


def test_every_checkpoint_incomplete_starts_fresh_loudly(tmp_path):
    _ckpt(tmp_path, 700, complete=False)
    assert pick_resume_checkpoint(tmp_path) is None


def test_unrelated_directories_and_files_are_ignored(tmp_path):
    _ckpt(tmp_path, 700)
    (tmp_path / "runs").mkdir()
    (tmp_path / "checkpoint-notanumber").mkdir()
    (tmp_path / "results.json").write_text("{}")
    (tmp_path / "checkpoint-999").write_text("a file, not a directory")
    assert [p.name for p in list_checkpoints(tmp_path)] == ["checkpoint-700"]


# ---------------------------------------------------------------------------
# Resuming past HF's problem_type validation
# ---------------------------------------------------------------------------


def test_hf_rejects_bacformers_problem_type_outside_the_block():
    """The bug this exists for. If transformers ever accepts it, this test fails and the shim goes."""
    from transformers import PretrainedConfig

    with pytest.raises(ValueError, match="problem_type"):
        PretrainedConfig(problem_type="binary_classification")


def test_inside_the_block_the_value_survives_verbatim():
    """Preserved, never dropped: problem_type=None would match no loss branch and train at zero grad."""
    from transformers import PretrainedConfig

    from bacpredict.engine.finetune.checkpoints import tolerate_custom_problem_type

    with tolerate_custom_problem_type():
        cfg = PretrainedConfig(problem_type="binary_classification", num_labels=1)
    assert cfg.problem_type == "binary_classification"
    assert cfg.num_labels == 1


def test_legal_values_still_take_the_untouched_code_path():
    from transformers import PretrainedConfig

    from bacpredict.engine.finetune.checkpoints import tolerate_custom_problem_type

    with tolerate_custom_problem_type():
        assert PretrainedConfig(problem_type="regression").problem_type == "regression"
        assert PretrainedConfig().problem_type is None


def test_the_patch_is_reverted_even_when_the_body_raises():
    from transformers import PretrainedConfig

    from bacpredict.engine.finetune.checkpoints import tolerate_custom_problem_type

    before = PretrainedConfig.__init__
    with pytest.raises(RuntimeError), tolerate_custom_problem_type():
        raise RuntimeError("boom")
    assert PretrainedConfig.__init__ is before
    with pytest.raises(ValueError, match="problem_type"):
        PretrainedConfig(problem_type="binary_classification")


def test_a_real_checkpoint_config_round_trips(tmp_path):
    """Exactly what Trainer._load_from_checkpoint does — read the file, then read its version."""
    import json

    from transformers import PretrainedConfig

    from bacpredict.engine.finetune.checkpoints import tolerate_custom_problem_type

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "architectures": ["BacformerLargeForGenomeClassification"],
        "model_type": "bacformer", "problem_type": "binary_classification",
        "num_labels": 1, "transformers_version": "4.57.6",
    }))
    with tolerate_custom_problem_type():
        cfg = PretrainedConfig.from_json_file(cfg_file)
    assert cfg.transformers_version == "4.57.6"
    assert cfg.problem_type == "binary_classification"
