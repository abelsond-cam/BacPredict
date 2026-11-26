"""The MIC arms' plumbing: which cells each objective keeps, what the dataset hands the loss, and sigma's life."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from bacpredict.engine.finetune.datasets import IntervalInjectingFileDataset
from bacpredict.engine.finetune.mic_trainer import (
    INIT_LOG_SIGMA,
    SIGMA_FILE,
    MicEvalMetrics,
    censor_map,
    interval_map,
    load_interval_table,
)

INF = float("inf")


def _table() -> pd.DataFrame:
    """One cell of each kind: exact, right-censored, left-censored, and a bare R/S call with no MIC."""
    return pd.DataFrame([
        {"Sample": "E", "ast_label": 1, "lo": 1.0, "hi": 2.0, "point": 2.0, "censor": "exact", "basis": "mic"},
        {"Sample": "R", "ast_label": 1, "lo": 3.0, "hi": INF, "point": 4.0, "censor": "right", "basis": "mic"},
        {"Sample": "L", "ast_label": 0, "lo": -INF, "hi": -1.0, "point": -1.0, "censor": "left", "basis": "mic"},
        {"Sample": "B", "ast_label": 0, "lo": -INF, "hi": 0.0, "point": np.nan, "censor": "bare",
         "basis": "bare_call"},
        {"Sample": "X", "ast_label": 0.5, "lo": 1.0, "hi": 2.0, "point": 2.0, "censor": "exact", "basis": "mic"},
    ])


# ------------------------------------------------------------------ which cells each objective can use


def test_the_censored_objective_keeps_the_bare_calls_as_open_intervals():
    """A bare R/S call is the one-sided interval at the breakpoint, which this likelihood can use directly."""
    assert sorted(interval_map(_table(), "censored")) == ["B", "E", "L", "R"]


def test_the_least_squares_objective_drops_what_it_cannot_impute():
    """Least squares needs a number; a bare call has none, and inventing one would be the bias under test."""
    assert sorted(interval_map(_table(), "mse")) == ["E", "L", "R"]


def test_a_fractional_label_is_never_scored_by_any_arm():
    """``load_splits`` drops conflicting cells from the binary arm; the MIC arms must drop the same ones."""
    assert "X" not in interval_map(_table(), "censored")


def test_the_interval_map_carries_the_split_tables_label_unchanged():
    """Every arm's R/S metrics are computed against the same labels, or the arms are not comparable."""
    assert {s: v[0] for s, v in interval_map(_table(), "censored").items()} == {
        "E": 1.0, "R": 1.0, "L": 0.0, "B": 0.0}


def test_load_interval_table_refuses_a_table_missing_a_column(tmp_path):
    """A silently absent ``censor`` column would make every agreement rate NaN and look like a data problem."""
    path = tmp_path / "t.csv"
    _table().drop(columns=["censor"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="censor"):
        load_interval_table(path)


def test_censor_map_is_the_class_per_sample():
    assert censor_map(_table())["R"] == "right"


# ------------------------------------------------------------------ the dataset


def _store(tmp_path, ids):
    for s in ids:
        torch.save({"prot_embeddings": torch.randn(5, 8), "attention_mask": torch.ones(5),
                    "contig_idx": torch.zeros(5, dtype=torch.long)},
                   tmp_path / f"{s}_esm_embeddings.pt")
    return tmp_path


def test_the_dataset_hands_the_loss_four_numbers_per_genome(tmp_path):
    """``(ast_label, lo, hi, point)`` — anything not in ``labels`` never reaches ``compute_metrics``."""
    ids = ["E", "R", "B"]
    ds = IntervalInjectingFileDataset(ids, _store(tmp_path, ids), interval_map(_table(), "censored"), "meropenem")
    assert len(ds) == 3
    assert list(ds[0]["labels"]) == [1.0, 1.0, 2.0, 2.0]
    assert ds[1]["labels"][2] == INF              # the infinities survive the tensor
    assert math.isnan(ds[2]["labels"][3])         # a bare call's point stays missing, not 0


def test_the_dataset_refuses_a_genome_with_no_interval_row(tmp_path):
    """Silently serving it the previous genome's label is the failure this prevents."""
    with pytest.raises(KeyError, match="no MIC interval row"):
        IntervalInjectingFileDataset(["E", "GHOST"], _store(tmp_path, ["E", "GHOST"]),
                                     interval_map(_table(), "censored"), "meropenem")


def test_the_embedding_payload_is_untouched(tmp_path):
    """Only the label changes: a MIC arm reads the same store, the same way, as the binary arm."""
    ds = IntervalInjectingFileDataset(["E"], _store(tmp_path, ["E"]), interval_map(_table(), "censored"), "m")
    assert ds[0]["protein_embeddings"].shape == (1, 5, 8)
    assert ds[0]["attention_mask"].shape[-1] == 5 and ds[0]["contig_ids"].shape[-1] == 5


# ------------------------------------------------------------------ the metrics


class _Preds:
    def __init__(self, mu, labels):
        self.predictions, self.label_ids = np.asarray(mu), np.asarray(labels)


class _Trainer:
    def __init__(self, sigma):
        self.sigma = sigma


def test_mic_metrics_emit_the_keys_the_stopping_rule_needs():
    """``metric_for_best_model='eval_auroc'`` is unchanged for every arm, so the key must be there."""
    labels = np.array([[1, 1.0, 2.0, 2.0], [0, -INF, 0.0, np.nan], [1, 0.0, INF, 1.0], [0, -2.0, -1.0, -1.0]])
    out = MicEvalMetrics([_Trainer(1.3)])(_Preds([1.5, -1.0, 0.8, -1.4], labels))
    assert {"eval_auroc", "eval_auprc", "eval_balanced_accuracy", "eval_f1", "eval_accuracy",
            "eval_nr_samples"} <= set(out)
    assert {"eval_sigma", "eval_censored_nll"} <= set(out)
    assert out["eval_sigma"] == 1.3


def test_sigma_cannot_move_the_metric_the_checkpoint_is_chosen_by():
    """If it could, the validate re-fit would be choosing the model as well as calibrating it."""
    labels = np.array([[1, 1.0, 2.0, 2.0], [0, -INF, 0.0, np.nan], [1, 0.0, INF, 1.0], [0, -2.0, -1.0, -1.0]])
    preds = _Preds([1.5, -1.0, 0.8, -1.4], labels)
    a = MicEvalMetrics([_Trainer(0.5)])(preds)
    b = MicEvalMetrics([_Trainer(4.0)])(preds)
    for key in ("eval_auroc", "eval_auprc", "eval_balanced_accuracy", "eval_accuracy"):
        assert a[key] == pytest.approx(b[key]), key
    assert a["eval_censored_nll"] != pytest.approx(b["eval_censored_nll"])   # but the likelihood does move


# ------------------------------------------------------------------ sigma's life across a checkpoint


class _TinyModel(torch.nn.Module):
    """Enough of a Bacformer to exercise the trainer: the same call signature, one linear head."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(8, 1)
        self.config = type("C", (), {"problem_type": "binary_classification", "num_labels": 1})()

    def forward(self, protein_embeddings=None, labels=None, attention_mask=None, contig_ids=None, **_):
        return {"logits": self.head(protein_embeddings.float().mean(dim=1))}


def _trainer(tmp_path, objective="censored"):
    from transformers import TrainingArguments

    from bacpredict.engine.finetune.mic_trainer import MicBacformerTrainer
    args = TrainingArguments(output_dir=str(tmp_path / "out"), report_to=[], use_cpu=True,
                             per_device_train_batch_size=2, remove_unused_columns=False)
    return MicBacformerTrainer(model=_TinyModel(), args=args, objective=objective)


def test_sigma_starts_at_one_dilution(tmp_path):
    """The assay's own reproducibility: "the model explains nothing the assay could not have mismeasured"."""
    assert _trainer(tmp_path).sigma == pytest.approx(math.exp(INIT_LOG_SIGMA))


def test_sigma_round_trips_through_a_checkpoint(tmp_path):
    """It is saved beside the weights, not inside them — a key the base model lacks is dropped on load."""
    t = _trainer(tmp_path)
    with torch.no_grad():
        t.log_sigma.fill_(math.log(2.5))
    t.save_sigma(tmp_path / "ckpt")
    assert json.loads((tmp_path / "ckpt" / SIGMA_FILE).read_text())["sigma"] == pytest.approx(2.5)
    fresh = _trainer(tmp_path)
    assert fresh.load_sigma(tmp_path / "ckpt") == pytest.approx(2.5)
    assert fresh.sigma == pytest.approx(2.5)


def test_resuming_a_checkpoint_with_no_sigma_is_loud(tmp_path):
    """Silently restarting sigma mid-run would change the model being fitted, invisibly."""
    (tmp_path / "ckpt").mkdir()
    with pytest.raises(FileNotFoundError, match="carries no sigma"):
        _trainer(tmp_path).load_sigma(tmp_path / "ckpt")


def test_sigma_gets_its_own_optimizer_group(tmp_path):
    """At the network's 1.5e-4 one scalar would move ~0.5 in log-space over a whole run and never arrive."""
    t = _trainer(tmp_path, "censored")
    groups = t.create_optimizer().param_groups
    assert any(g["params"] and g["params"][0] is t.log_sigma and g["lr"] == t.sigma_lr for g in groups)


def test_the_least_squares_arm_has_no_sigma_to_learn(tmp_path):
    """Its sigma is fitted on validate at the end like every other arm's, not by the optimizer."""
    t = _trainer(tmp_path, "mse")
    assert not t.log_sigma.requires_grad
    assert all(not (g["params"] and g["params"][0] is t.log_sigma) for g in t.create_optimizer().param_groups)


def test_the_least_squares_arm_refuses_a_cell_with_no_point(tmp_path):
    """Imputing a point for a bare call is exactly the bias this comparison is testing for."""
    t = _trainer(tmp_path, "mse")
    inputs = {"protein_embeddings": torch.randn(2, 5, 8), "attention_mask": torch.ones(2, 5),
              "contig_ids": torch.zeros(2, 5, dtype=torch.long),
              "labels": torch.tensor([[1.0, 1.0, 2.0, 2.0], [0.0, -INF, 0.0, float("nan")]])}
    with pytest.raises(ValueError, match="no imputed point"):
        t.compute_loss(t.model, inputs)
