"""``finetune_amr`` split-table mode, end to end on CPU with the model and the HF trainer replaced by fakes.

The fakes stand in only for Bacformer and ``Trainer.train``/``predict``: everything around them — the split
resolution through ``engine.splits.load_splits``, the datasets over real ``.pt`` files, the collate, the
post-training scoring, ``results.json`` and the score archives — is the trainer's own code. The same harness
pins the AST-sheet paths (k-fold, ``train_val_eval``, smoke), which ``--split-table`` must leave unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from bacpredict.engine.finetune import finetune_amr as ft
from bacpredict.engine.finetune.split_utils import generate_kfold_splits
from bacpredict.engine.splits.load_splits import load_split_ids, load_splits

# --------------------------------------------------------------------------- fakes


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(4, 1)
        self.config = SimpleNamespace(_commit_hash="fake-revision")


def _logit(sample_id: str, label: int) -> float:
    """A deterministic score that ranks R above S (AUROC 1), varied per sample."""
    return (2.0 * label - 1.0) * (1.0 + (sum(map(ord, sample_id)) % 7) / 10.0)


class _FakeTrainer:
    """Records what the trainer was handed; ``predict`` scores a dataset in its own order."""

    instances: list[_FakeTrainer] = []

    def __init__(self, *, model, data_collator, train_dataset, eval_dataset, args, compute_metrics, callbacks):
        self.model, self.collate = model, data_collator
        self.train_dataset, self.eval_dataset, self.args = train_dataset, eval_dataset, args
        self.predicted: list[list[str]] = []
        self.batch_dtypes: list[torch.dtype] = []
        self.calls: list[str] = []
        _FakeTrainer.instances.append(self)

    def train(self, resume_from_checkpoint=None):
        self.calls.append("train")
        n = min(len(self.train_dataset), 4)
        batch = self.collate([self.train_dataset[i] for i in range(n)])
        self.batch_dtypes.append(batch["protein_embeddings"].dtype)

    def save_state(self):
        self.calls.append("save_state")

    def predict(self, dataset):
        self.calls.append("predict")
        labels = [int(dataset[i]["labels"].item()) for i in range(len(dataset))]
        self.predicted.append(list(dataset.sample_ids))
        logits = [[_logit(s, y)] for s, y in zip(dataset.sample_ids, labels, strict=True)]
        return SimpleNamespace(predictions=np.asarray(logits, dtype=np.float32),
                               label_ids=np.asarray(labels, dtype=np.float32))


@pytest.fixture
def fakes(monkeypatch):
    _FakeTrainer.instances = []
    loads: list[str] = []
    load_kwargs: list[dict] = []

    def _from_pretrained(path, **kwargs):
        loads.append(path)
        load_kwargs.append(kwargs)
        return _FakeModel()

    monkeypatch.setattr(ft.AutoModelForSequenceClassification, "from_pretrained", _from_pretrained)
    monkeypatch.setattr(ft, "BacformerLargeTrainer", _FakeTrainer)
    return SimpleNamespace(loads=loads, load_kwargs=load_kwargs, trainers=_FakeTrainer.instances)


# --------------------------------------------------------------------------- data


def _write_pt(store: Path, sample: str, dtype: torch.dtype, n_proteins: int = 3) -> None:
    emb = (torch.arange(n_proteins * 4, dtype=torch.float32).reshape(1, n_proteins, 4) / 8).to(dtype)
    torch.save({"protein_embeddings": emb, "attention_mask": torch.ones(1, n_proteins),
                "contig_ids": torch.zeros(1, n_proteins, dtype=torch.long)}, store / f"{sample}_esm_embeddings.pt")


def _split_table(tmp_path: Path, species: tuple[str, ...] = ("paeruginosa",), n_per_split: int = 12) -> Path:
    """A ``<drug>_split.csv`` to the contract: every split, a fractional label, ``.pt`` for every row.

    Labels alternate within each split so every split holds both classes. Half the genomes are stored bf16 (the
    AMR store) and half fp32 (symlinked Kp files), as a cross-species store will be.
    """
    store = tmp_path / "esm"
    store.mkdir()
    rows = []
    for sp in species:
        for split in ("train", "validate", "holdout", "holdout_st"):
            for i in range(n_per_split):
                sample = f"{sp[:2].upper()}_{split}_{i:03d}"
                rows.append({"Sample": sample, "species": sp, "st_key": f"{sp}:{235 if split == 'holdout_st' else i}",
                             "ast_label": i % 2, "split": split})
    rows.append({"Sample": "FRACTIONAL", "species": species[0], "st_key": f"{species[0]}:1",
                 "ast_label": 0.5, "split": "train"})
    table = pd.DataFrame(rows)
    for k, sample in enumerate(table["Sample"]):
        _write_pt(store, sample, torch.bfloat16 if k % 2 else torch.float32)
    path = tmp_path / "ciprofloxacin_split.csv"
    table.to_csv(path, index=False)
    return path


def _run(tmp_path: Path, split_table: Path | None, **kw) -> Path:
    out = tmp_path / "run"
    ft.run("base-model", str(tmp_path / "esm"), str(out), kw.pop("ast_sheet_path", None), task=kw.pop("task", "pa_ast"),
           drug="ciprofloxacin", num_workers=0, split_table=None if split_table is None else str(split_table), **kw)
    return out


# --------------------------------------------------------------------------- split-table mode


def test_split_table_run_trains_on_the_tables_splits_and_scores_every_holdout(tmp_path, fakes):
    table = _split_table(tmp_path)
    out = _run(tmp_path, table)
    label_map, train, validate, holdout = load_splits(table)
    holdout_st = load_split_ids(table, "holdout_st")

    trainer = fakes.trainers[0]
    assert trainer.train_dataset.sample_ids == train and trainer.eval_dataset.sample_ids == validate
    assert "FRACTIONAL" not in train  # a conflicting label never trains
    assert trainer.predicted == [holdout, holdout_st]  # the best model scores both holdouts, holdout first
    # the whole run's history (<run>/trainer_state.json, which Stage B reads), saved once training has ended
    assert trainer.calls == ["train", "save_state", "predict", "predict"]
    assert trainer.batch_dtypes == [torch.float32]  # bf16 and fp32 .pt reach the model as one dtype

    res = json.loads((out / "results.json").read_text())
    assert res["task"] == "pa_ast" and res["split"]["source"] == "split_table"
    assert res["split"]["n_evaluate"] == len(holdout) and res["split"]["n_train"] == len(train)
    assert res["split"]["table"] == str(table) and len(res["split"]["table_md5"]) == 32
    assert list(res["holdouts"]) == ["holdout", "holdout_st"]
    assert res["metrics"] == res["holdouts"]["holdout"]["metrics"]
    for name, ids in (("holdout", holdout), ("holdout_st", holdout_st)):
        block = res["holdouts"][name]
        m = block["metrics"]
        assert block["n_samples"] == len(ids) == m["n_samples"]
        assert {"auroc", "auprc", "sensitivity", "specificity", "balanced_accuracy", "confusion_matrix",
                "calibration", "per_class"} <= set(m)
        assert m["auroc"] == pytest.approx(1.0)
        assert m["per_class"]["1"]["n"] + m["per_class"]["0"]["n"] == len(ids)
        assert "by_species" not in block  # one species in the table
        npz = np.load(out / block["scores"])
        assert list(npz["sample_ids"]) == ids
        assert list(npz["y_true"]) == [label_map[s] for s in ids]
        assert set(npz["split"]) == {name} and set(npz["species"]) == {"paeruginosa"}
    assert res["holdouts"]["holdout"]["scores"] == "eval_scores.npz"
    assert res["holdouts"]["holdout_st"]["scores"] == "eval_scores_holdout_st.npz"
    assert res["species"] == ["paeruginosa"]


def test_cross_species_table_adds_per_species_metrics_to_every_holdout(tmp_path, fakes):
    table = _split_table(tmp_path, species=("paeruginosa", "enterobacterales"))
    out = _run(tmp_path, table, task="pa_entero_ast")
    res = json.loads((out / "results.json").read_text())
    assert res["task"] == "pa_entero_ast" and res["species"] == ["enterobacterales", "paeruginosa"]
    for name in ("holdout", "holdout_st"):
        by = res["holdouts"][name]["by_species"]
        assert set(by) == {"enterobacterales", "paeruginosa"}
        assert sum(b["n_samples"] for b in by.values()) == res["holdouts"][name]["n_samples"]
        assert all({"auroc", "per_class", "confusion_matrix"} <= set(b) for b in by.values())


def test_smoke_with_a_split_table_is_train_equals_eval_on_a_balanced_ten(tmp_path, fakes):
    table = _split_table(tmp_path)
    out = _run(tmp_path, table, n_samples=10)
    trainer = fakes.trainers[0]
    train = trainer.train_dataset.sample_ids
    label_map = load_splits(table)[0]
    assert len(train) == 10 and sum(label_map[s] for s in train) == 5  # both classes: AUROC is defined
    assert trainer.eval_dataset.sample_ids == train and trainer.predicted[0] == train
    assert len(trainer.predicted[1]) == 10 and set(trainer.predicted[1]) <= set(load_split_ids(table, "holdout_st"))
    res = json.loads((out / "results.json").read_text())
    assert res["split"]["source"] == "smoke" and res["holdouts"]["holdout_st"]["n_samples"] == 10
    assert trainer.calls[:2] == ["train", "save_state"]  # the overfit check reads that history


def test_a_missing_embedding_fails_before_the_model_loads(tmp_path, fakes):
    table = _split_table(tmp_path)
    victim = load_split_ids(table, "holdout_st")[-1]
    (tmp_path / "esm" / f"{victim}_esm_embeddings.pt").unlink()
    with pytest.raises(FileNotFoundError, match=victim):
        _run(tmp_path, table)
    assert fakes.loads == []


def test_a_dangling_symlink_counts_as_missing(tmp_path):
    (tmp_path / "S1_esm_embeddings.pt").symlink_to(tmp_path / "gone.pt")
    assert ft.missing_embeddings(["S1"], tmp_path) == ["S1"]


def test_an_empty_extra_holdout_is_recorded_with_n_zero(tmp_path, fakes):
    table = _split_table(tmp_path)
    df = pd.read_csv(table)
    df = df[df["split"] != "holdout_st"]
    df.to_csv(table, index=False)
    out = _run(tmp_path, table)
    res = json.loads((out / "results.json").read_text())
    assert res["holdouts"]["holdout_st"]["n_samples"] == 0
    assert len(np.load(out / "eval_scores_holdout_st.npz")["sample_ids"]) == 0
    assert len(fakes.trainers[0].predicted) == 1


def test_no_extra_holdouts_scores_the_holdout_only(tmp_path, fakes):
    out = _run(tmp_path, _split_table(tmp_path), extra_holdouts=())
    assert list(json.loads((out / "results.json").read_text())["holdouts"]) == ["holdout"]
    assert not (out / "eval_scores_holdout_st.npz").exists()


@pytest.mark.parametrize("extra", [["holdout"], ["validate"], ["holdout-st"], ["holdout_st", "holdout_st"]])
def test_bad_extra_holdouts_are_refused(tmp_path, extra):
    with pytest.raises(ValueError):
        ft.resolve_split_table(_split_table(tmp_path), extra)


def test_split_table_refuses_n_folds(tmp_path):
    with pytest.raises(ValueError, match="n-folds"):
        _run(tmp_path, _split_table(tmp_path), n_folds=5)


def test_predict_order_mismatch_is_caught():
    class _Reversed:
        def predict(self, dataset):
            return SimpleNamespace(predictions=np.asarray([[0.1], [0.9]]), label_ids=np.asarray([0.0, 1.0]))

    with pytest.raises(RuntimeError, match="in order"):
        ft.predict_holdout(_Reversed(), None, ["R", "S"], {"R": 1, "S": 0})


# --------------------------------------------------------------------------- CLI path resolution


def test_task_paths_tb_kp_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("BACPREDICT_DATA_ROOT", str(tmp_path))
    for task, folder in (("tb_ast", "train_tb_ast"), ("kleb_ast", "train_kleb_ast")):
        root = tmp_path / "processed" / folder
        assert ft.resolve_task_paths(task, None, None, None) == (str(root / "esm"),
                                                                 str(root / "binary_ast_with_split.csv"))
        assert ft.resolve_task_paths(task, "E", "A", None) == ("E", "A")


def test_task_paths_pa_needs_a_split_table_and_defaults_to_the_flat_amr_store(monkeypatch, tmp_path):
    monkeypatch.setenv("BACPREDICT_AMR_ROOT", str(tmp_path))
    with pytest.raises(SystemExit, match="split table"):
        ft.resolve_task_paths("pa_ast", None, None, None)
    assert ft.resolve_task_paths("pa_ast", None, None, "t.csv") == (str(tmp_path / "processed" / "esm"), None)
    with pytest.raises(SystemExit, match="not both"):
        ft.resolve_task_paths("pa_ast", None, "sheet.csv", "t.csv")


def test_task_paths_a_cross_species_label_needs_both_paths():
    for emb, table in ((None, "t.csv"), ("E", None), (None, None)):
        with pytest.raises(SystemExit, match="cross-species"):
            ft.resolve_task_paths("pa_entero_ast", emb, None, table)
    assert ft.resolve_task_paths("pa_entero_ast", "E", None, "t.csv") == ("E", None)


def test_cli_parses_split_table_and_extra_holdouts():
    args = ft.ArgumentParser().parse_args(["--task", "pa_ast", "--split-table", "t.csv"])
    assert args.split_table == "t.csv" and args.extra_holdouts == ["holdout_st"]
    args = ft.ArgumentParser().parse_args(["--split-table", "t.csv", "--extra-holdouts"])
    assert args.extra_holdouts == []


# --------------------------------------------------------------------------- AST-sheet paths unchanged


def _ast_sheet(tmp_path: Path, n: int = 40) -> Path:
    store = tmp_path / "esm"
    store.mkdir(exist_ok=True)
    samples = [f"K{i:03d}" for i in range(n)]
    for s in samples:
        _write_pt(store, s, torch.float32)
    df = pd.DataFrame({"Sample": samples, "ciprofloxacin": [i % 2 for i in range(n)],
                       "train_val_eval": (["train"] * 28 + ["validate"] * 4 + ["evaluate"] * 8)[:n]})
    path = tmp_path / "binary_ast_with_split.csv"
    df.to_csv(path, index=False)
    return path


def test_kfold_path_unchanged(tmp_path, fakes):
    sheet = _ast_sheet(tmp_path)
    out = _run(tmp_path, None, ast_sheet_path=str(sheet), task="kleb_ast", n_folds=5)
    df = pd.read_csv(sheet).astype({"Sample": str})
    evaluate, folds = generate_kfold_splits(df, n_folds=5, seed=1, evaluate_seed=1)
    order = df["Sample"].tolist()
    trainer = fakes.trainers[0]
    assert trainer.train_dataset.sample_ids == [s for s in order if s in folds[0][0]]
    assert trainer.eval_dataset.sample_ids == [s for s in order if s in folds[0][1]]
    assert trainer.predicted == [[s for s in order if s in evaluate]]
    run_dir = Path(f"{out}_fold00_seed1")
    res = json.loads((run_dir / "results.json").read_text())
    assert res["split"] == {"source": "kfold", "evaluate_seed": 1, "n_folds": 5, "fold": 0,
                            "n_evaluate": len(evaluate)}
    assert "holdouts" not in res and "species" not in res and "per_class" not in res["metrics"]
    assert set(res["run"]) == {"precision", "pooling", "seed", "lr", "early_stopping_patience",
                               "early_stopping_threshold", "eval_steps", "warmup_proportion", "max_steps",
                               "model_revision_requested", "freeze_layernorm_gains",
                               "n_frozen_layernorm_gains"}
    # Unset by default: an unpinned run must not look pinned, and a run that froze nothing must say so.
    assert res["run"]["model_revision_requested"] is None
    assert res["run"]["freeze_layernorm_gains"] is False
    assert res["run"]["n_frozen_layernorm_gains"] == 0
    assert not list(run_dir.glob("eval_scores*.npz"))
    assert "save_state" not in trainer.calls  # AST-sheet runs write nothing new


def test_csv_and_smoke_paths_unchanged(tmp_path, fakes):
    sheet = _ast_sheet(tmp_path)
    df = pd.read_csv(sheet).astype({"Sample": str})
    _run(tmp_path, None, ast_sheet_path=str(sheet), task="tb_ast")
    trainer = fakes.trainers[-1]
    assert trainer.train_dataset.sample_ids == df.loc[df.train_val_eval == "train", "Sample"].tolist()
    assert trainer.predicted == [df.loc[df.train_val_eval == "evaluate", "Sample"].tolist()]
    assert "save_state" not in trainer.calls

    out = _run(tmp_path, None, ast_sheet_path=str(sheet), task="tb_ast", n_samples=10)
    trainer = fakes.trainers[-1]
    assert trainer.train_dataset.sample_ids == df.loc[df.train_val_eval == "train", "Sample"].tolist()[:10]
    assert "save_state" not in trainer.calls
    assert json.loads((out / "results.json").read_text())["split"]["source"] == "smoke"


# ------------------------------------------------- precision-ablation knobs (rifampicin three-arm experiment)


class _ToyBackboneModel(torch.nn.Module):
    """A model shaped like Bacformer's: a ``bacformer`` backbone with LayerNorms, plus a classifier head."""

    def __init__(self) -> None:
        super().__init__()
        self.bacformer = torch.nn.Module()
        self.bacformer.embeddings = torch.nn.LayerNorm(4)
        self.bacformer.encoder = torch.nn.Module()
        self.bacformer.encoder.final_ln = torch.nn.LayerNorm(4)
        self.bacformer.encoder.ffn = torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 4))
        self.classifier = torch.nn.Module()
        self.classifier.norm = torch.nn.LayerNorm(4)


def test_freeze_backbone_layernorm_gains_scope():
    """Backbone gains freeze; their biases, backbone matrices and the classifier head keep training."""
    model = _ToyBackboneModel()
    assert ft.freeze_backbone_layernorm_gains(model) == 3  # embeddings, final_ln, ffn.0 — not classifier.norm

    assert model.bacformer.embeddings.weight.requires_grad is False
    assert model.bacformer.encoder.final_ln.weight.requires_grad is False
    assert model.bacformer.encoder.ffn[0].weight.requires_grad is False
    # Biases are not gains, and were seen to move under bf16 — they must still train.
    assert model.bacformer.embeddings.bias.requires_grad is True
    assert model.bacformer.encoder.ffn[0].bias.requires_grad is True
    # A backbone matrix weight is not a LayerNorm gain.
    assert model.bacformer.encoder.ffn[1].weight.requires_grad is True
    # The freshly-initialised head is out of scope — pinning it would be a different intervention.
    assert model.classifier.norm.weight.requires_grad is True


def test_freeze_backbone_layernorm_gains_refuses_silent_zero():
    """A model with no backbone LayerNorm must raise, not quietly freeze nothing and train as the control."""
    with pytest.raises(RuntimeError, match="no backbone LayerNorm"):
        ft.freeze_backbone_layernorm_gains(_FakeModel())


def test_model_revision_is_pinned_on_a_hub_load(tmp_path, fakes):
    """The requested revision reaches from_pretrained and is recorded as requested in results.json."""
    table = _split_table(tmp_path)
    out = _run(tmp_path, table, model_revision="fake")  # _FakeModel resolves to "fake-revision"
    assert fakes.load_kwargs[0]["revision"] == "fake"
    res = json.loads((Path(out) / "results.json").read_text())
    assert res["run"]["model_revision_requested"] == "fake"
    assert res["model"]["revision"] == "fake-revision"


def test_model_revision_mismatch_refuses_to_train(tmp_path, fakes):
    """A pin that did not take must stop the run — a results file naming the wrong base looks authoritative."""
    table = _split_table(tmp_path)
    with pytest.raises(RuntimeError, match="Refusing to train"):
        _run(tmp_path, table, model_revision="deadbeef")
