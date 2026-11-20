"""Fine-tune Bacformer for AMR — the single trainer shared by every organism (TB, Kp, …).

Labels are injected at load time from the split CSV; no pre-built per-experiment ``.pt`` copies are
required. Pass ``--embeddings-dir`` at the organism's ESM-C store and ``--task`` (e.g. ``tb_ast`` /
``kleb_ast``) to label ``results.json``. The model trains in **bf16** master weights (the deployed AST
setting) — do Stage-A smokes on GPU. Drug column names are organism-specific (TB uses US spellings,
e.g. ``rifampin``).

**Split-table mode** (``--split-table <drug>_split.csv``, the multi-species AMR work). Train / validate /
holdout and the labels all come from the one table, read through :mod:`bacpredict.engine.splits` alone
(no AST sheet, no k-fold, no ``train_val_eval``). Each ``--extra-holdouts`` split (default ``holdout_st``, the
zero-shot held-out ST) is scored with the same best model as the holdout. ``results.json`` keeps the holdout's
metrics in ``metrics`` and adds a ``holdouts`` block with every holdout's metrics, per-class counts and, when the
table holds more than one species, per-species metrics. Per-sample scores go to ``eval_scores.npz`` (holdout)
and ``eval_scores_<split>.npz`` (each extra holdout). Without ``--split-table`` the trainer is unchanged.
"""
import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from bacformer.modeling.trainer import BacformerLargeTrainer, _extract_loss_and_logits
from tap import Tap
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    TrainingArguments,
)

from bacpredict.engine.config import organism
from bacpredict.engine.finetune.attention_pool import BacformerAttnPoolForGenomeClassification
from bacpredict.engine.finetune.datasets import LabelInjectingFileDataset, PanelInjectingFileDataset
from bacpredict.engine.finetune.metrics import (
    build_results_payload,
    compute_full_metrics,
    compute_metrics_binary_genome_pred,
    per_class_metrics,
    write_results_json,
)
from bacpredict.engine.finetune.split_utils import generate_kfold_splits
from bacpredict.engine.splits.load_splits import load_split_ids, load_split_species, load_splits
from bacpredict.engine.splits.subsample import subsample_balanced

# Maps the --task label to its OrganismConfig key, so store paths default off the chosen task.
_TASK_TO_ORGANISM = {"tb_ast": "tb", "kleb_ast": "kp", "pa_ast": "pa"}
# Tasks that train only from a split table: their labels hold fractional (conflicting) cells and genomes with no
# embedding, which the AST-sheet paths would cast to 0 with int() or fail on mid-run (PROJECT_STATE §3.6 Caveats).
_SPLIT_TABLE_ONLY_TASKS = {"pa_ast"}
# The extra holdouts scored by default in split-table mode: the zero-shot held-out ST.
DEFAULT_EXTRA_HOLDOUTS = ("holdout_st",)
_HOLDOUT = "holdout"


############################################################## Split-table mode ##############################################################


@dataclass
class SplitTableSplits:
    """What one ``<drug>_split.csv`` gives the trainer: labels, the three training splits, the extra holdouts."""

    path: str
    md5: str
    label_map: dict[str, int]
    train: list[str]
    validate: list[str]
    holdout: list[str]
    extra: dict[str, list[str]] = field(default_factory=dict)
    species: dict[str, str] = field(default_factory=dict)
    smoke: bool = False

    @property
    def holdouts(self) -> dict[str, list[str]]:
        """Every split scored after training, ``holdout`` first."""
        return {_HOLDOUT: self.holdout, **self.extra}

    @property
    def multi_species(self) -> bool:
        """True when the table holds more than one species (then every holdout is also scored per species)."""
        return len(set(self.species.values())) > 1


def _md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_split_table(
    split_table: str | Path,
    extra_holdouts: tuple[str, ...] | list[str] = DEFAULT_EXTRA_HOLDOUTS,
    *,
    n_samples: int = 10000,
    seed: int = 1,
) -> SplitTableSplits:
    """Read a drug's split table into the trainer's splits, through ``engine.splits`` only.

    ``load_splits`` gives the labels and train/validate/holdout, ``load_split_ids`` each extra holdout and
    ``load_split_species`` the species column: one read, one row filter (clean 0/1 labels, first row per Sample).

    ``n_samples == 10`` is the Stage A/B smoke: a class-balanced 10 of ``train`` (seeded), used as train, validate
    and holdout alike (the overfit check), and a class-balanced 10 of each extra holdout so its scoring path runs.
    Otherwise train, validate and holdout must be non-empty; an extra holdout may be empty (a drug no genome of the
    held-out ST is labelled for) and is then recorded with ``n_samples == 0``.
    """
    split_table = str(split_table)
    names = list(extra_holdouts)
    if len(set(names)) != len(names):
        raise ValueError(f"--extra-holdouts names a split twice: {names}")
    reserved = {"train", "validate", _HOLDOUT} & set(names)
    if reserved:
        raise ValueError(f"--extra-holdouts must name splits besides train/validate/holdout, got {sorted(reserved)}")

    label_map, train, validate, holdout = load_splits(split_table)
    extra = {name: load_split_ids(split_table, name) for name in names}
    species = load_split_species(split_table)

    smoke = n_samples == 10
    if smoke:
        train = subsample_balanced(train, label_map, max_n=10, seed=seed)
        validate = holdout = train
        extra = {name: subsample_balanced(ids, label_map, max_n=10, seed=seed) for name, ids in extra.items()}
        if not train:
            raise RuntimeError(f"{split_table}: no train rows with a 0/1 label")
    else:
        empty = [name for name, ids in (("train", train), ("validate", validate), (_HOLDOUT, holdout)) if not ids]
        if empty:
            raise RuntimeError(f"{split_table}: no rows with a 0/1 label in {empty}")
    return SplitTableSplits(path=split_table, md5=_md5(split_table), label_map=label_map, train=train,
                            validate=validate, holdout=holdout, extra=extra, species=species, smoke=smoke)


def missing_embeddings(sample_ids: list[str], embeddings_dir: str | Path) -> list[str]:
    """Samples with no ``<Sample>_esm_embeddings.pt`` on disk (a dangling symlink counts as missing)."""
    root = Path(embeddings_dir)
    return [s for s in dict.fromkeys(sample_ids) if not (root / f"{s}_esm_embeddings.pt").exists()]


def holdout_report(
    sample_ids: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    species: dict[str, str],
    by_species: bool,
    scores_file: str,
) -> dict:
    """One holdout's block in ``results.json``: §0.4 metrics + per-class, and per species when asked."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).astype(float).ravel()

    def _metrics(yt: np.ndarray, yp: np.ndarray) -> dict:
        return {**compute_full_metrics(yt, yp), "per_class": per_class_metrics(yt, yp)}

    block: dict = {"n_samples": len(sample_ids), "scores": scores_file, "metrics": _metrics(y_true, y_prob)}
    if by_species:
        sp = np.asarray([species.get(s, "") for s in sample_ids], dtype=object)
        block["by_species"] = {name: _metrics(y_true[sp == name], y_prob[sp == name]) for name in sorted(set(sp))}
    return block


def write_scores(
    path: str | Path,
    sample_ids: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    drug: str,
    split: str,
    species: dict[str, str],
) -> None:
    """Per-sample scores of one holdout, in the ``eval_scores.npz`` layout (row i is ``sample_ids[i]``).

    Adds ``split`` (per row, as ``score_cohort`` does) and ``species`` (``""`` when the table has none). There is
    no tuned threshold here, so ``operating_threshold`` is NaN.
    """
    np.savez(
        path,
        y_true=np.asarray(y_true).astype(int),
        y_prob=np.asarray(y_prob).astype(float),
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        drug=np.array(drug),
        split=np.asarray([split] * len(sample_ids), dtype=np.str_),
        species=np.asarray([species.get(s, "") for s in sample_ids], dtype=np.str_),
        operating_threshold=np.array(np.nan),
    )


def scores_filename(split: str) -> str:
    """``eval_scores.npz`` for the holdout (what every existing reader looks for), ``eval_scores_<split>.npz`` else."""
    return "eval_scores.npz" if split == _HOLDOUT else f"eval_scores_{split}.npz"


def predict_holdout(
    trainer, dataset, sample_ids: list[str], label_map: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """``(y_true, y_prob)`` for ``sample_ids`` in order, from ``trainer.predict`` on their dataset.

    Row i must be ``sample_ids[i]``: the scores file pairs them by position. So nothing is dropped (the trainer
    path's ``-100`` filter would shift every later row) and the predicted labels must equal the table's, in order.
    """
    preds = trainer.predict(dataset)
    logits = torch.as_tensor(preds.predictions).flatten()
    labels = torch.as_tensor(preds.label_ids).flatten().long()
    if logits.numel() != len(sample_ids) or labels.numel() != len(sample_ids):
        raise RuntimeError(f"predict returned {logits.numel()} scores and {labels.numel()} labels for "
                           f"{len(sample_ids)} samples")
    y_true = labels.cpu().numpy().astype(int)
    expected = np.asarray([label_map[s] for s in sample_ids], dtype=int)
    if not np.array_equal(y_true, expected):
        raise RuntimeError("predict's labels are not the split table's, in order: scores would mis-pair with samples")
    return y_true, torch.sigmoid(logits.float()).cpu().numpy().astype(float)


def write_split_table_results(
    trainer,
    make_dataset,
    splits: SplitTableSplits,
    *,
    output_dir: str,
    task: str,
    drug: str,
    model_name_or_path: str,
    model_revision: str | None,
    split_source: str,
    n_train: int,
    n_validate: int,
    run_config: dict,
) -> dict:
    """Score every holdout of a split table with the trained (best) model; write the npz files and ``results.json``.

    ``metrics`` is the holdout's, as in every other ``results.json``; ``holdouts`` holds each holdout's block
    (:func:`holdout_report`), ``holdout`` first. Returns the payload.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    blocks: dict[str, dict] = {}
    for name, ids in splits.holdouts.items():
        scores_file = scores_filename(name)
        if ids:
            print(f"Running post-training evaluation on {name}: {len(ids)} samples.")
            y_true, y_prob = predict_holdout(trainer, make_dataset(ids), ids, splits.label_map)
        else:
            print(f"No {name} samples for {drug}: recorded with n_samples 0.")
            y_true, y_prob = np.zeros(0, dtype=int), np.zeros(0, dtype=float)
        write_scores(out / scores_file, ids, y_true, y_prob, drug=drug, split=name, species=splits.species)
        blocks[name] = holdout_report(ids, y_true, y_prob, splits.species, splits.multi_species, scores_file)

    payload = build_results_payload(
        task=task,
        drug=drug,
        model_name_or_path=model_name_or_path,
        checkpoint_dir=str(out.resolve()),
        split_source=split_source,
        metrics=blocks[_HOLDOUT]["metrics"],
        n_evaluate=len(splits.holdout),
        model_revision=model_revision,
        run_config=run_config,
        versions={"torch": torch.__version__, "transformers": transformers.__version__},
    )
    payload["split"].update({
        "table": splits.path,
        "table_md5": splits.md5,
        "n_train": n_train,
        "n_validate": n_validate,
        "extra_holdouts": list(splits.extra),
    })
    if splits.species:
        payload["species"] = sorted(set(splits.species.values()))
    payload["holdouts"] = blocks
    results_path = out / "results.json"
    write_results_json(results_path, payload)
    print(f"Wrote results JSON: {results_path}")
    for name, block in blocks.items():
        m = block["metrics"]
        print(f"  {name}: AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} sens={m['sensitivity']:.4f} "
              f"spec={m['specificity']:.4f} bal_acc={m['balanced_accuracy']:.4f} n={m['n_samples']}")
        for sp, sm in block.get("by_species", {}).items():
            print(f"    {sp}: AUROC={sm['auroc']:.4f} n={sm['n_samples']}")
    return payload


############################################################## Panel-aware trainer ##############################################################


class PanelBacformerLargeTrainer(BacformerLargeTrainer):
    """:class:`BacformerLargeTrainer` that also forwards the per-protein surprisal ``panel``.

    The stock ``compute_loss`` passes only protein_embeddings/labels/attention_mask/contig_ids
    to the model, so a ``panel`` key in the batch would be silently dropped (and the
    panel-mode model would error for the missing tensor). This override mirrors the stock
    logic and additionally pops + forwards ``panel``. Used for both training and
    ``trainer.predict`` (stock HF ``prediction_step`` routes through ``compute_loss`` when
    labels are present).
    """

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):  # noqa: D102
        protein_embeddings = inputs.pop("protein_embeddings")
        labels = inputs.pop("labels")
        attention_mask = inputs.pop("attention_mask", None)
        contig_ids = inputs.pop("contig_ids", inputs.pop("token_type_ids", None))
        panel = inputs.pop("panel", None)
        outputs = model(
            protein_embeddings=protein_embeddings,
            labels=labels,
            attention_mask=attention_mask,
            contig_ids=contig_ids,
            panel=panel,
        )
        loss, logits = _extract_loss_and_logits(outputs)
        if return_outputs:
            return loss, {"logits": logits}
        return loss


############################################################## Main run function ##############################################################


def run(
    model_name_or_path: str,
    embeddings_dir: str,
    output_dir: str,
    ast_sheet_path: str | None,
    task: str = "ast",
    lr: float = 0.00015,
    batch_size: int = 1,
    grad_accumulation_steps: int = 8,
    drug: str = "rifampin",
    max_n_proteins: int = 6000,
    freeze_encoder: bool = False,
    precision: str = "bf16",
    pooling: str = "mean",
    attn_dim: int = 128,
    panel_mode: str = "none",
    panel_store: str | None = None,
    panel_stats: str | None = None,
    logging_steps: int = 10,
    n_samples: int = 10000,
    seed: int = 1,
    early_stopping_patience: int = 10,
    eval_steps: int = 150,
    num_workers: int = 16,
    warmup_proportion: float = 0.1,
    max_steps: int = 20000,
    max_epochs: int = 40,
    early_stopping_threshold: float = 0.005,
    n_folds: int | None = None,
    fold: int = 0,
    evaluate_seed: int = 1,
    resume_from_checkpoint: str | None = None,
    # deprecated — ignored; kept for call-site backward compat
    train_data_dir: str | None = None,
    val_data_dir: str | None = None,
    *,
    split_table: str | None = None,
    extra_holdouts: tuple[str, ...] | list[str] = DEFAULT_EXTRA_HOLDOUTS,
):
    """Fine-tune Bacformer on TB AMR data with dynamic label injection.

    When ``resume_from_checkpoint`` is set, training continues from that
    checkpoint (HF Trainer restores optimizer state, LR scheduler position,
    global step, best-metric tracking, and early-stopping counter) and the
    model is loaded from the checkpoint instead of ``model_name_or_path``.
    Pass ``output_dir`` = the run dir that already contains that checkpoint.

    When ``split_table`` is set (a ``<drug>_split.csv``), the splits and labels come from it alone
    (:func:`resolve_split_table`) and ``ast_sheet_path`` is ignored; ``extra_holdouts`` are scored after
    training beside the holdout (module docstring). ``split_table`` cannot be combined with ``n_folds``.
    """
    if train_data_dir or val_data_dir:
        warnings.warn(
            "--train-data-dir and --val-data-dir are deprecated and ignored. "
            "Use --embeddings-dir instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if pooling not in ("mean", "attention"):
        raise ValueError(f"--pooling must be 'mean' or 'attention', got {pooling!r}")

    if panel_mode not in ("none", "att_head", "e2e"):
        raise ValueError(f"--panel-mode must be 'none', 'att_head' or 'e2e', got {panel_mode!r}")
    if precision not in ("bf16", "fp32"):
        raise ValueError(f"--precision must be 'bf16' or 'fp32', got {precision!r}")
    use_panel = panel_mode != "none"
    panel_standardization = None
    panel_dim = 0
    if use_panel:
        if pooling != "attention":
            raise ValueError("--panel-mode requires --pooling attention.")
        if not panel_store or not panel_stats:
            raise ValueError("--panel-mode != none requires --panel-store and --panel-stats.")
        if panel_mode == "e2e" and freeze_encoder:
            raise ValueError("--panel-mode e2e fine-tunes the backbone end-to-end; do not set --freeze-encoder.")
        panel_standardization = json.loads(Path(panel_stats).read_text())
        panel_dim = len(panel_standardization["columns"])
        print(f"Panel mode: {panel_mode} (panel_dim={panel_dim}, store={panel_store})")

    if split_table is not None and n_folds is not None:
        raise ValueError("--split-table fixes the splits itself; do not combine it with --n-folds.")

    if n_folds is not None:
        output_dir = f"{output_dir}_fold{fold:02d}_seed{seed}"

    print(f"Loading model from: {model_name_or_path}")
    print(f"Predicting AMR for drug: {drug}")
    print(f"Embeddings dir: {embeddings_dir}")
    print(f"n_samples = {n_samples}")
    print(f"Freezing encoder: {freeze_encoder}")
    print(f"Learning rate: {lr}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Number of workers: {num_workers}")
    print(f"Eval steps: {eval_steps}")
    print(f"Grad accumulation steps: {grad_accumulation_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Max steps: {max_steps}")
    print(f"Warmup proportion: {warmup_proportion}")
    print(f"Output directory: {output_dir}")
    if split_table is not None:
        print(f"Split table: {split_table} (extra holdouts: {list(extra_holdouts)})")
    else:
        print(f"AST sheet: {ast_sheet_path}")
    if n_folds is not None:
        print(f"K-fold: n_folds={n_folds}, fold={fold}, seed={seed}, evaluate_seed={evaluate_seed}")
    print("------------------------------------------------\n")

    splits: SplitTableSplits | None = None
    if split_table is not None:
        splits = resolve_split_table(split_table, extra_holdouts, n_samples=n_samples, seed=seed)
        label_map = splits.label_map
        embeddings_path = Path(embeddings_dir)
        train_ids, val_ids, evaluate_ids = splits.train, splits.validate, splits.holdout
        print(f"Split table {splits.path} (md5 {splits.md5}): train {len(train_ids)}, validate {len(val_ids)}, "
              + ", ".join(f"{name} {len(ids)}" for name, ids in splits.holdouts.items())
              + (f"; species {sorted(set(splits.species.values()))}" if splits.species else ""))
        # Fail before the model loads, not at the holdout after hours of training: every genome the run will read
        needed = train_ids + val_ids + [s for ids in splits.holdouts.values() for s in ids]
        missing = missing_embeddings(needed, embeddings_path)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} genome(s) of {splits.path} have no <Sample>_esm_embeddings.pt in {embeddings_path}: "
                f"{missing[:10]}"
            )
        if splits.smoke:
            print("Using dummy test mode with 10 samples (class-balanced; train = validate = holdout).")
            split_source = "smoke"
            eval_strategy = "epoch"
            use_epochs = True
            num_train_epochs = 100
        else:
            split_source = "split_table"
            eval_strategy = "steps"
            use_epochs = False
    else:
        if not ast_sheet_path:
            raise ValueError("ast_sheet_path must be provided (binary_ast_with_split.csv)")
        if not os.path.exists(ast_sheet_path):
            raise FileNotFoundError(f"AST sheet not found at {ast_sheet_path}")

        ast_df = pd.read_csv(ast_sheet_path)
        if n_folds is None and "train_val_eval" not in ast_df.columns:
            raise ValueError(
                "AST sheet must contain 'train_val_eval' column. Run prepare_esmc_embeddings_and_labels_to_finetune_amr.py first, "
                "or use --n-folds to generate splits dynamically."
            )
        if "Sample" not in ast_df.columns:
            if "phenotype-BioSample_ID" in ast_df.columns:
                ast_df["Sample"] = ast_df["phenotype-BioSample_ID"].astype(str)
            else:
                raise ValueError("AST sheet must contain 'Sample' or 'phenotype-BioSample_ID'.")
        if drug not in ast_df.columns:
            raise ValueError(f"Drug column '{drug}' not found in AST sheet.")

        labeled = ast_df[ast_df[drug].notna()].copy()
        labeled["Sample"] = labeled["Sample"].astype(str)

        label_map: dict[str, int] = {row["Sample"]: int(row[drug]) for _, row in labeled.iterrows()}
        embeddings_path = Path(embeddings_dir)

        def build_sample_ids(split_name: str) -> list[str]:
            return labeled[labeled["train_val_eval"] == split_name]["Sample"].tolist()

        if n_samples == 10:
            print("Using dummy test mode with 10 samples.")
            train_ids = (build_sample_ids("train") if "train_val_eval" in labeled.columns else list(labeled["Sample"]))[:10]
            val_ids = train_ids
            evaluate_ids = train_ids  # No held-out evaluate set in smoke mode — reuse train for JSON sanity.
            split_source = "smoke"
            eval_strategy = "epoch"
            use_epochs = True
            num_train_epochs = 100
        elif n_folds is not None:
            print(f"K-fold mode: generating splits (n_folds={n_folds}, fold={fold}, seed={seed})")
            evaluate_ids_set, folds = generate_kfold_splits(
                labeled, n_folds=n_folds, seed=seed, evaluate_seed=evaluate_seed
            )
            train_ids_set, val_ids_set = folds[fold]
            train_ids = [sid for sid in labeled["Sample"].tolist() if sid in train_ids_set]
            val_ids = [sid for sid in labeled["Sample"].tolist() if sid in val_ids_set]
            evaluate_ids = [sid for sid in labeled["Sample"].tolist() if sid in evaluate_ids_set]
            split_source = "kfold"
            print(f"  train: {len(train_ids)}, val: {len(val_ids)}, evaluate holdout: {len(evaluate_ids)}")
            eval_strategy = "steps"
            use_epochs = False
        else:
            counts = labeled.groupby("train_val_eval")["Sample"].nunique().to_dict()
            print(
                f"Samples with non-missing '{drug}' - train: {counts.get('train', 0)}, "
                f"validate: {counts.get('validate', 0)}, evaluate: {counts.get('evaluate', 0)}"
            )
            print("Full set mode using LabelInjectingFileDataset")
            train_ids = build_sample_ids("train")
            val_ids = build_sample_ids("validate")
            evaluate_ids = build_sample_ids("evaluate")
            split_source = "csv"
            eval_strategy = "steps"
            use_epochs = False

    if not train_ids:
        raise RuntimeError(
            f"No training samples found (check AST sheet, '{drug}' column, and train_val_eval split)."
        )

    print(f"Number of train samples (with '{drug}'): {len(train_ids)}")
    print(f"Number of validation samples: {len(val_ids)}")

    def make_dataset(ids: list[str]):
        """Build a label dataset, augmented with the surprisal panel when panel-mode is on."""
        if use_panel:
            return PanelInjectingFileDataset(
                ids, embeddings_path, label_map, drug, Path(panel_store), panel_standardization
            )
        return LabelInjectingFileDataset(ids, embeddings_path, label_map, drug)

    train_dataset = make_dataset(train_ids)
    val_dataset = make_dataset(val_ids)

    # Verify structure
    try:
        sample = train_dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        emb = sample["protein_embeddings"]
        print(f"protein_embeddings shape: {emb.shape if hasattr(emb, 'shape') else len(emb)}")
        print(f"labels shape: {sample['labels'].shape}, value: {sample['labels'].item()}")
    except Exception as e:  # noqa: BLE001 (debug inspection only — warn and continue on any failure)
        print(f"WARNING: Could not inspect sample: {e}")

    # Load model (AutoModelForSequenceClassification loads BacformerLargeForGenomeClassification via auto_map).
    # The model is cast to bf16 master weights below — the proven, gold-standard setting for the deployed
    # AST models (Kp finetune AUROC 0.99); both organisms train identically. (bf16 weights mean CPU Stage-A
    # smokes don't run — do Stage A on GPU.)
    # On resume, load weights from the checkpoint directly (avoids a wasted base-model pull
    # before the trainer would overwrite them anyway).
    load_from = resume_from_checkpoint or model_name_or_path
    if resume_from_checkpoint:
        print(f"Resume mode: loading model weights from {resume_from_checkpoint}")

    if pooling == "attention":
        # Reuse the pretrained BacformerLarge backbone (contig-aware embeddings +
        # encoder, unchanged) and swap the mask-mean genome head for a gated-attention
        # MIL pool. Built from the base model id; on resume the Trainer restores the
        # full wrapper state (backbone + pool + head) from the checkpoint at train() time.
        print(f"Pooling: gated-attention MIL (attn_dim={attn_dim})")
        bacformer_model = BacformerAttnPoolForGenomeClassification.from_pretrained_backbone(
            model_name_or_path,
            num_labels=1,
            freeze_backbone=freeze_encoder,
            attn_dim=attn_dim,
            panel_mode=panel_mode,
            panel_dim=panel_dim if use_panel else 9,
            dtype="auto",
        )
    else:
        print("Pooling: mask-normalised mean (stock genome head)")
        bacformer_model = AutoModelForSequenceClassification.from_pretrained(
            load_from,
            num_labels=1,
            problem_type="binary_classification",
            return_dict=True,
            trust_remote_code=True,
        )
        if freeze_encoder:
            for param in bacformer_model.bacformer.parameters():
                param.requires_grad = False

    # Master-weight precision. bf16 (default, the deployed AST setting) casts here — applied uniformly to
    # the stock head and the attention-pool wrapper. fp32 keeps the native fp32 weights from_pretrained,
    # matching the pre-b047ed8 / CSD3 dtype="auto" condition; the bf16 AMP autocast on GPU
    # (TrainingArguments "bf16") is unchanged either way, so fp32-vs-bf16 isolates exactly that one commit
    # (fp32 also re-enables CPU Stage-A smokes).
    if precision == "bf16":
        bacformer_model = bacformer_model.to(torch.bfloat16)
    model_revision = getattr(getattr(bacformer_model, "config", None), "_commit_hash", None)

    print(f"Precision (master weights): {precision}")
    print("Nr of parameters:", sum(p.numel() for p in bacformer_model.parameters()))
    print("Nr of trainable:", sum(p.numel() for p in bacformer_model.parameters() if p.requires_grad))

    os.makedirs(output_dir, exist_ok=True)

    def collate_fn(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        # Squeeze batch dim for pad_sequence; each sample is [1, seq, dim] or [seq, dim]
        prot_list = [s["protein_embeddings"].squeeze(0) for s in samples]
        am_list = [s["attention_mask"].squeeze(0) for s in samples]
        contig_list = [s["contig_ids"].squeeze(0) for s in samples]

        batch = {
            "protein_embeddings": pad_sequence(prot_list, batch_first=True, padding_value=0.0),
            "labels": torch.stack([s["labels"] for s in samples], dim=0),
            "attention_mask": pad_sequence(am_list, batch_first=True, padding_value=0.0),
            "contig_ids": pad_sequence(contig_list, batch_first=True, padding_value=0),
        }
        # Pad the surprisal panel to the batch's protein length; 0.0 == the standardized
        # column mean, and padded rows are masked out of the attention softmax anyway.
        if "panel" in samples[0]:
            panel_list = [s["panel"].squeeze(0) for s in samples]
            batch["panel"] = pad_sequence(panel_list, batch_first=True, padding_value=0.0)
        return batch

    training_args_dict = {
        "output_dir": output_dir,
        "eval_strategy": eval_strategy,
        "save_strategy": eval_strategy,
        "save_total_limit": 1,
        "learning_rate": lr,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accumulation_steps,
        "seed": seed,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "dataloader_num_workers": num_workers,
        "dataloader_pin_memory": True,
        "dataloader_persistent_workers": bool(num_workers > 0),
        "bf16": bool(torch.cuda.is_available()),
        "metric_for_best_model": "eval_auroc",
        "load_best_model_at_end": True,
        "greater_is_better": True,
        "logging_steps": logging_steps,
        "logging_first_step": True,
        "logging_nan_inf_filter": False,
        "report_to": ["tensorboard"],
        "remove_unused_columns": False,
    }

    if use_epochs:
        total_batches = len(train_ids) // batch_size
        steps_per_epoch = max(1, total_batches // grad_accumulation_steps)
        calculated_max_steps = max(1, steps_per_epoch * num_train_epochs)
        training_args_dict["max_steps"] = calculated_max_steps
        training_args_dict["num_train_epochs"] = num_train_epochs
        print(f"num_train_epochs: {num_train_epochs}, calculated max_steps: {calculated_max_steps}")
    else:
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0 in full dataset mode. Pass --max-steps.")
        # Cap the flat max_steps by a dataset-aware epoch ceiling. A small cohort (Kp ~3k) otherwise
        # inherits the same 100k-step ceiling as TB (~35k) — hundreds of epochs — so it memorises the
        # train set (loss→0) many epochs past its best while early-stopping's patience counts down, and
        # warmup (scaled off max_steps) would itself run for dozens of epochs. Bounding by max_epochs keeps
        # the schedule, the warmup, and the worst-case wall sane per drug; early-stopping still ends most
        # runs earlier.
        steps_per_epoch = max(1, (len(train_ids) // batch_size) // grad_accumulation_steps)
        effective_max_steps = min(max_steps, steps_per_epoch * max_epochs)
        training_args_dict["max_steps"] = effective_max_steps
        training_args_dict["eval_steps"] = eval_steps
        training_args_dict["save_steps"] = eval_steps
        warmup_steps = int(effective_max_steps * warmup_proportion)
        training_args_dict["warmup_steps"] = warmup_steps
        training_args_dict["lr_scheduler_type"] = "linear"
        print(f"steps/epoch: {steps_per_epoch}, max_epochs cap: {max_epochs} -> effective max_steps: "
              f"{effective_max_steps} (of requested {max_steps}); warmup {warmup_steps}, eval_steps {eval_steps}")

    training_args = TrainingArguments(**training_args_dict)

    trainer_cls = PanelBacformerLargeTrainer if use_panel else BacformerLargeTrainer
    trainer = trainer_cls(
        model=bacformer_model,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        compute_metrics=compute_metrics_binary_genome_pred,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience,
                                         early_stopping_threshold=early_stopping_threshold)],
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    run_config = {
        "precision": precision,
        "pooling": pooling,
        "seed": seed,
        "lr": lr,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_threshold": early_stopping_threshold,
        "eval_steps": eval_steps,
        "warmup_proportion": warmup_proportion,
        "max_steps": max_steps,
    }
    if splits is not None:
        write_split_table_results(
            trainer,
            make_dataset,
            splits,
            output_dir=output_dir,
            task=task,
            drug=drug,
            model_name_or_path=model_name_or_path,
            model_revision=model_revision,
            split_source=split_source,
            n_train=len(train_ids),
            n_validate=len(val_ids),
            run_config={
                **run_config,
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "grad_accumulation_steps": grad_accumulation_steps,
                "n_samples": n_samples,
            },
        )
        return

    # Canonical §0.4 results JSON on the evaluate holdout. In smoke mode
    # (n_samples==10) there is no separate evaluate split so train_ids is reused
    # for end-to-end pipeline verification.
    if evaluate_ids:
        print(f"Running post-training evaluation on {len(evaluate_ids)} samples.")
        evaluate_dataset = make_dataset(evaluate_ids)
        preds = trainer.predict(evaluate_dataset)
        logits = torch.as_tensor(preds.predictions).flatten()
        labels = torch.as_tensor(preds.label_ids).flatten().long()
        keep = labels != -100
        if (~keep).any():
            logits, labels = logits[keep], labels[keep]
        y_prob = torch.sigmoid(logits.float()).cpu().numpy()
        y_true = labels.cpu().numpy()
        metrics_block = compute_full_metrics(y_true, y_prob)
        payload = build_results_payload(
            task=task,
            drug=drug,
            model_name_or_path=model_name_or_path,
            checkpoint_dir=str(Path(output_dir).resolve()),
            split_source=split_source,
            metrics=metrics_block,
            evaluate_seed=evaluate_seed if n_folds is not None else None,
            n_folds=n_folds,
            fold=fold if n_folds is not None else None,
            n_evaluate=len(evaluate_ids),
            model_revision=model_revision,
            run_config=run_config,
            versions={"torch": torch.__version__, "transformers": transformers.__version__},
        )
        results_path = Path(output_dir) / "results.json"
        write_results_json(results_path, payload)
        print(f"Wrote results JSON: {results_path}")
        print(
            f"  AUROC={metrics_block['auroc']:.4f} AUPRC={metrics_block['auprc']:.4f} "
            f"sens={metrics_block['sensitivity']:.4f} spec={metrics_block['specificity']:.4f} "
            f"bal_acc={metrics_block['balanced_accuracy']:.4f} n={metrics_block['n_samples']}"
        )
    else:
        print("No evaluate samples available; skipping results JSON write.")


############################################################## Argument parser ##############################################################


class ArgumentParser(Tap):
    """Argument parser for Bacformer TB AMR fine-tuning with dynamic label injection."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    model_name_or_path: str = "macwiatrak/bacformer-large-masked-complete-genomes"
    embeddings_dir: str | None = None
    """ESM-C store (default: <data-root>/processed/train_<task>/esm, keyed off --task)."""
    # deprecated — kept for backward compat; ignored if present
    train_data_dir: str | None = None
    val_data_dir: str | None = None
    output_dir: str = "/tmp/train-output/"
    task: str = "tb_ast"
    """Organism/task label recorded in results.json and keying the default store paths ('tb_ast', 'kleb_ast' or
    'pa_ast'; pa_ast needs --split-table). Any other label (a cross-species arm, e.g. 'pa_entero_ast') is accepted
    only with explicit --embeddings-dir and --split-table."""
    batch_size: int = 1
    grad_accumulation_steps: int = 8
    lr: float = 0.00015
    drug: str = "rifampin"
    max_n_proteins: int = 9000
    freeze_encoder: bool = False
    precision: str = "bf16"
    """Master-weight precision: 'bf16' (default, deployed setting) or 'fp32' (native weights + bf16 AMP,
    the pre-b047ed8 / CSD3 condition). fp32-vs-bf16 isolates the master-weight cast for the precision ablation."""
    pooling: str = "mean"
    """Genome pooling head: 'mean' (stock mask-normalised mean) or 'attention' (gated-attention MIL pool)."""
    attn_dim: int = 128
    """Attention hidden width when --pooling attention."""
    panel_mode: str = "none"
    """Surprisal-panel wiring: 'none', 'att_head' (panel steers the gate only) or 'e2e' (panel into pooled value + head). Requires --pooling attention."""
    panel_store: str | None = None
    """Directory of {sample}_panel.npz surprisal panels (required when --panel-mode != none)."""
    panel_stats: str | None = None
    """panel_standardization.json with train-only mean/std (required when --panel-mode != none)."""
    logging_steps: int = 10
    n_samples: int = 10000
    ast_sheet_path: str | None = None
    """Split sheet (default: <data-root>/processed/train_<task>/binary_ast_with_split.csv, keyed off --task)."""
    seed: int = 1
    max_steps: int = 100000
    max_epochs: int = 40
    """Epoch ceiling in full-dataset/k-fold mode: effective max_steps = min(max_steps, steps_per_epoch*max_epochs). Stops a small cohort inheriting a hundreds-of-epochs step ceiling (and dozens-of-epochs warmup)."""
    early_stopping_patience: int = 15
    early_stopping_threshold: float = 0.005
    """Min eval_auroc gain to count as an improvement, so a noisy plateau exhausts patience instead of resetting it."""
    eval_steps: int = 250
    num_workers: int = 15
    warmup_proportion: float = 0.1
    n_folds: int | None = None
    """Number of CV folds. When set, splits are generated dynamically; overrides train_val_eval column."""
    fold: int = 0
    """Which fold to use as validation set (0-indexed)."""
    evaluate_seed: int = 1
    """Seed controlling the fixed holdout set — do not change between folds/seeds in one experiment."""
    resume_from_checkpoint: str | None = None
    """If set, resume training from this checkpoint dir (true HF resume — preserves optimizer state, LR schedule, step counter, early-stopping counter). Pass --output-dir = the parent run dir."""
    split_table: str | None = None
    """A drug's <drug>_split.csv (engine.splits): train/validate/holdout and labels come from it alone, read by
    load_splits. Replaces --ast-sheet-path (do not pass both); cannot be combined with --n-folds."""
    extra_holdouts: list[str] = list(DEFAULT_EXTRA_HOLDOUTS)
    """With --split-table: further splits scored with the best model after training, beside the holdout
    (default holdout_st, the zero-shot held-out ST). Pass the flag with no value to score the holdout only."""


def resolve_task_paths(
    task: str,
    embeddings_dir: str | None,
    ast_sheet_path: str | None,
    split_table: str | None,
) -> tuple[str, str | None]:
    """``(embeddings_dir, ast_sheet_path)`` for the CLI: explicit paths win, else defaults keyed off ``--task``.

    Without ``--split-table`` this is the TB/Kp behaviour, unchanged. With it there is no AST sheet, and the store
    defaults to the organism's ESM-C directory (for ``pa_ast`` the flat multi-species store). ``pa_ast`` trains only
    from a split table; a task label with no organism (a cross-species arm) needs both paths given.
    """
    if task not in _TASK_TO_ORGANISM:
        if not (embeddings_dir and split_table):
            raise SystemExit(f"--task {task!r} is not one of {sorted(_TASK_TO_ORGANISM)}: a cross-species label needs "
                             "explicit --embeddings-dir and --split-table")
    elif task in _SPLIT_TABLE_ONLY_TASKS and not split_table:
        raise SystemExit(f"--task {task} trains only from a split table: pass --split-table <drug>_split.csv")
    if split_table:
        if ast_sheet_path:
            raise SystemExit("--split-table replaces --ast-sheet-path: pass one, not both")
        return embeddings_dir or str(organism(_TASK_TO_ORGANISM[task]).store_paths().esm_dir), None

    # Default the organism stores off --task; explicit --embeddings-dir/--ast-sheet-path win
    # (the root is resolved only when a path is actually missing).
    def _store() -> Path:
        return organism(_TASK_TO_ORGANISM[task]).data_root()

    return embeddings_dir or str(_store() / "esm"), ast_sheet_path or str(_store() / "binary_ast_with_split.csv")


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    print("Running AMR finetuning with dynamic label injection")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    embeddings_dir, ast_sheet_path = resolve_task_paths(
        args.task, args.embeddings_dir, args.ast_sheet_path, args.split_table
    )
    run(
        model_name_or_path=args.model_name_or_path,
        embeddings_dir=embeddings_dir,
        output_dir=args.output_dir,
        ast_sheet_path=ast_sheet_path,
        task=args.task,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation_steps,
        drug=args.drug,
        max_n_proteins=args.max_n_proteins,
        freeze_encoder=args.freeze_encoder,
        precision=args.precision,
        pooling=args.pooling,
        attn_dim=args.attn_dim,
        panel_mode=args.panel_mode,
        panel_store=args.panel_store,
        panel_stats=args.panel_stats,
        logging_steps=args.logging_steps,
        seed=args.seed,
        n_samples=args.n_samples,
        early_stopping_patience=args.early_stopping_patience,
        eval_steps=args.eval_steps,
        num_workers=args.num_workers,
        warmup_proportion=args.warmup_proportion,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        early_stopping_threshold=args.early_stopping_threshold,
        n_folds=args.n_folds,
        fold=args.fold,
        evaluate_seed=args.evaluate_seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
        split_table=args.split_table,
        extra_holdouts=args.extra_holdouts,
    )
