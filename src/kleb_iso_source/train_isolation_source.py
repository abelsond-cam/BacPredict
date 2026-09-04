"""
Train Bacformer for a binary isolation-source pair.

Labels are injected at load time from the split CSV; no pre-built per-experiment
``.pt`` copies are required. Pass ``--embeddings-dir`` pointing at the original
``klebsiella_esm_embeddings/`` directory.
"""
import argparse
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from bacformer.modeling.trainer import BacformerLargeTrainer
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    EvalPrediction,
    TrainingArguments,
)

from bacpredict.engine.finetune.checkpoints import pick_resume_checkpoint
from bacpredict.engine.finetune.datasets import LabelInjectingFileDataset
from bacpredict.engine.finetune.metrics import build_results_payload, compute_full_metrics, write_results_json
from bacpredict.engine.finetune.split_utils import generate_kfold_splits
from kleb_iso_source.isolation_source_cli_parsing import (
    sanitize_pair_name,
    slugify_isolation_source_token,
)

PROCESSED_DIR = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed")
PROCESSED_BASE_DIR_DEFAULT = PROCESSED_DIR / "train_iso_source"
EMBEDDINGS_DIR_DEFAULT = PROCESSED_DIR / "klebsiella_esm_embeddings"


############################################################## PyTorchFileDataset ##############################################################
class PyTorchFileDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that loads pytorch (.pt) files for isolation-source pair training."""

    def __init__(
        self,
        file_paths: list[Path],
        label_column: str,
    ):
        self.file_paths = file_paths
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict:
        file_path = self.file_paths[idx]
        data = torch.load(file_path, map_location="cpu", weights_only=False)

        if self.label_column not in data or pd.isna(data[self.label_column]):
            raise ValueError(
                f"Sample {data.get('Sample', 'unknown')} has no label for {self.label_column}"
            )

        label_val = data[self.label_column]
        if isinstance(label_val, torch.Tensor):
            label_val = label_val.item()
        elif hasattr(label_val, "item"):
            label_val = label_val.item()
        label_val = int(label_val)

        prot_embeddings = data.get("prot_embeddings", data.get("protein_embeddings"))
        if prot_embeddings is None:
            raise KeyError(
                f"Sample {data.get('Sample', 'unknown')} is missing 'prot_embeddings'/'protein_embeddings'."
            )

        if prot_embeddings.dim() == 2:
            prot_embeddings = prot_embeddings.unsqueeze(0)

        seq_len = prot_embeddings.shape[1]
        sample: dict[str, torch.Tensor] = {
            "protein_embeddings": prot_embeddings,
            "labels": torch.tensor(label_val, dtype=torch.float32),
        }

        am = data["attention_mask"] if "attention_mask" in data else None
        if am is not None and am.dim() == 1:
            am = am.unsqueeze(0)
        sample["attention_mask"] = am if am is not None else torch.ones(1, seq_len, dtype=torch.float32)
        contig_src = data.get("contig_idx", data.get("contig_ids", data.get("token_type_ids")))
        if contig_src is not None:
            sample["contig_ids"] = (
                contig_src.unsqueeze(0) if contig_src.dim() == 1 else contig_src
            )
        else:
            sample["contig_ids"] = torch.zeros(1, seq_len, dtype=torch.long)

        return sample


############################################################## Compute metrics ##############################################################


def compute_metrics_binary_genome_pred(
    preds: EvalPrediction, ignore_index: int = -100, prefix: str = "eval"
):
    """Compute metrics for a single-logit binary classifier."""
    with torch.no_grad():
        logits = torch.tensor(preds.predictions).flatten()
        labels = torch.tensor(preds.label_ids).flatten().long()
        if (labels == ignore_index).any():
            keep = labels != ignore_index
            logits, labels = logits[keep], labels[keep]
        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).long()
        from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

        y_true = labels.cpu().numpy()
        y_prob = prob.cpu().numpy()
        y_pred = pred.cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        try:
            auroc_val = roc_auc_score(y_true, y_prob)
        except ValueError:  # roc_auc_score raises when y_true is single-class
            auroc_val = float("nan")
        try:
            auprc = average_precision_score(y_true, y_prob)
        except ValueError:  # average_precision_score raises when y_true is single-class
            auprc = float("nan")
        f1 = f1_score(y_true, y_pred, average="binary")

    return {
        f"{prefix}_accuracy": acc,
        f"{prefix}_auroc": auroc_val,
        f"{prefix}_auprc": auprc,
        f"{prefix}_f1": f1,
        f"{prefix}_nr_samples": len(y_true),
    }


############################################################## Main run function ##############################################################


def run(
    model_name_or_path: str,
    embeddings_dir: str,
    output_dir: str,
    sheet_path: str,
    label_column: str,
    lr: float = 0.00015,
    batch_size: int = 1,
    grad_accumulation_steps: int = 8,
    max_n_proteins: int = 6000,
    freeze_encoder: bool = False,
    logging_steps: int = 10,
    n_samples: int = 10000,
    seed: int = 1,
    early_stopping_patience: int = 10,
    eval_steps: int = 150,
    num_workers: int = 16,
    warmup_proportion: float = 0.1,
    max_steps: int = 20000,
    n_folds: int | None = None,
    fold: int = 0,
    evaluate_seed: int = 1,
    precision: str = "bf16",
    resume_from_checkpoint: str = "none",
    # deprecated — ignored; kept for call-site backward compat
    train_data_dir: str | None = None,
    val_data_dir: str | None = None,
    pt_suffix: str | None = None,
):
    """Fine-tune Bacformer on a pair-specific isolation-source label.

    Labels are injected at load time; no pre-built per-experiment .pt copies needed.

    ``precision`` sets the master-weight dtype: ``"bf16"`` (default, the deployed AST setting) or
    ``"fp32"`` (the pre-``a817ac2`` condition that produced the 2026-05 results). It is recorded in
    ``results.json`` under ``run_config.precision``.

    ``resume_from_checkpoint`` continues an interrupted run: ``"auto"`` picks the newest checkpoint in
    ``output_dir`` and starts fresh if there is none, ``"none"`` always starts fresh, or pass an
    explicit checkpoint path. HF restores optimiser, LR schedule, RNG and the early-stopping counter,
    so a chained run follows the same trajectory as an uninterrupted one — which is what lets a 36 h
    fine-tune fit a 12 h queue.
    """
    if precision not in ("bf16", "fp32"):
        raise ValueError(f"--precision must be 'bf16' or 'fp32', got {precision!r}")

    if train_data_dir or val_data_dir or pt_suffix:
        warnings.warn(
            "--train-data-dir, --val-data-dir, and pt_suffix are deprecated and ignored. "
            "Use --embeddings-dir instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    if n_folds is not None:
        output_dir = f"{output_dir}_fold{fold:02d}_seed{seed}"

    print(f"Loading model from: {model_name_or_path}")
    print(f"Predicting {label_column}")
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
    print(f"Sheet: {sheet_path}")
    if n_folds is not None:
        print(f"K-fold: n_folds={n_folds}, fold={fold}, seed={seed}, evaluate_seed={evaluate_seed}")
    print("------------------------------------------------\n")

    if not sheet_path:
        raise ValueError("sheet_path must be provided (binary_<pair_slug>_with_split.csv).")
    if not os.path.exists(sheet_path):
        raise FileNotFoundError(f"Sheet not found at {sheet_path}")

    # low_memory=False: the split CSV carries the full wide metadata (hundreds of mixed-type
    # columns). The default chunked C parser can hard-crash (exit 1, no traceback) on the wider
    # cohorts — read in one pass to dtype-infer safely.
    df = pd.read_csv(sheet_path, low_memory=False)
    if n_folds is None and "train_val_eval" not in df.columns:
        raise ValueError(
            "Sheet must contain 'train_val_eval' column. "
            "Run prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py first, "
            "or use --n-folds to generate splits dynamically."
        )
    if "Sample" not in df.columns:
        if "sample_accession" in df.columns:
            df["Sample"] = df["sample_accession"].astype(str)
        elif "phenotype-BioSample_ID" in df.columns:
            df["Sample"] = df["phenotype-BioSample_ID"].astype(str)
        else:
            raise ValueError(
                "Sheet must contain 'Sample', 'sample_accession', or 'phenotype-BioSample_ID'."
            )
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in sheet.")

    labeled = df[df[label_column].notna()].copy()
    labeled["Sample"] = labeled["Sample"].astype(str)

    label_map: dict[str, int] = {
        row["Sample"]: int(row[label_column]) for _, row in labeled.iterrows()
    }
    embeddings_path = Path(embeddings_dir)

    def build_sample_ids(split_name: str) -> list[str]:
        return labeled[labeled["train_val_eval"] == split_name]["Sample"].tolist()

    if n_samples == 10:
        print("Using dummy test mode with 10 samples.")
        train_ids = (build_sample_ids("train") if "train_val_eval" in labeled.columns else list(labeled["Sample"]))[:10]
        val_ids = train_ids
        evaluate_ids = []  # smoke mode: no separate holdout
        split_source = "smoke"
        eval_strategy = "epoch"
        use_epochs = True
        num_train_epochs = 100
    elif n_folds is not None:
        print(f"K-fold mode: generating splits (n_folds={n_folds}, fold={fold}, seed={seed})")
        evaluate_set, folds = generate_kfold_splits(
            labeled, n_folds=n_folds, seed=seed, evaluate_seed=evaluate_seed
        )
        train_ids_set, val_ids_set = folds[fold]
        train_ids = [sid for sid in labeled["Sample"].tolist() if sid in train_ids_set]
        val_ids = [sid for sid in labeled["Sample"].tolist() if sid in val_ids_set]
        evaluate_ids = [sid for sid in labeled["Sample"].tolist() if sid in evaluate_set]
        split_source = "kfold"
        print(f"  train: {len(train_ids)}, val: {len(val_ids)}, evaluate holdout: {len(evaluate_ids)}")
        eval_strategy = "steps"
        use_epochs = False
    else:
        counts = labeled.groupby("train_val_eval")["Sample"].nunique().to_dict()
        print(
            f"Samples with non-missing '{label_column}' - train: {counts.get('train', 0)}, "
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
            f"No training samples found (check sheet, '{label_column}' column, and train_val_eval split)."
        )

    print(f"Number of train samples (with '{label_column}'): {len(train_ids)}")
    print(f"Number of validation samples: {len(val_ids)}")

    train_dataset = LabelInjectingFileDataset(train_ids, embeddings_path, label_map, label_column)
    val_dataset = LabelInjectingFileDataset(val_ids, embeddings_path, label_map, label_column)

    try:
        sample = train_dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        emb = sample["protein_embeddings"]
        print(f"protein_embeddings shape: {emb.shape if hasattr(emb, 'shape') else len(emb)}")
        print(f"labels shape: {sample['labels'].shape}, value: {sample['labels'].item()}")
    except Exception as e:  # noqa: BLE001 (debug inspection only — warn and continue on any failure)
        print(f"WARNING: Could not inspect sample: {e}")

    # Master-weight precision — mirrors bacpredict.engine.finetune.finetune_amr. bf16 (default) is the
    # deployed AST setting, where it beat fp32 by ~7pp AUROC on TB rifampin in a controlled A/B. fp32
    # keeps the native from_pretrained weights, which is what dtype="auto" resolved to for this model —
    # the underperforming pre-b047ed8 / CSD3 condition that produced every 2026-05 iso-source result.
    # The bf16 AMP autocast on GPU (TrainingArguments "bf16") is unchanged either way, so fp32-vs-bf16
    # isolates exactly the master-weight cast (fp32 also re-enables CPU Stage-A smokes).
    bacformer_model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=1,
        problem_type="binary_classification",
        return_dict=True,
        trust_remote_code=True,
        dtype="auto",
    )
    if precision == "bf16":
        bacformer_model = bacformer_model.to(torch.bfloat16)
    model_revision = getattr(getattr(bacformer_model, "config", None), "_commit_hash", None)
    print(f"Precision (master weights): {precision}")

    if freeze_encoder:
        for param in bacformer_model.bacformer.parameters():
            param.requires_grad = False

    print("Nr of parameters:", sum(p.numel() for p in bacformer_model.parameters()))
    print("Nr of trainable:", sum(p.numel() for p in bacformer_model.parameters() if p.requires_grad))

    os.makedirs(output_dir, exist_ok=True)

    def collate_fn(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        prot_list = [s["protein_embeddings"].squeeze(0) for s in samples]
        am_list = [s["attention_mask"].squeeze(0) for s in samples]
        contig_list = [s["contig_ids"].squeeze(0) for s in samples]
        return {
            "protein_embeddings": pad_sequence(prot_list, batch_first=True, padding_value=0.0),
            "labels": torch.stack([s["labels"] for s in samples], dim=0),
            "attention_mask": pad_sequence(am_list, batch_first=True, padding_value=0.0),
            "contig_ids": pad_sequence(contig_list, batch_first=True, padding_value=0),
        }

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
        training_args_dict["max_steps"] = max_steps
        training_args_dict["eval_steps"] = eval_steps
        training_args_dict["save_steps"] = eval_steps
        warmup_steps = int(max_steps * warmup_proportion)
        training_args_dict["warmup_steps"] = warmup_steps
        training_args_dict["lr_scheduler_type"] = "linear"
        print(f"Warmup steps: {warmup_steps}, max_steps: {max_steps}, eval_steps: {eval_steps}")

    training_args = TrainingArguments(**training_args_dict)

    trainer = BacformerLargeTrainer(
        model=bacformer_model,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        compute_metrics=compute_metrics_binary_genome_pred,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    # `auto` on a fresh output dir must mean "start from scratch", not crash: link 1 of a chained run
    # has no checkpoint, and every later link does. The newest checkpoint is the one to resume from —
    # not the best, which save_total_limit=1 also keeps. pick_resume_checkpoint rather than HF's
    # get_last_checkpoint because a link killed mid-save leaves a partial directory, and every later
    # link would otherwise pick that same corpse and take the chain down with it.
    resume: str | bool | None = None
    if resume_from_checkpoint == "auto":
        found = pick_resume_checkpoint(output_dir)
        resume = str(found) if found else None
        print(f"Resume: auto -> {found or 'no complete checkpoint found, starting fresh'}")
    elif resume_from_checkpoint != "none":
        resume = resume_from_checkpoint
        print(f"Resume: {resume}")
    trainer.train(resume_from_checkpoint=resume)

    # Canonical §0.4 results JSON on the held-out evaluate split (mirrors kleb_ast/train_amr.py).
    # Smoke mode (n_samples==10) has no separate evaluate split, so skip it there.
    if evaluate_ids:
        print(f"Running post-training evaluation on {len(evaluate_ids)} evaluate samples.")
        evaluate_dataset = LabelInjectingFileDataset(evaluate_ids, embeddings_path, label_map, label_column)
        preds = trainer.predict(evaluate_dataset)
        logits = torch.as_tensor(preds.predictions).flatten()
        labels = torch.as_tensor(preds.label_ids).flatten().long()
        keep = labels != -100
        if (~keep).any():
            logits, labels = logits[keep], labels[keep]
        y_prob = torch.sigmoid(logits.float()).cpu().numpy()
        y_true = labels.cpu().numpy()
        # Per-genome scores, keyed by Sample. Two things need them and neither can be reconstructed
        # from results.json: verifying this run really was scored on the materialised holdout (a
        # matching n is not proof — see materialise_kfold_splits), and the paired bootstrap against
        # the unitig model, which resamples the same genomes in both arms.
        scored_ids = np.asarray(evaluate_ids, dtype=np.str_)[keep.cpu().numpy()]
        scores_path = Path(output_dir) / "eval_scores.npz"
        np.savez(scores_path, sample_ids=scored_ids, y_true=y_true, y_prob=y_prob,
                 drug=np.asarray(label_column), operating_threshold=np.asarray(0.5))
        print(f"Wrote eval scores: {scores_path} ({len(scored_ids)} genomes)")
        metrics_block = compute_full_metrics(y_true, y_prob)
        payload = build_results_payload(
            task="kleb_iso_source",
            drug=label_column,
            model_name_or_path=model_name_or_path,
            checkpoint_dir=str(Path(output_dir).resolve()),
            split_source=split_source,
            metrics=metrics_block,
            evaluate_seed=evaluate_seed if n_folds is not None else None,
            n_folds=n_folds,
            fold=fold if n_folds is not None else None,
            n_evaluate=len(evaluate_ids),
            model_revision=model_revision,
            run_config={
                "precision": precision,
                "seed": seed,
                "lr": lr,
                "early_stopping_patience": early_stopping_patience,
                "eval_steps": eval_steps,
                "warmup_proportion": warmup_proportion,
                "max_steps": max_steps,
                "max_n_proteins": max_n_proteins,
                "freeze_encoder": freeze_encoder,
            },
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


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune Bacformer for an isolation-source pair from split .pt files "
        "(same layout as prepare_esmc_embeddings_and_labels_to_finetune_isolation_source.py)."
    )
    p.add_argument(
        "--isolation-sources",
        nargs=2,
        metavar=("TOKEN1", "TOKEN2"),
        required=True,
        help="Two isolation-source CLI tokens (same as the prepare script).",
    )
    p.add_argument(
        "--processed-base-dir",
        type=str,
        default=str(PROCESSED_BASE_DIR_DEFAULT),
        help="Directory containing <slug1>_<slug2>/ (default matches prepare script).",
    )
    p.add_argument(
        "--embeddings-dir",
        type=str,
        default=str(EMBEDDINGS_DIR_DEFAULT),
        help="Directory containing {sample_id}_esm_embeddings.pt files (original, shared across experiments).",
    )
    p.add_argument(
        "--train-data-dir",
        type=str,
        default=None,
        help="[DEPRECATED — ignored] Previously the train split .pt directory.",
    )
    p.add_argument(
        "--val-data-dir",
        type=str,
        default=None,
        help="[DEPRECATED — ignored] Previously the validation split .pt directory.",
    )
    p.add_argument(
        "--sheet-path",
        type=str,
        default=None,
        help="Override path to binary_<pair_slug>_with_split.csv (default under the <slug1>_<slug2>/ pair dir).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Checkpoint output dir. An absolute path is used verbatim; a relative one "
            "is a subdirectory name under the <slug1>_<slug2>/ pair dir (default includes learning rate)."
        ),
    )
    p.add_argument("--model-name-or-path", type=str, default="macwiatrak/bacformer-large-masked-complete-genomes")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accumulation-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.00015)
    p.add_argument("--max-n-proteins", type=int, default=9000)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--n-samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=100000)
    p.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default="none",
        help="'auto' resumes from the newest checkpoint in --output-dir (starting fresh if there is "
             "none), 'none' always starts fresh, or an explicit checkpoint path. 'auto' is what lets "
             "a 36 h fine-tune run as chained 12 h jobs.",
    )
    p.add_argument("--early-stopping-patience", type=int, default=30)
    p.add_argument("--eval-steps", type=int, default=250)
    p.add_argument("--num-workers", type=int, default=15)
    p.add_argument("--warmup-proportion", type=float, default=0.1)
    p.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help="Number of cross-validation folds. When set, splits are generated dynamically "
        "and --fold selects which fold to use as validation. Overrides train_val_eval column.",
    )
    p.add_argument("--fold", type=int, default=0, help="Which fold to use as the validation set (0-indexed).")
    p.add_argument(
        "--evaluate-seed",
        type=int,
        default=1,
        help="Seed controlling the fixed holdout set. Do not change between folds/seeds "
        "within one experiment — the holdout must remain constant.",
    )
    p.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="Master-weight precision. 'bf16' (default) is the deployed AST setting; 'fp32' reproduces "
        "the pre-a817ac2 dtype='auto' condition that produced the 2026-05 results. Recorded in results.json.",
    )
    return p


def _resolve_paths_from_tokens(
    token1: str,
    token2: str,
    processed_base_dir: str,
    sheet_path: str | None,
    output_dir: str | None,
    lr: float,
) -> tuple[str, str, str]:
    pair_slug = sanitize_pair_name(token1, token2)
    slug1 = slugify_isolation_source_token(token1)
    slug2 = slugify_isolation_source_token(token2)
    base = Path(processed_base_dir) / f"{slug1}_{slug2}"
    sheet = sheet_path or str(base / f"binary_{pair_slug}_with_split.csv")
    label_column = f"{pair_slug}_label"
    # An absolute --output-dir is used verbatim (lets cohorts live in a flat home like
    # train_iso_source/blood_faeces/<cohort>/models); a relative one nests under base.
    if output_dir and Path(output_dir).is_absolute():
        out = output_dir
    else:
        out = str(base / (output_dir or f"bacformer_finetuned_lr_{lr}"))
    return sheet, out, label_column


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    token1, token2 = args.isolation_sources
    sheet_path, output_dir, label_column = _resolve_paths_from_tokens(
        token1,
        token2,
        args.processed_base_dir,
        args.sheet_path,
        args.output_dir,
        args.lr,
    )
    print("Isolation-source pair finetuning with dynamic label injection")
    print(f"Tokens: {token1!r}, {token2!r} -> pair_slug={sanitize_pair_name(token1, token2)!r}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    run(
        model_name_or_path=args.model_name_or_path,
        embeddings_dir=args.embeddings_dir,
        output_dir=output_dir,
        sheet_path=sheet_path,
        label_column=label_column,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_n_proteins=args.max_n_proteins,
        freeze_encoder=args.freeze_encoder,
        logging_steps=args.logging_steps,
        seed=args.seed,
        n_samples=args.n_samples,
        early_stopping_patience=args.early_stopping_patience,
        eval_steps=args.eval_steps,
        num_workers=args.num_workers,
        warmup_proportion=args.warmup_proportion,
        max_steps=args.max_steps,
        n_folds=args.n_folds,
        fold=args.fold,
        evaluate_seed=args.evaluate_seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
        precision=args.precision,
    )
