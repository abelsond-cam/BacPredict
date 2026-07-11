"""
Train Bacformer AMR model.

Labels are injected at load time from the split CSV; no pre-built per-experiment
``.pt`` copies are required. Pass ``--embeddings-dir`` pointing at the original
``klebsiella_esm_embeddings/`` directory.
"""
import os
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from bacformer.modeling.trainer import BacformerLargeTrainer
from tap import Tap
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    TrainingArguments,
)

from bacpredict.engine.finetune.datasets import LabelInjectingFileDataset
from bacpredict.engine.finetune.metrics import (
    build_results_payload,
    compute_full_metrics,
    compute_metrics_binary_genome_pred,
    write_results_json,
)
from bacpredict.engine.finetune.split_utils import generate_kfold_splits

EMBEDDINGS_DIR_DEFAULT = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/klebsiella_esm_embeddings"
)


############################################################## PyTorchFileDataset ##############################################################
class PyTorchFileDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that loads pytorch (.pt) files directly for AMR training."""

    def __init__(
        self,
        file_paths: list[Path],
        drug: str,
    ):
        """
        Initialize dataset.

        Args:
            file_paths: List of paths to pytorch (.pt) files named {sample_id}_with_ast.pt.
            drug: Drug column name for the label.
        """
        self.file_paths = file_paths
        self.drug = drug

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict:
        file_path = self.file_paths[idx]
        data = torch.load(file_path, map_location="cpu", weights_only=False)

        # Skip if missing label for this drug
        if self.drug not in data or pd.isna(data[self.drug]):
            raise ValueError(f"Sample {data.get('Sample', 'unknown')} has no label for drug {self.drug}")

        label_val = data[self.drug]
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

        # Bacformer Large expects [batch, seq, dim]; ensure 3D
        if prot_embeddings.dim() == 2:
            prot_embeddings = prot_embeddings.unsqueeze(0)

        seq_len = prot_embeddings.shape[1]
        sample: dict[str, torch.Tensor] = {
            "protein_embeddings": prot_embeddings,
            "labels": torch.tensor(label_val, dtype=torch.float32),
        }

        # Bacformer Large uses attention_mask and contig_ids (no special_tokens_mask).
        # Synthesize when missing: all ones for attention, zeros for contig (single contig).
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


############################################################## Main run function ##############################################################


def run(
    model_name_or_path: str,
    embeddings_dir: str,
    output_dir: str,
    ast_sheet_path: str,
    lr: float = 0.00015,
    batch_size: int = 1,
    grad_accumulation_steps: int = 8,
    drug: str = "ampicillin",
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
    # deprecated — ignored; kept for call-site backward compat
    train_data_dir: str | None = None,
    val_data_dir: str | None = None,
):
    """Fine-tune Bacformer on AMR data with dynamic label injection."""
    if train_data_dir or val_data_dir:
        warnings.warn(
            "--train-data-dir and --val-data-dir are deprecated and ignored. "
            "Use --embeddings-dir instead.",
            DeprecationWarning,
            stacklevel=2,
        )

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
    print(f"AST sheet: {ast_sheet_path}")
    if n_folds is not None:
        print(f"K-fold: n_folds={n_folds}, fold={fold}, seed={seed}, evaluate_seed={evaluate_seed}")
    print("------------------------------------------------\n")

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

    train_dataset = LabelInjectingFileDataset(train_ids, embeddings_path, label_map, drug)
    val_dataset = LabelInjectingFileDataset(val_ids, embeddings_path, label_map, drug)

    # Verify structure
    try:
        sample = train_dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        emb = sample["protein_embeddings"]
        print(f"protein_embeddings shape: {emb.shape if hasattr(emb, 'shape') else len(emb)}")
        print(f"labels shape: {sample['labels'].shape}, value: {sample['labels'].item()}")
    except Exception as e:
        print(f"WARNING: Could not inspect sample: {e}")

    # Load model (AutoModelForSequenceClassification loads BacformerLargeForGenomeClassification via auto_map)
    bacformer_model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=1,
        problem_type="binary_classification",
        return_dict=True,
        trust_remote_code=True,
    ).to(torch.bfloat16)

    if freeze_encoder:
        for param in bacformer_model.bacformer.parameters():
            param.requires_grad = False

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
    trainer.train()

    # Canonical §0.4 results JSON on the evaluate holdout. In smoke mode
    # (n_samples==10) there is no separate evaluate split so train_ids is reused
    # for end-to-end pipeline verification.
    if evaluate_ids:
        print(f"Running post-training evaluation on {len(evaluate_ids)} samples.")
        evaluate_dataset = LabelInjectingFileDataset(
            evaluate_ids, embeddings_path, label_map, drug
        )
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
            task="kleb_ast",
            drug=drug,
            model_name_or_path=model_name_or_path,
            checkpoint_dir=str(Path(output_dir).resolve()),
            split_source=split_source,
            metrics=metrics_block,
            evaluate_seed=evaluate_seed if n_folds is not None else None,
            n_folds=n_folds,
            fold=fold if n_folds is not None else None,
            n_evaluate=len(evaluate_ids),
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
    """Argument parser for Bacformer AMR fine-tuning with dynamic label injection."""

    def __init__(self):
        super().__init__(underscores_to_dashes=True)

    model_name_or_path: str = "macwiatrak/bacformer-large-masked-complete-genomes"
    """HF model ID. Default = refreshed complete-genomes (CG) weights. Override with
    'macwiatrak/bacformer-large-masked-MAG' for the sub-step 3 MAG contrast runs."""
    embeddings_dir: str = str(EMBEDDINGS_DIR_DEFAULT)
    # deprecated — kept for backward compat; ignored if present
    train_data_dir: str | None = None
    val_data_dir: str | None = None
    output_dir: str = "/tmp/train-output/"
    batch_size: int = 1
    grad_accumulation_steps: int = 8
    lr: float = 0.00015
    drug: str = "ceftriaxone"
    max_n_proteins: int = 9000
    freeze_encoder: bool = False
    logging_steps: int = 10
    n_samples: int = 10000
    ast_sheet_path: str = "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/train_kleb_ast/binary_ast_with_split.csv"
    seed: int = 1
    max_steps: int = 100000
    early_stopping_patience: int = 30
    eval_steps: int = 250
    num_workers: int = 15
    warmup_proportion: float = 0.1
    n_folds: int | None = None
    """Number of CV folds. When set, splits are generated dynamically; overrides train_val_eval column."""
    fold: int = 0
    """Which fold to use as validation set (0-indexed)."""
    evaluate_seed: int = 1
    """Seed controlling the fixed holdout set — do not change between folds/seeds in one experiment."""


if __name__ == "__main__":
    args = ArgumentParser().parse_args()
    print("Running AMR finetuning with dynamic label injection")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    run(
        model_name_or_path=args.model_name_or_path,
        embeddings_dir=args.embeddings_dir,
        output_dir=args.output_dir,
        ast_sheet_path=args.ast_sheet_path,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation_steps,
        drug=args.drug,
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
    )
