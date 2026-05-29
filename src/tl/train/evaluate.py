r"""Evaluate a fine-tuned Bacformer AMR checkpoint on its held-out evaluate set.

Task-agnostic — shared by every AST task (kleb_ast, tb_ast, …). Loads a saved
checkpoint, reconstructs the *identical* evaluate holdout from the split CSV (via
the same ``generate_kfold_splits`` seed used at training time, or the CSV
``train_val_eval == "evaluate"`` rows), runs inference, computes the §0.4 metric
block, and writes ``eval_results.json`` + ``eval_scores.npz`` next to the
checkpoint. A second ``--combine`` mode renders a per-drug ROC | PR figure grid
from several saved ``eval_scores.npz`` files.

Examples
--------
Evaluate one checkpoint (writes eval_results.json + eval_scores.npz + a 1x2 figure)::

    uv run python src/tl/train/evaluate.py \\
      --checkpoint .../klebsiella_pneumoniae_ceftriaxone_..._fold00_seed1 \\
      --drug ceftriaxone --task kleb_ast --n-folds 5 --evaluate-seed 1 \\
      --ast-sheet-path .../train_kleb_ast/binary_ast_with_split.csv \\
      --embeddings-dir .../klebsiella_esm_embeddings

Combine several drugs into one ROC | PR grid (one row per drug)::

    uv run python src/tl/train/evaluate.py --combine \\
      ceftriaxone=.../ceftriaxone_.../eval_scores.npz \\
      gentamicin=.../gentamicin_.../eval_scores.npz \\
      meropenem=.../meropenem_.../eval_scores.npz \\
      --combine-out .../train_kleb_ast/eval_roc_pr_grid.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from tl.train.datasets import LabelInjectingFileDataset
from tl.train.metrics import build_results_payload, compute_full_metrics, write_results_json
from tl.train.split_utils import generate_kfold_splits


def collate_fn(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a list of per-sample dicts into a batch (matches train_amr.py)."""
    prot_list = [s["protein_embeddings"].squeeze(0) for s in samples]
    am_list = [s["attention_mask"].squeeze(0) for s in samples]
    contig_list = [s["contig_ids"].squeeze(0) for s in samples]
    return {
        "protein_embeddings": pad_sequence(prot_list, batch_first=True, padding_value=0.0),
        "labels": torch.stack([s["labels"] for s in samples], dim=0),
        "attention_mask": pad_sequence(am_list, batch_first=True, padding_value=0.0),
        "contig_ids": pad_sequence(contig_list, batch_first=True, padding_value=0),
    }


def resolve_evaluate_ids(
    ast_sheet_path: str,
    drug: str,
    n_folds: int | None,
    seed: int,
    evaluate_seed: int,
) -> tuple[list[str], dict[str, int], str]:
    """Reconstruct the evaluate-holdout sample IDs + label map for a drug.

    Mirrors ``train_amr.py``: k-fold mode derives the fixed holdout from
    ``evaluate_seed``; otherwise reads ``train_val_eval == "evaluate"`` from the CSV.
    """
    df = pd.read_csv(ast_sheet_path, low_memory=False)
    if "Sample" not in df.columns:
        if "phenotype-BioSample_ID" in df.columns:
            df["Sample"] = df["phenotype-BioSample_ID"].astype(str)
        else:
            raise ValueError("AST sheet must contain 'Sample' or 'phenotype-BioSample_ID'.")
    if drug not in df.columns:
        raise ValueError(f"Drug column {drug!r} not found in AST sheet.")

    labeled = df[df[drug].notna()].copy()
    labeled["Sample"] = labeled["Sample"].astype(str)
    label_map = {row["Sample"]: int(row[drug]) for _, row in labeled.iterrows()}

    if n_folds is not None:
        evaluate_set, _ = generate_kfold_splits(labeled, n_folds=n_folds, seed=seed, evaluate_seed=evaluate_seed)
        evaluate_ids = [sid for sid in labeled["Sample"].tolist() if sid in evaluate_set]
        return evaluate_ids, label_map, "kfold"

    if "train_val_eval" not in labeled.columns:
        raise ValueError("CSV has no 'train_val_eval' column; pass --n-folds to derive the holdout.")
    evaluate_ids = labeled[labeled["train_val_eval"] == "evaluate"]["Sample"].tolist()
    return evaluate_ids, label_map, "csv"


def resolve_checkpoint_dir(checkpoint: Path) -> Path:
    """Return the dir holding the model files.

    Trainer saves weights under ``checkpoint-<step>/`` (with ``save_total_limit=1``
    only the best/last is kept), while the run dir itself holds only
    ``results.json`` + ``runs/``. If ``checkpoint`` already has ``config.json`` use
    it; otherwise pick the highest-step ``checkpoint-*`` subdir that does.
    """
    checkpoint = Path(checkpoint)
    if (checkpoint / "config.json").exists():
        return checkpoint

    def _step(p: Path) -> int:
        tail = p.name.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    subs = sorted(
        (p for p in checkpoint.glob("checkpoint-*") if (p / "config.json").exists()),
        key=_step,
    )
    if not subs:
        raise FileNotFoundError(f"No config.json in {checkpoint} or any checkpoint-*/ subdir.")
    return subs[-1]


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob) over the dataloader."""
    model.eval()
    dtype = next(model.parameters()).dtype
    y_true_parts, y_prob_parts = [], []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            pe = batch["protein_embeddings"].to(device=device, dtype=dtype)
            am = batch["attention_mask"].to(device=device, dtype=dtype)
            cid = batch["contig_ids"].to(device=device)
            out = model(protein_embeddings=pe, attention_mask=am, contig_ids=cid)
            logits = out.logits.reshape(-1).float().cpu()
            y_prob_parts.append(torch.sigmoid(logits))
            y_true_parts.append(labels.reshape(-1).cpu())
    return (
        torch.cat(y_true_parts).numpy().astype(int),
        torch.cat(y_prob_parts).numpy().astype(float),
    )


def plot_roc_pr_grid(entries: list[tuple[str, np.ndarray, np.ndarray]], out_path: Path | str) -> None:
    """Render an N-row x 2-col grid: ROC (left) and PR (right) per drug.

    Parameters
    ----------
    entries
        List of ``(label, y_true, y_prob)`` tuples — one row per entry.
    out_path
        Destination PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )

    n = len(entries)
    fig, axes = plt.subplots(n, 2, figsize=(11, 4.2 * n), squeeze=False)
    for i, (label, y_true, y_prob) in enumerate(entries):
        ax_roc, ax_pr = axes[i]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)
        ax_roc.plot(fpr, tpr, lw=2, color="C0", label=f"AUROC = {auroc:.3f}")
        ax_roc.plot([0, 1], [0, 1], ls="--", lw=1, color="grey")
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1.02)
        ax_roc.set_xlabel("False positive rate")
        ax_roc.set_ylabel("True positive rate")
        ax_roc.set_title(f"{label} — ROC")
        ax_roc.legend(loc="lower right")

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        prevalence = float(np.mean(y_true))
        ax_pr.plot(rec, prec, lw=2, color="C1", label=f"AUPRC = {auprc:.3f}")
        ax_pr.axhline(prevalence, ls="--", lw=1, color="grey", label=f"prevalence = {prevalence:.3f}")
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1.02)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"{label} — PR")
        ax_pr.legend(loc="lower left")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _evaluate(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else checkpoint
    device = "cpu" if (args.no_cuda or not torch.cuda.is_available()) else "cuda"

    evaluate_ids, label_map, split_source = resolve_evaluate_ids(
        args.ast_sheet_path, args.drug, args.n_folds, args.seed, args.evaluate_seed
    )
    print(f"Drug {args.drug}: {len(evaluate_ids)} evaluate samples ({split_source} split), device={device}")
    if not evaluate_ids:
        raise RuntimeError(f"No evaluate samples for drug {args.drug!r}.")

    model_dir = resolve_checkpoint_dir(checkpoint)
    if model_dir != checkpoint:
        print(f"  loading weights from {model_dir.name}/")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        num_labels=1,
        problem_type="binary_classification",
        return_dict=True,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    if device == "cpu":
        model = model.float()
    model = model.to(device)

    dataset = LabelInjectingFileDataset(evaluate_ids, Path(args.embeddings_dir), label_map, args.drug)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    y_true, y_prob = run_inference(model, loader, device)
    metrics = compute_full_metrics(y_true, y_prob)

    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "eval_scores.npz"
    np.savez(scores_path, y_true=y_true, y_prob=y_prob, drug=np.array(args.drug))

    payload = build_results_payload(
        task=args.task,
        drug=args.drug,
        model_name_or_path=str(model_dir),
        checkpoint_dir=str(checkpoint),
        split_source=split_source,
        metrics=metrics,
        evaluate_seed=args.evaluate_seed if args.n_folds is not None else None,
        n_folds=args.n_folds,
        fold=args.fold if args.n_folds is not None else None,
        n_evaluate=len(evaluate_ids),
    )
    write_results_json(out_dir / "eval_results.json", payload)
    plot_roc_pr_grid([(args.drug, y_true, y_prob)], out_dir / f"eval_roc_pr_{args.drug}.png")

    print(
        f"  AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} "
        f"sens={metrics['sensitivity']:.4f} spec={metrics['specificity']:.4f} "
        f"bal_acc={metrics['balanced_accuracy']:.4f} n={metrics['n_samples']}"
    )
    print(f"  wrote {out_dir/'eval_results.json'}, {scores_path}, {out_dir/('eval_roc_pr_'+args.drug+'.png')}")


def _combine(args: argparse.Namespace) -> None:
    entries: list[tuple[str, np.ndarray, np.ndarray]] = []
    for item in args.combine:
        label, _, path = item.partition("=")
        if not path:  # bare path form: derive label from the npz's stored drug
            path = label
            data = np.load(path, allow_pickle=False)
            label = str(data["drug"])
        else:
            data = np.load(path, allow_pickle=False)
        entries.append((label, data["y_true"], data["y_prob"]))
    plot_roc_pr_grid(entries, args.combine_out)
    print(f"Wrote {len(entries)}-drug ROC|PR grid: {args.combine_out}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, help="Path to a fine-tuned checkpoint directory.")
    p.add_argument("--drug", type=str, help="Drug column to evaluate.")
    p.add_argument("--task", type=str, default="kleb_ast", help="Task slug for the results JSON (default: kleb_ast).")
    p.add_argument("--ast-sheet-path", type=str, help="Path to binary_ast_with_split.csv.")
    p.add_argument("--embeddings-dir", type=str, help="Directory of {sample}_esm_embeddings.pt files.")
    p.add_argument("--n-folds", type=int, default=None, help="If set, derive the fixed evaluate holdout via k-fold.")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--evaluate-seed", type=int, default=1, help="Pins the fixed holdout (match the training run).")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-cuda", action="store_true", help="Force CPU even if a GPU is available.")
    p.add_argument("--out-dir", type=str, default=None, help="Output dir (default: the checkpoint dir).")
    p.add_argument(
        "--combine",
        nargs="+",
        default=None,
        metavar="[LABEL=]NPZ",
        help="Plot mode: combine eval_scores.npz files into one ROC|PR grid (one row each).",
    )
    p.add_argument("--combine-out", type=str, default=None, help="Output PNG for --combine mode.")
    return p


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    if args.combine:
        if not args.combine_out:
            raise SystemExit("--combine requires --combine-out")
        _combine(args)
        return
    missing = [k for k in ("checkpoint", "drug", "ast_sheet_path", "embeddings_dir") if getattr(args, k) is None]
    if missing:
        raise SystemExit(f"Evaluate mode requires: {', '.join('--' + m.replace('_', '-') for m in missing)}")
    _evaluate(args)


if __name__ == "__main__":
    main()
