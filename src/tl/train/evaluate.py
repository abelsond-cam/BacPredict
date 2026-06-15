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
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from tl.train.datasets import LabelInjectingFileDataset
from tl.train.metrics import build_results_payload, compute_full_metrics, write_results_json, youden_threshold
from tl.train.split_utils import generate_kfold_splits

# The file-based dataset + DataLoader workers open many .pt files; the default
# file_descriptor sharing strategy exhausts FDs on large evaluate splits
# ("Too many open files"). file_system shares via temp files instead.
torch.multiprocessing.set_sharing_strategy("file_system")


def collate_fn(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a list of per-sample dicts into a batch (matches train_amr.py)."""
    prot_list = [s["protein_embeddings"].squeeze(0) for s in samples]
    am_list = [s["attention_mask"].squeeze(0) for s in samples]
    contig_list = [s["contig_ids"].squeeze(0) for s in samples]
    batch = {
        "protein_embeddings": pad_sequence(prot_list, batch_first=True, padding_value=0.0),
        "labels": torch.stack([s["labels"] for s in samples], dim=0),
        "attention_mask": pad_sequence(am_list, batch_first=True, padding_value=0.0),
        "contig_ids": pad_sequence(contig_list, batch_first=True, padding_value=0),
    }
    # Pass the surprisal panel through when present (panel-mode attention checkpoints).
    if "panel" in samples[0]:
        batch["panel"] = pad_sequence([s["panel"].squeeze(0) for s in samples], batch_first=True, padding_value=0.0)
    return batch


def resolve_holdouts(
    ast_sheet_path: str,
    drug: str,
    n_folds: int | None,
    fold: int,
    seed: int,
    evaluate_seed: int,
) -> tuple[list[str], list[str], dict[str, int], str]:
    """Reconstruct (evaluate_ids, validation_ids, label_map, source) for a drug.

    Mirrors ``train_amr.py``: k-fold mode derives the fixed evaluate holdout from
    ``evaluate_seed`` and the validation set from ``folds[fold]`` (with ``seed``);
    CSV mode reads ``train_val_eval``. Validation is needed to pick an operating
    threshold without peeking at the evaluate set.
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
    order = labeled["Sample"].tolist()

    if n_folds is not None:
        evaluate_set, folds = generate_kfold_splits(labeled, n_folds=n_folds, seed=seed, evaluate_seed=evaluate_seed)
        _, val_set = folds[fold]
        evaluate_ids = [sid for sid in order if sid in evaluate_set]
        validation_ids = [sid for sid in order if sid in val_set]
        return evaluate_ids, validation_ids, label_map, "kfold"

    if "train_val_eval" not in labeled.columns:
        raise ValueError("CSV has no 'train_val_eval' column; pass --n-folds to derive the holdout.")
    evaluate_ids = labeled[labeled["train_val_eval"] == "evaluate"]["Sample"].tolist()
    validation_ids = labeled[labeled["train_val_eval"] == "validate"]["Sample"].tolist()
    return evaluate_ids, validation_ids, label_map, "csv"


def resolve_evaluate_ids(
    ast_sheet_path: str,
    drug: str,
    n_folds: int | None,
    seed: int,
    evaluate_seed: int,
) -> tuple[list[str], dict[str, int], str]:
    """Back-compat shim: evaluate IDs + label map + source (no validation set).

    Evaluate IDs are independent of ``fold``, so fold 0 is used internally.
    """
    evaluate_ids, _validation_ids, label_map, source = resolve_holdouts(
        ast_sheet_path, drug, n_folds, fold=0, seed=seed, evaluate_seed=evaluate_seed
    )
    return evaluate_ids, label_map, source


def resolve_checkpoint_dir(checkpoint: Path) -> Path:
    """Return the dir holding the best model's files.

    Trainer saves weights under ``checkpoint-<step>/`` (the step varies per run —
    early stopping decides it), while the run dir itself holds only
    ``results.json`` + ``runs/``. Resolution order:

    1. ``checkpoint`` itself has ``config.json`` → use it.
    2. The Trainer's ``best_model_checkpoint`` (from any ``trainer_state.json``),
       matched by basename so a stale absolute path from another host still works.
    3. Fallback: highest-step ``checkpoint-*`` containing ``config.json``.

    With ``save_total_limit=1`` + ``load_best_model_at_end=True`` only the best
    checkpoint survives, so (2) and (3) agree; (2) matters if multiple are kept.
    """
    checkpoint = Path(checkpoint)
    if (checkpoint / "config.json").exists():
        return checkpoint

    def _step(p: Path) -> int:
        tail = p.name.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else -1

    candidates = sorted(
        (p for p in checkpoint.glob("checkpoint-*") if (p / "config.json").exists()),
        key=_step,
    )
    if not candidates:
        raise FileNotFoundError(f"No config.json in {checkpoint} or any checkpoint-*/ subdir.")

    by_name = {p.name: p for p in candidates}
    for c in candidates:
        state = c / "trainer_state.json"
        if not state.exists():
            continue
        try:
            best = json.loads(state.read_text()).get("best_model_checkpoint")
        except (json.JSONDecodeError, OSError):
            best = None
        if best and Path(best).name in by_name:
            return by_name[Path(best).name]
        break

    return candidates[-1]


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
            extra = {}
            if "panel" in batch:
                extra["panel"] = batch["panel"].to(device=device, dtype=dtype)
            out = model(protein_embeddings=pe, attention_mask=am, contig_ids=cid, **extra)
            logits = out.logits.reshape(-1).float().cpu()
            y_prob_parts.append(torch.sigmoid(logits))
            y_true_parts.append(labels.reshape(-1).cpu())
    return (
        torch.cat(y_true_parts).numpy().astype(int),
        torch.cat(y_prob_parts).numpy().astype(float),
    )


def plot_roc_pr_grid(
    entries: list[tuple],
    out_path: Path | str,
    prevalence_label: str = "prevalence",
) -> None:
    """Render an N-row x 2-col grid: ROC (left) and PR (right) per drug.

    Parameters
    ----------
    entries
        List of ``(label, y_true, y_prob)`` or ``(label, y_true, y_prob, threshold)``
        tuples — one row per entry. When a finite ``threshold`` is given, the
        Youden operating point is marked on both panels.
    out_path
        Destination PNG.
    prevalence_label
        Reader-facing name for the positive-class base rate on the PR panel
        (e.g. ``"resistance rate"`` for AST, ``"blood source ratio"`` for isolation source).
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
    for i, entry in enumerate(entries):
        label, y_true, y_prob, *rest = entry
        threshold = rest[0] if rest else None
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)
        ax_roc, ax_pr = axes[i]

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auroc = roc_auc_score(y_true, y_prob)
        ax_roc.plot(fpr, tpr, lw=2, color="C0", label=f"AUROC = {auroc:.3f}")
        ax_roc.plot([0, 1], [0, 1], ls="--", lw=1, color="grey")

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        prevalence = float(np.mean(y_true))
        ax_pr.plot(rec, prec, lw=2, color="C1", label=f"AUPRC = {auprc:.3f}")
        ax_pr.axhline(prevalence, ls="--", lw=1, color="grey", label=f"{prevalence_label} = {prevalence:.3f}")

        if threshold is not None and np.isfinite(threshold):
            y_pred = (y_prob >= threshold).astype(int)
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            sens = tp / (tp + fn) if (tp + fn) else 0.0
            spec = tn / (tn + fp) if (tn + fp) else 0.0
            prec_pt = tp / (tp + fp) if (tp + fp) else 0.0
            ax_roc.plot(
                1 - spec, sens, "o", color="C3", ms=7,
                label=f"J* @ {threshold:.2f} (sens {sens:.2f}, spec {spec:.2f})",
            )
            ax_pr.plot(sens, prec_pt, "o", color="C3", ms=7, label=f"J* @ {threshold:.2f}")

        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1.02)
        ax_roc.set_xlabel("False positive rate")
        ax_roc.set_ylabel("True positive rate")
        ax_roc.set_title(f"{label} — ROC")
        ax_roc.legend(loc="lower right")

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


def _build_loader(ids: list[str], args: argparse.Namespace, label_map: dict[str, int]) -> DataLoader:
    dataset = LabelInjectingFileDataset(ids, Path(args.embeddings_dir), label_map, args.drug)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )


def plot_auroc_bar(
    entries: list[tuple],
    out_path: Path | str,
    ylim: tuple[float, float] = (0.5, 1.05),
    title: str | None = None,
    colorbar_label: str | None = None,
    cmap: str = "YlOrRd",
) -> None:
    """Single-panel summary: AUROC per drug as a sorted vertical bar chart.

    Parameters
    ----------
    entries
        Same shape as :func:`plot_roc_pr_grid` — ``(label, y_true, y_prob, [threshold])``.
        Sorted internally by AUROC descending.
    out_path
        Destination PNG.
    ylim
        Y-axis limits. Default ``(0.5, 1.05)`` — 1.05 leaves room above the bars
        for the value labels.
    title
        Optional figure title.
    colorbar_label
        If given, color each bar by its positive-class prevalence
        (``y_true.mean()``) using ``cmap`` and attach a colorbar with this label
        (e.g. ``"resistance rate"`` for AST, ``"blood source ratio"`` for
        isolation source). When ``None`` the bars use a single colour.
    cmap
        Matplotlib colormap name when ``colorbar_label`` is set. Default
        ``"YlOrRd"`` (light at 0 → dark red at 1).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from sklearn.metrics import roc_auc_score

    items = []
    for entry in entries:
        label, y_true, y_prob, *_ = entry
        yt = np.asarray(y_true).astype(int)
        yp = np.asarray(y_prob).astype(float)
        items.append((label, float(roc_auc_score(yt, yp)), float(yt.mean())))
    items.sort(key=lambda x: -x[1])
    labels = [x[0] for x in items]
    aurocs = [x[1] for x in items]
    rates = [x[2] for x in items]

    fig, ax = plt.subplots(figsize=(max(8.0, 0.45 * len(items) + 2.5), 5.0))

    if colorbar_label is not None:
        norm = Normalize(vmin=0.0, vmax=1.0)
        cmap_obj = colormaps[cmap]
        ax.bar(range(len(items)), aurocs, color=[cmap_obj(norm(r)) for r in rates], edgecolor="0.4", linewidth=0.5)
        sm = ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=colorbar_label, fraction=0.025, pad=0.02)
    else:
        ax.bar(range(len(items)), aurocs, color="C0")

    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(*ylim)
    ax.set_ylabel("AUROC")
    if title:
        ax.set_title(title)
    for i, v in enumerate(aurocs):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _evaluate(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else checkpoint
    device = "cpu" if (args.no_cuda or not torch.cuda.is_available()) else "cuda"

    evaluate_ids, validation_ids, label_map, split_source = resolve_holdouts(
        args.ast_sheet_path, args.drug, args.n_folds, args.fold, args.seed, args.evaluate_seed
    )
    print(
        f"Drug {args.drug}: {len(evaluate_ids)} evaluate / {len(validation_ids)} validation "
        f"samples ({split_source} split), device={device}"
    )
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

    y_true, y_prob = run_inference(model, _build_loader(evaluate_ids, args, label_map), device)
    metrics = compute_full_metrics(y_true, y_prob)

    # Operating point: pick Youden's J threshold on validation, report it on evaluate.
    operating_point = None
    opt_thr = None
    if validation_ids:
        yv_true, yv_prob = run_inference(model, _build_loader(validation_ids, args, label_map), device)
        opt_thr = youden_threshold(yv_true, yv_prob)
        op_metrics = compute_full_metrics(y_true, y_prob, threshold=opt_thr)
        operating_point = {
            "objective": "youden_j",
            "selected_on": "validation",
            "threshold": opt_thr,
            "sensitivity": op_metrics["sensitivity"],
            "specificity": op_metrics["specificity"],
            "balanced_accuracy": op_metrics["balanced_accuracy"],
            "f1": op_metrics["f1"],
            "confusion_matrix": op_metrics["confusion_matrix"],
        }
    else:
        print("  no validation samples — skipping operating-point tuning.")

    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "eval_scores.npz"
    np.savez(
        scores_path,
        y_true=y_true,
        y_prob=y_prob,
        drug=np.array(args.drug),
        operating_threshold=np.array(opt_thr if opt_thr is not None else np.nan),
    )

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
        operating_point=operating_point,
    )
    write_results_json(out_dir / "eval_results.json", payload)
    plot_roc_pr_grid(
        [(args.drug, y_true, y_prob, opt_thr)],
        out_dir / f"eval_roc_pr_{args.drug}.png",
        prevalence_label=args.prevalence_label,
    )

    print(
        f"  @0.5: AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} "
        f"sens={metrics['sensitivity']:.4f} spec={metrics['specificity']:.4f} "
        f"bal_acc={metrics['balanced_accuracy']:.4f} n={metrics['n_samples']}"
    )
    if operating_point is not None:
        print(
            f"  @J*={opt_thr:.3f}: sens={operating_point['sensitivity']:.4f} "
            f"spec={operating_point['specificity']:.4f} bal_acc={operating_point['balanced_accuracy']:.4f}"
        )
    print(f"  wrote {out_dir/'eval_results.json'}, {scores_path}, {out_dir/('eval_roc_pr_'+args.drug+'.png')}")


def _combine(args: argparse.Namespace) -> None:
    entries: list[tuple] = []
    for item in args.combine:
        label, _, path = item.partition("=")
        if not path:  # bare path form: derive label from the npz's stored drug
            path = label
            data = np.load(path, allow_pickle=False)
            label = str(data["drug"])
        else:
            data = np.load(path, allow_pickle=False)
        thr = float(data["operating_threshold"]) if "operating_threshold" in data.files else None
        entries.append((label, data["y_true"], data["y_prob"], thr))
    plot_roc_pr_grid(entries, args.combine_out, prevalence_label=args.prevalence_label)
    print(f"Wrote {len(entries)}-drug ROC|PR grid: {args.combine_out}")
    if args.bar_out:
        plot_auroc_bar(entries, args.bar_out, title=args.bar_title, colorbar_label=args.prevalence_label)
        print(f"Wrote AUROC bar chart: {args.bar_out}")


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
        "--prevalence-label",
        type=str,
        default="prevalence",
        help='Reader-facing PR base-rate label, e.g. "resistance rate" (AST) or "blood source ratio".',
    )
    p.add_argument(
        "--combine",
        nargs="+",
        default=None,
        metavar="[LABEL=]NPZ",
        help="Plot mode: combine eval_scores.npz files into one ROC|PR grid (one row each).",
    )
    p.add_argument("--combine-out", type=str, default=None, help="Output PNG for --combine mode.")
    p.add_argument(
        "--bar-out",
        type=str,
        default=None,
        help="Optional in --combine mode: also write a single-panel AUROC bar chart (sorted desc, y=0.5..1.0).",
    )
    p.add_argument("--bar-title", type=str, default=None, help="Title for the --bar-out figure.")
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
