"""Shared unlabeled-inference helper.

Apply a fine-tuned binary-classification head (built on top of Bacformer) to a
batch of *unlabeled* sample IDs, returning per-sample positive-class probability
aligned with the input order. Used by per-task deployment scripts that want to
score every isolate in a cohort (e.g. write predicted AST back into a metadata
table).

This is intentionally a thin layer over the evaluator's internals — same
``resolve_checkpoint_dir`` / ``collate_fn`` / ``run_inference`` — so behaviour
matches the evaluator exactly. The only difference is that labels are discarded
because the caller does not have them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from bacpredict.engine.finetune.datasets import LabelInjectingFileDataset
from bacpredict.engine.finetune.evaluate import collate_fn, resolve_checkpoint_dir, run_inference


def predict_proba(
    checkpoint: str | Path,
    sample_ids: list[str],
    embeddings_dir: str | Path,
    device: str = "cpu",
    batch_size: int = 1,
    num_workers: int = 4,
) -> np.ndarray:
    """Run a fine-tuned classification head on unlabeled samples.

    Parameters
    ----------
    checkpoint
        Path to either a model dir containing ``config.json`` or a run dir with
        a ``checkpoint-<step>/`` subdir (the resolver handles both — including
        reading ``trainer_state.best_model_checkpoint`` when several are kept).
    sample_ids
        Sample IDs to score. Each must have a matching
        ``{embeddings_dir}/{sample_id}_esm_embeddings.pt`` file. The caller is
        responsible for pre-filtering to samples with embeddings on disk.
    embeddings_dir
        Directory of ``{sample}_esm_embeddings.pt`` files.
    device
        ``"cuda"`` or ``"cpu"``. The caller picks; on CPU the model is cast to
        float32, on CUDA it inherits the checkpoint's saved dtype.
    batch_size, num_workers
        Standard DataLoader knobs.

    Returns
    -------
    np.ndarray
        Positive-class probabilities (after sigmoid), shape ``(len(sample_ids),)``,
        in the same order as ``sample_ids``.
    """
    sample_ids = list(sample_ids)
    if not sample_ids:
        return np.array([], dtype=float)

    model_dir = resolve_checkpoint_dir(Path(checkpoint))
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

    # Dummy labels — discarded after inference (LabelInjectingFileDataset requires
    # a label_map, but the returned y_true is ignored here).
    label_map = dict.fromkeys(sample_ids, 0)
    dataset = LabelInjectingFileDataset(
        sample_ids, Path(embeddings_dir), label_map, label_column="__predict__"
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    _y_true, y_prob = run_inference(model, loader, device)
    # run_inference returns float in [0, 1] already (sigmoid applied).
    return np.asarray(y_prob, dtype=float)


def main() -> None:
    """Tiny CLI for ad-hoc verification: predict_proba over a sample-id list file."""
    import argparse

    p = argparse.ArgumentParser(description="Run predict_proba over a list of sample IDs (one per line).")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--sample-ids-file", required=True, help="Text file with one sample ID per line.")
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out", default=None, help="Optional output .npy of probabilities.")
    args = p.parse_args()

    ids = [ln.strip() for ln in Path(args.sample_ids_file).read_text().splitlines() if ln.strip()]
    device = "cpu" if (args.no_cuda or not torch.cuda.is_available()) else "cuda"
    y_prob = predict_proba(
        args.checkpoint, ids, args.embeddings_dir,
        device=device, batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"Predicted {len(y_prob)} samples; min={y_prob.min():.4f} max={y_prob.max():.4f} mean={y_prob.mean():.4f}")
    if args.out:
        np.save(args.out, y_prob)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
