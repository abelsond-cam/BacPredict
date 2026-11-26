r"""Score one named split of a split table with an already-trained checkpoint, keeping the **logits**.

This exists for post-hoc calibration. A run that has finished writes per-sample scores for its holdouts and
nothing for ``validate``, which is the split a one-parameter calibration must be fitted on
(:func:`bacpredict.engine.finetune.censored_mic_loss.fit_logit_temperature`). Without it an arm trained
before that was saved can only be compared *untempered* against one that was tempered — which credits the
estimator with the scaling rather than measuring it.

⚠ **It keeps the logit, not just the probability.** ``sigmoid`` saturates to exactly 1.0 in float32 from a
logit of about 17, and these models are confident, so inverting ``y_prob`` back to a logit gives ``inf`` for
precisely the genomes that determine the temperature. The logit is written as ``score``, the same key
``finetune_amr`` now writes.

Inference only: no gradients, no optimizer, nothing written into the scored run's own directory. GPU sbatch
rather than a login-node job — this is Bacformer-large over ~1,000 genomes per drug.

Usage::

    python -m bacpredict.engine.finetune.score_split \
        --checkpoint  $P/models/<drug> \
        --split-table $P/splits/<drug>_split.csv --split validate \
        --embeddings-dir $AMR/processed/esm \
        --out $Q/arm0_validate/<drug>_validate_scores.npz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

from bacpredict.engine.finetune.datasets import LabelInjectingFileDataset
from bacpredict.engine.finetune.evaluate import collate_fn, resolve_checkpoint_dir
from bacpredict.engine.splits.load_splits import SPLIT_VALUES, load_split_ids, load_splits

logger = logging.getLogger(__name__)


def run_inference_with_logits(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """``(y_true, logits)`` over the loader, in its order.

    Mirrors :func:`bacpredict.engine.finetune.evaluate.run_inference` exactly but returns the logit rather
    than the sigmoid of it, so nothing is lost to saturation before a calibration fit sees it.
    """
    model.eval()
    dtype = next(model.parameters()).dtype
    y_true_parts, logit_parts = [], []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            out = model(
                protein_embeddings=batch["protein_embeddings"].to(device=device, dtype=dtype),
                attention_mask=batch["attention_mask"].to(device=device, dtype=dtype),
                contig_ids=batch["contig_ids"].to(device=device),
            )
            logit_parts.append(out.logits.reshape(-1).float().cpu())
            y_true_parts.append(labels.reshape(-1).cpu())
    return torch.cat(y_true_parts).numpy().astype(int), torch.cat(logit_parts).numpy().astype(float)


def score_split(
    checkpoint: str | Path,
    split_table: str | Path,
    split: str,
    embeddings_dir: str | Path,
    *,
    device: str | None = None,
    num_workers: int = 8,
    species: str | None = None,
) -> dict[str, np.ndarray]:
    """Score ``split`` of ``split_table`` with ``checkpoint``; return the arrays to save.

    The split and its labels come from ``engine.splits`` alone, as everywhere else, so this scores exactly
    the genomes the training run called ``validate`` and in the same order.
    """
    if split not in SPLIT_VALUES:
        raise ValueError(f"--split must be one of {SPLIT_VALUES}, got {split!r}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    label_map, *_ = load_splits(split_table, species=species)
    ids = [s for s in load_split_ids(split_table, split, species=species) if s in label_map]
    if not ids:
        raise RuntimeError(f"{split_table}: no clean 0/1 rows in split {split!r}")

    model_dir = resolve_checkpoint_dir(Path(checkpoint))
    logger.info("scoring %d %s genomes with %s on %s", len(ids), split, model_dir.name, device)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), num_labels=1, problem_type="binary_classification", return_dict=True,
        trust_remote_code=True, torch_dtype="auto")
    if device == "cpu":
        model = model.float()
    model = model.to(device)

    dataset = LabelInjectingFileDataset(ids, Path(embeddings_dir), label_map, split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    y_true, logits = run_inference_with_logits(model, loader, device)
    expected = np.asarray([label_map[s] for s in ids], dtype=int)
    if not np.array_equal(y_true, expected):
        raise RuntimeError("inference returned labels that are not the split table's, in order")
    return {
        "y_true": y_true,
        "y_prob": 1.0 / (1.0 + np.exp(-logits)),
        "score": logits,
        "sample_ids": np.asarray(ids, dtype=np.str_),
        "split": np.asarray([split] * len(ids), dtype=np.str_),
        "operating_threshold": np.array(np.nan),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: score one split of a split table with a trained checkpoint, keeping the logits."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="a run dir or a checkpoint-<step> dir")
    p.add_argument("--split-table", required=True)
    p.add_argument("--split", default="validate")
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--species", help="restrict a cross-species table to one species' rows")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--out", required=True, help="where to write the npz (never inside the scored run)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arrays = score_split(args.checkpoint, args.split_table, args.split, args.embeddings_dir,
                         num_workers=args.num_workers, species=args.species)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, drug=np.array(Path(args.split_table).name.replace("_split.csv", "")),
             species=np.asarray([args.species or ""] * len(arrays["y_true"]), dtype=np.str_), **arrays)
    logger.info("wrote %s (%d genomes, %d resistant)", out, len(arrays["y_true"]), int(arrays["y_true"].sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
