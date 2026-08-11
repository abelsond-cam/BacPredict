r"""Score every labelled genome in a cohort with a deployed checkpoint, split labels retained.

``evaluate.py`` scores only the held-out split, which is the right thing for a headline metric but
too small to read *within-stratum* behaviour: a sublineage with 2,300 genomes in the cohort
contributes only ~470 to a 20% holdout, and a clonal group may not clear a useful count at all.
This scores the whole cohort so per-stratum tables can be built at full n.

**The train rows are not a held-out measurement and must never be reported as one.** The model was
fitted on them, so their AUROC is optimistically biased — every output here carries the split label
per genome precisely so downstream tables can separate `train` from `evaluate` rather than blend
them into one misleading number. Use the whole-cohort view to ask whether a *pattern* across strata
is stable at larger n; use the `evaluate` rows alone for any number that is quoted.

Writes an ``eval_scores.npz``-compatible archive (``sample_ids``/``y_true``/``y_prob``, plus a
``split`` array), so :mod:`bacpredict.engine.finetune.stratified_metrics` consumes it directly.

GPU sbatch, not a login-node job — this is Bacformer-large inference over ~14k genomes.

Usage::

    python -m bacpredict.engine.finetune.score_cohort \
        --checkpoint  <cohort>/models \
        --split-csv   <cohort>/binary_blood_vs_faeces_with_split.csv \
        --label-column blood_vs_faeces_label \
        --embeddings-dir .../klebsiella_esm_embeddings \
        --out <cohort>/models/cohort_scores.npz
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bacpredict.engine.finetune.metrics import compute_full_metrics
from bacpredict.engine.finetune.predict import predict_proba

logger = logging.getLogger(__name__)


def load_cohort(split_csv: Path, label_column: str, embeddings_dir: Path) -> pd.DataFrame:
    """Return the labelled cohort rows that have an embedding on disk.

    Genomes without an embedding are dropped and *counted* — a silent drop here would quietly
    change which genomes a per-stratum table is describing.
    """
    df = pd.read_csv(split_csv, low_memory=False)
    for col in ("Sample", label_column, "train_val_eval"):
        if col not in df.columns:
            raise ValueError(f"{split_csv} is missing required column {col!r}")
    df["Sample"] = df["Sample"].astype(str)
    df = df[df[label_column].isin([0, 1])].drop_duplicates(subset="Sample", keep="first")

    present = [(embeddings_dir / f"{s}_esm_embeddings.pt").exists() for s in df["Sample"]]
    n_missing = int(len(df) - sum(present))
    if n_missing:
        logger.warning("  %d/%d labelled genomes have no embedding on disk — dropped", n_missing, len(df))
    return df[present].reset_index(drop=True)


def _main_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split-csv", type=Path, required=True)
    p.add_argument("--label-column", type=str, required=True)
    p.add_argument("--embeddings-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-cuda", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cohort = load_cohort(args.split_csv, args.label_column, args.embeddings_dir)
    ids = cohort["Sample"].tolist()
    y_true = cohort[args.label_column].to_numpy().astype(int)
    split = cohort["train_val_eval"].to_numpy().astype(str)
    logger.info(
        "scoring %d genomes (train %d / validate %d / evaluate %d)",
        len(ids), (split == "train").sum(), (split == "validate").sum(), (split == "evaluate").sum(),
    )

    device = "cpu" if (args.no_cuda or not torch.cuda.is_available()) else "cuda"
    y_prob = predict_proba(
        args.checkpoint, ids, args.embeddings_dir,
        device=device, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        y_true=y_true, y_prob=y_prob,
        sample_ids=np.asarray(ids, dtype=np.str_),
        split=np.asarray(split, dtype=np.str_),
        drug=np.array(args.label_column),
        operating_threshold=np.array(np.nan),
    )

    summary = {"n_scored": len(ids), "checkpoint": str(args.checkpoint), "device": device, "by_split": {}}
    for name in ("train", "validate", "evaluate"):
        m = split == name
        if m.sum() and len(np.unique(y_true[m])) > 1:
            met = compute_full_metrics(y_true[m], y_prob[m])
            summary["by_split"][name] = {"n": int(m.sum()), "auroc": met["auroc"], "auprc": met["auprc"]}
            print(f"  {name:<9} n={int(m.sum()):<6} AUROC={met['auroc']:.4f}")
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")
    print("NOTE: the train AUROC above is fitted-on data, not a measurement. Quote 'evaluate' only.")


if __name__ == "__main__":
    _main_cli()
