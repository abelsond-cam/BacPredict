"""Predict AMR R/S calls for every kpsc_final_list sample for one drug.

Reads the v2 metadata TSV, filters to ``kpsc_final_list == True``, drops samples
whose ESM-C embedding file is missing on disk, runs Bacformer inference using
the drug's fine-tuned checkpoint via :func:`tl.train.predict.predict_proba`,
applies the per-drug **Youden's J** threshold (read from the checkpoint dir's
``eval_results.json``), and writes a per-drug parquet with columns:

    Sample, predicted_<drug>_AST_prob, predicted_<drug>_AST

where the AST column is the R/S string call.

Intended to run on an ampere GPU via the SLURM array
``src/kleb_ast/scripts/predict_amr_panel_on_slurm.sh`` (one array task per
drug); the merge step that pulls every drug's parquet into the v2 metadata
table lives in the BacHGT repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tl.train.predict import predict_proba


def _load_youden_threshold(checkpoint_dir: Path, drug: str) -> float:
    """Read the Youden-tuned threshold from the run dir's eval_results.json."""
    eval_json = Path(checkpoint_dir) / "eval_results.json"
    if not eval_json.exists():
        raise SystemExit(
            f"No eval_results.json in {checkpoint_dir}. Run the panel evaluator first "
            f"so {drug} has a Youden operating point."
        )
    payload = json.loads(eval_json.read_text())
    op = payload.get("operating_point") or {}
    thr = op.get("threshold")
    if thr is None:
        raise SystemExit(
            f"No operating_point.threshold in {eval_json}. Re-run the evaluator with "
            "validation-pass support so the Youden threshold is recorded."
        )
    return float(thr)


def _load_kpsc_samples(metadata_tsv: Path) -> list[str]:
    """Return Sample IDs flagged ``kpsc_final_list == True``."""
    # Read just the two columns we need — the v2 TSV is wide.
    df = pd.read_csv(
        metadata_tsv,
        sep="\t",
        low_memory=False,
        usecols=["Sample", "kpsc_final_list"],
        dtype={"Sample": str},
    )
    keep = df[df["kpsc_final_list"].astype(bool)]
    return keep["Sample"].astype(str).tolist()


def _filter_to_have_embeddings(sample_ids: list[str], embeddings_dir: Path) -> tuple[list[str], int]:
    """Drop sample IDs whose ``{sid}_esm_embeddings.pt`` file is absent."""
    emb_dir = Path(embeddings_dir)
    present = [sid for sid in sample_ids if (emb_dir / f"{sid}_esm_embeddings.pt").exists()]
    return present, len(sample_ids) - len(present)


def _binary_call(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Return an R/S string array given probabilities + a decision threshold."""
    return np.where(probabilities >= threshold, "R", "S")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--drug", required=True, help="Drug name (matches checkpoint dir + parquet basename).")
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Run dir containing eval_results.json + a checkpoint-<step>/ subdir.",
    )
    p.add_argument("--metadata-tsv", required=True, help="Path to metadata_v2_all_samples_and_columns.tsv.")
    p.add_argument("--embeddings-dir", required=True)
    p.add_argument("--out", required=True, help="Output parquet path.")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument(
        "--n-samples", type=int, default=None,
        help="Truncate the post-filter sample list to this many entries (canary/debug use only). "
             "Default: None = predict on every kpsc_final_list sample with an embedding.",
    )
    args = p.parse_args()

    print(f"[{args.drug}] loading kpsc_final_list samples from {args.metadata_tsv}")
    kpsc = _load_kpsc_samples(Path(args.metadata_tsv))
    print(f"[{args.drug}]   kpsc_final_list samples: {len(kpsc):,}")

    kept, missing = _filter_to_have_embeddings(kpsc, Path(args.embeddings_dir))
    print(f"[{args.drug}]   with embeddings on disk: {len(kept):,}  (missing: {missing:,})")
    if missing:
        print(
            f"[{args.drug}]   WARNING: {missing:,} kpsc samples lack embeddings; "
            "they will be absent from the parquet and will appear as NaN after merge."
        )
    if args.n_samples is not None and len(kept) > args.n_samples:
        print(f"[{args.drug}]   CANARY MODE: truncating to first {args.n_samples} samples (was {len(kept):,})")
        kept = kept[: args.n_samples]
    if not kept:
        raise SystemExit(f"[{args.drug}] no samples to predict.")

    threshold = _load_youden_threshold(Path(args.checkpoint), args.drug)
    device = "cpu" if (args.no_cuda or not torch.cuda.is_available()) else "cuda"
    print(f"[{args.drug}] Youden threshold = {threshold:.4f}; device = {device}")

    y_prob = predict_proba(
        checkpoint=args.checkpoint,
        sample_ids=kept,
        embeddings_dir=args.embeddings_dir,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    rs = _binary_call(y_prob, threshold)
    n_R = int((rs == "R").sum())
    print(f"[{args.drug}] predicted R: {n_R:,} / {len(rs):,} ({n_R / len(rs):.1%})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "Sample": kept,
        f"predicted_{args.drug}_AST_prob": y_prob.astype(float),
        f"predicted_{args.drug}_AST": rs,
    })
    out_df.to_parquet(out_path, index=False)
    print(f"[{args.drug}] wrote {out_path} ({len(out_df):,} rows)")


if __name__ == "__main__":
    sys.exit(main() or 0)
