"""Test A — score a trained attention-pool checkpoint on the FULL canonical AST holdout.

The 1000-genome **manifest** baseline (frozen gated-MIL, ``panel_mode="none"``, job ``30602029``)
scored eval AUROC **0.9768** on the manifest's own 200-genome holdout — far above the full-38k
localization ladder for the *same architecture* (mean 0.788, mean-pool FT 0.905, e2e gated-MIL
0.868, frozen gated-MIL ~0.78). A 0.78→0.977 jump is the open confound-vs-bug question
(``memory/tb_mini_set_0977_confound_vs_bug.md``; plan
``~/.claude/plans/i-d-like-to-start-crystalline-allen.md``). This entrypoint runs that **same
trained checkpoint** over the full cohort's canonical 20% ``evaluate`` holdout — the
non-confounded scoreboard:

- stays ~0.97  ⇒ the read-out is genuinely that good (**H2**, paradigm-shifting);
- drops to ~0.8 ⇒ the manifest number was a lineage/selection confound (**H1**).

Two guards make the number trustworthy:

1. **No train-on-eval leakage.** The manifest's ~1000 genomes were drawn from the *whole* cohort
   ignoring the full split, so ~20% of the manifest's train+validate genomes land in the full
   ``evaluate`` holdout. Pass ``--manifest-split-csv`` and those genomes are **excluded** from the
   scored set, so the AUROC is honest generalisation to genomes the checkpoint never saw.
2. **Path self-check.** With ``--manifest-split-csv`` we also re-score the manifest's *own* 200
   ``evaluate`` rows and report that AUROC — it must reproduce ~0.9768, proving this eval path
   matches the training-time evaluator before the full-eval number is trusted.

Only handles ``panel_mode="none"`` checkpoints: a panel checkpoint (``att_head``/``e2e``) needs a
per-protein surprisal panel for every eval genome, which is the deferred full-38k scan (plan D).
Read-only, GPU inference — one backbone forward per genome.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pangena_predict.bacformer_genome_vectors import _forward_inputs
from pangena_predict.head_pool_attention_probe import load_attn_pool_checkpoint
from pangena_predict.snp_vs_esm_prediction import resolve_clean_splits
from tl.train.metrics import (
    build_results_payload,
    compute_full_metrics,
    write_results_json,
    youden_threshold,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_manifest_splits(
    manifest_split_csv: str | Path, drug: str
) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    """Read the 1000-genome manifest split CSV → (train, validate, evaluate ids, label_map).

    The manifest split CSV (``tb_rif_1000_split.csv``) has the training-architecture columns
    ``Sample`` / ``<drug>`` / ``train_val_eval`` — the exact 700/100/200 split the manifest
    checkpoint trained on. The train+validate ids are the genomes the checkpoint *saw* (excluded
    from the full-eval scoring); the evaluate ids are its own 200-genome holdout (the path
    self-check). Only clean 0/1 labels are kept.
    """
    df = pd.read_csv(manifest_split_csv, low_memory=False)
    if "Sample" not in df.columns:
        if "phenotype-BioSample_ID" not in df.columns:
            raise ValueError("Manifest split CSV must contain 'Sample' or 'phenotype-BioSample_ID'.")
        df["Sample"] = df["phenotype-BioSample_ID"].astype(str)
    df["Sample"] = df["Sample"].astype(str)
    if drug not in df.columns:
        raise ValueError(f"Drug column {drug!r} not in manifest split CSV; has {list(df.columns)[:20]}")
    if "train_val_eval" not in df.columns:
        raise ValueError("Manifest split CSV has no 'train_val_eval' column.")

    clean = df[df[drug].isin([0, 1])].drop_duplicates(subset="Sample", keep="first")
    label_map = {row["Sample"]: int(row[drug]) for _, row in clean.iterrows()}
    by_split = {s: clean[clean["train_val_eval"] == s]["Sample"].tolist() for s in ("train", "validate", "evaluate")}
    return by_split["train"], by_split["validate"], by_split["evaluate"], label_map


def select_eval_ids(eval_ids: list[str], exclude_ids: set[str]) -> tuple[list[str], int]:
    """Drop any ``exclude_ids`` (manifest train+validate) from the full evaluate holdout.

    Returns ``(kept_ids, n_excluded)`` preserving the original order — so the scored set is the
    full canonical holdout minus the genomes the manifest checkpoint trained on.
    """
    kept = [s for s in eval_ids if s not in exclude_ids]
    return kept, len(eval_ids) - len(kept)


def _score_ids(
    model: torch.nn.Module,
    esm_store_dir: Path,
    ids: list[str],
    label_map: dict[str, int],
    *,
    device: str,
    pt_suffix: str = "_esm_embeddings.pt",
    log_every: int = 1000,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    """Forward the attention-pool checkpoint over ``ids`` → (y_true, y_prob, kept_ids, skips).

    One genome per forward (batch=1), mirroring the diagnostic probes: load the plain per-protein
    ``.pt``, build the backbone kwargs with :func:`_forward_inputs`, read ``output.logits`` and
    sigmoid it. Samples with no ``.pt`` on disk are skipped (counted in ``skips``).
    """
    model_dtype = next(model.parameters()).dtype
    y_true: list[int] = []
    y_prob: list[float] = []
    kept: list[str] = []
    skips: dict[str, int] = {}
    for i, sample_id in enumerate(ids):
        pt_path = esm_store_dir / f"{sample_id}{pt_suffix}"
        if not pt_path.exists():
            skips["missing_pt"] = skips.get("missing_pt", 0) + 1
            continue
        store = torch.load(pt_path, map_location="cpu")
        inputs = _forward_inputs(store, device, model_dtype)
        with torch.inference_mode():
            out = model(**inputs)
        prob = torch.sigmoid(out.logits.float().view(-1)[0]).item()
        y_true.append(int(label_map[sample_id]))
        y_prob.append(float(prob))
        kept.append(sample_id)
        if log_every and (i + 1) % log_every == 0:
            logger.info("  scored %d/%d genomes", i + 1, len(ids))
    if skips:
        logger.warning("scoring skipped %s", skips)
    return np.asarray(y_true), np.asarray(y_prob), kept, skips


def evaluate_checkpoint(
    *,
    ast_sheet_path: Path,
    esm_store_dir: Path,
    checkpoint_dir: str,
    drug: str,
    device: str,
    manifest_split_csv: Path | None,
    max_samples: int | None,
    pt_suffix: str = "_esm_embeddings.pt",
) -> dict:
    """Score the checkpoint on the full evaluate holdout (+ manifest-eval reproduction).

    Returns a dict with the full-eval ``metrics`` block (§0.4), an eval-fold-tuned operating point
    (no separate validate pass), the manifest-eval reproduction AUROC (if a manifest CSV is given),
    and split/exclusion counts.
    """
    label_map, _train_ids, _validate_ids, evaluate_ids, split_info = resolve_clean_splits(ast_sheet_path, drug)

    exclude_ids: set[str] = set()
    manifest_eval_ids: list[str] = []
    manifest_label_map: dict[str, int] = {}
    if manifest_split_csv is not None:
        m_train, m_val, manifest_eval_ids, manifest_label_map = load_manifest_splits(manifest_split_csv, drug)
        exclude_ids = set(m_train) | set(m_val)
        logger.info(
            "manifest split: train=%d validate=%d evaluate=%d → excluding %d seen genomes from full eval",
            len(m_train), len(m_val), len(manifest_eval_ids), len(exclude_ids),
        )

    eval_ids, n_excluded = select_eval_ids(evaluate_ids, exclude_ids)
    if max_samples is not None:
        eval_ids = eval_ids[:max_samples]
        manifest_eval_ids = manifest_eval_ids[: max_samples // 2]

    model = load_attn_pool_checkpoint(checkpoint_dir, device)
    panel_mode = getattr(model.config, "panel_mode", "none")
    if panel_mode not in (None, "none"):
        raise ValueError(
            f"checkpoint panel_mode={panel_mode!r}; this entrypoint scores bare (panel_mode=none) "
            "checkpoints — a panel checkpoint needs a per-protein surprisal panel for every full-eval "
            "genome, which is the deferred full-38k scan (plan D)."
        )

    # Path self-check: reproduce the manifest's own 200-genome eval AUROC (expect ~0.9768).
    manifest_repro: dict | None = None
    if manifest_eval_ids:
        logger.info("Path self-check — scoring the manifest's own %d evaluate genomes", len(manifest_eval_ids))
        y_t, y_p, kept, _ = _score_ids(
            model, esm_store_dir, manifest_eval_ids, manifest_label_map, device=device, pt_suffix=pt_suffix
        )
        m = compute_full_metrics(y_t, y_p)
        manifest_repro = {"auroc": m["auroc"], "auprc": m["auprc"], "n_scored": len(kept)}
        logger.info("manifest-eval reproduction AUROC=%.4f (expect ~0.9768) on %d genomes", m["auroc"], len(kept))

    # Decisive: full canonical evaluate holdout (minus manifest-seen genomes). The eval fold IS the
    # result — we do NOT score the validate split just to set a threshold (that was a second full
    # forward pass over thousands of genomes, delaying the headline AUROC by ~30 min for nothing).
    # AUROC/AUPRC are threshold-free, so compute and log them the instant scoring finishes.
    logger.info("Scoring the full canonical evaluate holdout — %d genomes (excluded %d seen)", len(eval_ids), n_excluded)
    y_true, y_prob, kept_eval, eval_skips = _score_ids(
        model, esm_store_dir, eval_ids, label_map, device=device, pt_suffix=pt_suffix
    )
    if len(kept_eval) == 0:
        raise RuntimeError("No evaluate genomes scored — check esm_store_dir / .pt suffix.")

    metrics = compute_full_metrics(y_true, y_prob)
    logger.info(
        "FULL-EVAL AUROC=%.4f AUPRC=%.4f on %d genomes (prevalence %.3f) — manifest-eval was 0.9768",
        metrics["auroc"], metrics["auprc"], metrics["n_samples"], metrics["prevalence"],
    )

    # Operating point tuned on the eval fold itself (Youden) — a labelled convenience for the §0.4
    # sensitivity/specificity figures, mildly optimistic by construction. No extra scoring pass.
    operating_point = None
    if np.unique(y_true).size == 2:
        thr = youden_threshold(y_true, y_prob)
        op = compute_full_metrics(y_true, y_prob, threshold=thr)
        operating_point = {
            "threshold": thr,
            "source": "evaluate_youden_selftuned",
            "sensitivity": op["sensitivity"],
            "specificity": op["specificity"],
            "balanced_accuracy": op["balanced_accuracy"],
            "f1": op["f1"],
        }
    return {
        "metrics": metrics,
        "operating_point": operating_point,
        "manifest_eval_reproduction": manifest_repro,
        "split_info": split_info,
        "n_excluded_manifest_seen": n_excluded,
        "n_full_eval_scored": len(kept_eval),
        "eval_skips": eval_skips,
        "panel_mode": panel_mode,
    }


def main() -> None:
    """CLI entry point — score a no-panel attention-pool checkpoint on the full AST holdout."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ast-sheet-path", type=Path, required=True,
                        help="binary_ast_with_split.csv — the full canonical 70/10/20 cohort.")
    parser.add_argument("--esm-store-dir", type=Path, required=True, help="Dir of {sample}_esm_embeddings.pt.")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Trained attention-pool run dir (best checkpoint-<step> auto-resolved).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--drug", type=str, default="rifampin", help="Phenotype column (default rifampin).")
    parser.add_argument("--manifest-split-csv", type=Path, default=None,
                        help="tb_rif_1000_split.csv — its train+validate ids are EXCLUDED from the full "
                             "eval (no leakage) and its evaluate ids re-scored as a path self-check (~0.9768).")
    parser.add_argument("--label", type=str, default=None, help="Checkpoint label for the results JSON (default: dir name).")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap genomes per split (smoke).")
    args = parser.parse_args()

    label = args.label or Path(args.checkpoint_dir).name
    result = evaluate_checkpoint(
        ast_sheet_path=args.ast_sheet_path,
        esm_store_dir=args.esm_store_dir,
        checkpoint_dir=args.checkpoint_dir,
        drug=args.drug,
        device=args.device,
        manifest_split_csv=args.manifest_split_csv,
        max_samples=args.max_samples,
    )

    payload = build_results_payload(
        task="pangena_predict",
        drug=args.drug,
        model_name_or_path="BacformerAttnPoolForGenomeClassification",
        checkpoint_dir=str(args.checkpoint_dir),
        split_source=f"{args.ast_sheet_path.name} (evaluate, manifest-seen excluded)",
        metrics=result["metrics"],
        n_evaluate=result["n_full_eval_scored"],
        operating_point=result["operating_point"],
        extra={
            "analysis": "eval_attn_pool_on_full_split",
            "label": label,
            "manifest_eval_reproduction": result["manifest_eval_reproduction"],
            "n_excluded_manifest_seen": result["n_excluded_manifest_seen"],
            "eval_skips": result["eval_skips"],
            "split_info": result["split_info"],
            "panel_mode": result["panel_mode"],
            "manifest_split_csv": str(args.manifest_split_csv) if args.manifest_split_csv else None,
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "full_eval_results.json"
    write_results_json(out_json, payload)
    (args.output_dir / "full_eval_summary.json").write_text(json.dumps({
        "label": label,
        "full_eval_auroc": result["metrics"]["auroc"],
        "full_eval_auprc": result["metrics"]["auprc"],
        "n_full_eval_scored": result["n_full_eval_scored"],
        "n_excluded_manifest_seen": result["n_excluded_manifest_seen"],
        "manifest_eval_reproduction": result["manifest_eval_reproduction"],
    }, indent=2))
    logger.info("Wrote results → %s", out_json)


if __name__ == "__main__":
    main()
