"""E1 — Concat probe: does *injecting* the ESM-C rpoB vector bypass the broken head?

The head-pool diagnostics showed the prediction head's learned pool never routes to rpoB — it
either hyper-concentrates on lineage markers (confound) or collapses to a uniform mean (the
deployable e2e). The rpoB signal is *present* upstream (frozen ESM-C mean-pooled rpoB ~0.971;
frozen Bacformer genome-mean ~0.788). This probe skips the head entirely and asks the simplest
deployable question: **concatenate the ESM-C rpoB 960-vector to the frozen Bacformer genome-mean
960-vector (→ 1,920-d) and fit a plain logistic regression.** If concat ≈ the ESM-rpoB ceiling, a
"top-K causal gene ⊕ genome mean" feature vector is a viable read-out that needs no attention head.

Three steps, all scored on the **same canonical evaluate fold** (``binary_ast_with_split.csv`` via
:func:`snp_embeddings.snp_vs_esm_prediction.resolve_clean_splits`) over the **same sample
intersection**, so the numbers are directly comparable:

================================  ================================================  ======
key                               features                                          ~AUROC
================================  ================================================  ======
``esm_rpob_only``                 frozen ESM-C mean-pooled rpoB 960-vector          ~0.971
``bacformer_mean_only``           frozen Bacformer genome-mean 960-vector           ~0.788
``concat_esm_rpob_plus_mean``     the two concatenated (1,920-d)                     the test
================================  ================================================  ======

The two ablations are the **harness sanity check**: ``esm_rpob_only`` must reproduce ~0.971 and
``bacformer_mean_only`` ~0.79 (the localization ladder) before the concat number is trusted.

Reuse only — no new science:
``resolve_clean_splits`` + ``load_pooled_rpob_vectors`` + ``fit_score_step`` (from
:mod:`snp_embeddings.snp_vs_esm_prediction`), ``build_genotype_table`` (rpoB flat index, single-copy
QC) and ``compute_bacformer_vectors`` (frozen genome mean). The Bacformer mean is GPU; pass a
pre-computed ``--bacformer-vectors`` NPZ (``mean_vectors``) to run the whole probe on CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from snp_embeddings.frozen_bacformer_rpob_vectors import compute_bacformer_vectors
from snp_embeddings.rpob_genotype import RIFAMPIN_COLUMN, build_genotype_table, load_reference
from snp_embeddings.snp_vs_esm_prediction import (
    fit_score_step,
    load_bacformer_vectors,
    load_pooled_rpob_vectors,
    resolve_clean_splits,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# The ladder numbers the two ablations must reproduce before the concat is believed (full run only).
SANITY_TARGETS = {"esm_rpob_only": 0.971, "bacformer_mean_only": 0.788}


def _slice_splits(
    train_ids: list[str], validate_ids: list[str], evaluate_ids: list[str], max_samples: int
) -> tuple[list[str], list[str], list[str]]:
    """Proportionally cap each split so all three stay represented (smoke). Mirrors the sibling probes."""
    total = len(train_ids) + len(validate_ids) + len(evaluate_ids)
    frac = min(1.0, max_samples / max(1, total))
    return (
        train_ids[: max(1, round(len(train_ids) * frac))],
        validate_ids[: max(1, round(len(validate_ids) * frac))],
        evaluate_ids[: max(1, round(len(evaluate_ids) * frac))],
    )


def _bacformer_mean_df(
    genotype: pd.DataFrame,
    esm_store_dir: Path,
    *,
    device: str,
    bacformer_vectors: Path | None,
    save_bacformer_vectors: Path | None,
) -> pd.DataFrame:
    """Frozen Bacformer genome-mean per sample — loaded from an NPZ (CPU) or computed on GPU.

    With ``--bacformer-vectors`` the whole probe is CPU-only. Without it, runs the frozen Bacformer
    forward over the genotyped genomes and (optionally) caches ``{sample_ids, rpob_vectors,
    mean_vectors}`` to ``save_bacformer_vectors`` for reuse by this and the sibling probes.
    """
    if bacformer_vectors is not None:
        logger.info("Loading frozen Bacformer genome-mean vectors from %s", bacformer_vectors)
        return load_bacformer_vectors(bacformer_vectors, key="mean_vectors")
    logger.info("Computing frozen Bacformer genome-mean over %d genomes on %s", len(genotype), device)
    _rpob_mat, mean_mat, kept = compute_bacformer_vectors(genotype, esm_store_dir, device=device)
    if save_bacformer_vectors is not None:
        save_bacformer_vectors.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_bacformer_vectors, sample_ids=np.array(kept), rpob_vectors=_rpob_mat, mean_vectors=mean_mat)
        logger.info("Cached %d Bacformer rpoB-token + genome-mean vectors to %s", len(kept), save_bacformer_vectors)
    return pd.DataFrame(mean_mat, index=pd.Index(kept, name="Sample"))


def run_concat_probe(
    ast_sheet_path: Path,
    parquet_dir: Path,
    esm_store_dir: Path,
    *,
    drug: str,
    device: str,
    bacformer_vectors: Path | None,
    save_bacformer_vectors: Path | None,
    qc_log_path: Path,
    pool_workers: int,
    max_samples: int | None,
) -> dict:
    """Run the three steps (ESM-rpoB, Bacformer-mean, concat) on the canonical eval fold."""
    reference = load_reference()
    label_map, train_ids, validate_ids, evaluate_ids, split_info = resolve_clean_splits(ast_sheet_path, drug)
    if max_samples is not None:
        train_ids, validate_ids, evaluate_ids = _slice_splits(train_ids, validate_ids, evaluate_ids, max_samples)
        keep = set(train_ids) | set(validate_ids) | set(evaluate_ids)
        label_map = {s: v for s, v in label_map.items() if s in keep}
    all_ids = [*train_ids, *validate_ids, *evaluate_ids]

    logger.info("Genotyping %d labelled samples (single-copy rpoB only)", len(all_ids))
    genotype = build_genotype_table(all_ids, parquet_dir, reference, qc_log_path=qc_log_path)
    logger.info("Genotyped %d single-copy genomes", len(genotype))

    esm_df = load_pooled_rpob_vectors(genotype, esm_store_dir, pool_workers=pool_workers)
    if esm_df.empty:
        raise RuntimeError("No ESM-C rpoB vectors recovered — check esm_store_dir / .pt suffix.")
    esm_df.columns = [f"esm_rpob_{i}" for i in range(esm_df.shape[1])]

    mean_df = _bacformer_mean_df(
        genotype, esm_store_dir, device=device,
        bacformer_vectors=bacformer_vectors, save_bacformer_vectors=save_bacformer_vectors,
    )
    if mean_df.empty:
        raise RuntimeError("No Bacformer genome-mean vectors recovered.")
    mean_df.columns = [f"bac_mean_{i}" for i in range(mean_df.shape[1])]

    # One common sample set so the three steps' evaluate subsets are identical → comparable AUROCs.
    common = sorted(set(esm_df.index) & set(mean_df.index))
    if not common:
        raise RuntimeError("No samples shared between the ESM-rpoB and Bacformer-mean feature frames.")
    esm_c, mean_c = esm_df.loc[common], mean_df.loc[common]
    concat_c = pd.concat([esm_c, mean_c], axis=1)
    logger.info("Feature frames aligned on %d common samples (concat dim=%d)", len(common), concat_c.shape[1])

    features = {
        "esm_rpob_only": esm_c,
        "bacformer_mean_only": mean_c,
        "concat_esm_rpob_plus_mean": concat_c,
    }
    payload: dict = {
        "schema_version": "1.0",
        "task": "snp_embeddings",
        "analysis": "eval_concat_rpob_mean",
        "label_column": drug,
        "sheet_path": str(ast_sheet_path),
        "esm_store_dir": str(esm_store_dir),
        "bacformer_vectors": str(bacformer_vectors) if bacformer_vectors else None,
        "split": split_info,
        "n_common_samples": len(common),
        "rpob_copy_qc": {"n_single_copy_genotyped": int(len(genotype)), "qc_log": str(qc_log_path)},
        "steps": {},
    }
    for key, feat_df in features.items():
        res = fit_score_step(
            feat_df, kind="numeric", standardise=True, label_map=label_map,
            train_ids=train_ids, validate_ids=validate_ids, evaluate_ids=evaluate_ids,
        )
        payload["steps"][key] = res
        if "metrics" in res:
            logger.info(
                "%s: AUROC=%.4f AUPRC=%.4f (n_eval=%d, n_feat=%d)",
                key, res["metrics"]["auroc"], res["metrics"]["auprc"], res["n_evaluate"], res["n_features"],
            )

    payload["headline"] = _build_headline(payload["steps"], smoke=max_samples is not None)
    return payload


def _build_headline(steps: dict, *, smoke: bool) -> dict:
    """Concat AUROC, the two ablation AUROCs, the lift over mean-only, and the sanity verdict."""
    auroc = {k: v["metrics"]["auroc"] for k, v in steps.items() if "metrics" in v}
    headline: dict = {"auroc": auroc}
    if "concat_esm_rpob_plus_mean" in auroc and "bacformer_mean_only" in auroc:
        headline["concat_minus_mean"] = float(auroc["concat_esm_rpob_plus_mean"] - auroc["bacformer_mean_only"])
    if "concat_esm_rpob_plus_mean" in auroc and "esm_rpob_only" in auroc:
        headline["concat_minus_esm_rpob"] = float(auroc["concat_esm_rpob_plus_mean"] - auroc["esm_rpob_only"])
    # Sanity: do the ablations reproduce the localization ladder? (Skipped on a smoke — n=10 AUROC is noise.)
    if not smoke:
        sanity = {
            k: {"observed": auroc[k], "target": t, "abs_diff": abs(auroc[k] - t), "ok": abs(auroc[k] - t) <= 0.02}
            for k, t in SANITY_TARGETS.items() if k in auroc
        }
        headline["ablation_sanity"] = sanity
        for k, s in sanity.items():
            (logger.info if s["ok"] else logger.warning)(
                "ablation sanity %s: observed %.4f vs target ~%.3f (|Δ|=%.4f) %s",
                k, s["observed"], s["target"], s["abs_diff"], "OK" if s["ok"] else "OFF — harness suspect",
            )
    return headline


def _write_probs_sidecar(output_json: Path, payload: dict) -> None:
    """Write ``<output>_eval_probs.npz`` (common-eval y_true + per-step probs) for plotting/calibration."""
    scored = {k: v for k, v in payload.get("steps", {}).items() if "eval_probs" in v}
    if not scored:
        return
    common = sorted(set.intersection(*(set(v["eval_probs"]) for v in scored.values())))
    if not common:
        return
    label_by_id: dict[str, int] = {}
    for v in scored.values():
        label_by_id.update(v["eval_labels"])
    arrays: dict[str, np.ndarray] = {
        "sample_ids": np.array(common),
        "y_true": np.array([label_by_id[s] for s in common], dtype=int),
    }
    for key, v in scored.items():
        arrays[f"prob_{key}"] = np.array([v["eval_probs"][s] for s in common], dtype=float)
    sidecar = output_json.with_name(output_json.stem + "_eval_probs.npz")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    np.savez(sidecar, **arrays)
    logger.info("Wrote %s", sidecar)


def _strip_probs_for_json(payload: dict) -> None:
    """Drop the bulky per-sample ``eval_probs``/``eval_labels`` dicts before writing JSON."""
    for step in payload.get("steps", {}).values():
        step.pop("eval_probs", None)
        step.pop("eval_labels", None)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ast-sheet-path", type=Path, required=True,
                        help="binary_ast_with_split.csv (Sample/phenotype-BioSample_ID, drug, train_val_eval).")
    parser.add_argument("--parquet-dir", type=Path, required=True, help="Dir of *_protein_sequences.parquet.")
    parser.add_argument("--esm-store-dir", type=Path, required=True, help="Dir of *_esm_embeddings.pt.")
    parser.add_argument("--output-json", type=Path, required=True, help="Where to write the results JSON.")
    parser.add_argument("--drug", type=str, default=RIFAMPIN_COLUMN, help="Phenotype column (default rifampin).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for the Bacformer mean (default cuda:0).")
    parser.add_argument("--bacformer-vectors", type=Path, default=None,
                        help="Pre-computed NPZ (mean_vectors) — supply to run the whole probe on CPU.")
    parser.add_argument("--save-bacformer-vectors", type=Path, default=None,
                        help="If computing on GPU, also cache the {rpob,mean}_vectors NPZ here for reuse.")
    parser.add_argument("--qc-log", type=Path, default=Path("rpob_copy_qc.log"),
                        help="Where to write the rpoB-copy QC log (default: ./rpob_copy_qc.log).")
    parser.add_argument("--pool-workers", type=int, default=1,
                        help="Parallel workers for the pooled ESM-C rpoB reads (default 1 = sequential).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the samples (quick smoke; default: all). On a smoke the ablation sanity is skipped.")
    args = parser.parse_args()

    payload = run_concat_probe(
        args.ast_sheet_path, args.parquet_dir, args.esm_store_dir,
        drug=args.drug, device=args.device,
        bacformer_vectors=args.bacformer_vectors, save_bacformer_vectors=args.save_bacformer_vectors,
        qc_log_path=args.qc_log, pool_workers=args.pool_workers, max_samples=args.max_samples,
    )

    _write_probs_sidecar(args.output_json, payload)
    _strip_probs_for_json(payload)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    payload["host"] = socket.gethostname()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()
