"""[ARCHIVED] rpoB rifampicin localization-ladder driver — the concluded TB SNP diagnostic.

This is the *driver* half of the former ``pangena_predict/snp_vs_esm_prediction.py``: the
rpoB/rifampicin-specific ladder (Steps 1/2/3a/2b/2c) that localised where the single-residue
signal is lost. Its conclusion — the signal survives in Bacformer's contextualised rpoB token
(0.953) and is destroyed by the protein→genome mean-pool (0.788) — is written up in
``docs/findings/ft_deficits.md`` §1–3 and ``docs/_archive/PROGRESS_REPORT.md``.

**Frozen snapshot.** The generic, still-live probe primitives it uses (``resolve_clean_splits``,
``load_pooled_gene_vectors``, ``load_bacformer_vectors``, ``fit_score_step``) stayed behind in the
engine; the rpoB genotyper it depends on is archived beside this file. The ``from pangena_predict.…``
imports below reflect the pre-consolidation layout and are **not maintained** — this module is a
record of what was run, not runnable code after the engine move.
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

from pangena_predict.rpob_genotype import (
    RIFAMPIN_COLUMN,
    RRDR_PANEL,
    build_genotype_table,
    load_reference,
    ref_index_for_codon,
    sample_codon_positions,
)
from pangena_predict.snp_vs_esm_prediction import (
    fit_score_step,
    load_bacformer_vectors,
    load_pooled_gene_vectors,
    resolve_clean_splits,
)
from tl.train.metrics import compute_full_metrics

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def masked_marginal_features(
    genotype: pd.DataFrame,
    reference: str,
    *,
    device: str,
    codons: list[int],
) -> pd.DataFrame:
    """Per-codon masked-marginal LLR features (Step 3a), indexed by Sample.

    For each sample, mask each panel codon in its own rpoB sequence and score
    ``log P(observed) − log P(wild-type)``. Wild-type sites score ~0; resistant
    substitutions score strongly negative if ESM-C "knows" the site.
    """
    # Imported lazily so the CPU steps run without loading the ESM model.
    from tl.embed.esm_residue_level import load_esmc_mlm, masked_marginals, substitution_llr

    model, tokenizer = load_esmc_mlm(device=device)
    wt_by_codon = {codon: reference[ref_index_for_codon(reference, codon)] for codon in codons}

    features: list[list[float]] = []
    kept: list[str] = []
    n_skipped = 0
    for sample_id, row in genotype.iterrows():
        seq = row["rpob_sequence"]
        positions = sample_codon_positions(seq, reference, codons)
        valid = {c: p for c, p in positions.items() if p is not None}
        if len(valid) != len(codons):
            n_skipped += 1
            continue
        log_probs = masked_marginals(
            model,
            tokenizer,
            seq,
            positions=list(valid.values()),
            device=device,
            expected_residues={p: seq[p] for p in valid.values()},
        )
        llrs = [
            substitution_llr(log_probs[valid[codon]], tokenizer, wt=wt_by_codon[codon], observed=seq[valid[codon]])
            for codon in codons
        ]
        features.append(llrs)
        kept.append(str(sample_id))
    if n_skipped:
        logger.warning("masked-marginal: skipped %d samples with a gapped panel codon", n_skipped)
    cols = [f"llr_codon_{c}" for c in codons]
    return pd.DataFrame(features, index=pd.Index(kept, name="Sample"), columns=cols)


STEP_META = {
    "onehot_rrdr": {"step": "1", "kind": "categorical", "standardise": False, "compute": "cpu",
                    "description": "one-hot RRDR codon genotype (the SNP ceiling)"},
    "pooled_esmc_rpob": {"step": "2", "kind": "numeric", "standardise": True, "compute": "cpu",
                         "description": "frozen ESM-C mean-pooled rpoB 960-vector"},
    "masked_marginal_llr": {"step": "3a", "kind": "numeric", "standardise": True, "compute": "gpu",
                            "description": "ESM-C masked-LM LLR at panel codons"},
    "bacformer_rpob_token": {"step": "2b", "kind": "numeric", "standardise": True, "compute": "gpu",
                             "description": "frozen Bacformer contextualised rpoB token 960-vector"},
    "bacformer_mean": {"step": "2c", "kind": "numeric", "standardise": True, "compute": "gpu",
                       "description": "frozen Bacformer genome mean-pooled 960-vector (vs the fine-tuned model)"},
}


def _public_meta(meta: dict) -> dict:
    """The JSON-facing slice of a step's static metadata (drops the internal ``kind``)."""
    return {k: meta[k] for k in ("step", "description", "standardise", "compute")}


def _build_headline(steps: dict) -> dict:
    """Compute AUROC per step on the **intersection** of every scored step's evaluate set."""
    scored = {k: v for k, v in steps.items() if "eval_probs" in v}
    if not scored:
        return {"common_evaluate_n": 0, "auroc_on_common_evaluate": {}}
    common_ids = sorted(set.intersection(*(set(v["eval_probs"]) for v in scored.values())))
    label_by_id: dict[str, int] = {}
    for v in scored.values():
        label_by_id.update(v["eval_labels"])

    headline: dict = {"common_evaluate_n": len(common_ids), "auroc_on_common_evaluate": {}}
    if not common_ids:
        return headline
    y_common = np.array([label_by_id[s] for s in common_ids], dtype=int)
    for key, v in scored.items():
        probs = np.array([v["eval_probs"][s] for s in common_ids], dtype=float)
        headline["auroc_on_common_evaluate"][key] = compute_full_metrics(y_common, probs)["auroc"]
    if "onehot_rrdr" in scored and "pooled_esmc_rpob" in scored:
        headline["auroc_onehot_minus_pooled"] = float(
            headline["auroc_on_common_evaluate"]["onehot_rrdr"]
            - headline["auroc_on_common_evaluate"]["pooled_esmc_rpob"]
        )
    return headline


def _load_reference_block(reference_results_json: Path, split_info: dict, drug: str) -> dict:
    """Read the deployed Bacformer evaluate AUROC and assert the split matches ours."""
    ref = json.loads(Path(reference_results_json).read_text())
    ref_split = ref.get("split", {})
    ref_source = ref_split.get("source")
    ref_n_eval = ref_split.get("n_evaluate")
    if ref_source != split_info["source"]:
        raise ValueError(
            f"Reference split source {ref_source!r} != probe split source {split_info['source']!r} — "
            "the probe is not on the deployed model's holdout."
        )
    if ref_n_eval is not None and ref_n_eval != split_info["n_evaluate_raw"]:
        raise ValueError(
            f"Reference n_evaluate ({ref_n_eval}) != probe raw evaluate count "
            f"({split_info['n_evaluate_raw']}) — different holdout or drug."
        )
    return {
        "source_results_json": str(reference_results_json),
        "drug": ref.get("drug", drug),
        "auroc": ref.get("metrics", {}).get("auroc"),
        "auprc": ref.get("metrics", {}).get("auprc"),
        "n_evaluate": ref_n_eval,
        "split_source": ref_source,
    }


def run_probes(
    ast_sheet_path: Path,
    parquet_dir: Path,
    esm_store_dir: Path,
    *,
    drug: str,
    steps: list[str],
    device: str,
    masked_marginal_codons: list[int],
    bacformer_vectors: Path | None,
    qc_log_path: Path,
    pool_workers: int,
    reference_results_json: Path | None,
    max_samples: int | None,
) -> dict:
    """Run the requested probe steps and assemble the results payload."""
    reference = load_reference()
    label_map, train_ids, validate_ids, evaluate_ids, split_info = resolve_clean_splits(ast_sheet_path, drug)

    if max_samples is not None:
        # Proportional slice so every split stays represented (a train-first
        # truncation would empty evaluate/validate and the probe would error).
        total = len(train_ids) + len(validate_ids) + len(evaluate_ids)
        frac = min(1.0, max_samples / max(1, total))
        train_ids = train_ids[: max(1, round(len(train_ids) * frac))]
        validate_ids = validate_ids[: max(1, round(len(validate_ids) * frac))]
        evaluate_ids = evaluate_ids[: max(1, round(len(evaluate_ids) * frac))]
        keep = set(train_ids) | set(validate_ids) | set(evaluate_ids)
        label_map = {s: v for s, v in label_map.items() if s in keep}
    all_ids = [*train_ids, *validate_ids, *evaluate_ids]

    logger.info("Genotyping %d labelled samples (single-copy rpoB only)", len(all_ids))
    genotype = build_genotype_table(all_ids, parquet_dir, reference, qc_log_path=qc_log_path)
    logger.info("Genotyped %d single-copy genomes", len(genotype))

    codon_cols = [c for c in genotype.columns if c.startswith("codon_")]
    payload: dict = {
        "schema_version": "2.0",
        "task": "pangena_predict",
        "analysis": "snp_vs_esm_prediction",
        "label_column": drug,
        "sheet_path": str(ast_sheet_path),
        "parquet_dir": str(parquet_dir),
        "esm_store_dir": str(esm_store_dir),
        "split": split_info,
        "panel": [{"codon": c, "wt": wt, "alt": alt} for c, wt, alt in RRDR_PANEL],
        "rpob_copy_qc": {"n_single_copy_genotyped": int(len(genotype)), "qc_log": str(qc_log_path)},
        "steps": {},
    }

    # Build each requested step's feature frame.
    feature_frames: dict[str, tuple[pd.DataFrame, dict]] = {}
    for key in steps:
        meta = STEP_META[key]
        if key == "onehot_rrdr":
            feat_df = genotype[codon_cols].astype(str)
        elif key == "pooled_esmc_rpob":
            feat_df = load_pooled_gene_vectors(
                genotype, esm_store_dir, flat_index_col="rpob_flat_index", pool_workers=pool_workers
            )
        elif key == "masked_marginal_llr":
            feat_df = masked_marginal_features(genotype, reference, device=device, codons=masked_marginal_codons)
        elif key == "bacformer_rpob_token":
            if bacformer_vectors is None:
                logger.warning("Step 2b (bacformer_rpob_token) requested but --bacformer-vectors not given; skipping")
                continue
            feat_df = load_bacformer_vectors(bacformer_vectors, key="gene_token_vectors")
        elif key == "bacformer_mean":
            if bacformer_vectors is None:
                logger.warning("Step 2c (bacformer_mean) requested but --bacformer-vectors not given; skipping")
                continue
            feat_df = load_bacformer_vectors(bacformer_vectors, key="mean_vectors")
        else:  # pragma: no cover - guarded by argparse choices
            raise ValueError(f"Unknown step {key!r}")
        if feat_df.empty:
            payload["steps"][key] = {**_public_meta(meta), "error": "no features produced"}
            logger.warning("Step %s (%s): no features produced", meta["step"], key)
            continue
        feature_frames[key] = (feat_df, meta)

    # Fit + score each step.
    for key, (feat_df, meta) in feature_frames.items():
        result = fit_score_step(
            feat_df,
            kind=meta["kind"],
            standardise=meta["standardise"],
            label_map=label_map,
            train_ids=train_ids,
            validate_ids=validate_ids,
            evaluate_ids=evaluate_ids,
        )
        payload["steps"][key] = {**_public_meta(meta), **result}
        if "metrics" in result:
            logger.info("Step %s (%s): AUROC=%.4f (n_eval=%d)",
                        meta["step"], key, result["metrics"]["auroc"], result["n_evaluate"])

    payload["headline"] = _build_headline(payload["steps"])
    if reference_results_json is not None:
        ref_block = _load_reference_block(reference_results_json, split_info, drug)
        payload["headline"]["reference_bacformer"] = ref_block
        if ref_block["auroc"] is not None:
            logger.info("Reference Bacformer evaluate AUROC: %.4f", ref_block["auroc"])

    return payload


def _write_probs_sidecar(output_json: Path, payload: dict) -> None:
    """Write ``<output>_eval_probs.npz`` with common-eval y_true + per-step probs (for plotting)."""
    scored = {k: v for k, v in payload.get("steps", {}).items() if "eval_probs" in v}
    if not scored or payload.get("headline", {}).get("common_evaluate_n", 0) == 0:
        return
    common = sorted(set.intersection(*(set(v["eval_probs"]) for v in scored.values())))
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


def _parse_codons(spec: str) -> list[int]:
    if spec == "panel":
        return [codon for codon, _wt, _alt in RRDR_PANEL]
    if spec == "all":
        from pangena_predict.rpob_genotype import RRDR_FIRST_CODON, RRDR_LAST_CODON

        return list(range(RRDR_FIRST_CODON, RRDR_LAST_CODON + 1))
    return [int(c) for c in spec.split(",")]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ast-sheet-path", type=Path, required=True,
                        help="binary_ast_with_split.csv (Sample/phenotype-BioSample_ID, drug, train_val_eval).")
    parser.add_argument("--parquet-dir", type=Path, required=True, help="Dir of *_protein_sequences.parquet.")
    parser.add_argument("--esm-store-dir", type=Path, required=True, help="Dir of *_esm_embeddings.pt.")
    parser.add_argument("--output-json", type=Path, required=True, help="Where to write the results JSON.")
    parser.add_argument("--drug", type=str, default=RIFAMPIN_COLUMN, help="Phenotype column (default rifampin).")
    parser.add_argument(
        "--steps", nargs="+", default=["onehot_rrdr", "pooled_esmc_rpob"], choices=list(STEP_META),
        help="Which probe steps to run (default: the two CPU steps 1 + 2).",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for masked-marginal (default cpu).")
    parser.add_argument("--masked-marginal-codons", type=str, default="panel",
                        help="'panel' (4 canonical codons), 'all' (RRDR window), or a comma list of codon numbers.")
    parser.add_argument("--bacformer-vectors", type=Path, default=None,
                        help="NPZ of Bacformer gene-token vectors (Step 2b; from bacformer_genome_vectors.py).")
    parser.add_argument("--qc-log", type=Path, default=Path("rpob_copy_qc.log"),
                        help="Where to write the rpoB-copy QC log (default: ./rpob_copy_qc.log).")
    parser.add_argument("--pool-workers", type=int, default=1,
                        help="Parallel workers for the pooled-vector reads (default 1 = sequential).")
    parser.add_argument("--reference-results-json", type=Path, default=None,
                        help="Deployed Bacformer eval_results.json — reference AUROC + split assertion.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the number of samples (quick login-node smoke; default: all).")
    args = parser.parse_args()

    payload = run_probes(
        args.ast_sheet_path,
        args.parquet_dir,
        args.esm_store_dir,
        drug=args.drug,
        steps=list(args.steps),
        device=args.device,
        masked_marginal_codons=_parse_codons(args.masked_marginal_codons),
        bacformer_vectors=args.bacformer_vectors,
        qc_log_path=args.qc_log,
        pool_workers=args.pool_workers,
        reference_results_json=args.reference_results_json,
        max_samples=args.max_samples,
    )

    # Sidecar npz before stripping the per-sample probs from the JSON payload.
    _write_probs_sidecar(args.output_json, payload)
    _strip_probs_for_json(payload)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    payload["host"] = socket.gethostname()

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", args.output_json)


if __name__ == "__main__":
    main()
