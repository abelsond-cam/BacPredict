"""Concat probe: ESM-C gene-protein vector ⊕ Bacformer genome-mean → logistic regression.

The head-pool diagnostics showed the prediction head's learned pool never routes to the causal gene
(*rpoB* for rifampicin) — it either hyper-concentrates on lineage markers or collapses to a uniform
mean. The gene signal is *present* upstream (frozen ESM-C mean-pooled rpoB ~0.971; frozen Bacformer
genome-mean ~0.788). This probe skips the head entirely and asks the simplest deployable question:
**concatenate one gene's ESM-C 960-vector to the Bacformer genome-mean 960-vector (→ 1,920-d) and fit
a plain logistic regression.** If concat ≈ the ESM-gene ceiling, a "causal gene ⊕ genome mean" feature
vector is a viable read-out that needs no attention head.

Two axes, both parameters (default = the rifampicin/rpoB setup that hit 0.975):

- **gene** (``--gene``, default ``rpoB``) — any single-copy gene, located generically via
  :func:`bacpredict.engine.gene_lr.locate_gene.build_gene_presence_table` (no rpoB-specific genotyping).
- **Bacformer mean variant** — **frozen** base model (default) or **fine-tuned** backbone of a deployed
  AMR checkpoint (``--bacformer-checkpoint``; the ~0.905 mean-pool model — A.1.i).

Three steps, all scored on the **same canonical evaluate fold** (``binary_ast_with_split.csv`` via
:func:`bacpredict.engine.finetune.holdout.resolve_clean_splits`) over the **same sample
intersection**, so the numbers are directly comparable:

============================  ===============================================  ======
key                           features                                         ~AUROC
============================  ===============================================  ======
``esm_gene_only``             ESM-C mean-pooled gene 960-vector                ~0.971 (rpoB)
``bacformer_mean_only``       Bacformer genome-mean 960-vector                 ~0.788 frozen / ~0.905 FT
``concat_esm_gene_plus_mean`` the two concatenated (1,920-d)                   the test
============================  ===============================================  ======

For rpoB the two ablations are a **harness sanity check** against the localization ladder (esm ~0.971,
mean ~0.788 frozen / ~0.905 FT); for other genes there is no ladder target, so the absolute sanity is
skipped. Pass a pre-computed ``--bacformer-vectors`` NPZ (``mean_vectors``) to run the whole probe on
CPU (and the ``--kfold`` significance pass).
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

from bacpredict.engine.concat.bacformer_genome_vectors import RIFAMPIN_COLUMN, compute_bacformer_vectors
from bacpredict.engine.finetune.holdout import resolve_clean_splits
from bacpredict.engine.gene_lr.kfold_probe import FeatureSpec, run_kfold_probe, summarise_kfold
from bacpredict.engine.gene_lr.linear_probe import fit_score_step
from bacpredict.engine.gene_lr.locate_gene import build_gene_presence_table
from bacpredict.engine.gene_lr.pooled_cds_vectors import load_pooled_gene_vectors

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Bacformer NPZ reader — the ONLY live user of the token-vector loader, kept local so it retires with this
# (concluded rpoB/rifampicin diagnostic) module. Token-vector NPZ keys, newest first — a gene-token request
# resolves to whichever the NPZ carries.
_TOKEN_KEY_ALIASES = ("gene_token_vectors", "rpob_vectors", "vectors")


def load_bacformer_vectors(path: str | Path, key: str = "gene_token_vectors") -> pd.DataFrame:
    """Load Bacformer vectors written by the GPU pass.

    Expects an ``.npz`` with ``sample_ids`` (str array) plus ``gene_token_vectors`` and ``mean_vectors``
    ([N, 960] each), as produced by :mod:`bacpredict.engine.concat.bacformer_genome_vectors`. ``key``
    selects which (``"gene_token_vectors"`` for the contextualised gene token, ``"mean_vectors"`` for the
    genome mean). A token request back-compat-resolves to whichever token alias the NPZ carries (legacy
    ``"rpob_vectors"`` / ``"vectors"``).
    """
    data = np.load(path, allow_pickle=False)
    ids = [str(s) for s in data["sample_ids"]]
    if key not in data.files:
        for alt in _TOKEN_KEY_ALIASES:
            if alt in data.files:
                key = alt
                break
    if key not in data.files:
        raise KeyError(f"{key!r} not in {path} (has {list(data.files)})")
    return pd.DataFrame(data[key], index=pd.Index(ids, name="Sample"))

# rpoB-only ladder targets the ablations must reproduce before the concat is believed (full run only).
# The esm-gene target (0.971) and the mean target (0.788 frozen / 0.905 FT) are rifampicin/rpoB-specific.
ESM_RPOB_SANITY_TARGET = 0.971


def _top_gene_from_ranking(ranking_csv: Path) -> str:
    """The highest out-of-fold-AUROC gene from a per-gene LR ranking CSV (``build_per_gene_lr_store``).

    The ranking table has columns ``gene_name`` and ``lr_auroc_<drug>`` (one auroc column). Returns the
    ``gene_name`` of the top-AUROC row — the auto-discovered causal-gene candidate for that drug, fed to
    the concat probe as ``--gene`` (rpoB tops the rifampin ranking; katG/embB/pncA/gyrA/rpsL their drugs).
    """
    df = pd.read_csv(ranking_csv)
    auroc_cols = [c for c in df.columns if c.startswith("lr_auroc_")]
    if not auroc_cols:
        raise ValueError(f"{ranking_csv} has no lr_auroc_<drug> column — not a per-gene ranking table.")
    top = df.sort_values(auroc_cols[0], ascending=False).iloc[0]
    logger.info("Top-ranked gene from %s: %s (%s=%.4f)", ranking_csv.name, top["gene_name"], auroc_cols[0], top[auroc_cols[0]])
    return str(top["gene_name"])


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
    gene_table: pd.DataFrame,
    esm_store_dir: Path,
    *,
    device: str,
    bacformer_vectors: Path | None,
    save_bacformer_vectors: Path | None,
    mode: str,
    bacformer_checkpoint: Path | None,
) -> pd.DataFrame:
    """Bacformer genome-mean per sample — loaded from an NPZ (CPU) or computed on GPU.

    With ``--bacformer-vectors`` the whole probe is CPU-only (the NPZ may hold a frozen *or* fine-tuned
    mean — say which with ``--mean-is-finetuned``). Otherwise runs a Bacformer forward over the genomes:
    the **fine-tuned** backbone of ``bacformer_checkpoint`` when ``mode == "finetuned"`` (A.1.i), else
    the **frozen** base model. The genome mean is gene-agnostic. Optionally caches ``{sample_ids,
    gene_token_vectors, mean_vectors}`` to ``save_bacformer_vectors`` (so the k-fold reruns are CPU-only).
    """
    if bacformer_vectors is not None:
        logger.info("Loading Bacformer genome-mean vectors from %s", bacformer_vectors)
        return load_bacformer_vectors(bacformer_vectors, key="mean_vectors")
    logger.info("Computing %s Bacformer genome-mean over %d genomes on %s", mode, len(gene_table), device)
    token_mat, mean_mat, kept = compute_bacformer_vectors(
        gene_table, esm_store_dir, device=device, mode=mode, checkpoint=bacformer_checkpoint
    )
    if save_bacformer_vectors is not None:
        save_bacformer_vectors.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_bacformer_vectors, sample_ids=np.array(kept), gene_token_vectors=token_mat, mean_vectors=mean_mat)
        logger.info("Cached %d Bacformer gene-token + genome-mean vectors to %s", len(kept), save_bacformer_vectors)
    return pd.DataFrame(mean_mat, index=pd.Index(kept, name="Sample"))


def run_concat_probe(
    ast_sheet_path: Path,
    parquet_dir: Path,
    esm_store_dir: Path,
    *,
    drug: str,
    gene: str,
    gene_aliases: tuple[str, ...],
    device: str,
    bacformer_vectors: Path | None,
    save_bacformer_vectors: Path | None,
    bacformer_checkpoint: Path | None,
    mean_is_finetuned: bool,
    qc_log_path: Path,
    pool_workers: int,
    max_samples: int | None,
    kfold: dict | None = None,
    kfold_on_eval_holdout: bool = False,
) -> dict:
    """Run the three steps (ESM-gene, Bacformer-mean, concat) on the canonical eval fold.

    When ``kfold`` is given (``{n_folds, seeds, evaluate_seed, evaluate_fraction}``) the same three
    aligned frames are *additionally* routed through the k-fold × m-seed harness (mean ± sd per frame
    + paired AUROC deltas), so one run yields both the canonical-holdout headline and the significance
    of the small top-of-ladder deltas. The k-fold block uses its own fixed holdout, not the canonical one.

    ``kfold_on_eval_holdout`` restricts the k-fold universe to the **canonical ``evaluate`` ids** — the
    genomes a fine-tuned backbone was *held out from*. On those FT-unseen genomes the fine-tuned mean is
    once again a label-blind feature, so re-splitting them is an **honest** k-fold (no GPU, no re-tuning):
    the only valid way to put error bars on the A.1.i FT-mean concat without re-fine-tuning per fold. For
    the frozen mean this restriction is unnecessary (the base model never saw any label) — leave it off.
    """
    # The mean is fine-tuned (A.1.i) when computed from a checkpoint, or flagged so for a loaded FT NPZ.
    finetuned = bacformer_checkpoint is not None or mean_is_finetuned
    mode = "finetuned" if bacformer_checkpoint is not None else "frozen"

    label_map, train_ids, validate_ids, evaluate_ids, split_info = resolve_clean_splits(ast_sheet_path, drug)
    if max_samples is not None:
        train_ids, validate_ids, evaluate_ids = _slice_splits(train_ids, validate_ids, evaluate_ids, max_samples)
        keep = set(train_ids) | set(validate_ids) | set(evaluate_ids)
        label_map = {s: v for s, v in label_map.items() if s in keep}
    all_ids = [*train_ids, *validate_ids, *evaluate_ids]

    logger.info("Locating single-copy %s in %d labelled samples", gene, len(all_ids))
    gene_table = build_gene_presence_table(all_ids, parquet_dir, gene, aliases=gene_aliases, qc_log_path=qc_log_path)
    logger.info("Found %d single-copy %s genomes", len(gene_table), gene)

    esm_df = load_pooled_gene_vectors(gene_table, esm_store_dir, flat_index_col="gene_flat_index", pool_workers=pool_workers)
    if esm_df.empty:
        raise RuntimeError(f"No ESM-C {gene} vectors recovered — check esm_store_dir / .pt suffix.")
    esm_df.columns = [f"esm_gene_{i}" for i in range(esm_df.shape[1])]

    mean_df = _bacformer_mean_df(
        gene_table, esm_store_dir, device=device,
        bacformer_vectors=bacformer_vectors, save_bacformer_vectors=save_bacformer_vectors,
        mode=mode, bacformer_checkpoint=bacformer_checkpoint,
    )
    if mean_df.empty:
        raise RuntimeError("No Bacformer genome-mean vectors recovered.")
    mean_df.columns = [f"bac_mean_{i}" for i in range(mean_df.shape[1])]

    # One common sample set so the three steps' evaluate subsets are identical → comparable AUROCs.
    common = sorted(set(esm_df.index) & set(mean_df.index))
    if not common:
        raise RuntimeError("No samples shared between the ESM-gene and Bacformer-mean feature frames.")
    esm_c, mean_c = esm_df.loc[common], mean_df.loc[common]
    concat_c = pd.concat([esm_c, mean_c], axis=1)
    logger.info("Feature frames aligned on %d common samples (concat dim=%d)", len(common), concat_c.shape[1])

    features = {
        "esm_gene_only": esm_c,
        "bacformer_mean_only": mean_c,
        "concat_esm_gene_plus_mean": concat_c,
    }
    payload: dict = {
        "schema_version": "2.0",
        "task": "pangena_predict",
        "analysis": "concatenate_bacformer_genome_esm_protein_emb",
        "label_column": drug,
        "gene": gene,
        "sheet_path": str(ast_sheet_path),
        "esm_store_dir": str(esm_store_dir),
        "bacformer_vectors": str(bacformer_vectors) if bacformer_vectors else None,
        "bacformer_checkpoint": str(bacformer_checkpoint) if bacformer_checkpoint else None,
        "mean_variant": "finetuned" if finetuned else "frozen",
        "split": split_info,
        "n_common_samples": len(common),
        "gene_presence_qc": {"n_single_copy": int(len(gene_table)), "qc_log": str(qc_log_path)},
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

    mean_target = 0.905 if finetuned else 0.788
    payload["headline"] = _build_headline(
        payload["steps"], smoke=max_samples is not None, mean_target=mean_target, gene=gene
    )

    if kfold is not None:
        specs = {key: FeatureSpec(feat_df, kind="numeric", standardise=True) for key, feat_df in features.items()}
        # Restrict the fold universe to the canonical evaluate holdout (FT-unseen) for an honest FT k-fold;
        # otherwise the harness builds its own holdout over the full common universe.
        kfold_universe: list[str] | None = None
        if kfold_on_eval_holdout:
            kfold_universe = sorted(set(evaluate_ids) & set(common) & set(label_map))
            logger.info(
                "k-fold restricted to the canonical evaluate holdout: %d FT-unseen genomes "
                "(the fine-tuned mean is label-blind here → honest error bars).", len(kfold_universe),
            )
        pool = kfold_universe if kfold_universe is not None else common
        if len(pool) < kfold["n_folds"] + 1:
            logger.warning("Skipping k-fold: %d samples < n_folds+1 (%d)", len(pool), kfold["n_folds"] + 1)
        else:
            # Leaky only when k-folding a FINE-TUNED mean over the WHOLE cohort: re-splitting puts
            # FT-training genomes into the new evaluate fold (representation leakage → optimistic).
            # Restricting to the canonical evaluate holdout removes that — those genomes were FT-unseen.
            leaky = finetuned and not kfold_on_eval_holdout
            if leaky:
                logger.warning(
                    "k-fold on a FINE-TUNED mean over the whole cohort is LEAKY: the backbone saw most "
                    "genomes' labels during fine-tuning. Treat these numbers as optimistic, not a valid "
                    "held-out estimate — use --kfold-on-eval-holdout for honest FT error bars."
                )
            logger.info("Running k-fold × m-seed harness over the three aligned frames")
            payload["kfold"] = run_kfold_probe(
                specs, label_map, universe_ids=kfold_universe,
                n_folds=kfold["n_folds"], seeds=kfold["seeds"],
                evaluate_seed=kfold["evaluate_seed"], evaluate_fraction=kfold["evaluate_fraction"],
            )
            payload["kfold"]["finetuned_mean_leakage_warning"] = leaky
            payload["kfold"]["restricted_to_eval_holdout"] = bool(kfold_on_eval_holdout)
            logger.info("\n%s", summarise_kfold(payload["kfold"]))

    return payload


def _build_headline(steps: dict, *, smoke: bool, mean_target: float, gene: str) -> dict:
    """Concat AUROC, the two ablation AUROCs, the lift over mean-only, and (for rpoB) the sanity verdict.

    ``mean_target`` is the expected ``bacformer_mean_only`` AUROC for the run's variant (~0.788 frozen,
    ~0.905 fine-tuned). The absolute ablation sanity is rifampicin/rpoB-specific, so it is only emitted
    when ``gene == "rpoB"`` (and never on a smoke — n≈10 AUROC is noise).
    """
    auroc = {k: v["metrics"]["auroc"] for k, v in steps.items() if "metrics" in v}
    headline: dict = {"auroc": auroc}
    if "concat_esm_gene_plus_mean" in auroc and "bacformer_mean_only" in auroc:
        headline["concat_minus_mean"] = float(auroc["concat_esm_gene_plus_mean"] - auroc["bacformer_mean_only"])
    if "concat_esm_gene_plus_mean" in auroc and "esm_gene_only" in auroc:
        headline["concat_minus_esm_gene"] = float(auroc["concat_esm_gene_plus_mean"] - auroc["esm_gene_only"])
    if not smoke and gene.lower() == "rpob":
        targets = {"esm_gene_only": ESM_RPOB_SANITY_TARGET, "bacformer_mean_only": mean_target}
        sanity = {
            k: {"observed": auroc[k], "target": t, "abs_diff": abs(auroc[k] - t), "ok": abs(auroc[k] - t) <= 0.02}
            for k, t in targets.items() if k in auroc
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
    parser.add_argument("--gene", type=str, default="rpoB",
                        help="Gene whose ESM-C vector to concat with the Bacformer mean (default rpoB).")
    parser.add_argument("--gene-from-ranking", type=Path, default=None,
                        help="Read --gene from this per-gene LR ranking CSV (top out-of-fold-AUROC gene). "
                             "Auto-discovers the causal gene per drug; overrides --gene.")
    parser.add_argument("--gene-aliases", type=str, nargs="*", default=[], help="Alternative accepted gene symbols.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for the Bacformer mean (default cuda:0).")
    parser.add_argument("--bacformer-vectors", type=Path, default=None,
                        help="Pre-computed NPZ (mean_vectors) — supply to run the whole probe on CPU.")
    parser.add_argument("--save-bacformer-vectors", type=Path, default=None,
                        help="If computing on GPU, also cache the {gene_token,mean}_vectors NPZ here for reuse.")
    parser.add_argument("--bacformer-checkpoint", type=Path, default=None,
                        help="A.1.i: compute the FT genome-mean from this AMR checkpoint (the 0.905 model) "
                             "instead of the frozen base. GPU; mutually exclusive with --bacformer-vectors.")
    parser.add_argument("--mean-is-finetuned", action="store_true",
                        help="Mark a loaded --bacformer-vectors NPZ as fine-tuned (sanity target 0.905 not 0.788).")
    parser.add_argument("--qc-log", type=Path, default=Path("gene_presence_qc.log"),
                        help="Where to write the gene-presence QC log (default: ./gene_presence_qc.log).")
    parser.add_argument("--pool-workers", type=int, default=1,
                        help="Parallel workers for the pooled ESM-C gene reads (default 1 = sequential).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the samples (quick smoke; default: all). On a smoke the ablation sanity is skipped.")
    parser.add_argument("--kfold", type=int, default=None, metavar="N",
                        help="Also run an N-fold × m-seed harness over the three frames (mean±sd + paired deltas).")
    parser.add_argument("--kfold-on-eval-holdout", action="store_true",
                        help="Restrict the k-fold universe to the canonical evaluate holdout (FT-unseen genomes) "
                             "— the honest way to error-bar a FINE-TUNED mean without re-fine-tuning per fold.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                        help="Seeds for the k-fold harness (default: 1 2 3). Only used with --kfold.")
    parser.add_argument("--evaluate-seed", type=int, default=1,
                        help="Pins the fixed k-fold evaluate holdout (default 1). Only used with --kfold.")
    parser.add_argument("--evaluate-fraction", type=float, default=0.20,
                        help="Fraction of the universe held out as the fixed k-fold evaluate set (default 0.20).")
    args = parser.parse_args()

    if args.bacformer_vectors is not None and args.bacformer_checkpoint is not None:
        parser.error("Pass either --bacformer-vectors (load a cached NPZ) or --bacformer-checkpoint (compute FT), not both.")
    if args.kfold_on_eval_holdout and args.kfold is None:
        parser.error("--kfold-on-eval-holdout requires --kfold N.")
    if args.gene_from_ranking is not None:
        args.gene = _top_gene_from_ranking(args.gene_from_ranking)

    kfold = None
    if args.kfold is not None:
        kfold = {
            "n_folds": args.kfold, "seeds": args.seeds,
            "evaluate_seed": args.evaluate_seed, "evaluate_fraction": args.evaluate_fraction,
        }

    payload = run_concat_probe(
        args.ast_sheet_path, args.parquet_dir, args.esm_store_dir,
        drug=args.drug, gene=args.gene, gene_aliases=tuple(args.gene_aliases), device=args.device,
        bacformer_vectors=args.bacformer_vectors, save_bacformer_vectors=args.save_bacformer_vectors,
        bacformer_checkpoint=args.bacformer_checkpoint, mean_is_finetuned=args.mean_is_finetuned,
        qc_log_path=args.qc_log, pool_workers=args.pool_workers, max_samples=args.max_samples,
        kfold=kfold, kfold_on_eval_holdout=args.kfold_on_eval_holdout,
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
