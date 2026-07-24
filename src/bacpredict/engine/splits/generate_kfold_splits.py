"""Generate the per-drug split TABLE — the single materialized source of sampling + labels.

``build_split_table`` reproduces the deployed model's canonical partition (fold 0, seed 1, evaluate_seed 1,
20% holdout) and writes ``<drug>_split.csv`` (``Sample, ast_label, split``). Everything downstream then reads
that one table via :func:`bacpredict.engine.splits.load_splits.load_splits` — no module re-derives a k-fold or
reads the retired CSV ``train_val_eval`` column. ``verify_table_matches_deployed`` is the one-time migration
check that a freshly generated table matches an already-deployed model's ``results.json`` holdout, so no model
is silently re-leaked.

The low-level ``generate_kfold_splits`` (the deterministic fold generator) also lives here — reused to build
the table AND, in its multi-fold form, for the orthogonal publication k-fold sweep. ``add_splits`` (the legacy
70/10/20 single-split writer) is retained only for callers not yet migrated (see the transition shim at
``finetune/split_utils.py``).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CANONICAL = {"n_folds": 5, "fold": 0, "seed": 1, "evaluate_seed": 1, "evaluate_fraction": 0.20}


def add_splits(df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """Add a ``train_val_eval`` column with a 70/10/20 split over unique Sample IDs (legacy single-split).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``Sample`` column of unique sample identifiers.
    seed : int
        Random seed controlling the shuffle.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with a ``train_val_eval`` column (``"train"`` / ``"validate"`` / ``"evaluate"``).
    """
    rng = np.random.default_rng(seed)
    sample_ids = df["Sample"].unique()
    rng.shuffle(sample_ids)

    n_total = len(sample_ids)
    n_train = int(0.7 * n_total)
    n_val = int(0.1 * n_total)
    train_ids = set(sample_ids[:n_train])
    val_ids = set(sample_ids[n_train : n_train + n_val])

    def _assign(sample_id: str) -> str:
        if sample_id in train_ids:
            return "train"
        if sample_id in val_ids:
            return "validate"
        return "evaluate"

    out = df.copy()
    out["train_val_eval"] = out["Sample"].map(_assign)
    return out


def generate_kfold_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 1,
    evaluate_fraction: float = 0.20,
    evaluate_seed: int = 1,
) -> tuple[set[str], list[tuple[set[str], set[str]]]]:
    """Return a fixed evaluate holdout and k-fold train/validate splits over unique ``Sample`` IDs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``Sample`` column.
    n_folds : int
        Number of cross-validation folds applied to the non-evaluate samples.
    seed : int
        Controls the shuffle before splitting into folds. Change this to generate different fold
        assignments without touching the evaluate set.
    evaluate_fraction : float
        Fraction of unique samples reserved as the fixed holdout.
    evaluate_seed : int
        Controls the shuffle used to select the evaluate set. Changing ``seed`` alone does NOT affect it.

    Returns
    -------
    evaluate_ids : set[str]
        Fixed holdout sample IDs (identical for any ``seed`` when ``evaluate_seed`` /
        ``evaluate_fraction`` are unchanged).
    folds : list[tuple[set[str], set[str]]]
        Length-``n_folds`` list of ``(train_ids, validate_ids)`` pairs. Fold *i* uses fold *i* as
        validation and the remaining folds as training.
    """
    sample_ids = np.array(df["Sample"].unique())

    # Fixed evaluate set — determined only by evaluate_seed.
    eval_rng = np.random.default_rng(evaluate_seed)
    eval_order = sample_ids.copy()
    eval_rng.shuffle(eval_order)
    n_evaluate = max(1, int(evaluate_fraction * len(eval_order)))
    evaluate_ids: set[str] = set(eval_order[-n_evaluate:])
    remaining = eval_order[:-n_evaluate].copy()

    # K-fold on the remainder — determined by seed.
    fold_rng = np.random.default_rng(seed)
    fold_rng.shuffle(remaining)
    fold_arrays = np.array_split(remaining, n_folds)

    folds: list[tuple[set[str], set[str]]] = []
    for i in range(n_folds):
        val_ids: set[str] = set(fold_arrays[i])
        train_ids: set[str] = set(np.concatenate([fold_arrays[j] for j in range(n_folds) if j != i]))
        folds.append((train_ids, val_ids))

    return evaluate_ids, folds


def _resolve_sample_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a string ``Sample`` column (accepting the ``phenotype-BioSample_ID`` alias)."""
    if "Sample" not in df.columns:
        if "phenotype-BioSample_ID" not in df.columns:
            raise ValueError("AST sheet must contain 'Sample' or 'phenotype-BioSample_ID'.")
        df["Sample"] = df["phenotype-BioSample_ID"].astype(str)
    df["Sample"] = df["Sample"].astype(str)
    return df


def build_split_table(
    ast_sheet_path: str | Path,
    drug: str,
    *,
    n_folds: int = 5,
    fold: int = 0,
    seed: int = 1,
    evaluate_seed: int = 1,
    evaluate_fraction: float = 0.20,
) -> pd.DataFrame:
    """Materialize the deployment split for one drug → ``DataFrame(Sample, ast_label, split)``.

    The partition is the deployed model's own: the fixed ``evaluate`` holdout (``evaluate_seed``) becomes
    ``split == "holdout"``; of the remainder, fold ``fold`` is ``"validate"`` and the rest is ``"train"``.
    Rows are every ``drug``-labelled genome (``notna``), deduplicated by ``Sample``; ``ast_label`` is the raw
    label (:func:`bacpredict.engine.splits.load_splits.load_splits` keeps only clean 0/1 rows at read time, so
    an ambiguous label still occupies its deployed split slot but is never scored).

    Defaults are the canonical deployment params — do not change them without regenerating + re-verifying
    against every deployed model (:func:`verify_table_matches_deployed`).
    """
    df = _resolve_sample_column(pd.read_csv(ast_sheet_path, low_memory=False))
    if drug not in df.columns:
        raise ValueError(f"Drug column {drug!r} not in AST sheet; has {list(df.columns)[:20]}")

    labeled = df[df[drug].notna()].drop_duplicates(subset="Sample", keep="first").copy()
    if labeled.empty:
        raise ValueError(f"No genomes labelled for drug {drug!r}.")
    evaluate_set, folds = generate_kfold_splits(
        labeled, n_folds=n_folds, seed=seed, evaluate_fraction=evaluate_fraction, evaluate_seed=evaluate_seed
    )
    _train_set, val_set = folds[fold]

    def _split(sid: str) -> str:
        if sid in evaluate_set:
            return "holdout"
        if sid in val_set:
            return "validate"
        return "train"

    out = pd.DataFrame({"Sample": labeled["Sample"].to_numpy()})
    out["ast_label"] = labeled[drug].to_numpy()
    out["split"] = out["Sample"].map(_split)
    counts = out["split"].value_counts().to_dict()
    logger.info("%s split table: train=%d validate=%d holdout=%d", drug,
                counts.get("train", 0), counts.get("validate", 0), counts.get("holdout", 0))
    return out


def write_split_table(ast_sheet_path: str | Path, drug: str, out_path: str | Path, **params) -> pd.DataFrame:
    """Build the drug's split table and write it to ``out_path`` (``<drug>_split.csv``); returns the table."""
    table = build_split_table(ast_sheet_path, drug, **params)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    logger.info("wrote %d-row split table -> %s", len(table), out_path)
    return table


def verify_table_matches_deployed(table: pd.DataFrame, run_dir: str | Path) -> dict:
    """One-time migration check: a freshly built table's holdout matches a deployed model's ``results.json``.

    Reads ``<run_dir>/results.json`` (or ``eval_results.json``) ``split`` block and compares its recorded
    ``n_evaluate`` (over the ``notna`` set — the same universe ``build_split_table`` partitions) to the table's
    ``holdout`` row count, and — when the run recorded ``holdout_ids`` — the exact id set. Returns a report
    dict with ``ok`` and the mismatch detail. A mismatch means the AST sheet changed since training: the model
    would be scored on genomes it trained on, so it must NOT be used until regenerated + retrained.
    """
    run_dir = Path(run_dir)
    payload = None
    for base in (run_dir, run_dir.parent):
        for name in ("results.json", "eval_results.json"):
            if (base / name).exists():
                payload = json.loads((base / name).read_text())
                break
        if payload is not None:
            break
    if payload is None:
        raise FileNotFoundError(f"No results.json / eval_results.json under {run_dir} (or its parent).")
    split = payload.get("split") or {}
    holdout = set(table.loc[table["split"] == "holdout", "Sample"].astype(str))
    report: dict = {"ok": True, "n_holdout_table": len(holdout), "n_evaluate_deployed": split.get("n_evaluate")}
    if split.get("n_evaluate") is not None and int(split["n_evaluate"]) != len(holdout):
        report.update(ok=False, reason="n_evaluate mismatch")
    recorded = split.get("holdout_ids") or split.get("evaluate_ids")
    if recorded is not None and set(map(str, recorded)) != holdout:
        report.update(ok=False, reason="holdout id-set mismatch")
    return report


def main() -> None:
    """CLI: write ``<drug>_split.csv`` (and optionally verify it against a deployed run)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ast-sheet", type=Path, required=True, help="AST sheet with Sample + the drug label column")
    ap.add_argument("--drug", required=True, help="drug label column")
    ap.add_argument("--out", type=Path, required=True, help="output <drug>_split.csv")
    ap.add_argument("--verify-run-dir", type=Path, default=None,
                    help="a deployed run dir; assert the table's holdout == its results.json holdout")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    table = write_split_table(args.ast_sheet, args.drug, args.out)
    if args.verify_run_dir is not None:
        report = verify_table_matches_deployed(table, args.verify_run_dir)
        logger.info("verify vs %s: %s", args.verify_run_dir, report)
        if not report["ok"]:
            raise SystemExit(f"split table does NOT match deployed model: {report}")


if __name__ == "__main__":
    main()
