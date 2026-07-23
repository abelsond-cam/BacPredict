"""Canonical train/validate/evaluate split resolution — the ONE place a holdout is decided.

Every downstream probe (per-gene/segment LR, the concat ladder, the driver panel, the FT genome-mean
cache) must score on the **same** holdout the deployed fine-tuned model was evaluated on. Score an
FT-derived feature on genomes the FT backbone trained on and its "held-out" AUROC is a train/test leak
that inflates every absolute number — the exact bug this module exists to prevent (81% of a CSV-holdout
"evaluate" set were k-fold TRAIN/VAL genomes; azithromycin read 0.918 leaked vs 0.799 honest).

The deployed AMR models are **k-fold**-trained, so the honest holdout is the k-fold evaluate set pinned by
``evaluate_seed`` — NOT the CSV ``train_val_eval`` single-split column. The two are different sets.

Resolvers, lowest → highest level:

* :func:`resolve_holdouts` — reconstruct (evaluate, validate, labels, source) for a drug in either mode.
* :func:`resolve_evaluate_ids` — evaluate ids + labels only (fold-independent).
* :func:`load_splits` — read the CSV ``train_val_eval`` column directly (label-blind ranking builders).
* :func:`resolve_deployed_holdout` — replay a run's *own* ``results.json`` split provenance, so a caller
  cannot silently pick the wrong holdout for an FT-derived feature.
* :func:`resolve_clean_splits` — clean-0/1 wrapper over either source; ``checkpoint_dir`` routes it through
  the deployed provenance.

They all live here so there is a single import surface for "who is in which split". This module stays
light (pandas + :mod:`bacpredict.engine.finetune.split_utils` only — no torch/sklearn/transformers) so any
probe can import a resolver without dragging in the model stack.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from bacpredict.engine.finetune.split_utils import generate_kfold_splits

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level resolvers (k-fold or CSV single-split)
# ---------------------------------------------------------------------------


def resolve_holdouts(
    ast_sheet_path: str,
    drug: str,
    n_folds: int | None,
    fold: int,
    seed: int,
    evaluate_seed: int,
) -> tuple[list[str], list[str], dict[str, int], str]:
    """Reconstruct (evaluate_ids, validation_ids, label_map, source) for a drug.

    Mirrors ``finetune_amr``: k-fold mode derives the fixed evaluate holdout from ``evaluate_seed`` and the
    validation set from ``folds[fold]`` (with ``seed``); CSV mode reads ``train_val_eval``. Validation is
    needed to pick an operating threshold without peeking at the evaluate set.
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
    """Back-compat shim: evaluate ids + label map + source (no validation set).

    Evaluate ids are independent of ``fold``, so fold 0 is used internally.
    """
    evaluate_ids, _validation_ids, label_map, source = resolve_holdouts(
        ast_sheet_path, drug, n_folds, fold=0, seed=seed, evaluate_seed=evaluate_seed
    )
    return evaluate_ids, label_map, source


def load_splits(
    split_csv: str | Path, drug: str
) -> tuple[dict[str, int], list[str], list[str], list[str]]:
    """Resolve ``(label_map, train_ids, validate_ids, evaluate_ids)`` from a CSV ``train_val_eval`` column.

    The CSV must carry a ``Sample`` (or ``phenotype-BioSample_ID``) id column, the binary ``drug`` label
    column, and a ``train_val_eval`` split column. Ambiguous (non-0/1, e.g. the 0.5 intermediate) labels are
    dropped; duplicate Samples keep the first row. Used by the **label-blind** ranking builders (baclm/ESM
    per-segment stores), whose features do not leak from the FT model — so the CSV single-split is a
    consistent-if-different holdout, not the deployed FT k-fold one (see :func:`resolve_deployed_holdout`).
    """
    df = pd.read_csv(split_csv, low_memory=False)
    if "Sample" not in df.columns:
        if "phenotype-BioSample_ID" not in df.columns:
            raise ValueError("Split CSV must contain 'Sample' or 'phenotype-BioSample_ID'.")
        df["Sample"] = df["phenotype-BioSample_ID"].astype(str)
    df["Sample"] = df["Sample"].astype(str)
    for col in (drug, "train_val_eval"):
        if col not in df.columns:
            raise ValueError(f"Split CSV is missing required column {col!r}; has {list(df.columns)[:20]}")

    clean = df[df[drug].isin([0, 1])].drop_duplicates(subset="Sample", keep="first")
    label_map = {row["Sample"]: int(row[drug]) for _, row in clean.iterrows()}

    def _ids(value: str) -> list[str]:
        return [s for s in clean.loc[clean["train_val_eval"] == value, "Sample"] if s in label_map]

    train_ids, validate_ids, evaluate_ids = _ids("train"), _ids("validate"), _ids("evaluate")
    logger.info(
        "splits (clean 0/1): train=%d validate=%d evaluate=%d", len(train_ids), len(validate_ids), len(evaluate_ids)
    )
    return label_map, train_ids, validate_ids, evaluate_ids


# ---------------------------------------------------------------------------
# Deployed-model provenance — the ONLY safe holdout for an FT-derived feature
# ---------------------------------------------------------------------------


def _find_results_json(run_dir: str | Path) -> Path:
    """Locate a run's split-provenance JSON — training's ``results.json`` else evaluate's ``eval_results.json``.

    Accepts the run root or a ``checkpoint-*/`` subdir (the provenance sits in the run root, so its parent is
    also searched).
    """
    run_dir = Path(run_dir)
    for base in (run_dir, run_dir.parent):
        for name in ("results.json", "eval_results.json"):
            cand = base / name
            if cand.exists():
                return cand
    raise FileNotFoundError(
        f"No results.json / eval_results.json under {run_dir} (or its parent). "
        f"resolve_deployed_holdout needs the deployed run's split provenance to reproduce its holdout."
    )


def read_split_provenance(run_dir: str | Path) -> dict:
    """The ``split`` block from a deployed run's results.json: ``{source, n_folds, fold, evaluate_seed, n_evaluate}``."""
    payload = json.loads(_find_results_json(run_dir).read_text())
    split = payload.get("split")
    if not split or "source" not in split:
        raise ValueError(
            f"{run_dir}: results.json has no 'split' provenance block; cannot reproduce the deployed holdout."
        )
    return split


def resolve_deployed_holdout(
    run_dir: str | Path,
    ast_sheet_path: str | Path,
    drug: str,
) -> tuple[list[str], list[str], dict[str, int], str, int | None]:
    """Reproduce the EXACT holdout a deployed model was evaluated on, from its results.json split block.

    Reads ``split.{source, n_folds, fold, evaluate_seed}`` and replays :func:`resolve_holdouts` with those —
    k-fold when the model was k-fold-trained (the honest FT-unseen holdout), CSV single-split otherwise.
    The evaluate holdout is independent of the training ``seed`` (it is selected first from ``evaluate_seed``),
    so ``seed=1`` is used internally; ``fold`` only steers the validation set.

    Returns ``(evaluate_ids, validate_ids, label_map, source, n_evaluate_expected)`` where
    ``n_evaluate_expected`` is the deployed model's recorded evaluate count — the target for the
    cache/ladder coverage guard (a CSV-scope cache will not contain these genomes).
    """
    split = read_split_provenance(run_dir)
    source = str(split.get("source", "")).lower()
    n_folds = split.get("n_folds") if source == "kfold" else None
    fold = int(split.get("fold") or 0)
    evaluate_seed = int(split.get("evaluate_seed") or 1)
    evaluate_ids, validate_ids, label_map, resolved_source = resolve_holdouts(
        str(ast_sheet_path), drug, n_folds=n_folds, fold=fold, seed=1, evaluate_seed=evaluate_seed
    )
    return evaluate_ids, validate_ids, label_map, resolved_source, split.get("n_evaluate")


# ---------------------------------------------------------------------------
# Clean-label wrapper (0/1 only) over either source
# ---------------------------------------------------------------------------


def resolve_clean_splits(
    ast_sheet_path: str | Path,
    drug: str,
    *,
    checkpoint_dir: str | Path | None = None,
    n_folds: int | None = None,
    fold: int = 0,
    seed: int = 1,
    evaluate_seed: int = 1,
) -> tuple[dict[str, int], list[str], list[str], list[str], dict]:
    """Clean-0/1 ``(label_map, train_ids, validate_ids, evaluate_ids, split_info)`` for a drug.

    The holdout comes from ONE of two sources, and for an FT-derived feature it MUST match the deployed
    model's:

    - ``checkpoint_dir`` given → :func:`resolve_deployed_holdout` replays that model's own recorded split
      (k-fold or CSV, from its results.json). **Use this for anything scoring FT-derived features** — it is
      the only way to guarantee the evaluate genomes are ones the FT backbone never trained on.
    - ``checkpoint_dir=None`` → :func:`resolve_holdouts` with the passed ``n_folds/fold/seed/evaluate_seed``
      (``n_folds=None`` = the CSV ``train_val_eval`` single-split). For **label-blind** features only.

    Train ids are the labelled remainder (everything not in evaluate/validate). Ambiguous (non-0/1) labels
    are dropped from all splits. ``split_info`` records the raw (pre-clean) evaluate count and, when a
    checkpoint was used, the deployed model's ``n_evaluate`` — the coverage-guard target.
    """
    n_evaluate_expected: int | None = None
    if checkpoint_dir is not None:
        evaluate_raw, validate_raw, _lm, source, n_evaluate_expected = resolve_deployed_holdout(
            checkpoint_dir, ast_sheet_path, drug
        )
    else:
        evaluate_raw, validate_raw, _lm, source = resolve_holdouts(
            str(ast_sheet_path), drug, n_folds=n_folds, fold=fold, seed=seed, evaluate_seed=evaluate_seed
        )

    df = pd.read_csv(ast_sheet_path, low_memory=False)
    if "Sample" not in df.columns:
        if "phenotype-BioSample_ID" not in df.columns:
            raise ValueError("AST sheet must contain 'Sample' or 'phenotype-BioSample_ID'.")
        df["Sample"] = df["phenotype-BioSample_ID"].astype(str)
    df["Sample"] = df["Sample"].astype(str)
    if drug not in df.columns:
        raise ValueError(f"Drug column {drug!r} not in AST sheet; has {list(df.columns)[:20]}")

    labelled = df[df[drug].notna()]
    n_ambiguous = int((~labelled[drug].isin([0, 1])).sum())
    clean = labelled[labelled[drug].isin([0, 1])].drop_duplicates(subset="Sample", keep="first")
    label_map = {row["Sample"]: int(row[drug]) for _, row in clean.iterrows()}

    evaluate_set, validate_set = set(evaluate_raw), set(validate_raw)
    evaluate_ids = [s for s in evaluate_raw if s in label_map]
    validate_ids = [s for s in validate_raw if s in label_map]
    train_ids = [s for s in label_map if s not in evaluate_set and s not in validate_set]

    split_info = {
        "source": source,
        "n_evaluate_raw": len(evaluate_raw),  # incl. ambiguous — matches the deployed model's n_evaluate
        "n_validate_raw": len(validate_raw),
        "n_ambiguous_dropped": n_ambiguous,
        "n_train": len(train_ids),
        "n_validate": len(validate_ids),
        "n_evaluate": len(evaluate_ids),
        "n_evaluate_expected": n_evaluate_expected,  # deployed model's recorded count (checkpoint mode only)
    }
    logger.info(
        "splits (%s, clean 0/1): train=%d validate=%d evaluate=%d (dropped %d ambiguous labels)",
        source, len(train_ids), len(validate_ids), len(evaluate_ids), n_ambiguous,
    )
    return label_map, train_ids, validate_ids, evaluate_ids, split_info
