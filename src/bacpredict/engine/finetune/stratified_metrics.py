r"""Per-stratum evaluation of a scored holdout — the "does it hold *within* a group?" test.

A pooled AUROC cannot distinguish a model that reads the phenotype from a model that reads the
population structure the phenotype happens to correlate with. Scoring the *same* holdout separately
within each lineage answers that directly: if discrimination survives inside a single clone, lineage
membership alone cannot be the explanation.

This is deliberately organism-agnostic — the same question is open for TB and Kp AST
("HGT-vs-vertical stratified performance", ToDo.md) as for *Klebsiella* isolation source. Any
``eval_scores.npz`` plus a metadata table carrying the grouping column will work.

Two things this module refuses to do quietly, because both mislead:

- **It never drops a small group silently.** Groups below ``--min-group-n`` are pooled into a single
  ``other`` row that is still reported, and the per-group ``n`` always sums to the holdout size.
- **It never reports a bare point estimate.** Every AUROC carries a bootstrap confidence interval.
  At n≈160 (a mid-sized sublineage in a 2.8k holdout) the 95% CI spans roughly ±0.08, so a group
  reading 0.74 against a pooled 0.79 is not a detected drop. Read the intervals, not the ranks.

Sample IDs come from the ``sample_ids`` array written into ``eval_scores.npz``. Runs scored before
that field existed are still usable: pass ``--checkpoint-dir`` and the deployed run's own recorded
split is replayed via :func:`~bacpredict.engine.finetune.holdout.resolve_deployed_holdout`, which
reproduces the exact ``evaluate_ids`` order the ``shuffle=False`` loader consumed. The row count is
asserted either way, so a mismatch fails loudly rather than silently mis-joining scores to genomes.

Usage::

    python -m bacpredict.engine.finetune.stratified_metrics \
        --eval-scores  <cohort>/models/eval_scores.npz \
        --metadata     <cohort>/binary_blood_vs_faeces_with_split.csv \
        --group-column Sublineage \
        --out          <cohort>/models/per_sublineage_metrics.csv \
        [--checkpoint-dir <cohort>/models] [--drug blood_vs_faeces_label] \
        [--min-group-n 100] [--n-boot 2000]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bacpredict.engine.finetune.metrics import compute_full_metrics

logger = logging.getLogger(__name__)

OTHER_LABEL = "other"
NA_LABEL = "no-call"
DEFAULT_MIN_GROUP_N = 100
DEFAULT_N_BOOT = 2000


# ---------------------------------------------------------------------------
# Loading + id resolution
# ---------------------------------------------------------------------------


def load_eval_scores(npz_path: str | Path) -> dict[str, Any]:
    """Read an ``eval_scores.npz`` into ``{y_true, y_prob, sample_ids|None, drug, threshold}``.

    ``sample_ids`` is ``None`` for archives written before the field was added; the caller then has
    to reconstruct the ids from the deployed run's split provenance.
    """
    data = np.load(npz_path, allow_pickle=False)
    sample_ids = [str(s) for s in data["sample_ids"]] if "sample_ids" in data.files else None
    threshold = float(data["operating_threshold"]) if "operating_threshold" in data.files else float("nan")
    return {
        "y_true": np.asarray(data["y_true"]).astype(int),
        "y_prob": np.asarray(data["y_prob"]).astype(float),
        "sample_ids": sample_ids,
        "drug": str(data["drug"]) if "drug" in data.files else None,
        "threshold": threshold,
    }


def resolve_sample_ids(
    scores: dict[str, Any],
    checkpoint_dir: str | Path | None,
    metadata_path: str | Path,
    drug: str,
) -> list[str]:
    """Return the holdout sample ids aligned row-for-row with ``scores['y_true']``.

    Prefers the ids stored in the npz. Falls back to replaying the deployed run's own split
    provenance, which regenerates ``evaluate_ids`` in the same order the ``shuffle=False`` loader
    used. Either way the length is checked against the score array, so a stale or wrong-cohort
    metadata file fails here rather than producing a silently mis-joined table.
    """
    n = len(scores["y_true"])
    if scores["sample_ids"] is not None:
        ids = scores["sample_ids"]
        source = "npz sample_ids"
    else:
        if checkpoint_dir is None:
            raise ValueError(
                "eval_scores.npz has no 'sample_ids' (written before that field existed) and no "
                "--checkpoint-dir was given. Pass --checkpoint-dir so the deployed run's recorded "
                "split can be replayed, or re-run evaluate.py to regenerate the npz with ids."
            )
        # Imported lazily: holdout pulls pandas only, but keeping it off the module import path
        # means a caller with ids already in the npz never pays for split reconstruction.
        from bacpredict.engine.finetune.holdout import resolve_deployed_holdout

        ids, _validate_ids, _label_map, source_kind, _n_expected = resolve_deployed_holdout(
            checkpoint_dir, metadata_path, drug
        )
        source = f"replayed {source_kind} split from {Path(checkpoint_dir).name}/results.json"

    if len(ids) != n:
        raise ValueError(
            f"Holdout id count ({len(ids)}, from {source}) != score row count ({n}). The scores and "
            f"the metadata/checkpoint do not describe the same run — refusing to join."
        )
    logger.info("resolved %d holdout sample ids (%s)", n, source)
    return list(ids)


def join_groups(
    sample_ids: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metadata_path: str | Path,
    group_column: str,
    id_column: str = "Sample",
) -> pd.DataFrame:
    """Attach the grouping column to the scored holdout rows.

    Missing/NaN group values become an explicit ``no-call`` group rather than being dropped —
    a genome with no lineage call is still a scored genome and must stay in the accounting.
    """
    meta = pd.read_csv(metadata_path, low_memory=False)
    if id_column not in meta.columns:
        raise ValueError(f"Metadata {metadata_path} has no {id_column!r} column.")
    if group_column not in meta.columns:
        raise ValueError(
            f"Metadata {metadata_path} has no {group_column!r} column. "
            f"Note 'Clonal group' contains a space. Available (first 30): {list(meta.columns)[:30]}"
        )
    meta[id_column] = meta[id_column].astype(str)
    lookup = meta.drop_duplicates(subset=id_column, keep="first").set_index(id_column)[group_column]

    scored = pd.DataFrame({"Sample": sample_ids, "y_true": y_true, "y_prob": y_prob})
    scored["group"] = scored["Sample"].map(lookup)
    n_unmatched = int(scored["group"].isna().sum())
    if n_unmatched:
        logger.warning(
            "  %d/%d holdout genomes have no %s value (kept as %r)",
            n_unmatched, len(scored), group_column, NA_LABEL,
        )
    scored["group"] = scored["group"].fillna(NA_LABEL).astype(str)
    return scored


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = 0.05,
    seed: int = 1,
) -> tuple[float, float, int]:
    """Percentile bootstrap CI for AUROC → ``(lo, hi, n_valid)``.

    Resamples genomes with replacement. Draws that end up single-class have no defined AUROC and are
    skipped, with ``n_valid`` reporting how many survived — for a small or heavily imbalanced group
    that count falls well below ``n_boot`` and the interval should be read as indicative only.
    Returns ``(nan, nan, 0)`` when the group itself is single-class.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), 0

    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        stats.append(roc_auc_score(yt, y_prob[idx]))
    if not stats:
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), len(stats)


# ---------------------------------------------------------------------------
# The per-group table
# ---------------------------------------------------------------------------


def _group_row(name: str, sub: pd.DataFrame, n_boot: int, seed: int, threshold: float) -> dict[str, Any]:
    """One output row: §0.4 metrics for a group plus its AUROC bootstrap CI."""
    y_true = sub["y_true"].to_numpy().astype(int)
    y_prob = sub["y_prob"].to_numpy().astype(float)
    n_pos = int(y_true.sum())
    row: dict[str, Any] = {
        "group": name,
        "n": int(len(sub)),
        "n_pos": n_pos,
        "n_neg": int(len(sub) - n_pos),
        "prevalence": float(y_true.mean()) if len(y_true) else float("nan"),
    }
    if len(np.unique(y_true)) < 2:
        # Single-class group: AUROC is undefined, not zero. Emit NaN and keep the row visible.
        row.update({
            "auroc": float("nan"), "auroc_ci_lo": float("nan"), "auroc_ci_hi": float("nan"),
            "n_boot_valid": 0, "auprc": float("nan"), "sensitivity": float("nan"),
            "specificity": float("nan"), "balanced_accuracy": float("nan"), "f1": float("nan"),
            "single_class": True,
        })
        return row

    thr = threshold if np.isfinite(threshold) else 0.5
    m = compute_full_metrics(y_true, y_prob, threshold=thr)
    lo, hi, n_valid = bootstrap_auroc_ci(y_true, y_prob, n_boot=n_boot, seed=seed)
    row.update({
        "auroc": m["auroc"], "auroc_ci_lo": lo, "auroc_ci_hi": hi, "n_boot_valid": n_valid,
        "auprc": m["auprc"], "sensitivity": m["sensitivity"], "specificity": m["specificity"],
        "balanced_accuracy": m["balanced_accuracy"], "f1": m["f1"], "single_class": False,
    })
    return row


def stratified_metrics(
    scored: pd.DataFrame,
    min_group_n: int = DEFAULT_MIN_GROUP_N,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 1,
    threshold: float = float("nan"),
) -> pd.DataFrame:
    """Per-group §0.4 metrics + AUROC CIs, with a pooled row and an ``other`` bucket.

    ``scored`` needs columns ``group``, ``y_true``, ``y_prob``. Groups with fewer than
    ``min_group_n`` genomes are merged into one ``other`` row (reported, never dropped), so the
    ``n`` column sums to the holdout size. The first row is always the pooled ``__pooled__``
    baseline, so every group can be read against the number it is meant to explain.
    """
    for col in ("group", "y_true", "y_prob"):
        if col not in scored.columns:
            raise ValueError(f"scored frame is missing required column {col!r}")

    counts = scored["group"].value_counts()
    big = [g for g in counts.index if counts[g] >= min_group_n]
    small = [g for g in counts.index if counts[g] < min_group_n]

    rows = [_group_row("__pooled__", scored, n_boot, seed, threshold)]
    rows[0]["n_groups"] = 1
    for g in sorted(big, key=lambda x: -counts[x]):
        row = _group_row(str(g), scored[scored["group"] == g], n_boot, seed, threshold)
        row["n_groups"] = 1
        rows.append(row)
    if small:
        sub = scored[scored["group"].isin(small)]
        row = _group_row(OTHER_LABEL, sub, n_boot, seed, threshold)
        row["n_groups"] = len(small)
        rows.append(row)
        logger.info("  %d groups below n=%d pooled into %r (n=%d)", len(small), min_group_n, OTHER_LABEL, len(sub))

    out = pd.DataFrame(rows)
    total = int(out.loc[out["group"] != "__pooled__", "n"].sum())
    if total != len(scored):
        raise AssertionError(f"per-group n sums to {total} but the holdout has {len(scored)} rows")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-scores", type=Path, required=True, help="Path to eval_scores.npz.")
    p.add_argument("--metadata", type=Path, required=True,
                   help="Table carrying the id column and the grouping column (e.g. the split CSV).")
    p.add_argument("--group-column", type=str, default="Sublineage",
                   help="Column to stratify by (default: Sublineage; 'Clonal group' has a space).")
    p.add_argument("--id-column", type=str, default="Sample")
    p.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="Deployed run dir (holding results.json). Needed only when the npz predates sample_ids.")
    p.add_argument("--drug", type=str, default=None,
                   help="Label column, for the fallback split replay. Defaults to the npz's stored drug.")
    p.add_argument("--min-group-n", type=int, default=DEFAULT_MIN_GROUP_N)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    scores = load_eval_scores(args.eval_scores)
    drug = args.drug or scores["drug"]
    if drug is None:
        raise SystemExit("Could not determine the label column; pass --drug.")

    sample_ids = resolve_sample_ids(scores, args.checkpoint_dir, args.metadata, drug)
    scored = join_groups(
        sample_ids, scores["y_true"], scores["y_prob"], args.metadata, args.group_column, args.id_column
    )
    table = stratified_metrics(
        scored, min_group_n=args.min_group_n, n_boot=args.n_boot, seed=args.seed, threshold=scores["threshold"]
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps({
        "eval_scores": str(args.eval_scores),
        "metadata": str(args.metadata),
        "group_column": args.group_column,
        "drug": drug,
        "n_holdout": int(len(scored)),
        "min_group_n": args.min_group_n,
        "n_boot": args.n_boot,
        "threshold": scores["threshold"],
    }, indent=2))

    pooled = table.iloc[0]
    print(f"\nPooled AUROC {pooled['auroc']:.4f} (n={pooled['n']})  — per-{args.group_column}:")
    for _, r in table.iloc[1:].iterrows():
        if r["single_class"]:
            print(f"  {r['group']:<24} n={r['n']:<6} SINGLE-CLASS (all {'pos' if r['n_pos'] else 'neg'}) — AUROC undefined")
        else:
            print(
                f"  {r['group']:<24} n={r['n']:<6} prev={r['prevalence']:.2f}  "
                f"AUROC {r['auroc']:.4f} [{r['auroc_ci_lo']:.3f}, {r['auroc_ci_hi']:.3f}]"
            )
    print(f"\nWrote {args.out} and {meta_path}")


if __name__ == "__main__":
    _main_cli()
