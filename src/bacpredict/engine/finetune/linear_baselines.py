r"""Linear-model baselines for binary phenotype prediction.

Fits ``sklearn.LogisticRegression`` on the SAME train/eval split the deep model
was scored on, across a *nested sequence* of feature blocks:

- ``country`` — one-hot ``country_parsed``
- ``sublineage`` — one-hot ``Sublineage``
- ``k_locus`` — one-hot Kleborate ``K_locus`` (capsular type, hundreds of categories)
- ``virulence_bsc`` — 6 binary flags: Yersiniabactin / Colibactin / Aerobactin /
  Salmochelin / RmpADC / rmpA2 (parsed via ``bac_kleborate.parsing``)
- ``amr_class`` — ~15 binary flags: every ``<class>_acquired`` column

plus their concatenations (``country+sublineage``, …, all-of-the-above) so the
**nested ΔAUROC** of each added block is directly comparable. Reuses
``bacpredict.engine.finetune.metrics.compute_full_metrics`` for §0.4 metrics (AUROC, AUPRC,
sensitivity, specificity, balanced accuracy, F1, confusion, calibration).

Output: a single JSON file with one entry per feature-set recipe; optionally
appends (idempotently) a Markdown table to a ``stratification_report.md`` with
a ΔAUROC-vs-previous-row column so the nested-ladder story reads top-to-bottom.

Reusable Kleborate parsing lives in ``bac_kleborate.parsing`` (BacHGT sibling
repo, installed via path dep) — single source of truth for the cell-presence
rule, virulence-locus schema, and acquired-AMR token splitting.

Usage::

    uv run python -m bacpredict.engine.finetune.linear_baselines \\
        --sheet-path <cohort>/binary_<pair>_with_split.csv \\
        --label-column blood_vs_faeces_label \\
        --metadata-file /home/.../metadata_v2_all_samples_and_columns.tsv \\
        --feature-sets country sublineage country+sublineage \\
                        k_locus virulence_bsc amr_class \\
                        country+sublineage+k_locus \\
                        country+sublineage+k_locus+virulence \\
                        country+sublineage+k_locus+virulence+amr \\
        --out <cohort>/linear_baselines.json \\
        [--also-score-validate] \\
        [--update-report <cohort>/stratification_report.md] \\
        [--task kleb_iso_source]
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import scipy.sparse as sp
from bac_kleborate.parsing import (
    KLEBORATE_VIRULENCE_LOCI,
    acquired_column_names,
    amr_class_presence,
    virulence_cluster_presence,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from bacpredict.engine.finetune.metrics import compute_full_metrics

DEFAULT_METADATA = Path(
    "/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_v2_all_samples_and_columns.tsv"
)

_NA_TOKEN = "__NA__"


# ---------------------------------------------------------------------------
# Feature-block definitions
# ---------------------------------------------------------------------------


@dataclass
class FeatureBlock:
    """One block in a linear-baseline feature stack.

    ``input_cols`` lists metadata columns the block's :attr:`materialise` reads
    from. For dynamically-discovered blocks (the AMR-class one), provide
    :attr:`discover_input_cols` instead — given the metadata's column list, it
    returns the set of columns the block needs.

    Exactly one of :attr:`categorical_materialise` / :attr:`binary_materialise`
    should be set. Categorical materialised frames go through ``OneHotEncoder``;
    binary frames are passed through verbatim as sparse 0/1 columns.
    """

    name: str
    input_cols: list[str] = field(default_factory=list)
    discover_input_cols: Callable[[Iterable[str]], list[str]] | None = None
    categorical_materialise: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    binary_materialise: Callable[[pd.DataFrame], pd.DataFrame] | None = None

    def resolve_input_cols(self, metadata_columns: Iterable[str]) -> list[str]:
        """Return the metadata columns this block actually needs given an available column set."""
        if self.discover_input_cols is not None:
            return self.discover_input_cols(metadata_columns)
        return list(self.input_cols)


def _identity_frame_factory(cols: list[str]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return ``df -> df[cols]`` — the trivial materialise for already-present categorical columns."""

    def _take(df: pd.DataFrame) -> pd.DataFrame:
        return df[cols]

    return _take


# Virulence allele columns spanning all 6 BSCs (input to bac_kleborate.virulence_cluster_presence).
_VIRULENCE_INPUT_COLS = sorted(
    {allele for info in KLEBORATE_VIRULENCE_LOCI.values() for allele in info["alleles"]}
)


FEATURE_BLOCKS: dict[str, FeatureBlock] = {
    "country": FeatureBlock(
        name="country",
        input_cols=["country_parsed"],
        categorical_materialise=_identity_frame_factory(["country_parsed"]),
    ),
    "sublineage": FeatureBlock(
        name="sublineage",
        input_cols=["Sublineage"],
        categorical_materialise=_identity_frame_factory(["Sublineage"]),
    ),
    "k_locus": FeatureBlock(
        name="k_locus",
        input_cols=["K_locus"],
        categorical_materialise=_identity_frame_factory(["K_locus"]),
    ),
    "virulence_bsc": FeatureBlock(
        name="virulence_bsc",
        input_cols=_VIRULENCE_INPUT_COLS,
        binary_materialise=virulence_cluster_presence,
    ),
    "amr_class": FeatureBlock(
        name="amr_class",
        discover_input_cols=acquired_column_names,
        binary_materialise=amr_class_presence,
    ),
}


FEATURE_SET_RECIPES: dict[str, list[str]] = {
    # Single blocks
    "country": ["country"],
    "sublineage": ["sublineage"],
    "k_locus": ["k_locus"],
    "virulence_bsc": ["virulence_bsc"],
    "amr_class": ["amr_class"],
    # Pairs
    "country+sublineage": ["country", "sublineage"],
    # Nested ladder (current ceiling → +k_locus → +virulence → +amr)
    "country+sublineage+k_locus": ["country", "sublineage", "k_locus"],
    "country+sublineage+k_locus+virulence": ["country", "sublineage", "k_locus", "virulence_bsc"],
    "country+sublineage+k_locus+virulence+amr": [
        "country",
        "sublineage",
        "k_locus",
        "virulence_bsc",
        "amr_class",
    ],
}


# ---------------------------------------------------------------------------
# Loaders + materialisation
# ---------------------------------------------------------------------------


def _load_split(sheet_path: Path, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(sheet_path, low_memory=False)
    for col in ("Sample", "train_val_eval", label_column):
        if col not in df.columns:
            raise ValueError(f"Split CSV {sheet_path} is missing required column {col!r}")
    df["Sample"] = df["Sample"].astype(str)
    return df


def _load_metadata_subset(metadata_file: Path, needed_cols: list[str]) -> pd.DataFrame:
    """Load only ``needed_cols`` (plus the join key) from the v2 metadata TSV.

    Reads the header first so we can skip columns the file doesn't carry without
    crashing the whole load, and so the AMR-class discovery has the full column
    list to scan.
    """
    header_cols = list(pd.read_csv(metadata_file, sep="\t", nrows=0, low_memory=False).columns)
    if "sample_accession" not in header_cols:
        raise ValueError(f"Metadata {metadata_file} lacks the sample_accession column")
    usecols = ["sample_accession", *(c for c in needed_cols if c in header_cols)]
    missing = [c for c in needed_cols if c not in header_cols]
    if missing:
        logging.warning("  metadata missing requested columns (ignored): %s", missing)
    meta = pd.read_csv(metadata_file, sep="\t", usecols=usecols, low_memory=False)
    # v2 carries ~115 duplicate sample_accessions; keep first.
    meta = meta.drop_duplicates("sample_accession", keep="first")
    meta = meta.set_index(meta["sample_accession"].astype(str))
    return meta.drop(columns=["sample_accession"])


def _materialise_blocks(
    joined: pd.DataFrame, blocks: dict[str, FeatureBlock]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Run every block's materialise callback once → (categorical_frames, binary_frames).

    Frames are indexed by ``joined.index`` (Sample) and contain only the block's
    feature columns. Caller selects the train/eval row subsets afterwards.
    """
    cat: dict[str, pd.DataFrame] = {}
    bin_: dict[str, pd.DataFrame] = {}
    for name, block in blocks.items():
        if block.categorical_materialise is not None:
            cat[name] = block.categorical_materialise(joined)
        if block.binary_materialise is not None:
            bin_[name] = block.binary_materialise(joined)
    return cat, bin_


# ---------------------------------------------------------------------------
# Fit + score
# ---------------------------------------------------------------------------


def _build_design_matrix(
    categorical_frames: list[pd.DataFrame],
    binary_frames: list[pd.DataFrame],
    train_idx: pd.Index,
    test_idx: pd.Index,
) -> tuple[sp.csr_matrix, sp.csr_matrix, int]:
    """Build sparse train/test design matrices, hstack-ing categorical (one-hot) + binary (passthrough)."""
    blocks_train: list[sp.csr_matrix] = []
    blocks_test: list[sp.csr_matrix] = []
    for frame in categorical_frames:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        blocks_train.append(enc.fit_transform(frame.loc[train_idx].astype(str).fillna(_NA_TOKEN)))
        blocks_test.append(enc.transform(frame.loc[test_idx].astype(str).fillna(_NA_TOKEN)))
    for frame in binary_frames:
        blocks_train.append(sp.csr_matrix(frame.loc[train_idx].fillna(0).astype(float).values))
        blocks_test.append(sp.csr_matrix(frame.loc[test_idx].fillna(0).astype(float).values))
    X_train = sp.hstack(blocks_train, format="csr") if blocks_train else sp.csr_matrix((len(train_idx), 0))
    X_test = sp.hstack(blocks_test, format="csr") if blocks_test else sp.csr_matrix((len(test_idx), 0))
    return X_train, X_test, X_train.shape[1]


def _fit_and_score_recipe(
    recipe_blocks: list[str],
    categorical_frames_all: dict[str, pd.DataFrame],
    binary_frames_all: dict[str, pd.DataFrame],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str,
) -> tuple[dict[str, Any], int, str]:
    """Fit LR on TRAIN, score on TEST for one recipe; returns (§0.4 metrics, n_features, model_repr)."""
    cats = [categorical_frames_all[b] for b in recipe_blocks if b in categorical_frames_all]
    bins = [binary_frames_all[b] for b in recipe_blocks if b in binary_frames_all]
    X_train, X_test, n_feat = _build_design_matrix(cats, bins, train_df.index, test_df.index)
    y_train = train_df[label_column].astype(int).to_numpy()
    y_test = test_df[label_column].astype(int).to_numpy()
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return compute_full_metrics(y_test, y_prob), int(n_feat), repr(model)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_baselines(
    sheet_path: Path,
    label_column: str,
    metadata_file: Path,
    feature_sets: list[str],
    also_score_validate: bool = False,
    task: str = "linear_baseline",
) -> dict[str, Any]:
    """Fit + score every requested feature-set recipe; return a JSON-ready payload."""
    unknown = [fs for fs in feature_sets if fs not in FEATURE_SET_RECIPES]
    if unknown:
        raise ValueError(f"Unknown feature sets: {unknown}. Known: {sorted(FEATURE_SET_RECIPES)}")
    needed_blocks = sorted({b for fs in feature_sets for b in FEATURE_SET_RECIPES[fs]})

    split_df = _load_split(sheet_path, label_column).set_index("Sample")
    # Discover the union of metadata columns the requested blocks need.
    header_cols = list(pd.read_csv(metadata_file, sep="\t", nrows=0, low_memory=False).columns)
    needed_meta_cols: set[str] = set()
    for b in needed_blocks:
        needed_meta_cols.update(FEATURE_BLOCKS[b].resolve_input_cols(header_cols))
    missing_in_split = sorted(needed_meta_cols - set(split_df.columns))
    if missing_in_split:
        meta_df = _load_metadata_subset(metadata_file, missing_in_split)
        joined = split_df.join(meta_df, how="left")
        logging.info("  joined %d metadata column(s): %s", len(missing_in_split), missing_in_split[:6])
    else:
        joined = split_df
        logging.info("  all needed columns present in split CSV — no metadata join")

    joined = joined.dropna(subset=[label_column])
    train_df = joined[joined["train_val_eval"] == "train"].copy()
    eval_df = joined[joined["train_val_eval"] == "evaluate"].copy()
    val_df = joined[joined["train_val_eval"] == "validate"].copy()
    logging.info("  splits: train=%d  validate=%d  evaluate=%d", len(train_df), len(val_df), len(eval_df))
    if train_df.empty or eval_df.empty:
        raise ValueError(
            f"After filter, train or evaluate split is empty (train={len(train_df)}, eval={len(eval_df)})."
        )

    selected_blocks = {b: FEATURE_BLOCKS[b] for b in needed_blocks}
    categorical_frames_all, binary_frames_all = _materialise_blocks(joined, selected_blocks)

    baselines: dict[str, Any] = {}
    for fs in feature_sets:  # preserve user-supplied order so ΔAUROC reads top-to-bottom
        blocks = FEATURE_SET_RECIPES[fs]
        logging.info("  fitting %s (blocks=%s)", fs, blocks)
        eval_metrics, n_feat, model_repr = _fit_and_score_recipe(
            blocks, categorical_frames_all, binary_frames_all, train_df, eval_df, label_column
        )
        entry: dict[str, Any] = {
            "blocks": blocks,
            "model_repr": model_repr,
            "n_features": n_feat,
            "metrics": eval_metrics,
        }
        if also_score_validate and not val_df.empty:
            val_metrics, _, _ = _fit_and_score_recipe(
                blocks, categorical_frames_all, binary_frames_all, train_df, val_df, label_column
            )
            entry["metrics_validate"] = val_metrics
        baselines[fs] = entry
        logging.info(
            "    eval AUROC=%.4f  AUPRC=%.4f  bal-acc=%.4f  (n_feat=%d)",
            eval_metrics["auroc"], eval_metrics["auprc"], eval_metrics["balanced_accuracy"], n_feat,
        )

    return {
        "schema_version": "2.0",
        "task": task,
        "label_column": label_column,
        "sheet_path": str(sheet_path),
        "metadata_file": str(metadata_file),
        "n_train": int(len(train_df)),
        "n_evaluate": int(len(eval_df)),
        "n_validate": int(len(val_df)),
        "baselines": baselines,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
    }


# ---------------------------------------------------------------------------
# Report append (idempotent)
# ---------------------------------------------------------------------------


_BASELINE_HEADING = "## Linear-model baseline (LogisticRegression on country / Sublineage / K_locus / virulence / AMR features)"
_PRIOR_HEADINGS = (
    _BASELINE_HEADING,
    "## Linear-model baseline (LogisticRegression on country / Sublineage one-hot features)",  # legacy v1
    "## Metadata-only baseline (LogisticRegression on one-hot features)",  # legacy pre-rename
)


def _strip_prior_section(report_text: str) -> str:
    """Remove any prior baseline section (delimited by the heading and the next ``## `` heading or EOF)."""
    lines = report_text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        if line in _PRIOR_HEADINGS:
            i += 1
            while i < len(lines) and not lines[i].startswith("## "):
                i += 1
            continue
        out.append(lines[i])
        i += 1
    text = "".join(out)
    while text.endswith("\n\n"):
        text = text[:-1]
    return text


def append_report_section(report_path: Path, payload: dict[str, Any]) -> None:
    """Append the linear-model baseline table to a ``stratification_report.md`` (idempotent).

    Adds a ``ΔAUROC vs prev`` column so the nested-ladder story (country → +SL →
    +K_locus → +virulence → +AMR) reads top-to-bottom.
    """
    if not report_path.exists():
        logging.warning("  --update-report: %s does not exist; skipping", report_path)
        return
    existing = report_path.read_text()
    stripped = _strip_prior_section(existing)
    if stripped != existing:
        logging.info("  --update-report: removed prior baseline section before re-appending")

    label = payload["label_column"]
    lines = [
        "",
        _BASELINE_HEADING,
        "",
        "Fit on the TRAIN split, scored on the held-out EVALUATE split. Pure sanity check: "
        "shows the AUROC achievable from each feature block (country, Sublineage, K_locus, "
        "virulence BSCs, acquired-AMR classes) alone or in combination, without any genomic "
        "embedding features. The ΔAUROC column reads the nested ladder: how much does adding "
        "each block on top of the previous row lift the linear ceiling?",
        "",
        "| Feature set | n_features | AUROC | AUPRC | bal-acc | F1 | ΔAUROC vs prev |",
        "|---|---|---|---|---|---|---|",
    ]
    prev_auroc: float | None = None
    for name, b in payload["baselines"].items():
        m = b["metrics"]
        auroc = float(m["auroc"])
        delta = "" if prev_auroc is None else f"{auroc - prev_auroc:+.3f}"
        lines.append(
            f"| {name} | {b['n_features']} | {auroc:.3f} | {m['auprc']:.3f} "
            f"| {m['balanced_accuracy']:.3f} | {m['f1']:.3f} | {delta} |"
        )
        prev_auroc = auroc
    lines += [
        "",
        f"_n_train={payload['n_train']:,}, n_evaluate={payload['n_evaluate']:,}, label={label}._",
        "",
    ]
    report_path.write_text(stripped + "\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main_cli() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sheet-path", type=Path, required=True,
                   help="Split CSV (Sample, <label>, train_val_eval).")
    p.add_argument("--label-column", required=True,
                   help="Binary label column name in the split CSV (e.g. blood_vs_faeces_label).")
    p.add_argument("--metadata-file", type=Path, default=DEFAULT_METADATA,
                   help="v2 metadata TSV (country_parsed + Sublineage + K_locus + Kleborate virulence/AMR columns).")
    p.add_argument(
        "--feature-sets", nargs="+",
        default=[
            "country", "sublineage", "country+sublineage",
            "k_locus", "virulence_bsc", "amr_class",
            "country+sublineage+k_locus",
            "country+sublineage+k_locus+virulence",
            "country+sublineage+k_locus+virulence+amr",
        ],
        choices=list(FEATURE_SET_RECIPES),
        help="Which feature-set recipes to fit. Order is preserved so the ΔAUROC column reads the nested ladder.",
    )
    p.add_argument("--out", type=Path, required=True,
                   help="Path to write linear_baselines.json.")
    p.add_argument("--also-score-validate", action="store_true",
                   help="Also score on the validate split (handy for direct val-curve comparison).")
    p.add_argument("--update-report", type=Path, default=None,
                   help="If set, append (idempotently) a 'Linear-model baseline' section to this stratification_report.md.")
    p.add_argument("--task", default="linear_baseline",
                   help="Free-form task label stored in the JSON (e.g. kleb_iso_source).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    payload = run_baselines(
        sheet_path=args.sheet_path,
        label_column=args.label_column,
        metadata_file=args.metadata_file,
        feature_sets=list(args.feature_sets),
        also_score_validate=args.also_score_validate,
        task=args.task,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    logging.info("wrote %s", args.out)
    if args.update_report is not None:
        append_report_section(args.update_report, payload)
        logging.info("appended baseline section to %s", args.update_report)


if __name__ == "__main__":
    _main_cli()
