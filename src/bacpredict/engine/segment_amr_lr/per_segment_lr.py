"""Rank AMR segments of one type by a per-segment logistic regression — the unified non-coding screen.

Folds the three non-coding ``gene_lr.build_*`` ranking stores into ONE driver over a per-type
:class:`bacpredict.engine.embedding.segment_locator.SegmentLocator`:

* :mod:`bacpredict.engine.gene_lr.build_per_igr_lr_store` — intergenic regions keyed by flank pair ``a→b``;
* :mod:`bacpredict.engine.gene_lr.build_upstream_region_lr_store` — 5′-anchored ``upstream:<gene>`` regions;
* :mod:`bacpredict.engine.gene_lr.build_per_unit_lr_store` — named non-CDS bodies ``<type>:<name>``.

Each was the SAME operation — locate every segment of a type in each genome, keep the single-copy (for units,
mean-pooled) occurrences in a prevalence band, fit one LR per segment, rank by out-of-fold train AUROC. They
differed only in the per-type keying / GFF / dedup, which now lives entirely in the locator and the uniform
two-pass sweep (:func:`bacpredict.engine.embedding.segment_embedding_extractor.collect_segment_matrices`); the
fit is the one shared engine (:func:`bacpredict.engine.segment_amr_lr.fit_lr.fit_per_segment`).

**The correctness spine.** Every segment's LR fits on the deployed ``train`` split, selects by its
out-of-fold train AUROC, and is scored on the deployed ``holdout`` — read from the one materialized
``<drug>_split.csv`` via :func:`bacpredict.engine.splits.load_splits.load_splits`. This closes the A1 gap (the
legacy per-IGR store reported no held-out number at all) and makes every non-coding rung report a real
``eval_auroc_<drug>`` on the *same* holdout the coding screen and the concat ladder use.

Three absence modes, orthogonal to the segment type (preserved verbatim from the siblings):

* **carrier-only** (``feature="embedding"``, default) — fit each segment over the genomes carrying it,
  conditioned on presence;
* **zero-imputed** (``impute_absent_zero=True``) — fit over the full read universe, a 0-vector for absent
  genomes, so the LR sees presence/absence + the embedding (*selection = usage*, what the concat consumes);
* **presence** (``feature="presence"``) — a single presence/absence one-hot, the lineage/synteny control.

Per-type output tables keep their legacy column schemas (so the ladder + plot modules read them by column
name unchanged) — the only change is that ``igr`` gains the held-out ``eval_auroc_<drug>`` / ``n_eval`` /
``n_eval_pos`` columns the other two already had. The de-"gene" vocabulary rename is a later, uniform pass,
not this fold-in.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bacpredict.engine.embedding.segment_embedding_extractor import collect_segment_matrices
from bacpredict.engine.embedding.segment_locator import IgrLocator, SegmentLocator, UnitLocator, UpstreamLocator
from bacpredict.engine.gene_lr.build_per_gene_lr_store import subsample_balanced  # relocates in c3
from bacpredict.engine.segment_amr_lr.fit_lr import fit_per_segment
from bacpredict.engine.splits.load_splits import load_splits

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass(frozen=True)
class SegmentTypeSpec:
    """Per-type configuration for the otherwise type-agnostic ranking driver.

    Attributes
    ----------
    id_column
        The segment-id column name in the ranking + prevalence tables (``igr_pair`` / ``upstream_gene`` /
        ``unit``).
    extra_id_columns
        Extra identity columns split out of the segment id (``left_gene``/``right_gene`` for igr, ``gene``
        for upstream, ``feature_type``/``feature_name`` for unit).
    needs_gff
        Whether the locator needs the per-sample Bakta GFF map (igr / upstream) or self-identifies (unit).
    """

    id_column: str
    extra_id_columns: tuple[str, ...]
    needs_gff: bool


SEGMENT_TYPES: dict[str, SegmentTypeSpec] = {
    "igr": SegmentTypeSpec("igr_pair", ("left_gene", "right_gene"), needs_gff=True),
    "upstream": SegmentTypeSpec("upstream_gene", ("gene",), needs_gff=True),
    "unit": SegmentTypeSpec("unit", ("feature_type", "feature_name"), needs_gff=False),
}


def _extra_id_values(segment_type: str, segment_id: str) -> dict[str, str]:
    """Split a segment id into its per-type extra identity columns (see :data:`SEGMENT_TYPES`)."""
    if segment_type == "igr":
        left, _, right = segment_id.partition("→")
        return {"left_gene": left, "right_gene": right}
    if segment_type == "upstream":
        gene = segment_id.split("upstream:", 1)[-1] if segment_id.startswith("upstream:") else segment_id
        return {"gene": gene}
    if segment_type == "unit":
        ftype, _, fname = segment_id.partition(":")
        return {"feature_type": ftype, "feature_name": fname}
    raise ValueError(f"unknown segment_type {segment_type!r}")


def _build_locator(
    segment_type: str,
    *,
    baclm_dir: Path,
    sample_gff: dict[str, str],
    boundary_tol: int,
    baclm_suffix: str,
    include_convergent: bool,
    unit_types: set[str] | None,
) -> SegmentLocator:
    """Construct the per-type :class:`SegmentLocator`, bound to its stores / GFF at build time."""
    baclm_dir = Path(baclm_dir)
    if segment_type == "igr":
        return IgrLocator(
            baclm_dir=baclm_dir, sample_gff=sample_gff, boundary_tol=boundary_tol, baclm_suffix=baclm_suffix
        )
    if segment_type == "upstream":
        return UpstreamLocator(
            baclm_dir=baclm_dir, sample_gff=sample_gff, boundary_tol=boundary_tol,
            include_convergent=include_convergent, baclm_suffix=baclm_suffix,
        )
    if segment_type == "unit":
        return UnitLocator(
            baclm_dir=baclm_dir, unit_types=frozenset(unit_types) if unit_types else None, baclm_suffix=baclm_suffix
        )
    raise ValueError(f"unknown segment_type {segment_type!r}")


def rank_segments(
    locator: SegmentLocator,
    sweep_ids: list[str],
    label_map: dict[str, int],
    *,
    id_column: str,
    eval_ids: set[str] | None = None,
    min_prevalence: float = 0.0,
    max_prevalence: float = 1.0,
    feature: str = "embedding",
    impute_absent_zero: bool = False,
    n_folds: int = 5,
    seed: int = 1,
    n_jobs: int = 1,
    store_dtype: str = "float32",
) -> tuple[dict[str, dict], pd.DataFrame, list[str], int, str]:
    """Sweep → (presence transform) → per-segment LR fit — the type-agnostic core of every screen.

    Parameters
    ----------
    locator
        The per-type :class:`SegmentLocator` bound to its stores / GFF.
    sweep_ids
        The genomes to sweep — the fit (train) genomes plus the held-out ``eval_ids``.
    label_map
        ``Sample -> 0/1`` AST label for every swept genome.
    id_column
        The segment-id column name (``igr_pair`` / ``upstream_gene`` / ``unit``), for the prevalence table.
    eval_ids
        The held-out split ids — excluded from prevalence / core selection, scored in pass 2 for the
        held-out ``eval_auroc``.
    min_prevalence, max_prevalence
        The single-copy prevalence band ``(min, max]`` over the fit genomes.
    feature
        ``"embedding"`` (fit the vector) or ``"presence"`` (fit a single presence/absence one-hot).
    impute_absent_zero
        Fit over the full read universe (a 0-vector for absent genomes) rather than carriers only.
    n_folds, seed, n_jobs, store_dtype
        Cross-fit folds, fold-assignment seed, per-segment fit worker processes, and design-matrix precision.

    Returns
    -------
    fitted
        ``{segment_id: fit_result}`` from :func:`...fit_lr.fit_per_segment` (single-class segments dropped).
    prevalence
        The ``[id_column, n_single_copy, prevalence]`` table over every single-copy segment.
    read_ids
        The impute universe — the fit ∪ eval genomes read in the sweep.
    n_core
        The number of in-band (core) segments actually fitted over.
    impute_mode
        The resolved absence-mode tag (``carrier_only`` / ``imputed_zero`` / ``presence``) for the table.
    """
    matrices, prevalence, read_ids = collect_segment_matrices(
        locator, sweep_ids, eval_ids=eval_ids, min_prevalence=min_prevalence,
        max_prevalence=max_prevalence, store_dtype=store_dtype, id_column=id_column,
    )
    n_core = len(matrices)
    presence = feature == "presence"
    impute_mode = "presence" if presence else ("imputed_zero" if impute_absent_zero else "carrier_only")
    if presence:
        # Replace each segment's embedding block with a ones-column; zero-imputing over the read universe
        # then makes the full design a 1/0 presence indicator (carrier=1, absent=0) — the pure one-hot LR.
        matrices = {k: (ids, np.ones((len(ids), 1), dtype=np.float32)) for k, (ids, _v) in matrices.items()}
        impute_absent_zero = True
    if not matrices:
        logger.warning("rank_segments[%s]: 0 core segments in band — nothing to fit", id_column)
        return {}, prevalence, read_ids, n_core, impute_mode
    fitted = fit_per_segment(
        matrices, label_map, n_folds=n_folds, seed=seed, n_jobs=n_jobs,
        all_ids=read_ids, impute_absent_zero=impute_absent_zero, eval_ids=eval_ids,
    )
    return fitted, prevalence, read_ids, n_core, impute_mode


def _write_prevalence(segment_type: str, prevalence: pd.DataFrame, out_path: Path) -> None:
    """Write the per-type prevalence table (``unit`` enriches with feature_type/feature_name + n_present)."""
    if segment_type == "unit":
        keys = prevalence["unit"].astype(str)
        out = prevalence.rename(columns={"n_single_copy": "n_present"}).copy()
        out.insert(1, "feature_type", [k.partition(":")[0] for k in keys])
        out.insert(2, "feature_name", [k.partition(":")[2] for k in keys])
        out = out[["unit", "feature_type", "feature_name", "n_present", "prevalence"]]
        out.to_csv(out_path, index=False)
    else:
        prevalence.to_csv(out_path, index=False)


def _write_ranking_table(
    segment_type: str,
    fitted: dict[str, dict],
    prevalence: pd.DataFrame,
    *,
    drug: str,
    auroc_col: str,
    filtered: set[str],
    impute_mode: str,
    out_path: Path,
) -> None:
    """Write the wide per-segment×drug ranking table (legacy per-type schema; igr now carries the eval cols).

    One row per fitted segment, ranked by out-of-fold train AUROC. The column layout is uniform across the
    three types — ``[id_column, *extra_id_columns, prevalence, <auroc_col>, eval_auroc_<drug>, n_train,
    n_pos, n_eval, n_eval_pos, kept_filtered, impute_mode]`` — reproducing the upstream/unit schema and
    upgrading igr to match (the A1 fix).
    """
    spec = SEGMENT_TYPES[segment_type]
    id_col = spec.id_column
    prev_map = dict(zip(prevalence[id_col], prevalence["prevalence"], strict=False))
    columns = [
        id_col, *spec.extra_id_columns, "prevalence", auroc_col, f"eval_auroc_{drug}",
        "n_train", "n_pos", "n_eval", "n_eval_pos", "kept_filtered", "impute_mode",
    ]
    rows = [
        {
            id_col: seg_id,
            **_extra_id_values(segment_type, seg_id),
            "prevalence": prev_map.get(seg_id, float("nan")),
            auroc_col: f["auroc"],
            f"eval_auroc_{drug}": f.get("eval_auroc", float("nan")),
            "n_train": f["n_train"],
            "n_pos": f["n_pos"],
            "n_eval": f.get("n_eval", 0),
            "n_eval_pos": f.get("n_eval_pos", 0),
            "kept_filtered": seg_id in filtered,
            "impute_mode": impute_mode,
        }
        for seg_id, f in sorted(fitted.items(), key=lambda kv: kv[1]["auroc"], reverse=True)
    ]
    pd.DataFrame(rows, columns=columns).to_csv(out_path, index=False)
    logger.info("wrote %s ranking (%d segments) to %s", segment_type, len(rows), out_path)


def run(
    segment_type: str,
    *,
    split_table: Path,
    drug: str,
    out_dir: Path,
    baclm_dir: Path,
    input_csv: Path | None = None,
    min_prevalence: float = 0.0,
    max_prevalence: float = 1.0,
    auroc_filter: float = 0.8,
    n_folds: int = 5,
    seed: int = 1,
    n_jobs: int = 1,
    max_train_genomes: int | None = None,
    sample_seed: int = 1,
    boundary_tol: int = 3,
    baclm_suffix: str = "_baclm_embeddings.pt",
    store_dtype: str = "float32",
    impute_absent_zero: bool = False,
    feature: str = "embedding",
    include_convergent: bool = False,
    unit_types: set[str] | None = None,
) -> dict:
    """Rank every segment of ``segment_type`` for one drug; write the ranking + prevalence tables + summary.

    Fits each segment's LR on the deployed ``train`` split, selects by out-of-fold train AUROC, and scores it
    on the deployed ``holdout`` (both read from ``split_table`` via
    :func:`bacpredict.engine.splits.load_splits.load_splits`) — so every rung reports a real held-out
    ``eval_auroc_<drug>`` on the same holdout as the coding screen and the ladder. The ``validate`` split is
    reserved for the trainer / concat operating-point and is not part of this screen's fit.

    Parameters
    ----------
    segment_type
        One of ``igr`` / ``upstream`` / ``unit`` (see :data:`SEGMENT_TYPES`).
    split_table
        The materialized ``<drug>_split.csv`` (``Sample, ast_label, split``).
    drug
        Drug name — used only to name the AUROC columns (``lr_auroc_<drug>`` / ``eval_auroc_<drug>``) and the
        output table; the split table is already drug-specific.
    out_dir
        Output directory for the ranking table, prevalence table, and build summary.
    baclm_dir
        Directory of ``{Sample}_baclm_embeddings.pt`` stores (the re-embed store for ``unit``).
    input_csv
        The ``Sample -> sr_gff_file`` map (required for ``igr`` / ``upstream``; unused for ``unit``).
    min_prevalence, max_prevalence
        The single-copy prevalence band ``(min, max]`` over the fit genomes.
    auroc_filter
        Mark segments with out-of-fold train AUROC above this as ``kept_filtered``.
    n_folds, seed, n_jobs
        Cross-fit folds, fold-assignment seed, and per-segment fit worker processes.
    max_train_genomes, sample_seed
        Fit on a random, class-balanced subsample of this many train genomes (``None`` = all), with this seed.
    boundary_tol
        Max bp gap between a region boundary and its abutting named flank / anchor gene (igr / upstream).
    baclm_suffix
        The ``{Sample}<suffix>`` store-file suffix.
    store_dtype
        Design-matrix storage precision (``float32`` default; ``float16`` halves the full-cohort footprint).
    impute_absent_zero
        Fit each segment over the full read universe, zero-imputing absent genomes (the zero-imputed block
        the concat consumes), rather than carriers only.
    feature
        ``"embedding"`` (default) or ``"presence"`` (a single presence/absence one-hot; the lineage control).
    include_convergent
        (upstream) also emit ``between:<a>→<b>`` for convergent regions with no 5′ anchor.
    unit_types
        (unit) restrict to this set of lower-cased feature types (e.g. ``{"rrna"}``); ``None`` = all bodies.

    Returns
    -------
    dict
        The build summary (also written as ``per_<segment_type>_lr_build_summary.json``).
    """
    if segment_type not in SEGMENT_TYPES:
        raise ValueError(f"segment_type must be one of {sorted(SEGMENT_TYPES)}; got {segment_type!r}")
    spec = SEGMENT_TYPES[segment_type]

    label_map, train_ids, _validate_ids, holdout_ids = load_splits(split_table)
    fit_train_ids = subsample_balanced(train_ids, label_map, max_n=max_train_genomes, seed=sample_seed)
    eval_ids = set(holdout_ids)
    sweep_ids = [*fit_train_ids, *holdout_ids]

    sample_gff: dict[str, str] = {}
    if spec.needs_gff:
        if input_csv is None:
            raise ValueError(f"segment_type={segment_type!r} needs input_csv (the Sample→sr_gff_file map).")
        inp = pd.read_csv(input_csv, usecols=["Sample", "sr_gff_file"])
        sample_gff = dict(zip(inp["Sample"].astype(str), inp["sr_gff_file"].astype(str), strict=True))

    locator = _build_locator(
        segment_type, baclm_dir=baclm_dir, sample_gff=sample_gff, boundary_tol=boundary_tol,
        baclm_suffix=baclm_suffix, include_convergent=include_convergent, unit_types=unit_types,
    )

    fitted, prevalence, read_ids, n_core, impute_mode = rank_segments(
        locator, sweep_ids, label_map, id_column=spec.id_column, eval_ids=eval_ids,
        min_prevalence=min_prevalence, max_prevalence=max_prevalence, feature=feature,
        impute_absent_zero=impute_absent_zero, n_folds=n_folds, seed=seed, n_jobs=n_jobs, store_dtype=store_dtype,
    )
    filtered = {k for k, f in fitted.items() if f["auroc"] > auroc_filter}
    logger.info("filter (AUROC > %.2f): %d of %d fitted %s kept", auroc_filter, len(filtered), len(fitted), segment_type)

    presence = feature == "presence"
    auroc_col = f"presence_lr_auroc_{drug}" if presence else f"lr_auroc_{drug}"
    table_name = f"per_{segment_type}_presence_lr_{drug}.csv" if presence else f"per_{segment_type}_lr_{drug}.csv"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_prevalence(segment_type, prevalence, out_dir / f"{segment_type}_prevalence.csv")
    _write_ranking_table(
        segment_type, fitted, prevalence, drug=drug, auroc_col=auroc_col,
        filtered=filtered, impute_mode=impute_mode, out_path=out_dir / table_name,
    )

    best_id, best_fit = max(fitted.items(), key=lambda kv: kv[1]["auroc"], default=(None, None))
    summary = {
        "analysis": "per_segment_lr",
        "segment_type": segment_type,
        "drug": drug,
        "split_table": str(split_table),
        "feature": feature,
        "impute_absent_zero": bool(impute_mode != "carrier_only"),  # effective: presence forces impute
        "impute_mode": impute_mode,
        "n_train_fit": len(fit_train_ids),
        "n_read_genomes": len(read_ids),
        "n_holdout": len(holdout_ids),
        "min_prevalence": min_prevalence,
        "max_prevalence": max_prevalence,
        "boundary_tol": boundary_tol if spec.needs_gff else None,
        "include_convergent": include_convergent if segment_type == "upstream" else None,
        "unit_types": sorted(unit_types) if (segment_type == "unit" and unit_types) else None,
        "n_prevalent": int(len(prevalence)),
        "n_core": n_core,
        "n_fitted": len(fitted),
        "auroc_filter": auroc_filter,
        "n_filtered": len(filtered),
        "best_segment": best_id,
        "best_auroc": best_fit["auroc"] if best_fit else None,
    }
    (out_dir / f"per_{segment_type}_lr_build_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "per_segment_lr[%s] %s: %s", segment_type, drug,
        json.dumps({k: summary[k] for k in ("n_core", "n_fitted", "n_filtered", "best_segment", "best_auroc")}),
    )
    return summary
