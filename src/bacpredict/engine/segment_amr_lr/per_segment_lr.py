"""Rank AMR segments of one type by a per-segment logistic regression — the unified screen.

ONE driver over a per-type :class:`bacpredict.engine.embedding.segment_locator.SegmentLocator`, covering
all four segment types:

* ``igr`` — intergenic regions keyed by flank pair ``a→b`` (baclm non-coding);
* ``upstream`` — 5′-anchored ``upstream:<gene>`` regions (baclm non-coding);
* ``unit`` — named non-CDS bodies ``<type>:<name>`` (baclm re-embed);
* ``coding`` — a gene's ESM-C or baclm coding vector keyed ``gene_name`` (the protein screen).

Each is the SAME operation — locate every segment of a type in each genome, keep the single-copy (for units,
mean-pooled) occurrences in a prevalence band, fit one LR per segment, rank by out-of-fold train AUROC. They
differ only in the per-type keying / GFF / dedup, which lives entirely in the locator and the uniform
two-pass sweep (:func:`bacpredict.engine.embedding.segment_embedding_extractor.collect_segment_matrices`); the
fit is the one shared engine (:func:`bacpredict.engine.segment_amr_lr.fit_lr.fit_per_segment`). This driver
replaced the four copy-forked ``gene_lr.build_*`` ranking stores.

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

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bacpredict.engine.config import store_paths
from bacpredict.engine.embedding.segment_embedding_extractor import (
    collect_core_subset,
    collect_segment_matrices,
    sweep_core_prevalence,
)
from bacpredict.engine.embedding.segment_locator import (
    IgrLocator,
    ProteinLocator,
    SegmentLocator,
    UnitLocator,
    UpstreamLocator,
)
from bacpredict.engine.segment_amr_lr.fit_lr import fit_per_segment
from bacpredict.engine.splits.load_splits import load_splits
from bacpredict.engine.splits.subsample import subsample_balanced

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@dataclass(frozen=True)
class SegmentTypeSpec:
    """Per-type configuration for the otherwise type-agnostic ranking driver.

    Attributes
    ----------
    id_column
        The segment-id column name in the **prevalence** table + the ``collect_segment_matrices`` sweep
        (``igr_pair`` / ``upstream_gene`` / ``unit`` / ``gene``). For the non-coding types it is also the
        ranking-table's first column; ``coding`` keys its ranking table ``gene_name`` instead (a legacy quirk
        preserved by the coding-specific writer).
    extra_id_columns
        Extra identity columns split out of the segment id (``left_gene``/``right_gene`` for igr, ``gene``
        for upstream, ``feature_type``/``feature_name`` for unit). Unused for ``coding`` (its ``annotation``
        column is a parquet lookup, not an id split — see :func:`_write_coding_table`).
    needs_gff
        Whether the locator needs the per-sample Bakta GFF map (igr / upstream) or self-identifies
        (unit / coding).
    is_coding
        The coding **protein** type — a dual (esm | baclm) store keyed ``gene_name``, with a parquet
        annotation join and an optional per-protein panel side-output, and **no presence mode**.
    offers_presence
        Whether ``feature="presence"`` (the one-hot lineage control) is valid for this type (all but coding).
    """

    id_column: str
    extra_id_columns: tuple[str, ...]
    needs_gff: bool
    is_coding: bool = False
    offers_presence: bool = True


SEGMENT_TYPES: dict[str, SegmentTypeSpec] = {
    "igr": SegmentTypeSpec("igr_pair", ("left_gene", "right_gene"), needs_gff=True),
    "upstream": SegmentTypeSpec("upstream_gene", ("gene",), needs_gff=True),
    "unit": SegmentTypeSpec("unit", ("feature_type", "feature_name"), needs_gff=False),
    "coding": SegmentTypeSpec("gene", (), needs_gff=False, is_coding=True, offers_presence=False),
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
    baclm_dir: Path | None,
    sample_gff: dict[str, str],
    boundary_tol: int,
    baclm_suffix: str,
    include_convergent: bool,
    unit_types: set[str] | None,
    embed_dir: Path | None = None,
    parquet_dir: Path | None = None,
    store_kind: str = "esm",
) -> SegmentLocator:
    """Construct the per-type :class:`SegmentLocator`, bound to its stores / GFF at build time."""
    if segment_type == "coding":
        return ProteinLocator(embed_dir=Path(embed_dir), parquet_dir=Path(parquet_dir), store_kind=store_kind)
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
    segment_batch_size: int | None = None,
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
    segment_batch_size
        If set, materialise + fit the core segments in memory-bounded batches of this many (one genome scan
        per batch, only that batch's matrices held at once) instead of all at once. Per-segment fits are
        independent, so the ``fitted`` result is **identical** to the single-shot path — only peak RAM changes.
        ``None`` (default) keeps the classic single-shot behaviour.

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
    presence = feature == "presence"
    impute_mode = "presence" if presence else ("imputed_zero" if impute_absent_zero else "carrier_only")

    def _fit_batch(matrices: dict[str, tuple[list[str], np.ndarray]], read_ids: list[str]) -> dict[str, dict]:
        """Presence-transform (if any) then fit one materialised batch of segment matrices."""
        m, impute = matrices, impute_absent_zero
        if presence:
            # Replace each segment's embedding block with a ones-column; zero-imputing over the read universe
            # then makes the full design a 1/0 presence indicator (carrier=1, absent=0) — the pure one-hot LR.
            m = {k: (ids, np.ones((len(ids), 1), dtype=np.float32)) for k, (ids, _v) in m.items()}
            impute = True
        if not m:
            return {}
        return fit_per_segment(
            m, label_map, n_folds=n_folds, seed=seed, n_jobs=n_jobs,
            all_ids=read_ids, impute_absent_zero=impute, eval_ids=eval_ids,
        )

    if segment_batch_size is None:
        # Single-shot: the whole core set is materialised at once (the classic path).
        matrices, prevalence, read_ids = collect_segment_matrices(
            locator, sweep_ids, eval_ids=eval_ids, min_prevalence=min_prevalence,
            max_prevalence=max_prevalence, store_dtype=store_dtype, id_column=id_column,
        )
        n_core = len(matrices)
        if not matrices:
            logger.warning("rank_segments[%s]: 0 core segments in band — nothing to fit", id_column)
            return {}, prevalence, read_ids, n_core, impute_mode
        return _fit_batch(matrices, read_ids), prevalence, read_ids, n_core, impute_mode

    # Batched: pass 1 once (core + prevalence), then materialise + fit the core set in memory-bounded slices
    # of `segment_batch_size` segments — one genome scan per batch, only that batch's matrices held at once.
    core, prevalence = sweep_core_prevalence(
        locator, sweep_ids, eval_ids=eval_ids, min_prevalence=min_prevalence,
        max_prevalence=max_prevalence, id_column=id_column,
    )
    n_core = len(core)
    core_sorted = sorted(core)
    if not core_sorted:
        _m, read_ids = collect_core_subset(locator, sweep_ids, set(), store_dtype=store_dtype)
        logger.warning("rank_segments[%s]: 0 core segments in band — nothing to fit", id_column)
        return {}, prevalence, read_ids, 0, impute_mode
    n_batches = math.ceil(len(core_sorted) / segment_batch_size)
    fitted: dict[str, dict] = {}
    read_ids: list[str] = []
    for bi in range(n_batches):
        subset = set(core_sorted[bi * segment_batch_size : (bi + 1) * segment_batch_size])
        matrices, batch_read_ids = collect_core_subset(locator, sweep_ids, subset, store_dtype=store_dtype)
        if not read_ids:  # identical across batches; capture once as the impute universe
            read_ids = batch_read_ids
        fitted.update(_fit_batch(matrices, read_ids))
        logger.info("segment batch %d/%d [%s]: fit %d segments (%d/%d core cumulative)",
                    bi + 1, n_batches, id_column, len(matrices), len(fitted), n_core)
        del matrices
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


def _coding_annotation(fit_ids: list[str], parquet_dir: Path) -> dict[str, str]:
    """``gene_name -> product`` from the fit genomes' parquet (the ``annotation`` column of the coding table).

    The uniform sweep keys only by ``gene_name``, so the human-readable product is recovered here in one
    light parquet-only pass over the fit genomes (first product seen per gene wins) — the ``annotation`` map
    the coding screen has always carried, via the parquet record extractor in the locator module.
    """
    from bacpredict.engine.embedding.segment_locator import _genome_segment_records

    annotation: dict[str, str] = {}
    for sid in fit_ids:
        for r in _genome_segment_records(str(sid), Path(parquet_dir)):
            gene, product = r.get("gene_name"), r.get("protein_name")
            if gene and product and gene not in annotation:
                annotation[gene] = product
    return annotation


def _write_coding_table(
    fitted: dict[str, dict],
    prevalence: pd.DataFrame,
    annotation: dict[str, str],
    *,
    drug: str,
    filtered: set[str],
    impute_mode: str,
    out_path: Path,
) -> None:
    """Write the coding ``per_gene_lr_<drug>.csv`` (keyed ``gene_name`` + a parquet ``annotation`` column).

    The coding ranking table is the one type whose id column (``gene_name``) differs from its prevalence key
    (``gene``) and which carries a free-text ``annotation`` (product) column, so it has a bespoke writer
    rather than the generic one — the ``per_gene_lr_<drug>.csv`` schema the ladder + catalogue plots read.
    """
    prev_by_gene = dict(zip(prevalence["gene"], prevalence["prevalence"], strict=False))
    columns = [
        "gene_name", "annotation", "prevalence", f"lr_auroc_{drug}", f"eval_auroc_{drug}",
        "n_train", "n_pos", "n_eval", "n_eval_pos", "kept_filtered", "impute_mode",
    ]
    rows = [
        {
            "gene_name": gene,
            "annotation": annotation.get(gene, ""),
            "prevalence": prev_by_gene.get(gene, float("nan")),
            f"lr_auroc_{drug}": f["auroc"],
            f"eval_auroc_{drug}": f.get("eval_auroc", float("nan")),
            "n_train": f["n_train"],
            "n_pos": f["n_pos"],
            "n_eval": f.get("n_eval", 0),
            "n_eval_pos": f.get("n_eval_pos", 0),
            "kept_filtered": gene in filtered,
            "impute_mode": impute_mode,
        }
        for gene, f in sorted(fitted.items(), key=lambda kv: kv[1]["auroc"], reverse=True)
    ]
    pd.DataFrame(rows, columns=columns).to_csv(out_path, index=False)
    logger.info("wrote coding ranking (%d genes, impute_mode=%s) to %s", len(rows), impute_mode, out_path)


def run(
    segment_type: str,
    *,
    split_table: Path,
    drug: str,
    out_dir: Path,
    baclm_dir: Path | None = None,
    input_csv: Path | None = None,
    embed_dir: Path | None = None,
    parquet_dir: Path | None = None,
    store_kind: str = "esm",
    write_panels: bool = False,
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
    segment_batch_size: int | None = None,
) -> dict:
    """Rank every segment of ``segment_type`` for one drug; write the ranking + prevalence tables + summary.

    Fits each segment's LR on the deployed ``train`` split, selects by out-of-fold train AUROC, and scores it
    on the deployed ``holdout`` (both read from ``split_table`` via
    :func:`bacpredict.engine.splits.load_splits.load_splits`) — so every rung reports a real held-out
    ``eval_auroc_<drug>`` on the same holdout as every other rung and the ladder. The ``validate`` split is
    reserved for the trainer / concat operating-point and is not part of this screen's fit.

    The three non-coding types (``igr`` / ``upstream`` / ``unit``) read the baclm store in ``baclm_dir``; the
    coding **protein** type reads the esm | baclm per-protein store in ``embed_dir`` + the parquet gene lists
    in ``parquet_dir`` (``store_kind``), joins a free-text ``annotation`` (product), keeps the deny-listed
    ``per_gene_lr_<drug>.csv`` / ``gene_prevalence.csv`` filenames, has **no presence mode**, and optionally
    writes the per-protein panel store (``write_panels``, the FT attention-head channel).

    Parameters
    ----------
    segment_type
        One of ``igr`` / ``upstream`` / ``unit`` / ``coding`` (see :data:`SEGMENT_TYPES`).
    split_table
        The materialized ``<drug>_split.csv`` (``Sample, ast_label, split``).
    drug
        Drug name — used only to name the AUROC columns (``lr_auroc_<drug>`` / ``eval_auroc_<drug>``) and the
        output table; the split table is already drug-specific.
    out_dir
        Output directory for the ranking table, prevalence table, and build summary.
    baclm_dir
        Directory of ``{Sample}_baclm_embeddings.pt`` stores (required for igr / upstream / unit).
    input_csv
        The ``Sample -> sr_gff_file`` map (required for ``igr`` / ``upstream``).
    embed_dir, parquet_dir, store_kind
        (coding) the per-protein embedding store dir, the ``{Sample}_protein_sequences.parquet`` dir, and the
        store kind (``esm`` | ``baclm``).
    write_panels
        (coding) also write the per-protein ``filtered/`` + ``unfiltered/`` panel store (heavy; off by default).
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
        The ``{Sample}<suffix>`` store-file suffix (non-coding).
    store_dtype
        Design-matrix storage precision (``float32`` default; ``float16`` halves the full-cohort footprint).
    impute_absent_zero
        Fit each segment over the full read universe, zero-imputing absent genomes (the zero-imputed block
        the concat consumes), rather than carriers only.
    feature
        ``"embedding"`` (default) or ``"presence"`` (a one-hot lineage control; **invalid for coding**).
    include_convergent
        (upstream) also emit ``between:<a>→<b>`` for convergent regions with no 5′ anchor.
    unit_types
        (unit) restrict to this set of lower-cased feature types (e.g. ``{"rrna"}``); ``None`` = all bodies.

    Returns
    -------
    dict
        The build summary (also written as ``per_<segment_type>_lr_build_summary.json``; ``per_gene`` for
        coding).
    """
    if segment_type not in SEGMENT_TYPES:
        raise ValueError(f"segment_type must be one of {sorted(SEGMENT_TYPES)}; got {segment_type!r}")
    spec = SEGMENT_TYPES[segment_type]
    if feature == "presence" and not spec.offers_presence:
        raise ValueError(f"segment_type={segment_type!r} does not support feature='presence'.")
    if spec.is_coding and (embed_dir is None or parquet_dir is None):
        raise ValueError("segment_type='coding' needs embed_dir + parquet_dir (the store + parquet dirs).")
    if not spec.is_coding and baclm_dir is None:
        raise ValueError(f"segment_type={segment_type!r} needs baclm_dir (the *_baclm_embeddings.pt dir).")

    label_map, train_ids, validate_ids, holdout_ids = load_splits(split_table)
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
        embed_dir=embed_dir, parquet_dir=parquet_dir, store_kind=store_kind,
    )

    fitted, prevalence, read_ids, n_core, impute_mode = rank_segments(
        locator, sweep_ids, label_map, id_column=spec.id_column, eval_ids=eval_ids,
        min_prevalence=min_prevalence, max_prevalence=max_prevalence, feature=feature,
        impute_absent_zero=impute_absent_zero, n_folds=n_folds, seed=seed, n_jobs=n_jobs, store_dtype=store_dtype,
        segment_batch_size=segment_batch_size,
    )
    filtered = {k for k, f in fitted.items() if f["auroc"] > auroc_filter}
    logger.info("filter (AUROC > %.2f): %d of %d fitted %s kept", auroc_filter, len(filtered), len(fitted), segment_type)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    if spec.is_coding:
        # Coding keeps the deny-listed legacy filenames (per_gene_lr_<drug> / gene_prevalence) + the parquet
        # annotation join; the sweep's prevalence is already the [gene, n_single_copy, prevalence] schema.
        annotation = _coding_annotation(fit_train_ids, Path(parquet_dir))
        prevalence.to_csv(out_dir / "gene_prevalence.csv", index=False)
        _write_coding_table(fitted, prevalence, annotation, drug=drug, filtered=filtered,
                            impute_mode=impute_mode, out_path=out_dir / f"per_gene_lr_{drug}.csv")
        if write_panels:
            from bacpredict.engine.segment_amr_lr.panel_store import build_panels

            filtered_dir, unfiltered_dir = out_dir / "filtered", out_dir / "unfiltered"
            for d in (filtered_dir, unfiltered_dir):
                d.mkdir(parents=True, exist_ok=True)
            n_written = build_panels(
                [*train_ids, *validate_ids, *holdout_ids], fitted, filtered, Path(embed_dir), Path(parquet_dir),
                train_set=set(fit_train_ids), filtered_dir=filtered_dir, unfiltered_dir=unfiltered_dir,
                store_kind=store_kind,
            )
        table_stem = "per_gene"
    else:
        presence = feature == "presence"
        auroc_col = f"presence_lr_auroc_{drug}" if presence else f"lr_auroc_{drug}"
        table_name = f"per_{segment_type}_presence_lr_{drug}.csv" if presence else f"per_{segment_type}_lr_{drug}.csv"
        _write_prevalence(segment_type, prevalence, out_dir / f"{segment_type}_prevalence.csv")
        _write_ranking_table(segment_type, fitted, prevalence, drug=drug, auroc_col=auroc_col,
                             filtered=filtered, impute_mode=impute_mode, out_path=out_dir / table_name)
        table_stem = f"per_{segment_type}"

    best_id, best_fit = max(fitted.items(), key=lambda kv: kv[1]["auroc"], default=(None, None))
    summary = {
        "analysis": "per_segment_lr",
        "segment_type": segment_type,
        "drug": drug,
        "split_table": str(split_table),
        "feature": feature,
        "impute_absent_zero": bool(impute_mode != "carrier_only"),  # effective: presence forces impute
        "impute_mode": impute_mode,
        "embedding_store": store_kind if spec.is_coding else None,
        "wrote_panels": write_panels if spec.is_coding else None,
        "n_genomes_written": n_written,
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
    (out_dir / f"{table_stem}_lr_build_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "per_segment_lr[%s] %s: %s", segment_type, drug,
        json.dumps({k: summary[k] for k in ("n_core", "n_fitted", "n_filtered", "best_segment", "best_auroc")}),
    )
    return summary


def main() -> None:
    """CLI: rank one drug's segments of one type via the deployed ``<drug>_split.csv`` (fit-train / eval-holdout).

    Store paths default off ``--species`` (:func:`bacpredict.engine.config.store_paths`) and are individually
    overridable; the coding ``--embed-dir`` defaults to the ESM or baclm store per ``--store-kind``. The
    non-coding types need ``--baclm-dir`` (pass the ``baclm_reembed`` store for the ladder's imputed rankings)
    and igr/upstream additionally need ``--input-csv`` (the ``Sample -> sr_gff_file`` map).
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--segment-type", required=True, choices=sorted(SEGMENT_TYPES),
                   help="coding (protein) / igr / upstream / unit.")
    p.add_argument("--split-table", type=Path, required=True,
                   help="Deployed per-drug <drug>_split.csv (Sample, ast_label, split) from splits.load_splits.")
    p.add_argument("--drug", required=True, help="Drug name — names the AUROC columns + output table.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--species", choices=["tb", "kp"], default=None, help="Resolve store-path defaults for this organism.")
    p.add_argument("--baclm-dir", type=Path, default=None,
                   help="baclm store for igr/upstream/unit (default: <species>.baclm_dir; pass baclm_reembed for the ladder).")
    p.add_argument("--input-csv", type=Path, default=None, help="Sample->sr_gff_file map (required for igr/upstream).")
    p.add_argument("--embed-dir", type=Path, default=None, help="coding per-protein store dir (default: <species> esm|baclm).")
    p.add_argument("--parquet-dir", type=Path, default=None, help="coding protein-sequence parquet dir.")
    p.add_argument("--store-kind", choices=["esm", "baclm"], default="esm", help="coding embedding store kind.")
    p.add_argument("--feature", choices=["embedding", "presence"], default="embedding",
                   help="embedding (fit the vector) or presence (a one-hot lineage control; invalid for coding).")
    p.add_argument("--impute-absent-zero", action="store_true",
                   help="Fit over the full read universe, zero-imputing absent genomes (selection = usage).")
    p.add_argument("--write-panels", action="store_true", help="(coding) also write the per-protein panel store.")
    p.add_argument("--include-convergent", action="store_true",
                   help="(upstream) also emit between:<a>->'<b>' convergent regions with no 5' anchor.")
    p.add_argument("--unit-types", nargs="*", default=None, help="(unit) restrict to these feature types, e.g. rrna.")
    p.add_argument("--min-prevalence", type=float, default=0.0)
    p.add_argument("--max-prevalence", type=float, default=1.0)
    p.add_argument("--auroc-filter", type=float, default=0.8)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--max-train-genomes", type=int, default=None)
    p.add_argument("--sample-seed", type=int, default=1)
    p.add_argument("--boundary-tol", type=int, default=3)
    p.add_argument("--baclm-suffix", default="_baclm_embeddings.pt")
    p.add_argument("--store-dtype", default="float32")
    p.add_argument("--segment-batch-size", type=int, default=None,
                   help="Materialise + fit core segments in memory-bounded batches of this many (default: all "
                        "at once). Bounds peak RAM for clonal cohorts (thousands of dense near-ubiquitous core "
                        "matrices) at the cost of one genome scan per batch; results are identical.")
    args = p.parse_args()

    sp = store_paths(args.species) if args.species else None
    baclm_dir = args.baclm_dir or (sp.baclm_dir if sp else None)
    input_csv = args.input_csv or (sp.input_csv if sp else None)
    parquet_dir = args.parquet_dir or (sp.parquet_dir if sp else None)
    if args.embed_dir is not None:
        embed_dir = args.embed_dir
    elif sp is not None:
        embed_dir = sp.baclm_dir if args.store_kind == "baclm" else sp.esm_dir
    else:
        embed_dir = None

    run(
        args.segment_type, split_table=args.split_table, drug=args.drug, out_dir=args.out_dir,
        baclm_dir=baclm_dir, input_csv=input_csv, embed_dir=embed_dir, parquet_dir=parquet_dir,
        store_kind=args.store_kind, write_panels=args.write_panels,
        min_prevalence=args.min_prevalence, max_prevalence=args.max_prevalence, auroc_filter=args.auroc_filter,
        n_folds=args.n_folds, seed=args.seed, n_jobs=args.n_jobs, max_train_genomes=args.max_train_genomes,
        sample_seed=args.sample_seed, boundary_tol=args.boundary_tol, baclm_suffix=args.baclm_suffix,
        store_dtype=args.store_dtype, impute_absent_zero=args.impute_absent_zero, feature=args.feature,
        include_convergent=args.include_convergent,
        unit_types=set(args.unit_types) if args.unit_types else None,
        segment_batch_size=args.segment_batch_size,
    )


if __name__ == "__main__":
    main()
