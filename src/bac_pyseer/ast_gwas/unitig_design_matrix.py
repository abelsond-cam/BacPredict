"""Significant unitigs → a sparse genomes × unitigs presence/absence design matrix.

The GWAS gives a list of hit unitig *sequences*; a logistic regression needs a matrix. Getting
there means one pass over the organism's ``unitigs.pyseer.gz`` (tens of GB) to pull out just the
hit rows, which :func:`~bac_pyseer.kleb_iso_source.unitig_placement.extract_hit_submatrix` already
does as a ``pigz | awk`` streaming hash-join and **caches forever** — so the big matrix is read once
per drug and never again, and re-running this module is cheap.

Two things are deliberate:

* **By default the matrix spans every genome in the split table — train, validate *and* holdout.**
  That is not a leak. Unitig *presence* is an unsupervised property of an assembly; withholding it
  from the holdout would mean having no features to score with. What must never touch the holdout is
  label information, and that is enforced upstream in
  :mod:`bac_pyseer.ast_gwas.build_ast_phenotype`. ``--splits train,validate`` narrows it for the
  train+validate-vocabulary arm, where the unitig matrix has no holdout carriers to find and those
  rows are scanned from sequence instead — see :mod:`bac_pyseer.ast_gwas.unitig_kmer_presence`.
* **Genomes carrying none of the hit unitigs become all-zero rows, not missing rows.** They are
  genuine negatives for every hit feature, exactly as a non-carrier is in the CARD/WHO catalogue
  one-hot this baseline is compared against. Dropping them would silently change the cohort.

``--dedupe-patterns`` keeps one representative per ``pattern_group`` (pyseer's perfect-LD block).
Hit counts are inflated by linkage — a niche-associated megaplasmid contributes thousands of
co-inherited unitigs carrying one signal — so the deduplicated matrix is the honest feature count,
and comparing the two fits tells you how much of the LR's performance is LD bookkeeping.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from bac_pyseer.kleb_iso_source.unitig_placement import extract_hit_submatrix
from bacpredict.engine.splits.load_splits import load_splits

logger = logging.getLogger(__name__)

VARIANT_COL = "variant"  # pyseer's id column; for unitigs the id IS the DNA sequence
# Per-hit statistics carried into id_map.tsv so a fitted coefficient can be traced back to its GWAS row.
_CARRY_COLS = ("var_explained_pct", "af", "beta", "lrt-pvalue", "pattern_group", "n_in_pattern", "direction")
SPLIT_SLICES = ("train", "validate", "holdout")


def read_hits(hits_tsv: Path, *, dedupe_patterns: bool = False, max_hits: int | None = None) -> pd.DataFrame:
    """Read a ``<drug>_hits_annotated.tsv`` → an id_map frame (``unitig_idx``, ``variant``, stats).

    Parameters
    ----------
    hits_tsv
        Output of ``pyseer_postprocess --feature-mode unitigs``, ranked by ``var_explained_pct``.
    dedupe_patterns
        Keep only the first hit of each ``pattern_group`` (one representative per perfect-LD block).
    max_hits
        Keep at most this many hits, taken in the file's existing rank order.

    Returns
    -------
    pandas.DataFrame
        One row per retained unitig, with a dense 0-based ``unitig_idx``.
    """
    hits = pd.read_csv(hits_tsv, sep="\t", low_memory=False)
    if VARIANT_COL not in hits.columns:
        raise SystemExit(f"{hits_tsv} has no '{VARIANT_COL}' column — is this a pyseer hits table?")
    hits = hits[hits[VARIANT_COL].notna()].copy()
    hits[VARIANT_COL] = hits[VARIANT_COL].astype(str).str.strip().str.upper()
    hits = hits[hits[VARIANT_COL].str.len() > 0]
    hits = hits.drop_duplicates(subset=VARIANT_COL, keep="first")

    if dedupe_patterns:
        if "pattern_group" not in hits.columns:
            raise SystemExit(f"--dedupe-patterns needs a 'pattern_group' column in {hits_tsv}")
        hits = hits.drop_duplicates(subset="pattern_group", keep="first")
    if max_hits is not None:
        hits = hits.head(max_hits)

    keep = [VARIANT_COL, *(c for c in _CARRY_COLS if c in hits.columns)]
    id_map = hits[keep].reset_index(drop=True)
    id_map.insert(0, "unitig_idx", np.arange(len(id_map), dtype=np.int64))
    id_map["unitig_len"] = id_map[VARIANT_COL].str.len()
    return id_map


def build_presence_matrix(
    submatrix_path: Path, id_map: pd.DataFrame, sample_ids: list[str]
) -> tuple[sparse.csr_matrix, set[str], int]:
    """Stream the cached hit sub-matrix → a ``(len(sample_ids), len(id_map))`` binary CSR.

    Each sub-matrix line is one unitig and its carriers, which is *column*-major for a genome ×
    unitig matrix — so the accumulation is CSC (``indptr`` from the per-unitig carrier counts) with
    a single conversion at the end, the pattern
    :mod:`bac_pyseer.kleb_iso_source.unitig_presence_model` uses. That avoids materialising either a
    COO triple or a dict-of-sets, roughly halving peak memory at ~10⁸ non-zeros.

    Columns are placed at their ``id_map`` position rather than in file order, so column *j* is
    always ``id_map`` row *j* — which is what lets a fitted coefficient be traced back to its GWAS
    row.

    Returns
    -------
    (scipy.sparse.csr_matrix, set of str, int)
        The presence matrix in ``sample_ids`` order; the hit sequences absent from the sub-matrix
        altogether (expected empty — a non-empty set means the hits table and the unitig matrix
        disagree); and the count of columns that are all-zero *within this cohort*, i.e. unitigs
        whose only carriers fall outside the split table. Those two failures look identical in the
        matrix but mean different things, so they are counted separately.
    """
    seq2idx = {row.variant: int(row.unitig_idx) for row in id_map.itertuples(index=False)}
    row_of = {s: i for i, s in enumerate(sample_ids)}
    n_cols = len(id_map)

    per_column: list[np.ndarray | None] = [None] * n_cols
    with submatrix_path.open() as fh:
        for line in fh:
            seq, sep, rest = line.partition(" | ")
            if not sep:
                continue
            col = seq2idx.get(seq)
            if col is None:
                continue
            rows = {row_of[s] for tok in rest.split() if (s := tok.rpartition(":")[0]) in row_of}
            # Sorted + de-duplicated: CSC wants ordered indices, and a sample can be listed twice
            # if a unitig was placed more than once.
            per_column[col] = np.fromiter(sorted(rows), dtype=np.int32, count=len(rows))

    counts = [0 if c is None else c.size for c in per_column]
    indptr = np.zeros(n_cols + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)
    present = [c for c in per_column if c is not None and c.size]
    indices = np.concatenate(present) if present else np.zeros(0, dtype=np.int32)
    matrix = sparse.csc_matrix(
        (np.ones(indices.size, dtype=np.int8), indices, indptr),
        shape=(len(sample_ids), n_cols),
        dtype=np.int8,
    ).tocsr()

    # A sequence missing from the sub-matrix is a join failure; a column that is present but empty
    # is a cohort fact (its only carriers fall outside the split table). The matrix alone cannot
    # tell those apart, so they are tracked separately during the parse.
    missing = {seq for seq, col in seq2idx.items() if per_column[col] is None}
    n_empty_columns = int(sum(1 for c in counts if c == 0))
    return matrix, missing, n_empty_columns


def check_holdout_coverage(
    matrix: sparse.csr_matrix, sample_ids: list[str], reference_ids: list[str], holdout_ids: list[str],
    *, min_ratio: float = 0.5, min_holdout_genomes: int = 30,
) -> dict[str, float]:
    """Assert the holdout rows carry hit unitigs at a rate comparable to the fitted rows'.

    The failure this exists to catch is silent and total. If the holdout rows are empty — a
    vocabulary built without those genomes, a scan that resolved no assembly, a sample-id join that
    matched nothing — the logistic regression still fits, still scores, and still reports a perfectly
    well-formed AUROC of about 0.5. Nothing downstream can tell "this genome carries no resistance
    unitig" apart from "this genome was never actually scored", because both are a row of zeros.

    A holdout mean far below the fitted mean is the one cheap signal that separates them, so it is
    checked here rather than left to be noticed as a disappointing result. The ratio is deliberately
    loose: a genuine out-of-vocabulary penalty *should* depress the holdout mean somewhat, and that
    penalty is part of what this experiment measures. It is a floor against nothing, not a test of
    the effect.

    Below ``min_holdout_genomes`` the ratio is not evidence and the assertion is skipped — a mean over
    a handful of genomes swings on one carrier. The skip is recorded as ``checked: False`` rather than
    reported as a pass, for the same reason the scanner's verification reports its shared-genome count:
    a gate that could not run must not look like a gate that ran. Every real drug clears this by an
    order of magnitude (the smallest Kp holdout is in the hundreds), so it only ever fires on fixtures.
    """
    row_of = {s: i for i, s in enumerate(sample_ids)}
    carriers = np.asarray(matrix.sum(axis=1)).ravel()
    ref_rows = [row_of[s] for s in reference_ids if s in row_of]
    hold_rows = [row_of[s] for s in holdout_ids if s in row_of]
    ref_mean = float(carriers[ref_rows].mean()) if ref_rows else 0.0
    hold_mean = float(carriers[hold_rows].mean()) if hold_rows else 0.0
    stats = {
        "reference_mean_carriers": ref_mean,
        "holdout_mean_carriers": hold_mean,
        "ratio": (hold_mean / ref_mean) if ref_mean else 0.0,
        "min_ratio": min_ratio,
        "n_holdout_all_zero": int((carriers[hold_rows] == 0).sum()) if hold_rows else 0,
        "n_holdout": len(hold_rows),
        "checked": bool(len(hold_rows) >= min_holdout_genomes and ref_mean),
    }
    if stats["checked"] and stats["ratio"] < min_ratio:
        raise SystemExit(
            f"holdout genomes carry {hold_mean:.1f} hit unitigs on average against {ref_mean:.1f} for the "
            f"fitted rows (ratio {stats['ratio']:.3f} < {min_ratio}). That is the signature of a holdout "
            f"that was never really scored; a model fitted on this would report a clean AUROC near 0.5. "
            f"{stats['n_holdout_all_zero']}/{len(hold_rows)} holdout rows are entirely empty."
        )
    return stats

def run(
    *, hits_tsv: Path, matrix_gz: Path, split_table: Path, out_dir: Path,
    dedupe_patterns: bool = False, max_hits: int | None = None, decomp_threads: int = 4,
    splits: Sequence[str] = SPLIT_SLICES, min_holdout_carrier_ratio: float = 0.5,
    min_holdout_genomes: int = 30,
) -> dict[str, object]:
    """Extract the hit sub-matrix, build the CSR over the selected split-table genomes, and persist both.

    ``splits`` restricts which slices get rows. It exists for the train+validate-vocabulary arm, where
    the unitig matrix contains no holdout carriers *by construction* — asking it for holdout rows there
    would silently yield zeros. Those rows are scanned from sequence instead and merged in by
    :mod:`bac_pyseer.ast_gwas.unitig_kmer_presence`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    id_map = read_hits(hits_tsv, dedupe_patterns=dedupe_patterns, max_hits=max_hits)
    if id_map.empty:
        raise SystemExit(f"{hits_tsv} yielded no usable hit unitigs — nothing to build a design matrix from")

    label_map, train_ids, validate_ids, holdout_ids = load_splits(split_table)
    unknown = [s for s in splits if s not in SPLIT_SLICES]
    if unknown:
        raise SystemExit(f"unknown split(s) {unknown}; expected a subset of {list(SPLIT_SLICES)}")
    by_split = {"train": train_ids, "validate": validate_ids, "holdout": holdout_ids}
    # Deterministic order: train, then validate, then holdout. Every selected genome gets a row —
    # non-carriers are all-zero, not absent.
    sample_ids = [s for name in SPLIT_SLICES if name in splits for s in by_split[name]]

    submatrix_path = out_dir / "hits_submatrix.tsv"
    n_rows = extract_hit_submatrix(
        matrix_gz, set(id_map[VARIANT_COL]), submatrix_path, decomp_threads=decomp_threads
    )
    matrix, missing, n_empty_columns = build_presence_matrix(submatrix_path, id_map, sample_ids)

    sparse.save_npz(out_dir / "presence.npz", matrix.tocsr())
    id_map.to_csv(out_dir / "id_map.tsv", sep="\t", index=False)
    (out_dir / "samples.txt").write_text("".join(f"{s}\n" for s in sample_ids))
    if missing:
        (out_dir / "unitig_join_misses.txt").write_text("".join(f"{s}\n" for s in sorted(missing)))

    carriers_per_sample = np.asarray(matrix.sum(axis=1)).ravel()
    coverage = check_holdout_coverage(
        matrix, sample_ids, [*train_ids, *validate_ids], holdout_ids if "holdout" in splits else [],
        min_ratio=min_holdout_carrier_ratio, min_holdout_genomes=min_holdout_genomes,
    )
    manifest = {
        "hits_tsv": str(hits_tsv),
        "matrix_gz": str(matrix_gz),
        "split_table": str(split_table),
        "submatrix_rows": n_rows,  # -1 when the cached sub-matrix was reused
        "dedupe_patterns": dedupe_patterns,
        "max_hits": max_hits,
        "n_unitigs": int(len(id_map)),
        "n_samples": len(sample_ids),
        "n_train": len(train_ids),
        "n_validate": len(validate_ids),
        "n_holdout": len(holdout_ids),
        "n_labelled": len(label_map),
        "nnz": int(matrix.nnz),
        "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])) if matrix.nnz else 0.0,
        "n_samples_with_no_hit_unitig": int((carriers_per_sample == 0).sum()),
        "n_unitigs_not_found_in_matrix": len(missing),
        "n_unitigs_absent_from_cohort": n_empty_columns,
        "splits": list(splits),
        "holdout_coverage": coverage,
        "outputs": {
            "presence_npz": str(out_dir / "presence.npz"),
            "id_map_tsv": str(out_dir / "id_map.tsv"),
            "samples_txt": str(out_dir / "samples.txt"),
        },
    }
    (out_dir / "design_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info(
        "design matrix: %d genomes x %d unitigs, nnz=%d (%.4f%% dense), %d genomes carry no hit",
        matrix.shape[0], matrix.shape[1], matrix.nnz, 100 * manifest["density"],
        manifest["n_samples_with_no_hit_unitig"],
    )
    if missing:
        logger.warning("%d hit unitig(s) absent from %s — see unitig_join_misses.txt", len(missing), matrix_gz)
    return manifest


def load_design(design_dir: Path) -> tuple[sparse.csr_matrix, list[str], pd.DataFrame]:
    """Read back a persisted design → ``(csr, sample_ids in row order, id_map)``."""
    matrix = sparse.load_npz(design_dir / "presence.npz").tocsr()
    sample_ids = (design_dir / "samples.txt").read_text().split()
    id_map = pd.read_csv(design_dir / "id_map.tsv", sep="\t")
    if matrix.shape[0] != len(sample_ids):
        raise SystemExit(f"{design_dir}: matrix has {matrix.shape[0]} rows but samples.txt has {len(sample_ids)}")
    if matrix.shape[1] != len(id_map):
        raise SystemExit(f"{design_dir}: matrix has {matrix.shape[1]} cols but id_map.tsv has {len(id_map)}")
    return matrix, sample_ids, id_map


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hits-tsv", type=Path, required=True, help="<drug>_hits_annotated.tsv from pyseer_postprocess.")
    p.add_argument("--matrix-gz", type=Path, required=True, help="The organism's unitigs.pyseer.gz.")
    p.add_argument("--split-table", type=Path, required=True, help="<drug>_split.csv — defines the genome universe.")
    p.add_argument("--out-dir", type=Path, required=True, help="Output directory for presence.npz + id_map.tsv.")
    p.add_argument("--dedupe-patterns", action="store_true",
                   help="Keep one unitig per pattern_group (perfect-LD block) — the honest feature count.")
    p.add_argument("--max-hits", type=int, default=None, help="Keep at most N hits, in rank order.")
    p.add_argument("--splits", default=",".join(SPLIT_SLICES),
                   help="Comma-separated split slices to give rows to. Use 'train,validate' when the "
                        "unitig matrix was built without holdout genomes; scan those rows separately.")
    p.add_argument("--min-holdout-carrier-ratio", type=float, default=0.5,
                   help="Fail if holdout genomes carry fewer than this fraction of the fitted rows' "
                        "mean hit count — the all-zero-holdout guard.")
    p.add_argument("--decomp-threads", type=int, default=4, help="pigz threads for the one big-matrix pass.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    manifest = run(
        hits_tsv=args.hits_tsv, matrix_gz=args.matrix_gz, split_table=args.split_table,
        out_dir=args.out_dir, dedupe_patterns=args.dedupe_patterns, max_hits=args.max_hits,
        decomp_threads=args.decomp_threads,
        splits=tuple(s.strip() for s in args.splits.split(",") if s.strip()),
        min_holdout_carrier_ratio=args.min_holdout_carrier_ratio,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
