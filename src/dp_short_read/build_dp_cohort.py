r"""Build the DefensePredictor run manifest from the paired LR/SR cohort.

Selects genomes from ``paired_index.tsv`` (the curated long-read-vs-short-read pairing,
``processed/complete_vs_sr_genomes/``), joins to ``metadata_v2`` to pull the four per-genome
file-path columns, and expands each row into up to **two** genome records — one for the
long-read (LR) assembly and one for the short-read (SR) assembly — each only when both its
GFF and assembly resolve and exist on disk.

The output manifest (one row per genome arm) is the single input to
:mod:`run_defense_predictor`. Path-resolution and row-expansion mirror BacHGT's
``panaroo_run_strain._genome_records_for_row`` so DefensePredictor sees the same gene models
Panaroo does, but the helpers are re-implemented here to keep the isolated DP venv free of a
BacHGT import.

Examples
--------
Smoke manifest — 10 highest-confidence reference genomes, both arms::

    python build_dp_cohort.py \\
        --paired-index <project_k>/david/processed/complete_vs_sr_genomes/paired_index.tsv \\
        --metadata     <project_k>/david/final/metadata_v2_all_samples_and_columns.tsv \\
        --base-dir     <project_k> \\
        --reference-only --limit 10 \\
        --out          <out_dir>/dp_manifest_smoke.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# metadata_v2 is mid-rename (METADATA_v2_README §12): the on-disk TSV may still carry the
# legacy column names until the next rebuild. Accept either spelling for each path column.
GFF_SR_COLS = ("sr_gff_file", "gff_file")
ASM_SR_COLS = ("sr_assembly_file", "assembly_file")
GFF_LR_COLS = ("lr_gff_file", "lra_gff_file")
ASM_LR_COLS = ("lr_assembly_file", "lra_assembly_file")

MANIFEST_COLS = [
    "panaroo_label",
    "arm",  # "lr" or "sr"
    "Sample",
    "sample_accession",
    "gff_abs",
    "assembly_abs",
]


def _first_present(row: pd.Series, candidates: tuple[str, ...]):
    """Return the first non-null value among ``candidates`` columns on ``row``."""
    for col in candidates:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                return val
    return None


def _abs_path(base: Path, rel) -> Path | None:
    """Resolve a metadata path column to an absolute Path.

    SR paths are stored relative to ``base`` (the project_k root); LR paths are stored
    absolute (``/home/dca36/...``). A leading ``/`` selects between the two. ``<project_k>``
    placeholder prefixes (README §12, path-relative rewrite still pending) are stripped and
    treated as relative.
    """
    if rel is None or pd.isna(rel) or not str(rel).strip():
        return None
    s = str(rel).strip()
    if s.startswith("<project_k>"):
        s = s[len("<project_k>") :].lstrip("/")
    p = Path(s) if s.startswith("/") else base / s
    return p


def _both_exist(gff: Path | None, assembly: Path | None) -> bool:
    return gff is not None and assembly is not None and gff.exists() and assembly.exists()


def genome_records_for_row(base: Path, row: pd.Series) -> list[dict]:
    """Expand one metadata row into up to two (LR + SR) genome records.

    A record is emitted only when both its GFF and assembly resolve and exist on disk, so a
    row missing one arm's files contributes just the other arm (graceful per-arm skipping).
    The Panaroo-style label is the accession that already matches the arm's files:
    ``Sample`` for the long-read genome, ``sample_accession`` for the short-read genome.
    """
    records: list[dict] = []
    sample = row.get("Sample")
    sample_accession = row.get("sample_accession")

    lr_gff = _abs_path(base, _first_present(row, GFF_LR_COLS))
    lr_asm = _abs_path(base, _first_present(row, ASM_LR_COLS))
    if _both_exist(lr_gff, lr_asm):
        records.append(
            {
                "panaroo_label": str(sample).strip(),
                "arm": "lr",
                "Sample": sample,
                "sample_accession": sample_accession,
                "gff_abs": str(lr_gff),
                "assembly_abs": str(lr_asm),
            }
        )

    sr_gff = _abs_path(base, _first_present(row, GFF_SR_COLS))
    sr_asm = _abs_path(base, _first_present(row, ASM_SR_COLS))
    if _both_exist(sr_gff, sr_asm):
        if pd.notna(sample_accession) and str(sample_accession).strip():
            records.append(
                {
                    "panaroo_label": str(sample_accession).strip(),
                    "arm": "sr",
                    "Sample": sample,
                    "sample_accession": sample_accession,
                    "gff_abs": str(sr_gff),
                    "assembly_abs": str(sr_asm),
                }
            )
    return records


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(
        lambda x: str(x).strip().lower() in ("true", "1", "yes", "t") if pd.notna(x) else False
    )


def build_manifest(
    paired_index: Path,
    metadata: Path,
    base: Path,
    reference_only: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    """Build the per-arm genome manifest for the paired cohort.

    Parameters
    ----------
    paired_index
        ``paired_index.tsv`` — the LR/SR pairing, keyed by ``lra_sample`` + ``sr_biosample``.
    metadata
        ``metadata_v2`` TSV holding the four per-genome file-path columns, keyed by ``Sample``.
    base
        project_k root, prepended to SR (relative) path columns.
    reference_only
        Keep only ``lra_is_reference_genome`` pairs (highest-confidence reference set).
    limit
        Keep at most this many genomes (pairs are taken in sorted ``lra_sample`` order for a
        deterministic smoke set).
    """
    pi = pd.read_csv(paired_index, sep="\t", dtype=str)
    if reference_only:
        if "lra_is_reference_genome" not in pi.columns:
            raise KeyError("paired_index has no 'lra_is_reference_genome' column")
        pi = pi[_as_bool(pi["lra_is_reference_genome"])]
    pi = pi.sort_values("lra_sample")

    meta = pd.read_csv(metadata, sep="\t", dtype=str, low_memory=False)
    if "Sample" not in meta.columns:
        raise KeyError("metadata has no 'Sample' column")
    meta_by_sample = meta.set_index("Sample")

    records: list[dict] = []
    seen_samples = 0
    for lra_sample in pi["lra_sample"].dropna().unique():
        if lra_sample not in meta_by_sample.index:
            print(f"  WARN: {lra_sample} not in metadata — skipping")
            continue
        row = meta_by_sample.loc[lra_sample]
        if isinstance(row, pd.DataFrame):  # duplicate Sample — take first
            row = row.iloc[0]
        recs = genome_records_for_row(base, row)
        if recs:
            records.extend(recs)
            seen_samples += 1
        if limit is not None and seen_samples >= limit:
            break

    manifest = pd.DataFrame(records, columns=MANIFEST_COLS)
    return manifest


def main() -> None:
    """CLI: write the DefensePredictor run manifest to TSV."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paired-index", required=True, type=Path)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--base-dir", required=True, type=Path, help="project_k root for relative SR paths")
    ap.add_argument("--reference-only", action="store_true", help="keep only lra_is_reference_genome pairs")
    ap.add_argument("--limit", type=int, default=None, help="cap number of genomes (sorted by lra_sample)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    manifest = build_manifest(
        paired_index=args.paired_index,
        metadata=args.metadata,
        base=args.base_dir,
        reference_only=args.reference_only,
        limit=args.limit,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, sep="\t", index=False)
    n_lr = (manifest["arm"] == "lr").sum()
    n_sr = (manifest["arm"] == "sr").sum()
    print(f"Wrote {len(manifest)} genome arms ({n_lr} LR + {n_sr} SR) -> {args.out}")


if __name__ == "__main__":
    main()
