r"""Build the DefensePredictor run manifest for the paired LR/SR complete-genome cohort.

Reads ``metadata_v2`` directly and selects rows that carry **all four** distinct per-genome
file paths — ``lr_assembly_file`` + ``lr_gff_file`` (long-read arm) and ``sr_assembly_file`` +
``sr_gff_file`` (short-read partner) — then filters by cohort (``complete`` / ``reference`` /
``all``). Each selected row expands into two genome records: one LR arm and one SR arm, sharing
the same ``Sample`` so their DefensePredictor outputs can be paired afterwards.

This is the post-fix design: metadata_v2 (2026-06-03) repopulated ``sr_assembly_file`` for the
merged RefSeq pairs, so all four columns are now genuinely distinct for the paired complete
cohort (1,454 rows; 709 reference). No ``paired_index.tsv`` join and no ``sr_shadow`` lookup are
needed — the four columns on one row are authoritative.

A record is emitted only when both its assembly and GFF resolve and exist on disk, so a Sample
whose SR file is missing on disk contributes only its LR arm (graceful per-arm skipping). The
manifest is the single input to :mod:`run_defense_predictor`.

Examples
--------
Smoke manifest — 10 reference genomes, both arms::

    python build_dp_cohort.py \\
        --metadata <project_k>/david/final/metadata_v2_all_samples_and_columns.tsv \\
        --base-dir <project_k> \\
        --cohort reference --limit 10 \\
        --out <out_dir>/dp_manifest_smoke.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Canonical names first; legacy spellings kept as fallback for older metadata snapshots.
ASM_LR_COLS = ("lr_assembly_file", "lra_assembly_file")
GFF_LR_COLS = ("lr_gff_file", "lra_gff_file")
ASM_SR_COLS = ("sr_assembly_file", "assembly_file")
GFF_SR_COLS = ("sr_gff_file", "gff_file")

_TRUE = {"true", "1", "yes", "t"}

MANIFEST_COLS = [
    "panaroo_label",  # output id: Sample (LR arm) / sr_biosample (SR arm)
    "arm",  # "lr" or "sr"
    "Sample",  # LR accession — shared by both arms of a pair (the pairing key)
    "sr_biosample",
    "is_reference",
    "gff_abs",
    "assembly_abs",
]


def _is_true(val) -> bool:
    return pd.notna(val) and str(val).strip().lower() in _TRUE


def _first_present(row: pd.Series, candidates: tuple[str, ...]):
    """Return the first non-empty value among ``candidates`` columns on ``row``."""
    for col in candidates:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                return val
    return None


def _abs_path(base: Path, rel) -> Path | None:
    """Resolve a metadata path value to an absolute Path.

    SR paths are stored relative to ``base`` (project_k root); LR paths are absolute. A leading
    ``/`` selects between the two; a ``<project_k>`` placeholder prefix is stripped and treated
    as relative.
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

    Each arm is emitted only when both its assembly and GFF resolve and exist on disk. The LR
    arm is labelled by ``Sample`` (the long-read accession); the SR arm by ``sr_biosample``
    (the short-read partner BioSample, which matches the SR assembly filename).
    """
    records: list[dict] = []
    sample = row.get("Sample")
    sr_biosample = row.get("sr_biosample")
    sample_accession = row.get("sample_accession")
    is_reference = _is_true(row.get("is_reference_genome"))
    sr_label = sr_biosample if (pd.notna(sr_biosample) and str(sr_biosample).strip()) else sample_accession

    lr_gff = _abs_path(base, _first_present(row, GFF_LR_COLS))
    lr_asm = _abs_path(base, _first_present(row, ASM_LR_COLS))
    if _both_exist(lr_gff, lr_asm):
        records.append(
            {
                "panaroo_label": str(sample).strip(),
                "arm": "lr",
                "Sample": sample,
                "sr_biosample": sr_biosample,
                "is_reference": is_reference,
                "gff_abs": str(lr_gff),
                "assembly_abs": str(lr_asm),
            }
        )

    sr_gff = _abs_path(base, _first_present(row, GFF_SR_COLS))
    sr_asm = _abs_path(base, _first_present(row, ASM_SR_COLS))
    if _both_exist(sr_gff, sr_asm) and pd.notna(sr_label) and str(sr_label).strip():
        records.append(
            {
                "panaroo_label": str(sr_label).strip(),
                "arm": "sr",
                "Sample": sample,
                "sr_biosample": sr_biosample,
                "is_reference": is_reference,
                "gff_abs": str(sr_gff),
                "assembly_abs": str(sr_asm),
            }
        )
    return records


def _effective(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    """Per-row first non-empty value among candidate columns (vectorised _first_present)."""
    out = pd.Series([None] * len(df), index=df.index, dtype=object)
    for col in candidates:
        if col in df.columns:
            v = df[col]
            fill = out.isna() & v.notna() & (v.astype(str).str.strip() != "")
            out = out.mask(fill, v)
    return out


def build_manifest(
    metadata: Path,
    base: Path,
    cohort: str = "complete",
    limit: int | None = None,
) -> pd.DataFrame:
    """Build the per-arm genome manifest from metadata_v2.

    Parameters
    ----------
    metadata
        ``metadata_v2`` TSV with the four per-genome file-path columns, keyed by ``Sample``.
    base
        project_k root, prepended to relative (SR) path columns.
    cohort
        ``complete`` (``is_complete``), ``reference`` (``is_reference_genome``), or ``all`` —
        applied on top of the always-required all-four-files-present filter.
    limit
        Keep at most this many genomes (sorted by ``Sample`` for a deterministic smoke set).
    """
    if cohort not in {"complete", "reference", "all"}:
        raise ValueError(f"--cohort must be complete/reference/all, got {cohort!r}")

    # metadata_v2 is wide (~450 cols) / large (~266 MB); read only the columns we need.
    header = pd.read_csv(metadata, sep="\t", nrows=0).columns
    if "Sample" not in header:
        raise KeyError("metadata has no 'Sample' column")
    path_cols = ASM_LR_COLS + GFF_LR_COLS + ASM_SR_COLS + GFF_SR_COLS
    flag_cols = ("sample_accession", "sr_biosample", "is_complete", "is_reference_genome")
    usecols = ["Sample"] + [c for c in (flag_cols + path_cols) if c in header]
    meta = pd.read_csv(metadata, sep="\t", dtype=str, usecols=usecols)

    # Require all four file paths present (the definition of a usable LR-vs-SR pair).
    lr_asm = _effective(meta, ASM_LR_COLS)
    lr_gff = _effective(meta, GFF_LR_COLS)
    sr_asm = _effective(meta, ASM_SR_COLS)
    sr_gff = _effective(meta, GFF_SR_COLS)
    all_four = lr_asm.notna() & lr_gff.notna() & sr_asm.notna() & sr_gff.notna()
    meta = meta[all_four]

    if cohort == "complete":
        meta = meta[meta["is_complete"].map(_is_true)]
    elif cohort == "reference":
        meta = meta[meta["is_reference_genome"].map(_is_true)]

    meta = meta.sort_values("Sample")
    print(f"Cohort '{cohort}': {len(meta)} paired genomes with all 4 distinct files")

    records: list[dict] = []
    seen = 0
    for _, row in meta.iterrows():
        recs = genome_records_for_row(base, row)
        if recs:
            records.extend(recs)
            seen += 1
        if limit is not None and seen >= limit:
            break

    return pd.DataFrame(records, columns=MANIFEST_COLS)


def main() -> None:
    """CLI: write the DefensePredictor run manifest to TSV."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--base-dir", required=True, type=Path, help="project_k root for relative SR paths")
    ap.add_argument("--cohort", choices=["complete", "reference", "all"], default="complete")
    ap.add_argument("--limit", type=int, default=None, help="cap number of genomes (sorted by Sample)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    manifest = build_manifest(
        metadata=args.metadata,
        base=args.base_dir,
        cohort=args.cohort,
        limit=args.limit,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, sep="\t", index=False)
    n_lr = (manifest["arm"] == "lr").sum()
    n_sr = (manifest["arm"] == "sr").sum()
    n_pairs = manifest.groupby("Sample")["arm"].nunique().eq(2).sum()
    print(f"Wrote {len(manifest)} arms ({n_lr} LR + {n_sr} SR; {n_pairs} complete LR+SR pairs) -> {args.out}")


if __name__ == "__main__":
    main()
