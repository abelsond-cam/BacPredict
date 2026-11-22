r"""Re-extract protein parquets under a changed extractor setting, rewriting only those whose proteins differ.

``preprocess_assemblies_to_protein_sequences --skip-existing`` tests a parquet's existence only, so a store built
before an extractor setting changed keeps its old proteins. This re-runs the extractor (Bakta GFF3 + contigs, as
that module does) for every genome of a cohort table and compares the result with the parquet in place: a genome
whose proteins, coordinates and contig grouping are the same is left alone — its file, and its mtime, which the
cohort checks date embeddings against, unchanged — and one that differs (or has no parquet) is written atomically.

Written for CABBAGE v2's ``--ambiguous-codons-as-x`` (PROJECT_STATE §6, 2026-11-22): only genomes with a non-ACGT
base inside a CDS change. ``--exclude`` (a CSV with a ``Sample`` column) leaves genomes out: another cohort's
parquets in a shared store are that cohort's.

Usage::

    python -m bacpredict.engine.embedding.reextract_proteins --input-csv genomes/bakta_annotation.csv \
        --exclude genomes/reused_genomes.csv --output-dir processed/protein_sequences --ambiguous-codons-as-x \
        --workers 32 --report reextract.csv

The report holds one row per genome: ``unchanged``, ``rewritten`` (with the protein count before and after),
``written`` (no parquet before) or ``error: …``, and ``nonstandard``: any residue letter outside ESM-C's protein
alphabet as the extractor can write it (the 20, ``X``, and the recoded ``U``/``O``) — Biopython's ``B``/``Z``/``J``
for an ambiguous codon, say, which the tokenizer would take as ``B``/``Z`` or not at all. Exit status 1 if any
genome errored or holds one.
"""

from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from bacpredict.engine.embedding.extract_proteins_from_gff_fna import extract_proteins_from_gff_fna

# what preprocess_assemblies_to_protein_sequences drops before writing, and what is compared
_DROPPED = ("strain_name", "accession_name", "protein_name", "accession_id")
_COMPARED = ("contig_idx", "start", "end", "protein_sequence")
ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY") | frozenset("XUO")


def parquet_path(output_dir: Path, sample: str) -> Path:
    """``<output_dir>/<sample>_protein_sequences.parquet``, the protein store's name."""
    return output_dir / f"{sample}_protein_sequences.parquet"


def genome_record(sample: str, contigs: str, gff: str, *, ambiguous_codons_as_x: bool) -> dict:
    """The one-row record ``preprocess_assemblies_to_protein_sequences`` writes for this genome."""
    info = extract_proteins_from_gff_fna(gff, contigs, ambiguous_codons_as_x=ambiguous_codons_as_x)
    for k in _DROPPED:
        info.pop(k, None)
    return {"sample_id": sample, **info}


def _as_lists(value) -> list:
    """A parquet nested-list cell (numpy arrays of arrays) or a Python list, as nested Python lists."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [_as_lists(v) if hasattr(v, "tolist") or isinstance(v, list) else v for v in value]


def same_proteins(existing: pd.Series, record: dict) -> bool:
    """Whether a parquet row holds the record's proteins, coordinates and contig grouping."""
    return all(k in existing.index and _as_lists(existing[k]) == _as_lists(record[k]) for k in _COMPARED)


def _n_proteins(cell) -> int:
    return sum(len(c) for c in cell)


def nonstandard(cell) -> str:
    """The residue letters of a record's proteins outside :data:`ALPHABET`, sorted and joined ('' if none)."""
    return "".join(sorted({a for contig in cell for p in contig for a in p} - ALPHABET))


def write_atomically(record: dict, path: Path) -> None:
    """As ``preprocess_assemblies_to_protein_sequences.save_to_parquet``: a hidden sibling, renamed into place."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        pd.DataFrame([record]).to_parquet(tmp, engine="pyarrow", compression="snappy")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def reextract_one(task: tuple[str, str, str, str, bool]) -> dict:
    """Re-extract one genome; write its parquet only if it is missing or its proteins differ."""
    sample, contigs, gff, output_dir, as_x = task
    path = parquet_path(Path(output_dir), sample)
    try:
        record = genome_record(sample, contigs, gff, ambiguous_codons_as_x=as_x)
        n_new = _n_proteins(record["protein_sequence"])
        odd = nonstandard(record["protein_sequence"])
        if not path.exists():
            write_atomically(record, path)
            return {"Sample": sample, "status": "written", "n_before": None, "n_after": n_new, "nonstandard": odd}
        old = pd.read_parquet(path).iloc[0]
        if same_proteins(old, record):
            return {"Sample": sample, "status": "unchanged", "n_before": n_new, "n_after": n_new, "nonstandard": odd}
        n_old = _n_proteins(old["protein_sequence"])
        write_atomically(record, path)
        return {"Sample": sample, "status": "rewritten", "n_before": n_old, "n_after": n_new, "nonstandard": odd}
    except Exception as e:  # noqa: BLE001 (one genome's failure is reported, and fails the run, after the rest)
        return {"Sample": sample, "status": f"error: {type(e).__name__}: {e}", "n_before": None, "n_after": None,
                "nonstandard": ""}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-csv", type=Path, required=True, help="Sample, sr_assembly_file, sr_gff_file")
    ap.add_argument("--exclude", type=Path, default=None, help="CSV with a Sample column: genomes to leave alone")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--ambiguous-codons-as-x", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args(argv)

    df = pd.read_csv(a.input_csv, dtype={"Sample": str}, usecols=["Sample", "sr_assembly_file", "sr_gff_file"])
    if df.isna().any().any() or not df["Sample"].is_unique:
        sys.exit(f"{a.input_csv}: every row needs a unique Sample, contigs and a GFF3")
    if a.exclude is not None:
        excluded = set(pd.read_csv(a.exclude, dtype=str, usecols=["Sample"])["Sample"])
        print(f"{int(df['Sample'].isin(excluded).sum())} of {len(df)} genomes excluded ({a.exclude})", flush=True)
        df = df[~df["Sample"].isin(excluded)]
    tasks = [(s, c, g, str(a.output_dir), a.ambiguous_codons_as_x)
             for s, c, g in df[["Sample", "sr_assembly_file", "sr_gff_file"]].itertuples(index=False)]
    with Pool(a.workers) as pool:
        rows = pool.map(reextract_one, tasks, chunksize=16)
    report = pd.DataFrame(rows, columns=["Sample", "status", "n_before", "n_after", "nonstandard"])
    a.report.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(a.report, index=False)
    kind = report["status"].str.split(":").str[0]
    print(f"{len(report)} genomes: {kind.value_counts().to_dict()}; report {a.report}", flush=True)
    changed = report[kind.eq("rewritten")]
    if len(changed):
        delta = (changed["n_after"] - changed["n_before"]).value_counts().sort_index().to_dict()
        print(f"rewritten, protein count after − before: {delta}", flush=True)
    odd = report["nonstandard"].fillna("").ne("")
    if odd.any():
        print(f"FAIL: {int(odd.sum())} genomes hold residues outside {''.join(sorted(ALPHABET))}: "
              f"{report.loc[odd, ['Sample', 'nonstandard']].head(10).values.tolist()}", flush=True)
    return 1 if kind.eq("error").any() or odd.any() else 0


if __name__ == "__main__":
    sys.exit(main())
