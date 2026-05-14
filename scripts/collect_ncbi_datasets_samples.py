#!/usr/bin/env python3
"""Resolve TB BioSamples to NCBI assembly accessions and prepare datasets batches.

Standalone helper for download_ncbi_datasets.sh. Runs under the `ncbi-datasets`
micromamba env (ncbi-datasets-cli + pandas; no project imports, no uv).

Flow:
1. Read the TB AMR records CSV; deduplicate to unique SAM*-prefixed BioSamples
2. Skip-existing: load the most recent biosample_to_accession_*.tsv cache (if any)
   and drop BioSamples whose mapped accession already has a populated dir under
   --output-dir
3. Resolve remaining BioSamples to assembly accessions via NCBI Entrez E-utilities
   (esearch on db=assembly with BioSamples as the query term, then esummary on the
   resulting UIDs). The datasets CLI's `summary genome accession` subcommand only
   accepts Assembly/BioProject accessions, not BioSamples, so we use Entrez for
   the lookup and feed the resolved GCF_/GCA_ accessions into the datasets CLI
   for the actual download. We then pick the best record per BioSample
   (prefer GCF_ over GCA_, higher assembly_level, then most recent release_date)
4. Write a fresh biosample_to_accession_*.tsv (cache for next run)
5. Write missing_samples_*.tsv for BioSamples with no resolvable accession
6. Write batch files of unique accessions for the datasets CLI

Never mutates the input CSV.
"""

import argparse
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

BIOSAMPLE_COL = "phenotype-BioSample_ID"

# Higher rank wins when picking a "best" assembly for a BioSample.
ASSEMBLY_LEVEL_RANK = {
    "Complete Genome": 4,
    "Chromosome": 3,
    "Scaffold": 2,
    "Contig": 1,
}


def load_biosamples(metadata_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load CSV and return (full_df, deduplicated SAM*-prefixed BioSample list)."""
    df = pd.read_csv(metadata_path, low_memory=False)
    initial_rows = len(df)

    if BIOSAMPLE_COL not in df.columns:
        print(f"ERROR: '{BIOSAMPLE_COL}' column not found in {metadata_path}", file=sys.stderr)
        sys.exit(1)

    biosamples = df[BIOSAMPLE_COL].astype(str).str.strip()
    sam_mask = biosamples.str.startswith("SAM")
    num_sam = int(sam_mask.sum())
    num_non_sam = int((~sam_mask).sum())

    print(
        f"Input CSV: {initial_rows:,} rows; SAM-prefixed BioSample rows: {num_sam:,}; "
        f"non-SAM rows: {num_non_sam:,}",
        file=sys.stderr,
    )
    if num_non_sam > 0:
        prefix_counts = biosamples[~sam_mask].str[:3].value_counts().head(10)
        print("Top non-SAM prefixes (first 3 chars):", file=sys.stderr)
        for prefix, count in prefix_counts.items():
            print(f"  {prefix or '<EMPTY>'}: {count}", file=sys.stderr)

    unique = (
        biosamples[sam_mask]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    print(f"Unique SAM* BioSamples: {len(unique):,}", file=sys.stderr)
    return df, unique


def load_accession_cache(output_dir: Path) -> dict[str, str]:
    """Merge all biosample_to_accession_*.tsv files in output_dir into one dict.

    Most recent value wins on conflict.
    """
    if not output_dir.is_dir():
        return {}
    caches = sorted(output_dir.glob("biosample_to_accession_*.tsv"))
    mapping: dict[str, str] = {}
    for cache in caches:
        try:
            df = pd.read_csv(cache, sep="\t", dtype=str)
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            print(f"WARNING: could not read cache {cache}: {exc}", file=sys.stderr)
            continue
        if "biosample" not in df.columns or "assembly_accession" not in df.columns:
            continue
        for bs, acc in zip(df["biosample"].fillna(""), df["assembly_accession"].fillna("")):
            if bs and acc:
                mapping[bs] = acc
    if mapping:
        print(f"Loaded accession cache: {len(mapping):,} BioSample->accession entries "
              f"from {len(caches)} file(s)", file=sys.stderr)
    return mapping


def existing_accession_dirs(output_dir: Path) -> set[str]:
    """Set of accession names already present as a non-empty directory in output_dir."""
    if not output_dir.is_dir():
        return set()
    out: set[str] = set()
    for p in output_dir.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith(("GCF_", "GCA_")):
            continue
        try:
            if next(p.iterdir(), None) is not None:
                out.add(p.name)
        except OSError:
            continue
    return out


def extract_biosample_id(record: dict) -> str | None:
    """Pull the BioSample accession out of a datasets summary JSONL record.

    Handles both nested (assembly_info.biosample.accession) and flat
    (biosample_accession) shapes across datasets CLI versions.
    """
    info = record.get("assembly_info") or {}
    bs = info.get("biosample")
    if isinstance(bs, dict):
        acc = bs.get("accession")
        if acc:
            return acc
    elif isinstance(bs, str) and bs:
        return bs
    for key in ("biosample_accession", "biosample"):
        val = record.get(key)
        if isinstance(val, str) and val:
            return val
    val = info.get("biosample_accession")
    if isinstance(val, str) and val:
        return val
    return None


def pick_best_record(records: list[dict]) -> dict | None:
    """Pick the best assembly for one BioSample.

    Priority: GCF_ (RefSeq) > GCA_ (GenBank); then assembly_level
    (Complete > Chromosome > Scaffold > Contig); then most recent release_date.
    """
    if not records:
        return None

    def key(r: dict) -> tuple:
        acc = (r.get("accession") or "").strip()
        info = r.get("assembly_info") or {}
        return (
            acc.startswith("GCF_"),
            ASSEMBLY_LEVEL_RANK.get(info.get("assembly_level"), 0),
            info.get("release_date") or "",
        )

    return max(records, key=key)


def parse_records_to_mapping(
    records: list[dict],
    queried: list[str],
) -> tuple[list[dict], list[str]]:
    """Group records by BioSample and pick the best assembly for each.

    Returns (rows, missing) where rows is a list of dicts ready for the
    biosample_to_accession sidecar TSV, and missing is a list of BioSamples
    in `queried` with no assembly hit.
    """
    by_biosample: dict[str, list[dict]] = {}
    for rec in records:
        bs = extract_biosample_id(rec)
        if bs:
            by_biosample.setdefault(bs, []).append(rec)

    rows: list[dict] = []
    found: set[str] = set()
    for bs in queried:
        recs = by_biosample.get(bs, [])
        best = pick_best_record(recs)
        if best is None:
            continue
        acc = (best.get("accession") or "").strip()
        if not acc:
            continue
        info = best.get("assembly_info") or {}
        rows.append({
            "biosample": bs,
            "assembly_accession": acc,
            "source": "refseq" if acc.startswith("GCF_") else "genbank",
            "assembly_level": info.get("assembly_level") or "",
            "release_date": info.get("release_date") or "",
        })
        found.add(bs)

    missing = [b for b in queried if b not in found]
    return rows, missing


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _entrez_get(endpoint: str, params: dict) -> bytes:
    """Single GET against NCBI E-utilities. Honours NCBI_API_KEY if set."""
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params = {**params, "api_key": api_key}
    url = f"{EUTILS}/{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def _entrez_sleep() -> None:
    """Rate-limit: 3 req/s without API key, 10 req/s with."""
    time.sleep(0.11 if os.environ.get("NCBI_API_KEY") else 0.34)


def _esearch_assembly_uids(biosamples: list[str], batch_size: int = 100) -> set[str]:
    """Return the set of assembly UIDs linked to any biosample in the input.

    Batches biosamples into OR-joined query terms so we make ~1 esearch call per
    `batch_size` BioSamples instead of one per BioSample.
    """
    uids: set[str] = set()
    for i in range(0, len(biosamples), batch_size):
        chunk = biosamples[i : i + batch_size]
        term = " OR ".join(chunk)
        xml = _entrez_get(
            "esearch.fcgi",
            {"db": "assembly", "term": term, "retmax": "100000"},
        )
        try:
            root = ET.fromstring(xml)
        except ET.ParseError as exc:
            print(f"WARNING: esearch XML parse error on batch {i}-{i + len(chunk)}: {exc}",
                  file=sys.stderr)
            _entrez_sleep()
            continue
        for el in root.findall(".//IdList/Id"):
            if el.text:
                uids.add(el.text)
        _entrez_sleep()
    return uids


def _esummary_assemblies(uids: list[str], batch_size: int = 200) -> list[dict]:
    """Fetch assembly metadata for `uids` and shape it into records compatible
    with `pick_best_record` / `extract_biosample_id`.

    Each returned record has the keys our parser expects:
      - accession (GCF_/GCA_)
      - assembly_info.assembly_level
      - assembly_info.release_date
      - assembly_info.biosample.accession
    """
    records: list[dict] = []
    for i in range(0, len(uids), batch_size):
        chunk = uids[i : i + batch_size]
        xml = _entrez_get(
            "esummary.fcgi",
            {"db": "assembly", "id": ",".join(chunk), "version": "2.0"},
        )
        try:
            root = ET.fromstring(xml)
        except ET.ParseError as exc:
            print(f"WARNING: esummary XML parse error on batch {i}-{i + len(chunk)}: {exc}",
                  file=sys.stderr)
            _entrez_sleep()
            continue
        for doc in root.findall(".//DocumentSummary"):
            acc = (doc.findtext("AssemblyAccession") or "").strip()
            if not acc:
                continue
            level = (doc.findtext("AssemblyStatus") or "").strip()
            # SeqReleaseDate / SubmissionDate look like "2022/01/20 00:00";
            # keep them as ISO-ish strings - lexicographic compare still orders correctly.
            release_date = (
                (doc.findtext("SeqReleaseDate") or "").strip()
                or (doc.findtext("SubmissionDate") or "").strip()
            )
            biosample_acc = (doc.findtext("BioSampleAccn") or "").strip()
            records.append({
                "accession": acc,
                "assembly_info": {
                    "assembly_level": level,
                    "release_date": release_date,
                    "biosample": {"accession": biosample_acc},
                },
            })
        _entrez_sleep()
    return records


def resolve_biosamples_via_entrez(biosamples: list[str]) -> list[dict]:
    """Resolve BioSamples -> assembly metadata via NCBI Entrez E-utilities.

    The NCBI datasets CLI's `summary genome accession` only accepts Assembly /
    BioProject accessions (not BioSamples), so we use Entrez E-utilities for
    the BioSample -> Assembly mapping:

      1. esearch(db=assembly, term=<biosample1> OR <biosample2> OR ...)
         -> assembly UIDs
      2. esummary(db=assembly, id=<uid1>,<uid2>,...)
         -> assembly metadata including BioSampleAccn

    Returns records in the shape expected by pick_best_record. Empty list on
    network failure for any specific call (other batches still tried).

    Honours NCBI_API_KEY env var for higher rate limits (10 req/s vs 3 req/s).
    """
    if not biosamples:
        return []
    print(f"Resolving {len(biosamples):,} BioSamples via NCBI Entrez "
          f"(esearch + esummary on db=assembly)...", file=sys.stderr)
    uids = _esearch_assembly_uids(biosamples)
    print(f"esearch returned {len(uids):,} unique assembly UIDs", file=sys.stderr)
    if not uids:
        return []
    records = _esummary_assemblies(sorted(uids))
    print(f"esummary returned metadata for {len(records):,} assemblies", file=sys.stderr)
    return records


def write_mapping_tsv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["biosample", "assembly_accession", "source", "assembly_level", "release_date"]
    pd.DataFrame(rows, columns=cols).to_csv(path, sep="\t", index=False)


def write_missing_sidecar(
    missing_biosamples: list[str],
    full_df: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not missing_biosamples:
        pd.DataFrame(columns=[BIOSAMPLE_COL]).to_csv(path, sep="\t", index=False)
        return
    sub = (
        full_df[full_df[BIOSAMPLE_COL].astype(str).str.strip().isin(missing_biosamples)]
        .drop_duplicates(subset=[BIOSAMPLE_COL])
        .copy()
    )
    sub.to_csv(path, sep="\t", index=False)


def write_batches(
    accessions: list[str],
    batch_dir: Path | None,
    batch_size: int,
    output: Path | None,
) -> None:
    if batch_dir is not None:
        batch_dir.mkdir(parents=True, exist_ok=True)
        num_batches = (len(accessions) + batch_size - 1) // batch_size if accessions else 0
        width = max(2, len(str(max(num_batches - 1, 0))))
        for i in range(num_batches):
            chunk = accessions[i * batch_size : (i + 1) * batch_size]
            (batch_dir / f"batch_{i:0{width}d}").write_text(
                "\n".join(chunk) + ("\n" if chunk else "")
            )
        print(f"Wrote {num_batches} batch files to {batch_dir}", file=sys.stderr)
        print(f"TOTAL={len(accessions)}", file=sys.stderr)
        print(f"NUM_BATCHES={num_batches}", file=sys.stderr)
    elif output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(accessions) + ("\n" if accessions else ""))
        print(f"Wrote {len(accessions)} accessions to {output}", file=sys.stderr)
    else:
        for acc in accessions:
            print(acc)


def run(
    metadata: Path,
    output_dir: Path | None,
    accession_map_path: Path | None,
    missing_output_path: Path | None,
    n: int,
    batch_dir: Path | None,
    batch_size: int,
    output: Path | None,
    resolver=resolve_biosamples_via_entrez,
) -> None:
    """Main entry point - factored for testability (resolver injectable)."""
    full_df, biosamples = load_biosamples(metadata)

    # 1. Apply cache + filesystem skip-existing
    cache = load_accession_cache(output_dir) if output_dir is not None else {}
    existing_dirs = existing_accession_dirs(output_dir) if output_dir is not None else set()
    if cache and existing_dirs:
        already_done = {bs for bs, acc in cache.items() if acc in existing_dirs}
        if already_done:
            before = len(biosamples)
            biosamples = [b for b in biosamples if b not in already_done]
            print(f"Skip-existing: dropped {before - len(biosamples):,} BioSamples "
                  f"already mapped + downloaded", file=sys.stderr)

    # 2. Apply --n limit before the costly resolution step
    if n >= 0:
        biosamples = biosamples[:n]
        print(f"Limiting to first {n} BioSamples", file=sys.stderr)

    # 3. Resolve via datasets CLI
    records = resolver(biosamples) if biosamples else []
    rows, missing = parse_records_to_mapping(records, biosamples)

    print(f"Resolved {len(rows):,} BioSamples to accessions; "
          f"{len(missing):,} unresolved", file=sys.stderr)

    # 4. Write sidecars
    if accession_map_path is not None:
        write_mapping_tsv(rows, accession_map_path)
        print(f"Wrote accession map: {accession_map_path}", file=sys.stderr)
    if missing_output_path is not None:
        write_missing_sidecar(missing, full_df, missing_output_path)
        print(f"Wrote missing-samples sidecar: {missing_output_path} ({len(missing):,} rows)",
              file=sys.stderr)

    # 5. Deduplicate accessions and emit batches
    seen: set[str] = set()
    accessions: list[str] = []
    for row in rows:
        acc = row["assembly_accession"]
        if acc and acc not in seen:
            seen.add(acc)
            accessions.append(acc)

    write_batches(accessions, batch_dir, batch_size, output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve TB BioSamples to NCBI assembly accessions and batch them "
            "for the datasets CLI (standalone, pandas only)."
        ),
    )
    parser.add_argument("--metadata", type=Path, required=True, help="Path to TB AMR records CSV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Assembly output directory (used for skip-existing via cache + dir scan)",
    )
    parser.add_argument(
        "--accession-map",
        type=Path,
        default=None,
        help="Write the BioSample->accession mapping TSV here",
    )
    parser.add_argument(
        "--missing-output",
        type=Path,
        default=None,
        help="Write the missing-samples sidecar TSV here",
    )
    parser.add_argument("--n", type=int, default=-1, help="Number of BioSamples (-1=all, default)")
    parser.add_argument("--output", type=Path, default=None, help="Write a single accession list file")
    parser.add_argument("--batch-dir", type=Path, default=None, help="Write batch_00, batch_01, ... here")
    parser.add_argument("--batch-size", type=int, default=100, help="Accessions per batch")

    args = parser.parse_args()
    run(
        metadata=args.metadata,
        output_dir=args.output_dir,
        accession_map_path=args.accession_map,
        missing_output_path=args.missing_output,
        n=args.n,
        batch_dir=args.batch_dir,
        batch_size=args.batch_size,
        output=args.output,
    )


if __name__ == "__main__":
    main()
