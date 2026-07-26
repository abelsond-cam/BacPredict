#!/usr/bin/env python3
"""Plan TB assembly downloads (ATB primary, NCBI fallback).

Standalone helper for run_download_assemblies.sh. Runs under the `ncbi-datasets`
micromamba env (ncbi-datasets-cli + pandas only; no project imports, no uv).

Sources, in priority order:

1. AllTheBacteria (ATB) v0.2 - short-read assemblies for ~2.7M public bacterial
   samples, keyed by INSDC BioSample (SAM*). Direct per-sample S3 download:
       https://allthebacteria-assemblies.s3.eu-west-2.amazonaws.com/<BIOSAMPLE>.fa.gz
   Used for every BioSample that appears in the ATB file_list.

2. NCBI (RefSeq + GenBank) - covers BioSamples that aren't in ATB. Useful in
   particular for long-read complete genomes in RefSeq that ATB doesn't hold.
   Resolved via NCBI Entrez E-utilities (esearch + esummary on db=assembly) and
   downloaded via the `datasets` CLI by GCF_/GCA_ accession.

Flow:

1. Read the TB AMR records CSV; deduplicate to unique SAM*-prefixed BioSamples.
2. Skip-existing: drop BioSamples whose <BIOSAMPLE>.fa.gz is already on disk.
3. Load (and cache locally) the ATB file_list TSV; intersect with our BioSamples
   to split into in_atb / not_in_atb sets.
4. For not_in_atb, run the NCBI Entrez resolver to find the best GCF_/GCA_
   accession per BioSample (prefer GCF_ > GCA_, higher assembly_level, more
   recent release_date).
5. Emit two batch directories:
     - <atb-batch-dir>/batch_NN  - one BioSample per line for `curl` on S3
     - <ncbi-batch-dir>/batch_NN - one accession per line for `datasets download`
6. Write sidecars next to the output directory:
     - manifest_<timestamp>.tsv          per-BioSample: source, filename,
                                          ncbi_accession (if applicable)
     - biosample_to_accession_<timestamp>.tsv  NCBI-only: BioSample -> accession
     - missing_samples_<timestamp>.tsv   BioSamples in neither ATB nor NCBI

Never mutates the input CSV.
"""

import argparse
import gzip
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

# ATB v0.2 hosts every assembly as <BIOSAMPLE>.fa.gz on AWS S3.
ATB_S3_BASE = "https://allthebacteria-assemblies.s3.eu-west-2.amazonaws.com"
# file_list.all.latest.tsv.gz under AllTheBacteria/Assembly on OSF.
# Discoverable by downloading https://osf.io/download/r6gcp (the project-wide
# meta-index) and grepping for filename 'file_list.all.latest.tsv.gz'.
ATB_FILE_LIST_URL = "https://osf.io/download/69a040c86a4dd653508ac769/"
ATB_FILE_LIST_NAME = "_atb_file_list.tsv.gz"


# ── input loading ─────────────────────────────────────────────────────────────

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


def existing_biosample_fastas(output_dir: Path) -> set[str]:
    """Set of BioSample IDs whose <BIOSAMPLE>.fa.gz is already on disk (and non-empty)."""
    if not output_dir.is_dir():
        return set()
    out: set[str] = set()
    for p in output_dir.glob("*.fa.gz"):
        try:
            if p.stat().st_size > 0:
                out.add(p.name[:-len(".fa.gz")])
        except OSError:
            continue
    return out


# ── ATB index ─────────────────────────────────────────────────────────────────

def _http_get_bytes(url: str, timeout: int = 600) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read()


def load_atb_biosamples(cache_path: Path) -> set[str]:
    """Return the set of BioSamples present in ATB.

    Downloads file_list.all.latest.tsv.gz to `cache_path` if it doesn't already
    exist, then parses BioSamples from the `sample` column (the file lists one
    row per BioSample assembly; columns: sample, sylph_species, filename_in_tar_xz,
    tar_xz, tar_xz_url, tar_xz_md5, tar_xz_size_MB).
    """
    if not cache_path.exists():
        print(f"Downloading ATB file_list to {cache_path}...", file=sys.stderr)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(_http_get_bytes(ATB_FILE_LIST_URL))
        print(f"  cached ({cache_path.stat().st_size / 1e6:.1f} MB)", file=sys.stderr)
    else:
        print(f"Using cached ATB file_list: {cache_path}", file=sys.stderr)

    print("Parsing ATB file_list (one row per BioSample; ~2.7M rows)...", file=sys.stderr)
    # The gzipped file expands to ~580 MB; usecols=[0] keeps it light.
    with gzip.open(cache_path, "rt") as fh:
        df = pd.read_csv(fh, sep="\t", usecols=[0], low_memory=False, dtype=str)
    sample_col = df.columns[0]
    samples = df[sample_col].astype(str).str.strip()
    mask = samples.str.startswith("SAM")
    biosamples = set(samples[mask])
    print(f"  ATB has {len(biosamples):,} BioSamples in {cache_path.name}", file=sys.stderr)
    return biosamples


# ── NCBI Entrez resolver (kept from earlier - used for the ATB fallback only) ─

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _entrez_get(endpoint: str, params: dict) -> bytes:
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params = {**params, "api_key": api_key}
    url = f"{EUTILS}/{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def _entrez_sleep() -> None:
    time.sleep(0.11 if os.environ.get("NCBI_API_KEY") else 0.34)


def _esearch_assembly_uids(biosamples: list[str], batch_size: int = 100) -> set[str]:
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
    """Map BioSamples to assembly metadata via NCBI Entrez.

    Honours NCBI_API_KEY for higher rate limits (10 req/s vs 3 req/s).
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


def extract_biosample_id(record: dict) -> str | None:
    """Pull the BioSample accession out of a datasets assembly record (``None`` if absent)."""
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
    """Prefer GCF_ over GCA_, then higher assembly_level, then most recent date."""
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

    Returns (rows, missing).
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


# ── output writing ────────────────────────────────────────────────────────────

def write_batches(
    items: list[str],
    batch_dir: Path,
    batch_size: int,
    label: str,
) -> int:
    """Write items as batch_00, batch_01, ... files under batch_dir.

    Returns the number of batch files written.
    """
    batch_dir.mkdir(parents=True, exist_ok=True)
    num_batches = (len(items) + batch_size - 1) // batch_size if items else 0
    width = max(2, len(str(max(num_batches - 1, 0))))
    for i in range(num_batches):
        chunk = items[i * batch_size : (i + 1) * batch_size]
        (batch_dir / f"batch_{i:0{width}d}").write_text(
            "\n".join(chunk) + ("\n" if chunk else "")
        )
    print(f"Wrote {num_batches} {label} batch files to {batch_dir} ({len(items):,} items)",
          file=sys.stderr)
    return num_batches


def write_mapping_tsv(rows: list[dict], path: Path) -> None:
    """Write the biosample→assembly mapping TSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["biosample", "assembly_accession", "source", "assembly_level", "release_date"]
    pd.DataFrame(rows, columns=cols).to_csv(path, sep="\t", index=False)


def write_manifest_tsv(rows: list[dict], path: Path) -> None:
    """Write the download manifest TSV (one row per file to fetch)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["biosample", "source", "filename", "ncbi_accession"]
    pd.DataFrame(rows, columns=cols).to_csv(path, sep="\t", index=False)


def write_missing_sidecar(
    missing_biosamples: list[str],
    full_df: pd.DataFrame,
    path: Path,
) -> None:
    """Write the missing-BioSamples sidecar TSV (samples with no assembly found)."""
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


# ── orchestration ─────────────────────────────────────────────────────────────

def run(
    metadata: Path,
    output_dir: Path,
    atb_batch_dir: Path,
    ncbi_batch_dir: Path,
    manifest_path: Path | None,
    accession_map_path: Path | None,
    missing_output_path: Path | None,
    atb_file_list_path: Path,
    n: int,
    batch_size: int,
    skip_atb: bool = False,
    skip_ncbi: bool = False,
    atb_biosamples_provider=load_atb_biosamples,
    ncbi_resolver=resolve_biosamples_via_entrez,
) -> None:
    """Plan a two-tier ATB+NCBI download.

    The two `*_provider` / `*_resolver` arguments are injection points for tests.
    """
    full_df, biosamples = load_biosamples(metadata)

    # Skip-existing: <BIOSAMPLE>.fa.gz already on disk
    existing = existing_biosample_fastas(output_dir) if output_dir is not None else set()
    if existing:
        before = len(biosamples)
        biosamples = [b for b in biosamples if b not in existing]
        print(f"Skip-existing: dropped {before - len(biosamples):,} BioSamples "
              f"with existing .fa.gz on disk", file=sys.stderr)

    if n >= 0:
        biosamples = biosamples[:n]
        print(f"Limiting to first {n} BioSamples", file=sys.stderr)

    # Tier 1: ATB
    if skip_atb:
        in_atb: list[str] = []
        not_in_atb = list(biosamples)
        print("ATB step disabled by flag; routing everything to NCBI fallback", file=sys.stderr)
    else:
        atb_present = atb_biosamples_provider(atb_file_list_path)
        in_atb = [b for b in biosamples if b in atb_present]
        not_in_atb = [b for b in biosamples if b not in atb_present]
        print(f"ATB coverage: {len(in_atb):,} / {len(biosamples):,} BioSamples "
              f"({100 * len(in_atb) / max(1, len(biosamples)):.1f}%); "
              f"{len(not_in_atb):,} routed to NCBI fallback", file=sys.stderr)

    write_batches(in_atb, atb_batch_dir, batch_size, label="ATB")

    # Tier 2: NCBI fallback for not_in_atb
    if skip_ncbi:
        ncbi_rows: list[dict] = []
        ncbi_missing = list(not_in_atb)
        print("NCBI step disabled by flag; not_in_atb BioSamples will be reported as missing",
              file=sys.stderr)
    else:
        records = ncbi_resolver(not_in_atb) if not_in_atb else []
        ncbi_rows, ncbi_missing = parse_records_to_mapping(records, not_in_atb)
        print(f"NCBI resolver: {len(ncbi_rows):,} resolved, {len(ncbi_missing):,} unresolved",
              file=sys.stderr)

    if accession_map_path is not None:
        write_mapping_tsv(ncbi_rows, accession_map_path)
        print(f"Wrote accession map: {accession_map_path}", file=sys.stderr)

    # Deduplicate NCBI accessions for the datasets CLI batch input
    seen: set[str] = set()
    ncbi_accessions: list[str] = []
    for row in ncbi_rows:
        acc = row["assembly_accession"]
        if acc and acc not in seen:
            seen.add(acc)
            ncbi_accessions.append(acc)
    write_batches(ncbi_accessions, ncbi_batch_dir, batch_size, label="NCBI")

    # Manifest: one row per BioSample we expect to end up on disk
    manifest_rows: list[dict] = []
    for bs in in_atb:
        manifest_rows.append({
            "biosample": bs,
            "source": "atb",
            "filename": f"{bs}.fa.gz",
            "ncbi_accession": "",
        })
    for row in ncbi_rows:
        manifest_rows.append({
            "biosample": row["biosample"],
            "source": f"ncbi-{row['source']}",   # ncbi-refseq or ncbi-genbank
            "filename": f"{row['biosample']}.fa.gz",
            "ncbi_accession": row["assembly_accession"],
        })
    if manifest_path is not None:
        write_manifest_tsv(manifest_rows, manifest_path)
        print(f"Wrote manifest: {manifest_path} ({len(manifest_rows):,} rows)",
              file=sys.stderr)

    if missing_output_path is not None:
        write_missing_sidecar(ncbi_missing, full_df, missing_output_path)
        print(f"Wrote missing-samples sidecar: {missing_output_path} "
              f"({len(ncbi_missing):,} rows)", file=sys.stderr)


def main() -> None:
    """CLI entry point: plan TB assembly downloads from ATB (primary) + NCBI (fallback)."""
    parser = argparse.ArgumentParser(
        description=(
            "Plan TB assembly downloads from ATB (primary) and NCBI (fallback). "
            "Writes batch files + manifest + missing-samples sidecar."
        ),
    )
    parser.add_argument("--metadata", type=Path, required=True, help="TB AMR records CSV")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Final assembly output directory (used for skip-existing and as the "
                             "default ATB file_list cache location)")
    parser.add_argument("--atb-batch-dir", type=Path, required=True,
                        help="Write ATB batch_NN files (BioSamples per line) here")
    parser.add_argument("--ncbi-batch-dir", type=Path, required=True,
                        help="Write NCBI batch_NN files (accessions per line) here")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Per-BioSample source/filename manifest TSV")
    parser.add_argument("--accession-map", type=Path, default=None,
                        help="BioSample -> NCBI accession TSV (used by post-processing to "
                             "rename downloaded files to <BIOSAMPLE>.fa.gz)")
    parser.add_argument("--missing-output", type=Path, default=None,
                        help="Missing-samples sidecar TSV (in neither ATB nor NCBI)")
    parser.add_argument("--atb-file-list", type=Path, default=None,
                        help=f"Path to cached ATB file_list (default: <output-dir>/{ATB_FILE_LIST_NAME})")
    parser.add_argument("--n", type=int, default=-1, help="Limit to first N BioSamples")
    parser.add_argument("--batch-size", type=int, default=100, help="Items per batch file")
    parser.add_argument("--skip-atb", action="store_true",
                        help="Skip the ATB tier (route everything to NCBI fallback)")
    parser.add_argument("--skip-ncbi", action="store_true",
                        help="Skip the NCBI fallback (only ATB)")

    args = parser.parse_args()
    atb_file_list = args.atb_file_list or (args.output_dir / ATB_FILE_LIST_NAME)
    run(
        metadata=args.metadata,
        output_dir=args.output_dir,
        atb_batch_dir=args.atb_batch_dir,
        ncbi_batch_dir=args.ncbi_batch_dir,
        manifest_path=args.manifest,
        accession_map_path=args.accession_map,
        missing_output_path=args.missing_output,
        atb_file_list_path=atb_file_list,
        n=args.n,
        batch_size=args.batch_size,
        skip_atb=args.skip_atb,
        skip_ncbi=args.skip_ncbi,
    )


if __name__ == "__main__":
    main()
