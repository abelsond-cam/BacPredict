r"""Decide where each labelled genome's assembly comes from — one row per genome, exactly one source.

Priority follows the **even-surface** decision of record (``PROJECT_STATE.md`` §6, 2026-11-18): every
genome should be a shovill assembly wherever that is attainable.

1. ``atb`` — AllTheBacteria holds the BioSample. ATB assemblies are shovill 1.1.0 / SPAdes, fetched one
   object per sample from S3 (the EBI AMR Portal's own ERZ assemblies *are* these, byte-for-byte).
2. ``portal_gff`` — ATB does not hold it, but the portal does (its genotype-table accession). These contigs
   are whatever the submitter assembled, so they are the uneven part of the surface. ``--prefer-reads``
   replaces them with ``shovill`` wherever usable reads exist; ``reads_depth`` is recorded for every such
   genome so that choice can be made with numbers.
3. ``shovill`` — no assembly we use exists: assemble the ENA reads. Only paired Illumina WGS/WGA runs are
   usable, pooled per BioSample, at ``>= --min-depth`` (``sum(base_count) / --genome-size``).
4. ``drop`` — none of the above; ``drop_reason`` says why (``rna_seq_only``, ``long_read_only``,
   ``single_end_only``, ``below_min_depth``, ``other_strategy``, ``no_runs``).

ENA is queried **by BioSample** (the portal API's ``filereport`` accepts sample accessions and returns every
run), so a sample-accession ``reads_accession`` never needs resolving. Answers — including "no runs" — are
cached in ``--ena-cache`` so re-planning is offline and reproducible; a query that still fails after retries
stops the plan rather than becoming a ``no_runs`` drop.

Usage (CSD3 login node; ~2 min of ENA queries the first time, at 8 concurrent)::

    uv run python -m bacpredict.engine.genomes.plan_genome_sources \
        --cohort $T/cohort.csv --atb-index $ATB/*.csv.gz --genome-size 6.6e6 --min-depth 10 \
        --ena-cache $T/genomes/ena_filereport_cache.tsv --out $T/genomes/genome_manifest.csv
"""

from __future__ import annotations

import argparse
import io
import re
import ssl
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ATB_URL = "https://allthebacteria-assemblies.s3.eu-west-2.amazonaws.com/{sample}.fa.gz"
PORTAL_GFF_URL = ("https://ftp.ebi.ac.uk/pub/databases/amr_portal/genomes/{prefix}/{d1}/{d2}/{acc}/"
                  "{acc}_annotations.gff.gz")
ENA_FILEREPORT = "https://www.ebi.ac.uk/ena/portal/api/filereport"
ENA_FIELDS = ["run_accession", "sample_accession", "instrument_platform", "instrument_model", "library_layout",
              "library_strategy", "library_source", "read_count", "base_count", "fastq_ftp", "fastq_bytes",
              "fastq_md5"]
USABLE_STRATEGIES = frozenset({"WGS", "WGA"})
QUERY_FAILED, NO_RUNS = "__QUERY_FAILED__", "__NO_RUNS__"  # cache sentinels, never real runs
LONG_READ_PLATFORMS = frozenset({"OXFORD_NANOPORE", "PACBIO_SMRT"})


def portal_gff_url(accession: str) -> str:
    """``GCA_000480865.1`` → ``…/genomes/GCA/000/480/GCA_000480865.1/…gff.gz``; ``ERZ4204997`` → ``ERZ/420/499/``."""
    m = re.fullmatch(r"(GCA)_(\d{9})\.\d+|(ERZ)(\d+)", accession)
    if not m:
        raise ValueError(f"not a GCA/ERZ accession: {accession!r}")
    prefix, digits = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
    return PORTAL_GFF_URL.format(prefix=prefix, d1=digits[:3], d2=digits[3:6], acc=accession)


def read_atb_index(paths: list[Path]) -> pd.DataFrame:
    """The ATB S3 inventory (bucket, key, size, last_modified, etag, checksum_algorithm) → one row per sample.

    The etag is the object's md5 unless it carries a ``-`` (multipart upload), in which case no md5 is kept.
    """
    cols = ["bucket", "key", "size", "last_modified", "etag", "checksum_algorithm"]
    idx = pd.concat([pd.read_csv(p, header=None, names=cols, dtype=str) for p in paths], ignore_index=True)
    idx = idx[idx["key"].str.endswith(".fa.gz")].copy()
    idx["Sample"] = idx["key"].str.removesuffix(".fa.gz")
    idx["atb_md5"] = idx["etag"].where(~idx["etag"].str.contains("-", regex=False))
    idx = idx.sort_values("last_modified").drop_duplicates("Sample", keep="last")
    return idx.set_index("Sample")[["size", "atb_md5", "last_modified"]].rename(
        columns={"size": "atb_bytes", "last_modified": "atb_last_modified"})


def _tls_context() -> ssl.SSLContext:
    """Verify against certifi's CA bundle where installed.

    uv's standalone Python looks for ``/etc/ssl/cert.pem``, which RHEL (CSD3) does not have, so the default
    context fails every HTTPS request with CERTIFICATE_VERIFY_FAILED while ``curl`` works.
    """
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _ena_query(sample: str, retries: int = 4, raise_on_fail: bool = False) -> pd.DataFrame:
    params = urllib.parse.urlencode({"accession": sample, "result": "read_run", "fields": ",".join(ENA_FIELDS),
                                     "format": "tsv"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{ENA_FILEREPORT}?{params}", timeout=60, context=_tls_context()) as r:
                text = r.read().decode()
            df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str) if text.strip() else pd.DataFrame(
                columns=ENA_FIELDS)
            if df.empty:  # a header-only answer: cache the "no runs" fact so re-planning stays offline
                df = pd.DataFrame({"run_accession": [NO_RUNS]})
            df.insert(0, "query", sample)
            return df
        except Exception as e:  # network: retry, then record (or raise) the failure
            if attempt == retries - 1:
                if raise_on_fail:
                    raise RuntimeError(f"ENA filereport failed for {sample} after {retries} attempts: {e!r}") from e
                return pd.DataFrame({"query": [sample], "run_accession": [QUERY_FAILED]})
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def ena_runs(samples: list[str], cache_path: Path, threads: int = 8) -> pd.DataFrame:
    """Every ENA run for each BioSample, from the cache where present; new answers are appended to it.

    The first uncached query runs alone and raises on failure, so a fault that fails every query (TLS, proxy,
    an API change) stops the plan in seconds, with its cause, instead of after every query's retries.
    """
    cached = pd.read_csv(cache_path, sep="\t", dtype=str) if cache_path.exists() else pd.DataFrame(
        columns=["query", *ENA_FIELDS])
    cached = cached[cached["run_accession"] != QUERY_FAILED]
    todo = sorted(set(samples) - set(cached["query"]))
    if todo:
        probe = _ena_query(todo[0], raise_on_fail=True)
        with ThreadPoolExecutor(max_workers=threads) as ex:
            new = [probe, *ex.map(_ena_query, todo[1:])]
        cached = pd.concat([cached, *new], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(cache_path, sep="\t", index=False)
    return cached[cached["query"].isin(samples)]


def summarise_reads(runs: pd.DataFrame, genome_size: float) -> pd.DataFrame:
    """Per BioSample: usable paired Illumina WGS/WGA runs pooled → depth and fastq URLs, or why there are none."""
    r = runs[runs["run_accession"].notna() & ~runs["run_accession"].isin([QUERY_FAILED, NO_RUNS])].copy()
    files = r["fastq_ftp"].fillna("").str.split(";")
    r["n_files"] = files.map(lambda f: sum(1 for x in f if x))
    # a paired run may also carry an unpaired file; keep only the _1/_2 pair
    r["pair"] = files.map(lambda f: [x for x in f if re.search(r"_[12]\.fastq\.gz$", x)])
    r["usable"] = (r["instrument_platform"].eq("ILLUMINA") & r["library_layout"].eq("PAIRED")
                   & r["library_strategy"].isin(USABLE_STRATEGIES) & r["pair"].map(len).eq(2))
    r["base_count"] = pd.to_numeric(r["base_count"], errors="coerce").fillna(0)
    out = {}
    for sample, g in r.groupby("query"):
        u = g[g["usable"]]
        if len(u):
            sizes = u["fastq_bytes"].fillna("").str.split(";")
            md5s = u["fastq_md5"].fillna("").str.split(";")
            urls, byts, mds = [], [], []
            for pair, f_all, s_all, m_all in zip(u["pair"], u["fastq_ftp"].str.split(";"), sizes, md5s, strict=True):
                for p in pair:
                    i = f_all.index(p)
                    urls.append("https://" + p)
                    byts.append(s_all[i] if i < len(s_all) else "")
                    mds.append(m_all[i] if i < len(m_all) else "")
            out[sample] = {"runs": ";".join(u["run_accession"]), "fastq_urls": ";".join(urls),
                           "fastq_bytes": ";".join(byts), "fastq_md5": ";".join(mds),
                           "reads_depth": round(u["base_count"].sum() / genome_size, 1), "no_reads_reason": pd.NA}
            continue
        if g["library_strategy"].eq("RNA-Seq").all():
            why = "rna_seq_only"
        elif g["instrument_platform"].isin(LONG_READ_PLATFORMS).all():
            why = "long_read_only"
        elif g["library_layout"].eq("SINGLE").all():
            why = "single_end_only"
        else:
            why = "other_strategy"
        out[sample] = {"runs": pd.NA, "fastq_urls": pd.NA, "fastq_bytes": pd.NA, "fastq_md5": pd.NA,
                       "reads_depth": np.nan, "no_reads_reason": why}
    return pd.DataFrame.from_dict(out, orient="index")


def plan_genome_sources(
    cohort: pd.DataFrame,
    atb: pd.DataFrame,
    reads: pd.DataFrame,
    min_depth: float,
    prefer_reads: bool = False,
) -> pd.DataFrame:
    """Assign one source per labelled genome (see the module docstring for the priority)."""
    c = cohort[cohort["n_drugs_labelled"] > 0].copy()
    c = c.join(atb, on="Sample").join(reads, on="Sample")
    in_atb = c["atb_bytes"].notna()
    has_portal = c["assembly_genotype"].notna()
    has_reads = c["fastq_urls"].notna()
    deep = has_reads & (c["reads_depth"] >= min_depth)

    portal_ok = has_portal & ~(prefer_reads & deep)
    c["source"] = np.select([in_atb, portal_ok, deep], ["atb", "portal_gff", "shovill"], default="drop")
    c["drop_reason"] = np.where(
        c["source"].eq("drop"),
        np.where(has_reads, "below_min_depth", c["no_reads_reason"].fillna("no_runs")), pd.NA)
    c["atb_url"] = np.where(c["source"].eq("atb"), c["Sample"].map(lambda s: ATB_URL.format(sample=s)), pd.NA)
    c["portal_accession"] = c["assembly_genotype"].where(c["source"].eq("portal_gff"))
    c["portal_url"] = c["portal_accession"].map(portal_gff_url, na_action="ignore")
    for col in ("runs", "fastq_urls", "fastq_bytes", "fastq_md5"):
        c[col] = c[col].where(c["source"].eq("shovill"))
    keep = ["Sample", "n_drugs_labelled", "genome_route", "source", "drop_reason", "atb_url", "atb_md5", "atb_bytes",
            "portal_accession", "portal_url", "runs", "fastq_urls", "fastq_bytes", "fastq_md5", "reads_depth",
            "no_reads_reason", "assembly_genotype", "assembly_phenotype", "reads_accession"]
    return c[keep].sort_values("Sample").reset_index(drop=True)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=Path, required=True, help="cohort.csv from build_binary_ast_labels")
    ap.add_argument("--atb-index", type=Path, nargs="+", required=True, help="ATB S3 inventory csv.gz file(s)")
    ap.add_argument("--genome-size", type=float, required=True, help="expected genome size (bp), for read depth")
    ap.add_argument("--min-depth", type=float, default=10.0, help="minimum pooled read depth to assemble")
    ap.add_argument("--prefer-reads", action="store_true",
                    help="assemble reads with shovill instead of using a portal (submitter) assembly when possible")
    ap.add_argument("--ena-cache", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8, help="concurrent ENA queries")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cohort = pd.read_csv(args.cohort, dtype={"Sample": str, "assembly_genotype": str, "reads_accession": str})
    atb = read_atb_index(args.atb_index)
    labelled = cohort[cohort["n_drugs_labelled"] > 0]
    need_reads = labelled.loc[~labelled["Sample"].isin(atb.index), "Sample"].tolist()
    runs = ena_runs(need_reads, args.ena_cache, args.threads)
    failed = sorted(runs.loc[runs["run_accession"].eq(QUERY_FAILED), "query"])
    if failed:  # a failed query must never become a silent "no_runs" drop
        raise SystemExit(f"{len(failed)} ENA queries failed (first: {failed[:5]}); re-run plan to retry them — "
                         "no manifest written")
    reads = summarise_reads(runs, args.genome_size)
    plan = plan_genome_sources(cohort, atb, reads, args.min_depth, args.prefer_reads)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(args.out, index=False)

    print(plan["source"].value_counts().to_string())
    print("\ndrop reasons:\n" + plan["drop_reason"].value_counts().to_string())
    pg = plan[plan["source"].eq("portal_gff")]
    print(f"\nportal_gff genomes with usable reads >= {args.min_depth}x (would be shovill under --prefer-reads): "
          f"{int((pg['reads_depth'] >= args.min_depth).sum())} of {len(pg)}")
    print(f"wrote {len(plan)} genomes → {args.out}")


if __name__ == "__main__":
    main()
