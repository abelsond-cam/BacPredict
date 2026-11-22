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
   ``--fallback-read-platforms DNBSEQ,BGISEQ`` also makes paired WGS/WGA runs of those short-read platforms usable,
   but only for a BioSample with no usable Illumina run, so every genome Illumina reads already cover is planned
   exactly as without it; the manifest then gains ``reads_platform``. Added for CABBAGE v2 (``PROJECT_STATE.md``
   §3.7, David 2026-11-22: assemble the DNBSEQ/BGISEQ genomes, ~920 of v2's drops).
4. ``ncbi_gca`` — only with ``--ncbi-fallback``: a genome that would otherwise drop, whose AST record names a GenBank
   assembly (``assembly_phenotype``, e.g. ``GCA_025835955.1``), takes that assembly from NCBI's FTP
   (``genomes/all/GCA/…/<accession>_<name>/<accession>_<name>_genomic.fna.gz``). Like ``portal_gff`` it is a submitter
   assembly, so it is the last resort: below ATB, the portal's assembly and shovill. Added for CABBAGE v2
   (``PROJECT_STATE.md`` §3.7, David 2026-11-21: "use NCBI when no reads exist"), where ~17,500 genomes have no reads in
   ENA or SRA at all; without the flag the manifest is exactly as before. Each accession's FTP directory is resolved
   once from NCBI's directory listing and cached in ``--ncbi-cache`` (an accession NCBI does not list keeps its drop,
   ``ncbi_status`` = ``not_found``).
5. ``drop`` — none of the above; ``drop_reason`` says why (``rna_seq_only``, ``long_read_only``,
   ``single_end_only``, ``below_min_depth``, ``other_strategy``, ``no_runs``).

ENA is queried **by BioSample** (the portal API's ``filereport`` accepts sample accessions and returns every
run), so a sample-accession ``reads_accession`` never needs resolving. Answers — including "no runs" — are
cached in ``--ena-cache`` so re-planning is offline and reproducible; a query that still fails after retries
stops the plan rather than becoming a ``no_runs`` drop.

Usage (CSD3 login node; ~2 min of ENA queries the first time, at 8 concurrent)::

    uv run python -m bacpredict.engine.genomes.plan_genome_sources \
        --cohort $T/cohort.csv --atb-index $ATB/*.csv.gz --genome-size 6.6e6 --min-depth 10 \
        --ena-cache $T/genomes/ena_filereport_cache.tsv --out $T/genomes/genome_manifest.csv

A multi-genus cohort gives ``--genome-size-by-genus Escherichia=5.1e6,Salmonella=4.9e6,…`` (the genus is the first
word of ``cohort.csv``'s ``species``); each genome's size then sets its read depth, and the manifest carries it as a
``genome_size`` column that the shovill build uses as ``--gsize``. A genus missing from the map falls back to
``--genome-size``; without the option the manifest is exactly as before. ``--genome-size-by-species "Genus epithet=bp,…"``
overrides the genus for a named species — for a species two label sources file under different genera (v2's
*Klebsiella aerogenes* is §3.6's *Enterobacter aerogenes*, sized as *Enterobacter*).
"""

from __future__ import annotations

import argparse
import io
import re
import ssl
import time
import urllib.error
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
NCBI_GCA_DIR = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/{d1}/{d2}/{d3}/"
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


def ncbi_listing_url(accession: str) -> str:
    """``GCA_025835955.1`` → ``https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/025/835/955/`` (the accession's parent)."""
    m = re.fullmatch(r"GCA_(\d{3})(\d{3})(\d{3})\.\d+", accession)
    if not m:
        raise ValueError(f"not a versioned GCA accession: {accession!r}")
    return NCBI_GCA_DIR.format(d1=m.group(1), d2=m.group(2), d3=m.group(3))


def ncbi_dir_from_listing(html: str, accession: str) -> str | None:
    """The ``<accession>_<assembly name>`` directory an NCBI FTP listing names for this exact accession.version."""
    found = sorted(set(re.findall(rf'href="({re.escape(accession)}_[^"/]+)/"', html)))
    if len(found) > 1:
        raise ValueError(f"{accession}: {len(found)} directories in the NCBI listing: {found}")
    return found[0] if found else None


def _ncbi_query(accession: str, retries: int = 4, raise_on_fail: bool = False) -> tuple[str, str]:
    url = ncbi_listing_url(accession)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60, context=_tls_context()) as r:
                html = r.read().decode(errors="replace")
            # a 200 that is not this directory's index (a maintenance or throttling page) must not be cached as
            # "not listed": it is retried, then fails the plan
            if f"Index of {urllib.parse.urlparse(url).path.rstrip('/')}" not in html:
                raise ValueError(f"not the NCBI index of {url}")
            return accession, ncbi_dir_from_listing(html, accession) or NO_RUNS
        except urllib.error.HTTPError as e:
            if e.code == 404:  # no directory for these digits at all
                return accession, NO_RUNS
            err = e
        except Exception as e:  # noqa: BLE001 — network: retry, then record (or raise) the failure
            err = e
        if attempt == retries - 1:
            if raise_on_fail:
                raise RuntimeError(f"NCBI listing failed for {accession} after {retries} attempts: {err!r}") from err
            return accession, QUERY_FAILED
        time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def ncbi_assembly_dirs(accessions: list[str], cache_path: Path, threads: int = 4) -> pd.Series:
    """Each accession's NCBI FTP directory name (``NO_RUNS`` where NCBI lists none), cached like the ENA answers."""
    cached = (pd.read_csv(cache_path, sep="\t", dtype=str) if cache_path.exists()
              else pd.DataFrame(columns=["accession", "ncbi_dir"]))
    cached = cached[cached["ncbi_dir"] != QUERY_FAILED]
    todo = sorted(set(accessions) - set(cached["accession"]))
    if todo:
        first = _ncbi_query(todo[0], raise_on_fail=True)
        with ThreadPoolExecutor(max_workers=threads) as ex:
            new = [first, *ex.map(_ncbi_query, todo[1:])]
        cached = pd.concat([cached, pd.DataFrame(new, columns=["accession", "ncbi_dir"])], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(cache_path, sep="\t", index=False)
    return cached.set_index("accession")["ncbi_dir"].reindex(accessions)


def apply_ncbi_fallback(plan: pd.DataFrame, ncbi_dirs: pd.Series) -> pd.DataFrame:
    """Give every ``drop`` genome whose ``assembly_phenotype`` NCBI lists the source ``ncbi_gca`` (module docstring, 4).

    Adds ``ncbi_accession``, ``ncbi_url`` (the ``_genomic.fna.gz``), ``ncbi_md5_url`` (that directory's
    ``md5checksums.txt``) and ``ncbi_status`` (``found`` / ``not_found`` for drops naming an assembly). A genome that
    takes the fallback keeps nothing of its drop but ``no_reads_reason``.
    """
    p = plan.copy()
    acc = p["assembly_phenotype"].where(p["source"].eq("drop") & p["assembly_phenotype"].notna())
    d = acc.map(ncbi_dirs)
    found = d.notna() & d.ne(NO_RUNS) & d.ne(QUERY_FAILED)
    if (failed := acc[d.eq(QUERY_FAILED)]).size:
        raise ValueError(f"{failed.size} NCBI listings failed (first: {failed.head(5).tolist()}); re-run the plan")
    base = acc[found].map(ncbi_listing_url) + d[found]
    p["ncbi_status"] = np.where(found, "found", np.where(acc.notna(), "not_found", None))
    p["ncbi_accession"] = acc.where(found)
    p["ncbi_url"] = (base + "/" + d[found] + "_genomic.fna.gz").reindex(p.index)
    p["ncbi_md5_url"] = (base + "/md5checksums.txt").reindex(p.index)
    p.loc[found, "source"] = "ncbi_gca"
    p.loc[found, "drop_reason"] = pd.NA
    return p


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


def summarise_reads(runs: pd.DataFrame, genome_size: float | pd.Series,
                    fallback_platforms: frozenset[str] | set[str] = frozenset()) -> pd.DataFrame:
    """Per BioSample: usable paired Illumina WGS/WGA runs pooled → depth and fastq URLs, or why there are none.

    ``genome_size`` is one size (bp) for every genome, or a Series of sizes indexed by BioSample.
    ``fallback_platforms``: paired WGS/WGA runs of these platforms are usable for a BioSample with no usable Illumina
    run (never pooled with Illumina runs), and a ``reads_platform`` column says which platform(s) a sample's usable
    runs are on. Empty (the default), the output is exactly as before.
    """
    r = runs[runs["run_accession"].notna() & ~runs["run_accession"].isin([QUERY_FAILED, NO_RUNS])].copy()
    files = r["fastq_ftp"].fillna("").str.split(";")
    r["n_files"] = files.map(lambda f: sum(1 for x in f if x))
    # a paired run may also carry an unpaired file; keep only the _1/_2 pair
    r["pair"] = files.map(lambda f: [x for x in f if re.search(r"_[12]\.fastq\.gz$", x)])
    paired = r["library_layout"].eq("PAIRED") & r["library_strategy"].isin(USABLE_STRATEGIES) & r["pair"].map(len).eq(2)
    illumina = paired & r["instrument_platform"].eq("ILLUMINA")
    has_illumina = illumina.groupby(r["query"]).transform("any").astype(bool)
    r["usable"] = illumina | (paired & r["instrument_platform"].isin(fallback_platforms) & ~has_illumina)
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
                           "reads_depth": round(u["base_count"].sum() / _size(genome_size, sample), 1),
                           "no_reads_reason": pd.NA,
                           "reads_platform": ";".join(sorted(u["instrument_platform"].unique()))}
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
                       "reads_depth": np.nan, "no_reads_reason": why, "reads_platform": pd.NA}
    res = pd.DataFrame.from_dict(out, orient="index")
    return res if fallback_platforms else res.drop(columns="reads_platform", errors="ignore")


def _size(genome_size: float | pd.Series, sample: str) -> float:
    return float(genome_size[sample]) if isinstance(genome_size, pd.Series) else float(genome_size)


def genome_sizes_by_genus(cohort: pd.DataFrame, by_genus: dict[str, float], default: float,
                          by_species: dict[str, float] | None = None) -> pd.Series:
    """Each BioSample's expected genome size: its species' entry, else its genus' (first word of ``species``).

    A BioSample with neither takes ``default``.
    """
    species = cohort["species"].fillna("").astype(str).str.strip()
    size = species.map(by_species or {}).fillna(species.str.split().str[0].map(by_genus))
    return size.fillna(default).set_axis(cohort["Sample"])


def plan_genome_sources(
    cohort: pd.DataFrame,
    atb: pd.DataFrame,
    reads: pd.DataFrame,
    min_depth: float,
    prefer_reads: bool = False,
    genome_size: pd.Series | None = None,
) -> pd.DataFrame:
    """Assign one source per labelled genome (see the module docstring for the priority).

    ``genome_size`` (per BioSample, from :func:`genome_sizes_by_genus`) is written as a ``genome_size`` column.
    """
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
    if genome_size is not None:
        c["genome_size"] = c["Sample"].map(genome_size)
        keep.append("genome_size")
    if "reads_platform" in c.columns:  # only from summarise_reads(..., fallback_platforms)
        keep.append("reads_platform")
    return c[keep].sort_values("Sample").reset_index(drop=True)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=Path, required=True, help="cohort.csv from build_binary_ast_labels")
    ap.add_argument("--atb-index", type=Path, nargs="+", required=True, help="ATB S3 inventory csv.gz file(s)")
    ap.add_argument("--genome-size", type=float, required=True, help="expected genome size (bp), for read depth")
    ap.add_argument("--genome-size-by-genus", default=None,
                    help="Genus=bp,... for a multi-genus cohort (genus = first word of cohort.csv species); "
                         "others fall back to --genome-size")
    ap.add_argument("--genome-size-by-species", default=None,
                    help="'Genus epithet=bp,...': a named species' size, over its genus'; needs --genome-size-by-genus")
    ap.add_argument("--min-depth", type=float, default=10.0, help="minimum pooled read depth to assemble")
    ap.add_argument("--prefer-reads", action="store_true",
                    help="assemble reads with shovill instead of using a portal (submitter) assembly when possible")
    ap.add_argument("--ena-cache", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8, help="concurrent ENA queries")
    ap.add_argument("--ncbi-fallback", action="store_true",
                    help="a genome that would drop takes its AST record's GenBank assembly from NCBI (source ncbi_gca)")
    ap.add_argument("--ncbi-cache", type=Path, default=None, help="NCBI FTP directory cache; required with --ncbi-fallback")
    ap.add_argument("--fallback-read-platforms", default="",
                    help="comma-separated short-read platforms (e.g. DNBSEQ,BGISEQ) whose paired runs are usable for a "
                         "genome with no usable Illumina run")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.genome_size_by_species and not args.genome_size_by_genus:
        raise SystemExit("--genome-size-by-species needs --genome-size-by-genus")
    cohort = pd.read_csv(args.cohort, dtype={"Sample": str, "assembly_genotype": str, "reads_accession": str})
    atb = read_atb_index(args.atb_index)
    labelled = cohort[cohort["n_drugs_labelled"] > 0]
    need_reads = labelled.loc[~labelled["Sample"].isin(atb.index), "Sample"].tolist()
    runs = ena_runs(need_reads, args.ena_cache, args.threads)
    failed = sorted(runs.loc[runs["run_accession"].eq(QUERY_FAILED), "query"])
    if failed:  # a failed query must never become a silent "no_runs" drop
        raise SystemExit(f"{len(failed)} ENA queries failed (first: {failed[:5]}); re-run plan to retry them — "
                         "no manifest written")
    sizes = None
    if args.genome_size_by_genus:
        by_genus = {k.strip(): float(v) for k, v in (x.split("=") for x in args.genome_size_by_genus.split(","))}
        by_species = ({k.strip(): float(v) for k, v in (x.split("=") for x in args.genome_size_by_species.split(","))}
                      if args.genome_size_by_species else None)
        sizes = genome_sizes_by_genus(cohort, by_genus, args.genome_size, by_species)
        genus = cohort.loc[cohort["n_drugs_labelled"] > 0, "species"].fillna("").str.split().str[0]
        unmapped = genus[~genus.isin(by_genus)].value_counts()
        if len(unmapped):
            print(f"genera without a size (use --genome-size {args.genome_size:.3g}): {unmapped.to_dict()}")
    fallback = frozenset(x.strip() for x in args.fallback_read_platforms.split(",") if x.strip())
    reads = summarise_reads(runs, sizes if sizes is not None else args.genome_size, fallback)
    plan = plan_genome_sources(cohort, atb, reads, args.min_depth, args.prefer_reads, sizes)
    if args.ncbi_fallback:
        if args.ncbi_cache is None:
            raise SystemExit("--ncbi-fallback needs --ncbi-cache")
        accs = sorted(plan.loc[plan["source"].eq("drop"), "assembly_phenotype"].dropna().unique())
        bad = [a for a in accs if not re.fullmatch(r"GCA_\d{9}\.\d+", a)]
        if bad:
            raise SystemExit(f"{len(bad)} assembly accessions are not one versioned GCA (first: {bad[:5]}); "
                             "no manifest written")
        try:
            plan = apply_ncbi_fallback(plan, ncbi_assembly_dirs(accs, args.ncbi_cache))
        except ValueError as e:
            raise SystemExit(f"{e}; no manifest written") from e
        print(f"NCBI fallback: {int(plan['ncbi_status'].eq('found').sum())} of {len(accs)} dropped genomes' "
              f"assemblies listed by NCBI → ncbi_gca; {int(plan['ncbi_status'].eq('not_found').sum())} not listed")
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
