"""Source planning: URL layout, ATB index parsing, ENA run selection, and the even-surface priority."""

from __future__ import annotations

import gzip
import ssl

import numpy as np
import pandas as pd
import pytest

import bacpredict.engine.genomes.plan_genome_sources as pgs
from bacpredict.engine.genomes.plan_genome_sources import (
    plan_genome_sources,
    portal_gff_url,
    read_atb_index,
    summarise_reads,
)


def test_portal_gff_url_layout():
    assert portal_gff_url("GCA_000480865.1").endswith(
        "/genomes/GCA/000/480/GCA_000480865.1/GCA_000480865.1_annotations.gff.gz")
    assert "/genomes/ERZ/420/499/ERZ4204997/" in portal_gff_url("ERZ4204997")      # 7-digit ERZ
    assert "/genomes/ERZ/256/317/ERZ25631748/" in portal_gff_url("ERZ25631748")
    with pytest.raises(ValueError):
        portal_gff_url("SAMN123")


def test_atb_index_keeps_latest_and_only_real_md5(tmp_path):
    p = tmp_path / "inv.csv.gz"
    with gzip.open(p, "wt") as fh:
        fh.write('"b","S1.fa.gz","100","2024-01-01T00:00:00.000Z","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",""\n')
        fh.write('"b","S1.fa.gz","120","2026-09-16T00:00:00.000Z","bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","CRC64NVME"\n')
        fh.write('"b","S2.fa.gz","300","2026-09-16T00:00:00.000Z","cccccccccccccccccccccccccccccccc-2",""\n')
        fh.write('"b","notes.txt","3","2026-09-16T00:00:00.000Z","x",""\n')
    idx = read_atb_index([p])
    assert list(idx.index) == ["S1", "S2"]
    assert idx.loc["S1", "atb_bytes"] == "120" and idx.loc["S1", "atb_md5"].startswith("b")
    assert pd.isna(idx.loc["S2", "atb_md5"])                                         # multipart etag


def _run(q, run, platform="ILLUMINA", layout="PAIRED", strategy="WGS", bases=6.6e8, files=2):
    ftp = ";".join(f"ftp.sra.ebi.ac.uk/vol1/{run}_{i}.fastq.gz" for i in range(1, files + 1))
    if files == 3:
        ftp = f"ftp.sra.ebi.ac.uk/vol1/{run}.fastq.gz;" + ftp.rsplit(";", 1)[0]
    return {"query": q, "run_accession": run, "instrument_platform": platform, "library_layout": layout,
            "library_strategy": strategy, "base_count": str(int(bases)), "fastq_ftp": ftp,
            "fastq_bytes": ";".join(["10"] * files), "fastq_md5": ";".join(["m"] * files)}


def test_summarise_reads_selects_paired_illumina_dna_and_pools_runs():
    runs = pd.DataFrame([
        _run("A", "SRR1"), _run("A", "SRR2"),                                   # pooled: 200x
        _run("B", "SRR3", strategy="RNA-Seq"),
        _run("C", "ERR4", platform="OXFORD_NANOPORE", layout="SINGLE", files=1),
        _run("D", "SRR5", files=3),                                             # unpaired file dropped
        _run("E", "SRR6", strategy="WGA", bases=3.3e7),                         # 5x
    ])
    r = summarise_reads(runs, genome_size=6.6e6)
    assert r.loc["A", "reads_depth"] == 200.0 and r.loc["A", "runs"] == "SRR1;SRR2"
    assert r.loc["A", "fastq_urls"].count("https://") == 4
    assert r.loc["B", "no_reads_reason"] == "rna_seq_only" and pd.isna(r.loc["B", "fastq_urls"])
    assert r.loc["C", "no_reads_reason"] == "long_read_only"
    assert r.loc["D", "fastq_urls"].count("https://") == 2 and "_1.fastq.gz" in r.loc["D", "fastq_urls"]
    assert r.loc["E", "reads_depth"] == 5.0
    cached = pd.concat([runs, pd.DataFrame({"query": ["F"], "run_accession": ["__NO_RUNS__"]})])
    assert "F" not in summarise_reads(cached, genome_size=6.6e6).index           # "no runs" sentinel is not a run


def _cohort():
    return pd.DataFrame({
        "Sample": ["inATB", "portal", "portal_reads", "reads", "shallow", "rna", "nolabel"],
        "n_drugs_labelled": [3, 3, 3, 3, 3, 3, 0],
        "genome_route": ["assembly", "assembly", "assembly", "reads", "reads", "reads", "reads"],
        "assembly_genotype": ["ERZ25631748", "GCA_000480865.1", "GCA_000480866.1", None, None, None, None],
        "assembly_phenotype": [None] * 7, "reads_accession": [None] * 7,
    })


def test_priority_atb_then_portal_then_shovill_then_drop():
    atb = pd.DataFrame({"atb_bytes": ["1"], "atb_md5": ["m"], "atb_last_modified": ["t"]}, index=["inATB"])
    reads = pd.DataFrame({
        "runs": ["R1", "R2", "R3", np.nan], "fastq_urls": ["u", "u", "u", np.nan], "fastq_bytes": ["1"] * 3 + [np.nan],
        "fastq_md5": ["m"] * 3 + [np.nan], "reads_depth": [50.0, 40.0, 6.0, np.nan],
        "no_reads_reason": [np.nan, np.nan, np.nan, "rna_seq_only"]}, index=["portal_reads", "reads", "shallow", "rna"])
    plan = plan_genome_sources(_cohort(), atb, reads, min_depth=10).set_index("Sample")
    assert "nolabel" not in plan.index
    assert plan.loc["inATB", "source"] == "atb" and plan.loc["inATB", "atb_url"].endswith("/inATB.fa.gz")
    assert plan.loc["portal", "source"] == "portal_gff" and plan.loc["portal", "portal_url"].endswith(".gff.gz")
    assert plan.loc["portal_reads", "source"] == "portal_gff" and plan.loc["portal_reads", "reads_depth"] == 50
    assert plan.loc["reads", "source"] == "shovill" and plan.loc["reads", "fastq_urls"] == "u"
    assert plan.loc["shallow", "source"] == "drop" and plan.loc["shallow", "drop_reason"] == "below_min_depth"
    assert plan.loc["rna", "drop_reason"] == "rna_seq_only"
    prefer = plan_genome_sources(_cohort(), atb, reads, min_depth=10, prefer_reads=True).set_index("Sample")
    assert prefer.loc["portal_reads", "source"] == "shovill" and prefer.loc["portal", "source"] == "portal_gff"


def test_ena_fault_that_fails_every_query_stops_at_the_first(tmp_path, monkeypatch):
    """A TLS fault (CSD3: no CA bundle where uv's Python looks) must stop the plan at once, with its cause."""
    calls = []

    def refuse(url, **kw):
        calls.append(url)
        raise ssl.SSLCertVerificationError("certificate verify failed: unable to get local issuer certificate")

    monkeypatch.setattr(pgs.urllib.request, "urlopen", refuse)
    monkeypatch.setattr(pgs.time, "sleep", lambda s: None)
    cache = tmp_path / "ena.tsv"
    # a cache holding an earlier run's failures: those must be retried, not trusted
    pd.DataFrame({"query": ["SAMN1", "SAMN2"], "run_accession": [pgs.QUERY_FAILED] * 2}).to_csv(
        cache, sep="\t", index=False)
    with pytest.raises(RuntimeError, match="CERTIFICATE|certificate verify failed"):
        pgs.ena_runs(["SAMN1", "SAMN2", "SAMN3"], cache, threads=2)
    assert len(calls) == 4  # the probe's retries only; the other two samples were never queried
