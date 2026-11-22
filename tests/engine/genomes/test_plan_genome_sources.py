"""Source planning: URL layout, ATB index parsing, ENA run selection, and the even-surface priority."""

from __future__ import annotations

import gzip
import ssl

import numpy as np
import pandas as pd
import pytest

import bacpredict.engine.genomes.plan_genome_sources as pgs
from bacpredict.engine.genomes.plan_genome_sources import (
    NO_RUNS,
    QUERY_FAILED,
    apply_ncbi_fallback,
    genome_sizes_by_genus,
    ncbi_dir_from_listing,
    ncbi_listing_url,
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


def test_fallback_read_platforms_only_for_genomes_without_usable_illumina_reads():
    runs = pd.DataFrame([
        _run("A", "SRR1"), _run("A", "SRR2", platform="DNBSEQ"),               # Illumina wins; never pooled
        _run("G", "SRR7", platform="DNBSEQ"), _run("G", "SRR8", platform="BGISEQ"),
        _run("H", "SRR9", platform="DNBSEQ", layout="SINGLE", files=1),
        _run("I", "SRR10", platform="ION_TORRENT"),
        _run("J", "SRR11", platform="DNBSEQ", strategy="RNA-Seq"),
    ])
    before = summarise_reads(runs, genome_size=6.6e6)
    assert "reads_platform" not in before.columns                             # without the option: as before
    assert before.loc["G", "no_reads_reason"] == "other_strategy" and pd.isna(before.loc["G", "fastq_urls"])
    r = summarise_reads(runs, genome_size=6.6e6, fallback_platforms=frozenset({"DNBSEQ", "BGISEQ"}))
    pd.testing.assert_series_equal(r.loc["A", before.columns], before.loc["A"])  # an Illumina genome is unchanged
    assert r.loc["A", "runs"] == "SRR1" and r.loc["A", "reads_platform"] == "ILLUMINA"
    assert r.loc["G", "runs"] == "SRR7;SRR8" and r.loc["G", "reads_depth"] == 200.0
    assert r.loc["G", "reads_platform"] == "BGISEQ;DNBSEQ"
    assert r.loc["H", "no_reads_reason"] == "single_end_only" and pd.isna(r.loc["H", "reads_platform"])
    assert r.loc["I", "no_reads_reason"] == "other_strategy"                  # a platform not named stays unusable
    assert r.loc["J", "no_reads_reason"] == "rna_seq_only"
    plan = plan_genome_sources(pd.DataFrame({
        "Sample": ["A", "G"], "n_drugs_labelled": [2, 2], "genome_route": ["reads"] * 2,
        "assembly_genotype": [None] * 2, "assembly_phenotype": [None] * 2, "reads_accession": [None] * 2}),
        pd.DataFrame(columns=["atb_bytes", "atb_md5", "atb_last_modified"]), r, min_depth=10).set_index("Sample")
    assert plan.loc["G", "source"] == "shovill" and plan.loc["G", "reads_platform"] == "BGISEQ;DNBSEQ"


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


def test_genome_size_by_genus_sets_depth_and_a_manifest_column():
    runs = pd.DataFrame([_run("A", "SRR1"), _run("B", "SRR2")])            # the same bases for both
    cohort = pd.DataFrame({"Sample": ["A", "B", "C"], "species": ["Escherichia coli", "Proteus mirabilis", None]})
    sizes = pgs.genome_sizes_by_genus(cohort, {"Escherichia": 5.0e6, "Proteus": 4.0e6}, default=6.6e6)
    assert sizes.to_dict() == {"A": 5.0e6, "B": 4.0e6, "C": 6.6e6}
    r = summarise_reads(runs, sizes)
    assert r.loc["B", "reads_depth"] == pytest.approx(r.loc["A", "reads_depth"] * 5.0 / 4.0, rel=1e-2)
    c = _cohort().assign(species="Escherichia coli")
    s = pgs.genome_sizes_by_genus(c, {"Escherichia": 5.0e6}, default=6.6e6)
    atb = pd.DataFrame({"atb_bytes": ["1"], "atb_md5": ["m"], "atb_last_modified": ["t"]}, index=["inATB"])
    plan = plan_genome_sources(c, atb, pd.DataFrame(columns=["runs", "fastq_urls", "fastq_bytes", "fastq_md5",
                                                             "reads_depth", "no_reads_reason"]), 10, genome_size=s)
    assert plan["genome_size"].eq(5.0e6).all()
    assert "genome_size" not in plan_genome_sources(c, atb, pd.DataFrame(columns=["runs", "fastq_urls", "fastq_bytes",
                                                    "fastq_md5", "reads_depth", "no_reads_reason"]), 10).columns


LISTING = """<a href="/genomes/all/GCA/025/835/">Parent</a>
<a href="GCA_025835955.1_ASM2583595v1/">GCA_025835955.1_ASM2583595v1/</a>
<a href="GCA_025835955.10_ASMother/">GCA_025835955.10_ASMother/</a>
<a href="GCA_025835955.2_ASM2583595v2/">GCA_025835955.2_ASM2583595v2/</a>"""


def test_ncbi_directory_of_one_exact_accession_version():
    assert ncbi_listing_url("GCA_025835955.1") == "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/025/835/955/"
    assert ncbi_dir_from_listing(LISTING, "GCA_025835955.1") == "GCA_025835955.1_ASM2583595v1"  # never .10
    assert ncbi_dir_from_listing(LISTING, "GCA_025835955.2") == "GCA_025835955.2_ASM2583595v2"
    assert ncbi_dir_from_listing(LISTING, "GCA_025835955.3") is None
    with pytest.raises(ValueError):
        ncbi_listing_url("ERZ123")


def test_ncbi_fallback_only_rescues_drops_whose_assembly_ncbi_lists():
    plan = pd.DataFrame({
        "Sample": ["a", "b", "c", "d"], "source": ["drop", "drop", "atb", "drop"],
        "drop_reason": ["no_runs", "other_strategy", None, "no_runs"],
        "assembly_phenotype": ["GCA_000000001.1", "GCA_000000002.1", "GCA_000000003.1", None]})
    dirs = pd.Series({"GCA_000000001.1": "GCA_000000001.1_A", "GCA_000000002.1": NO_RUNS})
    p = apply_ncbi_fallback(plan, dirs).set_index("Sample")
    assert p.loc["a", "source"] == "ncbi_gca" and pd.isna(p.loc["a", "drop_reason"])
    assert p.loc["a", "ncbi_url"] == ("https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/000/001/GCA_000000001.1_A/"
                                      "GCA_000000001.1_A_genomic.fna.gz")
    assert p.loc["a", "ncbi_md5_url"].endswith("/GCA_000000001.1_A/md5checksums.txt")
    assert p.loc["b", "source"] == "drop" and p.loc["b", "ncbi_status"] == "not_found"
    assert p.loc["c", "source"] == "atb" and pd.isna(p.loc["c", "ncbi_url"])   # a buildable genome is never moved
    assert p.loc["d", "source"] == "drop" and pd.isna(p.loc["d", "ncbi_status"])
    with pytest.raises(ValueError, match="re-run the plan"):
        apply_ncbi_fallback(plan, pd.Series({"GCA_000000001.1": QUERY_FAILED}))


def test_ncbi_dirs_are_cached(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pgs, "_ncbi_query", lambda a, **kw: calls.append(a) or (a, f"{a}_X"))
    cache = tmp_path / "ncbi.tsv"
    assert pgs.ncbi_assembly_dirs(["GCA_000000001.1"], cache)["GCA_000000001.1"] == "GCA_000000001.1_X"
    assert pgs.ncbi_assembly_dirs(["GCA_000000001.1"], cache)["GCA_000000001.1"] == "GCA_000000001.1_X"
    assert calls == ["GCA_000000001.1"]


def test_a_species_size_wins_over_its_genus():
    c = pd.DataFrame({"Sample": ["a", "b", "c"], "species": ["Klebsiella aerogenes", "Klebsiella pneumoniae", "Hafnia alvei"]})
    s = genome_sizes_by_genus(c, {"Klebsiella": 5.6e6}, 5.0e6, {"Klebsiella aerogenes": 4.9e6})
    assert s.to_dict() == {"a": 4.9e6, "b": 5.6e6, "c": 5.0e6}


def test_a_200_that_is_not_the_ncbi_index_is_a_failure_not_a_not_listed(monkeypatch):
    class Page:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.body.encode()

    monkeypatch.setattr(pgs.time, "sleep", lambda s: None)
    monkeypatch.setattr(pgs.urllib.request, "urlopen", lambda *a, **kw: Page("<html>Service maintenance</html>"))
    assert pgs._ncbi_query("GCA_025835955.1") == ("GCA_025835955.1", QUERY_FAILED)
    index = "<title>Index of /genomes/all/GCA/025/835/955</title>" + LISTING
    monkeypatch.setattr(pgs.urllib.request, "urlopen", lambda *a, **kw: Page(index))
    assert pgs._ncbi_query("GCA_025835955.1") == ("GCA_025835955.1", "GCA_025835955.1_ASM2583595v1")
    assert pgs._ncbi_query("GCA_025835955.3") == ("GCA_025835955.3", NO_RUNS)
