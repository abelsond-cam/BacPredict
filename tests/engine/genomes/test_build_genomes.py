"""The build worker: chunking, contig extraction/renaming, statistics, and one genome end to end with stub tools."""

from __future__ import annotations

import gzip
import stat
from pathlib import Path

import pandas as pd
import pytest

from bacpredict.engine.genomes.build_genomes import (
    assembly_stats,
    build_one,
    contigs_from_portal_gff,
    contigs_source,
    download,
    gff_contigs_md5,
    n_chunks,
    read_fasta,
    select_chunk,
    shovill_contigs,
    write_renamed_contigs,
)

GFF = """##gff-version 3
KI1.1\tProdigal:002006\tCDS\t1\t9\t.\t+\t0\tID=x_1
##FASTA
>KI1.1 some description
ACGTACGTAC
GG
>KI2.1
GGCC
"""


def test_chunks_are_disjoint_deterministic_and_cover_the_sources():
    m = pd.DataFrame({"Sample": [f"S{i:02d}" for i in range(10)], "source": ["atb", "shovill"] * 5})
    assert n_chunks(m, ["atb"], 2) == 3
    got = [select_chunk(m, ["atb"], 2, k)["Sample"].tolist() for k in range(3)]
    assert sum(got, []) == sorted(m.loc[m.source == "atb", "Sample"]) and len(set(sum(got, []))) == 5


def test_portal_gff_fasta_section_and_renaming(tmp_path):
    p = tmp_path / "a.gff.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(GFF)
    recs = contigs_from_portal_gff(p)
    assert recs == [("KI1.1 some description", "ACGTACGTACGG"), ("KI2.1", "GGCC")]
    out = tmp_path / "S1.fna.gz"
    write_renamed_contigs(recs, "S1", out, "portal_gff")
    back = read_fasta(out)
    assert [h for h, _ in back] == ["S1.contig00001 orig=KI1.1 src=portal_gff", "S1.contig00002 orig=KI2.1 src=portal_gff"]
    assert contigs_source(out) == "portal_gff"
    assert not list(tmp_path.glob("*.part"))
    assert back[0][1] == "ACGTACGTACGG"


def test_assembly_stats():
    s = assembly_stats([("a", "G" * 50), ("b", "A" * 30), ("c", "C" * 20)])
    assert s == {"n_contigs": 3, "total_length": 100, "n50": 50, "gc": 0.7}


def _stub_tools(bin_dir: Path) -> None:
    bin_dir.mkdir()
    tools = {
        "prodigal": '#!/bin/sh\nwhile [ "$1" ]; do [ "$1" = "-o" ] && out="$2"; shift; done\n'
                    'printf "##gff-version 3\\nc\\tProdigal_v2.6.3\\tCDS\\t1\\t9\\t.\\t+\\t0\\tID=1_1\\n" > "$out"\n',
        "mlst": '#!/bin/sh\nfor a; do f="$a"; done\nprintf "%s\\tpaeruginosa\\t235\\tacsA(38)\\taroE(11)\\n" "$f"\n',
    }
    for name, body in tools.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IEXEC)


def test_build_one_portal_genome_end_to_end_and_failure_is_a_row(tmp_path, monkeypatch):
    monkeypatch.setattr("bacpredict.engine.genomes.build_genomes.CURL_RETRIES", 0)
    src = tmp_path / "src.gff.gz"
    with gzip.open(src, "wt") as fh:
        fh.write(GFF)
    bin_dir = tmp_path / "bin"
    _stub_tools(bin_dir)
    out_root, tmp = tmp_path / "store", tmp_path / "tmp"
    for d in (out_root / "contigs", out_root / "prodigal", tmp):
        d.mkdir(parents=True)
    row = {"Sample": "S1", "source": "portal_gff", "portal_url": src.as_uri()}
    res = build_one(row, out_root, bin_dir, tmp, 1, 8, "paeruginosa")
    assert res["status"] == "ok", res["status"]
    assert res["st"] == "235" and res["n_cds"] == 1 and res["n_contigs"] == 2
    assert (out_root / "contigs" / "S1.fna.gz").exists() and (out_root / "prodigal" / "S1.gff.gz").exists()
    assert gff_contigs_md5(out_root / "prodigal" / "S1.gff.gz") is not None
    again = build_one(row | {"portal_url": "file:///nonexistent"}, out_root, bin_dir, tmp, 1, 8, "paeruginosa")
    assert again["status"] == "ok"                                   # same source: reused, not refetched
    moved = build_one(row | {"source": "atb", "atb_url": "file:///nonexistent"}, out_root, bin_dir, tmp, 1, 8,
                      "paeruginosa")
    assert moved["status"].startswith("fail:")                      # re-planned source: rebuilt, never relabelled
    assert not (out_root / "contigs" / "S1.fna.gz").exists()
    bad = build_one({"Sample": "S2", "source": "portal_gff", "portal_url": "file:///nonexistent"},
                    out_root, bin_dir, tmp, 1, 8, None)
    assert bad["status"].startswith("fail:") and bad["Sample"] == "S2"
    assert list(tmp.iterdir()) == []                                  # work dirs cleaned up


def test_download_retries_every_failure_without_flags_old_curl_rejects(tmp_path, monkeypatch):
    """CSD3 ships curl 7.61, which exits 2 on ``--retry-all-errors``; the retry loop must be ours."""
    bin_dir, calls = tmp_path / "bin", tmp_path / "calls"
    bin_dir.mkdir()
    # a curl 7.61 stand-in: unknown flag → exit 2; otherwise fail twice (as a throttled EBI would), then succeed
    (bin_dir / "curl").write_text(
        '#!/bin/sh\nfor a; do [ "$a" = "--retry-all-errors" ] && exit 2; done\n'
        f'echo x >> "{calls}"\n'
        f'[ "$(wc -l < "{calls}")" -lt 3 ] && {{ echo "curl: (7) Failed to connect" >&2; exit 7; }}\n'
        'while [ "$1" ]; do [ "$1" = "-o" ] && out="$2"; shift; done\nprintf hello > "$out"\n')
    (bin_dir / "curl").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:/usr/bin:/bin")
    monkeypatch.setattr("bacpredict.engine.genomes.build_genomes.CURL_RETRY_DELAY_S", 0)
    dest = tmp_path / "f"
    download("https://example.org/f", dest, md5="5d41402abc4b2a76b9719d911017c592", nbytes="5")
    assert dest.read_text() == "hello" and len(calls.read_text().split()) == 3


def _stub_shovill(bin_dir: Path, contigs: str) -> Path:
    """A shovill stand-in that records its arguments and OMP_NUM_THREADS, and writes ``contigs`` as its assembly."""
    record = bin_dir / "shovill.args"
    (bin_dir / "shovill").write_text(
        f'#!/bin/sh\necho "OMP=$OMP_NUM_THREADS $*" > "{record}"\n'
        'while [ "$1" ]; do [ "$1" = "--outdir" ] && out="$2"; shift; done\nmkdir -p "$out"\n'
        f'printf "{contigs}" > "$out/contigs.fa"\necho "[shovill] It contains 0 contigs" > "$out/shovill.log"\n')
    (bin_dir / "shovill").chmod(0o755)
    return record


def _fastq_row(tmp_path: Path) -> dict:
    urls = []
    for mate in (1, 2):
        f = tmp_path / f"R_{mate}.fastq.gz"
        with gzip.open(f, "wt") as fh:
            fh.write("@r\nACGT\n+\nIIII\n")
        urls.append(f.as_uri())
    return {"fastq_urls": ";".join(urls), "fastq_bytes": "", "fastq_md5": ""}


def test_shovill_gets_genome_size_and_every_thread(tmp_path, monkeypatch):
    """CSD3 exports OMP_NUM_THREADS=1, and shovill's own size estimate collapses on low-depth reads."""
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    bin_dir, work = tmp_path / "bin", tmp_path / "work"
    bin_dir.mkdir()
    work.mkdir()
    record = _stub_shovill(bin_dir, ">NODE_1\\nACGTACGT\\n")
    out = shovill_contigs(pd.Series(_fastq_row(tmp_path)), work, bin_dir, threads=8, ram_gb=48, genome_size=6.6e6)
    assert out == [("NODE_1", "ACGTACGT")]
    args = record.read_text().split()
    assert args[0] == "OMP=8"
    assert args[args.index("--gsize") + 1] == "6600000"


def test_empty_shovill_assembly_is_an_error_with_its_log(tmp_path):
    bin_dir, work = tmp_path / "bin", tmp_path / "work"
    bin_dir.mkdir()
    work.mkdir()
    _stub_shovill(bin_dir, "")
    with pytest.raises(RuntimeError, match="no contigs.*It contains 0 contigs"):
        shovill_contigs(pd.Series(_fastq_row(tmp_path)), work, bin_dir, threads=2, ram_gb=8, genome_size=6.6e6)
