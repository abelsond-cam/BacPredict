"""Bakta release checks, the BakRep lookup, and the ours-vs-BakRep gate on a synthetic genome."""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from bacpredict.engine.genomes import bakta
from bacpredict.engine.genomes.compare_bakta_to_bakrep import compare_one, gate, read_gff3

# contig00001: ATG AAA CCC GGG TTT TAA (+, MKPGF) then, on the minus strand, the reverse complement of
# ATG GAA TGG TAA (MEW); contig00002 carries a pseudogene.
C1 = "ATGAAACCCGGGTTTTAA" + "CC" + "TTACCATTCCAT" + "GG"
C2 = "ATGTTTTAAGG"
HEADER = "##gff-version 3\n# Annotated with Bakta\n# Software: v1.9.4\n# Database: v5.1, full\n"


def _gff(cds_start: int = 1) -> str:
    rows = [
        ("S1.contig00001", "region", 1, len(C1), "+", ".", "ID=S1.contig00001"),
        ("S1.contig00001", "CDS", cds_start, 18, "+", "0", "ID=L_1;locus_tag=L_00005;product=p1"),
        ("S1.contig00001", "CDS", 21, 32, "-", "0", "ID=L_2;locus_tag=L_00010;product=p2"),
        ("S1.contig00002", "CDS", 1, 9, "+", "0", "ID=L_3;locus_tag=L_00015;product=p3;pseudo=True"),
    ]
    body = "".join(f"{s}\tProdigal\t{t}\t{a}\t{b}\t.\t{st}\t{ph}\t{at}\n" for s, t, a, b, st, ph, at in rows)
    return HEADER + body + f"##FASTA\n>S1.contig00001\n{C1}\n>S1.contig00002\n{C2}\n"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        fh.write(text)
    return path


def _genome(gate_dir: Path, side: str, gff: str, amr_gene: str = "blaX") -> None:
    _write(gate_dir / side / "S1.bakta.gff3.gz", gff)
    # as Bakta 1.9.4 writes it: the pseudogene's translation is in the .faa too
    _write(gate_dir / side / "S1.bakta.faa.gz", ">L_00005 p1\nMKPGF\n>L_00010 p2\nMEW\n>L_00015 p3\nMF\n")
    feats = [{"type": "cds", "contig": "S1.contig00001", "start": 1, "stop": 18, "strand": "+",
              "expert": [{"type": "amrfinder", "gene": amr_gene, "product": "beta-lactamase"}]}]
    _write(gate_dir / side / "S1.bakta.json.gz", json.dumps({"features": feats}))


@pytest.fixture
def gate_dir(tmp_path):
    contigs = _write(tmp_path / "contigs" / "S1.fna.gz", f">S1.contig00001 orig=x src=atb\n{C1}\n>S1.contig00002\n{C2}\n")
    _genome(tmp_path, "ours", _gff())
    (tmp_path / "ours" / "S1.bakta.run.json").write_text(json.dumps({"seconds": 1.0, "max_rss_gb": 0.1}))
    row = pd.Series({"Sample": "S1", "group": "entero", "species": "E. coli", "contigs_path": str(contigs)})
    return tmp_path, row


def test_release_of_record_is_required(tmp_path):
    good = _write(tmp_path / "a.gff3.gz", _gff())
    assert bakta.gff3_release(good) == ("v1.9.4", "v5.1, full")
    bakta.require_release(good)
    newer = _write(tmp_path / "b.gff3.gz", _gff().replace("v1.9.4", "v1.12.0"))
    with pytest.raises(ValueError, match="v1.12.0"):
        bakta.require_release(newer)


def test_read_gff3_splits_features_from_fasta(tmp_path):
    df, fasta = read_gff3(_write(tmp_path / "a.gff3.gz", _gff()))
    assert (df["type"] == "CDS").sum() == 3 and df["start"].dtype.kind == "i"
    assert fasta == {"S1.contig00001": C1, "S1.contig00002": C2}


def test_identical_runs_pass_every_gated_check(gate_dir):
    d, row = gate_dir
    _genome(d, "bakrep", _gff())
    out, diffs = compare_one(row, d)
    assert out["contigs_equal_bakrep"] and out["embedding_input_equal"]
    assert out["cds_ours"] == 3 and out["pseudo_ours"] == 1 and out["prot_ours_n"] == 2
    assert out["prot_ours_vs_faa_diff"] == 0 and out["faa_ours_pseudo_n"] == 1 and out["amr_diff"] == 0 and not diffs
    assert all(n_bad == 0 for _, n_bad, _ in gate(pd.DataFrame([out])))


def test_a_moved_gene_call_fails_the_gate(gate_dir):
    d, row = gate_dir
    _genome(d, "bakrep", _gff(cds_start=4), amr_gene="blaY")
    out, diffs = compare_one(row, d)
    assert out["cds_only_ours"] == 1 and out["cds_only_bakrep"] == 1
    assert not out["embedding_input_equal"] and out["amr_diff"] == 2
    failing = {name for name, n_bad, _ in gate(pd.DataFrame([out])) if n_bad}
    assert {"G3 CDS sets equal", "G4 embedding input equal", "G5 extractor = Bakta .faa (BakRep)"} <= failing
    assert {"embedding_input", "amr_only_ours"} <= {x["kind"] for x in diffs}


def test_a_faa_protein_the_extractor_did_not_produce_fails_g5(gate_dir):
    d, row = gate_dir
    _genome(d, "bakrep", _gff())
    with gzip.open(d / "ours" / "S1.bakta.faa.gz", "at") as fh:
        fh.write(">L_00020 p4\nMQQ\n")
    out, _ = compare_one(row, d)
    assert out["faa_ours_not_extracted"] == 1 and out["faa_ours_pseudo_n"] == 1
    failing = {name for name, n_bad, _ in gate(pd.DataFrame([out])) if n_bad}
    assert failing == {"G5 extractor = Bakta .faa (ours)"}


def test_bakrep_404_means_absent_and_other_failures_raise(monkeypatch):
    def fake(code, rc=0):
        return lambda *a, **k: subprocess.CompletedProcess(a, rc, stdout=f'{{"results": []}}\n{code}', stderr="")
    monkeypatch.setattr(bakta.subprocess, "run", fake("404"))
    assert bakta.bakrep_urls("SAMX") == {}
    monkeypatch.setattr(bakta.subprocess, "run", fake("503"))
    with pytest.raises(RuntimeError, match="HTTP 503"):
        bakta.bakrep_urls("SAMX", retries=1, delay_s=0)


def test_a_run_is_current_only_for_the_same_contigs_tools_and_db(tmp_path):
    env, db = tmp_path / "env", tmp_path / "db"
    (env / "bin").mkdir(parents=True)
    (env / "conda-meta").mkdir()
    (env / "conda-meta" / "bakta-1.9.4-pyhdfd78af_0.json").write_text("{}")
    (db / "amrfinderplus-db" / "2024-07-22.1").mkdir(parents=True)
    (db / "amrfinderplus-db" / "latest").symlink_to("2024-07-22.1")
    (db / "version.json").write_text('{"major": 5, "minor": 1}')
    contigs = _write(tmp_path / "c.fna.gz", f">S1.contig00001\n{C1}\n")
    out = tmp_path / "out"
    _write(out / "S1.bakta.gff3.gz", _gff())
    fp = bakta.run_fingerprint(env / "bin", db)
    bakta.run_record(out, "S1").write_text(json.dumps({"fingerprint": fp, "contigs_md5": bakta._md5(contigs)}))
    assert bakta.is_current(out, "S1", contigs, fp)
    (env / "conda-meta" / "pyrodigal-3.7.1-py310h7c10099_1.json").write_text("{}")
    assert not bakta.is_current(out, "S1", contigs, bakta.run_fingerprint(env / "bin", db))
    other = _write(tmp_path / "d.fna.gz", f">S1.contig00001\n{C2}\n")
    assert not bakta.is_current(out, "S1", other, fp)


def _bakrep_serves(monkeypatch, gff_text: str) -> None:
    monkeypatch.setattr(bakta, "bakrep_urls", lambda s: {"gff3": f"https://x/{s}.bakta.gff3.gz"})
    monkeypatch.setattr(bakta, "download", lambda url, dest, **k: _write(Path(dest), gff_text))


def test_ingest_keeps_bakreps_file_only_on_our_contigs(tmp_path, monkeypatch):
    _bakrep_serves(monkeypatch, _gff())
    ours = _write(tmp_path / "c" / "S1.fna.gz", f">S1.contig00001 orig=x src=atb\n{C1}\n>S1.contig00002\n{C2}\n")
    rec = bakta.ingest_bakrep("S1", ours, tmp_path / "ann")
    assert rec["n_cds"] == 3 and rec["n_pseudo"] == 1 and rec["contigs_md5"] == bakta._md5(ours)
    assert (tmp_path / "ann" / "S1.bakta.gff3.gz").exists() and bakta.ingest_record(tmp_path / "ann", "S1").exists()
    other = _write(tmp_path / "c" / "S2.fna.gz", f">S1.contig00001\n{C1}\n>S1.contig00002\n{C2}A\n")
    with pytest.raises(ValueError, match="not ours"):
        bakta.ingest_bakrep("S2", other, tmp_path / "ann")
    assert not list((tmp_path / "ann").glob("S2*")) and not list((tmp_path / "ann").glob(".*part"))


def test_collate_takes_exactly_one_current_record_per_genome(tmp_path, monkeypatch):
    _bakrep_serves(monkeypatch, _gff())
    ann, fp = tmp_path / "ann", "f" * 32
    rows = []
    for s in ("S1", "S2", "S3", "S4"):
        c = _write(tmp_path / "c" / f"{s}.fna.gz", f">S1.contig00001\n{C1}\n>S1.contig00002\n{C2}\n")
        rows.append({"Sample": s, "sr_assembly_file": str(c)})
    bakta.ingest_bakrep("S1", Path(rows[0]["sr_assembly_file"]), ann)                  # BakRep's
    for s, f in (("S2", fp), ("S3", "0" * 32)):                                          # ours; S3 from another env
        _write(ann / f"{s}.bakta.gff3.gz", _gff())
        bakta.run_record(ann, s).write_text(json.dumps(
            {"fingerprint": f, "contigs_md5": bakta._md5(Path(rows[1]["sr_assembly_file"])), "n_cds": 3, "n_pseudo": 1}))
    good, bad = bakta.collate(pd.DataFrame(rows), ann, fp, workers=2)
    assert good.set_index("Sample")["origin"].to_dict() == {"S1": "bakrep", "S2": "bakta_run"}
    assert set(bad["Sample"]) == {"S3", "S4"}
    assert list(good.columns[:3]) == ["Sample", "sr_assembly_file", "sr_gff_file"]
