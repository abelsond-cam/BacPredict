"""Bakta release checks, the BakRep lookup, and the ours-vs-BakRep gate on a synthetic genome."""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
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
    assert good[["n_cds", "n_pseudo"]].to_numpy().tolist() == [[3, 1], [3, 1]]  # counted from the GFF3 itself
    bakta.ingest_record(ann, "S1").write_text('{"truncated')                      # a killed write → not current
    good, bad = bakta.collate(pd.DataFrame(rows), ann, fp, workers=2)
    assert "S1" in set(bad["Sample"])


# --- ingest outcomes (INGEST_STATUSES) -------------------------------------------------------------------------------

OURS = f">S1.contig00001 orig=x src=atb\n{C1}\n>S1.contig00002\n{C2}\n"


class _Bakrep:
    """BakRep as a table: ``files`` maps a sample to the gzipped GFF3 it serves; a sample not in it is a 404."""

    def __init__(self, root: Path):
        self.root, self.files, self.api_down, self.dl_down = root, {}, set(), set()
        self.asked, self.fetched = [], []

    def serve(self, sample: str, gff_text: str) -> None:
        self.files[sample] = _write(self.root / "served" / f"{sample}.gff3.gz", gff_text)

    def urls(self, sample: str) -> dict[str, str]:
        self.asked.append(sample)
        if sample in self.api_down:
            raise bakta.BakrepUnavailable(f"BakRep API {sample}: curl exit 0, HTTP 503")
        return {"gff3": f"https://x/{sample}.bakta.gff3.gz"} if sample in self.files or sample in self.dl_down else {}

    def download(self, url: str, dest: Path, **_) -> None:
        sample = url.rsplit("/", 1)[1].split(".")[0]
        self.fetched.append(sample)
        if sample in self.dl_down:
            raise subprocess.CalledProcessError(22, ["curl", "-fsSL", url])
        shutil.copyfile(self.files[sample], dest)


@pytest.fixture
def bakrep(tmp_path, monkeypatch):
    b = _Bakrep(tmp_path)
    monkeypatch.setattr(bakta, "bakrep_urls", b.urls)
    monkeypatch.setattr(bakta, "download", b.download)
    return b


def _contigs(tmp_path: Path, sample: str, text: str = OURS) -> Path:
    return _write(tmp_path / "contigs" / f"{sample}.fna.gz", text)


def test_ingest_gives_every_genome_one_outcome(tmp_path, bakrep):
    ann = tmp_path / "ann"
    roster = {}
    for s in ("OK", "ABS", "REL", "NAM", "CON", "API", "DL", "TRU", "RUN"):
        roster[s] = _contigs(tmp_path, s)
    roster["CON"] = _contigs(tmp_path, "CON", OURS.replace(C2, C2 + "A"))
    roster["MIS"] = tmp_path / "contigs" / "MIS.fna.gz"                       # our contigs are missing
    for s in ("OK", "RUN", "MIS"):
        bakrep.serve(s, _gff())
    bakrep.serve("REL", _gff().replace("v1.9.4", "v1.12.0"))
    # ATB numbering with a gap: BakRep calls our second contig contig00003
    bakrep.serve("NAM", _gff().replace("S1.contig00002", "S1.contig00003"))
    bakrep.serve("CON", _gff())
    gz = gzip.compress(_gff().encode())
    (tmp_path / "served" / "TRU.gff3.gz").write_bytes(gz[: len(gz) - 40])     # a truncated body, curl exit 0
    bakrep.files["TRU"] = tmp_path / "served" / "TRU.gff3.gz"
    bakrep.api_down.add("API")
    bakrep.dl_down.add("DL")
    ours_gff = _write(ann / "RUN.bakta.gff3.gz", _gff().replace("product=p1", "product=ours"))
    ours_md5 = bakta._md5(ours_gff)
    bakta.run_record(ann, "RUN").write_text("{}")

    res = bakta.ingest_roster(pd.DataFrame({"Sample": list(roster), "contigs": [str(p) for p in roster.values()]}),
                              ann, workers=4)
    status = res.set_index("Sample")["status"].to_dict()
    assert status == {"OK": "ingested", "ABS": "absent", "REL": "refused:release", "NAM": "refused:names",
                      "CON": "refused:content", "API": "error:transient", "DL": "error:transient",
                      "TRU": "error:transient", "RUN": "skipped:run", "MIS": "error:other"}
    assert set(status.values()) == set(bakta.INGEST_STATUSES)
    assert list(res["Sample"]) == list(roster)                                 # roster order
    assert not res["detail"].str.contains("\t|\n").any()
    # only the accepted genome has BakRep's file; our run's GFF3 is untouched and BakRep was never asked about it
    assert sorted(p.name for p in ann.glob("*.bakta.ingest.json")) == ["OK.bakta.ingest.json"]
    assert sorted(p.name for p in ann.glob("*.gff3.gz")) == ["OK.bakta.gff3.gz", "RUN.bakta.gff3.gz"]
    assert bakta._md5(ours_gff) == ours_md5 and "RUN" not in bakrep.asked
    assert not list(ann.glob(".*part"))


def _ingest_cli(monkeypatch, roster: Path, ann: Path, *extra: str) -> int:
    monkeypatch.setattr(sys, "argv", ["bakta", "ingest", "--roster", str(roster), "--out-dir", str(ann),
                                      "--workers", "2", *extra])
    with pytest.raises(SystemExit) as e:
        bakta.main()
    return e.value.code


def test_ingest_cli_exits_non_zero_only_on_errors_and_resumes(tmp_path, bakrep, monkeypatch):
    ann, status_tsv, roster = tmp_path / "ann", tmp_path / "ingest_status.tsv", tmp_path / "roster.tsv"
    bakrep.serve("OK", _gff())
    bakrep.serve("NAM", _gff().replace("S1.contig00002", "S1.contig00003"))
    rows = [("OK", _contigs(tmp_path, "OK")), ("NAM", _contigs(tmp_path, "NAM")), ("ABS", _contigs(tmp_path, "ABS"))]
    roster.write_text("".join(f"{s}\t{p}\n" for s, p in rows))
    assert _ingest_cli(monkeypatch, roster, ann, "--status-out", str(status_tsv)) == 0   # refusals are outcomes
    st = pd.read_csv(status_tsv, sep="\t", dtype=str)
    assert list(st.columns) == ["Sample", "status", "detail"]
    assert st.set_index("Sample")["status"].to_dict() == {"OK": "ingested", "NAM": "refused:names", "ABS": "absent"}

    bakrep.fetched.clear()                                                    # a re-run keeps the current record
    assert _ingest_cli(monkeypatch, roster, ann, "--status-out", str(status_tsv)) == 0
    assert "OK" not in bakrep.fetched
    assert pd.read_csv(status_tsv, sep="\t", dtype=str).set_index("Sample").loc["OK", "detail"] == "current record kept"

    bakrep.api_down.add("ABS")
    assert _ingest_cli(monkeypatch, roster, ann, "--status-out", str(status_tsv)) == 1   # an error:* fails the job
    st = pd.read_csv(status_tsv, sep="\t", dtype=str).set_index("Sample")["status"]
    assert st["ABS"] == "error:transient" and st["OK"] == "ingested"          # and is still written in full


def test_ingest_default_path_writes_what_it_always_wrote(tmp_path, bakrep, monkeypatch):
    """PsA's invocation (no --status-out): the same files, byte for byte, and the same record."""
    ann, roster = tmp_path / "ann", tmp_path / "roster.tsv"
    bakrep.serve("S1", _gff())
    contigs = _contigs(tmp_path, "S1")
    roster.write_text(f"S1\t{contigs}\n")
    assert _ingest_cli(monkeypatch, roster, ann) == 0
    assert sorted(p.name for p in ann.iterdir()) == ["S1.bakta.gff3.gz", "S1.bakta.ingest.json"]
    assert bakta._md5(ann / "S1.bakta.gff3.gz") == bakta._md5(bakrep.files["S1"])
    rec = json.loads(bakta.ingest_record(ann, "S1").read_text())
    assert set(rec) == {"sample", "origin", "url", "gff3_md5", "contigs", "contigs_md5", "software", "database",
                        "n_cds", "n_pseudo"}
    assert (rec["origin"], rec["contigs_md5"], rec["n_cds"], rec["n_pseudo"]) == ("bakrep", bakta._md5(contigs), 3, 1)
    # the exception types callers of ingest_bakrep have always caught
    with pytest.raises(LookupError):
        bakta.ingest_bakrep("ABS", contigs, ann)
    bakrep.serve("CON", _gff())
    with pytest.raises(ValueError, match="not ours"):
        bakta.ingest_bakrep("CON", _contigs(tmp_path, "CON", OURS.replace(C2, C2 + "A")), ann)
    bakrep.api_down.add("API")
    with pytest.raises(RuntimeError, match="HTTP 503"):
        bakta.ingest_bakrep("API", contigs, ann)


def test_ingest_never_replaces_our_runs_gff3(tmp_path, bakrep, monkeypatch):
    ann = tmp_path / "ann"
    contigs = _contigs(tmp_path, "S1")
    bakrep.serve("S1", _gff())
    ours = _write(ann / "S1.bakta.gff3.gz", _gff().replace("product=p1", "product=ours"))
    md5 = bakta._md5(ours)
    bakta.run_record(ann, "S1").write_text("{}")
    with pytest.raises(bakta.RunRecordExists):
        bakta.ingest_bakrep("S1", contigs, ann)
    assert bakrep.fetched == [] and bakta._md5(ours) == md5

    # our run finishing while BakRep's file downloads: its GFF3 is kept, no ingest record is written
    bakta.run_record(ann, "S1").unlink()
    served = bakrep.download

    def download_while_our_run_finishes(url, dest, **k):
        served(url, dest)
        bakta.run_record(ann, "S1").write_text("{}")

    monkeypatch.setattr(bakta, "download", download_while_our_run_finishes)
    with pytest.raises(FileExistsError):
        bakta.ingest_bakrep("S1", contigs, ann)
    assert bakta._md5(ours) == md5 and not bakta.ingest_record(ann, "S1").exists()
    assert not list(ann.glob(".*part"))


def test_ingest_status_does_not_read_any_lookup_error_as_absent():
    assert bakta.ingest_status(bakta.NotInBakrep("S: API 404")) == "absent"
    assert bakta.ingest_status(KeyError("attributes")) == "error:other"       # a changed API shape
    assert bakta.ingest_status(bakta.IngestRefused("names", "x")) == "refused:names"
    assert bakta.ingest_status(subprocess.CalledProcessError(22, ["curl"])) == "error:transient"


def test_fingerprint_cli_prints_the_run_fingerprint(tmp_path, monkeypatch, capsys):
    env, db = tmp_path / "env", tmp_path / "db"
    (env / "bin").mkdir(parents=True)
    (env / "conda-meta").mkdir()
    (env / "conda-meta" / "bakta-1.9.4-pyhdfd78af_0.json").write_text("{}")
    (db / "amrfinderplus-db" / "2024-07-22.1").mkdir(parents=True)
    (db / "amrfinderplus-db" / "latest").symlink_to("2024-07-22.1")
    (db / "version.json").write_text('{"major": 5, "minor": 1}')
    monkeypatch.setattr(sys, "argv", ["bakta", "fingerprint", "--tools-bin", str(env / "bin"), "--db", str(db)])
    bakta.main()
    assert capsys.readouterr().out.strip() == bakta.run_fingerprint(env / "bin", db)


def test_rerunning_psa_ingest_changes_no_file(tmp_path, bakrep, monkeypatch):
    """PsA's store after annotation: BakRep's accepted file (current ingest record) and a refused genome our run
    annotated. Re-running its ingest fetches nothing, rewrites nothing, and exits 0."""
    ann, roster = tmp_path / "ann", tmp_path / "roster.tsv"
    bakrep.serve("ACC", _gff())
    bakrep.serve("GAP", _gff().replace("S1.contig00002", "S1.contig00003"))
    acc, gap = _contigs(tmp_path, "ACC"), _contigs(tmp_path, "GAP")
    roster.write_text(f"ACC\t{acc}\nGAP\t{gap}\n")
    _ingest_cli(monkeypatch, roster, ann)
    _write(ann / "GAP.bakta.gff3.gz", _gff())                                  # our run of the refused genome
    bakta.run_record(ann, "GAP").write_text(json.dumps({"fingerprint": "f" * 32, "contigs_md5": bakta._md5(gap)}))
    before = {p.name: (bakta._md5(p), p.stat().st_mtime_ns) for p in ann.iterdir()}
    bakrep.asked.clear()
    bakrep.fetched.clear()
    assert _ingest_cli(monkeypatch, roster, ann) == 0
    assert {p.name: (bakta._md5(p), p.stat().st_mtime_ns) for p in ann.iterdir()} == before
    assert bakrep.asked == [] and bakrep.fetched == []
