r"""Bakta 1.9.4 for the AST-cohort genome store — the annotation of record (PROJECT_STATE §6).

Genomes BakRep v2 holds take BakRep's files: AllTheBacteria's own Bakta run (Bakta 1.9.4, database v5.1 full),
on contigs byte-identical to ATB's and so to ours. Every other genome gets our run of the same release with the
flags BakRep's files show. This module holds what both paths share:

- the release of record, and the check every GFF3 must pass before it is used (``# Software: v1.9.4``,
  ``# Database: v5.1, full``): a BakRep re-run with a newer Bakta is being prepared upstream, so a downloaded
  file is never trusted unread;
- BakRep's per-sample lookup (one API GET names the bucket each file sits in; 404 means BakRep has no data);
- one Bakta run on one genome. Bakta's defaults, plus ``--keep-contig-headers`` (BakRep's seqids are ATB's contig
  names, which are ours, so the GFF3 coordinates sit on our contigs) and ``--skip-plot`` (output we drop).

Tools come from the ``bakta`` environment of ``engine/genomes/pixi.toml``; its ``bin/`` is put first on ``PATH``
because Bakta calls amrfinder, diamond, cmscan, tRNAscan-SE, aragorn, pilercr and blastn by name.

Both paths write ``<store>/bakta/<Sample>.bakta.gff3.gz`` beside a record of where it came from and which contigs it
sits on (``.bakta.ingest.json`` for BakRep's, ``.bakta.run.json`` for ours); ``collate`` accepts a genome only if exactly
one record is current for the contigs the cohort now holds.

Usage::

    python -m bacpredict.engine.genomes.bakta run --contigs C.fna.gz --sample S --out-dir D \
        --tools-bin $BAKTA_ENV/bin --db $ROOT/raw/bakta_db/v5.1_full --threads 8      # one genome
    python -m bacpredict.engine.genomes.bakta ingest --roster bakrep_roster.tsv --out-dir D --workers 8
    python -m bacpredict.engine.genomes.bakta collate --cohort embedding_input.csv --annotation-dir D \
        --tools-bin $BAKTA_ENV/bin --db $DB --out bakta_annotation.csv
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from bacpredict.engine.genomes.build_genomes import _md5, _part, download, read_fasta

BAKTA_SOFTWARE = "v1.9.4"
BAKTA_DATABASE = "v5.1, full"
BAKTA_FLAGS = ("--keep-contig-headers", "--skip-plot")
BAKREP_API = "https://bakrep.computational.bio/api/v2/datasets/{sample}"
# What each run keeps (gzipped): the GFF3 is the annotation of record; the rest serve validation and audit.
DEFAULT_KEEP = ("gff3", "faa", "json", "txt")


def _open(path: Path):
    """Open text, gzipped or not, by content: a download's temporary name (``….part``) carries no ``.gz``."""
    with open(path, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt") if gz else open(path)


def gff3_release(path: Path) -> tuple[str | None, str | None]:
    """The ``# Software:`` and ``# Database:`` values from a Bakta GFF3's header (None where absent)."""
    software = database = None
    with _open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if line.startswith("# Software:"):
                software = line.split(":", 1)[1].strip()
            elif line.startswith("# Database:"):
                database = line.split(":", 1)[1].strip()
    return software, database


def require_release(path: Path) -> None:
    """Raise unless ``path`` was written by the Bakta release and database of record."""
    software, database = gff3_release(path)
    if (software, database) != (BAKTA_SOFTWARE, BAKTA_DATABASE):
        raise ValueError(f"{path}: Bakta {software} / database {database}, "
                         f"not the release of record {BAKTA_SOFTWARE} / {BAKTA_DATABASE}")


def gff3_fasta(path: Path) -> dict[str, str]:
    """The ``##FASTA`` section of a GFF3 as ``{first word of the header: sequence}``."""
    out, name, seq, in_fasta = {}, None, [], False
    with _open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not in_fasta:
                in_fasta = line.startswith("##FASTA")
            elif line.startswith(">"):
                if name is not None:
                    out[name] = "".join(seq)
                name, seq = line[1:].split()[0], []
            elif line:
                seq.append(line.strip())
    if name is not None:
        out[name] = "".join(seq)
    return out


def cds_counts(path: Path) -> dict[str, int]:
    """``n_cds`` (CDS lines, sORFs included) and ``n_pseudo`` (those marked ``pseudo=True``) in a GFF3."""
    n_cds = n_pseudo = 0
    with _open(path) as fh:
        for line in fh:
            if line.startswith("##FASTA"):
                break
            parts = line.split("\t")
            if len(parts) == 9 and parts[2] == "CDS":
                n_cds += 1
                n_pseudo += "pseudo=True" in parts[8]
    return {"n_cds": n_cds, "n_pseudo": n_pseudo}


def bakrep_urls(sample: str, retries: int = 5, delay_s: float = 15) -> dict[str, str]:
    """``{filetype: url}`` for the Bakta files BakRep serves for ``sample``; ``{}`` when BakRep has no data (404).

    Any other failure is retried, then raised: an unreachable API must never read as "not in BakRep".
    """
    url = BAKREP_API.format(sample=sample)
    for attempt in range(retries + 1):
        r = subprocess.run(["curl", "-sS", "--connect-timeout", "60", "-w", "\n%{http_code}", url],
                           capture_output=True, text=True)
        body, _, code = r.stdout.rpartition("\n")
        if r.returncode == 0 and code == "404":
            return {}
        if r.returncode == 0 and code == "200":
            res = json.loads(body).get("results", [])
            return {x["attributes"]["filetype"]: x["url"] for x in res if x["attributes"].get("tool") == "bakta"}
        if attempt == retries:
            raise RuntimeError(f"BakRep API {url}: curl exit {r.returncode}, HTTP {code or '?'} {r.stderr.strip()}")
        time.sleep(delay_s)
    raise AssertionError("unreachable")


def fetch_bakrep(sample: str, out_dir: Path, filetypes: tuple[str, ...] = ("gff3",)) -> dict[str, Path]:
    """Download BakRep's Bakta files for ``sample`` → ``<out_dir>/<sample>.bakta.<filetype>.gz``.

    Raises if BakRep lacks the sample or any requested file type. The GFF3 must be of the release of record.
    """
    urls = bakrep_urls(sample)
    if not urls:
        raise LookupError(f"{sample}: not in BakRep")
    out: dict[str, Path] = {}
    for ft in filetypes:
        if ft not in urls:
            raise LookupError(f"{sample}: BakRep has no {ft} (has {sorted(urls)})")
        dest = out_dir / f"{sample}.bakta.{ft}.gz"
        tmp = _part(dest)
        download(urls[ft], tmp)
        with gzip.open(tmp, "rb") as fh:  # a truncated or non-gzip body fails here, not at use
            while fh.read(1 << 22):
                pass
        tmp.replace(dest)
        out[ft] = dest
    if "gff3" in out:
        require_release(out["gff3"])
    return out


def run_bakta(contigs: Path, sample: str, out_dir: Path, tools_bin: Path, db: Path, threads: int,
              tmp_root: Path | None = None, keep: tuple[str, ...] = DEFAULT_KEEP) -> dict:
    """Bakta on one genome → ``<out_dir>/<sample>.bakta.<ext>.gz`` for each ``keep`` extension, plus the log.

    Returns wall seconds and the largest single child process's peak RSS (GB) — not the job's: sacct's MaxRSS
    for the same runs is ~4x higher (39 GB against 8-12 GB in the gate), and that is what --mem must cover. Raises on a non-zero exit, a missing
    output, or a GFF3 not of the release of record (a wrong env or database is caught here, not downstream).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"bakta_{sample}_", dir=tmp_root))
    try:
        # CSD3 exports OMP_NUM_THREADS=1 to every job; nothing Bakta runs should be held to it
        env = dict(os.environ, PATH=f"{tools_bin}:{os.environ.get('PATH', '')}", OMP_NUM_THREADS=str(threads))
        cmd = [str(tools_bin / "bakta"), "--db", str(db), "--output", str(work / "out"), "--prefix", sample,
               "--threads", str(threads), "--tmp-dir", str(work), *BAKTA_FLAGS, "--force", str(contigs)]
        log = work / "bakta.log"
        t0 = time.time()
        with open(log, "w") as fh:
            rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        seconds = round(time.time() - t0, 1)
        with open(log, "rb") as src, gzip.open(out_dir / f"{sample}.bakta.log.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        if rc != 0:
            tail = " | ".join(log.read_text(errors="replace").strip().splitlines()[-5:])
            raise RuntimeError(f"bakta exit {rc}: {tail}")
        require_release(work / "out" / f"{sample}.gff3")
        for ext in keep:
            src_path = work / "out" / f"{sample}.{ext}"
            if not src_path.exists():
                raise FileNotFoundError(f"bakta wrote no {src_path.name}")
            dest = out_dir / f"{sample}.bakta.{ext}.gz"
            tmp = _part(dest)
            with open(src_path, "rb") as src, gzip.open(tmp, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
            tmp.replace(dest)
        max_child_rss_gb = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 2)  # KiB on Linux
        return {"seconds": seconds, "max_child_rss_gb": max_child_rss_gb}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_fingerprint(tools_bin: Path, db: Path) -> str:
    """md5 over everything but the contigs that defines a run's output.

    Bakta's flags, the tool environment's exact package set (conda-meta names carry name, version and build), the
    database release and the AMRFinderPlus database.
    """
    pkgs = sorted(p.name for p in (tools_bin.parent / "conda-meta").glob("*.json"))
    if not pkgs:
        raise FileNotFoundError(f"no conda-meta beside {tools_bin}: not a pixi/conda env bin/")
    parts = [*BAKTA_FLAGS, *pkgs, (db / "version.json").read_text(), (db / "amrfinderplus-db" / "latest").resolve().name]
    return hashlib.md5("\n".join(parts).encode()).hexdigest()


def run_record(out_dir: Path, sample: str) -> Path:
    """``<out_dir>/<sample>.bakta.run.json``: what a finished run was given (contigs md5) and what it cost."""
    return out_dir / f"{sample}.bakta.run.json"


def is_current(out_dir: Path, sample: str, contigs: Path, fingerprint: str) -> bool:
    """A finished run of ``sample`` exists, made from these exact contigs with this exact tool env, flags and db."""
    rec, gff = run_record(out_dir, sample), out_dir / f"{sample}.bakta.gff3.gz"
    if not (rec.exists() and gff.exists()):
        return False
    try:
        r = json.loads(rec.read_text())
        return r.get("fingerprint") == fingerprint and r.get("contigs_md5") == _md5(contigs)
    except (OSError, ValueError):
        return False


def _write_json(path: Path, rec: dict) -> None:
    """Write a record atomically: a killed job leaves the old record or none, never a truncated one."""
    tmp = _part(path)
    tmp.write_text(json.dumps(rec) + "\n")
    tmp.replace(path)


def ingest_record(out_dir: Path, sample: str) -> Path:
    """``<out_dir>/<sample>.bakta.ingest.json``: where an ingested BakRep GFF3 came from and which contigs it sits on."""
    return out_dir / f"{sample}.bakta.ingest.json"


def ingest_bakrep(sample: str, contigs: Path, out_dir: Path) -> dict:
    """BakRep's GFF3 for ``sample`` → ``<out_dir>/<sample>.bakta.gff3.gz``, byte for byte as BakRep serves it.

    Accepted only if it is of the release of record and its embedded ``##FASTA`` equals ``contigs`` name for name
    and base for base, so its coordinates provably sit on our contigs (a mismatch would otherwise surface only as
    proteins silently missing from the parquet). Raises ``LookupError`` if BakRep has no GFF3 for the sample.
    """
    urls = bakrep_urls(sample)
    if "gff3" not in urls:
        raise LookupError(f"{sample}: BakRep has no GFF3" + (f" (has {sorted(urls)})" if urls else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{sample}.bakta.gff3.gz"
    contigs_md5 = _md5(contigs)  # before the comparison, re-checked after: the record must certify what was compared
    tmp = _part(dest)
    try:
        download(urls["gff3"], tmp)
        require_release(tmp)
        ours = {h.split()[0]: s for h, s in read_fasta(contigs)}
        theirs = gff3_fasta(tmp)  # reads the whole file, so a truncated gzip fails here
        if theirs != ours:
            same = sum(theirs.get(k) == v for k, v in ours.items())
            raise ValueError(f"{sample}: BakRep's contigs are not ours ({same}/{len(ours)} identical by name, "
                             f"{len(theirs)} in BakRep's GFF3)")
        counts = cds_counts(tmp)
        if _md5(contigs) != contigs_md5:
            raise RuntimeError(f"{sample}: {contigs} changed during ingest")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
    rec = {"sample": sample, "origin": "bakrep", "url": urls["gff3"], "gff3_md5": _md5(dest),
           "contigs": str(contigs), "contigs_md5": contigs_md5, "software": BAKTA_SOFTWARE,
           "database": BAKTA_DATABASE, **counts}
    _write_json(ingest_record(out_dir, sample), rec)
    return rec


def _record_if_current(path: Path, contigs_md5: str, fingerprint: str | None = None) -> dict | None:
    try:
        r = json.loads(path.read_text())
    except (OSError, ValueError):  # absent or unreadable: not current, so ingest re-fetches and collate reports it
        return None
    if r.get("contigs_md5") != contigs_md5 or (fingerprint is not None and r.get("fingerprint") != fingerprint):
        return None
    return r


def collate(cohort: pd.DataFrame, annotation_dir: Path, fingerprint: str, workers: int = 16
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One Bakta GFF3 per cohort genome → ``(annotation table, problems)``.

    ``cohort`` needs ``Sample`` and ``sr_assembly_file`` (the contigs). A genome is accepted when exactly one of its
    records is current: an ingest record on the same contigs, or a run record on the same contigs made with the
    tool env, flags and database of ``fingerprint``. The table has the protein stage's columns (``Sample``,
    ``sr_assembly_file``, ``sr_gff_file``) plus ``origin``, ``n_cds``, ``n_pseudo`` (counted here, from the GFF3
    handed on) and ``contigs_md5``.
    """
    def one(row: tuple[str, str]) -> dict:
        s, contigs = row
        gff = annotation_dir / f"{s}.bakta.gff3.gz"
        try:
            md5 = _md5(Path(contigs))
        except OSError as e:
            return {"Sample": s, "problem": f"contigs unreadable: {e}"}
        recs = [r for r in (_record_if_current(ingest_record(annotation_dir, s), md5),
                            _record_if_current(run_record(annotation_dir, s), md5, fingerprint)) if r]
        if len(recs) != 1 or not gff.exists():
            have = [p.name for p in (ingest_record(annotation_dir, s), run_record(annotation_dir, s)) if p.exists()]
            what = "two current annotations" if len(recs) > 1 else f"no current annotation (records: {have or 'none'})"
            return {"Sample": s, "problem": what}
        try:
            counts = cds_counts(gff)
        except (OSError, EOFError, ValueError) as e:
            return {"Sample": s, "problem": f"GFF3 unreadable: {type(e).__name__}"}
        if counts["n_cds"] == 0:
            return {"Sample": s, "problem": "GFF3 has no CDS"}
        return {"Sample": s, "sr_assembly_file": contigs, "sr_gff_file": str(gff),
                "origin": recs[0].get("origin", "bakta_run"), **counts, "contigs_md5": md5}

    rows = list(zip(cohort["Sample"].astype(str), cohort["sr_assembly_file"].astype(str), strict=True))
    with ThreadPoolExecutor(workers) as ex:
        res = pd.DataFrame(list(ex.map(one, rows)))
    if "problem" not in res:
        res["problem"] = pd.NA
    bad = res[res["problem"].notna()][["Sample", "problem"]]
    good = res[res["problem"].isna()].drop(columns="problem")
    return good, bad


def main() -> None:
    """CLI: ``run`` one genome, ``fetch`` one sample's BakRep files, ``ingest`` a roster, ``collate`` a cohort."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Bakta on one genome")
    r.add_argument("--contigs", type=Path, required=True)
    r.add_argument("--sample", required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--tools-bin", type=Path, required=True, help="bin/ of the engine/genomes `bakta` pixi env")
    r.add_argument("--db", type=Path, required=True, help="Bakta database v5.1 full")
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--keep", default="+".join(DEFAULT_KEEP),
                   help="Bakta output extensions to keep, gzipped, joined by '+' or ','")
    r.add_argument("--skip-existing", action="store_true",
                   help="skip if a finished run of this sample used these exact contigs, tools, flags and db")
    r.add_argument("--tmp-dir", type=Path, default=None, help="Bakta's working directory root (default $TMPDIR)")
    f = sub.add_parser("fetch", help="one sample's BakRep Bakta files")
    f.add_argument("--sample", required=True)
    f.add_argument("--out-dir", type=Path, required=True)
    f.add_argument("--filetypes", default="gff3", help="comma-separated BakRep filetypes, e.g. gff3,faa,json")
    i = sub.add_parser("ingest", help="BakRep GFF3s for a roster, each checked against our contigs")
    i.add_argument("--roster", type=Path, required=True, help="TSV, no header: Sample<TAB>contigs path")
    i.add_argument("--out-dir", type=Path, required=True)
    i.add_argument("--workers", type=int, default=8, help="concurrent BakRep streams (a 76-way pull was throttled)")
    c = sub.add_parser("collate", help="one current Bakta GFF3 per cohort genome → the protein stage's input")
    c.add_argument("--cohort", type=Path, required=True, help="CSV with Sample, sr_assembly_file")
    c.add_argument("--annotation-dir", type=Path, required=True)
    c.add_argument("--tools-bin", type=Path, required=True, help="the env our runs must have used")
    c.add_argument("--db", type=Path, required=True, help="the database our runs must have used")
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        fingerprint = run_fingerprint(a.tools_bin, a.db)
        if a.skip_existing and is_current(a.out_dir, a.sample, a.contigs, fingerprint):
            print(json.dumps({"sample": a.sample, "skipped": "current"}))
            return
        run_record(a.out_dir, a.sample).unlink(missing_ok=True)  # a run that dies leaves no record
        contigs_md5 = _md5(a.contigs)  # before the run, re-checked after: the record must certify what Bakta read
        tmp_root = a.tmp_dir or (Path(os.environ["TMPDIR"]) if os.environ.get("TMPDIR") else None)
        stats = run_bakta(a.contigs, a.sample, a.out_dir, a.tools_bin, a.db, a.threads, tmp_root=tmp_root,
                          keep=tuple(a.keep.replace("+", ",").split(",")))
        if _md5(a.contigs) != contigs_md5:
            raise SystemExit(f"{a.sample}: {a.contigs} changed during the run; no record written")
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        rec = {"sample": a.sample, "contigs": str(a.contigs), "contigs_md5": contigs_md5,
               "fingerprint": fingerprint, "flags": " ".join(BAKTA_FLAGS), "tools_bin": str(a.tools_bin.resolve()),
               "db": str(a.db.resolve()), "code_sha": sha or None,
               "software": BAKTA_SOFTWARE, "database": BAKTA_DATABASE,
               "amrfinderplus_db": (a.db / "amrfinderplus-db" / "latest").resolve().name,
               "threads": a.threads, "host": os.uname().nodename,
               **cds_counts(a.out_dir / f"{a.sample}.bakta.gff3.gz"), **stats}
        _write_json(run_record(a.out_dir, a.sample), rec)
        print(json.dumps(rec))
    elif a.cmd == "fetch":
        a.out_dir.mkdir(parents=True, exist_ok=True)
        print(json.dumps({k: str(v) for k, v in fetch_bakrep(a.sample, a.out_dir, tuple(a.filetypes.split(","))).items()}))
    elif a.cmd == "ingest":
        a.out_dir.mkdir(parents=True, exist_ok=True)
        roster = pd.read_csv(a.roster, sep="\t", header=None, names=["Sample", "contigs"], dtype=str)
        if roster.empty or roster["contigs"].isna().any():
            raise SystemExit(f"{a.roster}: empty, or a row without a contigs path")

        def one(r: tuple[str, str]) -> tuple[str, str]:
            s, contigs = r
            try:
                rec = _record_if_current(ingest_record(a.out_dir, s), _md5(Path(contigs)))
                if rec and (a.out_dir / f"{s}.bakta.gff3.gz").exists():
                    return s, "current"
                ingest_bakrep(s, Path(contigs), a.out_dir)
                return s, "ok"
            except Exception as e:  # noqa: BLE001 (one genome's failure is reported, the roster carries on)
                return s, f"fail: {type(e).__name__}: {e}"

        n_fail = 0
        with ThreadPoolExecutor(a.workers) as ex:
            for k, (s, status) in enumerate(ex.map(one, zip(roster["Sample"], roster["contigs"], strict=True)), 1):
                n_fail += status.startswith("fail")
                if status.startswith("fail") or k % 100 == 0:
                    print(f"[{k}/{len(roster)}] {s}: {status}", flush=True)
        print(f"ingest: {len(roster) - n_fail} ok or current, {n_fail} failed", flush=True)
        sys.exit(1 if n_fail else 0)
    else:
        cohort = pd.read_csv(a.cohort, dtype={"Sample": str})
        good, bad = collate(cohort, a.annotation_dir, run_fingerprint(a.tools_bin, a.db))
        print(f"collate: {len(good)} of {len(cohort)} genomes have one current Bakta annotation; "
              f"origin {good['origin'].value_counts().to_dict() if len(good) else {}}")
        if len(bad):
            problems = a.out.with_name(a.out.stem + "_problems.csv")
            bad.to_csv(problems, index=False)
            a.out.unlink(missing_ok=True)  # a previous table must not outlive a failed collate
            print(f"{len(bad)} without: {bad['problem'].value_counts().to_dict()} — listed in {problems}; "
                  f"{a.out} not written (any previous one removed)")
            sys.exit(1)
        tmp = _part(a.out)
        good.to_csv(tmp, index=False)
        tmp.replace(a.out)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
