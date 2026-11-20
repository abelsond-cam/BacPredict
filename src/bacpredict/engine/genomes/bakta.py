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

Usage (one genome)::

    python -m bacpredict.engine.genomes.bakta run --contigs C.fna.gz --sample S --out-dir D \
        --tools-bin $BAKTA_ENV/bin --db $ROOT/raw/bakta_db/v5.1_full --threads 10
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import resource
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from bacpredict.engine.genomes.build_genomes import _md5, _open, _part, download

BAKTA_SOFTWARE = "v1.9.4"
BAKTA_DATABASE = "v5.1, full"
BAKTA_FLAGS = ("--keep-contig-headers", "--skip-plot")
BAKREP_API = "https://bakrep.computational.bio/api/v2/datasets/{sample}"
# What each run keeps (gzipped): the GFF3 is the annotation of record; the rest serve validation and audit.
DEFAULT_KEEP = ("gff3", "faa", "json", "txt")


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

    Returns wall seconds and the largest child process's peak RSS (GB). Raises on a non-zero exit, a missing
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
        max_rss_gb = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 2)  # KiB on Linux
        return {"seconds": seconds, "max_rss_gb": max_rss_gb}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_record(out_dir: Path, sample: str) -> Path:
    """``<out_dir>/<sample>.bakta.run.json``: what a finished run was given (contigs md5) and what it cost."""
    return out_dir / f"{sample}.bakta.run.json"


def is_current(out_dir: Path, sample: str, contigs: Path) -> bool:
    """A finished run of ``sample`` exists and was made from these exact contigs."""
    rec, gff = run_record(out_dir, sample), out_dir / f"{sample}.bakta.gff3.gz"
    if not (rec.exists() and gff.exists()):
        return False
    try:
        return json.loads(rec.read_text()).get("contigs_md5") == _md5(contigs)
    except (OSError, ValueError):
        return False


def main() -> None:
    """CLI: ``run`` one genome, or ``fetch`` one sample's BakRep files."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Bakta on one genome")
    r.add_argument("--contigs", type=Path, required=True)
    r.add_argument("--sample", required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--tools-bin", type=Path, required=True, help="bin/ of the engine/genomes `bakta` pixi env")
    r.add_argument("--db", type=Path, required=True, help="Bakta database v5.1 full")
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--keep", default=",".join(DEFAULT_KEEP), help="Bakta output extensions to keep, gzipped")
    r.add_argument("--skip-existing", action="store_true",
                   help="skip if a finished run of this sample was made from these exact contigs (md5)")
    r.add_argument("--tmp-dir", type=Path, default=None, help="Bakta's working directory root (default $TMPDIR)")
    f = sub.add_parser("fetch", help="one sample's BakRep Bakta files")
    f.add_argument("--sample", required=True)
    f.add_argument("--out-dir", type=Path, required=True)
    f.add_argument("--filetypes", default="gff3", help="comma-separated BakRep filetypes, e.g. gff3,faa,json")
    a = ap.parse_args()
    if a.cmd == "run":
        if a.skip_existing and is_current(a.out_dir, a.sample, a.contigs):
            print(json.dumps({"sample": a.sample, "skipped": "current"}))
            return
        run_record(a.out_dir, a.sample).unlink(missing_ok=True)  # a run that dies leaves no record
        tmp_root = a.tmp_dir or (Path(os.environ["TMPDIR"]) if os.environ.get("TMPDIR") else None)
        stats = run_bakta(a.contigs, a.sample, a.out_dir, a.tools_bin, a.db, a.threads, tmp_root=tmp_root,
                          keep=tuple(a.keep.split(",")))
        rec = {"sample": a.sample, "contigs": str(a.contigs), "contigs_md5": _md5(a.contigs),
               "software": BAKTA_SOFTWARE, "database": BAKTA_DATABASE,
               "amrfinderplus_db": (a.db / "amrfinderplus-db" / "latest").resolve().name,
               "threads": a.threads, "host": os.uname().nodename, **stats}
        run_record(a.out_dir, a.sample).write_text(json.dumps(rec) + "\n")
        print(json.dumps(rec))
    else:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        print(json.dumps({k: str(v) for k, v in fetch_bakrep(a.sample, a.out_dir, tuple(a.filetypes.split(","))).items()}))


if __name__ == "__main__":
    main()
