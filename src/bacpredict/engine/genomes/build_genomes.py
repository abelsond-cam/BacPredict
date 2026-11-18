r"""SLURM-array worker: fetch or assemble one chunk of genomes, then Prodigal + mlst on each.

Per genome (rows of ``genome_manifest.csv`` whose ``source`` is in ``--sources``, sorted by Sample and cut
into ``--chunk-size`` chunks; task ``k`` takes chunk ``k``):

1. **Contigs** → ``<out-root>/contigs/<Sample>.fna.gz``, renamed ``<Sample>.contig00001`` … in input order
   so every source has the same naming (the original name is kept as a header comment):
   ``atb`` downloads the ATB object (md5-checked against the S3 etag where it is one); ``portal_gff`` takes
   the ``##FASTA`` section of the portal's annotation file; ``shovill`` downloads the paired fastq
   (size-checked against ENA), pools runs, and runs **shovill 1.1.0 with its defaults** — the same
   assembler and version as AllTheBacteria (``sw=shovill-spades/1.1.0`` in ATB headers).
2. **Genes** → ``<out-root>/prodigal/<Sample>.gff.gz``: one ``prodigal`` 2.6.3 call with **Prokka's
   flags** (``-c -m -g 11 -p single``), GFF output, on every genome whatever its source.
3. **ST** — ``mlst`` (``--scheme`` when given), and assembly statistics.

Existing contigs / GFFs are reused, so a re-run only fills gaps; mlst and the statistics are recomputed
for every genome of the chunk so the shard is always complete. One row per genome goes to
``<shard-dir>/<sources>_<task>.tsv`` — a failure is a row with ``status = fail:<reason>``, never a gap.

Usage::

    python -m bacpredict.engine.genomes.build_genomes --manifest M --sources atb,portal_gff \
        --chunk-size 40 --task-index $SLURM_ARRAY_TASK_ID --out-root $G --shard-dir $T/genomes/shards \
        --tools-bin $PIXI_ENV/bin --threads $SLURM_CPUS_PER_TASK --mlst-scheme paeruginosa
    python -m bacpredict.engine.genomes.build_genomes --manifest M --sources shovill --chunk-size 1 --count-chunks
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

PRODIGAL_PROKKA_FLAGS = ["-c", "-m", "-g", "11", "-p", "single", "-q"]
CURL_RETRIES, CURL_RETRY_DELAY_S = 6, 15
SHARD_COLUMNS = ["Sample", "source", "status", "n_contigs", "total_length", "n50", "gc", "n_cds", "mlst_scheme",
                 "st", "mlst_alleles", "contigs_path", "gff_path", "seconds"]


def select_chunk(manifest: pd.DataFrame, sources: list[str], chunk_size: int, task_index: int) -> pd.DataFrame:
    """The deterministic chunk ``task_index`` of the manifest rows whose source is in ``sources``."""
    rows = manifest[manifest["source"].isin(sources)].sort_values("Sample").reset_index(drop=True)
    return rows.iloc[task_index * chunk_size:(task_index + 1) * chunk_size]


def n_chunks(manifest: pd.DataFrame, sources: list[str], chunk_size: int) -> int:
    """How many array tasks cover ``sources`` at ``chunk_size`` genomes per task."""
    n = int(manifest["source"].isin(sources).sum())
    return (n + chunk_size - 1) // chunk_size


def _open(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """``[(header_without_>, sequence)]`` from a plain or gzipped FASTA."""
    out, name, seq = [], None, []
    with _open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if name is not None:
                    out.append((name, "".join(seq)))
                name, seq = line[1:], []
            elif line:
                seq.append(line.strip())
    if name is not None:
        out.append((name, "".join(seq)))
    return out


def contigs_from_portal_gff(path: Path) -> list[tuple[str, str]]:
    """The ``##FASTA`` section appended to an EBI AMR Portal annotation GFF."""
    out, name, seq, in_fasta = [], None, [], False
    with _open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not in_fasta:
                in_fasta = line.startswith("##FASTA")
                continue
            if line.startswith(">"):
                if name is not None:
                    out.append((name, "".join(seq)))
                name, seq = line[1:], []
            elif line:
                seq.append(line.strip())
    if name is not None:
        out.append((name, "".join(seq)))
    if not out:
        raise ValueError("no ##FASTA section")
    return out


def write_renamed_contigs(records: list[tuple[str, str]], sample: str, out_gz: Path) -> None:
    """Write ``>{sample}.contig{k:05d} orig=<first word>`` records, gzipped, atomically."""
    tmp = out_gz.with_suffix(out_gz.suffix + ".part")
    with gzip.open(tmp, "wt", compresslevel=6) as fh:
        for k, (header, seq) in enumerate(records, 1):
            fh.write(f">{sample}.contig{k:05d} orig={header.split()[0]}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i + 80] + "\n")
    tmp.replace(out_gz)


def assembly_stats(records: list[tuple[str, str]]) -> dict:
    """n_contigs, total_length, N50 and GC fraction."""
    lens = sorted((len(s) for _, s in records), reverse=True)
    total = sum(lens)
    acc, n50 = 0, 0
    for n in lens:
        acc += n
        if acc * 2 >= total:
            n50 = n
            break
    gc = sum(s.upper().count("G") + s.upper().count("C") for _, s in records)
    acgt = sum(sum(s.upper().count(b) for b in "ACGT") for _, s in records)
    return {"n_contigs": len(lens), "total_length": total, "n50": n50, "gc": round(gc / acgt, 4) if acgt else None}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, md5: str | None = None, nbytes: str | None = None) -> None:
    """Fetch with curl (retrying), then verify md5 and/or size when the source publishes them."""
    subprocess.run(["curl", "-fsSL", "--retry", str(CURL_RETRIES), "--retry-delay", str(CURL_RETRY_DELAY_S),
                    "--retry-all-errors",
                    "--connect-timeout", "60", "-o", str(dest), url], check=True)
    if nbytes and str(nbytes).isdigit() and dest.stat().st_size != int(nbytes):
        raise ValueError(f"size mismatch for {url}: {dest.stat().st_size} != {nbytes}")
    if md5 and isinstance(md5, str) and len(md5) == 32 and _md5(dest) != md5:
        raise ValueError(f"md5 mismatch for {url}")


def _split(field) -> list[str]:
    return [x for x in str(field).split(";") if x and x != "nan"] if pd.notna(field) else []


def shovill_contigs(row: pd.Series, work: Path, bin_dir: Path, threads: int, ram_gb: int) -> list[tuple[str, str]]:
    """Download the paired fastq (runs pooled) and assemble with shovill defaults."""
    urls, sizes, md5s = _split(row["fastq_urls"]), _split(row["fastq_bytes"]), _split(row["fastq_md5"])
    r1 = [u for u in urls if u.endswith("_1.fastq.gz")]
    r2 = [u for u in urls if u.endswith("_2.fastq.gz")]
    if not r1 or len(r1) != len(r2):
        raise ValueError("fastq pairs incomplete")
    pooled = {}
    for mate, group in (("R1", r1), ("R2", r2)):
        parts = []
        for u in group:
            i = urls.index(u)
            dest = work / Path(u).name
            download(u, dest, md5=md5s[i] if i < len(md5s) else None, nbytes=sizes[i] if i < len(sizes) else None)
            parts.append(dest)
        if len(parts) == 1:
            pooled[mate] = parts[0]
        else:  # concatenated gzip members are a valid gzip stream
            pooled[mate] = work / f"pooled_{mate}.fastq.gz"
            with open(pooled[mate], "wb") as out:
                for p in parts:
                    with open(p, "rb") as fh:
                        shutil.copyfileobj(fh, out)
    outdir = work / "shovill"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"}
    subprocess.run([str(bin_dir / "shovill"), "--R1", str(pooled["R1"]), "--R2", str(pooled["R2"]),
                    "--outdir", str(outdir), "--cpus", str(threads), "--ram", str(ram_gb),
                    "--tmpdir", str(work), "--force"], check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return read_fasta(outdir / "contigs.fa")


def run_prodigal(contigs_gz: Path, gff_gz: Path, bin_dir: Path, work: Path) -> int:
    """Prodigal 2.6.3 with Prokka's flags → gzipped GFF; returns the CDS count."""
    plain, gff = work / "contigs.fna", work / "prodigal.gff"
    with gzip.open(contigs_gz, "rb") as src, open(plain, "wb") as dst:
        shutil.copyfileobj(src, dst)
    subprocess.run([str(bin_dir / "prodigal"), "-i", str(plain), "-o", str(gff), "-f", "gff",
                    *PRODIGAL_PROKKA_FLAGS], check=True)
    n_cds = 0
    tmp = gff_gz.with_suffix(gff_gz.suffix + ".part")
    with open(gff) as src, gzip.open(tmp, "wt") as dst:
        for line in src:
            dst.write(line)
            if "\tCDS\t" in line:
                n_cds += 1
    tmp.replace(gff_gz)
    return n_cds


def run_mlst(contigs_gz: Path, bin_dir: Path, scheme: str | None) -> tuple[str, str, str]:
    """``(scheme, ST, alleles)`` from ``mlst``; ST is ``-`` when novel / incomplete."""
    cmd = [str(bin_dir / "mlst"), "--quiet", *(["--scheme", scheme] if scheme else []), str(contigs_gz)]
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"}
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env).stdout.strip().split("\t")
    return out[1], out[2], ";".join(out[3:])


def build_one(row: dict, out_root: Path, bin_dir: Path, tmp_root: Path, threads: int, ram_gb: int,
              mlst_scheme: str | None) -> dict:
    """Fetch/assemble, gene-call and type one genome; never raises — failures become a status."""
    t0, sample = time.time(), row["Sample"]
    contigs_gz = out_root / "contigs" / f"{sample}.fna.gz"
    gff_gz = out_root / "prodigal" / f"{sample}.gff.gz"
    res = dict.fromkeys(SHARD_COLUMNS)
    res.update(Sample=sample, source=row["source"], contigs_path=str(contigs_gz), gff_path=str(gff_gz))
    work = Path(tempfile.mkdtemp(prefix=f"{sample}_", dir=tmp_root))
    try:
        if not contigs_gz.exists():
            if row["source"] == "atb":
                dest = work / f"{sample}.fa.gz"
                download(row["atb_url"], dest, md5=row.get("atb_md5"), nbytes=row.get("atb_bytes"))
                records = read_fasta(dest)
            elif row["source"] == "portal_gff":
                dest = work / "annotations.gff.gz"
                download(row["portal_url"], dest)
                records = contigs_from_portal_gff(dest)
            elif row["source"] == "shovill":
                records = shovill_contigs(pd.Series(row), work, bin_dir, threads, ram_gb)
            else:
                raise ValueError(f"source {row['source']!r} is not buildable")
            write_renamed_contigs(records, sample, contigs_gz)
        records = read_fasta(contigs_gz)
        res.update(assembly_stats(records))
        if not gff_gz.exists():
            res["n_cds"] = run_prodigal(contigs_gz, gff_gz, bin_dir, work)
        else:
            with gzip.open(gff_gz, "rt") as fh:
                res["n_cds"] = sum(1 for line in fh if "\tCDS\t" in line)
        res["mlst_scheme"], res["st"], res["mlst_alleles"] = run_mlst(contigs_gz, bin_dir, mlst_scheme)
        res["status"] = "ok"
    except subprocess.CalledProcessError as e:
        tail = (e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else str(e.stderr or ""))[-300:]
        res["status"] = f"fail:{Path(str(e.cmd[0])).name} exit {e.returncode} {tail}".replace("\n", " ").replace("\t", " ")
    except Exception as e:  # noqa: BLE001 — one bad genome must not sink the chunk
        res["status"] = f"fail:{type(e).__name__}: {e}".replace("\n", " ").replace("\t", " ")[:400]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--sources", required=True, help="atb, portal_gff, shovill; joined by ',' or '+'")
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--task-index", type=int, default=None)
    ap.add_argument("--count-chunks", action="store_true", help="print the number of array tasks and exit")
    ap.add_argument("--out-root", type=Path, help="genome store: <out-root>/{contigs,prodigal}/")
    ap.add_argument("--shard-dir", type=Path)
    ap.add_argument("--tools-bin", type=Path, help="bin/ of the pixi env holding shovill, prodigal, mlst")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--ram-gb", type=int, default=32, help="shovill --ram")
    ap.add_argument("--tmp-dir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")))
    ap.add_argument("--mlst-scheme", default=None)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, dtype=str)
    # "+" as well as "," — sbatch --export cannot carry a comma inside a value
    sources = sorted(s.strip() for s in re.split(r"[,+]", args.sources) if s.strip())
    if args.count_chunks:
        print(n_chunks(manifest, sources, args.chunk_size))
        return
    chunk = select_chunk(manifest, sources, args.chunk_size, args.task_index)
    if chunk.empty:
        raise SystemExit(f"task {args.task_index} has no genomes ({len(sources)} sources, chunk {args.chunk_size})")
    for sub in ("contigs", "prodigal"):
        (args.out_root / sub).mkdir(parents=True, exist_ok=True)
    args.shard_dir.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = chunk.to_dict("records")
    if "shovill" in sources or args.threads == 1:  # shovill takes every thread; one genome at a time
        results = [build_one(r, args.out_root, args.tools_bin, args.tmp_dir, args.threads, args.ram_gb,
                             args.mlst_scheme) for r in rows]
    else:
        with ProcessPoolExecutor(max_workers=args.threads) as ex:
            futs = [ex.submit(build_one, r, args.out_root, args.tools_bin, args.tmp_dir, 1, args.ram_gb,
                              args.mlst_scheme) for r in rows]
            results = [f.result() for f in futs]
    shard = args.shard_dir / f"{'+'.join(sources)}_{args.task_index:05d}.tsv"
    pd.DataFrame(results, columns=SHARD_COLUMNS).to_csv(shard.with_suffix(".part"), sep="\t", index=False)
    shard.with_suffix(".part").replace(shard)
    n_ok = sum(r["status"] == "ok" for r in results)
    print(f"task {args.task_index}: {n_ok}/{len(results)} ok → {shard}")
    for r in results:
        if r["status"] != "ok":
            print(f"  {r['Sample']}: {r['status']}")


if __name__ == "__main__":
    main()
