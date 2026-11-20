r"""Check a cohort's protein-parquet and ESM-C stores against its annotation table, at ~68k genomes.

The check stages of the multi-species ESM-C drivers (``apps/entero/scripts/build_entero_embeddings.sh``; the PsA
driver keeps the same logic inline). Every count is against the cohort table, never against the files that exist:
the protein extractor logs a failed genome and exits 0, so a count of existing parquets would shrink with the
cohort and still read complete.

The cohort table is ``genomes/bakta_annotation.csv`` (``engine/genomes/bakta.py collate``): ``Sample``,
``sr_assembly_file`` (contigs), ``sr_gff_file`` (the Bakta GFF3), ``n_cds``, ``n_pseudo`` and ``contigs_md5``. Each
of those five columns is required (outside ``--presence-only``): a table without one fails, by name, for every
genome, rather than silently skipping the check that column feeds. A listed genome fails on:

- no parquet, an unreadable one, or one older than its GFF3 (built before the genome was re-annotated);
- a protein count other than CDS minus pseudogenes. A contig-name mismatch between GFF3 and contigs drops
  proteins silently, and every other check passes it;
- contigs changed (or gone) since annotation (``contigs_md5``);
- with ``--esm-dir``: no ``.pt``, an unreadable one (not a zip archive), or one older than its parquet; and, on a
  seeded sample of ``--n-sampled`` genomes, a ``protein_embeddings`` tensor whose protein axis is not
  ``min(proteins, --max-proteins)``, whose width is not 960, whose dtype is not ``--save-dtype``, or holding NaN.

With ``--qc-cohort`` the table must also hold exactly the QC table's genomes. Per-genome checks (stat, parquet
read, contigs md5, zip check) run in a process pool of ``--workers``; the sampled ``.pt`` loads run here after.
``--presence-only`` instead reads each directory's listing once (no per-file stat) and checks only that every
listed genome has its files: a cheap guard for a login node before an array is submitted.

Exit status is 1 on any problem or an empty sample set. ``--report OUT`` writes every (Sample, problem) pair to
OUT and, only on PASS, ``OUT.pass``: the md5 of the cohort table's bytes, the very bytes that were checked (a stale
``OUT.pass`` is removed first). A driver can then refuse to go on unless the latest check passed on the table as it
is now (``build_entero_embeddings.sh submit``).

Usage::

    python -m bacpredict.engine.embedding.check_cohort_store --cohort genomes/bakta_annotation.csv \
        --qc-cohort genomes/embedding_input.csv --protein-dir processed/protein_sequences \
        [--esm-dir processed/esm] [--samples-file esm_samples.txt [--n 10]] \
        --max-proteins 9000 --save-dtype bfloat16 --workers 32 --report problems.csv
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import random
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow.parquet as papq

from bacpredict.engine.embedding.sample_list import read_samples_file
from bacpredict.engine.genomes.build_genomes import _md5

PARQUET_SUFFIX = "_protein_sequences.parquet"
PT_SUFFIX = "_esm_embeddings.pt"
ESM_DIM = 960  # the pinned ESM-C (ESM++ small) width

DUPLICATE = "Sample duplicated in the cohort table"
NOT_IN_COHORT = "listed sample not in the cohort table"
NO_PARQUET = "no parquet"
GFF_MISSING = "GFF3 missing"
PARQUET_OLDER = "parquet older than its GFF"
UNREADABLE_PARQUET = "unreadable parquet (delete it and rerun proteins)"
COUNT_MISMATCH = "protein count != CDS - pseudogenes"
CONTIGS_UNREADABLE = "contigs unreadable"
CONTIGS_CHANGED = "contigs changed since annotation"
NO_EMBEDDING = "no embedding"
UNREADABLE_EMBEDDING = "unreadable embedding"
EMBEDDING_OLDER = "embedding older than its parquet"
SAMPLED_MISMATCH = "sampled embedding mismatch"
# the columns the per-genome checks need; outside --presence-only a missing one is a named failure, never a skip
REQUIRED_COLUMNS = ("sr_gff_file", "sr_assembly_file", "n_cds", "n_pseudo", "contigs_md5")
_ORDER = (DUPLICATE, NOT_IN_COHORT, NO_PARQUET, GFF_MISSING, PARQUET_OLDER, UNREADABLE_PARQUET, COUNT_MISMATCH,
          CONTIGS_UNREADABLE, CONTIGS_CHANGED, NO_EMBEDDING, UNREADABLE_EMBEDDING, EMBEDDING_OLDER, SAMPLED_MISMATCH)


@dataclass
class StoreCheck:
    """What :func:`check_cohort_store` found.

    Attributes
    ----------
    samples : list of str
        The genomes checked, in order.
    problems : dict of str to list of str
        Problem → the genomes that have it (sorted); only problems that occurred.
    n_proteins : dict of str to int
        Protein count of every readable parquet.
    with_parquet, with_embedding : int
        Genomes with a parquet, and with a ``.pt`` (0 without an ESM directory).
    count_checked, md5_checked : int
        Genomes whose protein count, and whose contigs md5, could be compared.
    sampled : list of str
        One line per sampled ``.pt`` loaded.
    """

    samples: list[str]
    problems: dict[str, list[str]] = field(default_factory=dict)
    n_proteins: dict[str, int] = field(default_factory=dict)
    with_parquet: int = 0
    with_embedding: int = 0
    count_checked: int = 0
    md5_checked: int = 0
    sampled: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if any genome was checked and none has a problem."""
        return bool(self.samples) and not any(self.problems.values())


def protein_count(parquet: Path) -> int:
    """Proteins in a genome's parquet: the summed lengths of row 0's per-contig lists, read without ``to_pylist``.

    ``protein_sequence`` is ``list<list<string>>`` (contigs, then proteins), one row per genome; the embedder reads
    row 0 only.
    """
    col = papq.read_table(parquet, columns=["protein_sequence"]).column(0).slice(0, 1)
    return sum(len(chunk.flatten().flatten()) for chunk in col.chunks)


def _check_one(task: tuple) -> tuple[str, int | None, bool, list[str]]:
    """One genome's file checks → ``(sample, protein count or None, has a .pt, problems)``."""
    sample, pq, gff, contigs, expect, contigs_md5, pt = task
    problems: list[str] = []
    n = None
    has_pt = False
    try:
        pq_mtime = pq.stat().st_mtime
    except FileNotFoundError:
        pq_mtime = None
        problems.append(NO_PARQUET)
    if pq_mtime is not None:
        if gff is not None:
            try:
                if pq_mtime < Path(gff).stat().st_mtime:
                    problems.append(PARQUET_OLDER)
            except OSError:
                problems.append(GFF_MISSING)
        try:
            n = protein_count(pq)
        except Exception:  # noqa: BLE001 (any failure to read is the finding; the check goes on)
            problems.append(UNREADABLE_PARQUET)
        else:
            if expect is not None and n != expect:
                problems.append(COUNT_MISMATCH)
    if contigs_md5 is not None and contigs is None:  # no contigs path to hash (e.g. no sr_assembly_file column)
        problems.append(CONTIGS_UNREADABLE)
    elif contigs_md5 is not None:
        try:
            if _md5(Path(contigs)) != contigs_md5:
                problems.append(CONTIGS_CHANGED)
        except OSError:
            problems.append(CONTIGS_UNREADABLE)
    if pt is not None and pq_mtime is not None:
        try:
            pt_mtime = pt.stat().st_mtime
        except FileNotFoundError:
            problems.append(NO_EMBEDDING)
        else:
            has_pt = True
            if not zipfile.is_zipfile(pt):
                problems.append(UNREADABLE_EMBEDDING)
            if pt_mtime < pq_mtime:
                problems.append(EMBEDDING_OLDER)
    return sample, n, has_pt, problems


def _check_batch(batch: list[tuple]) -> list[tuple[str, int | None, bool, list[str]]]:
    return [_check_one(t) for t in batch]


def _sampled_embedding(sample: str, pt: Path, n_prot: int, max_proteins: int, save_dtype: str) -> tuple[bool, str]:
    """Load one ``.pt`` and check its ``protein_embeddings`` shape, width, dtype and NaN → ``(ok, report line)``."""
    import torch  # here, not at import: the pool's workers never need it

    try:
        d = torch.load(pt, map_location="cpu", weights_only=True)
        pe = d["protein_embeddings"]
        ok = (pe.ndim == 3 and pe.shape[1] == min(n_prot, max_proteins) and pe.shape[2] == ESM_DIM
              and pe.dtype == getattr(torch, save_dtype) and not bool(torch.isnan(pe.float()).any()))
        line = (f"{sample}: {tuple(pe.shape)} {pe.dtype} proteins={n_prot} "
                f"contigs={int(d['contig_ids'].max()) + 1} {'ok' if ok else 'MISMATCH'}")
    except Exception as e:  # noqa: BLE001 (an unloadable file is a mismatch, reported with its error)
        ok, line = False, f"{sample}: unloadable ({type(e).__name__}: {e}) MISMATCH"
    return ok, line


def missing_column(col: str) -> str:
    """The problem a cohort table without ``col`` is reported under."""
    return f"cohort table has no {col} column"


def _listing(directory: Path, suffix: str) -> set[str]:
    """Sample names of ``directory``'s ``<Sample><suffix>`` files, from one directory read (no per-file stat)."""
    with os.scandir(directory) as it:
        return {e.name[: -len(suffix)] for e in it if e.name.endswith(suffix)}


def _finish(result: StoreCheck, found: dict[str, list[str]]) -> StoreCheck:
    order = [k for k in _ORDER if k in found] + sorted(k for k in found if k not in _ORDER)
    result.problems = {k: sorted(found[k]) for k in order if found[k]}
    return result


def check_cohort_store(cohort: pd.DataFrame, protein_dir: Path, samples: list[str], *, esm_dir: Path | None = None,
                       qc_samples: set[str] | None = None, qc_name: str = "the QC table", max_proteins: int = 9000,
                       save_dtype: str = "bfloat16", workers: int = 1, n_sampled: int = 10, seed: int = 0,
                       presence_only: bool = False, batch_size: int = 64) -> StoreCheck:
    """Check ``samples`` of a cohort table against the protein store and, optionally, the ESM-C store.

    Parameters
    ----------
    cohort : pandas.DataFrame
        The annotation table: ``Sample`` and :data:`REQUIRED_COLUMNS`. Outside ``presence_only`` a missing column
        is the problem ``"cohort table has no <column> column"`` for every genome (its check cannot run).
    protein_dir : Path
        Holds ``<Sample>_protein_sequences.parquet``.
    samples : list of str
        The genomes to check (the whole cohort, or the first N of an ESM array's list).
    esm_dir : Path, optional
        Holds ``<Sample>_esm_embeddings.pt``; omit to check proteins only.
    qc_samples : set of str, optional
        The QC table's genomes; the cohort table must hold exactly these.
    qc_name : str
        How to name the QC table in its problem.
    max_proteins, save_dtype : int, str
        What the embeddings were written with (``generate_embeddings --max-proteins/--save-dtype``).
    workers : int
        Processes for the per-genome checks; 1 runs them in this process.
    n_sampled, seed : int
        How many ``.pt`` files to load and check, drawn with this seed.
    presence_only : bool
        Only check that every genome has its files, from one listing per directory.
    batch_size : int
        Genomes per pool task.

    Returns
    -------
    StoreCheck
    """
    found: dict[str, list[str]] = {}
    sample_col = cohort["Sample"].astype(str)
    found[DUPLICATE] = sorted(set(sample_col[sample_col.duplicated()]))
    c = cohort.assign(Sample=sample_col).drop_duplicates("Sample").set_index("Sample")
    if qc_samples is not None:
        found[f"cohort differs from {qc_name}"] = sorted(set(qc_samples) ^ set(c.index))
    found[NOT_IN_COHORT] = [s for s in samples if s not in c.index]
    result = StoreCheck(samples=list(samples))

    if presence_only:
        have = _listing(protein_dir, PARQUET_SUFFIX)
        found[NO_PARQUET] = [s for s in samples if s not in have]
        result.with_parquet = len(samples) - len(found[NO_PARQUET])
        if esm_dir is not None:
            have_pt = _listing(esm_dir, PT_SUFFIX)
            found[NO_EMBEDDING] = [s for s in samples if s in have and s not in have_pt]
            result.with_embedding = sum(s in have_pt for s in samples)
        return _finish(result, found)

    for col in REQUIRED_COLUMNS:
        if col not in c.columns:
            found[missing_column(col)] = list(samples)
    sub = c.reindex(samples)

    def column(name: str) -> list:
        if name not in sub.columns:
            return [None] * len(samples)
        return [None if pd.isna(v) else v for v in sub[name]]

    if {"n_cds", "n_pseudo"} <= set(sub.columns):
        expect = [None if pd.isna(v) else int(v) for v in sub["n_cds"] - sub["n_pseudo"]]
    else:
        expect = [None] * len(samples)
    md5s = column("contigs_md5")
    tasks = [(s, protein_dir / f"{s}{PARQUET_SUFFIX}", gff, contigs, n, md5,
              None if esm_dir is None else esm_dir / f"{s}{PT_SUFFIX}")
             for s, gff, contigs, n, md5 in zip(samples, column("sr_gff_file"), column("sr_assembly_file"), expect,
                                                md5s, strict=True)]
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    if workers > 1 and len(batches) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = [r for batch in ex.map(_check_batch, batches) for r in batch]
    else:
        rows = [r for batch in batches for r in _check_batch(batch)]

    for s, n, has_pt, problems in rows:
        for p in problems:
            found.setdefault(p, []).append(s)
        if n is not None:
            result.n_proteins[s] = n
        result.with_embedding += has_pt
    result.with_parquet = len(samples) - len(found.get(NO_PARQUET, []))
    result.count_checked = sum(1 for s, n in zip(samples, expect, strict=True)
                               if n is not None and s in result.n_proteins)
    result.md5_checked = sum(m is not None for m in md5s)

    if esm_dir is not None and n_sampled > 0:
        bad = set(found.get(UNREADABLE_EMBEDDING, []))
        good = sorted(s for s, n, has_pt, _ in rows if has_pt and n is not None and s not in bad)
        found[SAMPLED_MISMATCH] = []
        for s in random.Random(seed).sample(good, min(n_sampled, len(good))):
            ok, line = _sampled_embedding(s, esm_dir / f"{s}{PT_SUFFIX}", result.n_proteins[s], max_proteins,
                                          save_dtype)
            result.sampled.append(line)
            if not ok:
                found[SAMPLED_MISMATCH].append(s)
    return _finish(result, found)


def main(argv: list[str] | None = None) -> int:
    """CLI; returns the exit status (0 only if every listed genome passes)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cohort", type=Path, required=True, help="the annotation table (bakta_annotation.csv)")
    ap.add_argument("--protein-dir", type=Path, required=True)
    ap.add_argument("--esm-dir", type=Path, default=None, help="omit to check the proteins only")
    ap.add_argument("--samples-file", type=Path, default=None,
                    help="check these genomes (sorted, unique; an ESM array's list) instead of the whole cohort")
    ap.add_argument("--n", type=int, default=None, help="only the first N genomes (of the list, or of the cohort)")
    ap.add_argument("--qc-cohort", type=Path, default=None,
                    help="a table whose Samples the cohort table must equal (embedding_input.csv)")
    ap.add_argument("--max-proteins", type=int, default=9000)
    ap.add_argument("--save-dtype", choices=["float32", "bfloat16"], default="bfloat16")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--n-sampled", type=int, default=10, help=".pt files to load and check (default 10)")
    ap.add_argument("--presence-only", action="store_true",
                    help="only check every genome has its files, from one listing per directory")
    ap.add_argument("--report", type=Path, default=None,
                    help="CSV of every (Sample, problem), written even if empty; on PASS also <report>.pass, the "
                    "md5 of the cohort table checked")
    a = ap.parse_args(argv)

    marker = None
    if a.report is not None:  # never leave a PASS marker from an earlier run beside this run's report
        marker = a.report.with_name(a.report.name + ".pass")
        marker.unlink(missing_ok=True)
    # read once: the marker's md5 is of the bytes that were checked, not of a table rewritten mid-check
    cohort_bytes = a.cohort.read_bytes()
    cohort = pd.read_csv(io.BytesIO(cohort_bytes), dtype={"Sample": str})
    if a.samples_file is not None:
        try:
            samples = read_samples_file(a.samples_file)
        except (OSError, ValueError) as e:
            ap.error(f"--samples-file: {e}")
        scope = a.samples_file.name
    else:
        samples = sorted(set(cohort["Sample"].dropna()))
        scope = a.cohort.name
    if a.n is not None:
        samples, scope = samples[: a.n], f"first {a.n} of {scope}"
    qc = None
    if a.qc_cohort is not None:
        qc = set(pd.read_csv(a.qc_cohort, dtype={"Sample": str}, usecols=["Sample"])["Sample"])

    r = check_cohort_store(cohort, a.protein_dir, samples, esm_dir=a.esm_dir, qc_samples=qc,
                           qc_name=a.qc_cohort.name if a.qc_cohort else "the QC table",
                           max_proteins=a.max_proteins, save_dtype=a.save_dtype, workers=a.workers,
                           n_sampled=a.n_sampled, presence_only=a.presence_only)

    print(f"cohort: {len(r.samples)} genomes ({scope}); with a protein parquet: {r.with_parquet}")
    if not a.presence_only:
        n_bad = len(r.problems.get(COUNT_MISMATCH, []))
        n_changed = len(r.problems.get(CONTIGS_CHANGED, [])) + len(r.problems.get(CONTIGS_UNREADABLE, []))
        print(f"protein count = CDS - pseudogenes: {r.count_checked - n_bad}/{r.count_checked}; "
              f"contigs unchanged since annotation: {r.md5_checked - n_changed}/{r.md5_checked}")
    if a.esm_dir is not None:
        print(f"with an embedding in {a.esm_dir}: {r.with_embedding}")
        for line in r.sampled:
            print(f"  {line}")
    for k, v in r.problems.items():
        print(f"FAIL {k}: {len(v)} {v[:10]}")
    if a.report is not None:
        rows = [(s, k) for k, v in r.problems.items() for s in v]
        pd.DataFrame(rows, columns=["Sample", "problem"]).to_csv(a.report, index=False)
        print(f"report: {a.report} ({len(rows)} rows)")
    if r.ok and marker is not None:
        marker.write_text(hashlib.md5(cohort_bytes).hexdigest() + "\n")
        print(f"pass marker: {marker}")
    print("PASS" if r.ok else "FAIL" + ("" if r.samples else ": no genomes to check"))
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
