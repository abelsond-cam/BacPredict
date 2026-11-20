r"""Symlink ESM-C embeddings from another store into a cohort's ESM store, where they align with our proteins.

David, 2026-11-20: the *Klebsiella* ESM-C embeddings already computed (the Kp store,
``david/bacformer_processed/esm/klebsiella/``) are reused for the Enterobacterales *Klebsiella* by symlink into the
flat ESM store, and the ESM-C array skips them (``apps/entero/scripts/build_entero_embeddings.sh link-kp``). This
module is the generic step: for each listed genome that has no ``.pt`` in the target store, look for
``<src-dir>/<Sample>_esm_embeddings.pt`` and link it in only if it is the embedding our pipeline would have written,
row for row.

What "aligns" means. A ``.pt`` is the Bacformer-input packing of one genome (``generate_embeddings``):
``protein_embeddings`` ``[1, n, 960]``, ``contig_ids`` and ``attention_mask`` ``[1, n]``, one row per protein in the
parquet's order (contigs, then proteins), cut to ``--max-proteins`` rows, contig ids clamped at
``--max-contigs - 1``. From our parquet's per-contig protein counts this module builds the contig-id vector our
embedder would write, and links a source ``.pt`` only when

- its protein count equals that vector's length (``count_mismatch`` otherwise), and
- its ``contig_ids`` equal that vector position for position, i.e. the same proteins per contig, contig by contig
  (``contig_mismatch`` otherwise).

Counts are not sequences. The Kp store was measured to hold our proteins in our order and count except three
selenoproteins, written ``*`` there and ``U`` by our extractor (PROJECT_STATE §3.6 *Caveats*: 39 of 40 sampled
``.pt`` had exactly our count; 1 lacked those three and fails here). A linked ``.pt`` keeps the source's embedding
of every protein, those three included, and the source's dtype (the Kp store is fp32; ours is bf16).

Nothing large is read: the source is opened with ``torch.load(mmap=True, weights_only=True)``, so the 960-wide
tensor's shape and dtype come from the archive's metadata and only ``contig_ids`` and ``attention_mask`` (``n``
values each) are paged in. Our side is the parquet's ``protein_sequence`` list lengths (pyarrow, no ``to_pylist``).

Never overwrites: a genome whose target name exists in any form (file, link, broken link) is ``exists`` and left
alone, and the link itself is ``os.symlink``, which fails rather than replace a name that appeared meanwhile. The
link target is absolute (``--src-dir`` resolved once, through any symlinked parent such as ``~/rds``), so the link
reads the same from any directory and for any user.
``check_cohort_store`` dates a link by its own mtime: the link is made after its parquet was compared.

Report (``--report``, TSV, written atomically; one row per listed genome): ``Sample``, ``status``, ``n_ours`` (our
parquet's protein count), ``n_src`` (the source's rows), ``src_dtype``, ``detail``. Statuses:

=================== ===========================================================================================
``linked``          the link was made (``would_link`` under ``--dry-run``)
``exists``          the target store already has a ``.pt`` under this name: not touched
``absent``          no source ``.pt``
``unreadable``      the source cannot be loaded, or lacks ``protein_embeddings``/``contig_ids``/``attention_mask``
``format_mismatch`` the source loads but is not ``[1, n, 960]`` float32/bfloat16 with ``n`` contig ids and an
                    all-ones mask
``count_mismatch``  the source's protein count is not ours (after the cap)
``contig_mismatch`` the same count, but other proteins per contig
``no_parquet``      our parquet is missing or unreadable: nothing to compare against (run the protein stage)
``error``           the link could not be made (an ``OSError`` other than the name existing)
=================== ===========================================================================================

Exit status 1 if any genome is ``no_parquet`` or ``error`` (the stage ran out of order, or the store is not
writable); every other status is an outcome, and the genome is embedded by the array as usual.

Usage::

    python -m bacpredict.engine.embedding.link_existing_embeddings --samples-file kp_link_samples.txt \
        --protein-dir processed/protein_sequences --src-dir /rds/project/…/bacformer_processed/esm/klebsiella \
        --esm-dir processed/esm --max-proteins 9000 --workers 16 --report genomes/esm_linked_kp.tsv [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pac
import pyarrow.parquet as papq

from bacpredict.engine.embedding.sample_list import read_samples_file

PARQUET_SUFFIX = "_protein_sequences.parquet"
PT_SUFFIX = "_esm_embeddings.pt"
ESM_DIM = 960  # the pinned ESM-C (ESM++ small) width
LINKABLE_DTYPES = ("float32", "bfloat16")  # what a linked .pt may hold: the Kp store's fp32, or our bf16

LINKED = "linked"
WOULD_LINK = "would_link"
EXISTS = "exists"
ABSENT = "absent"
UNREADABLE = "unreadable"
FORMAT_MISMATCH = "format_mismatch"
COUNT_MISMATCH = "count_mismatch"
CONTIG_MISMATCH = "contig_mismatch"
NO_PARQUET = "no_parquet"
ERROR = "error"
STATUSES = (LINKED, WOULD_LINK, EXISTS, ABSENT, UNREADABLE, FORMAT_MISMATCH, COUNT_MISMATCH, CONTIG_MISMATCH,
            NO_PARQUET, ERROR)
FAILING = (NO_PARQUET, ERROR)  # the statuses that make the exit status 1
REPORT_COLUMNS = ["Sample", "status", "n_ours", "n_src", "src_dtype", "detail"]


def contig_protein_counts(parquet: Path) -> np.ndarray:
    """Proteins per contig of a genome's parquet (row 0 of ``protein_sequence``, ``list<list<string>>``), in order."""
    col = papq.read_table(parquet, columns=["protein_sequence"]).column(0).slice(0, 1)
    parts = [pac.list_value_length(chunk.flatten()).fill_null(0).to_numpy(zero_copy_only=False)
             for chunk in col.chunks]
    return np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, dtype=np.int64)


def expected_contig_ids(counts: np.ndarray, max_proteins: int, max_contigs: int) -> np.ndarray:
    """The ``contig_ids`` row our embedder writes for these per-contig counts (bacformer ``large`` packing).

    Contig ``i`` contributes ``counts[i]`` copies of ``i``; the vector is cut to ``max_proteins`` and clamped at
    ``max_contigs - 1`` (``protein_embeddings_to_inputs``).
    """
    ids = np.repeat(np.arange(len(counts), dtype=np.int64), counts)[:max_proteins]
    return np.minimum(ids, max_contigs - 1)


def _read_source(pt: Path) -> dict:
    """The source ``.pt``'s shape, dtype, contig ids and mask, without reading its embedding tensor.

    Raises on anything that is not a loadable dict holding the three keys (the caller's ``unreadable``).
    """
    import torch  # here, not at import: the parent never needs it, so the pool never forks a torch process

    d = torch.load(pt, map_location="cpu", weights_only=True, mmap=True)
    pe, ci, am = d["protein_embeddings"], d["contig_ids"], d["attention_mask"]
    return {"shape": tuple(pe.shape), "dtype": str(pe.dtype).removeprefix("torch."),
            "contig_ids": ci.reshape(-1).to(torch.float64).numpy(),
            "mask_ok": tuple(am.reshape(-1).shape) == (pe.shape[1] if pe.ndim == 3 else -1,)
            and bool((am == 1).all())}


def link_one(task: tuple) -> dict:
    """One genome → its report row (and the link, if it aligns and ``dry_run`` is false)."""
    sample, pq, src, target, max_proteins, max_contigs, dry_run = task
    row = {"Sample": sample, "status": None, "n_ours": None, "n_src": None, "src_dtype": None, "detail": ""}

    def done(status: str, detail: str = "") -> dict:
        row.update(status=status, detail=detail)
        return row

    if os.path.lexists(target):
        return done(EXISTS, f"link -> {os.readlink(target)}" if os.path.islink(target) else "file")
    if not os.path.isfile(src):
        return done(ABSENT)
    try:
        counts = contig_protein_counts(pq)
    except Exception as e:  # noqa: BLE001 (missing or unreadable: either way nothing to compare)
        return done(NO_PARQUET, f"{type(e).__name__}: {e}")
    row["n_ours"] = int(counts.sum())
    try:
        s = _read_source(src)
    except Exception as e:  # noqa: BLE001 (any failure to load is the finding)
        return done(UNREADABLE, f"{type(e).__name__}: {e}")
    row["src_dtype"] = s["dtype"]
    shape = s["shape"]
    if len(shape) == 3:
        row["n_src"] = int(shape[1])
    if len(shape) != 3 or shape[0] != 1 or shape[2] != ESM_DIM:
        return done(FORMAT_MISMATCH, f"protein_embeddings {shape}, not (1, n, {ESM_DIM})")
    if s["dtype"] not in LINKABLE_DTYPES:
        return done(FORMAT_MISMATCH, f"dtype {s['dtype']}, not one of {LINKABLE_DTYPES}")
    if len(s["contig_ids"]) != shape[1] or not s["mask_ok"]:
        return done(FORMAT_MISMATCH, "contig_ids or attention_mask is not n long, or the mask is not all ones")
    want = expected_contig_ids(counts, max_proteins, max_contigs)
    if shape[1] != len(want):
        return done(COUNT_MISMATCH, f"source {shape[1]} proteins, ours {len(want)} after the {max_proteins} cap")
    if not np.array_equal(s["contig_ids"], want):
        first = int(np.flatnonzero(s["contig_ids"] != want)[0])
        return done(CONTIG_MISMATCH, f"contig ids differ from protein {first} on")
    if dry_run:
        return done(WOULD_LINK, str(src))
    try:
        os.symlink(src, target)
    except FileExistsError:
        return done(EXISTS, "appeared during this run")
    except OSError as e:
        return done(ERROR, f"{type(e).__name__}: {e}")
    return done(LINKED, str(src))


def _link_batch(batch: list[tuple]) -> list[dict]:
    return [link_one(t) for t in batch]


def link_existing_embeddings(samples: list[str], protein_dir: Path, src_dir: Path, esm_dir: Path, *,
                             max_proteins: int = 9000, max_contigs: int = 1000, workers: int = 1,
                             dry_run: bool = False, batch_size: int = 64) -> pd.DataFrame:
    """Link every listed genome's aligned source ``.pt`` into ``esm_dir``; return the report.

    Parameters
    ----------
    samples : list of str
        The genomes to consider (e.g. a cohort's *Klebsiella*).
    protein_dir : Path
        Holds our ``<Sample>_protein_sequences.parquet``.
    src_dir : Path
        The store to link from, ``<Sample>_esm_embeddings.pt``. Resolved here (absolute, through any symlinked
        parent such as ``~/rds``); the links point at the resolved path.
    esm_dir : Path
        The store to link into. Nothing in it is ever replaced.
    max_proteins, max_contigs : int
        What our embedder packs with (``generate_embeddings --max-proteins/--max-contigs``).
    workers : int
        Processes; 1 runs here.
    dry_run : bool
        Decide every genome but make no link (``would_link`` in place of ``linked``).
    batch_size : int
        Genomes per pool task.

    Returns
    -------
    pandas.DataFrame
        One row per sample, in order, with :data:`REPORT_COLUMNS`.
    """
    src_dir = Path(src_dir).resolve()
    tasks = [(s, protein_dir / f"{s}{PARQUET_SUFFIX}", src_dir / f"{s}{PT_SUFFIX}", esm_dir / f"{s}{PT_SUFFIX}",
              max_proteins, max_contigs, dry_run) for s in samples]
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
    if workers > 1 and len(batches) > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = [r for batch in ex.map(_link_batch, batches) for r in batch]
    else:
        rows = [r for batch in batches for r in _link_batch(batch)]
    return pd.DataFrame(rows, columns=REPORT_COLUMNS).astype({"n_ours": "Int64", "n_src": "Int64"})


def write_report(report: pd.DataFrame, path: Path) -> None:
    """Write the TSV atomically (a sibling ``.part``, then a rename)."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
    try:
        report.to_csv(tmp, sep="\t", index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """CLI; returns the exit status (1 if any genome is ``no_parquet`` or ``error``)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--samples-file", type=Path, required=True,
                    help="the genomes to consider: one per line, sorted and unique (sample_list.py)")
    ap.add_argument("--protein-dir", type=Path, required=True)
    ap.add_argument("--src-dir", type=Path, required=True, help="the store to link from")
    ap.add_argument("--esm-dir", type=Path, required=True, help="the store to link into")
    ap.add_argument("--max-proteins", type=int, default=9000)
    ap.add_argument("--max-contigs", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--report", type=Path, required=True, help="TSV, one row per listed genome")
    ap.add_argument("--dry-run", action="store_true", help="decide every genome, link none")
    a = ap.parse_args(argv)

    try:
        samples = read_samples_file(a.samples_file)
    except (OSError, ValueError) as e:
        ap.error(f"--samples-file: {e}")
    for name, d in (("--src-dir", a.src_dir), ("--esm-dir", a.esm_dir), ("--protein-dir", a.protein_dir)):
        if not d.is_dir():
            ap.error(f"{name} {d} is not a directory")
    if a.src_dir.resolve() == a.esm_dir.resolve():
        ap.error("--src-dir and --esm-dir are the same directory")

    report = link_existing_embeddings(samples, a.protein_dir, a.src_dir, a.esm_dir, max_proteins=a.max_proteins,
                                      max_contigs=a.max_contigs, workers=a.workers, dry_run=a.dry_run)
    write_report(report, a.report)
    counts = Counter(report["status"])
    print(f"{len(samples)} genomes ({a.samples_file.name}); source {a.src_dir.resolve()} → "
          f"{a.esm_dir}{' (dry run: no link made)' if a.dry_run else ''}")
    for status in STATUSES:
        if counts[status]:
            print(f"  {status}: {counts[status]}")
    dtypes = Counter(report.loc[report["status"].isin([LINKED, WOULD_LINK]), "src_dtype"])
    if dtypes:
        print(f"  dtype of the linked: {dict(sorted(dtypes.items()))}")
    print(f"report: {a.report}")
    bad = [s for s in FAILING if counts[s]]
    if bad:
        print(f"FAIL: {', '.join(f'{counts[s]} {s}' for s in bad)}")
        return 1
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
