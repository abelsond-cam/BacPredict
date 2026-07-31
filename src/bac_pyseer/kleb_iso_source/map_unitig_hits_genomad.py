r"""Map every invasion GWAS unitig hit onto geNomad MGE calls — the HGT-vs-chromosomal test, at scale.

The unitig (accessory) axis of the *Klebsiella* invasion GWAS concentrates its signal at **common
allele frequency** and is LD-redundant (``docs/PROGRESS_UNITIGS.md``). The hypothesis under test: is
that signal **plasmid/prophage-borne (acquired/HGT)** or **chromosomal**? This module maps **all**
significant hit unitigs into the genomes that carry them and reports, per unitig / per pattern_group /
overall, the fraction landing on a plasmid, a prophage, or the chromosome.

It is a genuine test of a hypothesis. Refutation (chromosomal / scattered / no geNomad call) is a
first-class outcome, reported plainly. "Co-inherited plasmid/prophage" stays an *untested* hypothesis
until this mapping supports it; nothing here asserts a causal-vs-lineage verdict.

Method
------
A coloured de-Bruijn unitig present in a sample is an **exact substring** of that sample's assembly
(and of geNomad's excised plasmid/virus sequences), so classification is **exact-substring matching**
on both strands — not alignment. At ~109M (unitig, carrier) lookups (≈33k unitigs × ~3.3k carriers),
each genome is streamed **once** through a single **Aho-Corasick** automaton of all unitigs (both
strands), so cost is ∝ genome size, not × unitigs.

Per carrier, unitigs are matched against three tagged sources and classified by priority
**PLASMID > VIRUS > ASM-only > unmapped** (nesting: a plasmid unitig also occurs in the assembly; a
prophage unitig occurs in both the excised ``_virus.fna`` and the assembly — matching ``_virus.fna``
is what catches *integrated* prophage). A carrier with no geNomad output is ``unknown_no_genomad`` —
**not** chromosomal, which would fabricate a hypothesis-refuting signal.

Cross-carrier caveats (recorded in the manifest, never hidden):

* ``seq_name`` is a **per-assembly contig name**, not comparable across genomes — "same plasmid /
  prophage across carriers" is summarised via cross-comparable geNomad fields: phage ``taxonomy`` and
  plasmid ``conjugation_genes``/``amr_genes`` (from ``genomad_{plasmid,virus}_summary_long.tsv``).
* ``chromosomal`` = *not on a geNomad-flagged plasmid/virus contig*; short-read geNomad plasmid
  recall is imperfect, so some may be unflagged plasmid contigs. The spot-check validates plasmid/
  prophage *positives*, not chromosomal *negatives*.

Phases (``--phase``): ``select`` (all unitigs → id_map + cached hit sub-matrix + resolved carriers),
``align`` (array shard of carriers → Aho-Corasick classify → per-unitig aggregate + per-pair parquet
part), ``combine`` (roll up to per-unitig / per-pattern_group / overall + spot-check), ``smoke``
(select + align a few carriers + combine, timed, for validation). Assembly resolution reuses
``resolve_assembly_paths.resolve()``. Results go to project_k / scratch only, never ``$HOME``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:  # package import when bac_pyseer is on the path (editable install)
    from bac_pyseer.kleb_iso_source.resolve_assembly_paths import resolve as resolve_assembly_paths
except ImportError:  # invoked as a bare script — add the script dir to sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from resolve_assembly_paths import resolve as resolve_assembly_paths  # type: ignore[no-redef]

# --- geNomad layout (verified on CSD3; centralised so a real-layout change is a one-line fix) -------
# Per-sample extracts: <root>/per_sample/<key>/<key>_summary/<key>_{plasmid,virus}.fna, key = Sample
# or "<Sample>__sr" (paired short-read). Keyed long tables at <root>/genomad_{plasmid,virus}_summary_long.tsv.
_SR_SUFFIX = "__sr"
_PLASMID_LONG = "genomad_plasmid_summary_long.tsv"
_VIRUS_LONG = "genomad_virus_summary_long.tsv"

# Classification tags for the three target sources (priority PLASMID > VIRUS > ASM-only).
_TAG_PLASMID, _TAG_VIRUS, _TAG_ASM = "PLASMID", "VIRUS", "ASM"
_CLASS_BY_TAG = {_TAG_PLASMID: "plasmid", _TAG_VIRUS: "prophage", _TAG_ASM: "chromosomal"}
_CLASSES = ("plasmid", "prophage", "chromosomal", "unmapped", "unknown_no_genomad")

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_PARQUET_BATCH = 500_000  # rows buffered before a parquet flush (bounds shard RAM)


def _revcomp(seq: str) -> str:
    """Reverse-complement a DNA string (ggcat emits canonical k-mers, so a unitig may be RC in a sample)."""
    return seq.translate(_COMPLEMENT)[::-1]


# --------------------------------------------------------------------------------------------------
# shared IO helpers
# --------------------------------------------------------------------------------------------------
def _iter_fasta(path: Path) -> Any:
    """Yield ``(header_first_token, sequence)`` from a plain or gzipped FASTA."""
    opener = gzip.open if str(path).endswith(".gz") else open
    header, chunks = None, []
    with opener(path, "rt") as fh:  # type: ignore[operator]
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:].strip().split()[0], []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def _load_contigs(path: Path | None) -> dict[str, str]:
    """Read a FASTA into ``{seq_name: UPPERCASE sequence}`` (empty dict if the path is None/missing)."""
    if path is None:
        return {}
    return {name: seq.upper() for name, seq in _iter_fasta(path)}


# --------------------------------------------------------------------------------------------------
# the one-time 77 GB → cached hit sub-matrix (pigz|awk hash-join)
# --------------------------------------------------------------------------------------------------
def extract_hit_submatrix(matrix_gz: Path, all_hit_seqs: set[str], submatrix_path: Path, *, decomp_threads: int = 4) -> int:
    """Extract only the hit-unitig rows from the 77 GB matrix into a small **cached** sub-matrix.

    A single C-speed streaming hash-join — ``pigz -dc`` (parallel gunzip; ``gzip -dc`` fallback) piped
    to an ``awk`` that keeps a matrix line iff its left field (the unitig sequence) is in the hit set.
    ``awk`` touches only ``$0`` (never ``$1``), so it does not field-split the giant carrier lists. The
    77 GB matrix is read exactly **once**, ever — the result is the reusable index, and this returns
    immediately (``-1``) if it already exists.
    """
    if submatrix_path.is_file() and submatrix_path.stat().st_size > 0:
        print(f"reusing cached hit sub-matrix {submatrix_path} (skipping the 77 GB pass)", file=sys.stderr)
        return -1
    targets = submatrix_path.with_suffix(".targets.txt")
    targets.write_text("".join(f"{s}\n" for s in all_hit_seqs))
    decomp = f"pigz -p {decomp_threads} -dc" if shutil.which("pigz") else "gzip -dc"
    awk_prog = r'NR==FNR{t[$0];next}{i=index($0," | ");if(i){k=substr($0,1,i-1);if(k in t)print}}'
    pipe = (f"set -o pipefail; {decomp} {shlex.quote(str(matrix_gz))} | "
            f"awk {shlex.quote(awk_prog)} {shlex.quote(str(targets))} -")
    print(f"extracting hit sub-matrix ({decomp.split()[0]} | awk hash-join, one 77 GB pass)…", file=sys.stderr)
    with submatrix_path.open("w") as out:
        res = subprocess.run(["bash", "-c", pipe], stdout=out, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"sub-matrix extraction failed (rc={res.returncode}): "
                           f"{res.stderr.decode('utf-8', errors='replace')}")
    targets.unlink(missing_ok=True)
    n = sum(1 for _ in submatrix_path.open())
    print(f"wrote {submatrix_path}: {n} hit rows", file=sys.stderr)
    return n


def carrier_union_from_submatrix(submatrix_path: Path, hit_seqs: set[str]) -> tuple[set[str], set[str]]:
    """Stream the small cached sub-matrix → (union of carrier Sample IDs, set of hit seqs actually present).

    Light on RAM (union caps at the cohort size; matched caps at the hit count) — the per-carrier
    expected sets are built later, per shard, in :func:`shard_expected`.
    """
    union: set[str] = set()
    matched: set[str] = set()
    with submatrix_path.open() as fh:
        for line in fh:
            seq, sep, rest = line.partition(" | ")
            if not sep or seq not in hit_seqs:
                continue
            matched.add(seq)
            union.update(tok.rpartition(":")[0] for tok in rest.split())
    return union, matched


def shard_expected(submatrix_path: Path, seq2idx: dict[str, int], my_carriers: set[str]) -> dict[str, set[int]]:
    """Stream the sub-matrix → ``{carrier: {unitig_idx…}}`` for this shard's carriers only (small RAM)."""
    expected: dict[str, set[int]] = defaultdict(set)
    with submatrix_path.open() as fh:
        for line in fh:
            seq, sep, rest = line.partition(" | ")
            if not sep:
                continue
            idx = seq2idx.get(seq)
            if idx is None:
                continue
            for tok in rest.split():
                s = tok.rpartition(":")[0]
                if s in my_carriers:
                    expected[s].add(idx)
    return expected


# --------------------------------------------------------------------------------------------------
# geNomad resolution + metadata
# --------------------------------------------------------------------------------------------------
def resolve_genomad_paths(sample: str, genomad_root: Path) -> dict[str, Any]:
    """Resolve a Sample's geNomad ``_plasmid.fna``/``_virus.fna``, trying ``<Sample>`` then ``__sr``.

    Returns ``key`` (resolved per-sample key or None), ``plasmid_fna``/``virus_fna`` (Path or None if
    absent/empty), and ``found`` (whether the summary dir exists at all).
    """
    for key in (sample, f"{sample}{_SR_SUFFIX}"):
        summ = genomad_root / "per_sample" / key / f"{key}_summary"
        if not summ.is_dir():
            continue
        plasmid = summ / f"{key}_plasmid.fna"
        virus = summ / f"{key}_virus.fna"
        return {"key": key,
                "plasmid_fna": plasmid if plasmid.is_file() and plasmid.stat().st_size > 0 else None,
                "virus_fna": virus if virus.is_file() and virus.stat().st_size > 0 else None,
                "found": True}
    return {"key": None, "plasmid_fna": None, "virus_fna": None, "found": False}


def load_genomad_meta(genomad_root: Path) -> dict[str, dict[tuple[str, str], str]]:
    """Load the summary_long tables → ``{(Sample, seq_name): descriptor}`` for plasmid and virus.

    Descriptors are the **cross-comparable** replicon/prophage identities: virus → ``taxonomy``;
    plasmid → ``conjugation_genes`` (MOB type; falls back to ``amr_genes``). Missing table → empty
    dict (records the descriptor as ``""``, not a spurious mismatch).
    """
    meta: dict[str, dict[tuple[str, str], str]] = {"plasmid": {}, "prophage": {}}
    pl = genomad_root / _PLASMID_LONG
    if pl.is_file():
        d = pd.read_csv(pl, sep="\t", usecols=lambda c: c in ("Sample", "seq_name", "conjugation_genes", "amr_genes"))
        for r in d.itertuples(index=False):
            desc = str(getattr(r, "conjugation_genes", "") or "") or str(getattr(r, "amr_genes", "") or "")
            meta["plasmid"][(str(r.Sample), str(r.seq_name))] = desc
    vi = genomad_root / _VIRUS_LONG
    if vi.is_file():
        d = pd.read_csv(vi, sep="\t", usecols=lambda c: c in ("Sample", "seq_name", "taxonomy"))
        for r in d.itertuples(index=False):
            meta["prophage"][(str(r.Sample), str(r.seq_name))] = str(getattr(r, "taxonomy", "") or "")
    return meta


# --------------------------------------------------------------------------------------------------
# Aho-Corasick automaton + classification
# --------------------------------------------------------------------------------------------------
def build_automaton(id_map: pd.DataFrame) -> Any:
    """Build one Aho-Corasick automaton over all unitigs (both strands); value = ``unitig_idx``."""
    import ahocorasick  # lazy: select/combine/--help don't need it

    aut = ahocorasick.Automaton()
    for row in id_map.itertuples(index=False):
        seq = str(row.variant).upper()
        idx = int(row.unitig_idx)
        aut.add_word(seq, idx)
        aut.add_word(_revcomp(seq), idx)
    aut.make_automaton()
    return aut


def scan_carrier(aut: Any, plasmid: dict[str, str], virus: dict[str, str],
                 asm: dict[str, str]) -> tuple[dict[int, dict[str, set[str]]], dict[int, list[tuple[str, int]]]]:
    """Stream a carrier's tagged contigs through the automaton.

    Returns ``(found, asm_pos)``: ``found[idx][tag] = {seq_name…}`` (for geNomad classification) and
    ``asm_pos[idx] = [(contig, end_off)…]`` — every ASM occurrence, for IS-overlap + copy number.
    """
    found: dict[int, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    asm_pos: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for tag, contigs in ((_TAG_PLASMID, plasmid), (_TAG_VIRUS, virus), (_TAG_ASM, asm)):
        for name, cseq in contigs.items():
            for end_off, idx in aut.iter(cseq):
                found[idx][tag].add(name)
                if tag == _TAG_ASM:
                    asm_pos[idx].append((name, end_off))
    return found, asm_pos


def classify(tag_hits: dict[str, set[str]]) -> dict[str, Any]:
    """Classify one unitig's per-tag hits by priority PLASMID > VIRUS > ASM-only > unmapped.

    Returns ``mge_class``, the winning class's ``seq_names`` (list), ``multi_replicon``, and
    ``asm_recall`` (was the unitig found in the assembly at all — the built-in ground-truth QC).
    """
    asm_recall = _TAG_ASM in tag_hits
    for tag in (_TAG_PLASMID, _TAG_VIRUS, _TAG_ASM):
        if tag in tag_hits:
            seqs = sorted(tag_hits[tag])
            return {"mge_class": _CLASS_BY_TAG[tag], "seq_names": seqs,
                    "multi_replicon": len(seqs) > 1, "asm_recall": asm_recall}
    return {"mge_class": "unmapped", "seq_names": [], "multi_replicon": False, "asm_recall": asm_recall}


# --------------------------------------------------------------------------------------------------
# IS-element (ISEScan) annotation — geNomad misses IS/transposons, so "chromosomal" absorbs them
# --------------------------------------------------------------------------------------------------
def load_is_intervals(csv_path: Path) -> dict[str, list[tuple[int, int, str, bool]]]:
    """Read an ISEScan per-genome ``.fa.csv`` → ``{contig: [(start, end, family, is_partial)…]}``.

    Uses ``isBegin``/``isEnd`` (1-based; swapped if start>end) — NOT ``start1/end1`` (0 for partial
    IS). ``type == 'p'`` marks a partial/degraded IS. Mirrors BacHGT ``isescan_gene_context._process``.
    """
    by_contig: dict[str, list[tuple[int, int, str, bool]]] = defaultdict(list)
    import csv as _csv
    with csv_path.open(newline="") as fh:
        for r in _csv.DictReader(fh):
            try:
                s, e = int(r["isBegin"]), int(r["isEnd"])
            except (KeyError, ValueError):
                continue
            if s > e:
                s, e = e, s
            by_contig[str(r["seqID"])].append((s, e, str(r.get("family", "")), str(r.get("type", "")) == "p"))
    return by_contig


def annotate_is(asm_positions: list[tuple[str, int]], unitig_len: int,
                is_by_contig: dict[str, list[tuple[int, int, str, bool]]] | None) -> dict[str, Any]:
    """Annotate a unitig's ASM occurrences against a carrier's IS intervals.

    Returns ``is_element`` (True/False, or None when the carrier has no ISEScan data), ``is_family``
    (of an overlapping IS), ``is_partial``, and ``n_copies`` (distinct ASM positions — IS-borne
    unitigs are typically multi-copy). Overlap is 1-based inclusive against ``isBegin``/``isEnd``.
    """
    n_copies = len({(c, p) for c, p in asm_positions})
    if is_by_contig is None:
        return {"is_element": None, "is_family": None, "is_partial": None, "n_copies": n_copies}
    hit = False
    fam: str | None = None
    partial = False
    for contig, end_off in asm_positions:
        u1s, u1e = end_off - unitig_len + 2, end_off + 1   # 0-based end_off → 1-based inclusive span
        for s, e, f, p in is_by_contig.get(contig, ()):
            if not (u1e < s or u1s > e):
                hit = True
                fam = fam or (f or None)
                partial = partial or p
    return {"is_element": hit, "is_family": fam, "is_partial": partial, "n_copies": n_copies}


def _is_state(is_element: bool | None) -> str:
    """IS state label for aggregation: 'IS' / 'nonIS' / 'na' (no ISEScan data)."""
    return "na" if is_element is None else ("IS" if is_element else "nonIS")


# --------------------------------------------------------------------------------------------------
# select
# --------------------------------------------------------------------------------------------------
def run_select(args: argparse.Namespace) -> None:
    """``select``: id_map for ALL hit unitigs; cached hit sub-matrix; resolved carrier→assembly list."""
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.scratch_dir.glob("align_shard_*"):  # avoid mixing a previous run's shard parts
        stale.unlink()
    hits = pd.read_csv(args.hits_tsv, sep="\t", low_memory=False)
    hits = hits.reset_index(drop=True)
    keep = ["variant", "pattern_group", "direction", "beta", "af", "var_explained_pct"]
    id_map = hits[[c for c in keep if c in hits.columns]].copy()
    id_map.insert(0, "unitig_idx", range(len(id_map)))
    id_map["unitig_len"] = id_map["variant"].astype(str).str.len()
    id_map.to_csv(out / "id_map.tsv", sep="\t", index=False)

    all_hit_seqs = set(id_map["variant"].astype(str))
    submatrix = out / "hits_submatrix.tsv"
    extract_hit_submatrix(args.unitig_matrix, all_hit_seqs, submatrix, decomp_threads=args.decomp_threads)
    union, matched = carrier_union_from_submatrix(submatrix, all_hit_seqs)
    missing = sorted(all_hit_seqs - matched)
    (out / "unitig_join_misses.txt").write_text("\n".join(missing) + ("\n" if missing else ""))

    carrier_union = sorted(union)
    tmp_csv = out / "_carrier_union.csv"
    pd.DataFrame({"Sample": carrier_union}).to_csv(tmp_csv, index=False)
    resolve_assembly_paths([tmp_csv], all_kpsc=False, metadata_path=args.metadata,
                           out_tsv=out / "_carriers.assembly.tsv", check_exists=True)
    tmp_csv.unlink(missing_ok=True)
    res = pd.read_csv(out / "_carriers.assembly.tsv", sep="\t", header=None, names=["Sample", "assembly_path"])
    res.to_csv(out / "carriers.resolved.tsv", sep="\t", index=False)

    probe = _probe_genomad_layout(carrier_union[: args.genomad_probe_n], args.genomad_root)
    pd.DataFrame(probe).to_csv(out / "genomad_layout_probe.tsv", sep="\t", index=False)

    manifest = {
        "phase": "select", "n_unitigs": int(len(id_map)),
        "n_pattern_groups": int(id_map["pattern_group"].nunique()) if "pattern_group" in id_map else None,
        "n_carrier_union": len(carrier_union), "n_resolved_assemblies": int(len(res)),
        "n_hit_seqs_matched": len(matched), "n_join_misses": len(missing),
        "match_method": "exact-substring (Aho-Corasick, fwd+revcomp)",
        "genomad_layout_found": sum(1 for p in probe if p["found"]),
    }
    (out / "select_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), file=sys.stderr)


def _probe_genomad_layout(samples: list[str], genomad_root: Path) -> list[dict[str, Any]]:
    """Probe a few carriers' geNomad dirs → which key (<Sample>/__sr) resolved + which extracts exist."""
    rows = []
    for s in samples:
        g = resolve_genomad_paths(s, genomad_root)
        rows.append({"Sample": s, "resolved_key": g["key"], "found": g["found"],
                     "has_plasmid": g["plasmid_fna"] is not None, "has_virus": g["virus_fna"] is not None})
    return rows


# --------------------------------------------------------------------------------------------------
# align
# --------------------------------------------------------------------------------------------------
def _read_carrier_shard(carriers_tsv: Path, shard_index: int, n_shards: int) -> pd.DataFrame:
    """Read the resolved carrier table and slice this shard (round-robin; carriers are independent)."""
    df = pd.read_csv(carriers_tsv, sep="\t")
    if n_shards > 1:
        df = df[df.reset_index(drop=True).index % n_shards == shard_index]
    return df.reset_index(drop=True)


class _ParquetSink:
    """Buffered pyarrow ParquetWriter for the per-(unitig, carrier) detail (bounded RAM)."""

    def __init__(self, path: Path):
        import pyarrow as pa  # noqa: F401  (import guarded here so select/combine don't need it)

        self.path = path
        self._writer = None
        self._buf: list[dict[str, Any]] = []

    def add(self, rec: dict[str, Any]) -> None:
        """Buffer one detail row; flush at the batch threshold."""
        self._buf.append(rec)
        if len(self._buf) >= _PARQUET_BATCH:
            self.flush()

    def flush(self) -> None:
        """Write the buffered rows as one row group."""
        if not self._buf:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(self._buf)
        if self._writer is None:
            self._writer = pq.ParquetWriter(str(self.path), table.schema, compression="zstd")
        self._writer.write_table(table)
        self._buf.clear()

    def close(self) -> None:
        """Flush and close (no file written if nothing was added)."""
        self.flush()
        if self._writer is not None:
            self._writer.close()


def run_align(args: argparse.Namespace) -> None:
    """``align`` shard: Aho-Corasick classify this shard's carriers → per-unitig agg + per-pair parquet."""
    out, scratch = args.out_dir, args.scratch_dir
    scratch.mkdir(parents=True, exist_ok=True)
    id_map = pd.read_csv(out / "id_map.tsv", sep="\t")
    seq2idx = {str(s).upper(): int(i) for s, i in zip(id_map["variant"], id_map["unitig_idx"], strict=True)}
    ulen = {int(i): int(n) for i, n in zip(id_map["unitig_idx"], id_map["unitig_len"], strict=True)}
    aut = build_automaton(id_map)
    meta = load_genomad_meta(args.genomad_root)
    is_lookup = dict(zip(*[pd.read_csv(args.isescan_lookup, sep="\t")[c].astype(str)
                           for c in ("Sample", "path")], strict=True)) if args.isescan_lookup else {}

    shard = _read_carrier_shard(out / "carriers.resolved.tsv", args.carrier_shard_index, args.n_shards)
    my_carriers = set(shard["Sample"].astype(str))
    expected = shard_expected(out / "hits_submatrix.tsv", seq2idx, my_carriers)

    i = args.carrier_shard_index
    sink = _ParquetSink(scratch / f"align_shard_{i:04d}.parquet")
    class_counts: dict[int, Counter] = defaultdict(Counter)              # unitig_idx -> (mge_class, is_state) -> n
    tax_counts: dict[int, Counter] = defaultdict(Counter)               # unitig_idx -> (class, descriptor) -> n
    isfam_counts: dict[int, Counter] = defaultdict(Counter)             # unitig_idx -> is_family -> n (IS-borne only)
    copies: dict[int, list[int]] = defaultdict(lambda: [0, 0])         # unitig_idx -> [sum_copies, n_obs]
    spot: list[dict[str, Any]] = []
    qc = {"n_carriers": 0, "n_pairs": 0, "asm_recall_num": 0, "asm_recall_den": 0,
          "found_not_expected": 0, "no_genomad_carriers": 0, "no_isescan_carriers": 0}
    t0 = time.perf_counter()

    for row in shard.itertuples(index=False):
        sample = str(row.Sample)
        exp = expected.get(sample, set())
        if not exp:
            continue
        qc["n_carriers"] += 1
        gp = resolve_genomad_paths(sample, args.genomad_root)
        is_path = is_lookup.get(sample)
        is_by_contig = load_is_intervals(Path(is_path)) if is_path and Path(is_path).is_file() else None
        if is_by_contig is None:
            qc["no_isescan_carriers"] += 1
        if not gp["found"]:
            qc["no_genomad_carriers"] += 1
            for idx in exp:
                class_counts[idx][("unknown_no_genomad", "na")] += 1
                sink.add({"unitig_idx": idx, "Sample": sample, "genomad_key": None, "mge_class": "unknown_no_genomad",
                          "seq_name": None, "descriptor": None, "asm_recall": None,
                          "is_element": None, "is_family": None, "is_partial": None, "n_copies": 0})
            qc["n_pairs"] += len(exp)
            continue

        key = gp["key"]
        found, asm_pos = scan_carrier(aut, _load_contigs(gp["plasmid_fna"]), _load_contigs(gp["virus_fna"]),
                                      _load_contigs(Path(str(row.assembly_path))))
        qc["found_not_expected"] += sum(1 for idx in found if idx not in exp)
        for idx in exp:
            c = classify(found.get(idx, {}))
            mge_class, seqs = c["mge_class"], c["seq_names"]
            seq_name = seqs[0] if seqs else None
            desc = ""
            if mge_class in ("plasmid", "prophage") and seq_name is not None:
                desc = next((meta[mge_class].get((key, s), "") for s in seqs if (key, s) in meta[mge_class]), "")
            isa = annotate_is(asm_pos.get(idx, []), ulen[idx], is_by_contig)
            state = _is_state(isa["is_element"])
            class_counts[idx][(mge_class, state)] += 1
            if mge_class in ("plasmid", "prophage"):
                tax_counts[idx][(mge_class, desc)] += 1
            if isa["is_element"]:
                isfam_counts[idx][isa["is_family"] or "unknown_family"] += 1
            copies[idx][0] += isa["n_copies"]
            copies[idx][1] += 1
            if c["asm_recall"] is not None:
                qc["asm_recall_den"] += 1
                qc["asm_recall_num"] += int(c["asm_recall"])
            if mge_class in ("plasmid", "prophage") and len(spot) < args.spot_n * 200:
                spot.append({"unitig_idx": idx, "Sample": sample, "genomad_key": key, "mge_class": mge_class,
                             "seq_name": seq_name, "in_genomad_long": (key, seq_name) in meta[mge_class]})
            sink.add({"unitig_idx": idx, "Sample": sample, "genomad_key": key, "mge_class": mge_class,
                      "seq_name": seq_name, "descriptor": desc, "asm_recall": c["asm_recall"],
                      "is_element": isa["is_element"], "is_family": isa["is_family"],
                      "is_partial": isa["is_partial"], "n_copies": isa["n_copies"]})
        qc["n_pairs"] += len(exp)

    sink.close()
    qc["seconds"] = round(time.perf_counter() - t0, 2)
    qc["sec_per_genome"] = round(qc["seconds"] / qc["n_carriers"], 4) if qc["n_carriers"] else None
    _write_class_counts(scratch / f"align_shard_{i:04d}.class.tsv", class_counts)
    _write_tax_counts(scratch / f"align_shard_{i:04d}.tax.tsv", tax_counts)
    _write_isfam_counts(scratch / f"align_shard_{i:04d}.isfam.tsv", isfam_counts)
    _write_copies(scratch / f"align_shard_{i:04d}.copies.tsv", copies)
    pd.DataFrame(spot).to_csv(scratch / f"align_shard_{i:04d}.spot.tsv", sep="\t", index=False)
    (scratch / f"align_shard_{i:04d}.qc.json").write_text(json.dumps(qc, indent=2))
    print(f"shard {i}: {qc['n_carriers']} carriers, {qc['n_pairs']} pairs, {qc['sec_per_genome']} s/genome", file=sys.stderr)


def _write_class_counts(path: Path, class_counts: dict[int, Counter]) -> None:
    """Write long per-(unitig_idx, mge_class, is_state) counts for this shard."""
    rows = [{"unitig_idx": idx, "mge_class": cls, "is_state": st, "n": n}
            for idx, cc in class_counts.items() for (cls, st), n in cc.items()]
    pd.DataFrame(rows, columns=["unitig_idx", "mge_class", "is_state", "n"]).to_csv(path, sep="\t", index=False)


def _write_isfam_counts(path: Path, isfam_counts: dict[int, Counter]) -> None:
    """Write long per-(unitig_idx, is_family) counts for IS-borne placements."""
    rows = [{"unitig_idx": idx, "is_family": fam, "n": n} for idx, cc in isfam_counts.items() for fam, n in cc.items()]
    pd.DataFrame(rows, columns=["unitig_idx", "is_family", "n"]).to_csv(path, sep="\t", index=False)


def _write_copies(path: Path, copies: dict[int, list[int]]) -> None:
    """Write per-unitig ASM copy-number sums (→ mean copies in combine)."""
    rows = [{"unitig_idx": idx, "copies_sum": s, "copies_n": n} for idx, (s, n) in copies.items()]
    pd.DataFrame(rows, columns=["unitig_idx", "copies_sum", "copies_n"]).to_csv(path, sep="\t", index=False)


def _write_tax_counts(path: Path, tax_counts: dict[int, Counter]) -> None:
    """Write long per-(unitig_idx, class, descriptor) counts (the cross-carrier convergence proxy)."""
    rows = [{"unitig_idx": idx, "mge_class": cls, "descriptor": desc, "n": n}
            for idx, cc in tax_counts.items() for (cls, desc), n in cc.items()]
    pd.DataFrame(rows, columns=["unitig_idx", "mge_class", "descriptor", "n"]).to_csv(path, sep="\t", index=False)


# --------------------------------------------------------------------------------------------------
# combine
# --------------------------------------------------------------------------------------------------
def run_combine(args: argparse.Namespace) -> None:
    """``combine``: sum shard aggregates → per-unitig / per-pattern_group / overall tables + spot-check.

    Classes are the **IS-refined** geNomad classes (``chromosomal`` split into ``chromosomal`` = truly
    chromosomal vs ``chromosomal_IS`` = on an ISEScan IS element; likewise plasmid/prophage).
    """
    out, scratch = args.out_dir, args.scratch_dir
    id_map = pd.read_csv(out / "id_map.tsv", sep="\t")

    cls3 = _sum_long(sorted(scratch.glob("align_shard_*.class.tsv")), ["unitig_idx", "mge_class", "is_state"])
    cls3["rclass"] = [_refined_class(c, s) for c, s in zip(cls3["mge_class"], cls3["is_state"], strict=True)]
    rcls = cls3.groupby(["unitig_idx", "rclass"], as_index=False)["n"].sum()
    classes = sorted(rcls["rclass"].unique())
    wide = rcls.pivot_table(index="unitig_idx", columns="rclass", values="n", fill_value=0).reset_index()
    for c in classes:
        if c not in wide.columns:
            wide[c] = 0
    wide["n_carriers"] = wide[classes].sum(axis=1)
    for c in classes:
        wide[f"frac_{c}"] = (wide[c] / wide["n_carriers"]).round(4)
    chrom_tot = sum(wide.get(c, 0) for c in ("chromosomal", "chromosomal_IS", "chromosomal_naIS"))
    wide["frac_chromosomal_is"] = (wide.get("chromosomal_IS", 0) / chrom_tot).round(4)  # IS share OF chromosomal

    tax = _sum_long(sorted(scratch.glob("align_shard_*.tax.tsv")), ["unitig_idx", "mge_class", "descriptor"])
    isfam = _sum_long(sorted(scratch.glob("align_shard_*.isfam.tsv")), ["unitig_idx", "is_family"])
    per_unitig = (id_map.merge(wide, on="unitig_idx", how="left")
                  .merge(_dominant_descriptor(tax), on="unitig_idx", how="left")
                  .merge(_dominant_is_family(isfam), on="unitig_idx", how="left")
                  .merge(_sum_copies(sorted(scratch.glob("align_shard_*.copies.tsv"))), on="unitig_idx", how="left"))
    per_unitig.to_csv(out / "mge_unitig_class.tsv", sep="\t", index=False)

    _rollup_pattern_group(per_unitig, classes).to_csv(out / "mge_pattern_group.tsv", sep="\t", index=False)
    _rollup_overall(per_unitig, classes).to_csv(out / "mge_overall.tsv", sep="\t", index=False)

    spot = pd.concat([pd.read_csv(f, sep="\t") for f in sorted(scratch.glob("align_shard_*.spot.tsv")) if f.stat().st_size > 1],
                     ignore_index=True) if list(scratch.glob("align_shard_*.spot.tsv")) else pd.DataFrame()
    spot.to_csv(out / "spotcheck.tsv", sep="\t", index=False)

    _assemble_parquet_dataset(scratch, out / "mge_hits.parquet")
    _write_combine_manifest(out, scratch, per_unitig, spot, classes)
    print(pd.read_csv(out / "mge_overall.tsv", sep="\t").to_string(index=False), file=sys.stderr)


def _refined_class(mge_class: str, is_state: str) -> str:
    """Split a geNomad class by IS state → ``chromosomal_IS`` / ``chromosomal`` (=non-IS) / ``chromosomal_naIS``."""
    if mge_class in ("plasmid", "prophage", "chromosomal"):
        return {"IS": f"{mge_class}_IS", "nonIS": mge_class, "na": f"{mge_class}_naIS"}[is_state]
    return mge_class


def _assemble_parquet_dataset(scratch: Path, dataset_dir: Path) -> None:
    """Move this run's per-shard parquet parts into a fresh durable dataset dir (never concatenated in RAM)."""
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True)
    for part in sorted(scratch.glob("align_shard_*.parquet")):
        shutil.move(str(part), str(dataset_dir / part.name))


def _sum_long(files: list[Path], keys: list[str]) -> pd.DataFrame:
    """Concatenate long shard tables and sum ``n`` over ``keys``."""
    parts = [pd.read_csv(f, sep="\t") for f in files if f.stat().st_size > 1]
    if not parts:
        return pd.DataFrame(columns=[*keys, "n"])
    return pd.concat(parts, ignore_index=True).groupby(keys, as_index=False)["n"].sum()


def _dominant_descriptor(tax: pd.DataFrame) -> pd.DataFrame:
    """Per unitig: the most common plasmid/prophage descriptor + its fraction of that unitig's MGE hits."""
    if tax.empty:
        return pd.DataFrame(columns=["unitig_idx", "dominant_mge_class", "dominant_descriptor", "dominant_descriptor_frac"])
    tot = tax.groupby("unitig_idx")["n"].transform("sum")
    tax = tax.assign(frac=tax["n"] / tot)
    top = tax.sort_values("n", ascending=False).drop_duplicates("unitig_idx")
    return top[["unitig_idx", "mge_class", "descriptor", "frac"]].rename(
        columns={"mge_class": "dominant_mge_class", "descriptor": "dominant_descriptor", "frac": "dominant_descriptor_frac"})


def _dominant_is_family(isfam: pd.DataFrame) -> pd.DataFrame:
    """Per unitig: the most common IS family among its IS-borne placements + how many carriers are IS-borne."""
    if isfam.empty:
        return pd.DataFrame(columns=["unitig_idx", "dominant_is_family", "n_is_carriers"])
    n_is = isfam.groupby("unitig_idx", as_index=False)["n"].sum().rename(columns={"n": "n_is_carriers"})
    top = isfam.sort_values("n", ascending=False).drop_duplicates("unitig_idx")[["unitig_idx", "is_family"]]
    return top.rename(columns={"is_family": "dominant_is_family"}).merge(n_is, on="unitig_idx")


def _sum_copies(files: list[Path]) -> pd.DataFrame:
    """Per unitig: mean ASM copy number across its carriers (IS-borne unitigs are typically multi-copy)."""
    parts = [pd.read_csv(f, sep="\t") for f in files if f.stat().st_size > 1]
    if not parts:
        return pd.DataFrame(columns=["unitig_idx", "mean_copies"])
    d = pd.concat(parts, ignore_index=True).groupby("unitig_idx", as_index=False).agg({"copies_sum": "sum", "copies_n": "sum"})
    d["mean_copies"] = (d["copies_sum"] / d["copies_n"]).round(3)
    return d[["unitig_idx", "mean_copies"]]


def _rollup_pattern_group(per_unitig: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Aggregate per-unitig counts to per-pattern_group (IS-refined) class fractions."""
    cnt = [c for c in classes if c in per_unitig.columns]
    g = per_unitig.groupby("pattern_group", as_index=False).agg(
        {"unitig_idx": "count", "n_carriers": "sum", **dict.fromkeys(cnt, "sum")})
    g = g.rename(columns={"unitig_idx": "n_member_unitigs"})
    for c in cnt:
        g[f"frac_{c}"] = (g[c] / g["n_carriers"]).round(4)
    return g


def _rollup_overall(per_unitig: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Global + by-direction + by-af-bin (IS-refined) class fractions (the headline 'where are they')."""
    cnt = [c for c in classes if c in per_unitig.columns]
    rows = []

    def _agg(label: str, sub: pd.DataFrame) -> dict[str, Any]:
        tot = sub["n_carriers"].sum()
        d = {"stratum": label, "n_unitigs": len(sub), "n_carrier_obs": int(tot)}
        for c in cnt:
            d[f"frac_{c}"] = round(sub[c].sum() / tot, 4) if tot else 0.0
        return d

    rows.append(_agg("ALL", per_unitig))
    if "direction" in per_unitig.columns:
        for dirn, sub in per_unitig.groupby("direction"):
            rows.append(_agg(f"direction={dirn}", sub))
    if "af" in per_unitig.columns:
        bins = pd.cut(per_unitig["af"], [0, 0.05, 0.2, 0.5, 0.7, 1.0])
        for b, sub in per_unitig.groupby(bins, observed=True):
            rows.append(_agg(f"af={b}", sub))
    return pd.DataFrame(rows)


def _write_combine_manifest(out: Path, scratch: Path, per_unitig: pd.DataFrame, spot: pd.DataFrame, classes: list[str]) -> None:
    """Aggregate shard QC → manifest (ASM-recall, discordances, class distribution, IS, timing)."""
    qcs = [json.loads(f.read_text()) for f in sorted(scratch.glob("align_shard_*.qc.json"))]
    den = sum(q.get("asm_recall_den", 0) for q in qcs)
    num = sum(q.get("asm_recall_num", 0) for q in qcs)
    cnt = [c for c in classes if c in per_unitig.columns]
    manifest = {
        "phase": "combine", "n_unitigs": int(len(per_unitig)),
        "n_carriers": sum(q.get("n_carriers", 0) for q in qcs),
        "n_pairs": sum(q.get("n_pairs", 0) for q in qcs),
        "asm_recall": round(num / den, 4) if den else None,
        "found_not_expected": sum(q.get("found_not_expected", 0) for q in qcs),
        "no_genomad_carriers": sum(q.get("no_genomad_carriers", 0) for q in qcs),
        "no_isescan_carriers": sum(q.get("no_isescan_carriers", 0) for q in qcs),
        "class_pair_totals": {c: int(per_unitig[c].sum()) for c in cnt},
        "spotcheck_pass": int((spot["in_genomad_long"] == True).sum()) if "in_genomad_long" in spot else 0,  # noqa: E712
        "spotcheck_fail": int((spot["in_genomad_long"] == False).sum()) if "in_genomad_long" in spot else 0,  # noqa: E712
        "total_align_seconds": round(sum(q.get("seconds", 0.0) for q in qcs), 1),
    }
    (out / "combine_manifest.json").write_text(json.dumps(manifest, indent=2))


# --------------------------------------------------------------------------------------------------
# stratify — post-hoc re-aggregation of the per-pair parquet by clonal structure
# --------------------------------------------------------------------------------------------------
_STRATA_COLS = {"sublineage": "Sublineage", "clonal_group": "Clonal group"}


def _load_strata(strata_csv: Path, min_group_size: int, carriers: set[str]) -> dict[str, dict[str, str]]:
    """Read Sample → {sublineage, clonal_group}; collapse groups with < ``min_group_size`` carriers to 'other'.

    'Big' is decided over the **carrier** set (the genomes actually mapped), matching the LMM's
    ≥100-sample sublineage rule. Missing / blank labels become 'unknown'.
    """
    df = pd.read_csv(strata_csv, usecols=["Sample", *_STRATA_COLS.values()], low_memory=False)
    df["Sample"] = df["Sample"].astype(str)
    df = df[df["Sample"].isin(carriers)].drop_duplicates("Sample")
    maps: dict[str, dict[str, str]] = {}
    for level, col in _STRATA_COLS.items():
        lab = df[col].fillna("unknown").astype(str).replace({"": "unknown", "nan": "unknown"})
        big = lab.value_counts()
        big = set(big.index[big >= min_group_size]) - {"unknown"}
        lab = lab.where(lab.isin(big | {"unknown"}), "other")
        maps[level] = dict(zip(df["Sample"], lab, strict=True))
    return maps


def run_stratify(args: argparse.Namespace) -> None:
    """``stratify``: re-aggregate the per-(unitig, carrier) parquet by big sublineage / clonal group.

    Answers "are the plasmid/prophage/chromosomal placements clonal-structure-specific?" — the direct
    follow-up to the within-sublineage shuffle test. No re-mapping: reads the existing parquet detail.
    """
    import pyarrow.parquet as pq

    out = args.out_dir
    id_map = pd.read_csv(out / "id_map.tsv", sep="\t", usecols=["unitig_idx", "direction"])
    dir_by_idx = dict(zip(id_map["unitig_idx"], id_map["direction"].astype(str), strict=True))
    carriers = set(pd.read_csv(out / "carriers.resolved.tsv", sep="\t")["Sample"].astype(str))
    strata = _load_strata(args.strata_csv, args.min_group_size, carriers)

    acc: Counter = Counter()                      # (level, group, direction, refined_class) -> n_pairs
    seen: dict[str, set[str]] = defaultdict(set)  # level -> refined classes observed
    n_carr: dict[tuple[str, str], int] = {}       # (level, group) -> distinct carriers
    for level, smap in strata.items():
        vc = Counter(smap.values())
        for g, n in vc.items():
            n_carr[(level, g)] = n
        n_carr[(level, "ALL")] = sum(vc.values())
    parts = sorted((out / "mge_hits.parquet").glob("*.parquet"))
    for part in parts:
        df = pq.read_table(part, columns=["unitig_idx", "Sample", "mge_class", "is_element"]).to_pandas()
        df["direction"] = df["unitig_idx"].map(dir_by_idx).fillna("NA")
        st = df["is_element"].map({True: "_IS", False: "", None: "_naIS"}).fillna("_naIS")
        mask = df["mge_class"].isin(["plasmid", "prophage", "chromosomal"])
        df["rclass"] = df["mge_class"].where(~mask, df["mge_class"] + st)
        for level, smap in strata.items():
            df["_g"] = df["Sample"].map(smap).fillna("unknown")
            for (g, d, cls), n in df.groupby(["_g", "direction", "rclass"]).size().items():
                seen[level].add(cls)
                for dk in (d, "all"):
                    acc[(level, g, dk, cls)] += n
                    acc[(level, "ALL", dk, cls)] += n

    for level in strata:
        classes = sorted(seen[level])
        rows = []
        for g, d in sorted({(k[1], k[2]) for k in acc if k[0] == level}):
            counts = {c: acc[(level, g, d, c)] for c in classes}
            tot = sum(counts.values())
            row = {level: g, "direction": d, "n_carriers": n_carr.get((level, g)), "n_pairs": tot}
            for c in classes:
                row[f"frac_{c}"] = round(counts[c] / tot, 4) if tot else 0.0
            rows.append(row)
        pd.DataFrame(rows).to_csv(out / f"mge_by_{level}.tsv", sep="\t", index=False)
    print(f"wrote mge_by_sublineage.tsv, mge_by_clonal_group.tsv (min_group_size={args.min_group_size})", file=sys.stderr)


# --------------------------------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------------------------------
def run_smoke(args: argparse.Namespace) -> None:
    """``smoke``: select + align (K carriers that carry many unitigs) + combine inline, timed."""
    run_select(args)
    out = args.out_dir
    resolved = pd.read_csv(out / "carriers.resolved.tsv", sep="\t")
    args.scratch_dir = args.scratch_dir / "smoke"   # isolate smoke shard parts from a full run's scratch
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.scratch_dir.glob("align_shard_*"):
        stale.unlink()
    # Point align at a 1-shard slice restricted to the first K carriers.
    args.carrier_shard_index, args.n_shards = 0, 1
    orig = out / "carriers.resolved.tsv"
    backup = out / "carriers.resolved.full.tsv"
    orig.rename(backup)
    resolved.head(args.smoke).to_csv(orig, sep="\t", index=False)
    try:
        run_align(args)
        run_combine(args)
    finally:
        backup.rename(orig)
    print("\n=== SMOKE combine_manifest ===\n" + (out / "combine_manifest.json").read_text(), file=sys.stderr)


# --------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    """CLI entry point — dispatch on ``--phase``."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", required=True, choices=("select", "align", "combine", "smoke", "stratify"))
    p.add_argument("--hits-tsv", type=Path, help="blood_vs_faeces_unitig_hits_annotated.tsv (select/smoke).")
    p.add_argument("--unitig-matrix", type=Path, help="unitigs.pyseer.gz (select/smoke).")
    p.add_argument("--genomad-root", type=Path, required=True, help="<DATA>/david/processed/genomad.")
    p.add_argument("--metadata", type=Path, help="metadata_v2 TSV for assembly resolution (select/smoke).")
    p.add_argument("--out-dir", type=Path, required=True, help="Durable outputs on project_k.")
    p.add_argument("--scratch-dir", type=Path, required=True, help="Per-shard scratch (RDS hpc-work).")
    p.add_argument("--decomp-threads", type=int, default=4, help="pigz threads for the one-time matrix scan.")
    p.add_argument("--carrier-shard-index", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--genomad-probe-n", type=int, default=20, help="Carriers to probe for geNomad layout.")
    p.add_argument("--spot-n", type=int, default=3, help="Spot-check records per class per group (approx).")
    p.add_argument("--smoke", type=int, default=30, help="smoke phase: number of carriers to map.")
    p.add_argument("--isescan-lookup", type=Path, help="TSV Sample<TAB>path → per-genome ISEScan .fa.csv (align).")
    p.add_argument("--strata-csv", type=Path, help="Cohort CSV with Sample, Sublineage, 'Clonal group' (stratify).")
    p.add_argument("--min-group-size", type=int, default=100, help="Min carriers for a 'big' sublineage/CG (stratify).")
    args = p.parse_args(argv)

    if args.phase in ("select", "smoke"):
        for req in ("hits_tsv", "unitig_matrix", "metadata"):
            if getattr(args, req) is None:
                p.error(f"--{req.replace('_', '-')} is required for --phase {args.phase}")
    if args.phase == "stratify" and args.strata_csv is None:
        p.error("--strata-csv is required for --phase stratify")

    {"select": run_select, "align": run_align, "combine": run_combine,
     "smoke": run_smoke, "stratify": run_stratify}[args.phase](args)


if __name__ == "__main__":
    main()
