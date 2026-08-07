r"""Generic unitig placement + select engine — the classifier-agnostic core of the unitig-mapping jobs.

A coloured de-Bruijn unitig present in a sample is an **exact substring** of that sample's assembly, so
locating every GWAS-hit unitig across the carriers that have it is exact-substring matching (both
strands) via a single **Aho-Corasick** automaton — cost ∝ genome size, not × unitigs. This module owns
that placement machinery and the one-time cohort **select** step (build the unitig id_map, extract the
cached hit sub-matrix from the 77 GB unitig matrix, resolve carriers → assemblies). It carries **no
classifier**: downstream jobs (`map_unitig_hits_genomad` = geNomad/IS class; `annotate_unitig_coding` =
CDS-vs-IGR) import these helpers and add their own align/combine. Naming stays generic on purpose — the
select/placement engine is not geNomad-specific, so it must not live under a geNomad name.

Phase here: ``select`` (id_map + cached sub-matrix + resolved carriers). align/combine live in the
classifier modules.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:  # package import when bac_pyseer is on the path (editable install)
    from bac_pyseer.kleb_iso_source.resolve_assembly_paths import resolve as resolve_assembly_paths
except ImportError:  # invoked as a bare script — add the script dir to sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from resolve_assembly_paths import resolve as resolve_assembly_paths  # type: ignore[no-redef]

# Source tags for the placement scan (a unitig may occur on a plasmid/virus contig and/or the assembly).
_TAG_PLASMID, _TAG_VIRUS, _TAG_ASM = "PLASMID", "VIRUS", "ASM"
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
    """Stream the small cached sub-matrix → (union of carrier Sample IDs, set of hit seqs actually present)."""
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
# Aho-Corasick automaton + carrier scan
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

    Returns ``(found, asm_pos)``: ``found[idx][tag] = {seq_name…}`` (for classification) and
    ``asm_pos[idx] = [(contig, end_off)…]`` — every ASM occurrence, for coordinate/overlap work.
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


# --------------------------------------------------------------------------------------------------
# shard IO + aggregation helpers shared by the classifier modules
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


def _sum_long(files: list[Path], keys: list[str]) -> pd.DataFrame:
    """Concatenate long shard tables and sum ``n`` over ``keys``."""
    parts = [pd.read_csv(f, sep="\t") for f in files if f.stat().st_size > 1]
    if not parts:
        return pd.DataFrame(columns=[*keys, "n"])
    return pd.concat(parts, ignore_index=True).groupby(keys, as_index=False)["n"].sum()


def _rollup_pattern_group(per_unitig: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Aggregate per-unitig counts to per-pattern_group class fractions."""
    cnt = [c for c in classes if c in per_unitig.columns]
    g = per_unitig.groupby("pattern_group", as_index=False).agg(
        {"unitig_idx": "count", "n_carriers": "sum", **dict.fromkeys(cnt, "sum")})
    g = g.rename(columns={"unitig_idx": "n_member_unitigs"})
    for c in cnt:
        g[f"frac_{c}"] = (g[c] / g["n_carriers"]).round(4)
    return g


def _rollup_overall(per_unitig: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Global + by-direction + by-af-bin class fractions (the headline 'where are they')."""
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


_STRATA_COLS = {"sublineage": "Sublineage", "clonal_group": "Clonal group"}


def _load_strata(strata_csv: Path, min_group_size: int, carriers: set[str]) -> dict[str, dict[str, str]]:
    """Read Sample → {sublineage, clonal_group}; collapse groups with < ``min_group_size`` carriers to 'other'."""
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


# --------------------------------------------------------------------------------------------------
# select (generic: id_map + cached sub-matrix + resolved carriers) — no classifier, no geNomad
# --------------------------------------------------------------------------------------------------
def run_select(args: argparse.Namespace) -> dict[str, Any]:
    """``select``: id_map for ALL hit unitigs; cached hit sub-matrix; resolved carrier→assembly list.

    Returns the manifest dict (also written to ``select_manifest.json``) so a classifier-specific
    wrapper can augment it (e.g. geNomad layout probe) without re-reading.
    """
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.scratch_dir.glob("align_shard_*"):  # avoid mixing a previous run's shard parts
        stale.unlink()
    hits = pd.read_csv(args.hits_tsv, sep="\t", low_memory=False).reset_index(drop=True)
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

    manifest = {
        "phase": "select", "n_unitigs": int(len(id_map)),
        "n_pattern_groups": int(id_map["pattern_group"].nunique()) if "pattern_group" in id_map else None,
        "n_carrier_union": len(carrier_union), "n_resolved_assemblies": int(len(res)),
        "n_hit_seqs_matched": len(matched), "n_join_misses": len(missing),
        "match_method": "exact-substring (Aho-Corasick, fwd+revcomp)",
    }
    (out / "select_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), file=sys.stderr)
    return manifest


def main(argv: list[str] | None = None) -> None:
    """CLI — the generic ``select`` phase (classifier align/combine live in the classifier modules)."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description="Generic unitig select (id_map + cached sub-matrix + carriers).")
    p.add_argument("--phase", required=True, choices=("select",))
    p.add_argument("--hits-tsv", type=Path, required=True, help="Hit unitig TSV (needs a 'variant' column).")
    p.add_argument("--unitig-matrix", type=Path, required=True, help="unitigs.pyseer.gz (the 77 GB matrix).")
    p.add_argument("--metadata", type=Path, required=True, help="metadata_v2 TSV for assembly resolution.")
    p.add_argument("--out-dir", type=Path, required=True, help="Durable outputs on project_k.")
    p.add_argument("--scratch-dir", type=Path, required=True, help="Scratch (RDS hpc-work).")
    p.add_argument("--decomp-threads", type=int, default=4, help="pigz threads for the one-time matrix scan.")
    args = p.parse_args(argv)
    run_select(args)


if __name__ == "__main__":
    main()
