r"""Map invasion-associated GWAS unitig hits back onto geNomad MGE calls — the HGT test.

The unitig (accessory) axis of the *Klebsiella* invasion GWAS concentrates its signal at **common
allele frequency** and is **LD-redundant**: λ≈24 is one or a few co-inherited biological events (a
niche-associated megaplasmid, the hypothesis) multiplied across thousands of unitigs
(``docs/PROGRESS_UNITIGS.md``). This module runs the **direct HGT-vs-chromosomal test**: do the
member unitigs of an invasion-associated ``pattern_group`` land on the **same plasmid replicon** or
the **same prophage** across the genomes that carry them — or do they **scatter** / sit on the
**chromosome**?

It is a genuine test of a hypothesis. Refutation (scatter, chromosomal, or no geNomad call) is a
first-class outcome and is reported plainly, never hidden. "Co-inherited plasmid/prophage" stays an
*untested* hypothesis until this mapping supports it; nothing here asserts a causal-vs-lineage verdict.

Method
------
For each carrier of a chosen unitig, align the unitig (an exact DNA substring of that carrier's
assembly by construction) into ONE tagged target FASTA built from that carrier's geNomad extracts +
full assembly:

* ``<Sample>_plasmid.fna`` contigs, headers tagged ``PLASMID|<seq_name>``;
* ``<Sample>_virus.fna`` sequences, tagged ``VIRUS|<seq_name>`` — geNomad **excises** integrated
  prophage into this file, so a prophage-borne unitig hits ``VIRUS|`` even though in the full
  assembly it sits inside a host chromosomal contig (mapping to the assembly alone would miss it);
* the full assembly contigs, tagged ``ASM|<contig>``.

A unitig hit is classified by priority **PLASMID > VIRUS > ASM-only** (nesting: a plasmid unitig also
hits ``ASM|``; a prophage unitig hits both ``VIRUS|`` and ``ASM|``). A carrier with no geNomad output
is ``unknown_no_genomad`` — **not** chromosomal, which would fabricate a hypothesis-refuting signal.

Phases (``--phase``), mirroring the sharded unitig-LMM job:

* ``select``  — pick the top ``pattern_group``\ s (invasion-oriented), write the query FASTA, scan the
  6.28M-row unitig matrix once for carrier sets, resolve carriers → assembly paths, probe the geNomad
  layout. Runs once.
* ``align``   — array body: one shard of the carrier list; build each carrier's tagged target, run one
  minimap2 over all query unitigs, classify → a per-shard long TSV in scratch.
* ``combine`` — concatenate shard TSVs → the master long TSV, per-group summary (same-plasmid /
  same-prophage / chromosomal / scatter fractions), a geNomad spot-check, and a Phase-2 estimate.
* ``smoke``   — select + align (``K`` carriers) + combine inline, timing seconds/genome & seconds/unitig
  and the ``ASM|`` recall QC, to validate the preset and size Phase-2 before the full array.

Assembly resolution reuses ``resolve_assembly_paths.resolve()``; minimap2 is the pixi ``bac_pyseer``
binary (pass ``--minimap2-bin``). Results go to project_k / scratch only, never ``$HOME``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

try:  # package import when bac_pyseer is on the path (editable install)
    from bac_pyseer.kleb_iso_source.resolve_assembly_paths import resolve as resolve_assembly_paths
except ImportError:  # invoked as a bare script — add the script dir to sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from resolve_assembly_paths import resolve as resolve_assembly_paths  # type: ignore[no-redef]

# --- geNomad layout (per the task brief; centralised so a real-layout surprise is a one-line fix) ---
# Per-sample extracts live at <root>/per_sample/<key>/<key>_summary/<key>_{plasmid,virus}.fna, where
# <key> is the Sample or, for paired short-read samples, "<Sample>__sr". The keyed long tables live at
# <root>/genomad_{plasmid,virus}_summary_long.tsv (columns include Sample, seq_name).
_SR_SUFFIX = "__sr"
_PLASMID_LONG = "genomad_plasmid_summary_long.tsv"
_VIRUS_LONG = "genomad_virus_summary_long.tsv"

# minimap2 preset for short, near-exact DNA queries (asm5/asm10 seed too sparsely for ~31 bp unitigs).
# -x sr (k=21,w=11) always seeds a >=k unitig; -c gives base-level alignment (accurate nmatch);
# secondary hits kept so a unitig mapping to several replicons/copies is fully recorded.
_MINIMAP2_PRESET = ("-c", "-x", "sr", "--secondary=yes", "-N", "50", "-p", "0.1")

# Classification tags written into target headers (routed on the prefix before the first "|").
_TAG_PLASMID, _TAG_VIRUS, _TAG_ASM = "PLASMID", "VIRUS", "ASM"
_CLASS_BY_TAG = {_TAG_PLASMID: "plasmid", _TAG_VIRUS: "prophage", _TAG_ASM: "chromosomal"}


# --------------------------------------------------------------------------------------------------
# select — target groups, query FASTA, carrier sets, assembly + geNomad resolution
# --------------------------------------------------------------------------------------------------
def select_target_groups(hits: pd.DataFrame, top_n: int, min_af: float) -> pd.DataFrame:
    """Pick the top invasion-oriented pattern groups and return all their member unitigs.

    Parameters
    ----------
    hits
        The annotated unitig-hits table (``blood_vs_faeces_unitig_hits_annotated.tsv``); every row is
        already Bonferroni-significant. ``variant`` is the unitig DNA sequence.
    top_n
        Number of pattern groups to keep.
    min_af
        Drop hits below this allele frequency before scoring (the "common-af" lever; HGT spreads wide).

    Returns
    -------
    pandas.DataFrame
        Member rows of the chosen groups, with an added integer ``group_rank`` (0 = strongest) and the
        per-group ``group_score``. Score per group = ``af * n_in_pattern * var_explained_pct`` on the
        group's representative row.
    """
    df = hits.copy()
    df["af"] = pd.to_numeric(df["af"], errors="coerce")
    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    df["var_explained_pct"] = pd.to_numeric(df["var_explained_pct"], errors="coerce")
    df["n_in_pattern"] = pd.to_numeric(df["n_in_pattern"], errors="coerce")

    # invasion orientation: unitig PRESENCE confers invasion (blood). invasion_allele=="ALT" == beta>0.
    if "invasion_allele" in df.columns:
        inv = df["invasion_allele"].astype(str).str.upper().eq("ALT")
    else:
        inv = df["beta"] > 0
    df = df[inv & (df["af"] >= min_af)].copy()
    if df.empty:
        raise SystemExit(f"no invasion-oriented hits with af >= {min_af}")

    score = df["af"] * df["n_in_pattern"] * df["var_explained_pct"]
    df = df.assign(group_score=score)
    per_group = df.groupby("pattern_group", sort=False)["group_score"].max().sort_values(ascending=False)
    chosen = list(per_group.index[:top_n])
    rank = {g: i for i, g in enumerate(chosen)}
    out = df[df["pattern_group"].isin(chosen)].copy()
    out["group_rank"] = out["pattern_group"].map(rank)
    out["group_score"] = out["pattern_group"].map(per_group)
    return out.sort_values(["group_rank", "var_explained_pct"], ascending=[True, False]).reset_index(drop=True)


def write_query_fasta(members: pd.DataFrame, out_fasta: Path) -> pd.DataFrame:
    """Write one query FASTA of all member unitigs; return the ``query_id`` id-map.

    Header is ``>pg{pattern_group}__u{idx}``; the sequence is the ``variant`` column.
    """
    id_rows: list[dict[str, Any]] = []
    with out_fasta.open("w") as fh:
        for idx, row in enumerate(members.itertuples(index=False)):
            qid = f"pg{row.pattern_group}__u{idx}"
            fh.write(f">{qid}\n{row.variant}\n")
            id_rows.append({
                "query_id": qid, "pattern_group": row.pattern_group, "group_rank": row.group_rank,
                "variant": row.variant, "af": row.af, "beta": row.beta,
                "var_explained_pct": row.var_explained_pct,
            })
    return pd.DataFrame(id_rows)


def extract_hit_submatrix(matrix_gz: Path, all_hit_seqs: set[str], submatrix_path: Path, *, decomp_threads: int = 4) -> int:
    """Extract only the hit-unitig rows from the 77 GB matrix into a small **cached** sub-matrix.

    A single C-speed streaming hash-join — ``pigz -dc`` (parallel gunzip; ``zcat`` fallback) piped to an
    ``awk`` that loads the ``all_hit_seqs`` into a hash and keeps a matrix line iff its left field
    (``substr($0, 1, index($0," | ")-1)``) is in that hash. ``awk`` touches only ``$0`` (never ``$1``),
    so it does **not** field-split the giant carrier lists — the reason the pure-Python ``gzip`` scan was
    ~30 min and this is a few. The 6.28M-row / 77 GB matrix is read exactly **once**, ever: the result is
    the reusable index (``index once, match once``) and this returns immediately if it already exists.

    Returns the row count of the sub-matrix (``-1`` if served from the existing cache).
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


def carriers_from_submatrix(submatrix_path: Path, target_seqs: dict[str, str]) -> tuple[dict[str, list[str]], set[str], int]:
    """Read the small cached sub-matrix → carrier sets for the target unitigs + the Phase-2 union.

    Parameters
    ----------
    submatrix_path
        The cached hit sub-matrix (``<unitig_seq> | SampleA:1 …`` rows) from :func:`extract_hit_submatrix`.
    target_seqs
        ``{unitig_sequence: query_id}`` for the chosen (top-group) unitigs — the ones we classify.

    Returns
    -------
    carriers : dict
        ``{query_id: [Sample, …]}`` for the target unitigs.
    phase2_union : set
        Union of carrier Sample IDs across **all** hit unitigs (the Phase-2 scaling driver).
    n_matched : int
        Number of hit rows read (join-coverage check vs the hits table).
    """
    carriers: dict[str, list[str]] = {}
    phase2_union: set[str] = set()
    matched = 0
    with submatrix_path.open() as fh:
        for line in fh:
            seq, sep, rest = line.partition(" | ")
            if not sep:
                continue
            matched += 1
            samples = [tok.rpartition(":")[0] for tok in rest.split()]
            phase2_union.update(samples)
            qid = target_seqs.get(seq)
            if qid is not None:
                carriers[qid] = samples
    return carriers, phase2_union, matched


def resolve_genomad_paths(sample: str, genomad_root: Path) -> dict[str, Any]:
    """Resolve a Sample's geNomad ``_plasmid.fna`` / ``_virus.fna``, trying ``<Sample>`` then ``__sr``.

    Returns a dict with ``key`` (the resolved per-sample key, or None), ``plasmid_fna`` / ``virus_fna``
    (Path or None if absent/empty), and ``found`` (whether the summary dir exists at all).
    """
    for key in (sample, f"{sample}{_SR_SUFFIX}"):
        summ = genomad_root / "per_sample" / key / f"{key}_summary"
        if not summ.is_dir():
            continue
        plasmid = summ / f"{key}_plasmid.fna"
        virus = summ / f"{key}_virus.fna"
        return {
            "key": key,
            "plasmid_fna": plasmid if plasmid.is_file() and plasmid.stat().st_size > 0 else None,
            "virus_fna": virus if virus.is_file() and virus.stat().st_size > 0 else None,
            "found": True,
        }
    return {"key": None, "plasmid_fna": None, "virus_fna": None, "found": False}


# --------------------------------------------------------------------------------------------------
# align — build tagged target, minimap2, classify
# --------------------------------------------------------------------------------------------------
def _iter_fasta(path: Path) -> Any:
    """Yield ``(header_without_'>', sequence)`` from a plain or gzipped FASTA."""
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


def build_combined_target(plasmid_fna: Path | None, virus_fna: Path | None, assembly_fna: Path, out_fna: Path) -> dict[str, int]:
    """Concatenate a carrier's plasmid/virus/assembly FASTAs into one tag-prefixed target.

    Headers become ``PLASMID|<seq>``, ``VIRUS|<seq>``, ``ASM|<contig>``. A missing/empty plasmid or
    virus class is simply omitted (its unitigs then fall through to the assembly = chromosomal).
    Returns per-tag sequence counts.
    """
    counts = {_TAG_PLASMID: 0, _TAG_VIRUS: 0, _TAG_ASM: 0}
    sources = [(_TAG_PLASMID, plasmid_fna), (_TAG_VIRUS, virus_fna), (_TAG_ASM, assembly_fna)]
    with out_fna.open("w") as out:
        for tag, path in sources:
            if path is None:
                continue
            for header, seq in _iter_fasta(path):
                out.write(f">{tag}|{header}\n{seq}\n")
                counts[tag] += 1
    return counts


def run_minimap2(minimap2_bin: str, target: Path, query: Path, out_paf: Path, *, threads: int) -> None:
    """Run minimap2 with the short-near-exact preset (target = carrier, query = all unitigs) → PAF."""
    cmd = [minimap2_bin, *_MINIMAP2_PRESET, "-t", str(threads), str(target), str(query)]
    with out_paf.open("w") as fh:
        res = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"minimap2 failed (rc={res.returncode}): {res.stderr.decode('utf-8', errors='replace')}")


def parse_paf(paf_path: Path, min_ident: float, min_cov: float, min_matchlen: int) -> dict[str, list[dict[str, Any]]]:
    """Parse a carrier's PAF → ``{query_id: [qualifying hit dicts]}`` (tag, seq_name, ident, cov).

    Target names are split on the first ``|`` into tag + seq_name (a contig literally containing ``|``
    keeps the remainder as its seq_name). Field math mirrors ``extract_proteins_from_gff_fna._parse_amr_paf``.
    """
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with paf_path.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 11:
                continue
            qname = parts[0]
            qlen, qstart, qend = int(parts[1]), int(parts[2]), int(parts[3])
            tname = parts[5]
            nmatch, alnlen = int(parts[9]), int(parts[10])
            if alnlen == 0 or qlen == 0:
                continue
            ident = nmatch / alnlen
            cov = (qend - qstart) / qlen
            if ident < min_ident or cov < min_cov or nmatch < min_matchlen:
                continue
            tag, _, seq_name = tname.partition("|")
            by_query[qname].append({
                "tag": tag, "seq_name": seq_name, "ident": round(ident, 4),
                "cov": round(cov, 4), "matchlen": nmatch,
            })
    return by_query


def classify_hit(hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify one unitig's qualifying hits in a carrier by priority PLASMID > VIRUS > ASM-only.

    Returns ``mge_class`` (plasmid / prophage / chromosomal / unmapped), the distinct ``seq_names`` of
    the winning class (the "same replicon/prophage" evidence), a ``multi_replicon`` flag, whether an
    ``ASM|`` hit was seen at all (``asm_recall`` — the built-in ground-truth QC), and best ident/cov.
    """
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for h in hits:
        by_tag[h["tag"]].append(h)
    asm_recall = _TAG_ASM in by_tag
    for tag in (_TAG_PLASMID, _TAG_VIRUS, _TAG_ASM):
        if tag in by_tag:
            winners = by_tag[tag]
            seqs = sorted({h["seq_name"] for h in winners})
            best = max(winners, key=lambda h: (h["ident"], h["cov"]))
            return {
                "mge_class": _CLASS_BY_TAG[tag], "seq_names": ";".join(seqs),
                "n_distinct_seqnames": len(seqs), "multi_replicon": len(seqs) > 1,
                "asm_recall": asm_recall, "best_ident": best["ident"], "best_cov": best["cov"],
            }
    return {
        "mge_class": "unmapped", "seq_names": "", "n_distinct_seqnames": 0,
        "multi_replicon": False, "asm_recall": asm_recall, "best_ident": None, "best_cov": None,
    }


def align_carrier(
    sample: str, assembly_fna: Path, expected_qids: list[str], id_map: pd.DataFrame,
    genomad_root: Path, query_fasta: Path, minimap2_bin: str, tmpdir: Path, *,
    min_ident: float, min_cov: float, min_matchlen: int, threads: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Map all query unitigs into one carrier's tagged target → classified records for expected pairs.

    ``expected_qids`` are the query IDs this carrier actually carries (from the matrix); only those
    (unitig, carrier) pairs are recorded — that is the ground-truth set the summary and ``ASM|`` recall
    QC use. Returns ``(records, timings)`` where timings has ``build``/``mm2``/``parse`` seconds.
    """
    gpaths = resolve_genomad_paths(sample, genomad_root)
    id_lookup = id_map.set_index("query_id")

    t0 = time.perf_counter()
    if not gpaths["found"]:
        # No geNomad output: cannot separate MGE from chromosome. Record honestly, do not align.
        recs = [
            {"Sample": sample, "genomad_key": None, **_id_fields(id_lookup, qid),
             "mge_class": "unknown_no_genomad", "seq_names": "", "n_distinct_seqnames": 0,
             "multi_replicon": False, "asm_recall": None, "best_ident": None, "best_cov": None}
            for qid in expected_qids
        ]
        return recs, {"build": 0.0, "mm2": 0.0, "parse": 0.0}

    target = tmpdir / f"{sample}_target.fna"
    build_combined_target(gpaths["plasmid_fna"], gpaths["virus_fna"], assembly_fna, target)
    t1 = time.perf_counter()
    paf = tmpdir / f"{sample}.paf"
    run_minimap2(minimap2_bin, target, query_fasta, paf, threads=threads)
    t2 = time.perf_counter()
    by_query = parse_paf(paf, min_ident, min_cov, min_matchlen)
    t3 = time.perf_counter()

    records: list[dict[str, Any]] = []
    for qid in expected_qids:
        cls = classify_hit(by_query.get(qid, []))
        records.append({"Sample": sample, "genomad_key": gpaths["key"], **_id_fields(id_lookup, qid), **cls})
    target.unlink(missing_ok=True)
    paf.unlink(missing_ok=True)
    return records, {"build": t1 - t0, "mm2": t2 - t1, "parse": t3 - t2}


def _id_fields(id_lookup: pd.DataFrame, qid: str) -> dict[str, Any]:
    """Pull the (pattern_group, group_rank, variant length) descriptors for a query_id."""
    row = id_lookup.loc[qid]
    return {
        "query_id": qid, "pattern_group": row["pattern_group"], "group_rank": row["group_rank"],
        "unitig_len": len(str(row["variant"])),
    }


# --------------------------------------------------------------------------------------------------
# combine — per-group summary, geNomad spot-check
# --------------------------------------------------------------------------------------------------
def summarise_groups(long: pd.DataFrame) -> pd.DataFrame:
    """Per ``pattern_group``: class fractions + the dominant plasmid/prophage seq_name (convergence).

    Fractions are over the geNomad-informative denominator (``unknown_no_genomad`` and ``unmapped``
    reported as their own fractions, not folded into chromosomal). ``top_plasmid_frac`` /
    ``top_prophage_frac`` are the "same plasmid / same prophage across carriers" convergence metric;
    high ``scatter_index`` or high ``frac_chromosomal`` = HGT hypothesis refuted / qualified.
    """
    rows: list[dict[str, Any]] = []
    for pg, g in long.groupby("pattern_group", sort=False):
        n = len(g)
        vc = g["mge_class"].value_counts()
        frac = {c: round(vc.get(c, 0) / n, 4) for c in ("plasmid", "prophage", "chromosomal", "unknown_no_genomad", "unmapped")}
        top_plasmid, top_plasmid_frac = _dominant_seqname(g, "plasmid", n)
        top_prophage, top_prophage_frac = _dominant_seqname(g, "prophage", n)
        rows.append({
            "pattern_group": pg, "group_rank": g["group_rank"].iloc[0], "n_obs": n,
            "n_carriers": g["Sample"].nunique(), "n_member_unitigs": g["query_id"].nunique(),
            "frac_plasmid": frac["plasmid"], "frac_prophage": frac["prophage"],
            "frac_chromosomal": frac["chromosomal"], "frac_unknown_no_genomad": frac["unknown_no_genomad"],
            "frac_unmapped": frac["unmapped"],
            "top_plasmid_seqname": top_plasmid, "top_plasmid_frac": top_plasmid_frac,
            "top_prophage_seqname": top_prophage, "top_prophage_frac": top_prophage_frac,
            "n_distinct_plasmid_seqnames": _n_distinct_seqnames(g, "plasmid"),
            "n_distinct_prophage_seqnames": _n_distinct_seqnames(g, "prophage"),
        })
    return pd.DataFrame(rows)


def _explode_seqnames(g: pd.DataFrame, mge_class: str) -> pd.Series:
    """All (semicolon-joined) seq_names for one class, exploded to one per row."""
    sub = g[g["mge_class"] == mge_class]["seq_names"]
    if sub.empty:
        return pd.Series(dtype=str)
    return sub[sub.astype(bool)].str.split(";").explode()


def _dominant_seqname(g: pd.DataFrame, mge_class: str, n: int) -> tuple[str | None, float]:
    """The most common seq_name of ``mge_class`` and its fraction over all ``n`` group observations."""
    exploded = _explode_seqnames(g, mge_class)
    if exploded.empty:
        return None, 0.0
    top = exploded.value_counts()
    return str(top.index[0]), round(int(top.iloc[0]) / n, 4)


def _n_distinct_seqnames(g: pd.DataFrame, mge_class: str) -> int:
    """Count distinct seq_names seen for a class within a group (scatter breadth)."""
    return int(_explode_seqnames(g, mge_class).nunique())


def crosscheck_genomad(long: pd.DataFrame, genomad_root: Path, n_spot: int) -> pd.DataFrame:
    """Spot-check classified (Sample, seq_name) against ``genomad_{plasmid,virus}_summary_long.tsv``.

    Confirms up to ``n_spot`` plasmid- and ``n_spot`` prophage-classified records per group appear in
    geNomad's own long table with the matching class. A mismatch is a header/tag bug, not biology —
    it is surfaced (``pass=False``), not swallowed.
    """
    keys = {}
    for cls, fname in (("plasmid", _PLASMID_LONG), ("prophage", _VIRUS_LONG)):
        p = genomad_root / fname
        if p.is_file():
            t = pd.read_csv(p, sep="\t", usecols=lambda c: c in ("Sample", "seq_name"))
            keys[cls] = set(zip(t["Sample"].astype(str), t["seq_name"].astype(str), strict=True))
        else:
            keys[cls] = None  # long table absent — record as unavailable rather than a spurious fail

    rows: list[dict[str, Any]] = []
    for cls in ("plasmid", "prophage"):
        sub = long[(long["mge_class"] == cls) & long["seq_names"].astype(bool)]
        for pg, g in sub.groupby("pattern_group", sort=False):
            for rec in g.head(n_spot).itertuples(index=False):
                for seq in str(rec.seq_names).split(";"):
                    kset = keys[cls]
                    ok = None if kset is None else ((str(rec.Sample), seq) in kset)
                    rows.append({"pattern_group": pg, "Sample": rec.Sample, "mge_class": cls,
                                 "seq_name": seq, "in_genomad_long": ok})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# phase drivers
# --------------------------------------------------------------------------------------------------
def _read_carrier_shard(carriers_tsv: Path, shard_index: int, n_shards: int) -> pd.DataFrame:
    """Read the resolved carrier table and slice this shard (round-robin; carriers are independent)."""
    df = pd.read_csv(carriers_tsv, sep="\t")
    if n_shards > 1:
        df = df[df.reset_index().index % n_shards == shard_index]
    return df.reset_index(drop=True)


def run_select(args: argparse.Namespace) -> None:
    """``select`` phase: groups → query FASTA + id_map; scan matrix → carriers; resolve assemblies."""
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    hits = pd.read_csv(args.hits_tsv, sep="\t")
    members = select_target_groups(hits, args.top_n_groups, args.min_af)
    id_map = write_query_fasta(members, out / "query_unitigs.fasta")
    id_map.to_csv(out / "id_map.tsv", sep="\t", index=False)

    target_seqs = dict(zip(id_map["variant"].astype(str), id_map["query_id"], strict=True))
    all_hit_seqs = set(hits["variant"].astype(str))
    submatrix = out / "hits_submatrix.tsv"
    extract_hit_submatrix(args.unitig_matrix, all_hit_seqs, submatrix, decomp_threads=args.decomp_threads)
    carriers, phase2_union, n_matched = carriers_from_submatrix(submatrix, target_seqs)

    # unitig_carriers.tsv: one row per (query_id, carrier) — the ground-truth (unitig, carrier) pairs.
    car_rows = [{"query_id": qid, "Sample": s} for qid, samples in carriers.items() for s in samples]
    pd.DataFrame(car_rows).to_csv(out / "unitig_carriers.tsv", sep="\t", index=False)
    join_misses = [q for q in id_map["query_id"] if q not in carriers]
    (out / "unitig_join_misses.txt").write_text("\n".join(join_misses) + ("\n" if join_misses else ""))

    carrier_union = sorted({s for samples in carriers.values() for s in samples})
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
        "phase": "select", "n_groups": int(members["pattern_group"].nunique()),
        "pattern_groups": [int(x) for x in members["pattern_group"].unique()],
        "n_member_unitigs": int(len(id_map)), "n_target_carriers": len(carrier_union),
        "n_resolved_assemblies": int(len(res)), "n_join_misses": len(join_misses),
        "n_all_hit_seqs": len(all_hit_seqs), "n_all_hit_seqs_matched": n_matched,
        "phase2_carrier_union": len(phase2_union), "min_af": args.min_af,
        "minimap2_preset": " ".join(_MINIMAP2_PRESET),
        "genomad_layout_found": sum(1 for p in probe if p["found"]),
    }
    (out / "select_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), file=sys.stderr)


def _probe_genomad_layout(samples: list[str], genomad_root: Path) -> list[dict[str, Any]]:
    """Probe a few carriers' geNomad dirs → which key (<Sample> / __sr) resolved + which extracts exist."""
    rows = []
    for s in samples:
        g = resolve_genomad_paths(s, genomad_root)
        rows.append({"Sample": s, "resolved_key": g["key"], "found": g["found"],
                     "has_plasmid": g["plasmid_fna"] is not None, "has_virus": g["virus_fna"] is not None})
    return rows


def run_align(args: argparse.Namespace) -> None:
    """``align`` phase: one shard of carriers → per-carrier classify → a shard long TSV in scratch."""
    out = args.out_dir
    id_map = pd.read_csv(out / "id_map.tsv", sep="\t")
    carriers = pd.read_csv(out / "unitig_carriers.tsv", sep="\t")
    by_sample = carriers.groupby("Sample")["query_id"].apply(list).to_dict()

    shard = _read_carrier_shard(out / "carriers.resolved.tsv", args.carrier_shard_index, args.n_shards)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    records, timings = _align_shard(shard, by_sample, id_map, args)
    shard_tsv = args.scratch_dir / f"align_shard_{args.carrier_shard_index:04d}.tsv"
    pd.DataFrame(records).to_csv(shard_tsv, sep="\t", index=False)
    (args.scratch_dir / f"align_shard_{args.carrier_shard_index:04d}.timing.json").write_text(json.dumps(timings, indent=2))
    print(f"shard {args.carrier_shard_index}: {len(records)} records, {shard['Sample'].nunique()} carriers -> {shard_tsv}", file=sys.stderr)


def _align_shard(shard: pd.DataFrame, by_sample: dict[str, list[str]], id_map: pd.DataFrame, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Align every carrier in a shard; aggregate records + timing sums (seconds/genome, /unitig)."""
    records: list[dict[str, Any]] = []
    totals = {"build": 0.0, "mm2": 0.0, "parse": 0.0}
    n_carriers, n_pairs = 0, 0
    with tempfile.TemporaryDirectory(prefix="mge_align_", dir=args.scratch_dir) as td:
        tmp = Path(td)
        for row in shard.itertuples(index=False):
            sample, asm = str(row.Sample), Path(str(row.assembly_path))
            expected = by_sample.get(sample, [])
            if not expected:
                continue
            recs, t = align_carrier(
                sample, asm, expected, id_map, args.genomad_root, args.out_dir / "query_unitigs.fasta",
                args.minimap2_bin, tmp, min_ident=args.min_ident, min_cov=args.min_cov,
                min_matchlen=args.min_matchlen, threads=args.threads)
            records.extend(recs)
            for k in totals:
                totals[k] += t[k]
            n_carriers += 1
            n_pairs += len(recs)
    timing = {
        **{f"total_{k}_s": round(v, 3) for k, v in totals.items()},
        "n_carriers": n_carriers, "n_pairs": n_pairs,
        "sec_per_genome": round(sum(totals.values()) / n_carriers, 4) if n_carriers else None,
        "sec_per_unitig_pair": round(sum(totals.values()) / n_pairs, 6) if n_pairs else None,
    }
    return records, timing


def run_combine(args: argparse.Namespace) -> None:
    """``combine`` phase: gather shard TSVs → master long TSV, per-group summary, spot-check, manifest."""
    out = args.out_dir
    shard_files = sorted(args.scratch_dir.glob("align_shard_*.tsv"))
    if not shard_files:
        raise SystemExit(f"no align_shard_*.tsv under {args.scratch_dir}")
    long = pd.concat([pd.read_csv(f, sep="\t") for f in shard_files], ignore_index=True)
    long.to_csv(out / "mge_hits_long.tsv", sep="\t", index=False)

    summary = summarise_groups(long)
    summary.to_csv(out / "mge_summary.tsv", sep="\t", index=False)
    spot = crosscheck_genomad(long, args.genomad_root, args.spot_n)
    spot.to_csv(out / "spotcheck.tsv", sep="\t", index=False)

    _write_combine_manifest(out, long, summary, spot, args.scratch_dir)
    print(summary.to_string(index=False), file=sys.stderr)


def _write_combine_manifest(out: Path, long: pd.DataFrame, summary: pd.DataFrame, spot: pd.DataFrame, scratch: Path) -> None:
    """Aggregate shard timings into a Phase-2 estimate + record QC counts to a JSON manifest."""
    timings = [json.loads(f.read_text()) for f in sorted(scratch.glob("align_shard_*.timing.json"))]
    tot_secs = sum(t.get(f"total_{k}_s", 0.0) for t in timings for k in ("build", "mm2", "parse"))
    tot_carriers = sum(t.get("n_carriers", 0) for t in timings)
    sec_per_genome = round(tot_secs / tot_carriers, 4) if tot_carriers else None
    asm = long[long["asm_recall"].notna()]
    manifest = {
        "phase": "combine", "n_observations": int(len(long)),
        "n_carriers": int(long["Sample"].nunique()), "n_pattern_groups": int(long["pattern_group"].nunique()),
        "class_counts": {k: int(v) for k, v in long["mge_class"].value_counts().items()},
        "asm_recall": round(float(asm["asm_recall"].mean()), 4) if len(asm) else None,
        "spotcheck_pass": int((spot["in_genomad_long"] == True).sum()),  # noqa: E712
        "spotcheck_fail": int((spot["in_genomad_long"] == False).sum()),  # noqa: E712
        "sec_per_genome": sec_per_genome,
    }
    (out / "combine_manifest.json").write_text(json.dumps(manifest, indent=2))
    if sec_per_genome:
        sel = json.loads((out / "select_manifest.json").read_text()) if (out / "select_manifest.json").is_file() else {}
        c2 = sel.get("phase2_carrier_union")
        if c2:
            serial_h = round(c2 * sec_per_genome / 3600, 2)
            est = {"phase2_carrier_union": c2, "sec_per_genome": sec_per_genome,
                   "serial_wall_hours": serial_h,
                   "suggested_n_shards": max(1, int((serial_h / 2) + 0.999)),  # target <= ~2h/shard
                   "note": "N_shards targets <=2h per array task; carriers are independent."}
            (out / "phase2_estimate.json").write_text(json.dumps(est, indent=2))


def run_smoke(args: argparse.Namespace) -> None:
    """``smoke`` phase: select + align (K carriers of the strongest group) + combine, inline + timed."""
    run_select(args)
    out = args.out_dir
    id_map = pd.read_csv(out / "id_map.tsv", sep="\t")
    carriers = pd.read_csv(out / "unitig_carriers.tsv", sep="\t")
    resolved = pd.read_csv(out / "carriers.resolved.tsv", sep="\t")

    top_qids = set(id_map[id_map["group_rank"] == 0]["query_id"])
    top_carriers = carriers[carriers["query_id"].isin(top_qids)]["Sample"].unique()[: args.smoke]
    shard = resolved[resolved["Sample"].isin(top_carriers)].reset_index(drop=True)
    if shard.empty:
        raise SystemExit("smoke: no resolved assemblies among the top group's carriers")

    by_sample = carriers.groupby("Sample")["query_id"].apply(list).to_dict()
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    records, timing = _align_shard(shard, by_sample, id_map, args)
    pd.DataFrame(records).to_csv(args.scratch_dir / "align_shard_0000.tsv", sep="\t", index=False)
    (args.scratch_dir / "align_shard_0000.timing.json").write_text(json.dumps(timing, indent=2))
    run_combine(args)
    print("\n=== SMOKE timing ===\n" + json.dumps(timing, indent=2), file=sys.stderr)


# --------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    """CLI entry point — dispatch on ``--phase``."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", required=True, choices=("select", "align", "combine", "smoke"))
    p.add_argument("--hits-tsv", type=Path, help="blood_vs_faeces_unitig_hits_annotated.tsv (select/smoke).")
    p.add_argument("--unitig-matrix", type=Path, help="unitigs.pyseer.gz (select/smoke).")
    p.add_argument("--genomad-root", type=Path, required=True, help="<DATA>/david/processed/genomad.")
    p.add_argument("--metadata", type=Path, help="metadata_v2 TSV for assembly resolution (select/smoke).")
    p.add_argument("--out-dir", type=Path, required=True, help="Durable outputs on project_k.")
    p.add_argument("--scratch-dir", type=Path, required=True, help="Per-shard scratch (RDS hpc-work).")
    p.add_argument("--top-n-groups", type=int, default=3)
    p.add_argument("--min-af", type=float, default=0.05)
    p.add_argument("--min-ident", type=float, default=0.95)
    p.add_argument("--min-cov", type=float, default=0.90)
    p.add_argument("--min-matchlen", type=int, default=25)
    p.add_argument("--minimap2-bin", default="minimap2")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--decomp-threads", type=int, default=4, help="pigz threads for the one-time matrix scan.")
    p.add_argument("--carrier-shard-index", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--genomad-probe-n", type=int, default=20, help="Carriers to probe for geNomad layout.")
    p.add_argument("--spot-n", type=int, default=3, help="Spot-check records per class per group.")
    p.add_argument("--smoke", type=int, default=5, help="smoke phase: carriers of the top group to map.")
    args = p.parse_args(argv)

    if args.phase in ("select", "smoke"):
        for req in ("hits_tsv", "unitig_matrix", "metadata"):
            if getattr(args, req) is None:
                p.error(f"--{req.replace('_', '-')} is required for --phase {args.phase}")

    {"select": run_select, "align": run_align, "combine": run_combine, "smoke": run_smoke}[args.phase](args)


if __name__ == "__main__":
    main()
