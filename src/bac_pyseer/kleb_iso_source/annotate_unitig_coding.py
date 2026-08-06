r"""Classify every invasion-GWAS unitig hit as CDS vs IGR (and which IGR type) in every carrier.

The geNomad + ISEScan mapping (``map_unitig_hits_genomad``) answered *plasmid / prophage / chromosomal
/ IS*, but not **coding vs non-coding**. This job asks: does the invasion (and faeces) unitig signal
sit in **protein-coding sequence (CDS)** or in **intergenic DNA (IGR)** — and, within IGR, in a
Bakta-annotated feature (tRNA/rRNA/ncRNA/CRISPR/regulatory/oriC…) or the **unclassified** gaps where
promoters (which Bakta does not annotate) live? Quantified against the genome-wide IGR/CDS base-pair
baseline (``genome_coding_fraction``), it says whether IGR is *enriched* for signal.

It is an **independent job**, not a layer inside the geNomad module: it reuses that module's cached
``select`` artifacts (``id_map.tsv`` / ``carriers.resolved.tsv`` / ``hits_submatrix.tsv`` — so the
``unitig_idx`` numbering is identical and the two parquets are joinable) and its Aho-Corasick placement
helpers, but classifies each placement through the shared :class:`genome_prep.CodingIndex` built from
the carrier's Bakta GFF3. Contig names are concordant (BakRep GFF ``seqID`` == the seb assembly the
unitigs were placed on — verified).

Per (unitig, carrier) pair we take the placement's occurrence(s) and assign one class: overlaps any
CDS → ``CDS`` (coding wins on a tie — conservative against over-claiming IGR); else the majority IGR
type → ``IGR_<type>``. Aggregates: per-unitig, overall / by direction / by af bin, and (stratify) by
big sublineage / clonal group. Refutation (signal is CDS) is a first-class outcome; nothing here
asserts a mechanism.

Phases (``--phase``): ``align`` (array shard of carriers → per-unitig coding aggregate + per-pair
parquet), ``combine`` (roll up → per-unitig / overall / pattern_group + manifest + parquet dataset),
``stratify`` (re-aggregate the parquet by clonal structure), ``smoke`` (align a few carriers + combine
inline, timed). Results go to project_k / scratch only, never ``$HOME``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from genome_prep import CDS_CLASS, CodingIndex

try:  # package import when bac_pyseer is on the path (editable install)
    from bac_pyseer.kleb_iso_source.map_unitig_hits_genomad import (
        _load_contigs,
        _load_strata,
        _read_carrier_shard,
        _rollup_overall,
        _rollup_pattern_group,
        _sum_long,
        build_automaton,
        scan_carrier,
        shard_expected,
    )
    from bac_pyseer.kleb_iso_source.map_unitig_hits_genomad import _ParquetSink as ParquetSink
except ImportError:  # invoked as a bare script — add the script dir to sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from map_unitig_hits_genomad import (
        _load_contigs,
        _load_strata,
        _read_carrier_shard,
        _rollup_overall,
        _rollup_pattern_group,
        _sum_long,
        build_automaton,
        scan_carrier,
        shard_expected,
    )
    from map_unitig_hits_genomad import (  # type: ignore[no-redef]
        _ParquetSink as ParquetSink,
    )

_UNKNOWN_NO_BAKTA = "unknown_no_bakta"  # carrier had no usable Bakta GFF (should be ~never — 100% coverage)
_NO_ASM_HIT = "no_asm_hit"              # unitig expected but not found in the assembly (asm-recall miss)


def classify_pair(occurrences: list[tuple[str, int]], unitig_len: int, cidx: CodingIndex | None) -> str:
    """Reduce a unitig's ASM occurrences in one carrier to a single coding class label (``rclass``).

    ``CDS`` if occurrences overlap coding sequence at least as often as IGR (coding wins ties); else
    ``IGR_<majority-type>``. ``unknown_no_bakta`` when the carrier has no CodingIndex; ``no_asm_hit``
    when the unitig was expected but not placed. Occurrence spans are the 1-based inclusive
    ``end_off - unitig_len + 2 … end_off + 1`` (same convention as the IS annotator).
    """
    if cidx is None:
        return _UNKNOWN_NO_BAKTA
    if not occurrences:
        return _NO_ASM_HIT
    n_cds = 0
    igr_types: Counter = Counter()
    for contig, end_off in occurrences:
        s, e = end_off - unitig_len + 2, end_off + 1
        coding_class, igr_type = cidx.classify_span(contig, s, e)
        if coding_class == CDS_CLASS:
            n_cds += 1
        else:
            igr_types[igr_type] += 1
    n_igr = sum(igr_types.values())
    if n_cds >= n_igr:  # coding wins ties
        return CDS_CLASS
    return f"IGR_{igr_types.most_common(1)[0][0]}"


# --------------------------------------------------------------------------------------------------
# align
# --------------------------------------------------------------------------------------------------
def _write_class_counts(path: Path, class_counts: dict[int, Counter]) -> None:
    """Write long per-(unitig_idx, rclass) coding-class counts for this shard."""
    rows = [{"unitig_idx": idx, "rclass": cls, "n": n} for idx, cc in class_counts.items() for cls, n in cc.items()]
    pd.DataFrame(rows, columns=["unitig_idx", "rclass", "n"]).to_csv(path, sep="\t", index=False)


def run_align(args: argparse.Namespace) -> None:
    """``align`` shard: Aho-Corasick placement of this shard's carriers → per-unitig coding aggregate."""
    sel, out, scratch = args.select_dir, args.out_dir, args.scratch_dir
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    id_map = pd.read_csv(sel / "id_map.tsv", sep="\t")
    seq2idx = {str(s).upper(): int(i) for s, i in zip(id_map["variant"], id_map["unitig_idx"], strict=True)}
    ulen = {int(i): int(n) for i, n in zip(id_map["unitig_idx"], id_map["unitig_len"], strict=True)}
    aut = build_automaton(id_map)
    bakta_lookup = dict(zip(*[pd.read_csv(args.bakta_lookup, sep="\t")[c].astype(str)
                             for c in ("Sample", "path")], strict=True)) if args.bakta_lookup else {}

    shard = _read_carrier_shard(sel / "carriers.resolved.tsv", args.carrier_shard_index, args.n_shards)
    my_carriers = set(shard["Sample"].astype(str))
    expected = shard_expected(sel / "hits_submatrix.tsv", seq2idx, my_carriers)

    i = args.carrier_shard_index
    sink = ParquetSink(scratch / f"coding_shard_{i:04d}.parquet")
    class_counts: dict[int, Counter] = defaultdict(Counter)
    qc = {"n_carriers": 0, "n_pairs": 0, "asm_hit_num": 0, "asm_hit_den": 0, "no_bakta_carriers": 0}
    t0 = time.perf_counter()

    for row in shard.itertuples(index=False):
        sample = str(row.Sample)
        exp = expected.get(sample, set())
        if not exp:
            continue
        qc["n_carriers"] += 1
        gff = bakta_lookup.get(sample)
        cidx = CodingIndex.from_gff(gff) if gff and Path(gff).is_file() else None
        if cidx is None:
            qc["no_bakta_carriers"] += 1
        _found, asm_pos = scan_carrier(aut, {}, {}, _load_contigs(Path(str(row.assembly_path))))
        for idx in exp:
            occ = asm_pos.get(idx, [])
            rclass = classify_pair(occ, ulen[idx], cidx)
            class_counts[idx][rclass] += 1
            qc["asm_hit_den"] += 1
            qc["asm_hit_num"] += int(bool(occ))
            sink.add({"unitig_idx": idx, "Sample": sample, "rclass": rclass, "n_copies": len(set(occ))})
        qc["n_pairs"] += len(exp)

    sink.close()
    qc["seconds"] = round(time.perf_counter() - t0, 2)
    qc["sec_per_genome"] = round(qc["seconds"] / qc["n_carriers"], 4) if qc["n_carriers"] else None
    _write_class_counts(scratch / f"coding_shard_{i:04d}.class.tsv", class_counts)
    (scratch / f"coding_shard_{i:04d}.qc.json").write_text(json.dumps(qc, indent=2))
    print(f"shard {i}: {qc['n_carriers']} carriers, {qc['n_pairs']} pairs, {qc['sec_per_genome']} s/genome",
          file=sys.stderr)


# --------------------------------------------------------------------------------------------------
# combine
# --------------------------------------------------------------------------------------------------
def _igr_columns(classes: list[str]) -> list[str]:
    """The IGR_* class columns among ``classes`` (everything that is not CDS / unknown / no_asm_hit)."""
    return [c for c in classes if c.startswith("IGR_")]


def _per_unitig(cls: pd.DataFrame, id_map: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pivot long (unitig_idx, rclass, n) → wide per-unitig counts + fracs, merged with hit metadata."""
    wide = cls.pivot(index="unitig_idx", columns="rclass", values="n").fillna(0).astype(int)
    classes = sorted(wide.columns)
    wide = wide.reset_index()
    keep = [c for c in ["unitig_idx", "variant", "pattern_group", "direction", "beta", "af", "var_explained_pct"]
            if c in id_map.columns]
    per = id_map[keep].merge(wide, on="unitig_idx", how="right")
    per["n_carriers"] = per[classes].sum(axis=1)
    igr_cols = _igr_columns(classes)
    denom = per["n_carriers"].replace(0, pd.NA)
    per["frac_CDS"] = (per.get("CDS", 0) / denom).astype(float).round(4)
    per["frac_IGR"] = (per[igr_cols].sum(axis=1) / denom).astype(float).round(4) if igr_cols else 0.0
    per["frac_IGR_unclassified"] = (per.get("IGR_unclassified", 0) / denom).astype(float).round(4)
    if igr_cols:
        dom = per[igr_cols].idxmax(axis=1).str.replace("IGR_", "", regex=False)
        per["dominant_igr_type"] = dom.where(per[igr_cols].sum(axis=1) > 0, "")
    else:
        per["dominant_igr_type"] = ""
    return per, classes


def _overall_with_igr_total(per: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """``_rollup_overall`` (per fine class) + a summed ``frac_IGR`` convenience column."""
    ov = _rollup_overall(per, classes)
    igr_frac_cols = [f"frac_{c}" for c in _igr_columns(classes) if f"frac_{c}" in ov.columns]
    ov.insert(3, "frac_IGR", ov[igr_frac_cols].sum(axis=1).round(4) if igr_frac_cols else 0.0)
    return ov


def _assemble_parquet_dataset(scratch: Path, out: Path) -> None:
    """Move the per-shard parquet parts into a single ``coding_hits.parquet/`` dataset dir."""
    dataset = out / "coding_hits.parquet"
    dataset.mkdir(parents=True, exist_ok=True)
    for part in sorted(scratch.glob("coding_shard_*.parquet")):
        shutil.copy2(part, dataset / part.name)


def run_combine(args: argparse.Namespace) -> None:
    """``combine``: sum shard aggregates → per-unitig / overall / pattern_group tables + parquet + manifest."""
    sel, out, scratch = args.select_dir, args.out_dir, args.scratch_dir
    id_map = pd.read_csv(sel / "id_map.tsv", sep="\t")
    cls = _sum_long(sorted(scratch.glob("coding_shard_*.class.tsv")), ["unitig_idx", "rclass"])
    per, classes = _per_unitig(cls, id_map)
    per.to_csv(out / "coding_unitig_class.tsv", sep="\t", index=False)
    _overall_with_igr_total(per, classes).to_csv(out / "coding_overall.tsv", sep="\t", index=False)
    if "pattern_group" in per.columns:
        _rollup_pattern_group(per, classes).to_csv(out / "coding_pattern_group.tsv", sep="\t", index=False)
    _assemble_parquet_dataset(scratch, out)

    qcs = [json.loads(f.read_text()) for f in sorted(scratch.glob("coding_shard_*.qc.json"))]
    den = sum(q.get("asm_hit_den", 0) for q in qcs)
    manifest = {
        "phase": "combine", "n_unitigs": int(len(per)),
        "n_carriers": sum(q.get("n_carriers", 0) for q in qcs),
        "n_pairs": sum(q.get("n_pairs", 0) for q in qcs),
        "asm_recall": round(sum(q.get("asm_hit_num", 0) for q in qcs) / den, 4) if den else None,
        "no_bakta_carriers": sum(q.get("no_bakta_carriers", 0) for q in qcs),
        "class_pair_totals": {c: int(per[c].sum()) for c in classes},
        "total_align_seconds": round(sum(q.get("seconds", 0.0) for q in qcs), 1),
    }
    (out / "combine_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), file=sys.stderr)


# --------------------------------------------------------------------------------------------------
# stratify
# --------------------------------------------------------------------------------------------------
def run_stratify(args: argparse.Namespace) -> None:
    """``stratify``: re-aggregate the per-pair parquet's coding class by big sublineage / clonal group."""
    import pyarrow.parquet as pq

    sel, out = args.select_dir, args.out_dir
    id_map = pd.read_csv(sel / "id_map.tsv", sep="\t", usecols=["unitig_idx", "direction"])
    dir_by_idx = dict(zip(id_map["unitig_idx"], id_map["direction"].astype(str), strict=True))
    carriers = set(pd.read_csv(sel / "carriers.resolved.tsv", sep="\t")["Sample"].astype(str))
    strata = _load_strata(args.strata_csv, args.min_group_size, carriers)

    acc: Counter = Counter()  # (level, group, direction, rclass) -> n_pairs
    seen: dict[str, set[str]] = defaultdict(set)
    dataset = out / "coding_hits.parquet"
    for part in sorted(dataset.glob("*.parquet")):
        t = pq.read_table(part, columns=["unitig_idx", "Sample", "rclass"]).to_pandas()
        t["direction"] = t["unitig_idx"].map(dir_by_idx).fillna("NA")
        t["Sample"] = t["Sample"].astype(str)
        for level, smap in strata.items():
            grp = t["Sample"].map(smap).fillna("unknown")
            for (g, dirn, rc), n in t.groupby([grp, "direction", "rclass"]).size().items():
                acc[(level, g, dirn, rc)] += int(n)
                seen[level].add(rc)

    for level in strata:
        classes = sorted(seen[level])
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        for (lvl, g, dirn, rc), n in acc.items():
            if lvl != level:
                continue
            r = rows.setdefault((g, dirn), {"group": g, "direction": dirn, **dict.fromkeys(classes, 0)})
            r[rc] = n
        df = pd.DataFrame(rows.values())
        if not df.empty:
            tot = df[classes].sum(axis=1)
            for c in classes:
                df[f"frac_{c}"] = (df[c] / tot).round(4)
            igr = [f"frac_{c}" for c in classes if c.startswith("IGR_")]
            df.insert(2, "n_pairs", tot)
            df.insert(3, "frac_IGR", df[igr].sum(axis=1).round(4) if igr else 0.0)
            df = df.sort_values(["group", "direction"])
        df.to_csv(out / f"coding_by_{level}.tsv", sep="\t", index=False)
    print(f"stratified: {[f'coding_by_{lv}.tsv' for lv in strata]}", file=sys.stderr)


# --------------------------------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------------------------------
def run_smoke(args: argparse.Namespace) -> None:
    """``smoke``: align the first ``--smoke`` carriers + combine inline (timed) to validate the pipeline."""
    args.n_shards = 1
    args.carrier_shard_index = 0
    # Restrict carriers.resolved to the first K by writing a tiny shard the align reads via _read_carrier_shard.
    sel = args.select_dir
    full = pd.read_csv(sel / "carriers.resolved.tsv", sep="\t").head(args.smoke)
    smoke_sel = args.out_dir / "_smoke_select"
    smoke_sel.mkdir(parents=True, exist_ok=True)
    full.to_csv(smoke_sel / "carriers.resolved.tsv", sep="\t", index=False)
    for f in ("id_map.tsv", "hits_submatrix.tsv"):
        (smoke_sel / f).unlink(missing_ok=True)
        (smoke_sel / f).symlink_to(sel / f)
    args.select_dir = smoke_sel
    run_align(args)
    run_combine(args)
    print("=== smoke coding_overall.tsv ===", file=sys.stderr)
    print(pd.read_csv(args.out_dir / "coding_overall.tsv", sep="\t").to_string(index=False), file=sys.stderr)


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    """CLI entry point — dispatch on ``--phase``."""
    p = argparse.ArgumentParser(description="Classify unitig hits CDS-vs-IGR across carriers (genome_prep).")
    p.add_argument("--phase", required=True, choices=["align", "combine", "stratify", "smoke"])
    p.add_argument("--select-dir", type=Path, required=True, help="geNomad select artifacts (id_map/carriers/submatrix).")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--scratch-dir", type=Path, required=True)
    p.add_argument("--bakta-lookup", type=Path, help="TSV Sample<TAB>path → per-genome Bakta GFF3 (align/smoke).")
    p.add_argument("--carrier-shard-index", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--strata-csv", type=Path, help="Cohort CSV with Sample/Sublineage/Clonal group (stratify).")
    p.add_argument("--min-group-size", type=int, default=100)
    p.add_argument("--smoke", type=int, default=30, help="Number of carriers for the smoke phase.")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    {"align": run_align, "combine": run_combine, "stratify": run_stratify, "smoke": run_smoke}[args.phase](args)


if __name__ == "__main__":
    main()
