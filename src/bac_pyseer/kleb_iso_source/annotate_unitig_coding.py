r"""Measure how much of each invasion-GWAS unitig hit lies in IGR vs CDS, across every carrier.

The geNomad + ISEScan mapping (``map_unitig_hits_genomad``) answered *plasmid / prophage / chromosomal
/ IS*, but not **coding vs non-coding**. This job asks: does the invasion (and faeces) unitig signal
sit in **protein-coding sequence (CDS)** or in **intergenic DNA (IGR)**? Rather than a binary vote, it
measures, per placement, the **base-pair fraction of the unitig that lies in IGR** (``igr_frac``), so a
boundary-spanning unitig is handled by *how much* IGR it covers, not an arbitrary tie-break. Reported
robustly across thresholds — ``% entirely within CDS`` (igr_frac = 0), ``% touching any IGR``,
``% covering a significant portion`` (≥ 0.25), ``% predominantly IGR`` (≥ 0.5), ``% entirely IGR`` — so
the headline is measure-insensitive.

The comparator is a **uniform-placement null** (``coding_null_model``): given the genomes' CDS
architecture and these unitigs' lengths, what fraction *would* land entirely within a CDS if the GWAS
were spatially uniform? Genes are long and unitigs short, so that null is high (near the CDS bp
fraction); observed entirely-CDS well below it means the signal avoids pure coding sequence.

Independent job (not a layer in the geNomad module): reuses that module's cached ``select`` artifacts
(identical ``unitig_idx``; joinable parquet) + its Aho-Corasick placement helpers, and measures each
placement through the shared :class:`genome_prep.CodingIndex` (``cds_overlap_bp``) built from the
carrier Bakta GFF3. Contig names are concordant (BakRep GFF ``seqID`` == the seb assembly the unitigs
were placed on — verified). Unit of report is the **hit unitig** (each counted once, by its behaviour
across the carriers it appears in), with a placement-weighted view alongside.

Phases (``--phase``): ``align`` (array shard → per-unitig igr-coverage aggregate + per-pair parquet),
``combine`` (roll up → per-unitig / overall / pattern_group + manifest + parquet dataset), ``stratify``
(re-aggregate by clonal structure), ``smoke`` (align a few carriers + combine inline, timed). Results
go to project_k / scratch only, never ``$HOME``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from genome_prep import CodingIndex

try:  # package import when bac_pyseer is on the path (editable install)
    from bac_pyseer.kleb_iso_source.map_unitig_hits_genomad import (
        _load_contigs,
        _load_strata,
        _read_carrier_shard,
        build_automaton,
        scan_carrier,
        shard_expected,
    )
    from bac_pyseer.kleb_iso_source.map_unitig_hits_genomad import _ParquetSink as ParquetSink
except ImportError:  # invoked as a bare script — add the script dir to sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from map_unitig_hits_genomad import (  # type: ignore[no-redef]
        _load_contigs,
        _load_strata,
        _read_carrier_shard,
        build_automaton,
        scan_carrier,
        shard_expected,
    )
    from map_unitig_hits_genomad import _ParquetSink as ParquetSink  # type: ignore[no-redef]

# igr_frac thresholds for the robust, measure-insensitive reporting.
_SIGNIFICANT = 0.25   # "covers a significant portion of IGR"
_PREDOMINANT = 0.5    # "predominantly IGR"
_ENTIRELY = 0.999     # "entirely IGR" (== 1.0, guarded against float noise)
# Per-unitig count columns accumulated per shard (all denominated by n_pairs).
_COUNT_COLS = ["n_pairs", "n_entirely_cds", "n_touch", "n_significant", "n_predominant", "n_entirely_igr"]


def pair_igr_frac(occurrences: list[tuple[str, int]], unitig_len: int, cidx: CodingIndex) -> float:
    """Mean IGR base-pair fraction of a unitig's ASM occurrences in one carrier.

    Each occurrence spans the 1-based inclusive ``end_off - unitig_len + 2 … end_off + 1``; its IGR
    fraction is ``(L - cds_overlap_bp) / L``. Returns the mean over occurrences (≈97% are single-copy).
    """
    fracs = []
    for contig, end_off in occurrences:
        s, e = end_off - unitig_len + 2, end_off + 1
        fracs.append((unitig_len - cidx.cds_overlap_bp(contig, s, e)) / unitig_len)
    return sum(fracs) / len(fracs)


# --------------------------------------------------------------------------------------------------
# align
# --------------------------------------------------------------------------------------------------
def run_align(args: argparse.Namespace) -> None:
    """``align`` shard: measure per-carrier igr_frac for each expected unitig → per-unitig aggregate."""
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
    # per unitig_idx -> [n_pairs, n_entirely_cds, n_touch, n_significant, n_predominant, n_entirely_igr, sum_frac]
    stats: dict[int, list[float]] = defaultdict(lambda: [0.0] * 7)
    qc = {"n_carriers": 0, "n_pairs": 0, "no_bakta_carriers": 0, "no_asm_hit_pairs": 0}
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
            continue
        _found, asm_pos = scan_carrier(aut, {}, {}, _load_contigs(Path(str(row.assembly_path))))
        for idx in exp:
            occ = asm_pos.get(idx, [])
            if not occ:
                qc["no_asm_hit_pairs"] += 1
                continue
            frac = pair_igr_frac(occ, ulen[idx], cidx)
            st = stats[idx]
            st[0] += 1
            st[1] += frac == 0
            st[2] += frac > 0
            st[3] += frac >= _SIGNIFICANT
            st[4] += frac >= _PREDOMINANT
            st[5] += frac >= _ENTIRELY
            st[6] += frac
            sink.add({"unitig_idx": idx, "Sample": sample, "igr_frac": round(frac, 4), "n_copies": len(set(occ))})
        qc["n_pairs"] += len(exp)

    sink.close()
    rows = [{"unitig_idx": idx, "n_pairs": int(s[0]), "n_entirely_cds": int(s[1]), "n_touch": int(s[2]),
             "n_significant": int(s[3]), "n_predominant": int(s[4]), "n_entirely_igr": int(s[5]),
             "sum_frac": s[6]} for idx, s in stats.items()]
    pd.DataFrame(rows, columns=["unitig_idx", *_COUNT_COLS, "sum_frac"]).to_csv(
        scratch / f"coding_shard_{i:04d}.perunitig.tsv", sep="\t", index=False)
    qc["seconds"] = round(time.perf_counter() - t0, 2)
    qc["sec_per_genome"] = round(qc["seconds"] / qc["n_carriers"], 4) if qc["n_carriers"] else None
    (scratch / f"coding_shard_{i:04d}.qc.json").write_text(json.dumps(qc, indent=2))
    print(f"shard {i}: {qc['n_carriers']} carriers, {qc['n_pairs']} pairs, {qc['sec_per_genome']} s/genome",
          file=sys.stderr)


# --------------------------------------------------------------------------------------------------
# combine
# --------------------------------------------------------------------------------------------------
def _per_unitig(scratch: Path, id_map: pd.DataFrame) -> pd.DataFrame:
    """Sum shard per-unitig aggregates → per-unitig fractions, merged with hit metadata."""
    parts = [pd.read_csv(f, sep="\t") for f in sorted(scratch.glob("coding_shard_*.perunitig.tsv"))
             if f.stat().st_size > 1]
    agg = pd.concat(parts, ignore_index=True).groupby("unitig_idx", as_index=False).sum()
    agg["n_carriers"] = agg["n_pairs"]
    agg["mean_igr_frac"] = (agg["sum_frac"] / agg["n_pairs"]).round(4)
    for c, p in [("n_entirely_cds", "p_entirely_cds"), ("n_touch", "p_touch"), ("n_significant", "p_significant"),
                 ("n_predominant", "p_predominant"), ("n_entirely_igr", "p_entirely_igr")]:
        agg[p] = (agg[c] / agg["n_pairs"]).round(4)
    keep = [c for c in ["unitig_idx", "variant", "pattern_group", "direction", "beta", "af", "var_explained_pct"]
            if c in id_map.columns]
    return id_map[keep].merge(agg, on="unitig_idx", how="right")


_PROBS = [("p_entirely_cds", "entirely_cds"), ("p_touch", "touch_igr"), ("p_significant", "significant_igr"),
          ("p_predominant", "predominant_igr"), ("p_entirely_igr", "entirely_igr")]
_COUNTS = [("n_entirely_cds", "entirely_cds"), ("n_touch", "touch_igr"), ("n_significant", "significant_igr"),
           ("n_predominant", "predominant_igr"), ("n_entirely_igr", "entirely_igr")]


def _rollup(per: pd.DataFrame) -> pd.DataFrame:
    """Overall + by-direction + by-af: unitig-level (majority-of-carriers) AND placement-weighted fractions."""
    rows = []

    def _agg(label: str, sub: pd.DataFrame) -> dict[str, Any]:
        n_pairs = int(sub["n_pairs"].sum())
        d: dict[str, Any] = {"stratum": label, "n_unitigs": len(sub), "n_pairs": n_pairs}
        d["mean_igr_frac"] = round(float(sub["mean_igr_frac"].mean()), 4) if len(sub) else 0.0
        for pcol, name in _PROBS:  # unitig-level: fraction of unitigs where the median carrier meets the bar
            d[f"unitig_frac_{name}"] = round(float((sub[pcol] >= 0.5).mean()), 4) if len(sub) else 0.0
        for ncol, name in _COUNTS:  # placement-weighted: fraction of (unitig,carrier) placements
            d[f"placement_frac_{name}"] = round(int(sub[ncol].sum()) / n_pairs, 4) if n_pairs else 0.0
        return d

    rows.append(_agg("ALL", per))
    if "direction" in per.columns:
        for dirn, sub in per.groupby("direction"):
            rows.append(_agg(f"direction={dirn}", sub))
    if "af" in per.columns:
        bins = pd.cut(per["af"], [0, 0.05, 0.2, 0.5, 0.7, 1.0])
        for b, sub in per.groupby(bins, observed=True):
            rows.append(_agg(f"af={b}", sub))
    return pd.DataFrame(rows)


def run_combine(args: argparse.Namespace) -> None:
    """``combine``: per-unitig fractions + overall/pattern_group rollups + parquet dataset + manifest."""
    sel, out, scratch = args.select_dir, args.out_dir, args.scratch_dir
    id_map = pd.read_csv(sel / "id_map.tsv", sep="\t")
    per = _per_unitig(scratch, id_map)
    per.to_csv(out / "coding_unitig_class.tsv", sep="\t", index=False)
    _rollup(per).to_csv(out / "coding_overall.tsv", sep="\t", index=False)
    if "pattern_group" in per.columns:
        pg = per.groupby("pattern_group", as_index=False).agg(
            n_member_unitigs=("unitig_idx", "count"), n_pairs=("n_pairs", "sum"),
            n_entirely_cds=("n_entirely_cds", "sum"), n_predominant=("n_predominant", "sum"))
        pg["placement_frac_entirely_cds"] = (pg["n_entirely_cds"] / pg["n_pairs"]).round(4)
        pg["placement_frac_predominant_igr"] = (pg["n_predominant"] / pg["n_pairs"]).round(4)
        pg.to_csv(out / "coding_pattern_group.tsv", sep="\t", index=False)

    dataset = out / "coding_hits.parquet"
    dataset.mkdir(parents=True, exist_ok=True)
    for part in sorted(scratch.glob("coding_shard_*.parquet")):
        shutil.copy2(part, dataset / part.name)

    qcs = [json.loads(f.read_text()) for f in sorted(scratch.glob("coding_shard_*.qc.json"))]
    manifest = {
        "phase": "combine", "n_unitigs": int(len(per)),
        "n_carriers": sum(q.get("n_carriers", 0) for q in qcs),
        "n_pairs": int(per["n_pairs"].sum()),
        "no_bakta_carriers": sum(q.get("no_bakta_carriers", 0) for q in qcs),
        "no_asm_hit_pairs": sum(q.get("no_asm_hit_pairs", 0) for q in qcs),
        "placement_totals": {name: int(per[ncol].sum()) for ncol, name in _COUNTS},
        "total_align_seconds": round(sum(q.get("seconds", 0.0) for q in qcs), 1),
    }
    (out / "combine_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), file=sys.stderr)


# --------------------------------------------------------------------------------------------------
# stratify
# --------------------------------------------------------------------------------------------------
def run_stratify(args: argparse.Namespace) -> None:
    """``stratify``: re-aggregate the per-pair parquet's igr_frac by big sublineage / clonal group."""
    import pyarrow.parquet as pq

    sel, out = args.select_dir, args.out_dir
    id_map = pd.read_csv(sel / "id_map.tsv", sep="\t", usecols=["unitig_idx", "direction"])
    dir_by_idx = dict(zip(id_map["unitig_idx"], id_map["direction"].astype(str), strict=True))
    carriers = set(pd.read_csv(sel / "carriers.resolved.tsv", sep="\t")["Sample"].astype(str))
    strata = _load_strata(args.strata_csv, args.min_group_size, carriers)

    acc: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # n, ent_cds, pred, sumfrac
    for part in sorted((out / "coding_hits.parquet").glob("*.parquet")):
        t = pq.read_table(part, columns=["unitig_idx", "Sample", "igr_frac"]).to_pandas()
        t["direction"] = t["unitig_idx"].map(dir_by_idx).fillna("NA")
        t["Sample"] = t["Sample"].astype(str)
        for level, smap in strata.items():
            t["_g"] = t["Sample"].map(smap).fillna("unknown")
            for (g, dirn), sub in t.groupby(["_g", "direction"]):
                a = acc[(level, g, dirn)]
                a[0] += len(sub)
                a[1] += float((sub["igr_frac"] == 0).sum())
                a[2] += float((sub["igr_frac"] >= _PREDOMINANT).sum())
                a[3] += float(sub["igr_frac"].sum())

    for level in strata:
        rows = []
        for (lvl, g, dirn), a in acc.items():
            if lvl != level or a[0] == 0:
                continue
            rows.append({"group": g, "direction": dirn, "n_pairs": int(a[0]),
                         "placement_frac_entirely_cds": round(a[1] / a[0], 4),
                         "placement_frac_predominant_igr": round(a[2] / a[0], 4),
                         "mean_igr_frac": round(a[3] / a[0], 4)})
        pd.DataFrame(rows).sort_values(["group", "direction"]).to_csv(
            out / f"coding_by_{level}.tsv", sep="\t", index=False)
    print(f"stratified: {[f'coding_by_{lv}.tsv' for lv in strata]}", file=sys.stderr)


# --------------------------------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------------------------------
def run_smoke(args: argparse.Namespace) -> None:
    """``smoke``: align the first ``--smoke`` carriers + combine inline (timed) to validate the pipeline."""
    args.n_shards = 1
    args.carrier_shard_index = 0
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
    p = argparse.ArgumentParser(description="Measure unitig-hit IGR-vs-CDS coverage across carriers (genome_prep).")
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
