r"""Build a second cohort's genomes into the shared per-BioSample store without touching an earlier cohort's.

The genome stores under ``$AMR/raw/`` (``contigs/``, ``prodigal/``, ``bakta/``) are flat, keyed by BioSample, and shared
by every cohort. A genome that an earlier cohort's ``genome_manifest.csv`` names belongs to that cohort: a later cohort
over the same BioSamples (CABBAGE v2, ``PROJECT_STATE.md`` §3.7) **reuses** its contigs, build row, CheckM2 and skani
results, and never builds it. That is enforced here rather than left to convention, because
``build_genomes.build_one`` deletes and rebuilds a genome whose stored ``src=`` header differs from its manifest row's
source: a second manifest naming the genome could destroy the first cohort's copy rather than skip it.

Subcommands:

``partition``
    ``cohort.csv`` → ``--out-cohort``, its rows that no ``--built`` manifest names (the input to
    ``plan_genome_sources``), and ``--out-reused``, one row per labelled genome that one does (``Sample``, ``owner``
    = that manifest's path, and ``source`` / ``drop_reason`` there). A genome named by two ``--built`` manifests is an
    error.
``check-disjoint``
    Exit 1, naming them, if ``--manifest`` names any genome in a ``--built`` manifest. Run before every submission as
    well as at plan: an earlier cohort's manifest can grow after this one was planned.
``split-by-scheme``
    The manifest's buildable rows, one file per MLST scheme (``<out-dir>/genome_manifest_<scheme>.csv``), the scheme
    taken from the genome's label genus (``--scheme-by-genus Escherichia=ecoli_achtman_4,…``; the genus is the first
    word of ``cohort.csv``'s ``species``). A genus with no scheme goes to ``genome_manifest_auto.csv``, built without
    ``--scheme`` (mlst picks). Each build array then runs with its scheme pinned: auto-detection typed 7.7 % of the
    2026-07 Enterobacterales genomes on another genus's scheme. Written read-only, since array tasks take rows by
    position.
``qc-list``
    Freeze the next CheckM2 list: every genome built ``ok`` under ``--shard-root`` that no ``--listed`` list holds, as
    ``Sample<TAB>contigs path``, sorted, read-only. ``--require-complete`` refuses while any buildable genome of
    ``--manifest`` has no shard row. Without ``--out`` it writes nothing and exits 1 unless every built genome is
    listed (the check before skani and collate).
``collate-inputs``
    One manifest, one shard table, one CheckM2 report and one skani table over the whole cohort — this cohort's builds
    plus each reused genome's rows from the cohort that built it — into ``--out-dir``, for ``collate_genomes``.
    Every reused genome its owner built must have a shard row in ``--reuse-shard-dir``, and the row's source must be
    the owner manifest's. CheckM2 and skani rows are kept for this cohort's genomes only.
``status``
    Per scheme and source: array tasks written against those expected, genomes ok and failed; then CheckM2 lists and
    their reports.

Usage: through ``engine/genomes/scripts/build_genomes_reusing_built.sh`` and a cohort's wrapper (e.g.
``apps/entero/scripts/build_entero_genomes_cabbage_v2.sh``), never by hand.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from bacpredict.engine.genomes.collate_genomes import read_shards

AUTO = "auto"  # the scheme name for genomes of a genus with no pinned scheme: built without --scheme
SOURCE_CHUNKS = {"atb": 100, "portal_gff": 40, "shovill": 1}  # genomes per array task, as the submit stage sizes them


def read_built(paths: Sequence[Path]) -> pd.DataFrame:
    """Every row of the earlier cohorts' manifests: ``Sample, source, drop_reason, owner`` (the manifest's path).

    A genome named by two of them is an error: the store would hold one copy for two owners.
    """
    frames = []
    for p in paths:
        m = pd.read_csv(p, dtype=str)
        if "Sample" not in m.columns or "source" not in m.columns:
            raise ValueError(f"{p} is not a genome manifest (no Sample/source column)")
        frames.append(m.reindex(columns=["Sample", "source", "drop_reason"]).assign(owner=str(p)))
    b = pd.concat(frames, ignore_index=True)
    dup = b.loc[b["Sample"].duplicated(keep=False)].sort_values("Sample")
    if len(dup):
        raise ValueError(f"{dup['Sample'].nunique()} genomes are in more than one --built manifest, e.g. "
                         f"{dup.head(4)[['Sample', 'owner']].values.tolist()}")
    return b


def partition(cohort: pd.DataFrame, built: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(cohort rows no earlier manifest names, the labelled genomes one does)``.

    The second table is ``Sample, source, drop_reason, owner``: the owner's build decision, reused as it stands —
    a genome the owner dropped stays dropped, and one it failed to build stays failed.
    """
    owned = cohort["Sample"].isin(built["Sample"])
    labelled = pd.to_numeric(cohort["n_drugs_labelled"], errors="coerce").fillna(0) > 0
    reused = cohort.loc[owned & labelled, ["Sample"]].merge(built, on="Sample", how="left")
    return cohort[~owned].reset_index(drop=True), reused.sort_values("Sample").reset_index(drop=True)


def overlap(manifest: pd.DataFrame, built: pd.DataFrame) -> list[str]:
    """Genomes of ``manifest`` (any row, buildable or not) that an earlier manifest names."""
    return sorted(set(manifest["Sample"]) & set(built["Sample"]))


def parse_mapping(text: str) -> dict[str, str]:
    """``"A=x,B=y"`` → ``{"A": "x", "B": "y"}``."""
    pairs = [x.split("=", 1) for x in text.split(",") if x.strip()]
    if any(len(p) != 2 for p in pairs):
        raise ValueError(f"expected Key=value,… — got {text!r}")
    return {k.strip(): v.strip() for k, v in pairs}


def split_by_scheme(manifest: pd.DataFrame, species: pd.Series, by_genus: dict[str, str]) -> dict[str, pd.DataFrame]:
    """The manifest's buildable rows grouped by MLST scheme, from each genome's label genus (``AUTO`` if unmapped).

    ``species`` is the label species per Sample (``cohort.csv``); a buildable genome with no species is an error.
    """
    if AUTO in by_genus.values():
        raise ValueError(f"{AUTO!r} names the unpinned group and cannot be a scheme")
    m = manifest[manifest["source"].ne("drop")]
    sp = m["Sample"].map(species)
    if sp.isna().any():
        raise ValueError(f"{int(sp.isna().sum())} buildable genomes have no species in the cohort, e.g. "
                         f"{m.loc[sp.isna(), 'Sample'].head(5).tolist()}")
    scheme = sp.str.split().str[0].map(by_genus).fillna(AUTO)
    return {k: g.reset_index(drop=True) for k, g in m.groupby(scheme, sort=True)}


def read_shard_dirs(dirs: Sequence[Path]) -> pd.DataFrame:
    """The newest shard row per genome in each directory; a genome in two directories is an error."""
    rows = pd.concat([read_shards(d).assign(shard_dir=str(d)) for d in dirs], ignore_index=True)
    dup = rows["Sample"].duplicated(keep=False)
    if dup.any():
        raise ValueError(f"{int(rows.loc[dup, 'Sample'].nunique())} genomes have shard rows in more than one of "
                         f"{len(dirs)} shard directories, e.g. {sorted(rows.loc[dup, 'Sample'].unique())[:5]}")
    return rows


def existing_shard_dirs(root: Path) -> list[Path]:
    """The per-scheme shard directories under ``root`` that hold at least one shard."""
    return sorted(d for d in root.iterdir() if d.is_dir() and any(d.glob("*.tsv"))) if root.exists() else []


def next_qc_list(manifest: pd.DataFrame, shards: pd.DataFrame, listed: set[str], require_complete: bool) -> pd.DataFrame:
    """``Sample, contigs_path`` of every genome built ok and in no CheckM2 list yet, sorted by Sample."""
    want = set(manifest.loc[manifest["source"].ne("drop"), "Sample"])
    s = shards[shards["Sample"].isin(want)]
    missing = want - set(s["Sample"])
    if missing and require_complete:
        raise ValueError(f"{len(missing)} of {len(want)} buildable genomes have no shard row yet, e.g. "
                         f"{sorted(missing)[:5]}: finish the build first")
    ok = s[s["status"].eq("ok") & ~s["Sample"].isin(listed)].sort_values("Sample")
    if not ok["contigs_path"].astype(str).str.startswith("/").all():
        raise ValueError("contigs paths must be absolute")
    return ok[["Sample", "contigs_path"]].reset_index(drop=True)


def _sample_from(path_col: pd.Series) -> pd.Series:
    return path_col.astype(str).map(lambda p: Path(p).name).str.replace(r"\.fna(\.gz)?$", "", regex=True)


def collate_inputs(
    cohort: pd.DataFrame,
    manifest: pd.DataFrame,
    shards: pd.DataFrame,
    reused: pd.DataFrame,
    reuse_shards: pd.DataFrame,
    checkm2: pd.DataFrame,
    skani: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(manifest, shards, checkm2, skani)`` over the whole cohort: this cohort's builds plus the reused genomes.

    ``n_drugs_labelled`` of a reused row is this cohort's (from ``cohort``), not the owner's. ``checkm2`` and ``skani``
    are the raw reports (``Name`` / ``Query_file`` columns), concatenated; only this cohort's genomes are kept.
    """
    if dup := sorted(set(manifest["Sample"]) & set(reused["Sample"])):
        raise ValueError(f"{len(dup)} genomes are both planned here and reused, e.g. {dup[:5]}")
    n_lab = cohort.set_index("Sample")["n_drugs_labelled"]
    r = reused.assign(n_drugs_labelled=reused["Sample"].map(n_lab), built_by=reused["owner"])
    m = pd.concat([manifest.assign(built_by="this cohort"), r.drop(columns="owner")], ignore_index=True)

    mine = shards[shards["Sample"].isin(manifest["Sample"])]
    r_built = reused[reused["source"].ne("drop")]
    theirs = reuse_shards[reuse_shards["Sample"].isin(r_built["Sample"])]
    if missing := sorted(set(r_built["Sample"]) - set(theirs["Sample"])):
        raise ValueError(f"{len(missing)} reused genomes have no shard row in the owners' shard directories, e.g. "
                         f"{missing[:5]}: pass the shard directory of every cohort that built them")
    src = theirs.set_index("Sample")["source"]
    moved = r_built[r_built["source"].ne(r_built["Sample"].map(src))]
    if len(moved):
        raise ValueError(f"{len(moved)} reused genomes were built from a source other than their owner manifest's, "
                         f"e.g. {moved['Sample'].head(5).tolist()}")
    sh = pd.concat([mine, theirs], ignore_index=True).drop(columns="shard_dir", errors="ignore")

    keep = set(m["Sample"])
    q = checkm2.assign(Sample=_sample_from(checkm2["Name"]))
    q = q[q["Sample"].isin(keep)]
    if (d := q["Sample"].duplicated()).any():
        raise ValueError(f"{int(d.sum())} genomes have more than one CheckM2 row, e.g. {q.loc[d, 'Sample'].head(5).tolist()}")
    s = skani.assign(Sample=_sample_from(skani["Query_file"]))
    s = s[s["Sample"].isin(keep)]
    return m, sh, q.drop(columns="Sample"), s.drop(columns="Sample")


def status(by_scheme: Path, shard_root: Path, qc_dir: Path) -> str:
    """Build and CheckM2 progress, as text."""
    lines = []
    for f in sorted(by_scheme.glob("genome_manifest_*.csv")):
        scheme = f.stem.removeprefix("genome_manifest_")
        m = pd.read_csv(f, dtype=str)
        d = shard_root / scheme
        shards = sorted(d.glob("*.tsv")) if d.exists() else []
        for src, chunk in SOURCE_CHUNKS.items():
            n = -(-int(m["source"].eq(src).sum()) // chunk)
            if n == 0:
                continue
            have = {int(p.stem.rsplit("_", 1)[1]) for p in shards if p.stem.startswith(src + "_")}
            missing = sorted(set(range(n)) - have)
            more = " …" if len(missing) > 30 else ""
            lines.append(f"{scheme:>16} {src:<10} {len(have):>5}/{n:<5} tasks written"
                         + (f"; missing {','.join(map(str, missing[:30]))}{more}" if missing else ""))
        if shards:
            st = read_shards(d)["status"]
            lines.append(f"{scheme:>16} genomes: {int(st.eq('ok').sum())} ok, {int(st.ne('ok').sum())} failed, "
                         f"of {len(m)} to build")
    for lst in sorted(qc_dir.glob("checkm2_list_*.tsv")):
        k = lst.stem.removeprefix("checkm2_list_")
        n_rows = len(lst.read_text().splitlines())
        reports = list((qc_dir / f"list_{k}" / "checkm2").glob("checkm2_quality_report_*.tsv"))
        got = sum(max(len(p.read_text().splitlines()) - 1, 0) for p in reports)
        lines.append(f"CheckM2 list {k}: {n_rows} genomes, {len(reports)} reports, {got} genomes reported")
    sk = qc_dir / "skani" / "skani_dist.tsv"
    lines.append(f"skani: {'written' if sk.exists() else 'not written'} ({sk})")
    return "\n".join(lines)


def _write_frozen(df: pd.DataFrame, path: Path, **kw) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists and is frozen (array tasks take its rows by position)")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    df.to_csv(tmp, **kw)
    tmp.replace(path)
    path.chmod(0o444)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("partition")
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--built", type=Path, nargs="+", required=True, help="every earlier cohort's genome_manifest.csv")
    p.add_argument("--out-cohort", type=Path, required=True)
    p.add_argument("--out-reused", type=Path, required=True)
    p = sub.add_parser("check-disjoint")
    p.add_argument("--manifest", type=Path, nargs="+", required=True)
    p.add_argument("--built", type=Path, nargs="+", required=True)
    p = sub.add_parser("split-by-scheme")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--scheme-by-genus", required=True, help="Genus=scheme,…")
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("qc-list")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--shard-root", type=Path, required=True, help="the per-scheme shard directories' parent")
    p.add_argument("--listed", type=Path, nargs="*", default=[])
    p.add_argument("--require-complete", action="store_true")
    p.add_argument("--out", type=Path, default=None, help="omit to check that nothing is left to list")
    p = sub.add_parser("collate-inputs")
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--shard-root", type=Path, required=True, help="the per-scheme shard directories' parent")
    p.add_argument("--reused", type=Path, required=True)
    p.add_argument("--reuse-shard-dir", type=Path, nargs="+", required=True)
    p.add_argument("--checkm2", type=Path, nargs="+", required=True)
    p.add_argument("--skani", type=Path, nargs="+", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("status")
    p.add_argument("--by-scheme", type=Path, required=True)
    p.add_argument("--shard-root", type=Path, required=True)
    p.add_argument("--qc-dir", type=Path, required=True)
    a = ap.parse_args(argv)

    if a.cmd == "partition":
        cohort = pd.read_csv(a.cohort, dtype={"Sample": str, "assembly_genotype": str, "reads_accession": str})
        to_plan, reused = partition(cohort, read_built(a.built))
        for df, path in ((to_plan, a.out_cohort), (reused, a.out_reused)):
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
        n_lab = int((pd.to_numeric(to_plan["n_drugs_labelled"], errors="coerce") > 0).sum())
        print(f"{len(cohort)} cohort genomes: {n_lab} labelled genomes to plan → {a.out_cohort}; "
              f"{len(reused)} labelled genomes reused → {a.out_reused}")
        if len(reused):
            print(reused.groupby(["owner", "source"]).size().to_string())
    elif a.cmd == "check-disjoint":
        built = read_built(a.built)
        bad = {str(f): overlap(pd.read_csv(f, dtype=str, usecols=["Sample"]), built) for f in a.manifest}
        bad = {f: s for f, s in bad.items() if s}
        if bad:
            sys.exit("NOT DISJOINT — " + "; ".join(f"{f}: {len(s)} genomes in an earlier manifest, e.g. {s[:5]}"
                                                   for f, s in bad.items()))
        print(f"disjoint: {len(a.manifest)} manifest(s) name no genome of the {len(built)} in {len(a.built)} "
              "earlier manifest(s)")
    elif a.cmd == "split-by-scheme":
        manifest = pd.read_csv(a.manifest, dtype=str)
        species = pd.read_csv(a.cohort, dtype=str, usecols=["Sample", "species"]).set_index("Sample")["species"]
        by = split_by_scheme(manifest, species, parse_mapping(a.scheme_by_genus))
        for scheme, g in by.items():
            if not re.fullmatch(r"[A-Za-z0-9_]+", scheme):
                raise SystemExit(f"scheme name {scheme!r} is not a plain identifier")
            _write_frozen(g, a.out_dir / f"genome_manifest_{scheme}.csv", index=False)
        print(pd.DataFrame({s: g["source"].value_counts() for s, g in by.items()}).fillna(0).astype(int).T
              .to_string())
    elif a.cmd == "qc-list":
        manifest = pd.read_csv(a.manifest, dtype=str)
        listed: set[str] = set()
        for f in a.listed:
            listed |= set(pd.read_csv(f, sep="\t", header=None, dtype=str)[0])
        dirs = existing_shard_dirs(a.shard_root)
        if not dirs:
            raise SystemExit(f"no shards under {a.shard_root}")
        try:
            out = next_qc_list(manifest, read_shard_dirs(dirs), listed, a.require_complete)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        if a.out is None:
            if len(out):
                raise SystemExit(f"{len(out)} built genomes are in no CheckM2 list, e.g. {out['Sample'].head(5).tolist()}")
            print(f"every built genome is in a CheckM2 list ({len(listed)} listed)")
            return
        if out.empty:
            raise SystemExit("no built genome is outside the existing CheckM2 lists: nothing to freeze")
        _write_frozen(out, a.out, sep="\t", header=False, index=False)
        print(f"CheckM2 list: {len(out)} built genomes ({len(listed)} already listed) → {a.out}")
    elif a.cmd == "collate-inputs":
        cohort = pd.read_csv(a.cohort, dtype={"Sample": str})
        manifest = pd.read_csv(a.manifest, dtype=str)
        reused = pd.read_csv(a.reused, dtype=str)
        try:
            m, sh, q, s = collate_inputs(
                cohort, manifest, read_shard_dirs(existing_shard_dirs(a.shard_root)), reused,
                pd.concat([read_shards(d) for d in a.reuse_shard_dir], ignore_index=True)
                .drop_duplicates("Sample", keep="last"),
                pd.concat([pd.read_csv(f, sep="\t") for f in a.checkm2], ignore_index=True),
                pd.concat([pd.read_csv(f, sep="\t") for f in a.skani], ignore_index=True))
        except ValueError as e:
            raise SystemExit(str(e)) from e
        (a.out_dir / "shards").mkdir(parents=True, exist_ok=True)
        m.to_csv(a.out_dir / "genome_manifest.csv", index=False)
        sh.to_csv(a.out_dir / "shards" / "genome_shards.tsv", sep="\t", index=False)
        q.to_csv(a.out_dir / "checkm2_quality_report.tsv", sep="\t", index=False)
        s.to_csv(a.out_dir / "skani_dist.tsv", sep="\t", index=False)
        print(f"collate inputs → {a.out_dir}: {len(m)} genomes ({int(m['built_by'].eq('this cohort').sum())} planned "
              f"here, {int(m['built_by'].ne('this cohort').sum())} reused); {len(sh)} shard rows, {len(q)} CheckM2 rows, "
              f"{s['Query_file'].nunique()} genomes with a skani hit")
    elif a.cmd == "status":
        print(status(a.by_scheme, a.shard_root, a.qc_dir))


if __name__ == "__main__":
    main()
