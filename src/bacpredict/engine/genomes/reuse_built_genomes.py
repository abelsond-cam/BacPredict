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
    taken from the genome's label species (``--scheme-by-species "Klebsiella aerogenes=kaerogenes,…"``), else its
    genus (``--scheme-by-genus Escherichia=ecoli_achtman_4,…``; the genus is the first word of ``cohort.csv``'s
    ``species``). A genome matching neither goes to ``genome_manifest_auto.csv``, built without ``--scheme`` (mlst
    picks). Each build array then runs with its scheme pinned: auto-detection typed 7.7 % of the
    2026-07 Enterobacterales genomes on another genus's scheme. Written read-only, since array tasks take rows by
    position.
``check-schemes``
    After collate, before its outputs are published. A genome **built here** that passes QC must be on the scheme its
    label species or genus pins (a species typed on two schemes puts one clone under two ``st`` keys in a split table).
    A **reused** genome must carry exactly the ``mlst_scheme`` and ``st`` its owner's ``genome_qc.csv`` records: the
    owner's re-typed STs win in its shard directory only by file mtime (``collate_genomes.read_shards``), so this is
    the check that catches a copy or restore that lost them. Reused genomes whose v2 label pins another scheme (the
    owner typed them under its own label) and unpinned species typed on several schemes are listed, not failed: they
    are species questions, not build faults.
``qc-list``
    Freeze the next CheckM2 list: every genome built ``ok`` under ``--shard-root`` that no ``--listed`` list holds, as
    ``Sample<TAB>contigs path``, sorted, read-only. ``--require-complete`` refuses while any buildable genome of
    ``--manifest`` has no shard row. Without ``--out`` it writes nothing and exits 1 unless every built genome is
    listed (the check before skani and collate).
``collate-inputs``
    One manifest, one shard table, one CheckM2 report and one skani table over the whole cohort — this cohort's builds
    plus each reused genome's rows from the cohort that built it — into ``--out-dir``, for ``collate_genomes``.
    Every reused genome its owner built must have a shard row in ``--reuse-shard-dir``, and the row's source must be
    the owner manifest's. CheckM2 and skani rows are kept for this cohort's genomes only. Refuses unless the skani job
    ran on exactly the genomes of the current CheckM2 lists (``--skani-list`` against ``--lists``: a list frozen after
    skani would leave its genomes ``unknown``), and unless this cohort's skani references are, by file name, the ones
    the owner's skani rows name (``--refs-tsv`` against ``--reuse-skani``): reused and new genomes must be called
    against one panel.
``missing-tasks``
    The array indices of one scheme × source with no shard file holding exactly the task's genomes, as a SLURM
    ``--array`` list (``0-4,7``); empty when every task has written. ``submit`` sends only these, so running it again
    fills gaps and never duplicates a task that finished, and a shard left by an earlier plan whose rows shifted is not
    mistaken for done.
``replan-diff``
    Before a re-plan is published (``build_genomes_reusing_built.sh replan``): every genome whose manifest row changed
    must have moved between the source pairs ``--allow`` names, and ``reused_genomes.csv`` must be unchanged.
``status``
    Per scheme and source: array tasks written against those expected (names as ``submit`` takes them), the tasks
    holding a failed genome, genomes ok and failed; then CheckM2 lists, their reports, and the skani result.

The Bakta stage keeps the same split — a reused genome's annotation is its owner's:

``bakta-plan``
    ``built_genomes.csv`` (every genome built here so far) and ``bakrep_roster.tsv`` (the ATB-built ones whose
    BioSample BakRep's own export lists — not §3.6's ``in_bakrep_ids.txt``, which is §3.6's cohorts joined to it and
    lists no genome only a later cohort has). Re-run as more genomes are built; written atomically, never frozen.
``bakta-roster``
    The run roster under ``all``/``planned``/``refused`` (``bakta_roster``): genomes built here that BakRep cannot
    cover or that ingest did not take, and reused genomes with no annotation record at all, minus those our run
    already has. Refuses past ``--max-run``; written read-only, as array tasks take rows by position.
``bakta-status``
    Built genomes, the BakRep roster and its ingest outcomes, and each cohort genome's records (built here / reused).
``bakta-annotated``
    The genomes built here whose annotation has finished (``bakta_annotated``): a cohort to collate, extract proteins
    from and embed before QC exists, while the rest are still annotating. Written atomically, never frozen.

Usage: through ``engine/genomes/scripts/build_genomes_reusing_built.sh`` and
``engine/genomes/scripts/annotate_genomes_reusing_built.sh`` and a cohort's wrappers (e.g.
``apps/entero/scripts/build_entero_genomes_cabbage_v2.sh``), never by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from bacpredict.engine.genomes.bakta import ingest_record, run_record, running_marker, status_part
from bacpredict.engine.genomes.build_genomes import select_chunk
from bacpredict.engine.genomes.collate_genomes import read_shards

AUTO = "auto"  # the scheme name for genomes of a genus with no pinned scheme: built without --scheme
# genomes per array task, as the submit stage sizes them; the key is build_genomes' source, the value's name submit's
SOURCE_CHUNKS = {"atb": 100, "portal_gff": 40, "ncbi_gca": 40, "shovill": 1}
SUBMIT_NAME = {"atb": "atb", "portal_gff": "portal", "ncbi_gca": "ncbi", "shovill": "shovill"}


def read_built(paths: Sequence[Path]) -> pd.DataFrame:
    """Every row of the earlier cohorts' manifests: ``Sample, source, drop_reason, owner`` (the manifest's path).

    A genome named by two of them is an error: the store would hold one copy for two owners.
    """
    frames = []
    for p in paths:
        m = pd.read_csv(p, dtype=str, usecols=lambda c: c in {"Sample", "source", "drop_reason"})
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


def pinned_scheme(species: pd.Series, by_genus: dict[str, str], by_species: dict[str, str] | None = None) -> pd.Series:
    """Each label species' pinned MLST scheme: its species entry, else its genus entry, else ``AUTO``."""
    by_species = by_species or {}
    if AUTO in {*by_genus.values(), *by_species.values()}:
        raise ValueError(f"{AUTO!r} names the unpinned group and cannot be a scheme")
    s = species.fillna("").astype(str).str.strip()
    return s.map(by_species).fillna(s.str.split().str[0].map(by_genus)).fillna(AUTO)


def split_by_scheme(manifest: pd.DataFrame, species: pd.Series, by_genus: dict[str, str],
                    by_species: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """The manifest's buildable rows grouped by pinned MLST scheme (:func:`pinned_scheme`).

    ``species`` is the label species per Sample (``cohort.csv``); a buildable genome with no species is an error.
    """
    m = manifest[manifest["source"].ne("drop")]
    sp = m["Sample"].map(species)
    if sp.isna().any():
        raise ValueError(f"{int(sp.isna().sum())} buildable genomes have no species in the cohort, e.g. "
                         f"{m.loc[sp.isna(), 'Sample'].head(5).tolist()}")
    scheme = pinned_scheme(sp, by_genus, by_species)
    return {k: g.reset_index(drop=True) for k, g in m.groupby(scheme, sort=True)}


def scheme_mismatches(qc: pd.DataFrame, reused: set[str], owner_qc: pd.DataFrame, by_genus: dict[str, str],
                      by_species: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """The ``check-schemes`` tables (module docstring); the first two fail the check, the last two are reported.

    ``qc`` is this cohort's ``genome_qc.csv``; ``reused`` the Samples reused from an owner; ``owner_qc`` the owners'
    ``genome_qc.csv`` rows (``Sample, mlst_scheme, st``).

    - ``built_off_pin``: counts per (species, pin, mlst_scheme) of QC-passing genomes built here, off their pin.
    - ``reused_changed``: reused genomes built ok whose (mlst_scheme, st) differ from the owner's, or that the owner's
      table lacks.
    - ``reused_relabelled``: counts per (species, pin, mlst_scheme) of QC-passing reused genomes off their v2 pin.
    - ``unpinned_several``: counts per (species, mlst_scheme) of QC-passing unpinned species on more than one scheme.
    """
    def norm(s: pd.Series) -> pd.Series:
        return s.astype("string").fillna("-").str.strip()

    q = qc.assign(mlst_scheme=norm(qc["mlst_scheme"]), st=norm(qc["st"]), is_reused=qc["Sample"].isin(reused))
    ok = q["status"].eq("ok")
    passing = q[q["qc_pass"].astype(str).str.lower().eq("true")].copy()
    passing["pin"] = pinned_scheme(passing["species"], by_genus, by_species)
    off = passing[passing["pin"].ne(AUTO) & passing["mlst_scheme"].ne(passing["pin"])]

    def counts(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
        return df.groupby(by, dropna=False).size().rename("n").reset_index()

    o = owner_qc.assign(mlst_scheme=norm(owner_qc["mlst_scheme"]), st=norm(owner_qc["st"])).drop_duplicates("Sample")
    r = q[q["is_reused"] & ok][["Sample", "species", "mlst_scheme", "st"]].merge(
        o[["Sample", "mlst_scheme", "st"]], on="Sample", how="left", suffixes=("", "_owner"), indicator=True)
    changed = r[r["_merge"].ne("both") | r["mlst_scheme"].ne(r["mlst_scheme_owner"]) | r["st"].ne(r["st_owner"])]
    auto = passing[passing["pin"].eq(AUTO)]
    several = auto[auto.groupby("species")["mlst_scheme"].transform("nunique") > 1]
    return {
        "built_off_pin": counts(off[~off["is_reused"]], ["species", "pin", "mlst_scheme"]),
        "reused_changed": changed.drop(columns="_merge").reset_index(drop=True),
        "reused_relabelled": counts(off[off["is_reused"]], ["species", "pin", "mlst_scheme"]),
        "unpinned_several": counts(several, ["species", "mlst_scheme"]),
    }


def array_list(indices: Sequence[int]) -> str:
    """``[0, 1, 2, 4, 7, 8]`` → ``"0-2,4,7-8"``: a SLURM ``--array`` list."""
    out, idx = [], sorted(set(indices))
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        out.append(str(idx[i]) if i == j else f"{idx[i]}-{idx[j]}")
        i = j + 1
    return ",".join(out)


def task_files(shard_dir: Path, source: str) -> dict[int, Path]:
    """The shard files one source's array tasks have written in ``shard_dir``, by task index."""
    pat = re.compile(rf"{re.escape(source)}_(\d+)\.tsv")
    found = {}
    if shard_dir.exists():
        for f in shard_dir.iterdir():
            if m := pat.fullmatch(f.name):
                found[int(m.group(1))] = f
    return found


def missing_tasks(manifest: pd.DataFrame, source: str, chunk: int, shard_dir: Path) -> list[int]:
    """Indices of the tasks covering ``source`` that have no shard file holding exactly their genomes.

    A shard that holds other genomes than its task now covers was written under an earlier plan whose rows have since
    shifted: its task counts as not done.
    """
    n = -(-int(manifest["source"].eq(source).sum()) // chunk)
    files = task_files(shard_dir, source)
    todo = []
    for k in range(n):
        want = set(select_chunk(manifest, [source], chunk, k)["Sample"])
        f = files.get(k)
        if f is None or set(pd.read_csv(f, sep="\t", dtype=str, usecols=["Sample"])["Sample"]) != want:
            todo.append(k)
    return todo


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
    reuse_skani: pd.DataFrame | None = None,
    refs: Sequence[str] | None = None,
    skani_list: set[str] | None = None,
    listed: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(manifest, shards, checkm2, skani)`` over the whole cohort: this cohort's builds plus the reused genomes.

    ``n_drugs_labelled`` of a reused row is this cohort's (from ``cohort``), not the owner's. ``checkm2`` and ``skani``
    are the raw reports (``Name`` / ``Query_file`` columns), concatenated; only this cohort's genomes are kept.
    ``reuse_skani`` (the owners' skani rows) joins ``skani``; given ``refs`` (this cohort's reference paths), the
    owners' rows must name the same reference files. Given ``skani_list`` and ``listed``, the genomes skani ran on must
    be exactly the genomes of the CheckM2 lists.
    """
    if skani_list is not None and listed is not None and skani_list != listed:
        raise ValueError(f"skani ran on {len(skani_list)} genomes but the CheckM2 lists hold {len(listed)} "
                         f"({len(listed - skani_list)} not in skani's list, e.g. {sorted(listed - skani_list)[:5]}): "
                         "rerun qc skani")
    if reuse_skani is not None:
        if refs is not None:
            mine = {Path(r).name for r in refs}
            theirs = set(reuse_skani["Ref_file"].map(lambda p: Path(p).name))
            if mine != theirs:
                raise ValueError(f"this cohort's skani references differ from the owner's: only here "
                                 f"{sorted(mine - theirs)}, only the owner's {sorted(theirs - mine)}")
        skani = pd.concat([skani, reuse_skani], ignore_index=True)
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
            files = task_files(d, src)
            missing = array_list(sorted(set(range(n)) - set(files)))
            failed = array_list([k for k, f in files.items()
                                 if pd.read_csv(f, sep="\t", usecols=["status"])["status"].ne("ok").any()])
            lines.append(f"{scheme:>16} {SUBMIT_NAME[src]:<8} {len(files):>5}/{n:<5} tasks written"
                         + (f"; not written: {missing}" if missing else "")
                         + (f"; holding a failed genome: {failed}" if failed else ""))
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


def replan_diff(old: pd.DataFrame, new: pd.DataFrame, allowed: set[tuple[str, str]]
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(changed, disallowed)``: the genomes a re-plan moved, and those whose move ``allowed`` does not name.

    ``changed`` holds the ``old`` and ``new`` source per Sample; a move is ``(old source, new source)``.
    A row is changed if any column of the old manifest differs (columns only the new one has are ignored, e.g.
    ``reads_platform``). Both manifests must hold the same genomes.
    """
    if set(old["Sample"]) != set(new["Sample"]) or old["Sample"].duplicated().any():
        raise ValueError(f"the re-plan holds other genomes: {len(set(old['Sample']) - set(new['Sample']))} only in "
                         f"the old, {len(set(new['Sample']) - set(old['Sample']))} only in the new")
    if missing := sorted(set(old.columns) - set(new.columns)):
        raise ValueError(f"the new manifest lacks columns of the old: {missing}")
    o = old.set_index("Sample").fillna("")
    n = new.set_index("Sample").loc[o.index, o.columns].fillna("")
    moved = (o != n).any(axis=1)
    changed = pd.DataFrame({"old": o.loc[moved, "source"], "new": n.loc[moved, "source"]})
    ok = pd.Series([(a, b) in allowed for a, b in zip(changed["old"], changed["new"], strict=True)],
                   index=changed.index, dtype=bool)
    return changed, changed[~ok]


# ---- Bakta: the annotation stage over the same split between genomes built here and genomes reused -------------------

RUN_MODES = ("all", "planned", "refused")
# the record names bakta.py writes beside a genome's GFF3, as suffixes of its Sample
INGEST_SUFFIX = ingest_record(Path(), "").name  # ".bakta.ingest.json": BakRep's GFF3, ingested
RUN_SUFFIX = run_record(Path(), "").name  # ".bakta.run.json": our own run
RUNNING_SUFFIX = running_marker(Path(), "").name  # ".bakta.running": our run has started and not finished
GFF3_SUFFIX = ".bakta.gff3.gz"
REUSED_WHY = "reused, and no annotation record exists"


def bakrep_export_ids(path: Path) -> set[str]:
    """The BioSamples BakRep's export lists (its first column, ``#id``)."""
    ids = pd.read_csv(path, sep="\t", usecols=[0], dtype=str, quoting=csv.QUOTE_NONE)
    if ids.columns[0] != "#id":
        raise ValueError(f"{path}: its first column is {ids.columns[0]!r}, not BakRep's export's '#id'")
    return set(ids.iloc[:, 0].dropna())


def bakta_plan(manifest: pd.DataFrame, shards: pd.DataFrame, reused: pd.DataFrame, bakrep_ids: set[str]
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(built, bakrep_roster)``: the genomes this cohort has built so far, and those BakRep's GFF3 may stand for.

    ``built`` (``Sample, sr_assembly_file, source``) is every genome whose shard row is ``ok`` under the source this
    cohort's manifest planned for it (a row under another source was written under an earlier plan). ``bakrep_roster``
    (``Sample, sr_assembly_file``) is the ATB-built ones whose BioSample BakRep's export lists: BakRep's GFF3 sits on
    ATB's contigs, so it cannot stand for a genome built from any other source. A reused genome is never on either —
    its annotation is its owner's — and finding one among this cohort's builds is an error.
    """
    planned = manifest.set_index("Sample")["source"]
    ok = shards[shards["status"].eq("ok") & shards["source"].eq(shards["Sample"].map(planned))]
    if both := sorted(set(ok["Sample"]) & set(reused["Sample"])):
        raise ValueError(f"{len(both)} genomes are both built by this cohort and reused, e.g. {both[:5]}")
    built = (ok.rename(columns={"contigs_path": "sr_assembly_file"})[["Sample", "sr_assembly_file", "source"]]
             .sort_values("Sample").reset_index(drop=True))
    if not built["sr_assembly_file"].astype(str).str.startswith("/").all():
        raise ValueError("contigs paths must be absolute")
    roster = built.loc[built["source"].eq("atb") & built["Sample"].isin(bakrep_ids), ["Sample", "sr_assembly_file"]]
    return built, roster.reset_index(drop=True)


def _annotation_names(annotation_dir: Path) -> set[str]:
    """One directory read, not a stat per genome (the store holds several files per genome of every cohort)."""
    return set(os.listdir(annotation_dir)) if annotation_dir.is_dir() else set()


def _run_done(annotation_dir: Path, names: set[str], sample: str, contigs: str, fingerprint: str) -> bool:
    """Our run of ``sample`` with this env and database exists, written after its contigs.

    Not an md5 check: collate makes that one, over every cohort genome.
    """
    if sample + RUN_SUFFIX not in names or sample + GFF3_SUFFIX not in names:
        return False
    rec = annotation_dir / (sample + RUN_SUFFIX)
    try:
        return (json.loads(rec.read_text()).get("fingerprint") == fingerprint
                and rec.stat().st_mtime > os.stat(contigs).st_mtime)
    except (OSError, ValueError):
        return False


def bakta_roster(cohort: pd.DataFrame, built: pd.DataFrame, reused: pd.DataFrame, bakrep_roster: pd.DataFrame,
                 status: pd.DataFrame | None, annotation_dir: Path, fingerprint: str, mode: str,
                 allow_errors: bool = False, skip_reused: bool = False) -> pd.DataFrame:
    """The cohort genomes our Bakta must annotate under ``mode``: ``Sample, sr_assembly_file, why, done``.

    ``done`` marks those our run already has (``_run_done``); the run roster is the rest.

    - **Built here, not on the BakRep roster** (``planned``, ``all``): no BakRep file can stand for them.
    - **Built here, on the BakRep roster, not ingested** (``refused``, ``all``): ``absent``, ``refused:*`` and
      ``skipped:run``; ``error:*`` raises unless ``allow_errors`` (re-running ingest usually clears it). Needs
      ``status``, ingest's outcome per genome; a roster genome without one raises (plan has run since ingest did).
    - **Reused, with no annotation record at all** (``planned``, ``all``): its owner never annotated it. Every other
      reused genome is left to its owner's record, which collate then checks is current — this cohort never fetches
      or re-runs a genome its owner has annotated.

    ``skip_reused`` leaves that last class off (while an owner's Bakta job is live, it may be writing their files).
    A cohort genome that is neither reused nor in ``built`` raises (plan has not seen it), as does one in both.
    """
    if mode not in RUN_MODES:
        raise ValueError(f"mode {mode!r}: one of {RUN_MODES}")
    c = cohort[["Sample", "sr_assembly_file"]].reset_index(drop=True)
    if not c["Sample"].is_unique or c["sr_assembly_file"].isna().any():
        raise ValueError("the cohort holds a Sample twice, or a genome without contigs")
    is_reused = c["Sample"].isin(set(reused["Sample"]))
    source = c["Sample"].map(built.set_index("Sample")["source"])
    if (bad := is_reused & source.notna()).any():
        raise ValueError(f"{int(bad.sum())} cohort genomes are both reused and built here, e.g. "
                         f"{c.loc[bad, 'Sample'].head(3).tolist()}")
    if (bad := ~is_reused & source.isna()).any():
        raise ValueError(f"{int(bad.sum())} cohort genomes are neither reused nor in built_genomes.csv, e.g. "
                         f"{c.loc[bad, 'Sample'].head(3).tolist()}: re-run plan")
    bakrep = c["Sample"].isin(set(bakrep_roster["Sample"]))
    names = _annotation_names(annotation_dir)
    has_record = (c["Sample"] + INGEST_SUFFIX).isin(names) | (c["Sample"] + RUN_SUFFIX).isin(names)
    bare_reused = is_reused & ~has_record & (not skip_reused)
    not_in_bakrep = ~is_reused & ~bakrep
    st = pd.Series(pd.NA, index=c.index, dtype=object)
    if mode == "planned":
        take = not_in_bakrep | bare_reused
    else:
        if status is None:
            raise ValueError("no ingest status: run ingest first (or run mode planned, which needs none)")
        st = c["Sample"].map(status.set_index("Sample")["status"])
        if (none := bakrep & st.isna()).any():
            raise ValueError(f"{int(none.sum())} BakRep-roster cohort genomes have no ingest status, e.g. "
                             f"{c.loc[none, 'Sample'].head(3).tolist()}: re-run ingest")
        err = bakrep & st.str.startswith("error:", na=False)
        if err.any() and not allow_errors:
            raise ValueError(f"ingest ended {st[err].value_counts().to_dict()}: re-run ingest (it resumes) until none "
                             "do, or allow errors to run ours on them (the same annotation of record)")
        refused = bakrep & st.ne("ingested")
        take = refused if mode == "refused" else refused | not_in_bakrep | bare_reused
    why = (("not in BakRep: " + source.fillna("")).where(~bakrep, "BakRep " + st.fillna(""))
           .where(~is_reused, REUSED_WHY))
    want = c[take].assign(why=why[take])
    want["done"] = pd.Series([_run_done(annotation_dir, names, s, p, fingerprint)
                              for s, p in zip(want["Sample"], want["sr_assembly_file"], strict=True)],
                             index=want.index, dtype=bool)
    return want.sort_values("Sample").reset_index(drop=True)


def _newer(path: Path, than: str) -> bool:
    try:
        return path.stat().st_mtime > os.stat(than).st_mtime
    except OSError:
        return False


def read_sample_list(path: Path) -> set[str]:
    """Samples from a CSV with a ``Sample`` column (e.g. a check's problems.csv), or one per line."""
    first = path.read_text().split("\n", 1)[0].strip()
    if first.split(",")[0] == "Sample":
        return set(pd.read_csv(path, dtype=str, usecols=["Sample"])["Sample"].dropna())
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def bakta_annotated(built: pd.DataFrame, annotation_dir: Path, fingerprint: str, workers: int = 16
                    ) -> pd.DataFrame:
    """Every genome built here with its annotation state: ``Sample, sr_assembly_file, state``.

    ``state`` is ``ingest`` (BakRep's GFF3, ingested) or ``run`` (our run of ``fingerprint``) for a finished
    annotation — exactly one record, written after the genome's contigs, and no run of ours in progress — and
    otherwise why not: ``running``, ``both`` (collate refuses two records), ``stale`` (a record older than the contigs,
    or our run under another env) or ``none``. The finished ones are a cohort ``bakta.py collate`` can take before QC
    exists. Not an md5 check: collate makes that one. Per genome a stat or two and, for our run, its record: threaded,
    as on Lustre the metadata round trips, not the bytes, are the cost.
    """
    names = _annotation_names(annotation_dir)
    b = built[["Sample", "sr_assembly_file"]].reset_index(drop=True)
    if not b["Sample"].is_unique or b["sr_assembly_file"].isna().any():
        raise ValueError("built genomes: a Sample twice, or a genome without contigs")

    def one(row: tuple[str, str]) -> str:
        s, contigs = row
        if s + RUNNING_SUFFIX in names:
            return "running"
        has_ingest, has_run = s + INGEST_SUFFIX in names, s + RUN_SUFFIX in names
        if not (has_ingest or has_run):
            return "none"
        if has_ingest and has_run:
            return "both"
        if has_run:
            return "run" if _run_done(annotation_dir, names, s, contigs, fingerprint) else "stale"
        ok = s + GFF3_SUFFIX in names and _newer(annotation_dir / (s + INGEST_SUFFIX), contigs)
        return "ingest" if ok else "stale"

    rows = list(zip(b["Sample"].astype(str), b["sr_assembly_file"].astype(str), strict=True))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        state = list(pool.map(one, rows))
    return b.assign(state=state)


def read_roster(path: Path) -> pd.DataFrame:
    """A header-less ``Sample<TAB>contigs`` roster; an empty file is an empty roster."""
    if path.stat().st_size == 0:
        return pd.DataFrame({"Sample": pd.Series(dtype=str), "sr_assembly_file": pd.Series(dtype=str)})
    return pd.read_csv(path, sep="\t", header=None, names=["Sample", "sr_assembly_file"], dtype=str)


def bakta_status(plan_dir: Path, annotation_dir: Path, cohort_csv: Path | None, reused: set[str]) -> str:
    """Annotation progress, as text: built genomes, the BakRep roster and ingest, the cohort's records, run rosters."""
    lines = []
    if (f := plan_dir / "built_genomes.csv").exists():
        b = pd.read_csv(f, dtype=str)
        lines.append(f"built here: {len(b)} {b['source'].value_counts().to_dict()}")
    roster = read_roster(plan_dir / "bakrep_roster.tsv") if (plan_dir / "bakrep_roster.tsv").exists() else None
    final = plan_dir / "ingest_status.tsv"
    src = final if final.exists() else status_part(final) if status_part(final).exists() else None
    st = pd.read_csv(src, sep="\t", dtype=str) if src is not None else None
    if st is None:
        lines.append("no ingest status yet")
    else:
        lines.append(f"ingest status ({src.name}, {'final' if src == final else 'NOT final: ingest is running, or was killed'}"
                     f"): {len(st)} genomes {st['status'].value_counts().to_dict()}")
    if roster is not None:
        n = len(roster) if st is None else int((~roster["Sample"].isin(st["Sample"])).sum())
        lines.append(f"BakRep roster: {len(roster)} genomes, {n} without an ingest status")
    if cohort_csv is not None and cohort_csv.exists():
        names = _annotation_names(annotation_dir)
        c = pd.read_csv(cohort_csv, dtype={"Sample": str}, usecols=["Sample"])["Sample"]
        ing, run, r = (c + INGEST_SUFFIX).isin(names), (c + RUN_SUFFIX).isin(names), c.isin(reused)
        for label, m in (("built here", ~r), ("reused", r)):
            lines.append(f"cohort {cohort_csv.name}, {label}: {int(m.sum())} — ingest record {int((ing & m).sum())}, "
                         f"our run record {int((run & m).sum())}, both {int((ing & run & m).sum())}, "
                         f"neither {int((~ing & ~run & m).sum())}")
    else:
        lines.append(f"no cohort at {cohort_csv}")
    for f in sorted(plan_dir.glob("run_roster_*.tsv")):
        lines.append(f"  {f.name}: {len(read_roster(f))} genomes")
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
    p.add_argument("--scheme-by-species", default="", help="Genus epithet=scheme,…; wins over the genus")
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("check-schemes")
    p.add_argument("--genome-qc", type=Path, required=True)
    p.add_argument("--reused", type=Path, required=True)
    p.add_argument("--owner-genome-qc", type=Path, nargs="+", required=True)
    p.add_argument("--scheme-by-genus", required=True)
    p.add_argument("--scheme-by-species", default="")
    p = sub.add_parser("missing-tasks")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--source", required=True, choices=sorted(SOURCE_CHUNKS))
    p.add_argument("--shard-dir", type=Path, required=True)
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
    p.add_argument("--skani", type=Path, required=True, help="this cohort's skani_dist.tsv")
    p.add_argument("--reuse-skani", type=Path, nargs="+", required=True, help="the owners' skani_dist.tsv")
    p.add_argument("--refs-tsv", type=Path, required=True, help="the reference list this cohort's skani ran with")
    p.add_argument("--skani-list", type=Path, required=True, help="the genome list this cohort's skani ran on")
    p.add_argument("--lists", type=Path, nargs="+", required=True, help="this cohort's CheckM2 lists")
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("status")
    p.add_argument("--by-scheme", type=Path, required=True)
    p.add_argument("--shard-root", type=Path, required=True)
    p.add_argument("--qc-dir", type=Path, required=True)
    p = sub.add_parser("replan-diff")
    p.add_argument("--old-manifest", type=Path, required=True)
    p.add_argument("--new-manifest", type=Path, required=True)
    p.add_argument("--old-reused", type=Path, required=True)
    p.add_argument("--new-reused", type=Path, required=True)
    p.add_argument("--allow", required=True, help="old source:new source,… — the only moves the re-plan may make")
    p = sub.add_parser("bakta-plan")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--shard-root", type=Path, required=True, help="the per-scheme shard directories' parent")
    p.add_argument("--reused", type=Path, required=True)
    p.add_argument("--bakrep-export", type=Path, required=True, help="BakRep's bakrep-export.tsv.gz")
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("bakta-roster")
    p.add_argument("--cohort", type=Path, required=True, help="CSV with Sample, sr_assembly_file")
    p.add_argument("--built", type=Path, required=True, help="bakta-plan's built_genomes.csv")
    p.add_argument("--reused", type=Path, required=True)
    p.add_argument("--bakrep-roster", type=Path, required=True, help="bakta-plan's bakrep_roster.tsv")
    p.add_argument("--status", type=Path, default=None, help="ingest_status.tsv (not needed for mode planned)")
    p.add_argument("--annotation-dir", type=Path, required=True)
    p.add_argument("--fingerprint", required=True, help="bakta.py fingerprint of the env and database to run")
    p.add_argument("--mode", choices=RUN_MODES, required=True)
    p.add_argument("--max-run", type=int, required=True, help="refuse to write a roster of more genomes than this")
    p.add_argument("--allow-errors", action="store_true", help="run ours on genomes whose ingest ended error:*")
    p.add_argument("--skip-reused", action="store_true",
                   help="leave reused genomes off the roster (an owner's Bakta job is live); a later submit takes them")
    p.add_argument("--out", type=Path, required=True, help="the run roster to write (new; read-only once written)")
    p = sub.add_parser("bakta-status")
    p.add_argument("--plan-dir", type=Path, required=True)
    p.add_argument("--annotation-dir", type=Path, required=True)
    p.add_argument("--cohort", type=Path, default=None)
    p.add_argument("--reused", type=Path, required=True)
    p = sub.add_parser("bakta-annotated")
    p.add_argument("--built", type=Path, required=True, help="bakta-plan's built_genomes.csv, re-run just before")
    p.add_argument("--annotation-dir", type=Path, required=True)
    p.add_argument("--fingerprint", required=True, help="the fingerprint our runs must carry (§3.6's)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--hold", type=Path, default=None,
                   help="Samples to leave out (one per line, or a CSV with a Sample column), e.g. a check's problems")
    p.add_argument("--out", type=Path, required=True, help="Sample, sr_assembly_file, origin of the finished ones")
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
        by = split_by_scheme(manifest, species, parse_mapping(a.scheme_by_genus), parse_mapping(a.scheme_by_species))
        for scheme, g in by.items():
            if not re.fullmatch(r"[A-Za-z0-9_]+", scheme):
                raise SystemExit(f"scheme name {scheme!r} is not a plain identifier")
            _write_frozen(g, a.out_dir / f"genome_manifest_{scheme}.csv", index=False)
        print(pd.DataFrame({s: g["source"].value_counts() for s, g in by.items()}).fillna(0).astype(int).T
              .to_string())
    elif a.cmd == "check-schemes":
        qc = pd.read_csv(a.genome_qc, dtype={"Sample": str, "st": str, "mlst_scheme": str})
        reused = set(pd.read_csv(a.reused, dtype=str, usecols=["Sample"])["Sample"])
        owner = pd.concat([pd.read_csv(f, dtype=str, usecols=["Sample", "mlst_scheme", "st"]) for f in a.owner_genome_qc],
                          ignore_index=True)
        t = scheme_mismatches(qc, reused, owner, parse_mapping(a.scheme_by_genus), parse_mapping(a.scheme_by_species))
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            for k, why in (("reused_relabelled", "reused genomes off their v2-label pin (typed by the owner under its "
                                                 "own label; a species question, not an error)"),
                           ("unpinned_several", "unpinned species on more than one scheme (not an error)")):
                if len(t[k]):
                    print(f"{why}:\n{t[k].to_string(index=False)}")
            bad = [f"{k}:\n{t[k].head(50).to_string(index=False)}" for k in ("built_off_pin", "reused_changed") if len(t[k])]
        if bad:
            sys.exit("MLST CHECK FAILED — " + "\n".join(bad))
        print(f"MLST: every QC-passing genome built here is on its pin, and all {len(reused & set(qc['Sample']))} reused "
              "genomes carry their owner's scheme and ST")
    elif a.cmd == "missing-tasks":
        manifest = pd.read_csv(a.manifest, dtype=str, usecols=["Sample", "source"])
        print(array_list(missing_tasks(manifest, a.source, SOURCE_CHUNKS[a.source], a.shard_dir)))
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
        listed: set[str] = set()
        for f in a.lists:
            listed |= set(pd.read_csv(f, sep="\t", header=None, dtype=str)[0])
        try:
            m, sh, q, s = collate_inputs(
                cohort, manifest, read_shard_dirs(existing_shard_dirs(a.shard_root)), reused,
                pd.concat([read_shards(d) for d in a.reuse_shard_dir], ignore_index=True)
                .drop_duplicates("Sample", keep="last"),
                pd.concat([pd.read_csv(f, sep="\t") for f in a.checkm2], ignore_index=True),
                pd.read_csv(a.skani, sep="\t"),
                reuse_skani=pd.concat([pd.read_csv(f, sep="\t") for f in a.reuse_skani], ignore_index=True),
                refs=pd.read_csv(a.refs_tsv, sep="\t", header=None, dtype=str)[1].tolist(),
                skani_list=set(pd.read_csv(a.skani_list, sep="\t", header=None, dtype=str)[0]),
                listed=listed)
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
    elif a.cmd == "replan-diff":
        allowed = {tuple(x.strip().split(":", 1)) for x in a.allow.split(",") if x.strip()}
        if any(len(t) != 2 for t in allowed):
            raise SystemExit(f"--allow {a.allow!r}: expected old:new,…")
        old_r, new_r = (pd.read_csv(f, dtype=str) for f in (a.old_reused, a.new_reused))
        if not old_r.equals(new_r):
            raise SystemExit(f"reused_genomes.csv changed ({len(old_r)} → {len(new_r)} rows): an owner manifest moved")
        try:
            changed, bad = replan_diff(pd.read_csv(a.old_manifest, dtype=str), pd.read_csv(a.new_manifest, dtype=str),
                                       allowed)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        print(f"re-plan: {len(changed)} genomes change source" + (":" if len(changed) else ""))
        if len(changed):
            print(changed.value_counts().to_string())
        if len(bad):
            raise SystemExit(f"{len(bad)} changes are not among {sorted(allowed)}, e.g. "
                             f"{bad.head(5).reset_index().values.tolist()}: not published")
    elif a.cmd == "bakta-plan":
        manifest = pd.read_csv(a.manifest, dtype=str, usecols=["Sample", "source"])
        reused = pd.read_csv(a.reused, dtype=str, usecols=["Sample"])
        dirs = existing_shard_dirs(a.shard_root)
        if not dirs:
            raise SystemExit(f"no shards under {a.shard_root}: nothing is built yet")
        ids = bakrep_export_ids(a.bakrep_export)
        try:
            built, roster = bakta_plan(manifest, read_shard_dirs(dirs), reused, ids)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        a.out_dir.mkdir(parents=True, exist_ok=True)
        # atomically, never frozen: plan is re-run as genomes are built, and a running ingest may be reading the roster
        for df, name, kw in ((built, "built_genomes.csv", {}), (roster, "bakrep_roster.tsv", {"sep": "\t", "header": False})):
            tmp = a.out_dir / f".{name}.part"
            df.to_csv(tmp, index=False, **kw)
            tmp.replace(a.out_dir / name)
        n_atb = int(built["source"].eq("atb").sum())
        print(f"{len(dirs)} shard directories: {len(built)} of {int(manifest['source'].ne('drop').sum())} genomes to "
              f"build are built {built['source'].value_counts().to_dict()} → {a.out_dir / 'built_genomes.csv'}")
        print(f"BakRep ({len(ids)} BioSamples in {a.bakrep_export.name}): {len(roster)} of the {n_atb} ATB-built "
              f"genomes → {a.out_dir / 'bakrep_roster.tsv'}; the other {len(built) - len(roster)} built genomes take "
              "our run")
    elif a.cmd == "bakta-roster":
        cohort = pd.read_csv(a.cohort, dtype={"Sample": str}, usecols=["Sample", "sr_assembly_file"])
        status_df = pd.read_csv(a.status, sep="\t", dtype=str) if a.status is not None else None
        try:
            want = bakta_roster(cohort, pd.read_csv(a.built, dtype=str), pd.read_csv(a.reused, dtype=str),
                                read_roster(a.bakrep_roster), status_df, a.annotation_dir, a.fingerprint, a.mode,
                                a.allow_errors, a.skip_reused)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        todo = want[~want["done"]]
        print(f"cohort {len(cohort)} ({a.cohort}): {a.mode}: {len(want)} genomes for our run, of which "
              f"{int(want['done'].sum())} already have it → {len(todo)} to run")
        if todo.empty:
            return
        print(todo["why"].value_counts().to_string())
        if len(todo) > a.max_run:
            raise SystemExit(f"{len(todo)} genomes to run is more than the {a.max_run} allowed: no roster written. If "
                             f"the counts above are meant, re-run with MAX_RUN={len(todo)}")
        _write_frozen(todo[["Sample", "sr_assembly_file"]], a.out, sep="\t", header=False, index=False)
        print(f"run roster: {a.out}")
    elif a.cmd == "bakta-status":
        reused = set(pd.read_csv(a.reused, dtype=str, usecols=["Sample"])["Sample"])
        print(bakta_status(a.plan_dir, a.annotation_dir, a.cohort, reused))
    elif a.cmd == "bakta-annotated":
        built = pd.read_csv(a.built, dtype=str, usecols=["Sample", "sr_assembly_file"])
        try:
            t = bakta_annotated(built, a.annotation_dir, a.fingerprint, a.workers)
        except ValueError as e:
            raise SystemExit(str(e)) from e
        done = t[t["state"].isin(["ingest", "run"])].rename(columns={"state": "origin"})
        print(f"{len(t)} genomes built here ({a.built}): {t['state'].value_counts().to_dict()}")
        if a.hold is not None:
            hold = read_sample_list(a.hold)
            held = done["Sample"].isin(hold)
            print(f"held back ({a.hold}): {int(held.sum())} of the {len(hold)} listed are annotated; "
                  f"{len(hold - set(t['Sample']))} listed are not built here")
            done = done[~held]
        if done.empty:
            raise SystemExit("no genome built here has a finished annotation: nothing written")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = a.out.with_name(f".{a.out.name}.part")
        done.to_csv(tmp, index=False)
        tmp.replace(a.out)
        print(f"{len(done)} annotated → {a.out}")


if __name__ == "__main__":
    main()
