r"""Build the multi-species AMR split tables: an exact-ST zero-shot holdout beside a random holdout.

Writes one ``<drug>_split.csv`` per drug into ``--out-dir``, plus ``split_manifest.json``. The tables are read
by :func:`bacpredict.engine.splits.load_splits.load_splits` (``train``/``validate``/``holdout``) and
:func:`~bacpredict.engine.splits.load_splits.load_split_ids` (``holdout_st``) and by nothing else.

**Table.** Columns ``Sample, species, st_key, ast_label, split``, sorted by ``species`` then ``Sample``.

- ``species`` is the ``--species`` NAME (the species folder: ``paeruginosa``, ``enterobacterales``).
- ``st_key`` is ``<mlst_scheme>:<st>`` from the QC table. An untyped genome is ``<scheme>:-``, and a genome with
  no scheme is ``-:-``. ST numbers collide across schemes, so an ST is never keyed on ``st`` alone.
- ``ast_label`` is the label sheet's value, written as ``0``/``1`` or the fraction (``0.5``). A fractional label
  means conflicting tests: it keeps its row, but ``load_splits`` never scores it.
- ``split`` is one of ``train``, ``validate``, ``holdout``, ``holdout_st``.
- A drug's table holds every eligible genome with a label for that drug, and nothing else.

**Eligible** = listed in the species' eligible-genomes CSV (for PsA, ``genomes/bakta_annotation.csv``: annotated
and embedded) and not in any ``--exclude`` file.

**Label sheet** = ``binary_ast.csv`` as :mod:`bacpredict.engine.ast_labels.build_binary_ast_labels` writes it: the
genome key is its index column, headed ``phenotype-BioSample_ID``, then one column per drug. A sheet keyed
``Sample`` is accepted too. If it has both columns, they must agree on every row.

**Assignment is per genome and per species, never per drug.** For each species:

1. Genomes whose ``st_key`` is one of that species' ``--holdout-st`` → ``holdout_st``. Membership is the exact ST
   (David, 2026-11-20), so a single-locus variant stays in the random pool.
2. The rest are shuffled, as units: a ``--groups`` group (one isolate under several BioSamples) is one unit, so its
   genomes always share a split. Units are taken in shuffled order into ``holdout`` until it holds at least
   ``floor(holdout_fraction × genomes)`` genomes (exactly that many when no group is drawn). The remaining units are cut into ``n_folds`` folds
   (``numpy.array_split``): fold ``--fold`` → ``validate``, the others → ``train``.
3. The permutation comes from one RNG per species, seeded from ``(seed, sha256(species name))``. Units are sorted
   by key before shuffling, so input row order does not matter.

Each drug's table is that one assignment restricted to the genomes labelled for the drug. A genome's split is the
same in every drug's table, and ``--fold`` never moves a genome into or out of ``holdout``.

**Arms.** An arm is one run of this builder over one or more species. A cross-species arm is the union of its
species' assignments. Because each species has its own RNG and its own inputs, a species' rows are the same in
every arm that contains it. ``--assert-same-as NAME=<other arm dir>`` checks that before anything is written: this
arm's NAME rows must equal the NAME rows of the other arm's table, row for row and character for character, for
every drug, or nothing is written. Build the PsA-only arm first, then the PsA + Enterobacterales arm with
``--assert-same-as paeruginosa=<PsA-only dir>``.

A cross-species table pools its species in every split. To score one species, read it with
``load_splits(path, species=NAME)`` / ``load_split_ids(path, split, species=NAME)``, which return exactly that
species' own arm. Never filter the ``species`` column outside the reader.

**Fails loudly on:** a Sample duplicated in any input, or eligible in two species; an eligible genome missing from
its QC table; a label sheet with no key column, or with ``Sample`` and ``phenotype-BioSample_ID`` disagreeing; a
drug column missing from a label sheet, or a label outside 0–1; a held-out ST with no eligible
genomes, or an untyped one; a group spanning a held-out ST and not, or two species; an empty
``train``/``validate``/``holdout`` for a species; and output tables that already exist with different content
(``--overwrite`` replaces them) or a ``*_split.csv`` in ``--out-dir`` for a drug not in ``--drugs``.

``split_manifest.json`` records the arguments, every input path with its md5, the git SHA, numpy and pandas
versions, per-species genome counts (eligible, excluded, held-out ST, per split, groups used), each table's md5,
and per drug × species × split the rows, scoreable (0/1) rows and resistant rows.

Example (the PsA-only arm; a CSD3 login node is enough)::

    uv run python -m bacpredict.engine.splits.build_amr_holdout_splits \
      --species paeruginosa=$AMR/processed/paeruginosa/binary_ast.csv,$AMR/processed/paeruginosa/genomes/bakta_annotation.csv,$AMR/processed/paeruginosa/genomes/genome_qc.csv \
      --holdout-st paeruginosa=paeruginosa:235 \
      --exclude src/bacpredict/apps/pa/excluded_genomes.tsv \
      --groups src/bacpredict/apps/pa/duplicate_isolate_groups.csv \
      --drugs ciprofloxacin ceftazidime meropenem tobramycin amikacin imipenem piperacillin-tazobactam \
              aztreonam levofloxacin cefepime \
      --out-dir $AMR/processed/paeruginosa/splits
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bacpredict.engine.ast_labels.build_binary_ast_labels import ID_COL as LABEL_ID_ALIAS
from bacpredict.engine.splits.load_splits import SPLIT_VALUES

logger = logging.getLogger(__name__)

TABLE_COLUMNS = ("Sample", "species", "st_key", "ast_label", "split")
TABLE_SUFFIX = "_split.csv"
MANIFEST_NAME = "split_manifest.json"
UNTYPED = "-"


@dataclass(frozen=True)
class SpeciesInputs:
    """One species' three input tables (``--species NAME=<labels>,<eligible>,<qc>``)."""

    name: str
    labels: Path
    eligible: Path
    qc: Path


# --------------------------------------------------------------------------------------------- parsing helpers


def _split_name_spec(spec: str, flag: str) -> tuple[str, list[str]]:
    """``NAME=a,b,c`` → ``(NAME, [a, b, c])``; refuses an empty name or value."""
    name, sep, rest = spec.partition("=")
    values = [v.strip() for v in rest.split(",") if v.strip()]
    if not sep or not name.strip() or not values:
        raise ValueError(f"{flag} {spec!r}: expected NAME=<value>[,<value>…]")
    return name.strip(), values


def parse_species_spec(spec: str) -> SpeciesInputs:
    """Parse ``NAME=<labels csv>,<eligible genomes csv>,<qc csv>``."""
    name, paths = _split_name_spec(spec, "--species")
    if len(paths) != 3:
        raise ValueError(f"--species {spec!r}: expected NAME=<labels csv>,<eligible genomes csv>,<qc csv>")
    return SpeciesInputs(name, Path(paths[0]), Path(paths[1]), Path(paths[2]))


def parse_holdout_st_specs(specs: Iterable[str]) -> dict[str, list[str]]:
    """Parse repeatable ``NAME=<scheme:st>[,…]`` into ``{NAME: [st_key, …]}``."""
    out: dict[str, list[str]] = {}
    for spec in specs:
        name, keys = _split_name_spec(spec, "--holdout-st")
        for key in keys:
            scheme, sep, st = key.partition(":")
            if not sep or not scheme or not st:
                raise ValueError(f"--holdout-st {spec!r}: {key!r} is not <scheme>:<st>")
        out.setdefault(name, []).extend(keys)
    return out


def st_key(scheme: object, st: object) -> str:
    """``<scheme>:<st>``, with ``-`` for a missing scheme or ST."""
    def _clean(v: object) -> str:
        s = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()
        return s or UNTYPED

    return f"{_clean(scheme)}:{_clean(st)}"


def _st_keys(scheme: pd.Series, st: pd.Series) -> list[str]:
    """Vectorised :func:`st_key` over string columns (read with ``keep_default_na=False``: blank, never NaN)."""
    s = scheme.astype(str).str.strip().replace("", UNTYPED)
    t = st.astype(str).str.strip().replace("", UNTYPED)
    return (s + ":" + t).tolist()


def species_rng(seed: int, species: str) -> np.random.Generator:
    """The species' own RNG, from ``(seed, sha256(species))``; adding another species never changes it."""
    name_entropy = int.from_bytes(hashlib.sha256(species.encode("utf-8")).digest(), "big")
    return np.random.default_rng(np.random.SeedSequence([int(seed), name_entropy]))


# ------------------------------------------------------------------------------------------------- input readers


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_keyed(path: Path, columns: Sequence[str], *, sep: str = ",", what: str) -> pd.DataFrame:
    """Read ``columns`` of a table as strings, refusing a missing column or a duplicated ``Sample``."""
    df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, low_memory=False)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{what} {path} is missing column(s) {missing}; has {list(df.columns)[:20]}")
    df = df[list(columns)].copy()
    df["Sample"] = df["Sample"].str.strip()
    if (df["Sample"] == "").any():
        raise ValueError(f"{what} {path} has a row with an empty Sample")
    dup = df.loc[df["Sample"].duplicated(), "Sample"].unique().tolist()
    if dup:
        raise ValueError(f"{what} {path}: Sample duplicated: {dup[:10]}{' …' if len(dup) > 10 else ''}")
    return df


def read_sample_map(paths: Iterable[Path], value_column: str, *, sep: str, what: str) -> dict[str, str]:
    """``Sample -> value`` over one or more files; a Sample in two files is refused like one listed twice."""
    out: dict[str, str] = {}
    for path in paths:
        df = _read_keyed(Path(path), ["Sample", value_column], sep=sep, what=what)
        clash = sorted(set(df["Sample"]) & set(out))
        if clash:
            raise ValueError(f"{what} {path}: Sample already listed in an earlier file: {clash[:10]}")
        if (df[value_column].str.strip() == "").any():
            raise ValueError(f"{what} {path} has a row with an empty {value_column}")
        out.update(zip(df["Sample"], df[value_column].str.strip(), strict=True))
    return out


def read_labels(path: Path, drugs: Sequence[str]) -> pd.DataFrame:
    """The label sheet's ``drugs`` columns as floats, indexed by ``Sample``; NaN = not labelled.

    The genome key is ``Sample`` or, as :mod:`~bacpredict.engine.ast_labels.build_binary_ast_labels` writes it,
    ``phenotype-BioSample_ID`` (:data:`LABEL_ID_ALIAS`), the alias every other label reader accepts. A sheet with
    neither is refused, and so is one with both whose values differ on any row.
    """
    df = pd.read_csv(path, dtype={"Sample": str, LABEL_ID_ALIAS: str}, low_memory=False)
    has_sample, has_alias = "Sample" in df.columns, LABEL_ID_ALIAS in df.columns
    if not has_sample and not has_alias:
        raise ValueError(f"label sheet {path} has neither a Sample nor a {LABEL_ID_ALIAS} column; "
                         f"has {list(df.columns)[:20]}")
    missing = [d for d in drugs if d not in df.columns]
    if missing:
        raise ValueError(f"label sheet {path} is missing drug column(s) {missing}")
    if has_alias:
        alias = df[LABEL_ID_ALIAS].astype(str).str.strip()
        if has_sample:
            differ = df["Sample"].astype(str).str.strip() != alias
            if differ.any():
                pairs = list(zip(df.loc[differ, "Sample"], alias[differ], strict=True))[:5]
                raise ValueError(f"label sheet {path}: Sample and {LABEL_ID_ALIAS} disagree on {int(differ.sum())} "
                                 f"row(s), e.g. {pairs}")
        df["Sample"] = alias
    df["Sample"] = df["Sample"].astype(str).str.strip()
    dup = df.loc[df["Sample"].duplicated(), "Sample"].unique().tolist()
    if dup:
        raise ValueError(f"label sheet {path}: Sample duplicated: {dup[:10]}")
    labels = df.set_index("Sample")[list(drugs)].apply(pd.to_numeric, errors="raise").astype(float)
    rows, cols = np.nonzero(((labels < 0) | (labels > 1)).to_numpy())
    if len(rows):
        examples = [(labels.index[r], labels.columns[c], labels.iat[r, c]) for r, c in zip(rows[:5], cols[:5], strict=True)]
        raise ValueError(f"label sheet {path}: {len(rows)} label(s) outside 0–1, e.g. {examples}")
    return labels


def format_label(value: float) -> str:
    """``0``/``1`` for a whole label, else the shortest float repr (``0.5``): fixed per value, whatever the dtype."""
    v = float(value)
    return str(int(v)) if v.is_integer() else repr(v)


# ------------------------------------------------------------------------------------------- genome assignment


def assign_species_genomes(
    species: str,
    st_keys: Mapping[str, str],
    *,
    holdout_sts: Iterable[str] = (),
    groups: Mapping[str, str] | None = None,
    holdout_fraction: float = 0.2,
    n_folds: int = 5,
    fold: int = 0,
    seed: int = 1,
) -> pd.DataFrame:
    """Assign one species' eligible genomes to splits → ``DataFrame(Sample, species, st_key, split)``.

    Parameters
    ----------
    species : str
        The species name; with ``seed`` it seeds this species' RNG (:func:`species_rng`).
    st_keys : Mapping[str, str]
        ``Sample -> st_key`` for exactly the species' eligible genomes.
    holdout_sts : Iterable[str]
        ``st_key`` values held out whole (``holdout_st``). Each must have at least one genome and be typed.
    groups : Mapping[str, str], optional
        ``Sample -> group``; genomes of one group share a split. Members not in ``st_keys`` are ignored.
    holdout_fraction, n_folds, fold, seed
        Random holdout fraction (of the genomes not in a held-out ST), folds over the rest, the fold that is
        ``validate``, and the seed.

    Returns
    -------
    pd.DataFrame
        One row per genome in ``st_keys``, sorted by ``Sample``.
    """
    groups = groups or {}
    held = list(dict.fromkeys(holdout_sts))
    present = set(st_keys.values())
    for key in held:
        if key.endswith(f":{UNTYPED}"):
            raise ValueError(f"{species}: held-out ST {key!r} is untyped, which is not an ST")
        if key not in present:
            raise ValueError(f"{species}: held-out ST {key!r} has no eligible genomes")
    held_set = set(held)

    members: dict[str, list[str]] = {}
    for sample in sorted(st_keys):
        group = groups.get(sample)
        members.setdefault(f"group:{group}" if group is not None else f"sample:{sample}", []).append(sample)

    split_of: dict[str, str] = {}
    pool: list[str] = []
    for key in sorted(members):
        in_held = {st_keys[s] in held_set for s in members[key]}
        if in_held == {True, False}:
            detail = {s: st_keys[s] for s in members[key]}
            raise ValueError(f"{species}: {key} spans a held-out ST and not: {detail}")
        if True in in_held:
            split_of.update(dict.fromkeys(members[key], "holdout_st"))
        else:
            pool.append(key)

    order = [pool[i] for i in species_rng(seed, species).permutation(len(pool))]
    n_pool = sum(len(members[k]) for k in pool)
    target = int(np.floor(holdout_fraction * n_pool + 1e-9))
    taken = n_units = 0
    while n_units < len(order) and taken < target:
        split_of.update(dict.fromkeys(members[order[n_units]], "holdout"))
        taken += len(members[order[n_units]])
        n_units += 1
    rest = order[n_units:]
    for f, chunk in enumerate(np.array_split(np.arange(len(rest)), n_folds)):
        name = "validate" if f == fold else "train"
        for j in chunk:
            split_of.update(dict.fromkeys(members[rest[j]], name))

    out = pd.DataFrame({"Sample": sorted(st_keys)})
    out["species"] = species
    out["st_key"] = out["Sample"].map(st_keys)
    out["split"] = out["Sample"].map(split_of)
    empty = [s for s in ("train", "validate", "holdout") if not (out["split"] == s).any()]
    if empty:
        raise ValueError(f"{species}: no genomes in {empty} ({len(out)} eligible) — too few genomes for this split")
    return out


def drug_tables(assignment: pd.DataFrame, labels: pd.DataFrame, drugs: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Restrict one species' assignment to each drug's labelled genomes → ``{drug: table}`` (TABLE_COLUMNS)."""
    out: dict[str, pd.DataFrame] = {}
    for drug in drugs:
        y = assignment["Sample"].map(labels[drug])
        rows = assignment[y.notna()].copy()
        rows["ast_label"] = y[y.notna()].map(format_label)
        out[drug] = rows[list(TABLE_COLUMNS)].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------------------------------- the arm


def build_arm(
    species_inputs: Sequence[SpeciesInputs],
    drugs: Sequence[str],
    *,
    holdout_sts: Mapping[str, Sequence[str]] | None = None,
    exclusions: Mapping[str, str] | None = None,
    groups: Mapping[str, str] | None = None,
    holdout_fraction: float = 0.2,
    n_folds: int = 5,
    fold: int = 0,
    seed: int = 1,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Build every drug's table for one arm → ``({drug: table}, report)``. Writes nothing.

    Tables are sorted by ``species`` then ``Sample``. ``report`` holds the per-species genome counts and the per
    drug × species × split counts that go into ``split_manifest.json``.
    """
    holdout_sts = dict(holdout_sts or {})
    exclusions = dict(exclusions or {})
    groups = dict(groups or {})
    names = [s.name for s in species_inputs]
    if not names:
        raise ValueError("at least one --species is required")
    if len(set(names)) != len(names):
        raise ValueError(f"a species is given twice: {names}")
    unknown = sorted(set(holdout_sts) - set(names))
    if unknown:
        raise ValueError(f"--holdout-st names species not given with --species: {unknown}")
    if not drugs or len(set(drugs)) != len(drugs):
        raise ValueError(f"--drugs must be non-empty and unique: {list(drugs)}")
    if not 0 < holdout_fraction < 1:
        raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")
    if n_folds < 2 or not 0 <= fold < n_folds:
        raise ValueError(f"need n_folds >= 2 and 0 <= fold < n_folds, got n_folds={n_folds} fold={fold}")
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")

    eligible_of: dict[str, dict[str, str]] = {}  # species -> {Sample: st_key}
    report: dict = {"species": {}, "counts": {d: {} for d in drugs}}
    seen: dict[str, str] = {}
    for inp in sorted(species_inputs, key=lambda s: s.name):
        listed = _read_keyed(inp.eligible, ["Sample"], what="eligible genomes")["Sample"].tolist()
        clash = sorted(s for s in listed if s in seen)
        if clash:
            raise ValueError(f"Sample eligible in two species ({inp.name} and {seen[clash[0]]}): {clash[:10]}")
        seen.update(dict.fromkeys(listed, inp.name))
        excluded = sorted(s for s in listed if s in exclusions)
        eligible = [s for s in listed if s not in exclusions]
        qc = _read_keyed(inp.qc, ["Sample", "mlst_scheme", "st"], what="QC table").set_index("Sample")
        no_qc = sorted(set(eligible) - set(qc.index))
        if no_qc:
            raise ValueError(f"{inp.name}: {len(no_qc)} eligible genomes have no QC row: {no_qc[:10]}")
        sub_qc = qc.loc[eligible]
        keys = dict(zip(eligible, _st_keys(sub_qc["mlst_scheme"], sub_qc["st"]), strict=True))
        eligible_of[inp.name] = keys
        report["species"][inp.name] = {
            "n_eligible_listed": len(listed),
            "n_excluded": len(excluded),
            "excluded": excluded,
            "n_eligible": len(eligible),
            "n_untyped": sum(k.endswith(f":{UNTYPED}") for k in keys.values()),
            "holdout_sts": {k: sum(v == k for v in keys.values()) for k in holdout_sts.get(inp.name, [])},
        }

    # A group is one isolate: its eligible members must sit in one species (then one split, below).
    group_species: dict[str, set[str]] = {}
    for sample, group in groups.items():
        if sample in seen and sample not in exclusions:
            group_species.setdefault(group, set()).add(seen[sample])
    spanning = {g: sorted(s) for g, s in group_species.items() if len(s) > 1}
    if spanning:
        raise ValueError(f"group(s) span species: {spanning}")
    unmatched = sorted(set(exclusions) - set(seen))
    if unmatched:
        logger.warning("%d excluded Sample(s) are not eligible in any species (nothing to exclude): %s",
                       len(unmatched), unmatched[:10])
    report["excluded_not_eligible"] = unmatched

    tables_by_species: dict[str, dict[str, pd.DataFrame]] = {}
    for inp in sorted(species_inputs, key=lambda s: s.name):
        keys = eligible_of[inp.name]
        assignment = assign_species_genomes(
            inp.name, keys, holdout_sts=holdout_sts.get(inp.name, ()), groups=groups,
            holdout_fraction=holdout_fraction, n_folds=n_folds, fold=fold, seed=seed,
        )
        labels = read_labels(inp.labels, drugs)
        tables_by_species[inp.name] = drug_tables(assignment, labels, drugs)
        used: dict[str, list[str]] = {}
        for s in sorted(set(keys) & set(groups)):
            used.setdefault(groups[s], []).append(s)
        rep = report["species"][inp.name]
        rep["groups"] = dict(sorted(used.items()))
        rep["genome_splits"] = {s: int((assignment["split"] == s).sum()) for s in SPLIT_VALUES}
        for drug in drugs:
            labelled = set(labels.index[labels[drug].notna()])
            table = tables_by_species[inp.name][drug]
            y = table["ast_label"].astype(float)
            report["counts"][drug][inp.name] = {
                "n_labelled_not_eligible": len(labelled - set(keys)),
                **{s: {"n": int((table["split"] == s).sum()),
                       "n_scoreable": int(((table["split"] == s) & y.isin([0.0, 1.0])).sum()),
                       "n_R": int(((table["split"] == s) & (y == 1.0)).sum())} for s in SPLIT_VALUES},
            }

    tables = {
        drug: pd.concat([tables_by_species[n][drug] for n in sorted(tables_by_species)], ignore_index=True)
        for drug in drugs
    }
    return tables, report


def render_table(table: pd.DataFrame) -> str:
    """The exact CSV text written for a table (newline line ends, every field already a string)."""
    return table[list(TABLE_COLUMNS)].to_csv(index=False, lineterminator="\n")


def assert_same_as(tables: Mapping[str, pd.DataFrame], species: str, other_dir: Path) -> None:
    """Refuse unless ``species``' rows equal its rows in ``other_dir``'s table, for every drug in ``tables``."""
    other_dir = Path(other_dir)
    for drug, table in tables.items():
        path = other_dir / f"{drug}{TABLE_SUFFIX}"
        if not path.exists():
            raise ValueError(f"--assert-same-as {species}={other_dir}: no {path.name} there")
        theirs = pd.read_csv(path, dtype=str, keep_default_na=False)
        if tuple(theirs.columns) != TABLE_COLUMNS:
            raise ValueError(f"{path}: columns {list(theirs.columns)}, expected {list(TABLE_COLUMNS)}")
        ours = pd.read_csv(io.StringIO(render_table(table)), dtype=str, keep_default_na=False)
        theirs = theirs[theirs["species"] == species].reset_index(drop=True)
        ours = ours[ours["species"] == species].reset_index(drop=True)
        if not ours.equals(theirs):
            diff = ours.merge(theirs, how="outer", indicator=True)
            diff = diff[diff["_merge"] != "both"].head(5).to_dict("records")
            raise ValueError(
                f"{drug}: {species} rows differ from {path} ({len(ours)} here vs {len(theirs)} there); "
                f"first differences: {diff}"
            )
        logger.info("%s: %d %s rows equal %s", drug, len(ours), species, path)


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_arm(out_dir: Path, tables: Mapping[str, pd.DataFrame], manifest: dict, *, overwrite: bool = False) -> None:
    """Write ``<drug>_split.csv`` for every table, then ``split_manifest.json`` (with each table's md5).

    Refuses (writing nothing) if a table exists with different content and ``overwrite`` is false: a new split
    invalidates every model and read-out built on the old one. Always refuses a ``*_split.csv`` already in
    ``out_dir`` for a drug not in ``tables``, which would otherwise sit there looking current.
    """
    out_dir = Path(out_dir)
    rendered = {drug: render_table(t) for drug, t in tables.items()}
    if out_dir.exists():
        foreign = sorted(p.name for p in out_dir.glob(f"*{TABLE_SUFFIX}") if p.name[: -len(TABLE_SUFFIX)] not in tables)
        if foreign:
            raise FileExistsError(f"{out_dir} holds split tables for drugs not in this run: {foreign}")
        changed = [d for d, text in rendered.items()
                   if (out_dir / f"{d}{TABLE_SUFFIX}").exists() and (out_dir / f"{d}{TABLE_SUFFIX}").read_text() != text]
        if changed and not overwrite:
            raise FileExistsError(f"{out_dir}: existing split tables differ for {changed}; a new split invalidates "
                                  "every model trained on the old one — pass --overwrite only if that is intended")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {**manifest, "tables": {}}
    for drug, text in rendered.items():
        path = out_dir / f"{drug}{TABLE_SUFFIX}"
        _atomic_write(path, text)
        manifest["tables"][drug] = {"path": path.name, "md5": hashlib.md5(text.encode()).hexdigest(),
                                    "n_rows": len(tables[drug])}
    _atomic_write(out_dir / MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    logger.info("wrote %d split tables + %s -> %s", len(rendered), MANIFEST_NAME, out_dir)


def _counts_frame(report: dict) -> pd.DataFrame:
    rows = [{"drug": d, "species": sp, **{s: c[s]["n"] for s in SPLIT_VALUES}}
            for d, by_sp in report["counts"].items() for sp, c in by_sp.items()]
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: build one arm's split tables (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--species", action="append", required=True, metavar="NAME=LABELS,ELIGIBLE,QC",
                    help="repeatable: label sheet (Sample or phenotype-BioSample_ID, + drug columns), eligible "
                         "genomes (Sample), QC table (Sample, mlst_scheme, st)")
    ap.add_argument("--holdout-st", action="append", default=[], metavar="NAME=SCHEME:ST[,…]",
                    help="repeatable: STs of species NAME held out whole (exact ST)")
    ap.add_argument("--exclude", action="append", type=Path, default=[], metavar="TSV",
                    help="repeatable: TSV with Sample, reason; these genomes are not eligible")
    ap.add_argument("--groups", action="append", type=Path, default=[], metavar="CSV",
                    help="repeatable: CSV with Sample, group; a group's genomes share a split")
    ap.add_argument("--drugs", nargs="+", required=True, help="drug label columns (space- or comma-separated)")
    ap.add_argument("--holdout-fraction", type=float, default=0.2)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--fold", type=int, default=0, help="the fold that becomes validate")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--assert-same-as", action="append", default=[], metavar="NAME=DIR",
                    help="repeatable: refuse unless species NAME's rows equal those in DIR's tables, every drug")
    ap.add_argument("--overwrite", action="store_true", help="replace existing tables whose content differs")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    species = [parse_species_spec(s) for s in args.species]
    drugs = [d.strip() for item in args.drugs for d in item.split(",") if d.strip()]
    holdout_sts = parse_holdout_st_specs(args.holdout_st)
    same_as = []
    for spec in args.assert_same_as:
        name, dirs = _split_name_spec(spec, "--assert-same-as")
        if len(dirs) != 1 or name not in {s.name for s in species}:
            raise SystemExit(f"--assert-same-as {spec!r}: expected NAME=DIR with NAME one of the --species")
        same_as.append((name, Path(dirs[0])))

    exclusions = read_sample_map(args.exclude, "reason", sep="\t", what="exclusions")
    groups = read_sample_map(args.groups, "group", sep=",", what="groups")
    tables, report = build_arm(species, drugs, holdout_sts=holdout_sts, exclusions=exclusions, groups=groups,
                               holdout_fraction=args.holdout_fraction, n_folds=args.n_folds, fold=args.fold,
                               seed=args.seed)
    for name, other in same_as:
        assert_same_as(tables, name, other)

    inputs = [{"role": role, "species": s.name, "path": str(p.resolve()), "md5": _md5(p)}
              for s in species for role, p in (("labels", s.labels), ("eligible", s.eligible), ("qc", s.qc))]
    inputs += [{"role": "exclude", "path": str(p.resolve()), "md5": _md5(p)} for p in args.exclude]
    inputs += [{"role": "groups", "path": str(p.resolve()), "md5": _md5(p)} for p in args.groups]
    manifest = {
        "builder": "bacpredict.engine.splits.build_amr_holdout_splits",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "argv": list(sys.argv if argv is None else argv),
        "arguments": {
            "species": [s.name for s in species], "drugs": drugs, "holdout_sts": holdout_sts,
            "holdout_fraction": args.holdout_fraction, "n_folds": args.n_folds, "fold": args.fold,
            "seed": args.seed, "assert_same_as": {n: str(d.resolve()) for n, d in same_as},
        },
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "columns": list(TABLE_COLUMNS),
        "inputs": inputs,
        **report,
    }
    write_arm(args.out_dir, tables, manifest, overwrite=args.overwrite)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(_counts_frame(report).to_string(index=False))
    for name, rep in report["species"].items():
        print(f"{name}: genomes {rep['genome_splits']}; held-out STs {rep['holdout_sts']}; "
              f"excluded {rep['n_excluded']}; groups {len(rep['groups'])}")


if __name__ == "__main__":
    main()
