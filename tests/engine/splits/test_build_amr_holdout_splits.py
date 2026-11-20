"""Unit tests for the multi-species AMR split builder (``engine.splits.build_amr_holdout_splits``).

Synthetic two-species fixtures: determinism, a species' RNG independent of the other species, groups kept
together, the exact-ST holdout (scheme-keyed), exclusions, a union arm equal to its per-species arms, no Sample in
two splits, split fractions, the fail-loudly cases, and the written tables read back through ``load_splits``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")
import pandas as pd

from bacpredict.engine.splits import build_amr_holdout_splits as b
from bacpredict.engine.splits.load_splits import load_split_ids, load_splits

DRUGS = ("ciprofloxacin", "meropenem")
REPO = Path(__file__).resolve().parents[3]


def _species(tmp_path: Path, name: str, *, n: int = 400, scheme: str = "paeruginosa", prefix: str = "P",
             held_st: str = "235", n_held: int = 40, seed: int = 0) -> b.SpeciesInputs:
    """A synthetic species: eligible genomes + 5 not eligible, an ST mix with ``n_held`` in ``held_st``."""
    rng = np.random.default_rng(seed)
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    samples = [f"{prefix}{i:04d}" for i in range(n + 5)]  # the last 5 are labelled + QC'd but not eligible
    sts = [held_st] * n_held + [str(x) for x in rng.choice([1, 17, 111, 2350, 23], size=n + 5 - n_held)]
    sts[n_held] = "-"  # one untyped genome
    sts[n_held + 1] = f"{held_st}0"  # an ST that only starts like the held one
    qc = pd.DataFrame({"Sample": samples, "mlst_scheme": scheme, "st": sts, "qc_pass": True})
    qc.to_csv(d / "genome_qc.csv", index=False)
    pd.DataFrame({"Sample": samples[:n], "sr_gff_file": "x.gff3"}).to_csv(d / "bakta_annotation.csv", index=False)
    labels = pd.DataFrame({"Sample": samples,
                           "ciprofloxacin": rng.integers(0, 2, len(samples)).astype(float),
                           "meropenem": np.where(rng.random(len(samples)) < 0.7, rng.integers(0, 2, len(samples)),
                                                 np.nan)})
    labels.loc[3, "ciprofloxacin"] = 0.5  # conflicting tests: a row, never scored
    labels.to_csv(d / "binary_ast.csv", index=False)
    return b.SpeciesInputs(name, d / "binary_ast.csv", d / "bakta_annotation.csv", d / "genome_qc.csv")


@pytest.fixture
def pa(tmp_path: Path) -> b.SpeciesInputs:
    return _species(tmp_path, "paeruginosa")


@pytest.fixture
def entero(tmp_path: Path) -> b.SpeciesInputs:
    # same ST number in another scheme: must NOT be held out by paeruginosa:235
    return _species(tmp_path, "enterobacterales", n=600, scheme="klebsiella", prefix="E", held_st="235",
                    n_held=30, seed=1)


def _genome_splits(tables: dict[str, pd.DataFrame]) -> dict[str, str]:
    """Sample -> split over every drug table, asserting one split per genome across drugs."""
    out: dict[str, str] = {}
    for table in tables.values():
        assert table["Sample"].is_unique
        for s, split in zip(table["Sample"], table["split"], strict=True):
            assert out.setdefault(s, split) == split, f"{s} in two splits across drugs"
    return out


def test_table_contract_columns_and_sort(pa: b.SpeciesInputs) -> None:
    """Columns are the contract's, rows sorted species then Sample, st_key is scheme:st, splits known."""
    tables, _ = b.build_arm([pa], DRUGS, holdout_sts={"paeruginosa": ["paeruginosa:235"]})
    for t in tables.values():
        assert tuple(t.columns) == b.TABLE_COLUMNS
        assert t["Sample"].tolist() == sorted(t["Sample"])
        assert set(t["split"]) <= set(b.SPLIT_VALUES)
        assert t["st_key"].str.startswith("paeruginosa:").all()
    assert "paeruginosa:-" in set(tables["ciprofloxacin"]["st_key"])


def test_deterministic_and_independent_of_input_row_order(tmp_path: Path, pa: b.SpeciesInputs) -> None:
    """Same inputs → identical text; shuffling every input file's rows changes nothing."""
    kw = {"holdout_sts": {"paeruginosa": ["paeruginosa:235"]}}
    first = {d: b.render_table(t) for d, t in b.build_arm([pa], DRUGS, **kw)[0].items()}
    again = {d: b.render_table(t) for d, t in b.build_arm([pa], DRUGS, **kw)[0].items()}
    assert first == again
    for path in (pa.labels, pa.eligible, pa.qc):
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df.sample(frac=1, random_state=7).to_csv(path, index=False)
    shuffled = {d: b.render_table(t) for d, t in b.build_arm([pa], DRUGS, **kw)[0].items()}
    assert shuffled == first


def test_species_rng_is_per_species_and_seeded() -> None:
    """The RNG depends on (seed, species) only: same pair → same stream, different name or seed → different."""
    def draw(seed: int, name: str) -> list[int]:
        return b.species_rng(seed, name).permutation(50).tolist()

    assert draw(1, "paeruginosa") == draw(1, "paeruginosa")
    assert draw(1, "paeruginosa") != draw(1, "enterobacterales")
    assert draw(1, "paeruginosa") != draw(2, "paeruginosa")


def test_union_arm_rows_equal_single_species_arms(pa: b.SpeciesInputs, entero: b.SpeciesInputs) -> None:
    """Adding a species changes no row of the other: the union is exactly the per-species arms' rows."""
    held = {"paeruginosa": ["paeruginosa:235"], "enterobacterales": ["klebsiella:235"]}
    union, _ = b.build_arm([entero, pa], DRUGS, holdout_sts=held)
    pa_only, _ = b.build_arm([pa], DRUGS, holdout_sts={"paeruginosa": held["paeruginosa"]})
    en_only, _ = b.build_arm([entero], DRUGS, holdout_sts={"enterobacterales": held["enterobacterales"]})
    for drug in DRUGS:
        expected = pd.concat([en_only[drug], pa_only[drug]], ignore_index=True)  # species sorted: e < p
        assert b.render_table(union[drug]) == b.render_table(expected)
        assert union[drug]["Sample"].is_unique


def test_exact_st_holdout_keyed_on_scheme(pa: b.SpeciesInputs, entero: b.SpeciesInputs) -> None:
    """holdout_st is exactly the held scheme:ST — not ST2350, not untyped, not ST235 of another scheme."""
    tables, report = b.build_arm([pa, entero], DRUGS, holdout_sts={"paeruginosa": ["paeruginosa:235"]})
    splits = _genome_splits(tables)
    cip = tables["ciprofloxacin"]
    held = set(cip.loc[cip["st_key"] == "paeruginosa:235", "Sample"])
    assert len(held) == 40 and all(splits[s] == "holdout_st" for s in held)
    assert {s for s, v in splits.items() if v == "holdout_st"} == held
    assert (cip.loc[cip["st_key"] == "klebsiella:235", "split"] != "holdout_st").all()
    assert report["species"]["paeruginosa"]["holdout_sts"] == {"paeruginosa:235": 40}
    assert report["species"]["paeruginosa"]["genome_splits"]["holdout_st"] == 40


def test_split_fractions_and_disjointness(pa: b.SpeciesInputs) -> None:
    """Random holdout = floor(20 %) of the non-held genomes; validate ≈ a fifth of the rest; disjoint, complete."""
    _, report = b.build_arm([pa], DRUGS, holdout_sts={"paeruginosa": ["paeruginosa:235"]})
    g = report["species"]["paeruginosa"]["genome_splits"]
    pool = 400 - 40
    assert g["holdout"] == int(0.2 * pool) == 72
    rest = pool - g["holdout"]
    assert abs(g["validate"] - rest / 5) <= 1
    assert g["train"] + g["validate"] + g["holdout"] + g["holdout_st"] == 400


def test_fold_moves_only_train_and_validate(pa: b.SpeciesInputs) -> None:
    """Another fold swaps train/validate; holdout and holdout_st never move, and folds' validates are disjoint."""
    kw = {"holdout_sts": {"paeruginosa": ["paeruginosa:235"]}}
    by_fold = [_genome_splits(b.build_arm([pa], DRUGS, fold=f, **kw)[0]) for f in range(5)]
    for split in ("holdout", "holdout_st"):
        sets = [{s for s, v in f.items() if v == split} for f in by_fold]
        assert all(s == sets[0] for s in sets)
    validates = [{s for s, v in f.items() if v == "validate"} for f in by_fold]
    assert sum(len(v) for v in validates) == len(set().union(*validates))  # each genome validates in one fold


def test_groups_share_a_split(tmp_path: Path) -> None:
    """Every group's genomes land in one split (many groups, so some land in each split)."""
    sp = _species(tmp_path, "paeruginosa", n=300, n_held=20)
    eligible = pd.read_csv(sp.eligible)["Sample"].tolist()
    groups = {s: f"g{i // 3}" for i, s in enumerate(eligible[20:290])}  # 90 groups of 3, none in the held ST
    tables, report = b.build_arm([sp], DRUGS, holdout_sts={"paeruginosa": ["paeruginosa:235"]}, groups=groups)
    splits = _genome_splits(tables)
    by_group: dict[str, set[str]] = {}
    for s, g in groups.items():
        if s in splits:
            by_group.setdefault(g, set()).add(splits[s])
    assert all(len(v) == 1 for v in by_group.values())
    assert {next(iter(v)) for v in by_group.values()} >= {"train", "validate", "holdout"}
    assert len(report["species"]["paeruginosa"]["groups"]) == 90


def test_exclusions_are_not_eligible(pa: b.SpeciesInputs) -> None:
    """Excluded genomes appear in no table; the report counts them; an unknown exclusion is only reported."""
    excl = {"P0050": "long-read", "P0051": "long-read", "NOT_A_GENOME": "typo"}
    tables, report = b.build_arm([pa], DRUGS, holdout_sts={"paeruginosa": ["paeruginosa:235"]}, exclusions=excl)
    for t in tables.values():
        assert not set(t["Sample"]) & set(excl)
    rep = report["species"]["paeruginosa"]
    assert rep["n_excluded"] == 2 and rep["n_eligible"] == 398
    assert report["excluded_not_eligible"] == ["NOT_A_GENOME"]


def test_only_eligible_labelled_genomes_appear(pa: b.SpeciesInputs) -> None:
    """A drug's table = eligible ∩ labelled for that drug; the not-eligible genomes never appear."""
    tables, report = b.build_arm([pa], DRUGS, holdout_sts={"paeruginosa": ["paeruginosa:235"]})
    labels = pd.read_csv(pa.labels).set_index("Sample")
    eligible = set(pd.read_csv(pa.eligible)["Sample"])
    for drug in DRUGS:
        assert set(tables[drug]["Sample"]) == {s for s in eligible if pd.notna(labels.at[s, drug])}
    assert report["counts"]["ciprofloxacin"]["paeruginosa"]["n_labelled_not_eligible"] == 5
    assert "0.5" in set(tables["ciprofloxacin"]["ast_label"])  # kept, as text, never scored


def test_assert_same_as_passes_and_refuses(tmp_path: Path, pa: b.SpeciesInputs, entero: b.SpeciesInputs) -> None:
    """--assert-same-as accepts the PsA-only arm's tables and refuses a PsA arm built differently."""
    held = {"paeruginosa": ["paeruginosa:235"]}
    pa_only, _ = b.build_arm([pa], DRUGS, holdout_sts=held)
    other = tmp_path / "pa_only"
    b.write_arm(other, pa_only, {})
    union, _ = b.build_arm([pa, entero], DRUGS, holdout_sts=held)
    b.assert_same_as(union, "paeruginosa", other)

    reseeded, _ = b.build_arm([pa, entero], DRUGS, holdout_sts=held, seed=2)
    with pytest.raises(ValueError, match="rows differ"):
        b.assert_same_as(reseeded, "paeruginosa", other)
    # the union run forgot nothing but gained an exclusion: PsA's assignment moves, and the check catches it
    excluded, _ = b.build_arm([pa, entero], DRUGS, holdout_sts=held, exclusions={"P0100": "x"})
    with pytest.raises(ValueError, match="rows differ"):
        b.assert_same_as(excluded, "paeruginosa", other)
    # a species' rows never depend on the other species: Entero's exclusion leaves PsA's rows equal
    b.assert_same_as(b.build_arm([pa, entero], DRUGS, holdout_sts=held, exclusions={"E0100": "x"})[0],
                     "paeruginosa", other)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"holdout_sts": {"paeruginosa": ["paeruginosa:9999"]}}, "no eligible genomes"),
        ({"holdout_sts": {"paeruginosa": ["paeruginosa:-"]}}, "untyped"),
        ({"holdout_sts": {"nope": ["paeruginosa:235"]}}, "not given with --species"),
        ({"holdout_sts": {"paeruginosa": ["paeruginosa:235"]}, "groups": {"P0000": "g", "P0100": "g"}},
         "spans a held-out ST and not"),
        ({"fold": 5}, "fold"),
        ({"holdout_fraction": 0.0}, "holdout_fraction"),
    ],
)
def test_build_arm_fails_loudly(pa: b.SpeciesInputs, kwargs: dict, match: str) -> None:
    """Each contract violation raises rather than writing a quietly different split."""
    with pytest.raises(ValueError, match=match):
        b.build_arm([pa], DRUGS, **kwargs)


def test_fails_on_missing_label_column(pa: b.SpeciesInputs) -> None:
    with pytest.raises(ValueError, match="missing drug column"):
        b.build_arm([pa], [*DRUGS, "colistin"])


def test_fails_on_duplicated_sample(pa: b.SpeciesInputs) -> None:
    """A Sample listed twice in the eligible list, the label sheet or the QC table is refused."""
    for path in (pa.eligible, pa.labels, pa.qc):
        original = path.read_text()
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        pd.concat([df, df.iloc[[10]]]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="duplicated"):
            b.build_arm([pa], DRUGS)
        path.write_text(original)


def test_fails_on_sample_in_two_species_or_group_across_species(tmp_path: Path) -> None:
    a = _species(tmp_path, "a", prefix="X")
    c = _species(tmp_path, "c", prefix="X", seed=3)
    with pytest.raises(ValueError, match="eligible in two species"):
        b.build_arm([a, c], DRUGS)
    c = _species(tmp_path, "c", prefix="Y", seed=3)
    with pytest.raises(ValueError, match="span species"):
        b.build_arm([a, c], DRUGS, groups={"X0100": "g", "Y0100": "g"})


def test_fails_on_eligible_genome_without_qc_row(pa: b.SpeciesInputs) -> None:
    qc = pd.read_csv(pa.qc, dtype=str)
    qc[qc["Sample"] != "P0007"].to_csv(pa.qc, index=False)
    with pytest.raises(ValueError, match="no QC row"):
        b.build_arm([pa], DRUGS)


def test_fails_on_label_outside_unit_interval(pa: b.SpeciesInputs) -> None:
    labels = pd.read_csv(pa.labels)
    labels.loc[5, "meropenem"] = 8.0  # an MIC, not a label
    labels.to_csv(pa.labels, index=False)
    with pytest.raises(ValueError, match="outside 0"):
        b.build_arm([pa], DRUGS)


def test_cli_writes_tables_that_load_splits_reads(tmp_path: Path, pa: b.SpeciesInputs) -> None:
    """End to end: the CLI writes tables + manifest; load_splits / load_split_ids read every split back."""
    out = tmp_path / "splits"
    excl = tmp_path / "excluded.tsv"
    excl.write_text("Sample\treason\nP0050\tlong-read\n")
    groups = tmp_path / "groups.csv"
    groups.write_text("Sample,group\nP0100,iso1\nP0101,iso1\n")
    argv = ["--species", f"paeruginosa={pa.labels},{pa.eligible},{pa.qc}",
            "--holdout-st", "paeruginosa=paeruginosa:235", "--exclude", str(excl), "--groups", str(groups),
            "--drugs", ",".join(DRUGS), "--out-dir", str(out)]
    b.main(argv)

    manifest = json.loads((out / b.MANIFEST_NAME).read_text())
    assert manifest["arguments"]["holdout_sts"] == {"paeruginosa": ["paeruginosa:235"]}
    assert {i["role"] for i in manifest["inputs"]} == {"labels", "eligible", "qc", "exclude", "groups"}
    assert all(len(i["md5"]) == 32 for i in manifest["inputs"])
    assert set(manifest["tables"]) == set(DRUGS) and manifest["git_sha"]

    for drug in DRUGS:
        path = out / f"{drug}_split.csv"
        table = pd.read_csv(path, dtype=str)
        label_map, train, validate, holdout = load_splits(path)
        held = load_split_ids(path, "holdout_st")
        parts = [set(train), set(validate), set(holdout), set(held)]
        assert sum(map(len, parts)) == len(set().union(*parts)) == len(label_map)  # disjoint, all scored rows
        assert "P0050" not in table["Sample"].tolist()
        assert set(held) <= set(table.loc[table["st_key"] == "paeruginosa:235", "Sample"])
        counts = manifest["counts"][drug]["paeruginosa"]
        assert counts["holdout_st"]["n_scoreable"] == len(held) and counts["train"]["n_scoreable"] == len(train)

    b.main(argv)  # identical rerun: allowed
    with pytest.raises(FileExistsError, match="differ"):
        b.main([*argv, "--seed", "2"])
    b.main([*argv, "--seed", "2", "--overwrite"])
    (out / "colistin_split.csv").write_text("Sample,species,st_key,ast_label,split\n")
    with pytest.raises(FileExistsError, match="not in this run"):
        b.main(argv)


def test_committed_pa_inputs_parse() -> None:
    """apps/pa's exclusion list (15 Ecuador long-read genomes) and duplicate-isolate groups (3 × 2) are valid."""
    excl = b.read_sample_map([REPO / "src/bacpredict/apps/pa/excluded_genomes.tsv"], "reason", sep="\t",
                             what="exclusions")
    assert len(excl) == 15 and all(s.startswith("SAMN338421") for s in excl)
    groups = b.read_sample_map([REPO / "src/bacpredict/apps/pa/duplicate_isolate_groups.csv"], "group", sep=",",
                               what="groups")
    assert sorted(pd.Series(groups).value_counts().to_dict().items()) == [("S54485", 2), ("U2504", 2), ("X24509", 2)]


def test_st_key_formats_untyped_and_missing() -> None:
    """``<scheme>:<st>``; untyped ``<scheme>:-``; no scheme ``-:-``; the vectorised form agrees."""
    cases = [("paeruginosa", "235", "paeruginosa:235"), ("paeruginosa", "-", "paeruginosa:-"),
             ("-", "-", "-:-"), ("", "", "-:-"), ("klebsiella", " 11 ", "klebsiella:11")]
    assert [b.st_key(s, t) for s, t, _ in cases] == [k for *_, k in cases]
    assert b.st_key(None, float("nan")) == "-:-"
    scheme, st = pd.Series([s for s, _, _ in cases]), pd.Series([t for _, t, _ in cases])
    assert b._st_keys(scheme, st) == [k for *_, k in cases]
