"""A second cohort over the shared genome store: plan only what no earlier cohort builds, and collate over both."""

from __future__ import annotations

import os
import stat

import pandas as pd
import pytest

from bacpredict.engine.genomes.reuse_built_genomes import (
    AUTO,
    collate_inputs,
    main,
    next_qc_list,
    overlap,
    partition,
    read_built,
    read_shard_dirs,
    split_by_scheme,
)

SHARD = ["Sample", "source", "status", "contigs_path"]


def _csv(path, df, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kw)
    return path


@pytest.fixture
def built(tmp_path):
    pa = _csv(tmp_path / "pa/genome_manifest.csv", pd.DataFrame(
        {"Sample": ["P1", "P2"], "source": ["atb", "drop"], "drop_reason": [None, "rna_seq_only"]}))
    en = _csv(tmp_path / "en/genome_manifest.csv", pd.DataFrame(
        {"Sample": ["E1", "E2"], "source": ["atb", "shovill"], "drop_reason": [None, None]}))
    return [pa, en]


@pytest.fixture
def cohort():
    return pd.DataFrame({
        "Sample": ["P1", "P2", "E2", "N1", "N2", "N3", "U1"],
        "species": ["Pseudomonas aeruginosa"] * 2 + ["Escherichia coli", "Escherichia coli", "Salmonella enterica",
                                                     "Kluyvera ascorbata", "Escherichia coli"],
        "n_drugs_labelled": [3, 2, 1, 4, 2, 1, 0],
    })


def test_partition_reuses_every_owned_genome_and_its_decision(built, cohort):
    to_plan, reused = partition(cohort, read_built(built))
    assert list(to_plan["Sample"]) == ["N1", "N2", "N3", "U1"]  # unlabelled rows pass through; the planner drops them
    assert list(reused["Sample"]) == ["E2", "P1", "P2"]
    r = reused.set_index("Sample")
    assert r.loc["P2", "source"] == "drop" and r.loc["P2", "drop_reason"] == "rna_seq_only"  # dropped stays dropped
    assert r.loc["E2", "owner"].endswith("en/genome_manifest.csv")


def test_a_genome_in_two_earlier_manifests_is_an_error(tmp_path, built):
    again = _csv(tmp_path / "x/genome_manifest.csv", pd.DataFrame({"Sample": ["E1"], "source": ["atb"]}))
    with pytest.raises(ValueError, match="more than one --built manifest"):
        read_built([*built, again])


def test_overlap_counts_drop_rows_too(built):
    m = pd.DataFrame({"Sample": ["N1", "P2"], "source": ["atb", "drop"]})
    assert overlap(m, read_built(built)) == ["P2"]


def test_check_disjoint_cli(tmp_path, built):
    ok = _csv(tmp_path / "v2/ok.csv", pd.DataFrame({"Sample": ["N1"], "source": ["atb"]}))
    bad = _csv(tmp_path / "v2/bad.csv", pd.DataFrame({"Sample": ["N1", "E1"], "source": ["atb", "atb"]}))
    main(["check-disjoint", "--manifest", str(ok), "--built", *map(str, built)])
    with pytest.raises(SystemExit, match="NOT DISJOINT.*E1"):
        main(["check-disjoint", "--manifest", str(ok), str(bad), "--built", *map(str, built)])


def test_split_by_scheme(cohort):
    m = pd.DataFrame({"Sample": ["N1", "N2", "N3", "U1"], "source": ["atb", "shovill", "atb", "drop"]})
    by = split_by_scheme(m, cohort.set_index("Sample")["species"],
                         {"Escherichia": "ecoli_achtman_4", "Salmonella": "salmonella"})
    assert {k: list(v["Sample"]) for k, v in by.items()} == {
        AUTO: ["N3"], "ecoli_achtman_4": ["N1"], "salmonella": ["N2"]}  # drop rows are never built
    with pytest.raises(ValueError, match="no species"):
        split_by_scheme(pd.DataFrame({"Sample": ["ZZ"], "source": ["atb"]}), cohort.set_index("Sample")["species"], {})


def test_split_by_scheme_cli_writes_read_only_and_never_overwrites(tmp_path, cohort):
    c = _csv(tmp_path / "cohort.csv", cohort)
    m = _csv(tmp_path / "genome_manifest.csv", pd.DataFrame({"Sample": ["N1", "N2"], "source": ["atb", "atb"]}))
    argv = ["split-by-scheme", "--manifest", str(m), "--cohort", str(c),
            "--scheme-by-genus", "Escherichia=ecoli_achtman_4,Salmonella=salmonella", "--out-dir", str(tmp_path / "bys")]
    main(argv)
    f = tmp_path / "bys/genome_manifest_ecoli_achtman_4.csv"
    assert not (os.stat(f).st_mode & stat.S_IWUSR)
    with pytest.raises(SystemExit, match="frozen"):
        main(argv)


def _shards(d, rows):
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=SHARD).to_csv(d / "atb_00000.tsv", sep="\t", index=False)
    return d


def test_qc_list_takes_built_genomes_not_yet_listed(tmp_path):
    m = pd.DataFrame({"Sample": ["A", "B", "C", "D"], "source": ["atb", "atb", "shovill", "drop"]})
    sh = read_shard_dirs([_shards(tmp_path / "s/x", [["A", "atb", "ok", "/c/A"], ["B", "atb", "fail:x", "/c/B"]])])
    with pytest.raises(ValueError, match="1 of 3 buildable genomes have no shard row"):
        next_qc_list(m, sh, set(), require_complete=True)
    assert list(next_qc_list(m, sh, set(), require_complete=False)["Sample"]) == ["A"]
    assert next_qc_list(m, sh, {"A"}, require_complete=False).empty


def test_a_genome_in_two_shard_dirs_is_an_error(tmp_path):
    a = _shards(tmp_path / "s/a", [["A", "atb", "ok", "/c/A"]])
    b = _shards(tmp_path / "s/b", [["A", "atb", "ok", "/c/A"]])
    with pytest.raises(ValueError, match="more than one"):
        read_shard_dirs([a, b])


def _collate_args(cohort):
    manifest = pd.DataFrame({"Sample": ["N1", "N2"], "source": ["atb", "shovill"], "n_drugs_labelled": ["4", "2"]})
    shards = pd.DataFrame([["N1", "atb", "ok", "/c/N1"], ["N2", "shovill", "fail:x", "/c/N2"]], columns=SHARD)
    reused = pd.DataFrame({"Sample": ["E2", "P2"], "source": ["shovill", "drop"], "drop_reason": [None, "rna_seq_only"],
                           "owner": ["en", "pa"]})
    reuse_shards = pd.DataFrame([["E2", "shovill", "ok", "/c/E2"], ["E1", "atb", "ok", "/c/E1"]], columns=SHARD)
    checkm2 = pd.DataFrame({"Name": ["N1", "E2.fna", "E1"], "Completeness": [99.0, 98.0, 97.0],
                            "Contamination": [0.1, 0.2, 0.3]})
    skani = pd.DataFrame({"Ref_file": ["/r/K12.fna.gz"] * 3, "Query_file": ["/l/N1.fna.gz", "/l/E2.fna.gz", "/l/E1.fna.gz"],
                          "ANI": [99.0, 98.5, 99.1]})
    return [cohort, manifest, shards, reused, reuse_shards, checkm2, skani]


def test_collate_inputs_joins_this_cohorts_builds_with_the_reused(cohort):
    m, sh, q, s = collate_inputs(*_collate_args(cohort))
    m = m.set_index("Sample")
    assert list(m.index) == ["N1", "N2", "E2", "P2"]
    assert m.loc["E2", "n_drugs_labelled"] == 1  # this cohort's label count, not the owner's
    assert m.loc["E2", "built_by"] == "en" and m.loc["N1", "built_by"] == "this cohort"
    assert sorted(sh["Sample"]) == ["E2", "N1", "N2"]  # E1 is the owner's genome, not this cohort's
    assert sorted(q["Name"]) == ["E2.fna", "N1"] and list(q.columns) == ["Name", "Completeness", "Contamination"]
    assert sorted(s["Query_file"]) == ["/l/E2.fna.gz", "/l/N1.fna.gz"]


def test_collate_inputs_refuses_a_reused_genome_with_no_owner_row(cohort):
    args = _collate_args(cohort)
    args[4] = args[4][args[4]["Sample"].ne("E2")]
    with pytest.raises(ValueError, match="no shard row in the owners"):
        collate_inputs(*args)


def test_collate_inputs_refuses_a_reused_genome_the_owner_rebuilt_from_another_source(cohort):
    args = _collate_args(cohort)
    args[4] = args[4].assign(source=["atb", "atb"])
    with pytest.raises(ValueError, match="source other than"):
        collate_inputs(*args)


def test_qc_list_cli_freezes_then_checks(tmp_path):
    m = _csv(tmp_path / "genome_manifest.csv", pd.DataFrame({"Sample": ["A", "B"], "source": ["atb", "atb"]}))
    _shards(tmp_path / "shards/klebsiella", [["A", "atb", "ok", "/c/A"], ["B", "atb", "ok", "/c/B"]])
    base = ["qc-list", "--manifest", str(m), "--shard-root", str(tmp_path / "shards"), "--require-complete"]
    with pytest.raises(SystemExit, match="2 built genomes are in no CheckM2 list"):
        main(base)
    main([*base, "--out", str(tmp_path / "qc/checkm2_list_01.tsv")])
    assert (tmp_path / "qc/checkm2_list_01.tsv").read_text() == "A\t/c/A\nB\t/c/B\n"
    main([*base, "--listed", str(tmp_path / "qc/checkm2_list_01.tsv")])  # nothing left: exits 0
