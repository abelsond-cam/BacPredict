"""A second cohort over the shared genome store: plan only what no earlier cohort builds, and collate over both."""

from __future__ import annotations

import gzip
import json
import os
import stat
from pathlib import Path

import pandas as pd
import pytest

from bacpredict.engine.genomes.reuse_built_genomes import (
    AUTO,
    REUSED_WHY,
    array_list,
    bakrep_export_ids,
    bakta_annotated,
    bakta_plan,
    bakta_roster,
    bakta_status,
    collate_inputs,
    main,
    missing_tasks,
    next_qc_list,
    overlap,
    partition,
    read_built,
    read_shard_dirs,
    replan_diff,
    scheme_mismatches,
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


def test_a_species_pin_wins_over_its_genus(cohort):
    c = pd.concat([cohort, pd.DataFrame({"Sample": ["KA"], "species": ["Klebsiella aerogenes"], "n_drugs_labelled": [2]})])
    m = pd.DataFrame({"Sample": ["KA", "N1"], "source": ["atb", "atb"]})
    by = split_by_scheme(m, c.set_index("Sample")["species"], {"Klebsiella": "klebsiella", "Escherichia": "ecoli_achtman_4"},
                         {"Klebsiella aerogenes": "kaerogenes"})
    assert {k: list(v["Sample"]) for k, v in by.items()} == {"ecoli_achtman_4": ["N1"], "kaerogenes": ["KA"]}


def test_scheme_mismatches_judges_built_genomes_by_pin_and_reused_by_their_owner():
    qc = pd.DataFrame({
        "Sample": ["B1", "B2", "R1", "R2", "R3", "U1", "U2", "F1"],
        "species": ["Klebsiella aerogenes", "Escherichia coli", "Enterobacter hormaechei", "Escherichia coli",
                    "Escherichia coli", "Kluyvera ascorbata", "Kluyvera ascorbata", "Escherichia coli"],
        "mlst_scheme": ["klebsiella", "ecoli_achtman_4", "kaerogenes", "ecoli_achtman_4", "ecoli_achtman_4",
                        "ecloacae", None, "salmonella"],
        "st": ["-", "10", "4", "131", "73", "5", "-", "1"],
        "status": ["ok"] * 8, "qc_pass": [True] * 7 + [False],
    })
    owner = pd.DataFrame({"Sample": ["R1", "R2", "R3"], "mlst_scheme": ["kaerogenes", "ecoli_achtman_4", "ecoli_achtman_4"],
                          "st": ["4", "131", "1193"]})
    t = scheme_mismatches(qc, {"R1", "R2", "R3"}, owner, {"Klebsiella": "klebsiella", "Escherichia": "ecoli_achtman_4",
                                                         "Enterobacter": "ecloacae"},
                          {"Klebsiella aerogenes": "kaerogenes"})
    assert t["built_off_pin"][["species", "mlst_scheme", "n"]].values.tolist() == [["Klebsiella aerogenes", "klebsiella", 1]]
    assert t["reused_changed"]["Sample"].tolist() == ["R3"]          # the owner records ST1193: an mtime-lost re-type
    assert t["reused_relabelled"][["species", "mlst_scheme"]].values.tolist() == [["Enterobacter hormaechei", "kaerogenes"]]
    assert sorted(t["unpinned_several"]["mlst_scheme"]) == ["-", "ecloacae"]  # QC-fail F1 is never counted


def test_a_reused_genome_missing_from_the_owners_table_fails():
    qc = pd.DataFrame({"Sample": ["R1"], "species": ["Escherichia coli"], "mlst_scheme": ["ecoli_achtman_4"],
                       "st": ["10"], "status": ["ok"], "qc_pass": [True]})
    owner = pd.DataFrame({"Sample": ["X"], "mlst_scheme": ["ecoli_achtman_4"], "st": ["10"]})
    assert scheme_mismatches(qc, {"R1"}, owner, {"Escherichia": "ecoli_achtman_4"})["reused_changed"]["Sample"].tolist() == ["R1"]


def test_array_list_and_missing_tasks(tmp_path):
    assert array_list([7, 0, 1, 2, 4, 8]) == "0-2,4,7-8" and array_list([]) == ""
    m = pd.DataFrame({"Sample": [f"S{i:03d}" for i in range(250)], "source": ["atb"] * 230 + ["shovill"] * 20})
    d = tmp_path / "shards"
    d.mkdir()

    def shard(name, samples):
        (d / name).write_text("Sample\tstatus\n" + "".join(f"{s}\tok\n" for s in samples))

    shard("atb_00000.tsv", [f"S{i:03d}" for i in range(100)])
    shard("atb_00002.tsv", [f"S{i:03d}" for i in range(200, 230)])
    shard("shovill_00003.tsv", ["S233"])
    assert missing_tasks(m, "atb", 100, d) == [1]
    assert array_list(missing_tasks(m, "shovill", 1, d)) == "0-2,4-19"
    shard("atb_00000.tsv", [f"S{i:03d}" for i in range(1, 101)])  # written under a plan whose rows have since shifted
    assert missing_tasks(m, "atb", 100, d) == [0, 1]


def test_collate_inputs_refuses_a_stale_skani_and_a_different_panel(cohort):
    args = _collate_args(cohort)
    reuse_skani = args[6][args[6]["Query_file"].ne("/l/N1.fna.gz")]
    args[6] = args[6][args[6]["Query_file"].eq("/l/N1.fna.gz")]
    with pytest.raises(ValueError, match="rerun qc skani"):
        collate_inputs(*args, reuse_skani=reuse_skani, refs=["/x/K12.fna.gz"], skani_list={"N1"}, listed={"N1", "N9"})
    with pytest.raises(ValueError, match="only here"):
        collate_inputs(*args, reuse_skani=reuse_skani, refs=["/x/K12.fna.gz", "/x/B2.fna.gz"], skani_list={"N1"},
                       listed={"N1"})
    m, sh, q, s = collate_inputs(*args, reuse_skani=reuse_skani, refs=["/elsewhere/K12.fna.gz"], skani_list={"N1"},
                                 listed={"N1"})
    assert sorted(s["Query_file"]) == ["/l/E2.fna.gz", "/l/N1.fna.gz"]


# ---- Bakta ------------------------------------------------------------------------------------------------------------

FP = "fp-of-record"


def test_bakrep_export_ids_reads_the_id_column(tmp_path):
    f = tmp_path / "bakrep-export.tsv.gz"
    with gzip.open(f, "wt") as fh:  # a stray quote in a later column must not swallow lines
        fh.write('#id\tbakta.genome.genus\nSAMN1\tE"scherichia\nSAMN2\t\nSAMN3\tSalmonella\n')
    assert bakrep_export_ids(f) == {"SAMN1", "SAMN2", "SAMN3"}
    bad = tmp_path / "not_export.tsv"
    bad.write_text("Sample\tx\nS\t1\n")
    with pytest.raises(ValueError, match="#id"):
        bakrep_export_ids(bad)


def test_bakta_plan_takes_ok_builds_and_rosters_only_the_atb_ones_bakrep_lists():
    samples = ["A1", "A2", "N1", "S1", "F1", "X1"]
    manifest = pd.DataFrame({"Sample": samples, "source": ["atb", "atb", "ncbi_gca", "shovill", "atb", "shovill"]})
    shards = pd.DataFrame({"Sample": samples, "source": ["atb", "atb", "ncbi_gca", "shovill", "atb", "atb"],
                           "status": ["ok", "ok", "ok", "ok", "fail:x", "ok"],
                           "contigs_path": [f"/c/{s}.fna.gz" for s in samples]})
    built, roster = bakta_plan(manifest, shards, pd.DataFrame({"Sample": ["R1"]}), {"A1", "N1", "R1"})
    assert list(built["Sample"]) == ["A1", "A2", "N1", "S1"]  # F1 failed; X1's row is from a plan since changed
    assert list(built.columns) == ["Sample", "sr_assembly_file", "source"]
    # N1 is in BakRep but built from NCBI: BakRep's GFF3 sits on ATB's contigs, not on these
    assert roster.to_dict("list") == {"Sample": ["A1"], "sr_assembly_file": ["/c/A1.fna.gz"]}
    with pytest.raises(ValueError, match="both built by this cohort and reused"):
        bakta_plan(manifest, shards, pd.DataFrame({"Sample": ["A2"]}), set())
    with pytest.raises(ValueError, match="absolute"):
        bakta_plan(manifest, shards.assign(contigs_path="c/x.fna.gz"), pd.DataFrame({"Sample": []}), set())


@pytest.fixture
def annotation(tmp_path):
    """A cohort over the shared store: genomes built here (A*, N*) and reused (R*), with the records each has."""
    ann, contigs = tmp_path / "bakta", tmp_path / "contigs"
    ann.mkdir()
    contigs.mkdir()
    cohort = ["A1", "A2", "A3", "N1", "N2", "R1", "R2", "R3"]
    for s in cohort:
        (contigs / f"{s}.fna.gz").write_bytes(b">c1\nACGT\n")
        os.utime(contigs / f"{s}.fna.gz", (1_000_000, 1_000_000))

    def rec(sample, kind, **body):
        (ann / f"{sample}.bakta.{kind}.json").write_text(json.dumps(body))
        (ann / f"{sample}.bakta.gff3.gz").write_bytes(b"gff")

    rec("A1", "ingest")
    rec("A3", "run", fingerprint=FP)  # ours, finished after its contigs: done
    rec("N2", "run", fingerprint="another env")  # ours, under another env: not done
    rec("R1", "ingest")  # the owner's
    rec("R3", "run", fingerprint=FP)
    built = pd.DataFrame({"Sample": ["A1", "A2", "A3", "N1", "N2"],
                          "source": ["atb", "atb", "atb", "ncbi_gca", "shovill"]})
    return {
        "cohort": pd.DataFrame({"Sample": cohort, "sr_assembly_file": [str(contigs / f"{s}.fna.gz") for s in cohort]}),
        "built": built,
        "reused": pd.DataFrame({"Sample": ["R1", "R2", "R3"]}),
        "bakrep_roster": pd.DataFrame({"Sample": ["A1", "A2", "A3"]}),
        "status": pd.DataFrame({"Sample": ["A1", "A2", "A3"], "status": ["ingested", "refused:names", "absent"]}),
        "annotation_dir": ann,
        "fingerprint": FP,
    }


def _roster(a, mode, **kw):
    want = bakta_roster(**{**a, **kw}, mode=mode).set_index("Sample")
    return want, sorted(want.index[~want["done"]])


def test_bakta_roster_modes(annotation):
    want, todo = _roster(annotation, "planned")
    assert sorted(want.index) == ["N1", "N2", "R2"] and todo == ["N1", "N2", "R2"]
    assert want.loc["R2", "why"] == REUSED_WHY and want.loc["N1", "why"] == "not in BakRep: ncbi_gca"
    want, todo = _roster(annotation, "refused")
    assert sorted(want.index) == ["A2", "A3"] and todo == ["A2"]  # A3: our run of this env already stands
    assert want.loc["A2", "why"] == "BakRep refused:names"
    want, todo = _roster(annotation, "all")
    assert todo == ["A2", "N1", "N2", "R2"]  # never A1 (ingested), R1 or R3 (their owner's records stand)


def test_bakta_roster_reruns_ours_when_the_contigs_are_newer(annotation):
    os.utime(annotation["cohort"].set_index("Sample").loc["A3", "sr_assembly_file"], None)  # rebuilt just now
    assert _roster(annotation, "refused")[1] == ["A2", "A3"]


def test_bakta_roster_refuses_what_it_cannot_place(annotation):
    a = annotation
    with pytest.raises(ValueError, match="no ingest status: run ingest"):
        bakta_roster(**{**a, "status": None}, mode="all")
    assert _roster(a, "planned", status=None)[1] == ["N1", "N2", "R2"]  # planned needs none
    with pytest.raises(ValueError, match="1 BakRep-roster cohort genomes have no ingest status"):
        bakta_roster(**{**a, "status": a["status"].iloc[:2]}, mode="refused")
    err = a["status"].assign(status=["ingested", "error:transient", "absent"])
    with pytest.raises(ValueError, match="error:transient"):
        bakta_roster(**{**a, "status": err}, mode="all")
    assert _roster(a, "refused", status=err, allow_errors=True)[1] == ["A2"]
    with pytest.raises(ValueError, match="neither reused nor in built_genomes"):
        bakta_roster(**{**a, "built": a["built"].iloc[1:]}, mode="planned")
    with pytest.raises(ValueError, match="both reused and built here"):
        bakta_roster(**{**a, "reused": pd.DataFrame({"Sample": ["R1", "N1"]})}, mode="planned")
    with pytest.raises(ValueError, match="twice"):
        bakta_roster(**{**a, "cohort": pd.concat([a["cohort"], a["cohort"].head(1)])}, mode="planned")


def test_bakta_roster_cli_writes_a_read_only_roster_and_refuses_past_max_run(tmp_path, annotation):
    a = annotation
    files = {k: _csv(tmp_path / f"in/{k}.csv", a[k]) for k in ("cohort", "built", "reused")}
    roster = tmp_path / "in/bakrep_roster.tsv"
    a["bakrep_roster"].assign(c="/x").to_csv(roster, sep="\t", header=False, index=False)
    status = tmp_path / "in/ingest_status.tsv"
    a["status"].to_csv(status, sep="\t", index=False)
    base = ["bakta-roster", "--cohort", str(files["cohort"]), "--built", str(files["built"]), "--reused",
            str(files["reused"]), "--bakrep-roster", str(roster), "--status", str(status), "--annotation-dir",
            str(a["annotation_dir"]), "--fingerprint", FP, "--mode", "all"]
    out = tmp_path / "run_roster_1.tsv"
    with pytest.raises(SystemExit, match="more than the 3 allowed"):
        main([*base, "--max-run", "3", "--out", str(out)])
    assert not out.exists()
    main([*base, "--max-run", "4", "--out", str(out)])
    assert [r.split("\t")[0] for r in out.read_text().splitlines()] == ["A2", "N1", "N2", "R2"]
    assert not out.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    empty = tmp_path / "run_roster_2.tsv"
    main([*base[:-1], "refused", "--status", str(status), "--max-run", "9", "--out", str(empty),
          "--cohort", str(_csv(tmp_path / "in/c2.csv", a["cohort"][a["cohort"]["Sample"].isin(["A1", "A3"])]))])
    assert not empty.exists()  # nothing to run: no roster


def test_bakta_plan_cli_and_status(tmp_path):
    gen = tmp_path / "genomes"
    m = _csv(gen / "genome_manifest.csv", pd.DataFrame({"Sample": ["A", "B", "C"], "source": ["atb", "atb", "drop"]}))
    _shards(gen / "shards/klebsiella", [["A", "atb", "ok", "/c/A"], ["B", "atb", "ok", "/c/B"]])
    reused = _csv(gen / "reused_genomes.csv", pd.DataFrame({"Sample": ["R"]}))
    export = tmp_path / "bakrep-export.tsv.gz"
    with gzip.open(export, "wt") as fh:
        fh.write("#id\tx\nA\t1\nR\t1\n")
    main(["bakta-plan", "--manifest", str(m), "--shard-root", str(gen / "shards"), "--reused", str(reused),
          "--bakrep-export", str(export), "--out-dir", str(gen / "bakta")])
    assert (gen / "bakta/bakrep_roster.tsv").read_text() == "A\t/c/A\n"
    assert list(pd.read_csv(gen / "bakta/built_genomes.csv")["Sample"]) == ["A", "B"]
    cohort = _csv(gen / "embedding_input.csv", pd.DataFrame({"Sample": ["A", "B", "R"], "sr_assembly_file": "/c"}))
    (gen / "bakta/ingest_status.tsv.part").write_text("Sample\tstatus\tdetail\nA\tingested\tfetched\n")
    (tmp_path / "ann").mkdir()
    (tmp_path / "ann/R.bakta.ingest.json").write_text("{}")
    text = bakta_status(gen / "bakta", tmp_path / "ann", cohort, {"R"})
    assert "NOT final" in text and "BakRep roster: 1 genomes, 0 without an ingest status" in text
    assert "built here: 2 — ingest record 0, our run record 0, both 0, neither 2" in text
    assert "reused: 1 — ingest record 1" in text


def test_replan_diff_names_every_moved_genome_and_refuses_an_unexpected_move(tmp_path):
    old = pd.DataFrame({"Sample": ["A", "B", "C", "D"], "source": ["atb", "drop", "ncbi_gca", "shovill"],
                        "drop_reason": [None, "other_strategy", None, None], "runs": [None, None, None, "R1"]})
    new = old.assign(source=["atb", "shovill", "shovill", "shovill"], drop_reason=None,
                     runs=[None, "R2", "R3", "R1"], reads_platform=[None, "DNBSEQ", "DNBSEQ", "ILLUMINA"])
    changed, bad = replan_diff(old, new, {("drop", "shovill"), ("ncbi_gca", "shovill")})
    assert changed.to_dict("index") == {"B": {"old": "drop", "new": "shovill"}, "C": {"old": "ncbi_gca", "new": "shovill"}}
    assert bad.empty  # D: only a column the old manifest lacks is new
    _, bad = replan_diff(old, new.assign(runs=[None, "R2", "R3", "R9"]), {("drop", "shovill"), ("ncbi_gca", "shovill")})
    assert list(bad.index) == ["D"]  # a genome whose reads changed is not an allowed move
    with pytest.raises(ValueError, match="other genomes"):
        replan_diff(old, new.iloc[:3], set())
    files = {k: _csv(tmp_path / f"{k}.csv", df) for k, df in
             (("old", old), ("new", new), ("r1", pd.DataFrame({"Sample": ["X"]})), ("r2", pd.DataFrame({"Sample": ["Y"]})))}
    base = ["replan-diff", "--old-manifest", str(files["old"]), "--new-manifest", str(files["new"]),
            "--old-reused", str(files["r1"])]
    main([*base, "--new-reused", str(files["r1"]), "--allow", "drop:shovill,ncbi_gca:shovill"])
    with pytest.raises(SystemExit, match="not among"):
        main([*base, "--new-reused", str(files["r1"]), "--allow", "drop:shovill"])
    with pytest.raises(SystemExit, match="reused_genomes.csv changed"):
        main([*base, "--new-reused", str(files["r2"]), "--allow", "drop:shovill,ncbi_gca:shovill"])


def test_bakta_roster_can_leave_reused_genomes_for_later(annotation):
    want, todo = _roster(annotation, "all", skip_reused=True)
    assert todo == ["A2", "N1", "N2"]  # R2 waits: an owner's job may be writing it


def test_bakta_annotated_takes_only_finished_single_current_records(tmp_path, annotation):
    a = annotation
    ann = a["annotation_dir"]
    paths = a["cohort"].set_index("Sample")["sr_assembly_file"]
    for s, kind in (("B1", "ingest"), ("B2", "run"), ("B2", "ingest"), ("B3", "ingest")):
        Path(paths["A1"]).with_name(f"{s}.fna.gz").write_bytes(b">c1\nACGT\n")
        os.utime(Path(paths["A1"]).with_name(f"{s}.fna.gz"), (1_000_000, 1_000_000))
        (ann / f"{s}.bakta.{kind}.json").write_text(json.dumps({"fingerprint": FP}))
        (ann / f"{s}.bakta.gff3.gz").write_bytes(b"gff")
    (ann / "B3.bakta.running").write_text("")  # our run of B3 has started: not finished, whatever else is there
    built = pd.DataFrame({"Sample": ["A1", "A2", "A3", "N1", "N2", "B1", "B2", "B3"]})
    built["sr_assembly_file"] = [paths.get(s, str(Path(paths["A1"]).with_name(f"{s}.fna.gz"))) for s in built["Sample"]]
    os.utime(built.set_index("Sample").loc["B1", "sr_assembly_file"], None)  # rebuilt after its ingest
    t = bakta_annotated(built, ann, FP, workers=2).set_index("Sample")["state"].to_dict()
    assert t == {"A1": "ingest", "A2": "none", "A3": "run", "N1": "none", "N2": "stale", "B1": "stale",
                 "B2": "both", "B3": "running"}
    f = _csv(tmp_path / "built_genomes.csv", built.assign(source="atb"))
    out = tmp_path / "genomes/preqc_annotated.csv"
    main(["bakta-annotated", "--built", str(f), "--annotation-dir", str(ann), "--fingerprint", FP, "--out", str(out)])
    got = pd.read_csv(out)
    assert list(got.columns) == ["Sample", "sr_assembly_file", "origin"]
    assert got[["Sample", "origin"]].values.tolist() == [["A1", "ingest"], ["A3", "run"]]
    with pytest.raises(SystemExit, match="nothing written"):
        main(["bakta-annotated", "--built", str(_csv(tmp_path / "b2.csv", built.iloc[[1, 3]])), "--annotation-dir",
              str(ann), "--fingerprint", FP, "--out", str(tmp_path / "none.csv")])
