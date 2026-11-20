"""Genome QC: the ATB high-quality rules, the skani species call, ST counts per drug, and the multi-species options."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from bacpredict.engine.genomes import collate_genomes
from bacpredict.engine.genomes.collate_genomes import (
    apply_qc,
    read_checkm2,
    read_shards,
    read_skani,
    species_crosstab,
    st_counts_by_drug,
    untyped,
)

from .collate_fixtures import EC, KP, MM, PA, PARA, SF, entero_argv, psa_argv, write_entero_inputs, write_psa_inputs


def test_skani_species_call(tmp_path):
    p = tmp_path / "skani.tsv"
    pd.DataFrame({
        "Ref_file": ["/r/PAO1.fna.gz", "/r/PA7.fna.gz", "/r/PAO1.fna.gz", "/r/PA7.fna.gz", "/r/PAO1.fna.gz"],
        "Query_file": ["/l/S1.fna.gz", "/l/S1.fna.gz", "/l/S2.fna.gz", "/l/S2.fna.gz", "/l/S3.fna.gz"],
        "ANI": [99.1, 93.4, 93.5, 99.2, 88.0],
    }).to_csv(p, sep="\t", index=False)
    s = read_skani(p, {PA: "/x/PAO1.fna.gz", PARA: "/x/PA7.fna.gz"})
    assert s.loc["S1", "species_call"] == PA and s.loc["S2", "species_call"] == PARA
    assert s.loc["S3", "species_call"] == "unknown"


def test_qc_rules():
    g = pd.DataFrame({
        "Sample": ["ok", "contam", "frag", "para", "broken"],
        "status": ["ok", "ok", "ok", "ok", "fail:x"],
        "checkm2_completeness": [99.0, 99.0, 95.0, 99.0, None],
        "checkm2_contamination": [0.5, 7.0, 1.0, 0.4, None],
        "total_length": [6.5e6, 6.6e6, 6.4e6, 6.6e6, None],
        "n_contigs": [80, 90, 2500, 70, None], "n50": [2e5, 2e5, 1500, 3e5, None],
        "species_call": [PA, PA, PA, PARA, None],
    })
    q = apply_qc(g, PA).set_index("Sample")
    assert q.loc["ok", "qc_pass"] and q.loc["ok", "qc_fail"] == ""
    assert q.loc["contam", "qc_fail"] == "contamination"
    assert q.loc["frag", "qc_fail"] == "contigs;n50"
    assert q.loc["para", "qc_fail"] == "species"
    assert q.loc["broken", "qc_fail"] == "build"  # missing statistics are not counted as failing every rule
    assert not q.loc["broken", "qc_pass"]


def test_read_shards_keeps_the_newest_row(tmp_path):
    a, b = tmp_path / "a.tsv", tmp_path / "b.tsv"
    pd.DataFrame({"Sample": ["S1", "S2"], "status": ["fail:x", "ok"], "st": ["-", "235"]}).to_csv(a, sep="\t", index=False)
    pd.DataFrame({"Sample": ["S1"], "status": ["ok"], "st": ["175"]}).to_csv(b, sep="\t", index=False)
    import os
    os.utime(a, (1, 1))
    s = read_shards(tmp_path).set_index("Sample")
    assert s.loc["S1", "status"] == "ok" and s.loc["S1", "st"] == "175" and s.loc["S2", "st"] == "235"


def test_st_counts_by_drug_only_counts_passing_clean_labels():
    labels = pd.DataFrame({"phenotype-BioSample_ID": ["a", "b", "c", "d"], "meropenem": [1.0, 0.0, 0.5, 1.0],
                           "amikacin": [0.0, None, 1.0, 1.0]})
    g = pd.DataFrame({"Sample": ["a", "b", "c", "d"], "st": ["235", "235", "235", "111"],
                      "qc_pass": [True, True, True, False]})
    st = st_counts_by_drug(g, labels).set_index(["drug", "st"])
    assert st.loc[("meropenem", "235"), "n"] == 2 and st.loc[("meropenem", "235"), "n_R"] == 1
    assert ("meropenem", "111") not in st.index
    assert st.loc[("amikacin", "235"), "n"] == 2 and st.loc[("amikacin", "235"), "n_R"] == 1


# --- PsA's call is unchanged by the multi-species options -----------------------------------------------------------
# data/collate_psa_golden/ was written by collate_genomes at fbc500a (before --species-mode, --cohort, multi-file
# --checkm2/--skani and repeated --species-ref existed) from collate_fixtures.write_psa_inputs, with the arguments of
# build_pa_genomes.sh collate.

GOLDEN = Path(__file__).parent / "data" / "collate_psa_golden"


def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv)
    collate_genomes.main()


def test_psa_call_writes_byte_identical_outputs(tmp_path, monkeypatch):
    p = write_psa_inputs(tmp_path / "in")
    _run(psa_argv(p, tmp_path / "out"), monkeypatch)
    for name in ("genome_qc.csv", "embedding_input.csv", "st_counts_by_drug.csv"):
        assert (tmp_path / "out" / name).read_bytes() == (GOLDEN / name).read_bytes(), name
    assert not (tmp_path / "out" / "species_crosstab.csv").exists()
    assert not (tmp_path / "out" / "embedding_input_unfiltered.csv").exists()  # report mode's name only


def test_psa_call_with_split_qc_files_equals_one_file(tmp_path, monkeypatch):
    """The same CheckM2 and skani rows in two files each give the same frames as one file."""
    p = write_psa_inputs(tmp_path / "in")
    for key in ("checkm2", "skani"):
        df = pd.read_csv(p[key], sep="\t")
        parts = [p[key].with_suffix(f".{i}.tsv") for i in (0, 1)]
        df.iloc[:3].to_csv(parts[0], sep="\t", index=False)
        df.iloc[3:].to_csv(parts[1], sep="\t", index=False)
        p[key] = parts
    argv = psa_argv({k: v[0] if isinstance(v, list) else v for k, v in p.items()}, tmp_path / "out")
    i = argv.index("--checkm2")
    argv[i + 1:i + 2] = map(str, p["checkm2"])
    i = argv.index("--skani")
    argv[i + 1:i + 2] = map(str, p["skani"])
    _run(argv, monkeypatch)
    for name in ("genome_qc.csv", "embedding_input.csv", "st_counts_by_drug.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "out" / name), pd.read_csv(GOLDEN / name))


def test_st_counts_with_one_scheme_keep_bare_st():
    labels = pd.DataFrame({"phenotype-BioSample_ID": ["a", "b"], "meropenem": [1.0, 0.0]})
    g = pd.DataFrame({"Sample": ["a", "b", "c"], "st": ["235", "-", None], "mlst_scheme": ["paeruginosa"] * 2 + [None],
                      "qc_pass": [True, True, False]})
    assert set(st_counts_by_drug(g, labels)["st"]) == {"235", "-"}


# --- multi-file QC inputs -------------------------------------------------------------------------------------------

def test_read_checkm2_concatenates_and_rejects_a_genome_in_two_reports(tmp_path):
    a, b = tmp_path / "a.tsv", tmp_path / "b.tsv"
    pd.DataFrame({"Name": ["S1", "S2"], "Completeness": [99.0, 98.0], "Contamination": [0.1, 0.2]}
                 ).to_csv(a, sep="\t", index=False)
    pd.DataFrame({"Name": ["S3"], "Completeness": [97.0], "Contamination": [0.3]}).to_csv(b, sep="\t", index=False)
    q = read_checkm2([a, b])
    assert list(q.index) == ["S1", "S2", "S3"] and q.loc["S3", "checkm2_completeness"] == 97.0
    pd.DataFrame({"Name": ["S2"], "Completeness": [97.0], "Contamination": [0.3]}).to_csv(b, sep="\t", index=False)
    with pytest.raises(ValueError, match=r"\['S2'\]"):
        read_checkm2([a, b])


def test_read_skani_rejects_a_pair_in_two_files(tmp_path):
    a, b = tmp_path / "a.tsv", tmp_path / "b.tsv"
    row = {"Ref_file": ["/r/PAO1.fna.gz"], "Query_file": ["/l/S1.fna.gz"], "ANI": [99.1]}
    pd.DataFrame(row).to_csv(a, sep="\t", index=False)
    pd.DataFrame(row).to_csv(b, sep="\t", index=False)
    with pytest.raises(ValueError, match="appear twice"):
        read_skani([a, b], {PA: "/x/PAO1.fna.gz"})


def test_read_skani_rejects_a_reference_it_was_not_given(tmp_path):
    """A reference list that has drifted from the skani run stops collate, naming the reference, rather than
    dropping its hits; a given reference skani found nothing for is fine (its column is empty)."""
    p = tmp_path / "skani.tsv"
    pd.DataFrame({"Ref_file": ["/r/k12.fna.gz", "/r/sflex.fna.gz", "/r/kpne.fna.gz"],
                  "Query_file": ["/l/A.fna.gz", "/l/A.fna.gz", "/l/B.fna.gz"],
                  "ANI": [97.0, 96.0, 99.0]}).to_csv(p, sep="\t", index=False)
    with pytest.raises(ValueError, match=r"1 reference file\(s\) .* not among the 2 references given: \['kpne.fna.gz'\]"):
        read_skani(p, [(EC, "/x/k12.fna.gz"), (SF, "/x/sflex.fna.gz")])
    s = read_skani(p, [(EC, "/x/k12.fna.gz"), (SF, "/x/sflex.fna.gz"), (KP, "/x/kpne.fna.gz"), (MM, "/x/mmor.fna.gz")])
    assert s.loc["B", "species_call"] == KP and s[f"ani_{MM}"].isna().all()


def test_collate_stops_when_the_reference_list_drifted_from_skani(tmp_path, monkeypatch):
    p = write_entero_inputs(tmp_path / "in")
    argv = entero_argv(p, tmp_path / "out")
    i = argv.index(f"{MM}=/x/GCF_006094455.1.fna.gz")  # E7's only hit
    with pytest.raises(ValueError, match="GCF_006094455.1.fna.gz"):
        _run(argv[:i - 1] + argv[i + 1:], monkeypatch)
    assert not (tmp_path / "out").exists()


def test_repeated_species_name_takes_its_best_reference(tmp_path):
    p = tmp_path / "skani.tsv"
    pd.DataFrame({"Ref_file": ["/r/k12.fna.gz", "/r/sakai.fna.gz", "/r/sflex.fna.gz", "/r/k12.fna.gz", "/r/sflex.fna.gz"],
                  "Query_file": ["/l/A.fna.gz", "/l/A.fna.gz", "/l/A.fna.gz", "/l/B.fna.gz", "/l/B.fna.gz"],
                  "ANI": [97.0, 97.8, 97.5, 97.2, 98.0]}).to_csv(p, sep="\t", index=False)
    s = read_skani(p, [(EC, "/x/k12.fna.gz"), (EC, "/x/sakai.fna.gz"), (SF, "/x/sflex.fna.gz")])
    assert list(s.columns) == [f"ani_{EC}", f"ani_{SF}", "species_call"]
    assert s.loc["A", f"ani_{EC}"] == 97.8 and s.loc["A", "species_call"] == EC  # Sakai beats S. flexneri
    assert s.loc["B", "species_call"] == SF
    with pytest.raises(ValueError, match="given for both"):
        read_skani(p, [(EC, "/x/k12.fna.gz"), (SF, "/y/k12.fna.gz")])


def test_built_genome_without_checkm2_row_stops_collate(tmp_path, monkeypatch, capsys):
    p = write_entero_inputs(tmp_path / "in", drop_checkm2=("E5",))
    with pytest.raises(SystemExit, match=r"1 built genomes have no row in the 2 CheckM2 report\(s\): \['E5'\]"):
        _run(entero_argv(p, tmp_path / "out"), monkeypatch)
    _run(entero_argv(p, tmp_path / "out", "--allow-incomplete"), monkeypatch)  # inspectable, and E5 then fails
    q = pd.read_csv(tmp_path / "out" / "genome_qc.csv").set_index("Sample")
    assert q.loc["E5", "qc_fail"] == "completeness;contamination"


# --- species mode report --------------------------------------------------------------------------------------------

def test_report_mode_never_fails_species():
    g = pd.DataFrame({"Sample": ["a", "b"], "status": ["ok", "ok"], "checkm2_completeness": [99.0, 99.0],
                      "checkm2_contamination": [0.5, 0.5], "total_length": [5e6, 5e6], "n_contigs": [80, 80],
                      "n50": [2e5, 2e5], "species_call": [EC, "unknown"]})
    assert apply_qc(g.copy(), None)["qc_pass"].all()
    assert apply_qc(g.copy(), EC)["qc_fail"].tolist() == ["", "species"]


def test_report_mode_end_to_end(tmp_path, monkeypatch, capsys):
    p = write_entero_inputs(tmp_path / "in")
    _run(entero_argv(p, tmp_path / "out"), monkeypatch)
    out = tmp_path / "out"
    assert "embedding_input.csv is not written in report mode" in capsys.readouterr().out
    q = pd.read_csv(out / "genome_qc.csv", dtype={"st": str}).set_index("Sample")
    assert list(q.columns[:5]) == ["source", "n_drugs_labelled", "species", "species_genotype", "species_conflict"]
    assert not q["qc_fail"].fillna("").str.contains("species").any()
    assert q.loc["E6", "species_call"] == "unknown" and q.loc["E6", "qc_pass"]  # no skani hit, still passes
    assert q.loc["E3", "species_call"] == SF and q.loc["E1", f"ani_{EC}"] == 97.8
    assert q.loc["E8", "qc_fail"] == "build"
    # no species rule yet: the pass set is written under its own name, never as the embedding input
    assert not (out / "embedding_input.csv").exists()
    unfiltered = pd.read_csv(out / "embedding_input_unfiltered.csv")
    assert list(unfiltered.columns) == ["Sample", "sr_assembly_file", "sr_gff_file"]
    assert sorted(unfiltered["Sample"]) == ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]

    xt = pd.read_csv(out / "species_crosstab.csv").set_index(["label_species", "species_call"])
    assert xt.loc[("Escherichia coli", EC), "n"] == 1 and xt.loc[("Escherichia coli", SF), "n"] == 1
    assert xt.loc[("Escherichia coli", SF), "n_species_conflict"] == 1
    assert xt.loc[("Escherichia coli", EC), "best_ani_median"] == pytest.approx(97.8)
    assert xt.loc[("Escherichia coli", EC), "gap_to_runner_up_median"] == pytest.approx(0.9)  # 97.8 vs S. flexneri
    assert xt.loc[("Klebsiella pneumoniae", "unknown"), "n_no_skani_hit"] == 1
    assert xt["n"].sum() == 7  # built genomes only

    st = pd.read_csv(out / "st_counts_by_drug.csv", dtype={"st": str})
    assert {"klebsiella:11", "senterica_achtman_2:11", "ecoli_achtman_4:-", "-:-"} <= set(st["st"])
    assert "11" not in set(st["st"])


def test_species_crosstab_quantiles():
    g = pd.DataFrame({"Sample": list("abcd"), "status": ["ok", "ok", "ok", "fail:x"],
                      "species": [EC, EC, EC, EC], "species_conflict": [False, "True", False, False],
                      f"ani_{EC}": [97.0, 98.0, 96.0, None], f"ani_{SF}": [96.0, None, 96.5, None],
                      "species_call": [EC, EC, SF, None], "qc_pass": [True, False, True, False]})
    xt = species_crosstab(g).set_index(["label_species", "species_call"])
    assert xt.loc[(EC, EC), "n"] == 2 and xt.loc[(EC, EC), "n_qc_pass"] == 1
    assert xt.loc[(EC, EC), "n_species_conflict"] == 1
    assert xt.loc[(EC, EC), "best_ani_min"] == 97.0 and xt.loc[(EC, EC), "best_ani_max"] == 98.0
    assert xt.loc[(EC, EC), "gap_to_runner_up_median"] == pytest.approx(1.0)  # b has one species only: NaN, skipped
    assert xt.loc[(EC, SF), "best_ani_median"] == 96.5


def test_untyped_never_a_candidate():
    assert untyped(pd.Series(["-", "ecoli_achtman_4:-", "-:-", "235", "klebsiella:11"])).tolist() == [
        True, True, True, False, False]


@pytest.mark.parametrize("extra, message", [
    (["--target-species", EC], "drop --target-species"),
])
def test_report_mode_argument_checks(tmp_path, monkeypatch, extra, message):
    p = write_entero_inputs(tmp_path / "in")
    with pytest.raises(SystemExit, match=message):
        _run(entero_argv(p, tmp_path / "out", *extra), monkeypatch)


def test_report_mode_needs_cohort_and_target_mode_needs_target(tmp_path, monkeypatch):
    p = write_entero_inputs(tmp_path / "in")
    argv = entero_argv(p, tmp_path / "out")
    i = argv.index("--cohort")
    with pytest.raises(SystemExit, match="needs --cohort"):
        _run(argv[:i] + argv[i + 2:], monkeypatch)
    i = argv.index("--species-mode")
    with pytest.raises(SystemExit, match="needs --target-species"):
        _run(argv[:i] + argv[i + 2:], monkeypatch)
