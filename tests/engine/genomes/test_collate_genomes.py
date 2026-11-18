"""Genome QC: the ATB high-quality rules, the skani species call, and ST counts per drug."""

from __future__ import annotations

import pandas as pd

from bacpredict.engine.genomes.collate_genomes import apply_qc, read_shards, read_skani, st_counts_by_drug

PA, PARA = "Pseudomonas aeruginosa", "Pseudomonas paraeruginosa"


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
    assert set(q.loc["broken", "qc_fail"].split(";")) >= {"build", "completeness", "species"}


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
