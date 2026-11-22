"""reextract_proteins rewrites only the parquets whose proteins change, and never an excluded genome's."""

from __future__ import annotations

import os

import pandas as pd

from bacpredict.engine.embedding.reextract_proteins import genome_record, main, parquet_path, write_atomically


def _genome(d, name, seq):
    gff, fna = d / f"{name}.gff3", d / f"{name}.fna"
    gff.write_text(f"##gff-version 3\nc1\tprodigal\tCDS\t1\t{len(seq)}\t.\t+\t0\tID=x\n")
    fna.write_text(f">c1\n{seq}\n")
    return str(fna), str(gff)


def test_only_changed_parquets_are_rewritten(tmp_path):
    out = tmp_path / "prot"
    out.mkdir()
    genomes = {"PLAIN": "ATGAAAAAATAA", "AMBIG": "ATGRAYAAATAA", "STOP": "ATGTRAAAATAA", "NEW": "ATGAAATAA",
               "OWNER": "ATGRAYAAATAA"}
    rows = []
    for s, seq in genomes.items():
        fna, gff = _genome(tmp_path, s, seq)
        rows.append({"Sample": s, "sr_assembly_file": fna, "sr_gff_file": gff})
        if s != "NEW":  # the old store: Biopython's resolution (STOP's CDS dropped)
            write_atomically(genome_record(s, fna, gff, ambiguous_codons_as_x=False), parquet_path(out, s))
            os.utime(parquet_path(out, s), (1_000_000, 1_000_000))
    cohort = tmp_path / "cohort.csv"
    pd.DataFrame(rows).to_csv(cohort, index=False)
    exclude = tmp_path / "reused.csv"
    pd.DataFrame({"Sample": ["OWNER"]}).to_csv(exclude, index=False)
    report = tmp_path / "report.csv"
    assert main(["--input-csv", str(cohort), "--exclude", str(exclude), "--output-dir", str(out),
                 "--ambiguous-codons-as-x", "--workers", "2", "--report", str(report)]) == 0
    r = pd.read_csv(report).set_index("Sample")
    assert r["status"].to_dict() == {"PLAIN": "unchanged", "AMBIG": "rewritten", "STOP": "rewritten", "NEW": "written"}
    assert (r.loc["STOP", "n_before"], r.loc["STOP", "n_after"]) == (0, 1)
    assert os.stat(parquet_path(out, "PLAIN")).st_mtime == 1_000_000  # an unchanged parquet is not touched
    assert os.stat(parquet_path(out, "OWNER")).st_mtime == 1_000_000  # nor an excluded genome's
    seq = pd.read_parquet(parquet_path(out, "AMBIG"))["protein_sequence"].iloc[0]
    assert [list(c) for c in seq] == [["MXK"]]
    again = tmp_path / "again.csv"
    main(["--input-csv", str(cohort), "--output-dir", str(out), "--ambiguous-codons-as-x", "--workers", "2",
          "--report", str(again), "--exclude", str(exclude)])
    assert set(pd.read_csv(again)["status"]) == {"unchanged"}  # idempotent
