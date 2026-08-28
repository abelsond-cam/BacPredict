"""The build summary must never report an unrun assertion as a passed one."""

from __future__ import annotations

import json

import pytest

from bac_pyseer.ast_gwas.summarise_vocab_build import run, summarise_drug

CLEAN = {
    "reflist": {
        "n_reflist": 1128, "n_train": 902, "n_validate": 226, "n_holdout": 282,
        "n_holdout_in_reflist": 0, "n_outside_trainval": 0, "min_samples_floor": 12,
    },
    "vocabulary": {
        "n_colors": 1128, "n_missing_from_graph": 0, "n_extra_in_graph": 0,
        "n_holdout_coloured": 0, "n_holdout_in_matrix_head": 0,
    },
    "clusters": {"n_clusters": 5, "n_in_other": 848, "n_missing": 0, "n_extra": 0},
}


def _drug(tmp_path, name="colistin", audit=None, matrix=b"x" * 16):
    """Lay out one drug directory: audit file plus the artifacts the summary stats."""
    d = tmp_path / name
    (d / "unitigs").mkdir(parents=True)
    (d / "structure").mkdir(parents=True)
    if audit is not None:
        (d / "leakage_audit.json").write_text(json.dumps(audit))
    if matrix:
        (d / "unitigs" / "unitigs.pyseer.gz").write_bytes(matrix)
    return d


def test_clean_build_is_ok(tmp_path):
    _drug(tmp_path, audit=CLEAN)
    row = summarise_drug(tmp_path, "colistin")
    assert row["status"] == "ok"
    assert row["notes"] == ""
    assert row["n_reflist"] == 1128
    assert row["min_samples_floor"] == 12
    assert row["matrix_bytes"] == 16


def test_a_holdout_genome_in_the_colour_set_fails(tmp_path):
    """The assertion the whole rebuild rests on: one coloured holdout genome must fail the drug."""
    audit = json.loads(json.dumps(CLEAN))
    audit["vocabulary"]["n_holdout_coloured"] = 1
    _drug(tmp_path, audit=audit)
    row = summarise_drug(tmp_path, "colistin")
    assert row["status"] == "FAIL"
    assert "n_holdout_coloured=1" in row["notes"]


def test_missing_vocabulary_section_is_unchecked_not_ok(tmp_path):
    """A GGCAT job that died leaves no vocabulary block. That is UNCHECKED, never a pass."""
    audit = {k: v for k, v in CLEAN.items() if k != "vocabulary"}
    _drug(tmp_path, audit=audit)
    row = summarise_drug(tmp_path, "colistin")
    assert row["status"] == "unchecked"
    assert row["status"] != "ok"
    assert "vocabulary" in row["notes"]


def test_absent_audit_file_is_unchecked(tmp_path):
    _drug(tmp_path, audit=None)
    row = summarise_drug(tmp_path, "colistin")
    assert row["status"] == "unchecked"
    assert "never started" in row["notes"]


def test_colour_count_disagreeing_with_the_reflist_fails(tmp_path):
    """The silent-reuse backstop: a graph built from a different cohort still colours cleanly."""
    audit = json.loads(json.dumps(CLEAN))
    audit["vocabulary"]["n_colors"] = 7080          # the full-cohort graph, reused by accident
    audit["vocabulary"]["n_holdout_coloured"] = 0   # ... and its own holdout check says nothing
    _drug(tmp_path, audit=audit)
    row = summarise_drug(tmp_path, "colistin")
    assert row["status"] == "FAIL"
    assert "n_colors=7080" in row["notes"]


def test_empty_matrix_alongside_a_vocabulary_section_fails(tmp_path):
    _drug(tmp_path, audit=CLEAN, matrix=b"")
    row = summarise_drug(tmp_path, "colistin")
    assert row["status"] == "FAIL"
    assert "unitigs.pyseer.gz" in row["notes"]


def test_run_exit_code_is_nonzero_when_any_drug_is_not_clean(tmp_path, capsys):
    _drug(tmp_path, "colistin", audit=CLEAN)
    _drug(tmp_path, "gentamicin", audit={k: v for k, v in CLEAN.items() if k != "clusters"})
    out_tsv = tmp_path / "out" / "summary.tsv"
    assert run(tmp_path, out_tsv) == 1
    text = capsys.readouterr().out
    assert "1/2 clean" in text
    assert "1 unchecked" in text
    assert out_tsv.exists()
    header, *body = out_tsv.read_text().splitlines()
    assert "n_holdout_coloured" in header.split("\t")
    assert len(body) == 2


def test_run_exit_code_is_zero_when_every_drug_is_clean(tmp_path):
    _drug(tmp_path, "colistin", audit=CLEAN)
    _drug(tmp_path, "gentamicin", audit=CLEAN)
    assert run(tmp_path, None) == 0


def test_empty_root_is_an_error_not_a_silent_pass(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, None)


GOOD_MERGE = {
    "rows_from_ggcat": 1537, "rows_from_scanner": 384, "n_features": 4785, "nnz": 2418618,
    "shard_completeness": {"n_shards": 8, "shard_files": [f"s{i}" for i in range(8)]},
    "verification": {"n_shared": 1537, "cells": 7354545, "n_mismatch_cells": 0},
    "holdout_coverage": {"checked": True, "ratio": 1.0096},
}


def _readout(tmp_path, drug="azithromycin", merge=None, n_holdout=384):
    d = tmp_path / drug
    (d / drug / "design_merged").mkdir(parents=True)
    if merge is not None:
        (d / drug / "design_merged" / "merge_manifest.json").write_text(json.dumps(merge))
    (d / "leakage_audit.json").write_text(json.dumps({"reflist": {"n_holdout": n_holdout}}))
    return d


def test_a_healthy_readout_is_ok(tmp_path):
    from bac_pyseer.ast_gwas.summarise_vocab_build import summarise_readout
    _readout(tmp_path, merge=GOOD_MERGE)
    r = summarise_readout(tmp_path, "azithromycin")
    assert r["readout_status"] == "ok"
    assert r["rows_from_scanner"] == 384 and r["expected_holdout"] == 384
    assert r["n_mismatch_cells"] == 0


def test_a_truncated_scan_fails_even_though_every_statistic_looks_fine(tmp_path):
    """The 2026-08-28 failure, as it actually presented: 1 shard of 8, 36 of 282 holdout genomes,
    zero mismatches, and a perfectly healthy carrier ratio."""
    from bac_pyseer.ast_gwas.summarise_vocab_build import summarise_readout
    merge = json.loads(json.dumps(GOOD_MERGE))
    merge["rows_from_scanner"] = 36
    merge["shard_completeness"] = {"n_shards": 8, "shard_files": ["s0"]}
    merge["verification"] = {"n_shared": 141, "cells": 795381, "n_mismatch_cells": 0}
    merge["holdout_coverage"] = {"checked": True, "ratio": 1.156}
    _readout(tmp_path, merge=merge, n_holdout=282)
    r = summarise_readout(tmp_path, "azithromycin")
    assert r["readout_status"] == "FAIL"
    assert "1/8 scan shards" in r["readout_notes"]
    assert "36/282 holdout genomes" in r["readout_notes"]


def test_a_missing_shard_record_fails_rather_than_passing_quietly(tmp_path):
    from bac_pyseer.ast_gwas.summarise_vocab_build import summarise_readout
    merge = {k: v for k, v in GOOD_MERGE.items() if k != "shard_completeness"}
    _readout(tmp_path, merge=merge)
    r = summarise_readout(tmp_path, "azithromycin")
    assert r["readout_status"] == "FAIL"
    assert "cannot be shown" in r["readout_notes"]


def test_one_or_two_unscannable_genomes_are_tolerated(tmp_path):
    """A genome with no assembly can never be scanned; 593/594 is normal, not a truncation."""
    from bac_pyseer.ast_gwas.summarise_vocab_build import summarise_readout
    merge = json.loads(json.dumps(GOOD_MERGE))
    merge["rows_from_scanner"] = 593
    _readout(tmp_path, merge=merge, n_holdout=594)
    assert summarise_readout(tmp_path, "azithromycin")["readout_status"] == "ok"


def test_a_drug_without_a_readout_reads_as_absent_not_ok(tmp_path):
    from bac_pyseer.ast_gwas.summarise_vocab_build import summarise_readout
    _readout(tmp_path, merge=None)
    r = summarise_readout(tmp_path, "azithromycin")
    assert r["readout_status"] == "absent"
    assert r["readout_status"] != "ok"


def test_run_readout_exit_code_and_totals(tmp_path, capsys):
    from bac_pyseer.ast_gwas.summarise_vocab_build import run_readout
    _readout(tmp_path, "azithromycin", merge=GOOD_MERGE)
    _readout(tmp_path, "colistin", merge=GOOD_MERGE, n_holdout=384)
    assert run_readout(tmp_path, None) == 0
    out = capsys.readouterr().out
    assert "2/2 read-outs clean" in out
    assert "14,709,090 cells compared, 0 mismatched" in out


def test_drug_enumeration_ignores_directories_that_are_not_drugs(tmp_path):
    """An output directory written beside the drugs is not a drug whose read-out never ran.

    Placing the comparison output inside the vocab root made it a 23rd "drug" with no read-out,
    which stopped the C6 driver while all 22 genuine gates were passing.
    """
    from bac_pyseer.ast_gwas.summarise_vocab_build import drug_dirs
    _drug(tmp_path, "colistin", audit=CLEAN)
    _drug(tmp_path, "ertapenem", audit=CLEAN)
    (tmp_path / "comparison").mkdir()          # output dir: no audit, no nested <drug>/<drug>/
    (tmp_path / "logs").mkdir()
    (tmp_path / "comparison" / "figures").mkdir()   # ... even with children of its own
    assert drug_dirs(tmp_path) == ["colistin", "ertapenem"]


def test_run_skips_non_drug_directories(tmp_path, capsys):
    from bac_pyseer.ast_gwas.summarise_vocab_build import run
    _drug(tmp_path, "colistin", audit=CLEAN)
    (tmp_path / "comparison").mkdir()
    assert run(tmp_path, None) == 0
    assert "1/1 clean" in capsys.readouterr().out
