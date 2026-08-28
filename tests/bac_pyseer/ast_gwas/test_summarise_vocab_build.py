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
