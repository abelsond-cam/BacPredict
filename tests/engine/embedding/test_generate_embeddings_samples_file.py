"""generate_embeddings --samples-file: a sample list instead of the directory listing; the default path unchanged.

The model and the per-genome work are stubbed, so these run on CPU in milliseconds: what is under test is which
parquets a task is given, in which order, and that a bad list stops the task before the model loads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import bacpredict.engine.embedding.generate_embeddings as ge
from bacpredict.engine.embedding.sample_list import parse_samples, read_samples_file

# SAMN123 < SAMN1234 as IDs, but SAMN1234_… < SAMN123_… as file names ('4' < '_'): the two orders differ
SAMPLES = ["SAMEA001", "SAMN123", "SAMN1234", "SAMN2", "SAMN3"]


def _store(tmp_path: Path, samples=SAMPLES) -> Path:
    d = tmp_path / "protein_sequences"
    d.mkdir()
    for s in samples:
        (d / f"{s}_protein_sequences.parquet").write_bytes(b"stub")
    return d


def _run(monkeypatch, tmp_path: Path, *argv: str) -> list[str]:
    """main() with the model stubbed; returns the samples it embedded, in order."""
    done: list[str] = []

    def process(path, *a, **kw):
        sid = ge.extract_sample_id(path)
        done.append(sid)
        return sid, True, ""

    monkeypatch.setattr(ge, "load_esmc_pinned", lambda device: (None, None))
    monkeypatch.setattr(ge, "process_genome_from_protein_sequences", process)
    monkeypatch.setattr(sys, "argv", ["generate_embeddings", "--esm-dir", str(tmp_path / "esm"), *argv])
    ge.main()
    return done


def _no_model(monkeypatch):
    def boom(device):
        raise AssertionError("the model was loaded")

    monkeypatch.setattr(ge, "load_esmc_pinned", boom)


# ---------------------------------------------------------------------------------------------------------------
# the list format


def test_parse_skips_blank_lines_and_strips_whitespace():
    assert parse_samples("A\n\n  B \t\r\n\nC") == ["A", "B", "C"]


@pytest.mark.parametrize("text, what", [("A\nC\nB\n", "out of order"), ("A\nB\nB\n", "duplicate"), ("\n \n", "no")])
def test_read_refuses_unsorted_duplicate_or_empty(tmp_path, text, what):
    f = tmp_path / "s.txt"
    f.write_text(text)
    with pytest.raises(ValueError, match=what):
        read_samples_file(f)


# ---------------------------------------------------------------------------------------------------------------
# the default path (no --samples-file) is what it was


def test_default_path_is_the_sorted_directory_listing(tmp_path, monkeypatch):
    d = _store(tmp_path)
    expected = [ge.extract_sample_id(p) for p in sorted(d.glob("*_protein_sequences.parquet"))]
    assert expected.index("SAMN1234") < expected.index("SAMN123")  # file-name order, as before
    assert _run(monkeypatch, tmp_path, "--input-dir", str(d)) == expected


@pytest.mark.parametrize("extra, sl", [
    (["--start-idx", "1", "--end-idx", "3"], slice(1, 3)),
    (["--n", "4", "--start-idx", "2", "--end-idx", "9"], slice(2, 4)),
    (["--start-idx", "2"], slice(None)),  # slicing needs both ends, as before
])
def test_default_path_slices_the_listing_as_before(tmp_path, monkeypatch, extra, sl):
    d = _store(tmp_path)
    listing = [ge.extract_sample_id(p) for p in sorted(d.glob("*_protein_sequences.parquet"))]
    n = int(extra[extra.index("--n") + 1]) if "--n" in extra else None
    assert _run(monkeypatch, tmp_path, "--input-dir", str(d), *extra) == listing[:n][sl]


def test_default_path_skip_existing_never_loads_the_model_when_done(tmp_path, monkeypatch):
    d = _store(tmp_path)
    esm = tmp_path / "esm"
    esm.mkdir()
    for s in SAMPLES:
        (esm / f"{s}_esm_embeddings.pt").write_bytes(b"x")
    _no_model(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["g", "--input-dir", str(d), "--esm-dir", str(esm), "--skip-existing"])
    ge.main()


# ---------------------------------------------------------------------------------------------------------------
# --samples-file


def test_samples_file_embeds_only_the_list_in_list_order(tmp_path, monkeypatch):
    d = _store(tmp_path, SAMPLES + ["OTHER_SPECIES"])
    lst = tmp_path / "esm_samples.txt"
    lst.write_text("SAMN123\nSAMN1234\n\nSAMN3\n")
    assert _run(monkeypatch, tmp_path, "--input-dir", str(d), "--samples-file", str(lst)) == [
        "SAMN123", "SAMN1234", "SAMN3"]


def test_samples_file_slices_and_n_index_the_list(tmp_path, monkeypatch):
    d = _store(tmp_path, SAMPLES + ["OTHER_SPECIES"])
    lst = tmp_path / "esm_samples.txt"
    lst.write_text("\n".join(SAMPLES) + "\n")
    got = _run(monkeypatch, tmp_path, "--input-dir", str(d), "--samples-file", str(lst),
               "--start-idx", "1", "--end-idx", "3")
    assert got == SAMPLES[1:3]
    got = _run(monkeypatch, tmp_path, "--input-dir", str(d), "--samples-file", str(lst), "--n", "2")
    assert got == SAMPLES[:2]


def test_samples_file_skip_existing_filters_within_the_slice(tmp_path, monkeypatch):
    d = _store(tmp_path)
    esm = tmp_path / "esm"
    esm.mkdir()
    (esm / "SAMN123_esm_embeddings.pt").write_bytes(b"x")
    lst = tmp_path / "esm_samples.txt"
    lst.write_text("\n".join(SAMPLES) + "\n")
    got = _run(monkeypatch, tmp_path, "--input-dir", str(d), "--samples-file", str(lst), "--skip-existing",
               "--start-idx", "0", "--end-idx", "3")
    assert got == ["SAMEA001", "SAMN1234"]


def test_a_missing_listed_parquet_exits_before_the_model_loads(tmp_path, monkeypatch):
    d = _store(tmp_path, ["SAMN2"])
    lst = tmp_path / "esm_samples.txt"
    lst.write_text("SAMN1\nSAMN2\n")
    _no_model(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["g", "--input-dir", str(d), "--esm-dir", str(tmp_path / "esm"),
                                      "--samples-file", str(lst)])
    with pytest.raises(SystemExit) as e:
        ge.main()
    assert "SAMN1_protein_sequences.parquet" in str(e.value.code)


def test_a_missing_parquet_outside_the_slice_is_another_tasks_to_report(tmp_path, monkeypatch):
    d = _store(tmp_path, ["A", "B", "D"])
    lst = tmp_path / "esm_samples.txt"
    lst.write_text("A\nB\nC\nD\n")
    got = _run(monkeypatch, tmp_path, "--input-dir", str(d), "--samples-file", str(lst),
               "--start-idx", "0", "--end-idx", "2")
    assert got == ["A", "B"]


@pytest.mark.parametrize("text", ["B\nA\n", "A\nA\n", "\n"])
def test_a_bad_list_exits_before_the_model_loads(tmp_path, monkeypatch, text):
    d = _store(tmp_path, ["A", "B"])
    lst = tmp_path / "esm_samples.txt"
    lst.write_text(text)
    _no_model(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["g", "--input-dir", str(d), "--esm-dir", str(tmp_path / "esm"),
                                      "--samples-file", str(lst)])
    with pytest.raises(SystemExit) as e:
        ge.main()
    assert str(e.value.code).startswith("--samples-file:")
