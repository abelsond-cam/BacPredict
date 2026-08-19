"""Tests for the MiST result parser.

The two invariants worth pinning are the ones that fail *silently*: an unrecognised output shape
and an unnormalised sample id both yield a table that looks fine and joins to nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

from genome_prep.lin_typing.parse_mist_json import parse_one, run, strip_extensions

_METADATA = [
    ["scgST", "37993"],
    ["LINcode", "2_0_220_0_0_0_0_0_0_0"],
    ["Phylogroup", "Kp2"],
    ["Sublineage", "SL12005"],
    ["Clonal group", "CG13650"],
]


def _profile(nb_matches: int = 629) -> dict:
    return {
        "alleles": {},
        "metadata": _METADATA,
        "name": "37993",
        "nb_matches": nb_matches,
        "pct_match": 100 * nb_matches / 629,
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_reads_the_pre_1_3_single_profile_shape(tmp_path: Path) -> None:
    """MiST <= 1.2 wrote one `profile` object; the CSD3 archive is entirely this shape."""
    p = _write(tmp_path / "SAMEA4780982.fa.gz.json", {"alleles": {}, "metadata": {}, "profile": _profile()})
    row = parse_one(p, max_mismatched=30)
    assert row is not None
    assert row["Sublineage"] == "SL12005"
    assert row["scgST"] == "37993"


def test_reads_the_1_3_profiles_list_shape(tmp_path: Path) -> None:
    """1.3 writes a `profiles` list so several equally good STs can be reported."""
    p = _write(tmp_path / "x.json", {"alleles": {}, "metadata": {}, "profiles": [_profile(), _profile()]})
    row = parse_one(p, max_mismatched=30)
    assert row is not None
    assert row["Sublineage"] == "SL12005"
    assert row["n_equivalent_profiles"] == 2


def test_a_result_that_typed_nothing_yields_no_row(tmp_path: Path) -> None:
    p = _write(tmp_path / "y.json", {"alleles": {}, "metadata": {}, "profiles": []})
    assert parse_one(p, max_mismatched=30) is None


def test_sample_id_is_normalised_even_when_mist_recorded_one(tmp_path: Path) -> None:
    """Without --sample-id MiST records the input filename, which joins to nothing on its own."""
    payload = {
        "alleles": {}, "profile": _profile(),
        "metadata": {"input": {"sample_id": "SAMEA4780982.fa.gz", "path": "/tmp/x"}},
    }
    p = _write(tmp_path / "SAMEA4780982.fa.gz.json", payload)
    assert parse_one(p, max_mismatched=30)["Sample"] == "SAMEA4780982"


def test_strip_extensions_handles_stacked_suffixes() -> None:
    assert strip_extensions("SAMN1.fa.gz.json") == "SAMN1"
    assert strip_extensions("SAMN1.fasta") == "SAMN1"
    assert strip_extensions("SAMN1") == "SAMN1"


def test_the_gate_flags_rather_than_deletes(tmp_path: Path) -> None:
    """A poor match keeps its row so the cut can be re-made without re-running MiST."""
    _write(tmp_path / "good.json", {"alleles": {}, "metadata": {}, "profile": _profile(629)})
    _write(tmp_path / "poor.json", {"alleles": {}, "metadata": {}, "profile": _profile(400)})

    out = tmp_path / "out.tsv"
    run(json_dirs=[tmp_path], out_tsv=out, max_mismatched=30)

    rows = [line.split("\t") for line in out.read_text().splitlines()]
    header, body = rows[0], rows[1:]
    gate = {r[header.index("Sample")]: r[header.index("passes_gate")] for r in body}
    assert gate == {"good": "True", "poor": "False"}

    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["n_rows"] == 2
    assert manifest["n_passes_gate"] == 1


def test_tied_profiles_that_disagree_fail_the_gate(tmp_path: Path) -> None:
    """Taking profiles[0] is an arbitrary pick unless the ties agree, so a disagreement is flagged."""
    other = _profile()
    other["metadata"] = [["scgST", "1"], ["Sublineage", "SL11"], ["Clonal group", "CG11"]]
    p = _write(tmp_path / "t.json", {"alleles": {}, "metadata": {}, "profiles": [_profile(), other]})

    row = parse_one(p, max_mismatched=30)
    assert row["n_equivalent_profiles"] == 2
    assert row["tied_profiles_agree"] is False
    assert row["passes_gate"] is False


def test_tied_profiles_that_agree_still_pass(tmp_path: Path) -> None:
    """Several scgSTs inside one sublineage is the normal case and must not be penalised."""
    p = _write(tmp_path / "t.json", {"alleles": {}, "metadata": {}, "profiles": [_profile(), _profile()]})
    row = parse_one(p, max_mismatched=30)
    assert row["tied_profiles_agree"] is True
    assert row["passes_gate"] is True
