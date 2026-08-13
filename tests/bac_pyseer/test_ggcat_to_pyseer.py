"""Unit tests for the GGCAT → pyseer ``--kmers`` matrix converter.

Covers the prevalence band (``min_samples`` / ``max_samples``) that decides which colour
subsets are expanded at all — the main lever on output size, since a near-universal subset
expands to a presence string proportional to the whole cohort.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bac_pyseer.kleb_iso_source.ggcat_to_pyseer import convert

_K = 4
_N_KMERS = 3  # segment length = n_kmers + k - 1 = 6 bases

# Three unitigs over a 5-sample cohort, one colour segment each, spanning the prevalence range:
#   subset 1 -> 1 carrier  (rare: below a 2-sample floor)
#   subset 2 -> 3 carriers (testable middle)
#   subset 3 -> 5 carriers (universal: above a 4-sample ceiling)
_FASTA = """\
>0 LN:i:6 C:1:3
ACGTAC
>1 LN:i:6 C:2:3
TTGGCC
>2 LN:i:6 C:3:3
GGAATT
"""
_COLORMAP = "1,0\n2,0-2\n3,0-4\n"


@pytest.fixture
def ggcat_build(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write a minimal (fasta, color_names, colormap) GGCAT artifact triple."""
    fasta = tmp_path / "unitigs.fa"
    fasta.write_text(_FASTA)
    names = tmp_path / "color_names.jsonl"
    names.write_text(
        "".join(json.dumps({"color_index": i, "color_name": f"SAM{i}"}) + "\n" for i in range(5))
    )
    colormap = tmp_path / "colormap.csv"
    colormap.write_text(_COLORMAP)
    return fasta, names, colormap


def _run(build: tuple[Path, Path, Path], tmp_path: Path, **kw) -> list[str]:
    """Convert and return the written matrix lines."""
    fasta, names, colormap = build
    out = tmp_path / "matrix.txt"
    convert(fasta, names, colormap, out, kmer_length=_K, tmp_dir=tmp_path, **kw)
    return out.read_text().splitlines()


def test_convert_emits_pyseer_line_format(ggcat_build, tmp_path: Path) -> None:
    """With no bounds every subset is emitted as ``<seq> | <Sample>:1 …``."""
    lines = _run(ggcat_build, tmp_path)
    assert len(lines) == 3
    by_seq = {line.split(" | ")[0]: line.split(" | ")[1] for line in lines}
    assert by_seq["ACGTAC"] == "SAM0:1"
    assert by_seq["TTGGCC"] == "SAM0:1 SAM1:1 SAM2:1"
    assert by_seq["GGAATT"] == "SAM0:1 SAM1:1 SAM2:1 SAM3:1 SAM4:1"


def test_min_samples_drops_rare_subsets(ggcat_build, tmp_path: Path) -> None:
    """The existing MAF floor still drops the 1-carrier subset and nothing else."""
    lines = _run(ggcat_build, tmp_path, min_samples=2)
    assert {line.split(" | ")[0] for line in lines} == {"TTGGCC", "GGAATT"}


def test_max_samples_drops_near_universal_subsets(ggcat_build, tmp_path: Path) -> None:
    """The new ceiling drops the 5-carrier subset, which pyseer could not test anyway."""
    lines = _run(ggcat_build, tmp_path, max_samples=4)
    assert {line.split(" | ")[0] for line in lines} == {"ACGTAC", "TTGGCC"}


def test_bounds_combine_to_a_testable_band(ggcat_build, tmp_path: Path) -> None:
    """Floor and ceiling together keep only the subsets inside the band."""
    lines = _run(ggcat_build, tmp_path, min_samples=2, max_samples=4)
    assert [line.split(" | ")[0] for line in lines] == ["TTGGCC"]


def test_no_cap_by_default_preserves_previous_behaviour(ggcat_build, tmp_path: Path) -> None:
    """max_samples=None must not filter — existing builds are unchanged."""
    assert len(_run(ggcat_build, tmp_path, max_samples=None)) == 3


def test_inverted_bounds_are_rejected(ggcat_build, tmp_path: Path) -> None:
    """A ceiling below the floor admits nothing — fail loudly rather than emit an empty matrix."""
    with pytest.raises(ValueError, match="no unitig can pass"):
        _run(ggcat_build, tmp_path, min_samples=4, max_samples=2)
