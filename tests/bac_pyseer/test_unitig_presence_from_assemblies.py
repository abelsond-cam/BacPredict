"""Tests for calling hit-unitig presence straight from assembly FASTAs.

This is the path that lets the fitted unitig model score genomes the GWAS never saw. The failure
modes are all quiet ones: a unitig present only on the reverse strand being called absent, the
column order drifting away from the model's, or a missing assembly silently becoming an all-zero
genome that then looks confidently faecal.
"""

from __future__ import annotations

import gzip

import pandas as pd
import pytest

from bac_pyseer.kleb_iso_source.unitig_presence_from_assemblies import (
    load_unitig_order,
    scan_assemblies,
)

pytest.importorskip("ahocorasick", reason="Aho-Corasick matching needs pyahocorasick")

# 30-mers, long enough to be unique in the toy contigs below.
U1 = "ACGTACGTACGTACGTACGTACGTACGTAC"
U2 = "TTTTGGGGCCCCAAAATTTTGGGGCCCCAA"
U3 = "GATTACAGATTACAGATTACAGATTACAGA"


def _revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _write_fasta(path, contigs: dict[str, str], gz: bool = False):
    text = "".join(f">{name}\n{seq}\n" for name, seq in contigs.items())
    if gz:
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        path.write_text(text)
    return path


def test_presence_is_called_on_both_strands(tmp_path):
    """A unitig is present if the genome carries it in either orientation — assemblies are unoriented."""
    fwd = _write_fasta(tmp_path / "fwd.fa", {"c1": "AAAA" + U1 + "TTTT"})
    rev = _write_fasta(tmp_path / "rev.fa", {"c1": "AAAA" + _revcomp(U1) + "TTTT"})
    asm = pd.DataFrame({"Sample": ["fwd", "rev"], "assembly_path": [str(fwd), str(rev)]})

    X, ids, qc = scan_assemblies(asm, [U1, U2])
    assert ids == ["fwd", "rev"]
    dense = X.toarray()
    assert dense[0, 0] == 1 and dense[1, 0] == 1, "reverse-complement carriage must count as present"
    assert dense[:, 1].sum() == 0, "an absent unitig must stay absent"


def test_column_order_follows_the_model_not_the_scan(tmp_path):
    """Coefficients are positional: column j must be the model's unitig j regardless of what is found."""
    a = _write_fasta(tmp_path / "a.fa", {"c1": U3 + "AAAA" + U1})
    asm = pd.DataFrame({"Sample": ["a"], "assembly_path": [str(a)]})

    order = [U1, U2, U3]
    X, _, _ = scan_assemblies(asm, order)
    dense = X.toarray()[0]
    assert dense.tolist() == [1, 0, 1], "presence must land in the model's column positions"

    reordered = [U3, U2, U1]
    X2, _, _ = scan_assemblies(asm, reordered)
    assert X2.toarray()[0].tolist() == [1, 0, 1]


def test_gzipped_assemblies_are_read(tmp_path):
    gz = _write_fasta(tmp_path / "a.fa.gz", {"c1": U1}, gz=True)
    asm = pd.DataFrame({"Sample": ["a"], "assembly_path": [str(gz)]})
    X, _, qc = scan_assemblies(asm, [U1])
    assert X.toarray()[0, 0] == 1
    assert qc.iloc[0]["assembly_found"]


def test_missing_assembly_is_flagged_not_silently_zero(tmp_path):
    """An all-zero row scores as confidently non-invasive; it must be distinguishable from a real one."""
    real = _write_fasta(tmp_path / "a.fa", {"c1": U1})
    asm = pd.DataFrame({"Sample": ["a", "gone"],
                        "assembly_path": [str(real), str(tmp_path / "nope.fa")]})
    X, ids, qc = scan_assemblies(asm, [U1, U2])

    assert X.shape == (2, 2)
    q = qc.set_index("Sample")
    assert q.loc["gone", "assembly_found"] is False or not q.loc["gone", "assembly_found"]
    assert q.loc["gone", "n_unitigs_present"] == 0
    assert q.loc["a", "assembly_found"]


def test_multiple_copies_count_once(tmp_path):
    """Presence is binary — a unitig repeated across contigs is still a single feature."""
    a = _write_fasta(tmp_path / "a.fa", {"c1": U1 + "AAAA" + U1, "c2": U1})
    asm = pd.DataFrame({"Sample": ["a"], "assembly_path": [str(a)]})
    X, _, qc = scan_assemblies(asm, [U1])
    assert X.toarray()[0, 0] == 1
    assert qc.iloc[0]["n_unitigs_present"] == 1


def test_load_unitig_order_round_trips(tmp_path):
    p = tmp_path / "unitigs.csv"
    pd.DataFrame({"unitig": [U1, U2, U3]}).to_csv(p, index=False)
    assert load_unitig_order(p) == [U1, U2, U3]
