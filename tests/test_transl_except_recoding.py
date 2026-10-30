"""A declared translational exception must reach the protein, or the encoder sees an `<unk>`.

A GFF3 carries no protein sequence — it carries the annotator's decision, and for a recoded codon that
decision is ``transl_except=(pos:...,aa:Sec)``. Translating with a codon table and ignoring the
attribute writes ``*``, which is absent from the ESM/ESM-C vocabulary and so tokenises to ``<unk>``.
That is not a hypothetical: it put an `<unk>` into 261 stored protein vectors before anyone noticed,
and the only reason it was ever found is that the annotator's own file disagreed.

The cases below are the two genes walked by hand plus the four ways the parse can go wrong.
"""

from __future__ import annotations

import pytest

from bacpredict.engine.embedding.extract_proteins_from_gff_fna import _recode_declared_exceptions

# GCA_003855335.1 fdhF: CDS 4895142..4897289 (+), codon 4895559..4895561 -> protein offset 139
PLUS = {"start": 4895142, "end": 4897289, "strand": "+", "phase": 0}
# SAMD00049595 fdnG: CDS 115749..118796 (-), codon complement(118209..118211) -> offset 195
MINUS = {"start": 115749, "end": 118796, "strand": "-", "phase": 0}
SEC_PLUS = "ID=cds;transl_except=(pos:4895559..4895561%2Caa:Sec);transl_table=11"
SEC_MINUS = "ID=cds;transl_except=(pos:complement(118209..118211)%2Caa:Sec)"


def _protein(offset: int, length: int, residue: str = "*") -> str:
    return "A" * offset + residue + "A" * (length - offset - 1)


def test_plus_strand_walked_gene_gets_its_U():
    out, recoded, skipped = _recode_declared_exceptions(_protein(139, 715), SEC_PLUS, **PLUS)
    assert (recoded, skipped) == (1, 0)
    assert out[139] == "U" and "*" not in out


def test_minus_strand_walked_gene_gets_its_U():
    out, recoded, skipped = _recode_declared_exceptions(_protein(195, 1015), SEC_MINUS, **MINUS)
    assert (recoded, skipped) == (1, 0)
    assert out[195] == "U" and "*" not in out


def test_the_two_strands_do_not_agree_on_the_offset():
    """⛔ A fixture where both strands give the same offset cannot test that the strand is read."""
    plus, _, _ = _recode_declared_exceptions(_protein(139, 715), SEC_PLUS, **PLUS)
    same_span_as_minus, _, skipped = _recode_declared_exceptions(
        _protein(139, 715), SEC_PLUS, **{**PLUS, "strand": "-"}
    )
    assert plus[139] == "U"
    assert skipped == 1 and "U" not in same_span_as_minus  # the minus formula lands elsewhere, so it refuses


def test_two_declarations_on_one_cds_are_both_written():
    """⛔ `_parse_gff_attributes` keeps only the LAST value for a repeated key, so this reads the raw field."""
    attrs = "ID=c;transl_except=(pos:4895559..4895561%2Caa:Sec);transl_except=(pos:4895562..4895564%2Caa:Sec)"
    out, recoded, _ = _recode_declared_exceptions("A" * 139 + "**" + "A" * 574, attrs, **PLUS)
    assert recoded == 2
    assert out[139] == "U" and out[140] == "U"


def test_a_declaration_that_does_not_land_on_a_stop_is_refused():
    """⚠ Verified, not trusted: writing blind would corrupt a residue nothing downstream reveals."""
    out, recoded, skipped = _recode_declared_exceptions("A" * 715, SEC_PLUS, **PLUS)
    assert (recoded, skipped) == (0, 1)
    assert out == "A" * 715


def test_pyrrolysine_is_written_as_O():
    attrs = "transl_except=(pos:4895559..4895561%2Caa:Pyl)"
    out, recoded, _ = _recode_declared_exceptions(_protein(139, 715), attrs, **PLUS)
    assert recoded == 1 and out[139] == "O"


@pytest.mark.parametrize("aa", ["OTHER", "TERM"])
def test_an_exception_we_do_not_write_is_counted_not_guessed(aa):
    attrs = f"transl_except=(pos:4895559..4895561%2Caa:{aa})"
    out, recoded, skipped = _recode_declared_exceptions(_protein(139, 715), attrs, **PLUS)
    assert (recoded, skipped) == (0, 1)
    assert out[139] == "*"


def test_a_cds_with_no_declaration_is_untouched():
    out, recoded, skipped = _recode_declared_exceptions(_protein(139, 715), "ID=c;product=x", **PLUS)
    assert (recoded, skipped) == (0, 0)
    assert out[139] == "*"


def test_a_nonzero_phase_shifts_the_offset():
    """The phase bases are dropped before translation, so the protein starts at start+phase."""
    out, recoded, _ = _recode_declared_exceptions(
        _protein(138, 715), SEC_PLUS, **{**PLUS, "phase": 3}
    )
    assert recoded == 1 and out[138] == "U"
