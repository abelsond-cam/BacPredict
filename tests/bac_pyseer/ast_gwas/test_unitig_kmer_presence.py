"""Tests for scoring unitig presence by k-mer containment — the rule GGCAT colouring encodes.

A train+validate-only vocabulary has no holdout rows, so the holdout is scored from sequence. The
failure modes are quiet: a rule that silently disagrees with the one that produced the training rows,
a reverse-strand carrier called absent, or an ambiguity code counted as a base. The case that
separates k-mer containment from exact-substring matching is narrower than "a contig break" — a clean
break destroys the spanning k-mers, so both rules agree — and it is pinned down below, because it is
the whole reason the two rules had to be measured against each other before one was chosen.
"""

from __future__ import annotations

import numpy as np
import pytest

from bac_pyseer.ast_gwas.unitig_kmer_presence import (
    K,
    canonical_kmers,
    contains_all,
    feature_kmer_table,
    genome_kmer_index,
)
from bac_pyseer.kleb_iso_source.unitig_placement import _revcomp


def _rand(rng, n: int) -> str:
    return "".join(rng.choice(list("ACGT"), n))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_canonical_kmers_are_strand_symmetric(rng):
    """Assemblies are unoriented, so a sequence and its reverse complement must give one k-mer set."""
    s = _rand(rng, 500)
    assert set(canonical_kmers(s).tolist()) == set(canonical_kmers(_revcomp(s)).tolist())


def test_substring_features_are_present_on_either_strand(rng):
    """A feature carried forward or reverse is present; an unrelated feature is not."""
    genome = {"c1": _rand(rng, 20000)}
    feats = [genome["c1"][1000:1040], _revcomp(genome["c1"][3000:3060]), _rand(rng, 60)]
    flat, offsets = feature_kmer_table(feats)
    assert contains_all(genome_kmer_index(genome), flat, offsets).tolist() == [True, True, False]


def test_overlap_repeated_break_is_where_the_two_rules_diverge(rng):
    """All k-mers present but no contiguous occurrence — k-mer containment says present.

    This is the only shape that separates the rules, and it needs the break to be *overlap-repeated*:
    contig 1 ends with ``feat[:35]`` and contig 2 starts with ``feat[5:]``, so every k-mer survives
    while the 40 bp feature appears nowhere as a substring. Aho-Corasick would call this absent.
    """
    feat = _rand(rng, 40)
    frag = {"c1": _rand(rng, 500) + feat[:35], "c2": feat[5:] + _rand(rng, 500)}
    flat, offsets = feature_kmer_table([feat])
    assert contains_all(genome_kmer_index(frag), flat, offsets)[0]
    assert all(feat not in c and _revcomp(feat) not in c for c in frag.values())


def test_clean_contig_break_is_absent_under_both_rules(rng):
    """A non-overlapping break destroys the spanning k-mers, so containment agrees with substring."""
    feat = _rand(rng, 40)
    clean = {"c1": _rand(rng, 500) + feat[:20], "c2": feat[20:] + _rand(rng, 500)}
    flat, offsets = feature_kmer_table([feat])
    assert not contains_all(genome_kmer_index(clean), flat, offsets)[0]


def test_ambiguity_codes_and_short_sequences_yield_no_kmers():
    """GGCAT never emits a k-mer spanning an N, and a sequence shorter than k has none to emit."""
    assert canonical_kmers("N" * 40).size == 0
    assert canonical_kmers("ACGT" * 5).size == 0
    assert canonical_kmers("ACGT" * 8).size == 32 - K + 1


def test_untestable_feature_scores_absent_rather_than_true(rng):
    """``logical_and.reduceat`` over an empty span returns True — a sub-k feature must not ride that."""
    genome = {"c1": _rand(rng, 5000)}
    flat, offsets = feature_kmer_table(["ACGT", genome["c1"][100:140]])
    assert contains_all(genome_kmer_index(genome), flat, offsets).tolist() == [False, True]


def test_empty_genome_scores_everything_absent(rng):
    """A missing or unreadable assembly must be all-zero, not silently confident."""
    flat, offsets = feature_kmer_table([_rand(rng, 40), _rand(rng, 40)])
    assert not contains_all(genome_kmer_index({}), flat, offsets).any()
