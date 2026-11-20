"""Unit tests for the per-gene attribution statistics and the lineage control.

The lineage control is the test that decides what the whole analysis is allowed to claim, so the
case that matters most here is the adversarial one: a gene that looks like a determinant only
because it marks a lineage in which every genome happens to be resistant.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pandas")
pytest.importorskip("scipy")
pytest.importorskip("statsmodels")
import numpy as np
import pandas as pd

from bacpredict.engine.explain.attribution_statistics import (
    collapse_gene_copies,
    normalise_within_genome,
    rank_genes,
    report_unnamed_share,
    resistant_vs_susceptible,
    summarise_lineage_control,
)


def _cohort(
    n_genomes: int = 60,
    *,
    determinant_effect: float = 0.0,
    lineage_marker: bool = False,
    seed: int = 0,
    n_background: int = 200,
) -> pd.DataFrame:
    """Build a synthetic cohort of 4 named genes plus background, with a known ground truth.

    ``determinant`` gets ``determinant_effect`` extra attribution in resistant genomes regardless of
    lineage. ``marker`` gets a boost in lineage A only — and lineage A is entirely resistant, so
    pooled it mimics a determinant exactly and only stratification tells them apart.

    ``n_background`` defaults to 200 because the within-genome z-score is compositional: one gene
    taking more attribution lowers every other gene's z in that genome. The size of that artifact
    goes as 1/n_genes, so a realistic genome (~5,390 genes) is unaffected, while a toy genome with a
    dozen genes shows it strongly. See ``test_normalisation_is_compositional_at_small_gene_counts``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_genomes):
        lineage = "A" if i % 2 == 0 else "B"
        # Lineage A is all resistant; lineage B is mixed. This is the real confounding structure.
        label = 1 if lineage == "A" else int(i % 4 == 1)
        for gene in ("determinant", "marker", "boring1", "boring2"):
            value = rng.normal(0.0, 1.0)
            if gene == "determinant" and label == 1:
                value += determinant_effect
            if gene == "marker" and lineage_marker and lineage == "A":
                value += 3.0
            rows.append({"Sample": f"S{i:03d}", "gene_name": gene, "is_named_gene": True,
                         "attribution_logit": value, "ast_label": label, "lineage": lineage})
        for j in range(n_background):  # background so the within-genome z-score is well defined
            rows.append({"Sample": f"S{i:03d}", "gene_name": f"LOCUS_{i}_{j}", "is_named_gene": False,
                         "attribution_logit": rng.normal(0.0, 1.0), "ast_label": label, "lineage": lineage})
    return pd.DataFrame(rows)


def test_normalisation_is_per_genome_and_flags_flat_genomes() -> None:
    """A genome with no spread carries no information and must be flagged, not zero-filled."""
    frame = pd.DataFrame({
        "Sample": ["A", "A", "A", "B", "B", "B"],
        "attribution_logit": [1.0, 2.0, 3.0, 5.0, 5.0, 5.0],
    })
    out = normalise_within_genome(frame)
    assert out.loc[out["Sample"] == "A", "genome_has_spread"].all()
    assert not out.loc[out["Sample"] == "B", "genome_has_spread"].any()
    assert out.loc[out["Sample"] == "B", "z_abs"].isna().all()
    # Genome A's z-scores are its own, independent of genome B's much larger raw values.
    assert out.loc[out["Sample"] == "A", "z_abs"].mean() == pytest.approx(0.0, abs=1e-9)


def test_rank_genes_drops_genes_below_the_occurrence_floor() -> None:
    """A gene seen once must not top the table; the published implementation ranked one at z=75.8."""
    frame = normalise_within_genome(_cohort(40))
    extra = frame.iloc[[0]].copy()
    extra["gene_name"] = "singleton"
    extra["z_abs"] = 99.0
    ranked = rank_genes(pd.concat([frame, extra], ignore_index=True), min_genomes=20)
    assert "singleton" not in set(ranked["gene_name"])
    assert set(ranked["gene_name"]) <= {"determinant", "marker", "boring1", "boring2"}


def test_resistant_vs_susceptible_detects_a_real_effect() -> None:
    """A gene genuinely read more in resistant genomes must separate, and boring genes must not."""
    frame = normalise_within_genome(_cohort(120, determinant_effect=2.5, seed=1))
    out = resistant_vs_susceptible(frame, min_per_class=5).set_index("gene_name")
    assert out.loc["determinant", "p_adj"] < 0.05
    assert out.loc["determinant", "delta"] > 0
    assert out.loc["boring1", "p_adj"] > 0.05
    assert (out["value_column"] == "signed_score").all(), "the signed score is the default contrast"


def test_lineage_control_separates_a_determinant_from_a_lineage_marker() -> None:
    """The case the whole analysis turns on.

    ``marker`` is boosted only in lineage A, and lineage A is entirely resistant, so pooled it looks
    exactly like a determinant. Only the within-lineage repeat can tell them apart: the determinant
    keeps its effect inside the mixed lineage, the marker has no within-lineage contrast at all.
    """
    frame = normalise_within_genome(_cohort(160, determinant_effect=2.5, lineage_marker=True, seed=2))
    overall = resistant_vs_susceptible(frame, min_per_class=5)
    within = resistant_vs_susceptible(frame, min_per_class=5, stratum="lineage")
    control = summarise_lineage_control(overall, within, "lineage").set_index("gene_name")

    pooled = overall.set_index("gene_name")
    # Pooled, both look significant — which is exactly why pooled evidence is not enough.
    assert pooled.loc["determinant", "p_adj"] < 0.05
    assert pooled.loc["marker", "p_adj"] < 0.05
    # Stratified, only the determinant survives.
    assert bool(control.loc["determinant", "survives_stratification"])
    assert not bool(control.loc["marker", "survives_stratification"])


def test_normalisation_is_compositional_at_small_gene_counts() -> None:
    """Document a real property of the primary statistic, so it is never rediscovered as a bug.

    Normalising within a genome makes gene scores compositional: a gene that takes a large share of
    the attribution raises that genome's mean and sd, which pushes every other gene's z down. With a
    handful of genes per genome the artifact is large enough to make an inert gene look significant
    in the *opposite* direction to the true determinant. It shrinks as 1/n_genes, so at the ~5,390
    genes of a real Kp genome it is negligible — but a reader must know the effect exists and which
    way it points.
    """
    tiny = normalise_within_genome(_cohort(120, determinant_effect=2.5, seed=1, n_background=8))
    out = resistant_vs_susceptible(tiny, value_column="z_abs", min_per_class=5).set_index("gene_name")
    assert out.loc["determinant", "delta"] > 0
    # The inert gene is dragged the other way — that is the compositional artifact, not a signal.
    assert out.loc["boring1", "delta"] < 0

    realistic = normalise_within_genome(_cohort(120, determinant_effect=2.5, seed=1, n_background=400))
    big = resistant_vs_susceptible(realistic, value_column="z_abs", min_per_class=5).set_index("gene_name")
    assert abs(big.loc["boring1", "delta"]) < abs(out.loc["boring1", "delta"])



def test_unnamed_share_is_reported() -> None:
    """The share of attribution in unnameable genes is what only a gene-family method can reach."""
    share = report_unnamed_share(normalise_within_genome(_cohort(30)))
    assert 0.0 < share < 1.0


def test_empty_result_is_returned_not_raised() -> None:
    """Too few genomes is a legitimate outcome and must not crash the stage."""
    frame = normalise_within_genome(_cohort(6))
    assert rank_genes(frame, min_genomes=50).empty
    assert resistant_vs_susceptible(frame, min_per_class=50).empty


def _wild_type_negative_cohort(n_genomes: int = 60, seed: int = 7, n_background: int = 200) -> pd.DataFrame:
    """A toy of the measured ``rpoB`` pattern: wild-type strongly negative, mutant mildly positive.

    On the TB rifampicin cohort the gene's signed attribution was median -10.10 in genomes with no
    ``rpoB`` variant and +3.85 in those carrying one. Magnitude therefore runs the *opposite* way to
    the phenotype, which is the whole reason the contrast is computed on the signed score.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_genomes):
        label = int(i % 2 == 0)
        lineage = "A" if i % 4 < 2 else "B"  # both lineages mixed, so both are testable
        rows.append({"Sample": f"G{i:03d}", "gene_name": "rpoB_like", "is_named_gene": True,
                     "attribution_logit": (4.0 if label else -10.0) + rng.normal(0.0, 0.5),
                     "ast_label": label, "lineage": lineage})
        rows.append({"Sample": f"G{i:03d}", "gene_name": "inert", "is_named_gene": True,
                     "attribution_logit": rng.normal(0.0, 1.0), "ast_label": label, "lineage": lineage})
        for j in range(n_background):
            rows.append({"Sample": f"G{i:03d}", "gene_name": f"LOCUS_{i}_{j}", "is_named_gene": False,
                         "attribution_logit": rng.normal(0.0, 1.0),
                         "ast_label": label, "lineage": lineage})
    return pd.DataFrame(rows)


def test_absolute_value_inverts_the_contrast_the_signed_score_recovers() -> None:
    """The measured failure, pinned as a regression: |.| does not merely weaken this, it reverses it.

    A determinant whose wild-type form carries a large negative attribution and whose mutant form
    carries a smaller positive one separates cleanly on the signed score and **confidently
    backwards** on the absolute value. A reader of the ``|.|`` table alone would conclude the model
    reads the gene more in susceptible genomes.
    """
    frame = normalise_within_genome(_wild_type_negative_cohort())

    signed = resistant_vs_susceptible(frame, min_per_class=5).set_index("gene_name")
    assert signed.loc["rpoB_like", "delta"] > 0
    assert signed.loc["rpoB_like", "p_adj"] < 1e-6
    assert bool(signed.loc["rpoB_like", "disjoint"]), "the two classes do not overlap at all"
    assert signed.loc["inert", "p_adj"] > 0.05

    magnitude = resistant_vs_susceptible(frame, value_column="z_abs", min_per_class=5).set_index("gene_name")
    assert magnitude.loc["rpoB_like", "delta"] < 0, "this is the inversion, not a weaker effect"
    assert magnitude.loc["rpoB_like", "p_adj"] < 1e-6, "and it is reported with full confidence"


def test_signed_contrast_survives_stratification_where_the_magnitude_one_cannot() -> None:
    """The lineage control must be read on the statistic that carries the signal."""
    frame = normalise_within_genome(_wild_type_negative_cohort())
    overall = resistant_vs_susceptible(frame, min_per_class=5)
    within = resistant_vs_susceptible(frame, min_per_class=5, stratum="lineage")
    control = summarise_lineage_control(overall, within, "lineage").set_index("gene_name")

    assert control.loc["rpoB_like", "n_strata_tested"] == 2
    assert control.loc["rpoB_like", "n_strata_significant"] == 2
    assert bool(control.loc["rpoB_like", "survives_stratification"])
    assert control.loc["rpoB_like", "n_strata_disjoint"] == 2


def _multi_copy_frame() -> pd.DataFrame:
    """30 genomes where ``multi`` is annotated three times and ``single`` once."""
    rng = np.random.default_rng(3)
    rows = []
    for i in range(30):
        label = int(i % 2 == 0)
        for copy_value in (-5.0, 2.0, 1.0):
            rows.append({"Sample": f"M{i:02d}", "gene_name": "multi", "is_named_gene": True,
                         "attribution_logit": copy_value + rng.normal(0.0, 0.01), "ast_label": label})
        rows.append({"Sample": f"M{i:02d}", "gene_name": "single", "is_named_gene": True,
                     "attribution_logit": rng.normal(0.0, 1.0), "ast_label": label})
        for j in range(100):
            rows.append({"Sample": f"M{i:02d}", "gene_name": f"LOCUS_{i}_{j}", "is_named_gene": False,
                         "attribution_logit": rng.normal(0.0, 1.0), "ast_label": label})
    return pd.DataFrame(rows)


def test_multi_copy_genes_collapse_to_one_value_per_genome() -> None:
    """``n_genomes`` must count genomes. Counting rows made ``rpoC`` read 190 across 98 genomes."""
    frame = normalise_within_genome(_multi_copy_frame())
    ranked = rank_genes(frame, min_genomes=20).set_index("gene_name")
    assert ranked.loc["multi", "n_genomes"] == 30, "three copies in one genome are one observation"
    assert ranked.loc["multi", "median_n_copies"] == 3.0
    assert ranked.loc["single", "n_genomes"] == 30


def test_collapse_sums_copies_by_default_and_max_keeps_the_strongest_copys_sign() -> None:
    """Summing is the IG-additive choice; ``max`` must not return |.| stripped of its sign."""
    frame = normalise_within_genome(_multi_copy_frame())
    named = frame[frame["is_named_gene"]]

    summed = collapse_gene_copies(named, how="sum")
    multi = summed[summed["gene_name"] == "multi"]
    assert len(multi) == 30
    assert (multi["n_copies"] == 3).all()
    # -5 + 2 + 1 = -2, the gene's total contribution to the logit.
    assert multi["signed_score"].mean() == pytest.approx(-2.0, abs=0.05)

    peak = collapse_gene_copies(named, how="max")
    multi_peak = peak[peak["gene_name"] == "multi"]
    assert multi_peak["abs_score"].mean() == pytest.approx(5.0, abs=0.05)
    assert multi_peak["signed_score"].mean() == pytest.approx(-5.0, abs=0.05), "sign must be kept"
