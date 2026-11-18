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
    assert out.loc["determinant", "delta_z"] > 0
    assert out.loc["boring1", "p_adj"] > 0.05


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
    out = resistant_vs_susceptible(tiny, min_per_class=5).set_index("gene_name")
    assert out.loc["determinant", "delta_z"] > 0
    # The inert gene is dragged the other way — that is the compositional artifact, not a signal.
    assert out.loc["boring1", "delta_z"] < 0

    realistic = normalise_within_genome(_cohort(120, determinant_effect=2.5, seed=1, n_background=400))
    big = resistant_vs_susceptible(realistic, min_per_class=5).set_index("gene_name")
    assert abs(big.loc["boring1", "delta_z"]) < abs(out.loc["boring1", "delta_z"])


def test_unnamed_share_is_reported() -> None:
    """The share of attribution in unnameable genes is what only a gene-family method can reach."""
    share = report_unnamed_share(normalise_within_genome(_cohort(30)))
    assert 0.0 < share < 1.0


def test_empty_result_is_returned_not_raised() -> None:
    """Too few genomes is a legitimate outcome and must not crash the stage."""
    frame = normalise_within_genome(_cohort(6))
    assert rank_genes(frame, min_genomes=50).empty
    assert resistant_vs_susceptible(frame, min_per_class=50).empty
