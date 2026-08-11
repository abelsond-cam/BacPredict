"""Tests for the per-drug kinship subset and the final comparison table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bac_pyseer.ast_gwas.collect_comparison import collect, read_unitig_results
from bac_pyseer.ast_gwas.mash_kinship import distances_for_samples, similarity_for_samples
from bac_pyseer.ast_gwas.mash_kinship import run as kinship_run


def _write_triangle(path: Path, names: list[str], d: np.ndarray) -> None:
    """Write a mash-triangle lower-triangular PHYLIP file."""
    lines = [f"\t{len(names)}"]
    for i, name in enumerate(names):
        lines.append("\t".join([name, *(f"{d[i, j]:.6f}" for j in range(i))]))
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def triangle(tmp_path: Path) -> Path:
    """A 4-genome mash triangle with distinguishable distances."""
    names = ["A", "B", "C", "D"]
    d = np.array([
        [0.00, 0.01, 0.20, 0.21],
        [0.01, 0.00, 0.19, 0.22],
        [0.20, 0.19, 0.00, 0.02],
        [0.21, 0.22, 0.02, 0.00],
    ])
    path = tmp_path / "mash_triangle.txt"
    _write_triangle(path, names, d)
    return path


def test_similarity_is_one_minus_distance(triangle: Path) -> None:
    """pyseer --similarity takes S = 1 - D, with a unit diagonal."""
    sim = similarity_for_samples(triangle, ["A", "B", "C", "D"])
    assert sim.loc["A", "B"] == pytest.approx(0.99)
    assert sim.loc["A", "C"] == pytest.approx(0.80)
    assert np.allclose(np.diag(sim.to_numpy()), 1.0)


def test_similarity_subsets_to_the_phenotyped_samples(triangle: Path) -> None:
    """The cohort-wide triangle is cut per drug — this is what keeps TB's n^2 tractable."""
    sim = similarity_for_samples(triangle, ["A", "C"])
    assert list(sim.index) == ["A", "C"]
    assert sim.shape == (2, 2)
    assert sim.loc["A", "C"] == pytest.approx(0.80)


def test_similarity_preserves_requested_order(triangle: Path) -> None:
    """Row/column order follows the request, not the triangle — pyseer joins by label, but a
    silently reordered matrix would be very hard to debug if that ever changed."""
    sim = similarity_for_samples(triangle, ["D", "A"])
    assert list(sim.index) == list(sim.columns) == ["D", "A"]
    assert sim.loc["D", "A"] == pytest.approx(0.79)


def test_missing_sample_is_a_hard_error(triangle: Path) -> None:
    """A phenotyped genome absent from the kinship would be silently dropped by pyseer."""
    with pytest.raises(SystemExit, match="absent from"):
        similarity_for_samples(triangle, ["A", "ZZZ"])


def test_distances_are_the_raw_matrix_not_the_similarity(triangle: Path) -> None:
    """pyseer --lineage takes distances, --lmm takes similarity; they must not be confused."""
    dist = distances_for_samples(triangle, ["A", "C"])
    sim = similarity_for_samples(triangle, ["A", "C"])
    assert dist.loc["A", "C"] == pytest.approx(0.20)
    assert dist.loc["A", "C"] == pytest.approx(1 - sim.loc["A", "C"])
    assert np.allclose(np.diag(dist.to_numpy()), 0.0)


def test_kinship_run_writes_both_matrices_over_the_same_samples(triangle: Path, tmp_path: Path) -> None:
    """One call emits the pair pyseer needs, guaranteed to cover identical genomes in the same order."""
    pheno = tmp_path / "phenotype.tsv"
    pd.DataFrame({"samples": ["A", "C"], "ertapenem_label": [1, 0]}).to_csv(pheno, sep="\t", index=False)
    manifest = kinship_run(
        triangle_path=triangle, out_tsv=tmp_path / "similarity.tsv",
        phenotype_tsv=pheno, distances_tsv=tmp_path / "distances.tsv",
    )
    sim = pd.read_csv(tmp_path / "similarity.tsv", sep="\t", index_col=0)
    dist = pd.read_csv(tmp_path / "distances.tsv", sep="\t", index_col=0)
    assert list(sim.index) == list(dist.index) == ["A", "C"]
    assert manifest["distances_output"].endswith("distances.tsv")


def test_kinship_run_uses_the_phenotype_sample_list(triangle: Path, tmp_path: Path) -> None:
    """Driving the subset off the phenotype TSV guarantees kinship and GWAS cover the same genomes."""
    pheno = tmp_path / "phenotype.tsv"
    pd.DataFrame({"samples": ["A", "B", "C"], "ertapenem_label": [1, 0, 1]}).to_csv(
        pheno, sep="\t", index=False
    )
    out = tmp_path / "similarity.tsv"
    manifest = kinship_run(triangle_path=triangle, out_tsv=out, phenotype_tsv=pheno)

    assert manifest["n_samples"] == 3
    written = pd.read_csv(out, sep="\t", index_col=0)
    assert list(written.index) == ["A", "B", "C"]
    assert json.loads((tmp_path / "similarity.manifest.json").read_text())["n_samples"] == 3


# --------------------------------------------------------------------------------------- #
# comparison table
# --------------------------------------------------------------------------------------- #
def _results_json(path: Path, drug: str, auroc: float, auprc: float) -> Path:
    """A minimal unitig-LR results.json, as unitig_lr.run would write."""
    path.write_text(json.dumps({
        "schema_version": "1.2", "task": "kleb_ast", "drug": drug,
        "model": {"name_or_path": "unitig_lr"},
        "split": {"source": "split_table", "n_evaluate": 418},
        "metrics": {
            "auroc": auroc, "auprc": auprc, "sensitivity": 0.9, "specificity": 0.9,
            "balanced_accuracy": 0.9,
        },
        "extra": {
            "n_unitigs": 1200, "n_train": 1500,
            "gwas_summary": {
                "n_unique_patterns": 500_000, "bonferroni_threshold": 1e-7,
                "genomic_inflation_lambda": 1.05, "pheno_var": 0.232,
            },
        },
        "timestamp": "2026-08-11T00:00:00+00:00", "host": "test",
    }))
    return path


def test_read_unitig_results_flattens_gwas_provenance(tmp_path: Path) -> None:
    """λ, the pattern count and pheno_var travel with the AUROC so a row is self-describing."""
    row = read_unitig_results(_results_json(tmp_path / "r.json", "ertapenem", 0.985, 0.982))
    assert row["drug"] == "ertapenem"
    assert row["unitig_auroc"] == 0.985
    assert row["n_unique_patterns"] == 500_000
    assert row["lambda_gc"] == 1.05
    assert row["pheno_var"] == 0.232


def test_collect_joins_ft_and_ceiling_and_computes_deltas(tmp_path: Path) -> None:
    """The output is the head-to-head table: unitig-LR vs fine-tune vs catalogue ceiling."""
    results = [
        _results_json(tmp_path / "ert.json", "ertapenem", 0.985, 0.982),
        _results_json(tmp_path / "col.json", "colistin", 0.860, 0.720),
    ]
    panel = tmp_path / "amr_summary_panel.csv"
    pd.DataFrame({
        "drug": ["ertapenem", "colistin"],
        "ceiling_auroc": [0.97701, 0.67987], "ceiling_auprc": [0.98490, 0.49568],
        "ft_auroc": [0.9870, 0.8072], "ft_auprc": [0.9843, 0.6855],
        "concat_auroc": [1.0, 0.92536],  # known-unreliable for Kp; must be ignored
    }).to_csv(panel, index=False)

    table = collect(results, panel).set_index("drug")
    assert "concat_auroc" not in table.columns
    # Colistin is the case with real headroom: unitigs beat both the fine-tune and the catalogue.
    assert table.loc["colistin", "delta_vs_ft_auroc"] == pytest.approx(0.860 - 0.8072)
    assert table.loc["colistin", "delta_vs_ceiling_auroc"] == pytest.approx(0.860 - 0.67987)
    # Ertapenem is the saturated positive control: essentially a tie with the fine-tune.
    assert abs(table.loc["ertapenem", "delta_vs_ft_auroc"]) < 0.01


def test_collect_without_a_panel_still_emits_unitig_columns(tmp_path: Path) -> None:
    """Missing comparators must not lose the results we do have."""
    table = collect([_results_json(tmp_path / "r.json", "ertapenem", 0.985, 0.982)], None)
    assert list(table["drug"]) == ["ertapenem"]
    assert "delta_vs_ft_auroc" not in table.columns


def test_collect_rejects_an_empty_input() -> None:
    """An empty run is a mistake, not an empty table."""
    with pytest.raises(SystemExit, match="no results.json"):
        collect([], None)
