"""Tests for the per-drug kinship subset and the final comparison table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bac_pyseer.ast_gwas.collect_comparison import collect, paired_ci_against_ft, read_unitig_results
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


def test_panel_supplies_the_ceiling_only_never_the_fine_tune(tmp_path: Path) -> None:
    """The real panels carry `concat_*`, not `ft_*` — reading those as the fine-tune compares
    against the concat-ladder model instead, which is a different model entirely."""
    results = [
        _results_json(tmp_path / "ert.json", "ertapenem", 0.985, 0.982),
        _results_json(tmp_path / "col.json", "colistin", 0.860, 0.720),
    ]
    panel = tmp_path / "amr_summary_panel.csv"
    pd.DataFrame({
        "drug": ["ertapenem", "colistin"],
        "ceiling_auroc": [0.97701, 0.67987], "ceiling_auprc": [0.98490, 0.49568],
        "concat_auroc": [1.0, 0.92536],  # a DIFFERENT model — must never be read as the fine-tune
    }).to_csv(panel, index=False)

    table = collect(results, panel).set_index("drug")
    assert "concat_auroc" not in table.columns
    assert "ft_auroc" not in table.columns          # no --ft-scores given, so no fine-tune arm
    assert "delta_vs_ft_auroc" not in table.columns  # and therefore no fine-tune delta invented
    assert table.loc["colistin", "delta_vs_ceiling_auroc"] == pytest.approx(0.860 - 0.67987)


def test_rifampin_matches_the_panels_rifampicin_spelling(tmp_path: Path) -> None:
    """The AST column is `rifampin` (US), the panels key on `rifampicin` (UK). Without the alias
    the headline TB drug merges to NaN and the table looks like it simply has no ceiling."""
    results = [_results_json(tmp_path / "rif.json", "rifampin", 0.94, 0.90)]
    panel = tmp_path / "tb_panel.csv"
    pd.DataFrame({"drug": ["rifampicin"], "ceiling_auroc": [0.96658],
                  "ceiling_auprc": [0.93916]}).to_csv(panel, index=False)

    table = collect(results, panel).set_index("drug")
    assert table.loc["rifampin", "ceiling_auroc"] == pytest.approx(0.96658)
    assert table.loc["rifampin", "delta_vs_ceiling_auroc"] == pytest.approx(0.94 - 0.96658)


def test_a_drug_absent_from_a_partial_panel_is_warned_about(tmp_path: Path, caplog) -> None:
    """Kp's panel covers 7 of 22 drugs and has no ertapenem at all; a silent NaN would read as
    'no catalogue ceiling exists' rather than 'nobody added this drug to the panel'."""
    results = [_results_json(tmp_path / "ert.json", "ertapenem", 0.985, 0.982)]
    panel = tmp_path / "panel.csv"
    pd.DataFrame({"drug": ["colistin"], "ceiling_auroc": [0.67987],
                  "ceiling_auprc": [0.49568]}).to_csv(panel, index=False)

    with caplog.at_level("WARNING"):
        table = collect(results, panel)
    assert pd.isna(table.loc[0, "ceiling_auroc"])
    assert "ertapenem" in caplog.text


def test_collect_without_a_panel_still_emits_unitig_columns(tmp_path: Path) -> None:
    """Missing comparators must not lose the results we do have."""
    table = collect([_results_json(tmp_path / "r.json", "ertapenem", 0.985, 0.982)], None)
    assert list(table["drug"]) == ["ertapenem"]
    assert "delta_vs_ft_auroc" not in table.columns


def _scores_npz(path: Path, ids: list[str], y_true: list[int], y_prob: list[float]) -> Path:
    """An eval_scores.npz in the shape both comparators write."""
    np.savez(path, y_true=np.array(y_true), y_prob=np.array(y_prob),
             sample_ids=np.asarray(ids, dtype=np.str_), drug=np.array("ertapenem"),
             operating_threshold=np.array(0.5))
    return path


def test_paired_ci_pairs_by_sample_id_not_position(tmp_path: Path) -> None:
    """The two models' holdouts overlap partially and are in different orders — align by id."""
    ids = [f"S{i}" for i in range(20)]
    y = [i % 2 for i in range(20)]
    unitig = _scores_npz(tmp_path / "u.npz", ids, y, [0.9 if v else 0.1 for v in y])
    # Fine-tune covers a shifted, reordered subset — pairing by position would be wrong.
    ft_ids = list(reversed(ids[5:]))
    ft_y = [int(s[1:]) % 2 for s in ft_ids]
    ft = _scores_npz(tmp_path / "f.npz", ft_ids, ft_y, [0.8 if v else 0.2 for v in ft_y])

    ci = paired_ci_against_ft(unitig, ft, seed=1)
    assert ci is not None
    assert ci["n_common_genomes"] == 15
    # Both models separate the classes perfectly on the common set, so the delta is exactly 0.
    assert ci["unitig_auroc_on_common"] == pytest.approx(1.0)
    assert ci["ft_auroc_on_common"] == pytest.approx(1.0)
    assert ci["delta_unitig_minus_ft"] == pytest.approx(0.0)
    assert ci["separates_from_zero"] is False


def test_paired_ci_needs_sample_ids(tmp_path: Path) -> None:
    """Without sample_ids the models could only be aligned by position — refuse rather than guess."""
    ids = [f"S{i}" for i in range(10)]
    y = [i % 2 for i in range(10)]
    unitig = _scores_npz(tmp_path / "u.npz", ids, y, [0.9 if v else 0.1 for v in y])
    legacy = tmp_path / "legacy.npz"
    np.savez(legacy, y_true=np.array(y), y_prob=np.array([0.5] * 10))  # pre-sample_ids schema
    assert paired_ci_against_ft(unitig, legacy) is None


def test_collect_adds_the_ci_columns(tmp_path: Path) -> None:
    """A drug with a fine-tune npz gains a delta and a separates_from_zero verdict."""
    lr_dir = tmp_path / "lr"
    lr_dir.mkdir()
    results = _results_json(lr_dir / "results.json", "ertapenem", 0.985, 0.982)
    ids = [f"S{i}" for i in range(20)]
    y = [i % 2 for i in range(20)]
    _scores_npz(lr_dir / "eval_scores.npz", ids, y, [0.9 if v else 0.1 for v in y])
    ft = _scores_npz(tmp_path / "ft.npz", ids, y, [0.6 if v else 0.4 for v in y])

    table = collect([results], None, ft_scores={"ertapenem": ft})
    assert table.loc[0, "n_common_genomes"] == 20
    assert "delta_unitig_minus_ft" in table.columns
    assert table.loc[0, "separates_from_zero"] in (True, False)


def test_collect_without_ft_scores_omits_the_ci(tmp_path: Path) -> None:
    """No fine-tune predictions means no CI columns — not a column of NaNs implying one was tried."""
    table = collect([_results_json(tmp_path / "r.json", "ertapenem", 0.985, 0.982)], None)
    assert "delta_unitig_minus_ft" not in table.columns


def test_collect_rejects_an_empty_input() -> None:
    """An empty run is a mistake, not an empty table."""
    with pytest.raises(SystemExit, match="no results.json"):
        collect([], None)
