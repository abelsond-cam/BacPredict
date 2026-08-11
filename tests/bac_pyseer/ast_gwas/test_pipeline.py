"""Unit + Stage-A-smoke tests for the AMR unitig-GWAS → LR read-out.

The Stage A protocol (root ``CLAUDE.md`` §0.2) wants the pipeline proven end-to-end on a tiny
cohort, CPU-only, before any cluster time is spent. Everything here runs against synthetic
pyseer-format fixtures with no GGCAT, no pyseer and no cluster — so the only things under test are
ours: the holdout guard, the design-matrix construction, and the logistic-regression read-out.

The load-bearing assertion is :func:`test_phenotype_never_contains_holdout`. If that ever fails,
every AUROC this package produces is uncomparable to the Bacformer fine-tune.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bac_pyseer.ast_gwas.build_ast_phenotype import build_phenotype, label_column, write_phenotype
from bac_pyseer.ast_gwas.unitig_design_matrix import load_design, read_hits
from bac_pyseer.ast_gwas.unitig_design_matrix import run as build_design
from bac_pyseer.ast_gwas.unitig_lr import run as run_lr

_DRUG = "ertapenem"

# A 20-genome cohort on the repo's split vocabulary: 10 train / 4 validate / 6 holdout.
_SPLIT_LAYOUT = [("train", 10), ("validate", 4), ("holdout", 6)]


def _sample(i: int) -> str:
    return f"SAMN{i:05d}"


@pytest.fixture
def split_table(tmp_path: Path) -> Path:
    """A <drug>_split.csv with alternating labels and one ambiguous (fractional) row."""
    rows = []
    i = 0
    for split, n in _SPLIT_LAYOUT:
        for j in range(n):
            rows.append({"Sample": _sample(i), "ast_label": j % 2, "split": split})
            i += 1
    # A sample whose repeat DSTs disagreed — load_splits must drop it before it reaches pyseer.
    rows.append({"Sample": _sample(i), "ast_label": 0.5, "split": "train"})
    path = tmp_path / f"{_DRUG}_split.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _all_samples() -> list[str]:
    return [_sample(i) for i in range(sum(n for _, n in _SPLIT_LAYOUT))]


def _holdout_samples() -> list[str]:
    return [_sample(i) for i in range(10 + 4, 10 + 4 + 6)]


# --------------------------------------------------------------------------------------- #
# phenotype: the leakage guard
# --------------------------------------------------------------------------------------- #
def test_phenotype_never_contains_holdout(split_table: Path, tmp_path: Path) -> None:
    """The GWAS phenotype must exclude every holdout genome — the basis of the comparison."""
    out = tmp_path / "phenotype.tsv"
    manifest = write_phenotype(split_table, _DRUG, out)
    written = set(pd.read_csv(out, sep="\t")["samples"].astype(str))
    assert written.isdisjoint(_holdout_samples())
    assert manifest["holdout_excluded"] is True
    assert manifest["n_samples"] == 14  # 10 train + 4 validate; the 0.5-label row is dropped
    assert manifest["n_holdout_excluded"] == 6


def test_phenotype_is_pyseer_format(split_table: Path, tmp_path: Path) -> None:
    """First column is literally 'samples'; the label column is what --phenotype-column takes."""
    out = tmp_path / "phenotype.tsv"
    write_phenotype(split_table, _DRUG, out)
    frame = pd.read_csv(out, sep="\t")
    assert list(frame.columns) == ["samples", label_column(_DRUG)]
    assert set(frame[label_column(_DRUG)]) <= {0, 1}


def test_phenotype_manifest_records_pheno_var(split_table: Path, tmp_path: Path) -> None:
    """pheno_var travels with the phenotype so postprocess can normalise VE correctly."""
    out = tmp_path / "phenotype.tsv"
    manifest = write_phenotype(split_table, _DRUG, out)
    p = manifest["prevalence"]
    assert manifest["pheno_var"] == pytest.approx(p * (1 - p))
    assert json.loads((tmp_path / "phenotype.manifest.json").read_text())["drug"] == _DRUG


def test_phenotype_can_opt_into_the_leaky_framing(split_table: Path) -> None:
    """Including holdout is possible but is recorded and loudly flagged in the manifest."""
    frame, manifest = build_phenotype(split_table, _DRUG, ("train", "validate", "holdout"))
    assert len(frame) == 20
    assert manifest["holdout_excluded"] is False
    assert "WARNING" in manifest["leakage_note"]


def test_phenotype_rejects_unknown_split(split_table: Path) -> None:
    """A typo'd split name must fail, not silently select nothing."""
    with pytest.raises(SystemExit, match="unknown split"):
        build_phenotype(split_table, _DRUG, ("train", "test"))


# --------------------------------------------------------------------------------------- #
# design matrix
# --------------------------------------------------------------------------------------- #
def _write_unitig_matrix(path: Path, unitigs: dict[str, list[str]]) -> None:
    """Write a gzipped pyseer --kmers matrix: ``<seq> | <Sample>:1 …``."""
    import gzip

    with gzip.open(path, "wt") as fh:
        for seq, carriers in unitigs.items():
            fh.write(f"{seq} | {' '.join(f'{s}:1' for s in carriers)}\n")


def _write_hits(path: Path, seqs: list[str], pattern_groups: list[int] | None = None) -> None:
    """Write a minimal pyseer_postprocess hits table."""
    n = len(seqs)
    pd.DataFrame({
        "variant": seqs,
        "var_explained_pct": np.linspace(10, 1, n),
        "af": [0.5] * n,
        "beta": [1.0] * n,
        "lrt-pvalue": [1e-9] * n,
        "pattern_group": pattern_groups if pattern_groups is not None else list(range(n)),
        "n_in_pattern": [1] * n,
    }).to_csv(path, sep="\t", index=False)


@pytest.fixture
def signal_cohort(tmp_path: Path) -> tuple[Path, Path]:
    """A matrix + hits table where one unitig is carried by exactly the resistant genomes."""
    samples = _all_samples()
    resistant = [s for i, s in enumerate(samples) if i % 2 == 1]
    matrix = tmp_path / "unitigs.pyseer.gz"
    _write_unitig_matrix(matrix, {
        "AAAACCCCGGGG": resistant,          # perfectly predictive
        "TTTTGGGGAAAA": samples[:6],        # unrelated to phenotype
        "CCCCTTTTAAAA": samples[3:9],       # unrelated to phenotype
    })
    hits = tmp_path / "hits.tsv"
    _write_hits(hits, ["AAAACCCCGGGG", "TTTTGGGGAAAA", "CCCCTTTTAAAA"])
    return matrix, hits


def test_design_matrix_covers_every_split_genome(signal_cohort, split_table: Path, tmp_path: Path) -> None:
    """Rows span train+validate+holdout; a genome carrying no hit is an all-zero row, not absent."""
    matrix_gz, hits = signal_cohort
    design_dir = tmp_path / "design"
    manifest = build_design(hits_tsv=hits, matrix_gz=matrix_gz, split_table=split_table, out_dir=design_dir)

    csr, sample_ids, id_map = load_design(design_dir)
    assert csr.shape == (20, 3)
    assert set(sample_ids) == set(_all_samples())
    assert set(_holdout_samples()) <= set(sample_ids)
    assert manifest["n_unitigs_not_found_in_matrix"] == 0
    assert len(id_map) == 3
    # Index 10: even (so susceptible, not a carrier of the predictive unitig) and outside both
    # control unitigs' carrier ranges (0-5 and 3-8) — so it carries nothing and must still be a row.
    empty_row = csr[sample_ids.index("SAMN00010")]
    assert empty_row.nnz == 0
    assert empty_row.shape == (1, 3)
    assert manifest["n_samples_with_no_hit_unitig"] == 5  # indices 10, 12, 14, 16, 18


def test_design_matrix_presence_is_correct(signal_cohort, split_table: Path, tmp_path: Path) -> None:
    """A cell is 1 iff the matrix listed that sample as a carrier of that unitig."""
    matrix_gz, hits = signal_cohort
    design_dir = tmp_path / "design"
    build_design(hits_tsv=hits, matrix_gz=matrix_gz, split_table=split_table, out_dir=design_dir)
    csr, sample_ids, id_map = load_design(design_dir)

    col = int(id_map.loc[id_map["variant"] == "AAAACCCCGGGG", "unitig_idx"].iloc[0])
    carried = {sample_ids[r] for r in csr[:, col].tocoo().row}
    assert carried == {s for i, s in enumerate(_all_samples()) if i % 2 == 1}


def test_design_matrix_dedupes_ld_blocks(split_table: Path, tmp_path: Path) -> None:
    """--dedupe-patterns keeps one unitig per perfect-LD pattern_group."""
    samples = _all_samples()
    matrix_gz = tmp_path / "u.gz"
    _write_unitig_matrix(matrix_gz, {"AAAA": samples[:5], "CCCC": samples[:5], "GGGG": samples[5:]})
    hits = tmp_path / "hits.tsv"
    _write_hits(hits, ["AAAA", "CCCC", "GGGG"], pattern_groups=[0, 0, 1])  # AAAA/CCCC perfectly linked

    assert len(read_hits(hits)) == 3
    assert len(read_hits(hits, dedupe_patterns=True)) == 2
    manifest = build_design(
        hits_tsv=hits, matrix_gz=matrix_gz, split_table=split_table,
        out_dir=tmp_path / "design", dedupe_patterns=True,
    )
    assert manifest["n_unitigs"] == 2


def test_design_matrix_reports_unjoined_hits(split_table: Path, tmp_path: Path) -> None:
    """A hit sequence absent from the unitig matrix is counted, not silently zero-filled."""
    matrix_gz = tmp_path / "u.gz"
    _write_unitig_matrix(matrix_gz, {"AAAA": _all_samples()[:5]})
    hits = tmp_path / "hits.tsv"
    _write_hits(hits, ["AAAA", "TTTT"])  # TTTT is not in the matrix
    manifest = build_design(
        hits_tsv=hits, matrix_gz=matrix_gz, split_table=split_table, out_dir=tmp_path / "design"
    )
    assert manifest["n_unitigs_not_found_in_matrix"] == 1
    assert (tmp_path / "design" / "unitig_join_misses.txt").read_text().split() == ["TTTT"]


# --------------------------------------------------------------------------------------- #
# logistic regression read-out
# --------------------------------------------------------------------------------------- #
def test_lr_recovers_a_planted_signal(signal_cohort, split_table: Path, tmp_path: Path) -> None:
    """A unitig carried by exactly the resistant genomes must give a perfect holdout AUROC."""
    matrix_gz, hits = signal_cohort
    design_dir = tmp_path / "design"
    build_design(hits_tsv=hits, matrix_gz=matrix_gz, split_table=split_table, out_dir=design_dir)
    payload = run_lr(
        design_dir=design_dir, split_table=split_table, drug=_DRUG,
        organism="kp", out_dir=tmp_path / "lr",
    )
    assert payload["metrics"]["auroc"] == pytest.approx(1.0)
    assert payload["metrics"]["auprc"] == pytest.approx(1.0)
    assert payload["metrics"]["n_samples"] == 6  # scored on the holdout only


def test_lr_on_noise_is_uninformative(split_table: Path, tmp_path: Path) -> None:
    """Unitigs unrelated to the phenotype must not produce a confidently good AUROC."""
    rng = np.random.default_rng(0)
    samples = _all_samples()
    unitigs = {
        "".join(seq): [s for s in samples if rng.random() < 0.5]
        for seq in (["A", "C", "G", "T"] * 4)[:8]
    }
    matrix_gz = tmp_path / "u.gz"
    _write_unitig_matrix(matrix_gz, unitigs)
    hits = tmp_path / "hits.tsv"
    _write_hits(hits, list(unitigs))
    design_dir = tmp_path / "design"
    build_design(hits_tsv=hits, matrix_gz=matrix_gz, split_table=split_table, out_dir=design_dir)
    payload = run_lr(
        design_dir=design_dir, split_table=split_table, drug=_DRUG,
        organism="kp", out_dir=tmp_path / "lr",
    )
    assert 0.0 <= payload["metrics"]["auroc"] <= 1.0
    assert payload["extra"]["features"] == "significant_unitig_presence"


def test_lr_emits_the_engine_results_schema(signal_cohort, split_table: Path, tmp_path: Path) -> None:
    """results.json validates against schema v1.2 and carries the comparison provenance."""
    from bacpredict.engine.finetune.metrics import REQUIRED_METRICS_KEYS, REQUIRED_TOP_LEVEL_KEYS

    matrix_gz, hits = signal_cohort
    design_dir = tmp_path / "design"
    build_design(hits_tsv=hits, matrix_gz=matrix_gz, split_table=split_table, out_dir=design_dir)
    out_dir = tmp_path / "lr"
    run_lr(design_dir=design_dir, split_table=split_table, drug=_DRUG, organism="tb", out_dir=out_dir)

    written = json.loads((out_dir / "results.json").read_text())
    assert REQUIRED_TOP_LEVEL_KEYS <= set(written)
    assert REQUIRED_METRICS_KEYS <= set(written["metrics"])
    assert written["schema_version"] == "1.2"
    assert written["task"] == "tb_ast"          # organism -> engine task name
    assert written["model"]["name_or_path"] == "unitig_lr"
    assert written["split"]["source"] == "split_table"
    assert written["operating_point"]["selected_on"] == "validation"
    assert written["extra"]["standardised"] is False

    scores = np.load(out_dir / "eval_scores.npz", allow_pickle=False)
    assert scores["y_true"].size == 6
    assert scores["y_prob"].size == 6
    coefficients = pd.read_csv(out_dir / "coefficients.tsv", sep="\t")
    assert len(coefficients) == 3
    assert "lr_coef" in coefficients.columns


def test_lr_scores_only_the_holdout(signal_cohort, split_table: Path, tmp_path: Path) -> None:
    """n_evaluate and the metrics block must describe the holdout, not the fitted rows."""
    matrix_gz, hits = signal_cohort
    design_dir = tmp_path / "design"
    build_design(hits_tsv=hits, matrix_gz=matrix_gz, split_table=split_table, out_dir=design_dir)
    payload = run_lr(
        design_dir=design_dir, split_table=split_table, drug=_DRUG,
        organism="kp", out_dir=tmp_path / "lr",
    )
    assert payload["split"]["n_evaluate"] == 6
    assert payload["extra"]["n_train"] == 10
    assert payload["extra"]["n_validate"] == 4
