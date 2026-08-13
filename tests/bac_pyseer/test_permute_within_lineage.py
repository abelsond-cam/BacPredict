"""Tests for the within-lineage permutation null, including excluding the ``other`` bucket."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bac_pyseer.kleb_iso_source.permute_phenotype_within_lineage import (
    UNASSIGNED,
    permute_within_lineage,
)

LABEL = "ertapenem_label"


def _write_pheno(path: Path, samples: list[str], labels: list[int]) -> None:
    pd.DataFrame({"samples": samples, LABEL: labels}).to_csv(path, sep="\t", index=False)


def _write_clusters(path: Path, mapping: dict[str, str]) -> None:
    path.write_text("".join(f"{s}\t{c}\n" for s, c in mapping.items()))


def test_case_count_is_preserved_within_every_cluster(tmp_path: Path) -> None:
    """The null must keep the phenotype-lineage correlation exactly, or it tests the wrong thing."""
    samples = [f"s{i}" for i in range(8)]
    labels = [1, 1, 0, 0, 1, 0, 0, 0]
    clusters = {**dict.fromkeys(samples[:4], "SL258"), **dict.fromkeys(samples[4:], "SL307")}
    pheno, cl = tmp_path / "p.tsv", tmp_path / "c.tsv"
    _write_pheno(pheno, samples, labels)
    _write_clusters(cl, clusters)

    out = permute_within_lineage(pheno, cl, LABEL, seed=1)
    merged = out.assign(grp=out["samples"].map(clusters))
    per_cluster = merged.groupby("grp")[LABEL].sum().to_dict()
    assert per_cluster == {"SL258": 2, "SL307": 1}  # identical to the input, per cluster


def test_default_behaviour_still_shuffles_other(tmp_path: Path) -> None:
    """Backward compatibility: without --exclude-cluster nothing is dropped."""
    samples = [f"s{i}" for i in range(6)]
    clusters = {**dict.fromkeys(samples[:3], "SL258"), **dict.fromkeys(samples[3:], "other")}
    pheno, cl = tmp_path / "p.tsv", tmp_path / "c.tsv"
    _write_pheno(pheno, samples, [1, 0, 0, 1, 0, 0])
    _write_clusters(cl, clusters)

    out = permute_within_lineage(pheno, cl, LABEL, seed=1)
    assert len(out) == 6


def test_exclude_cluster_drops_those_samples_entirely(tmp_path: Path) -> None:
    """'other' is a bag of unrelated sublineages; shuffling within it is near-unrestricted."""
    samples = [f"s{i}" for i in range(6)]
    clusters = {**dict.fromkeys(samples[:3], "SL258"), **dict.fromkeys(samples[3:], "other")}
    pheno, cl = tmp_path / "p.tsv", tmp_path / "c.tsv"
    _write_pheno(pheno, samples, [1, 0, 0, 1, 0, 0])
    _write_clusters(cl, clusters)

    out = permute_within_lineage(pheno, cl, LABEL, seed=1, exclude={"other"})
    assert list(out["samples"]) == ["s0", "s1", "s2"]
    assert out[LABEL].sum() == 1  # SL258's case count still preserved


def test_samples_missing_from_the_cluster_file_can_be_excluded(tmp_path: Path) -> None:
    """A sample with no cluster row is unassigned, and is excludable by that name."""
    pheno, cl = tmp_path / "p.tsv", tmp_path / "c.tsv"
    _write_pheno(pheno, ["a", "b", "c"], [1, 0, 1])
    _write_clusters(cl, {"a": "SL258", "b": "SL258"})  # c absent

    out = permute_within_lineage(pheno, cl, LABEL, seed=1, exclude={UNASSIGNED})
    assert sorted(out["samples"]) == ["a", "b"]


def test_excluding_everything_is_an_error_not_an_empty_table(tmp_path: Path) -> None:
    """An empty phenotype would fail deep inside pyseer; fail here with the reason instead."""
    pheno, cl = tmp_path / "p.tsv", tmp_path / "c.tsv"
    _write_pheno(pheno, ["a", "b"], [1, 0])
    _write_clusters(cl, {"a": "other", "b": "other"})
    with pytest.raises(SystemExit, match="removed every sample"):
        permute_within_lineage(pheno, cl, LABEL, seed=1, exclude={"other"})


def test_a_singleton_cluster_cannot_be_shuffled(tmp_path: Path) -> None:
    """A cluster of one keeps its label — worth asserting, since it silently weakens the null."""
    pheno, cl = tmp_path / "p.tsv", tmp_path / "c.tsv"
    _write_pheno(pheno, ["a", "b", "c"], [1, 0, 1])
    _write_clusters(cl, {"a": "SL1", "b": "SL2", "c": "SL2"})

    out = permute_within_lineage(pheno, cl, LABEL, seed=3)
    assert out.set_index("samples")[LABEL]["a"] == 1
