"""Stage-A smoke for the Kleborate-determinant ceiling module — synthetic data, CPU, no HPC.

Exercises the Kleborate cell tokeniser, the determinant one-hot builder, and the full ``run()`` through
the real k-fold harness, on a tiny synthetic metadata TSV + AST sheet (real metadata_v2 is HPC-only).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from bacpredict.apps.kleb.kleborate_determinant_lr import (
    ALL_KEY,
    build_determinant_onehot,
    run,
    tokenize_cell,
)


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("-", []),
        ("", []),
        ("0", []),
        (None, []),
        ("CTX-M-15", ["CTX-M-15"]),
        ("CTX-M-15*", ["CTX-M-15"]),                 # imperfect-match marker stripped
        ("CTX-M-15*-42%", ["CTX-M-15"]),             # coverage suffix stripped
        ("KPC-2;NDM-1", ["KPC-2", "NDM-1"]),         # ';'-separated classes
        ("OXA-1,OXA-1", ["OXA-1", "OXA-1"]),         # ',' multi-copy list
        ("qnrS1;-", ["qnrS1"]),                      # mixed present + absent token
    ],
)
def test_tokenize_cell(cell, expected):
    """The tokeniser splits on ;/, , drops absent sentinels, and strips quality/coverage markers."""
    assert tokenize_cell(cell) == expected


def test_build_determinant_onehot_shapes():
    """One-hot is genomes × '<column>:<token>'; susceptible genomes are all-zero rows over the universe."""
    meta = pd.DataFrame(
        {
            "Sample": ["s1", "s2", "s3"],
            "Flq_acquired": ["qnrS1", "-", "qnrB1"],
            "Flq_mutations": ["GyrA-83Y", "-", "-"],
        }
    )
    universe = ["s1", "s2", "s3", "s4"]  # s4 labelled but absent from metadata rows → all-zero
    oh = build_determinant_onehot(meta, ["Flq_acquired", "Flq_mutations"], universe)
    assert list(oh.index) == universe
    assert "Flq_acquired:qnrS1" in oh.columns
    assert "Flq_mutations:GyrA-83Y" in oh.columns
    assert oh.loc["s1", "Flq_acquired:qnrS1"] == 1
    assert oh.loc["s2"].sum() == 0          # susceptible → all zero
    assert oh.loc["s4"].sum() == 0          # not in metadata → all zero


def _synthetic_inputs(tmp_path, n: int = 60):
    """Write a tiny metadata TSV + AST CSV with a clean determinant→label signal for ciprofloxacin."""
    samples = [f"g{i:03d}" for i in range(n)]
    resistant = [i % 2 == 0 for i in range(n)]  # alternate R/S, balanced
    meta = pd.DataFrame(
        {
            "Sample": samples,
            "Flq_acquired": ["qnrS1" if r else "-" for r in resistant],
            "Flq_mutations": ["GyrA-83Y" if r else "-" for r in resistant],
        }
    )
    meta_path = tmp_path / "metadata.tsv"
    meta.to_csv(meta_path, sep="\t", index=False)

    ast = pd.DataFrame({"Sample": samples, "ciprofloxacin": [1 if r else 0 for r in resistant]})
    ast_path = tmp_path / "binary_ast_with_split.csv"
    ast.to_csv(ast_path, index=False)
    return meta_path, ast_path


def test_run_end_to_end(tmp_path):
    """run() scores per-column bars + the ceiling and writes the per-drug CSV + manifest."""
    meta_path, ast_path = _synthetic_inputs(tmp_path)
    out_dir = tmp_path / "out"
    run(meta_path, ast_path, out_dir, drugs=["ciprofloxacin"], seeds=(1, 2))

    csv = out_dir / "kp_ciprofloxacin" / "kleborate_determinant_lr_ciprofloxacin.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    assert ALL_KEY in set(df["gene_name"])                      # ceiling row present
    assert {"Flq_acquired", "Flq_mutations"} <= set(df["gene_name"])
    # Clean synthetic signal → high AUROC for the ceiling.
    ceiling = df[df["gene_name"] == ALL_KEY]["mut_auroc"].iloc[0]
    assert ceiling > 0.9
    # Category tagging: acquired = HGT/embeddable; mutations = chromosomal/non-embeddable.
    acq = df[df["gene_name"] == "Flq_acquired"].iloc[0]
    mut = df[df["gene_name"] == "Flq_mutations"].iloc[0]
    assert acq["category"] == "acquired_hgt" and bool(acq["embeddable"]) is True
    assert mut["category"] == "chromosomal_mutation" and bool(mut["embeddable"]) is False

    manifest = json.loads((out_dir / "kleborate_determinant_lr_manifest.json").read_text())
    assert "kp_ciprofloxacin/kleborate_determinant_lr_ciprofloxacin.csv" in manifest["files"]
