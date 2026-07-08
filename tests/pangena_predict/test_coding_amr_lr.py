"""Unit smoke for the Stage-2a baclm-vs-ESM coding probe.

Fabricates tiny ESM + baclm + parquet + AST fixtures where one gene's vector carries a label
signal, then runs the real ESM-vs-baclm k-fold comparison end-to-end. Covers the only new I/O
(the baclm ``[n_cds, 960]`` reader), the gene→flat-index bridge, and the delta orientation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from pangena_predict.coding_amr_lr import (
    GeneTarget,
    SpeciesPaths,
    _baclm_minus_esm,
    build_multi_gene_presence,
    load_baclm_gene_vectors,
    run_gene_comparison,
)
from pangena_predict.locate_gene import build_gene_presence_table

GENES = ["geneA", "rpoB", "geneC", "geneD"]  # rpoB at flat_index 1
RPOB = 1
DIM = 960


def _make_fixtures(root, n=30, signal=2.5):
    rng = np.random.default_rng(0)
    for sub in ("esm", "baclm", "pq"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        sid = f"s{i:03d}"
        label = i % 2

        def gene_vec(sig):
            v = rng.normal(0, 1, (len(GENES), DIM)).astype(np.float32)
            v[RPOB, :20] += sig * label
            return v

        torch.save(
            {"protein_embeddings": torch.from_numpy(gene_vec(signal)).unsqueeze(0),
             "attention_mask": torch.ones(1, len(GENES), dtype=torch.long)},
            root / "esm" / f"{sid}_esm_embeddings.pt",
        )
        torch.save(
            {"protein_embeddings": torch.from_numpy(gene_vec(signal * 0.9)).to(torch.bfloat16),
             "n_proteins": len(GENES)},
            root / "baclm" / f"{sid}_baclm_embeddings.pt",
        )
        pd.DataFrame([{
            "contig_idx": [0],
            "gene_name": [GENES],
            "protein_name": [GENES],
            "protein_sequence": [["M" * 10] * len(GENES)],
            "start": [[1, 100, 200, 300]],
            "end": [[90, 190, 290, 390]],
            "protein_id": [[f"p{j}" for j in range(len(GENES))]],
        }]).to_parquet(root / "pq" / f"{sid}_protein_sequences.parquet")
        tve = "evaluate" if i < 6 else ("validate" if i < 9 else "train")
        rows.append({"Sample": sid, "testdrug": label, "train_val_eval": tve})
    ast = root / "ast.csv"
    pd.DataFrame(rows).to_csv(ast, index=False)
    return SpeciesPaths(ast_sheet=ast, esm_dir=root / "esm", baclm_dir=root / "baclm", parquet_dir=root / "pq")


def test_baclm_reader_reads_correct_row(tmp_path):
    paths = _make_fixtures(tmp_path, n=6)
    table = build_gene_presence_table([f"s{i:03d}" for i in range(6)], paths.parquet_dir, "rpoB")
    assert (table["gene_flat_index"] == RPOB).all()
    vecs = load_baclm_gene_vectors(table, paths.baclm_dir)
    assert vecs.shape == (6, DIM)  # one 960-vector per sample


def test_esm_vs_baclm_comparison_recovers_signal(tmp_path):
    paths = _make_fixtures(tmp_path, n=30)
    res = run_gene_comparison(GeneTarget("rpoB", "testdrug"), paths, n_folds=3, seeds=(1, 2))
    assert res.error is None
    assert res.n_esm == 30 and res.n_baclm == 30
    for name in ("esm", "baclm"):
        assert res.kfold["frames"][name]["aggregate"]["auroc"]["mean"] > 0.6


def test_multi_gene_presence_matches_single_gene(tmp_path):
    paths = _make_fixtures(tmp_path, n=8)
    ids = [f"s{i:03d}" for i in range(8)]
    tables = build_multi_gene_presence(ids, paths.parquet_dir, [("rpoB", ()), ("geneC", ())])
    # one sweep locates every gene at its true flat index (rpoB=1, geneC=2)
    assert (tables["rpoB"]["gene_flat_index"] == 1).all()
    assert (tables["geneC"]["gene_flat_index"] == 2).all()
    # agrees row-for-row with the per-gene builder
    single = build_gene_presence_table(ids, paths.parquet_dir, "rpoB")
    assert tables["rpoB"]["gene_flat_index"].equals(single["gene_flat_index"])


def test_delta_orientation_is_baclm_minus_esm():
    paired = {"esm__minus__baclm": {
        "metric": "auroc", "mean_delta": -0.01, "sd_delta": 0.0,
        "n_runs": 10, "n_first_wins": 2, "win_fraction": 0.2,
    }}
    d = _baclm_minus_esm(paired)
    assert d["mean_delta"] == 0.01 and d["win_fraction"] == 0.8 and d["n_first_wins"] == 8
