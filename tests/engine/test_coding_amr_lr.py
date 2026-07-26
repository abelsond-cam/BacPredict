"""Unit smoke for the Stage-2a baclm-vs-ESM coding probe.

Fabricates tiny ESM + baclm + parquet + AST fixtures where one gene's vector carries a label
signal, then runs the real ESM-vs-baclm k-fold comparison end-to-end. Covers the only new I/O
(the baclm ``[n_cds, 960]`` reader), the gene→flat-index bridge, and the delta orientation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from bacpredict.engine.gene_lr.coding_amr_lr import (
    GeneTarget,
    SpeciesPaths,
    _baclm_minus_esm,
    _ladder_grid,
    _stratified_order,
    build_multi_gene_presence,
    load_baclm_gene_vectors,
    run_gene_comparison,
    run_gene_ladder,
)
from bacpredict.engine.gene_lr.locate_gene import build_gene_presence_table

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

        def gene_vec(sig, label=label):  # bind label per-iteration (called immediately below)
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


def test_ladder_grid_dense_then_coarse_and_endpoint():
    # fine step up to fine_until, then 4x coarser; full pool always the last rung; strictly increasing
    grid = _ladder_grid(pool_size=9000, step=500, fine_until=3000)
    assert grid[0] == 500
    assert grid[-1] == 9000  # endpoint always included
    assert grid == sorted(set(grid))
    fine = [b - a for a, b in zip(grid, grid[1:], strict=False) if b <= 3000]
    assert all(d == 500 for d in fine)  # 500-increments in the fine regime
    # fine_until >= pool forces literal step throughout (+ endpoint)
    assert _ladder_grid(pool_size=1500, step=500, fine_until=10000) == [500, 1000, 1500]


def test_stratified_order_prefix_preserves_class_ratio():
    label_map = {f"s{i:03d}": (1 if i < 30 else 0) for i in range(100)}  # 30% positive
    order = _stratified_order(list(label_map), label_map, seed=1)
    assert len(order) == 100 and set(order) == set(label_map)
    # any prefix keeps roughly the 30% positive rate (nested, stratified)
    for n in (10, 40, 70):
        pos = sum(label_map[s] for s in order[:n])
        assert abs(pos / n - 0.30) <= 0.12
    # nested: the length-40 prefix contains the length-10 prefix
    assert set(order[:10]) <= set(order[:40])


def test_ladder_recovers_signal_and_endpoint_matches_kfold(tmp_path):
    paths = _make_fixtures(tmp_path, n=60)
    lad = run_gene_ladder(GeneTarget("rpoB", "testdrug"), paths, seeds=(1, 2),
                          step=10, fine_until=1000)
    assert lad.get("error") is None
    assert lad["rungs"][0]["n_train"] == 10
    assert lad["rungs"][-1]["n_train"] == lad["n_train_pool"]  # endpoint = full pool
    # the signal is recoverable at the top rung for both frames
    assert lad["rungs"][-1]["esm"]["mean"] > 0.6
    assert lad["rungs"][-1]["baclm"]["mean"] > 0.6


def test_delta_orientation_is_baclm_minus_esm():
    paired = {"esm__minus__baclm": {
        "metric": "auroc", "mean_delta": -0.01, "sd_delta": 0.0,
        "n_runs": 10, "n_first_wins": 2, "win_fraction": 0.2,
    }}
    d = _baclm_minus_esm(paired)
    assert d["mean_delta"] == 0.01 and d["win_fraction"] == 0.8 and d["n_first_wins"] == 8
