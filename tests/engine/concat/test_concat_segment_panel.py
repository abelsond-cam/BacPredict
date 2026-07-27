"""Stage-A smoke for the FT-mean ⊕ top-k gene panel — fit-on-train / test-on-holdout scoring.

The heavy store reads (FT-mean NPZ, ESM store+parquet, split table) are monkeypatched so the test exercises
the panel ASSEMBLY, the fit-on-FT-train / eval-on-holdout LR, the cache-coverage guard, and the reported
``n_eval`` = holdout count (the crux of the leak fix — the AUROC is over the FT-unseen holdout only, not the
whole cache universe). The FT gene block is a real synthetic ``.npz`` on disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bacpredict.engine.concat import concat_segment_panel as P


def _signal(ids_subset, label_map, strength, seed):
    r = np.random.default_rng(seed)
    x = r.normal(size=(len(ids_subset), 4))
    x[:, 0] += np.array([strength if label_map[s] else -strength for s in ids_subset])
    return x.astype(np.float32)


def test_panel_fits_train_scores_holdout(tmp_path, monkeypatch):
    n = 60
    all_ids = [f"g{i}" for i in range(n)]
    y = np.array([i % 2 for i in range(n)])
    label_map = dict(zip(all_ids, y.tolist(), strict=True))
    holdout_ids = all_ids[:20]  # 10 pos / 10 neg; the other 40 are the FT-train the LR fits on

    mean_block = _signal(all_ids, label_map, 0.6, 0)
    # split table + FT-mean + ESM collector are all patched (no cluster stores).
    monkeypatch.setattr(P, "load_splits", lambda t: (label_map, all_ids[20:], [], holdout_ids))
    monkeypatch.setattr(P, "load_ft_mean", lambda cache, drug, lm, scope=None: (all_ids, mean_block))
    monkeypatch.setattr(P, "collect_esm_blocks", lambda *a, **k: {})  # ESM side empty → only ft_top{k} rows

    # comparison ranks the FT panel (rpoB); a real FT gene .npz on disk (carried by 45 genomes).
    comp = tmp_path / "cmp.csv"
    pd.DataFrame([{"gene_name": "rpoB", "ft_lr_auroc": 0.9, "esm_lr_auroc": 0.5}]).to_csv(comp, index=False)
    pd.DataFrame([{"gene_name": "rpoB", "sanitized": "rpoB"}]).to_csv(
        tmp_path / "top_gene_manifest_rifampin.csv", index=False)
    gene_dir = tmp_path / "gene_emb"
    gene_dir.mkdir()
    gene_ids = all_ids[:45]
    np.savez(gene_dir / "rpoB.npz", sample_ids=np.array(gene_ids),
             vectors=_signal(gene_ids, label_map, 1.5, 1))

    df = P.run(split_table=tmp_path / "split.csv", drug="rifampin", parquet_dir=tmp_path, esm_dir=tmp_path,
               ft_cache_dir=tmp_path, comparison_csv=comp, out_dir=tmp_path, panel_sizes=[1])

    assert "mean_only" in df["config"].values
    assert (df["config"] == "ft_top1").any()
    # THE crux: the reported AUROC is over the 20-genome holdout, not the 60-genome cache universe.
    assert (df["n_eval"] == 20).all()
    assert df["auroc"].between(0, 1).all() and df["auroc"].notna().all()
    assert "delta_vs_mean" in df.columns
    assert (tmp_path / "concat_panel_rifampin.csv").exists()
