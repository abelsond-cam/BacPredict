"""Stage-A smoke for the AMR concat ladder (ft_mean, +baclm gene, +baclm noncoding, +both vs ceiling).

The heavy store loaders (FT-mean NPZ, baclm ``.pt`` + parquet + GFF) are monkeypatched so the test
exercises the ladder ASSEMBLY, fit-on-train/test-on-holdout scoring, the cache-coverage guard, ceiling
parse, and table shape on tiny synthetic vectors — no cluster stores. Separate tests pin the ranking
selection (``_best_from_ranking`` selects on the **train-OOF** ``lr_auroc`` — leakage-free w.r.t. the FT
holdout; ``_select_noncoding`` picks the top-imputed-AUROC region across upstream ∪ per_unit ∪ igr, no gate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bacpredict.engine.concat import build_amr_ladder as L


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_best_from_ranking_prefers_lr_auroc(tmp_path):
    csv = tmp_path / "per_gene_lr_rifampin.csv"
    pd.DataFrame([
        {"gene_name": "rpoB", "lr_auroc_rifampin": 0.90, "eval_auroc_rifampin": 0.97},
        {"gene_name": "katG", "lr_auroc_rifampin": 0.95, "eval_auroc_rifampin": 0.60},
    ]).to_csv(csv, index=False)
    # train-OOF lr_auroc preferred (leakage-free w.r.t. the FT holdout) → katG (0.95) wins even though rpoB
    # has the higher held-out eval_auroc. Selecting on eval_auroc would taint the number with the test set.
    assert L._best_from_ranking(csv, key_col="gene_name") == ("katG", 0.95)

    csv2 = tmp_path / "oof_only.csv"
    pd.DataFrame([{"gene_name": "a", "lr_auroc_x": 0.7}, {"gene_name": "b", "lr_auroc_x": 0.8}]).to_csv(csv2, index=False)
    assert L._best_from_ranking(csv2, key_col="gene_name") == ("b", 0.8)
    # an lr_auroc column that is ALL-NaN must fall back to eval_auroc, not empty the frame.
    csv3 = tmp_path / "empty_lr.csv"
    pd.DataFrame([{"gene_name": "a", "lr_auroc_x": np.nan, "eval_auroc_x": 0.7},
                  {"gene_name": "b", "lr_auroc_x": np.nan, "eval_auroc_x": 0.8}]).to_csv(csv3, index=False)
    assert L._best_from_ranking(csv3, key_col="gene_name") == ("b", 0.8)
    assert L._best_from_ranking(tmp_path / "missing.csv", key_col="gene_name") is None


def test_select_noncoding_picks_top_imputed_across_keys(tmp_path):
    """Top-imputed-AUROC region across upstream ∪ per_unit ∪ igr → (kind, key, auroc); no prevalence gate."""
    up = tmp_path / "per_upstream_lr_ethionamide.csv"
    pd.DataFrame([{"gene": "fabg1", "prevalence": 0.997, "n_pos": 100,
                   "lr_auroc_ethionamide": 0.80}]).to_csv(up, index=False)
    un = tmp_path / "per_unit_lr_ethionamide.csv"
    pd.DataFrame([{"unit": "rrna:rrs", "prevalence": 0.997, "n_pos": 100,
                   "lr_auroc_ethionamide": 0.60}]).to_csv(un, index=False)
    ig = tmp_path / "per_igr_lr_ethionamide.csv"
    pd.DataFrame([{"igr_pair": "mura→ogt", "prevalence": 0.99, "n_pos": 100,
                   "lr_auroc_ethionamide": 0.70}]).to_csv(ig, index=False)
    assert L._select_noncoding(up, un, ig) == ("upstream", "fabg1", 0.80)
    # if the merged convergent IGR scores highest, it wins and the kind flips to "igr".
    pd.DataFrame([{"igr_pair": "mura→ogt", "prevalence": 0.99, "n_pos": 100,
                   "lr_auroc_ethionamide": 0.92}]).to_csv(ig, index=False)
    assert L._select_noncoding(up, un, ig) == ("igr", "mura→ogt", 0.92)
    # NO gate: a low-prevalence accessory region with the top imputed AUROC still wins (the imputed zeros
    # already penalise it, so the arg-max is safe — this is the point of dropping the ≥0.9 core gate).
    pd.DataFrame([{"unit": "ncrna:acc", "prevalence": 0.05, "n_pos": 40,
                   "lr_auroc_ethionamide": 0.95}]).to_csv(un, index=False)
    assert L._select_noncoding(up, un, ig) == ("per_unit", "ncrna:acc", 0.95)
    assert L._select_noncoding(tmp_path / "n1.csv", tmp_path / "n2.csv", tmp_path / "n3.csv") is None


def _write_cache_summary(tmp_path, drug, scope="trainholdout"):
    """The FT cache provenance the ladder reads (the checkpoint value is ignored — resolve_clean_splits is patched)."""
    (tmp_path / f"cache_summary_{drug}.json").write_text(
        f'{{"checkpoint": "{tmp_path}/ckpt", "scope": "{scope}"}}'
    )


def test_run_builds_four_config_ladder_with_ceiling(tmp_path, monkeypatch):
    n = 60
    all_ids = [f"g{i}" for i in range(n)]
    y = np.array([1 if i % 2 == 0 else 0 for i in range(n)])
    label_map = dict(zip(all_ids, y.tolist(), strict=True))
    holdout_ids = all_ids[:20]  # 10 pos / 10 neg; the other 40 are the FT-train the LR fits on

    def _signal(ids_subset, strength, seed):
        r = _rng(seed)
        x = r.normal(size=(len(ids_subset), 4))
        x[:, 0] += np.array([strength if label_map[s] else -strength for s in ids_subset])
        return x.astype(np.float32)

    # FT-mean over the whole train+holdout universe (weak signal).
    mean_block = _signal(all_ids, 0.6, 0)
    monkeypatch.setattr(L, "resolve_clean_splits",
                        lambda *a, **k: (label_map, all_ids[20:], [], holdout_ids,
                                         {"source": "kfold", "n_evaluate_expected": len(holdout_ids)}))
    monkeypatch.setattr(L, "load_ft_mean", lambda cache, drug, lm, scope=None: (all_ids, mean_block))
    _write_cache_summary(tmp_path, "ethionamide")
    # best gene carried by 45 genomes; best upstream region by 50; the per-unit loader is unused here.
    gene_ids = all_ids[:45]
    igr_ids = all_ids[:50]
    monkeypatch.setattr(L, "load_baclm_gene_block", lambda ids, gene, **k: (gene_ids, _signal(gene_ids, 1.5, 1)))
    monkeypatch.setattr(L, "load_baclm_upstream_block", lambda ids, gene, **k: (igr_ids, _signal(igr_ids, 1.2, 2)))
    monkeypatch.setattr(L, "load_baclm_unit_block", lambda ids, unit, **k: ([], np.zeros((0, 0), np.float32)))
    monkeypatch.setattr(L, "load_baclm_igr_block", lambda ids, pair, **k: ([], np.zeros((0, 0), np.float32)))

    # synthetic ranking + catalogue inputs. Upstream fabg1 (0.80) beats per-unit rrs (0.60) + igr a→b (0.50)
    # → upstream non-coding rung (no prevalence gate now; top imputed AUROC wins).
    gcsv = tmp_path / "gene.csv"
    pd.DataFrame([{"gene_name": "rpoB", "lr_auroc_ethionamide": 0.9}]).to_csv(gcsv, index=False)
    ucsv = tmp_path / "upstream.csv"
    pd.DataFrame([{"gene": "fabg1", "prevalence": 0.997, "n_pos": 25,
                   "lr_auroc_ethionamide": 0.80}]).to_csv(ucsv, index=False)
    nucsv = tmp_path / "unit.csv"
    pd.DataFrame([{"unit": "rrna:rrs", "prevalence": 0.997, "n_pos": 25,
                   "lr_auroc_ethionamide": 0.60}]).to_csv(nucsv, index=False)
    igcsv = tmp_path / "igr.csv"
    pd.DataFrame([{"igr_pair": "a→b", "prevalence": 0.5, "n_pos": 25,
                   "lr_auroc_ethionamide": 0.50}]).to_csv(igcsv, index=False)
    ccsv = tmp_path / "cat.csv"
    pd.DataFrame([
        {"gene_name": "__ALL_WHO_one_hot__", "mut_auroc": 0.99, "mut_auprc": 0.98},
        {"gene_name": "inhA", "mut_auroc": 0.95, "mut_auprc": 0.9},
    ]).to_csv(ccsv, index=False)
    inp = tmp_path / "input.csv"
    pd.DataFrame({"Sample": all_ids, "sr_gff_file": ["/dev/null"] * n}).to_csv(inp, index=False)

    table = L.run(
        species="tb", drug="ethionamide", ast_sheet=tmp_path / "sheet.csv", ft_cache_dir=tmp_path,
        baclm_dir=tmp_path, noncoding_dir=tmp_path, parquet_dir=tmp_path, input_csv=inp,
        gene_ranking_csv=gcsv, upstream_ranking_csv=ucsv, unit_ranking_csv=nucsv, igr_ranking_csv=igcsv,
        catalogue_csv=ccsv, out_dir=tmp_path,
    )

    assert list(table["rung"]) == [1, 2, 3, 4]
    assert list(table["config"]) == ["ft_mean", "ft_mean+baclm_gene", "ft_mean+baclm_noncoding",
                                     "ft_mean+baclm_gene+baclm_noncoding"]
    assert table.loc[1, "block"] == "rpoB"
    assert table.loc[2, "block"] == "upstream:fabg1" and table.loc[2, "noncoding_kind"] == "upstream"
    assert table.loc[3, "block"] == "rpoB | upstream:fabg1"
    # +gene and +noncoding each add ONE block (equal width); +both stacks both → widest; ft_mean narrowest.
    assert table.loc[0, "n_features"] < table.loc[1, "n_features"] < table.loc[3, "n_features"]
    assert table.loc[2, "n_features"] == table.loc[1, "n_features"]
    assert (table["ceiling_auroc"] == 0.99).all()
    assert table.loc[0, "lift_vs_ft"] == 0.0  # ft_mean is the baseline
    # adding the strong gene block lifts AUROC over FT-mean alone.
    assert table.loc[1, "auroc"] >= table.loc[0, "auroc"]
    assert (tmp_path / "ethionamide_amr_ladder_table.csv").exists()


def test_run_guards_against_leaky_cache(tmp_path, monkeypatch):
    """The cache MUST contain the deployed k-fold holdout; a cache holding ~none of it (the leak signature) raises."""
    n = 40
    all_ids = [f"g{i}" for i in range(n)]
    label_map = {s: i % 2 for i, s in enumerate(all_ids)}
    mean_block = _rng(3).normal(size=(n, 4)).astype(np.float32)
    # The deployed k-fold holdout is 30 genomes NOT present in the cache (the CSV-vs-kfold mismatch, azithro's
    # 69-of-384 leak signature) → coverage 0/30 → the ladder must refuse rather than score a leaky set.
    holdout_ids = [f"h{i}" for i in range(30)]
    monkeypatch.setattr(L, "resolve_clean_splits",
                        lambda *a, **k: (label_map, [], [], holdout_ids, {"source": "kfold"}))
    monkeypatch.setattr(L, "load_ft_mean", lambda *a, **k: (all_ids, mean_block))
    _write_cache_summary(tmp_path, "azithromycin")
    with pytest.raises(ValueError, match="leak signature"):
        L.run(species="kp", drug="azithromycin", ast_sheet=tmp_path / "s.csv", ft_cache_dir=tmp_path,
              baclm_dir=tmp_path, noncoding_dir=tmp_path, parquet_dir=tmp_path, input_csv=tmp_path / "i.csv",
              gene_ranking_csv=tmp_path / "g.csv", upstream_ranking_csv=tmp_path / "u.csv",
              unit_ranking_csv=tmp_path / "nu.csv", igr_ranking_csv=tmp_path / "ig.csv",
              catalogue_csv=tmp_path / "c.csv", out_dir=tmp_path)


def test_run_survives_missing_rankings(tmp_path, monkeypatch):
    """No gene/upstream/unit/igr ranking on disk → all four configs fall back to the FT-mean features (no crash)."""
    n = 40
    all_ids = [f"g{i}" for i in range(n)]
    y = np.array([i % 2 for i in range(n)])
    label_map = dict(zip(all_ids, y.tolist(), strict=True))
    holdout_ids = all_ids[:16]  # the other 24 are FT-train the LR fits on
    mean_block = _rng(3).normal(size=(n, 4)).astype(np.float32)
    monkeypatch.setattr(L, "resolve_clean_splits",
                        lambda *a, **k: (label_map, all_ids[16:], [], holdout_ids, {"source": "kfold"}))
    monkeypatch.setattr(L, "load_ft_mean", lambda *a, **k: (all_ids, mean_block))
    _write_cache_summary(tmp_path, "kanamycin")
    inp = tmp_path / "input.csv"
    pd.DataFrame({"Sample": all_ids, "sr_gff_file": ["/dev/null"] * n}).to_csv(inp, index=False)

    table = L.run(
        species="tb", drug="kanamycin", ast_sheet=tmp_path / "s.csv", ft_cache_dir=tmp_path,
        baclm_dir=tmp_path, noncoding_dir=tmp_path, parquet_dir=tmp_path, input_csv=inp,
        gene_ranking_csv=tmp_path / "none_g.csv", upstream_ranking_csv=tmp_path / "none_u.csv",
        unit_ranking_csv=tmp_path / "none_nu.csv", igr_ranking_csv=tmp_path / "none_ig.csv",
        catalogue_csv=tmp_path / "none_c.csv", out_dir=tmp_path,
    )
    assert list(table["rung"]) == [1, 2, 3, 4]
    assert (table["block"] == "").all()  # nothing matched → all configs are the FT-mean only
    assert table["n_features"].nunique() == 1
    assert np.isnan(table.loc[0, "ceiling_auroc"])


def test_default_gene_ranking_requires_imputed(tmp_path):
    """Selection must match usage — HARD: resolve the zero-imputed ranking, else RAISE (no carrier fallback)."""
    drug = "ciprofloxacin"
    imp = tmp_path / "per_gene_lr_ranking_imputed_baclm" / drug
    ev = tmp_path / "per_gene_lr_ranking_baclm_eval" / drug
    for d in (imp, ev):
        d.mkdir(parents=True)
        pd.DataFrame([{"gene_name": "x", "lr_auroc_ciprofloxacin": 0.9}]).to_csv(d / f"per_gene_lr_{drug}.csv",
                                                                                 index=False)
    csv, flavour = L.default_gene_ranking(tmp_path, drug)
    assert flavour == "imputed_zero" and "imputed" in str(csv)

    # only a carrier-only ranking present → RAISE, never silently select on it (the tetracycline/iME4 footgun).
    import shutil
    shutil.rmtree(tmp_path / "per_gene_lr_ranking_imputed_baclm")
    with pytest.raises(FileNotFoundError, match="zero-imputed gene ranking"):
        L.default_gene_ranking(tmp_path, drug)
    with pytest.raises(FileNotFoundError):
        L.default_gene_ranking(tmp_path / "empty", drug)


def test_assert_imputed_ranking_guards_override(tmp_path):
    """--gene-ranking-csv override: accept only the zero-imputed ranking; reject a carrier-only one."""
    ok = tmp_path / "imp.csv"
    pd.DataFrame([{"gene_name": "gyrA", "lr_auroc_x": 0.9, "impute_mode": "imputed_zero"}]).to_csv(ok, index=False)
    L._assert_imputed_ranking(ok, "x")  # no raise

    bad = tmp_path / "carrier.csv"
    pd.DataFrame([{"gene_name": "recE", "lr_auroc_x": 0.95, "impute_mode": "carrier_only"}]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="not the zero-imputed ranking"):
        L._assert_imputed_ranking(bad, "x")

    # legacy file (no impute_mode column) is trusted ONLY under an *_imputed_* path.
    legacy_dir = tmp_path / "per_gene_lr_ranking_imputed_baclm" / "x"
    legacy_dir.mkdir(parents=True)
    legacy_ok = legacy_dir / "per_gene_lr_x.csv"
    pd.DataFrame([{"gene_name": "gyrA", "lr_auroc_x": 0.9}]).to_csv(legacy_ok, index=False)
    L._assert_imputed_ranking(legacy_ok, "x")  # no raise (path carries 'imputed')

    legacy_bad = tmp_path / "per_gene_lr_x.csv"  # unmarked AND not under an imputed path → reject
    pd.DataFrame([{"gene_name": "gyrA", "lr_auroc_x": 0.9}]).to_csv(legacy_bad, index=False)
    with pytest.raises(ValueError):
        L._assert_imputed_ranking(legacy_bad, "x")

    with pytest.raises(FileNotFoundError):
        L._assert_imputed_ranking(tmp_path / "nope.csv", "x")
