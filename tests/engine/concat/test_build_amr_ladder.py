"""Stage-A smoke for the AMR concat ladder (ft_mean, +baclm gene, +baclm noncoding, +both vs ceiling).

The heavy store loaders (FT-mean NPZ, baclm ``.pt`` + parquet + GFF) are monkeypatched so the test
exercises the ladder ASSEMBLY, scoring, ceiling parse, and table shape on tiny synthetic vectors — no
cluster stores. Separate tests pin the ranking selection (``_best_from_ranking`` prefers held-out eval
AUROC; ``_best_core_from_ranking`` / ``_select_core_noncoding`` apply the core-prevalence + n_pos guard).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bacpredict.engine.concat import build_amr_ladder as L


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_best_from_ranking_prefers_eval_auroc(tmp_path):
    csv = tmp_path / "per_gene_lr_rifampin.csv"
    pd.DataFrame([
        {"gene_name": "rpoB", "lr_auroc_rifampin": 0.90, "eval_auroc_rifampin": 0.97},
        {"gene_name": "katG", "lr_auroc_rifampin": 0.95, "eval_auroc_rifampin": 0.60},
    ]).to_csv(csv, index=False)
    # eval_auroc preferred → rpoB (0.97) wins even though katG has the higher OOF lr_auroc.
    assert L._best_from_ranking(csv, key_col="gene_name") == ("rpoB", 0.97)

    csv2 = tmp_path / "oof_only.csv"
    pd.DataFrame([{"gene_name": "a", "lr_auroc_x": 0.7}, {"gene_name": "b", "lr_auroc_x": 0.8}]).to_csv(csv2, index=False)
    assert L._best_from_ranking(csv2, key_col="gene_name") == ("b", 0.8)
    assert L._best_from_ranking(tmp_path / "missing.csv", key_col="gene_name") is None


def test_best_core_from_ranking_guards_prevalence_and_n_pos(tmp_path):
    """Core selection keeps prevalence>=min and n_pos>=min — the low-n low-prev artifact is dropped."""
    csv = tmp_path / "per_unit_lr_streptomycin.csv"
    pd.DataFrame([
        {"unit": "crispr:artifact", "prevalence": 0.003, "n_pos": 3, "lr_auroc_streptomycin": 1.0},
        {"unit": "rrna:rrs", "prevalence": 0.997, "n_pos": 90, "lr_auroc_streptomycin": 0.65},
        {"unit": "ncrna:acc", "prevalence": 0.30, "n_pos": 80, "lr_auroc_streptomycin": 0.80},
    ]).to_csv(csv, index=False)
    # artifact excluded (prev<0.9 AND n_pos<50); accessory excluded (prev<0.9); rrs is the only core survivor.
    assert L._best_core_from_ranking(csv, key_col="unit", min_prevalence=0.9, min_n_pos=50) == ("rrna:rrs", 0.65)
    # loosen prevalence to admit the accessory but keep the n_pos floor → the 0.80 accessory now wins.
    assert L._best_core_from_ranking(csv, key_col="unit", min_prevalence=0.2, min_n_pos=50) == ("ncrna:acc", 0.80)
    assert L._best_core_from_ranking(tmp_path / "none.csv", key_col="unit", min_prevalence=0.9, min_n_pos=50) is None


def test_select_core_noncoding_picks_best_across_keys(tmp_path):
    """Best core region across upstream ∪ per_unit → (kind, key, auroc); the higher AUROC wins."""
    up = tmp_path / "per_upstream_lr_ethionamide.csv"
    pd.DataFrame([{"gene": "fabg1", "prevalence": 0.997, "n_pos": 100,
                   "lr_auroc_ethionamide": 0.80}]).to_csv(up, index=False)
    un = tmp_path / "per_unit_lr_ethionamide.csv"
    pd.DataFrame([{"unit": "rrna:rrs", "prevalence": 0.997, "n_pos": 100,
                   "lr_auroc_ethionamide": 0.60}]).to_csv(un, index=False)
    assert L._select_core_noncoding(up, un, min_prevalence=0.9, min_n_pos=50) == ("upstream", "fabg1", 0.80)
    # if the per_unit body scores higher, it wins and the kind flips.
    pd.DataFrame([{"unit": "rrna:rrs", "prevalence": 0.997, "n_pos": 100,
                   "lr_auroc_ethionamide": 0.90}]).to_csv(un, index=False)
    assert L._select_core_noncoding(up, un, min_prevalence=0.9, min_n_pos=50) == ("per_unit", "rrna:rrs", 0.90)
    assert L._select_core_noncoding(tmp_path / "n1.csv", tmp_path / "n2.csv",
                                    min_prevalence=0.9, min_n_pos=50) is None


def test_run_builds_four_config_ladder_with_ceiling(tmp_path, monkeypatch):
    n = 60
    all_ids = [f"g{i}" for i in range(n)]
    y = np.array([1 if i % 2 == 0 else 0 for i in range(n)])
    label_map = dict(zip(all_ids, y.tolist(), strict=True))

    def _signal(ids_subset, strength, seed):
        r = _rng(seed)
        x = r.normal(size=(len(ids_subset), 4))
        x[:, 0] += np.array([strength if label_map[s] else -strength for s in ids_subset])
        return x.astype(np.float32)

    # FT-mean over the whole holdout universe (weak signal).
    mean_block = _signal(all_ids, 0.6, 0)
    monkeypatch.setattr(L, "resolve_clean_splits", lambda *a, **k: (label_map, [], [], all_ids, {}))
    monkeypatch.setattr(L, "load_ft_mean", lambda cache, drug, lm: (all_ids, mean_block))
    # best gene carried by 45 genomes; best upstream region by 50; the per-unit loader is unused here.
    gene_ids = all_ids[:45]
    igr_ids = all_ids[:50]
    monkeypatch.setattr(L, "load_baclm_gene_block", lambda ids, gene, **k: (gene_ids, _signal(gene_ids, 1.5, 1)))
    monkeypatch.setattr(L, "load_baclm_upstream_block", lambda ids, gene, **k: (igr_ids, _signal(igr_ids, 1.2, 2)))
    monkeypatch.setattr(L, "load_baclm_unit_block", lambda ids, unit, **k: ([], np.zeros((0, 0), np.float32)))

    # synthetic ranking + catalogue inputs. Upstream fabg1 (0.80) beats per-unit rrs (0.60) → upstream rung.
    gcsv = tmp_path / "gene.csv"
    pd.DataFrame([{"gene_name": "rpoB", "lr_auroc_ethionamide": 0.9}]).to_csv(gcsv, index=False)
    ucsv = tmp_path / "upstream.csv"
    pd.DataFrame([{"gene": "fabg1", "prevalence": 0.997, "n_pos": 25,
                   "lr_auroc_ethionamide": 0.80}]).to_csv(ucsv, index=False)
    nucsv = tmp_path / "unit.csv"
    pd.DataFrame([{"unit": "rrna:rrs", "prevalence": 0.997, "n_pos": 25,
                   "lr_auroc_ethionamide": 0.60}]).to_csv(nucsv, index=False)
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
        gene_ranking_csv=gcsv, upstream_ranking_csv=ucsv, unit_ranking_csv=nucsv,
        catalogue_csv=ccsv, out_dir=tmp_path, core_min_prevalence=0.9, core_min_n_pos=20,
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


def test_run_survives_missing_rankings(tmp_path, monkeypatch):
    """No gene/upstream/unit ranking on disk → all four configs fall back to the FT-mean features (no crash)."""
    n = 40
    all_ids = [f"g{i}" for i in range(n)]
    y = np.array([i % 2 for i in range(n)])
    label_map = dict(zip(all_ids, y.tolist(), strict=True))
    mean_block = _rng(3).normal(size=(n, 4)).astype(np.float32)
    monkeypatch.setattr(L, "resolve_clean_splits", lambda *a, **k: (label_map, [], [], all_ids, {}))
    monkeypatch.setattr(L, "load_ft_mean", lambda *a, **k: (all_ids, mean_block))
    inp = tmp_path / "input.csv"
    pd.DataFrame({"Sample": all_ids, "sr_gff_file": ["/dev/null"] * n}).to_csv(inp, index=False)

    table = L.run(
        species="tb", drug="kanamycin", ast_sheet=tmp_path / "s.csv", ft_cache_dir=tmp_path,
        baclm_dir=tmp_path, noncoding_dir=tmp_path, parquet_dir=tmp_path, input_csv=inp,
        gene_ranking_csv=tmp_path / "none_g.csv", upstream_ranking_csv=tmp_path / "none_u.csv",
        unit_ranking_csv=tmp_path / "none_nu.csv", catalogue_csv=tmp_path / "none_c.csv", out_dir=tmp_path,
    )
    assert list(table["rung"]) == [1, 2, 3, 4]
    assert (table["block"] == "").all()  # nothing matched → all configs are the FT-mean only
    assert table["n_features"].nunique() == 1
    assert np.isnan(table.loc[0, "ceiling_auroc"])


def test_default_gene_ranking_prefers_imputed_over_carrier_only(tmp_path):
    """Selection must match usage: the zero-imputed whole-cohort ranking wins over the carrier-only ones."""
    drug = "ciprofloxacin"
    imp = tmp_path / "per_gene_lr_ranking_imputed_baclm" / drug
    ev = tmp_path / "per_gene_lr_ranking_baclm_eval" / drug
    for d in (imp, ev):
        d.mkdir(parents=True)
        pd.DataFrame([{"gene_name": "x", "lr_auroc_ciprofloxacin": 0.9}]).to_csv(d / f"per_gene_lr_{drug}.csv",
                                                                                 index=False)
    csv, flavour = L.default_gene_ranking(tmp_path, drug)
    assert flavour == "imputed_whole_cohort" and "imputed" in str(csv)

    # no imputed ranking → fall back to the carrier-only eval one (logged as selection≠usage)
    import shutil
    shutil.rmtree(tmp_path / "per_gene_lr_ranking_imputed_baclm")
    csv2, flavour2 = L.default_gene_ranking(tmp_path, drug)
    assert flavour2 == "carrier_only_eval" and "_eval" in str(csv2)

    assert L.default_gene_ranking(tmp_path / "empty", drug) == (None, "none")
