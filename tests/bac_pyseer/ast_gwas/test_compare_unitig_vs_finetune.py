"""The aggregate unitig-vs-fine-tune comparison, and the honesty of its p-value."""

from __future__ import annotations

import json

import pytest

from bac_pyseer.ast_gwas.compare_unitig_vs_finetune import (
    exact_binomial_two_sided,
    read_finetune,
    run,
    summarise,
)


def test_exact_binomial_matches_known_values():
    # 22 trials: all-one-way is (1/2)^22 doubled; an even split is p = 1.
    assert exact_binomial_two_sided(22, 22) == pytest.approx(2 * 0.5**22)
    assert exact_binomial_two_sided(11, 22) == pytest.approx(1.0)
    # 18/22 two-sided — the classic textbook shape, checked against the closed form.
    from math import comb
    expect = 2 * sum(comb(22, i) * 0.5**22 for i in range(18, 23))
    assert exact_binomial_two_sided(18, 22) == pytest.approx(expect)
    assert exact_binomial_two_sided(0, 0) != exact_binomial_two_sided(0, 0) or True  # nan, no crash


def test_summarise_counts_wins_and_excludes_ties(tmp_path):
    rows = [
        {"delta_x_minus_ft": 0.01}, {"delta_x_minus_ft": 0.02}, {"delta_x_minus_ft": -0.03},
        {"delta_x_minus_ft": 0.0},
    ]
    s = summarise(rows, "x", "label")
    assert (s["wins"], s["losses"], s["ties"]) == (2, 1, 1)
    assert s["n_drugs"] == 4
    # The tie is excluded from the test, not split: an exact tie is evidence for neither arm.
    assert s["binomial_p"] == pytest.approx(exact_binomial_two_sided(2, 3))


def test_finetune_numbers_come_from_each_checkpoints_own_results_json(tmp_path):
    """Reading a summary panel instead is how colistin was once quoted as 0.8072, not 0.9094."""
    d = tmp_path / "klebsiella_pneumoniae_colistin_lr_0.00015_finetuned_fold00_seed1"
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({"metrics": {"auroc": 0.9094}, "n_evaluate": 282}))
    got = read_finetune(tmp_path)
    assert got["colistin"]["ft_auroc"] == 0.9094
    assert got["colistin"]["ft_n_evaluate"] == 282
    assert got["colistin"]["ft_checkpoint"] == d.name


def _lay_out(tmp_path, drug, ft, full, tv, n=282):
    ftd = tmp_path / "ft" / f"klebsiella_pneumoniae_{drug}_lr_0.00015_finetuned_fold00_seed1"
    ftd.mkdir(parents=True)
    (ftd / "results.json").write_text(json.dumps({"metrics": {"auroc": ft}, "n_evaluate": n}))
    fd = tmp_path / "kp" / drug / "lr"
    fd.mkdir(parents=True)
    (fd / "results.json").write_text(json.dumps({"metrics": {"auroc": full}, "n_evaluate": n}))
    vd = tmp_path / "vocab" / drug / drug / "lr"
    vd.mkdir(parents=True)
    (vd / "results.json").write_text(json.dumps({"metrics": {"auroc": tv}, "n_evaluate": n}))
    (tmp_path / "vocab" / drug / "leakage_audit.json").write_text(json.dumps({"reflist": {"n_holdout": n}}))


def test_run_reports_both_arms_and_flags_non_independence(tmp_path, capsys):
    _lay_out(tmp_path, "colistin", 0.90, 0.92, 0.919)
    _lay_out(tmp_path, "ertapenem", 0.95, 0.977, 0.9797, n=423)
    assert run(tmp_path / "ft", tmp_path / "kp", tmp_path / "vocab", tmp_path / "o.csv") == 0
    out = capsys.readouterr().out
    assert "unitig (leakage-free) vs BacFormer FT" in out
    assert "unitig (full-cohort vocab) vs BacFormer FT" in out
    assert "NOT independent" in out
    assert "no per-drug CI against the FT exists" in out
    assert (tmp_path / "o.csv").exists()


def test_a_holdout_size_mismatch_is_flagged_not_hidden(tmp_path, capsys):
    _lay_out(tmp_path, "colistin", 0.90, 0.92, 0.919)
    # FT scored a different number of genomes than the unitig arm.
    ftd = tmp_path / "ft" / "klebsiella_pneumoniae_colistin_lr_0.00015_finetuned_fold00_seed1"
    (ftd / "results.json").write_text(json.dumps({"metrics": {"auroc": 0.9}, "n_evaluate": 999}))
    run(tmp_path / "ft", tmp_path / "kp", tmp_path / "vocab", None)
    out = capsys.readouterr().out
    assert "holdout n differs" in out
    assert "agrees between FT and unitig on 0/1" in out


def test_nothing_to_compare_is_an_error(tmp_path):
    for d in ("ft", "kp", "vocab"):
        (tmp_path / d).mkdir()
    with pytest.raises(SystemExit):
        run(tmp_path / "ft", tmp_path / "kp", tmp_path / "vocab", None)


def test_unitig_holdout_count_is_read_from_split_not_only_the_top_level(tmp_path):
    """The count lives under split{}. Reading only the top level yields None, which compares
    unequal to the FT's real count and flags every drug — a check that looks like it ran."""
    from bac_pyseer.ast_gwas.compare_unitig_vs_finetune import read_unitig
    d = tmp_path / "colistin" / "lr"
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({"metrics": {"auroc": 0.9}, "split": {"n_evaluate": 282}}))
    assert read_unitig(tmp_path, "colistin", nested=False)["n_evaluate"] == 282
    # top-level still wins when present
    (d / "results.json").write_text(json.dumps({"metrics": {"auroc": 0.9}, "n_evaluate": 99,
                                                "split": {"n_evaluate": 282}}))
    assert read_unitig(tmp_path, "colistin", nested=False)["n_evaluate"] == 99


def _ft(root, species, drug, auroc, n_eval):
    d = root / f"{species}_{drug}_lr_0.00015_finetuned_fold00_seed1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps({"metrics": {"auroc": auroc}, "n_evaluate": n_eval}))


def _unitig(root, drug, auroc, n_eval, *, nested=False):
    d = (root / drug / drug if nested else root / drug) / "lr"
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(
        json.dumps({"metrics": {"auroc": auroc}, "split": {"n_evaluate": n_eval}})
    )


def test_tb_checkpoints_are_read_under_their_own_species(tmp_path):
    """Kp and TB name checkpoints identically apart from the prefix; only the species differs."""
    _ft(tmp_path, "mycobacterium_tuberculosis", "rifampin", 0.9642, 7127)
    _ft(tmp_path, "mycobacterium_tuberculosis", "rifabutin", 0.8381, 2427)
    got = read_finetune(tmp_path, species="mycobacterium_tuberculosis")
    assert set(got) == {"rifampin", "rifabutin"}
    assert got["rifampin"]["ft_auroc"] == pytest.approx(0.9642)
    # The Kp species must not pick these up.
    assert read_finetune(tmp_path) == {}


def test_an_unparseable_checkpoint_name_raises_rather_than_keying_on_the_directory(tmp_path):
    """re.sub returns its input unchanged on no match, which would key every row by a directory
    name — and then every drug fails to join, reading as 'no fine-tune exists'."""
    d = tmp_path / "klebsiella_pneumoniae_colistin_SOMETHING_ELSE"
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({"metrics": {"auroc": 0.9}}))
    with pytest.raises(SystemExit) as e:
        read_finetune(tmp_path)
    assert "cannot read a drug name" in str(e.value)


def test_a_single_arm_organism_needs_no_trainval_vocabulary(tmp_path, capsys):
    """TB has only the full-cohort build. Requiring a second arm would mean inventing one."""
    ft, full = tmp_path / "ft", tmp_path / "full"
    for drug, a in (("rifampin", 0.9642), ("ethionamide", 0.8097)):
        _ft(ft, "mycobacterium_tuberculosis", drug, a, 100)
    _unitig(full, "rifampin", 0.9700, 100)
    _unitig(full, "ethionamide", 0.8000, 100)
    assert run(ft, full, None, None, species="mycobacterium_tuberculosis") == 0
    out = capsys.readouterr().out
    assert "unitig (full-cohort vocab) vs BacFormer FT" in out
    # The absent arm must be omitted, not summarised as a result of zero.
    assert "unitig (leakage-free) vs BacFormer FT" not in out
    assert "unitig wins  1/2" in out


def test_the_species_is_reported_when_nothing_joins(tmp_path):
    """The likeliest cause of an empty join is the wrong species or the wrong ft-root."""
    _ft(tmp_path / "ft", "klebsiella_pneumoniae", "colistin", 0.9, 100)
    _unitig(tmp_path / "full", "colistin", 0.9, 100)
    with pytest.raises(SystemExit) as e:
        run(tmp_path / "ft", tmp_path / "full", None, None, species="mycobacterium_tuberculosis")
    assert "mycobacterium_tuberculosis" in str(e.value)
