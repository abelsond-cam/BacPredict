"""The trainer reads what the split builder writes: ``build_amr_holdout_splits`` → ``finetune_amr --split-table``.

Built on synthetic inputs with the real builder (two species, a held-out ST each, an exclusion, a duplicate-isolate
group), then read by the trainer's own split resolution and run once end to end with the fakes of
``test_finetune_amr_split_table``. Skipped where the builder is absent.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from bacpredict.engine.finetune import finetune_amr as ft

from .test_finetune_amr_split_table import _write_pt, fakes  # noqa: F401 (fakes is a fixture)

builder = pytest.importorskip("bacpredict.engine.splits.build_amr_holdout_splits")

DRUGS = ["ciprofloxacin", "meropenem"]


def _species_inputs(tmp_path, name: str, prefix: str, scheme: str, n: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    samples = [f"{prefix}{i:04d}" for i in range(n)]
    st = np.where(np.arange(n) < 15, "235", rng.choice(["1", "2", "3", "-"], n))
    labels = pd.DataFrame({"Sample": samples, **{d: rng.integers(0, 2, n).astype(float) for d in DRUGS}})
    labels.loc[rng.random(n) < 0.2, "meropenem"] = np.nan
    labels.loc[20, "ciprofloxacin"] = 0.5
    d = tmp_path / name
    d.mkdir()
    labels.to_csv(d / "binary_ast.csv", index=False)
    pd.DataFrame({"Sample": samples}).to_csv(d / "eligible.csv", index=False)
    pd.DataFrame({"Sample": samples, "mlst_scheme": scheme, "st": st}).to_csv(d / "qc.csv", index=False)
    for k, s in enumerate(samples):
        _write_pt(tmp_path / "esm", s, torch.bfloat16 if k % 2 else torch.float32)
    return f"{name}={d}/binary_ast.csv,{d}/eligible.csv,{d}/qc.csv"


@pytest.fixture
def arm(tmp_path):
    (tmp_path / "esm").mkdir()
    pa = _species_inputs(tmp_path, "paeruginosa", "PA", "paeruginosa", 120, 1)
    en = _species_inputs(tmp_path, "enterobacterales", "EN", "ecoli_achtman_4", 150, 2)
    (tmp_path / "exclude.tsv").write_text("Sample\treason\nPA0100\ttest\n")
    (tmp_path / "groups.csv").write_text("Sample,group\nPA0050,iso1\nPA0051,iso1\n")
    common = ["--holdout-st", "paeruginosa=paeruginosa:235", "--exclude", str(tmp_path / "exclude.tsv"),
              "--groups", str(tmp_path / "groups.csv"), "--drugs", *DRUGS]
    builder.main(["--species", pa, *common, "--out-dir", str(tmp_path / "pa")])
    builder.main(["--species", pa, "--species", en, *common, "--holdout-st", "enterobacterales=ecoli_achtman_4:235",
                  "--assert-same-as", f"paeruginosa={tmp_path / 'pa'}", "--out-dir", str(tmp_path / "pa_en")])
    return tmp_path


@pytest.mark.parametrize("arm_dir", ["pa", "pa_en"])
def test_the_trainers_splits_are_the_tables_splits(arm, arm_dir):
    for drug in DRUGS:
        table = arm / arm_dir / f"{drug}_split.csv"
        df = pd.read_csv(table, dtype={"Sample": str})
        clean = df[df["ast_label"].isin([0, 1])]
        s = ft.resolve_split_table(table)
        for name, ids in (("train", s.train), ("validate", s.validate), ("holdout", s.holdout),
                          ("holdout_st", s.extra["holdout_st"])):
            assert ids == clean.loc[clean["split"] == name, "Sample"].tolist(), (drug, name)
        assert s.species == dict(zip(clean["Sample"], clean["species"], strict=True))
        assert s.multi_species == (arm_dir == "pa_en")
        assert "PA0100" not in s.label_map  # excluded
        assert ft.missing_embeddings(list(s.label_map), arm / "esm") == []


def test_a_cross_species_arm_trains_end_to_end_and_scores_psa_on_the_same_holdouts(arm, fakes):  # noqa: F811
    out = arm / "run"
    table = arm / "pa_en" / "ciprofloxacin_split.csv"
    ft.run("base", str(arm / "esm"), str(out), None, task="pa_entero_ast", drug="ciprofloxacin", num_workers=0,
           split_table=str(table))
    res = json.loads((out / "results.json").read_text())
    pa_only = ft.resolve_split_table(arm / "pa" / "ciprofloxacin_split.csv")
    for name in ("holdout", "holdout_st"):
        by = res["holdouts"][name]["by_species"]
        assert set(by) == {"enterobacterales", "paeruginosa"}
        # the PsA rows of the cross-species arm are the PsA-only arm's: same genomes scored for PsA in both arms
        assert by["paeruginosa"]["n_samples"] == len(pa_only.holdouts[name])
        npz = np.load(out / res["holdouts"][name]["scores"])
        assert sorted(npz["sample_ids"][npz["species"] == "paeruginosa"]) == sorted(pa_only.holdouts[name])
