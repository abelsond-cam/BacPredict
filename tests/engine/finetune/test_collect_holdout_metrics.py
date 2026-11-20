"""``collect_holdout_metrics``: one row per drug × holdout (× species), and a missing drug is a row, not a gap."""

from __future__ import annotations

import json

from bacpredict.engine.finetune.collect_holdout_metrics import collect


def _metrics(n: int, n_r: int, auroc: float) -> dict:
    return {"n_samples": n, "auroc": auroc, "auprc": 0.5, "sensitivity": 0.5, "specificity": 0.5,
            "balanced_accuracy": 0.5, "f1": 0.5, "prevalence": n_r / n,
            "per_class": {"0": {"n": n - n_r}, "1": {"n": n_r}}}


def _write(models, drug: str, payload: dict) -> None:
    (models / drug).mkdir(parents=True)
    (models / drug / "results.json").write_text(json.dumps(payload))


def test_rows_per_holdout_and_species_and_missing_drugs(tmp_path):
    split = {"source": "split_table", "table_md5": "abc"}
    _write(tmp_path, "ciprofloxacin", {
        "split": split, "model": {"git_sha": "s"}, "metrics": _metrics(10, 4, 0.9),
        "holdouts": {"holdout": {"scores": "eval_scores.npz", "metrics": _metrics(10, 4, 0.9)},
                     "holdout_st": {"scores": "eval_scores_holdout_st.npz", "metrics": _metrics(6, 3, 0.7),
                                    "by_species": {"paeruginosa": _metrics(6, 3, 0.7)}}}})
    _write(tmp_path, "rifampin", {"split": {"source": "kfold"}, "model": {}, "metrics": _metrics(8, 2, 0.8)})
    t = collect(tmp_path, ["ciprofloxacin", "meropenem", "rifampin"])

    cip = t[t.drug == "ciprofloxacin"]
    assert list(zip(cip.holdout, cip.species, strict=True)) == [
        ("holdout", "all"), ("holdout_st", "all"), ("holdout_st", "paeruginosa")]
    st = cip[(cip.holdout == "holdout_st") & (cip.species == "all")].iloc[0]
    assert (st.n_samples, st.n_resistant, st.n_susceptible, st.auroc) == (6, 3, 3, 0.7)
    assert st.scores.endswith("ciprofloxacin/eval_scores_holdout_st.npz") and st.status == "ok"
    assert t[t.drug == "meropenem"].status.tolist() == ["missing"]
    assert t[t.drug == "rifampin"].status.tolist() == ["no_holdouts_block"]
    assert str(t["n_samples"].dtype) == "Int64"
