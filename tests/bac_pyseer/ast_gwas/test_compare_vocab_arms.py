"""The paired comparison must refuse two arms that did not score the same genomes."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bac_pyseer.ast_gwas.compare_vocab_arms import compare_drug, load_arm, run

RNG = np.random.default_rng(0)


def _scores(path, ids, y, p):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, sample_ids=np.asarray(ids, dtype=np.str_),
             y_true=np.asarray(y, dtype=int), y_prob=np.asarray(p, dtype=float))
    return path


def _summary(path, **kw):
    path.parent.mkdir(parents=True, exist_ok=True)
    base = {"n_variants": 2_486_812, "n_unique_patterns": 960_320,
            "bonferroni_threshold": 5.2e-08, "n_significant": 9277,
            "genomic_inflation_lambda": 1.232, "pheno_var": 0.19397}
    base.update(kw)
    path.write_text(json.dumps(base))
    return path


def _cohort(n=120, seed=0):
    """A holdout with real signal, so AUROC is well away from 0.5 and the CI is meaningful."""
    rng = np.random.default_rng(seed)
    ids = [f"S{i:04d}" for i in range(n)]
    y = (rng.random(n) < 0.4).astype(int)
    strong = np.clip(y * 0.45 + rng.normal(0.3, 0.12, n), 0, 1)   # the flattered arm
    weak = np.clip(y * 0.30 + rng.normal(0.3, 0.16, n), 0, 1)     # the honest arm
    return ids, y, strong, weak


def test_pairs_by_id_regardless_of_row_order(tmp_path):
    """The two arms write rows in whatever order load_splits gave; pairing must not assume a match."""
    ids, y, a, b = _cohort()
    order = RNG.permutation(len(ids))
    _scores(tmp_path / "a.npz", ids, y, a)
    _scores(tmp_path / "b.npz", [ids[i] for i in order], y[order], b[order])

    row = compare_drug("colistin", tmp_path / "a.npz", tmp_path / "b.npz", n_boot=200)
    # Same comparison with both arms written in the same order must give an identical delta.
    _scores(tmp_path / "b2.npz", ids, y, b)
    same = compare_drug("colistin", tmp_path / "a.npz", tmp_path / "b2.npz", n_boot=200)
    assert row["delta"] == pytest.approx(same["delta"])
    assert row["full_cohort_auroc"] > row["trainval_vocab_auroc"]
    assert row["delta"] > 0          # positive == the old arm scored higher
    assert row["n_holdout"] == len(ids)


def test_load_arm_sorts_by_sample_id(tmp_path):
    _scores(tmp_path / "s.npz", ["S3", "S1", "S2"], [1, 0, 1], [0.9, 0.1, 0.8])
    ids, y, p = load_arm(tmp_path / "s.npz")
    assert list(ids) == ["S1", "S2", "S3"]
    assert list(y) == [0, 1, 1]
    assert list(p) == [0.1, 0.8, 0.9]


def test_wholly_different_holdout_sets_of_equal_size_are_refused(tmp_path):
    """The failure counts alone cannot see: same n, different genomes."""
    ids, y, a, b = _cohort(n=100)
    other = [f"T{i:04d}" for i in range(100)]
    _scores(tmp_path / "a.npz", ids, y, a)
    _scores(tmp_path / "b.npz", other, y, b)
    with pytest.raises(SystemExit) as e:
        compare_drug("colistin", tmp_path / "a.npz", tmp_path / "b.npz", n_boot=50)
    assert "substantially different holdout genomes" in str(e.value)
    assert "100 vs 100" in str(e.value)


def test_a_genome_the_rebuild_could_not_score_is_paired_out_and_counted(tmp_path):
    """~0.16% of the comparator's rows are genomes with no assembly and an all-zero feature row.

    The full-cohort arm scores them from the intercept; the rebuild's scanner cannot score them at
    all. Pairing on the intersection is right — but it must be counted, not done in silence.
    """
    ids, y, a, b = _cohort(n=200)
    _scores(tmp_path / "a.npz", ids, y, a)                       # comparator: all 200
    _scores(tmp_path / "b.npz", ids[:-1], y[:-1], b[:-1])        # rebuild: 199, one unscannable
    row = compare_drug("cefazolin", tmp_path / "a.npz", tmp_path / "b.npz", n_boot=200)
    assert row["n_holdout"] == 199                # paired on the intersection
    assert row["n_only_full_cohort"] == 1
    assert row["n_only_trainval_vocab"] == 0
    assert ids[-1] in row["unpaired_examples"]


def test_the_tolerance_is_a_fraction_not_a_free_pass(tmp_path):
    """Two genomes out of 200 is fine at 2%; twenty is not."""
    ids, y, a, b = _cohort(n=200)
    _scores(tmp_path / "a.npz", ids, y, a)
    _scores(tmp_path / "b.npz", ids[:-20], y[:-20], b[:-20])
    with pytest.raises(SystemExit, match="substantially different"):
        compare_drug("cefazolin", tmp_path / "a.npz", tmp_path / "b.npz", n_boot=50)


def test_disagreeing_labels_are_refused(tmp_path):
    ids, y, a, b = _cohort(n=100)
    flipped = y.copy()
    flipped[0] = 1 - flipped[0]
    _scores(tmp_path / "a.npz", ids, y, a)
    _scores(tmp_path / "b.npz", ids, flipped, b)
    with pytest.raises(SystemExit) as e:
        compare_drug("colistin", tmp_path / "a.npz", tmp_path / "b.npz", n_boot=50)
    assert "different label" in str(e.value)


def test_identical_arms_give_zero_delta_that_does_not_separate(tmp_path):
    ids, y, a, _ = _cohort()
    _scores(tmp_path / "a.npz", ids, y, a)
    _scores(tmp_path / "b.npz", ids, y, a)
    row = compare_drug("colistin", tmp_path / "a.npz", tmp_path / "b.npz", n_boot=300)
    assert row["delta"] == pytest.approx(0.0)
    assert row["separates_from_zero"] is False


def test_confound_columns_are_carried_from_both_arms(tmp_path):
    ids, y, a, b = _cohort()
    _scores(tmp_path / "a.npz", ids, y, a)
    _scores(tmp_path / "b.npz", ids, y, b)
    (tmp_path / "fr.json").write_text(json.dumps({"extra": {"n_unitigs": 40000, "C": 0.01}}))
    (tmp_path / "tr.json").write_text(json.dumps({"extra": {"n_unitigs": 51000, "C": 0.1}}))
    (tmp_path / "audit.json").write_text(json.dumps({"reflist": {"min_samples_floor": 12, "n_reflist": 1128}}))
    _summary(tmp_path / "fs.json", n_unique_patterns=960_320)
    _summary(tmp_path / "ts.json", n_unique_patterns=967_109)

    row = compare_drug(
        "colistin", tmp_path / "a.npz", tmp_path / "b.npz",
        full_results=tmp_path / "fr.json", trainval_results=tmp_path / "tr.json",
        trainval_audit=tmp_path / "audit.json",
        full_summary=tmp_path / "fs.json", trainval_summary=tmp_path / "ts.json", n_boot=100,
    )
    assert row["full_n_unitigs"] == 40000
    assert row["trainval_n_unitigs"] == 51000
    # Read from the summary FILE, under the key names the file really uses.
    assert row["full_n_unique_patterns"] == 960_320
    assert row["trainval_n_unique_patterns"] == 967_109
    # The MAF-floor confound must be visible in the row, not only in the docstring.
    assert row["trainval_min_samples"] == 12
    assert row["trainval_n_reflist"] == 1128


def test_summary_file_wins_over_the_unreliable_embedded_copy(tmp_path):
    """results.json's embedded gwas_summary is dropped silently when the path is wrong."""
    ids, y, a, b = _cohort()
    _scores(tmp_path / "a.npz", ids, y, a)
    _scores(tmp_path / "b.npz", ids, y, b)
    (tmp_path / "r.json").write_text(json.dumps(
        {"extra": {"n_unitigs": 1, "gwas_summary": {"n_unique_patterns": 111}}}))
    _summary(tmp_path / "s.json", n_unique_patterns=960_320)
    row = compare_drug("colistin", tmp_path / "a.npz", tmp_path / "b.npz",
                       full_results=tmp_path / "r.json", full_summary=tmp_path / "s.json", n_boot=50)
    assert row["full_n_unique_patterns"] == 960_320

    # ... and with no summary file it still falls back rather than reporting nothing.
    row2 = compare_drug("colistin", tmp_path / "a.npz", tmp_path / "b.npz",
                        full_results=tmp_path / "r.json", n_boot=50)
    assert row2["full_n_unique_patterns"] == 111


def test_run_lays_out_the_rebuild_directory_shape(tmp_path, capsys):
    """vocab arm lives at <vocab>/<drug>/<drug>/lr, full arm at <full>/<drug>/lr."""
    full, vocab = tmp_path / "kp", tmp_path / "kp_trainval_vocab"
    for drug in ("colistin", "ertapenem"):
        ids, y, a, b = _cohort(seed=hash(drug) % 100)
        _scores(full / drug / "lr" / "eval_scores.npz", ids, y, a)
        _scores(vocab / drug / drug / "lr" / "eval_scores.npz", ids, y, b)
    out = tmp_path / "cmp.csv"
    assert run(full, vocab, out, n_boot=100) == 0
    text = capsys.readouterr().out
    assert "2 drug(s) compared" in text
    assert "NOT all leakage" in text
    assert out.exists()
    assert len(out.read_text().strip().splitlines()) == 3


def test_run_reports_a_drug_missing_one_arm_rather_than_dropping_it(tmp_path, capsys):
    full, vocab = tmp_path / "kp", tmp_path / "kp_trainval_vocab"
    ids, y, a, b = _cohort()
    _scores(full / "colistin" / "lr" / "eval_scores.npz", ids, y, a)
    _scores(vocab / "colistin" / "colistin" / "lr" / "eval_scores.npz", ids, y, b)
    (vocab / "ertapenem").mkdir(parents=True)          # built, but no read-out yet
    assert run(full, vocab, None, n_boot=100) == 0
    text = capsys.readouterr().out
    assert "skipped: ertapenem" in text
    assert "1 drug(s) compared" in text


def test_run_with_nothing_to_compare_is_an_error(tmp_path):
    (tmp_path / "kp").mkdir()
    (tmp_path / "kp_trainval_vocab" / "colistin").mkdir(parents=True)
    with pytest.raises(SystemExit):
        run(tmp_path / "kp", tmp_path / "kp_trainval_vocab", None, n_boot=10)


def test_gwas_row_carries_both_arms_and_their_ratios(tmp_path):
    from bac_pyseer.ast_gwas.compare_vocab_arms import gwas_row
    a = _summary(tmp_path / "a.json")
    b = _summary(tmp_path / "b.json", n_variants=1_145_309, n_unique_patterns=967_109,
                 n_significant=5641, genomic_inflation_lambda=1.254)
    row = gwas_row("colistin", a, b)
    assert row["full_n_variants"] == 2_486_812
    assert row["trainval_n_variants"] == 1_145_309
    assert row["ratio_n_variants"] == pytest.approx(0.4605, abs=1e-3)
    # Patterns barely move while variants halve — that contrast is the point of the table.
    assert row["ratio_n_unique_patterns"] == pytest.approx(1.007, abs=1e-3)


def test_gwas_row_refuses_arms_whose_phenotype_differs(tmp_path):
    """The rebuild changes the vocabulary, never the phenotype. pheno_var is the control."""
    from bac_pyseer.ast_gwas.compare_vocab_arms import gwas_row
    a = _summary(tmp_path / "a.json")
    b = _summary(tmp_path / "b.json", pheno_var=0.2501)
    with pytest.raises(SystemExit) as e:
        gwas_row("colistin", a, b)
    assert "pheno_var differs" in str(e.value)


def test_run_gwas_reads_the_rebuild_layout(tmp_path, capsys):
    from bac_pyseer.ast_gwas.compare_vocab_arms import run_gwas
    full, vocab = tmp_path / "kp", tmp_path / "kp_trainval_vocab"
    for drug in ("colistin", "ertapenem"):
        _summary(full / drug / "gwas" / f"{drug}_gwas_summary.json")
        _summary(vocab / drug / drug / "gwas" / f"{drug}_gwas_summary.json", n_variants=1_145_309)
    out = tmp_path / "gwas.csv"
    assert run_gwas(full, vocab, out, None) == 0
    text = capsys.readouterr().out
    assert "2 drug(s)" in text
    assert out.exists()


def test_a_degenerate_bootstrap_does_not_crash_the_whole_comparison(tmp_path, capsys, monkeypatch):
    """paired_delta_ci drops separates_from_zero when every resample was single-class."""
    import bac_pyseer.ast_gwas.compare_vocab_arms as mod

    def degenerate(y_true, a, b, **kw):
        return {"delta": 0.0, "ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot_valid": 0}

    monkeypatch.setattr(mod, "paired_delta_ci", degenerate)
    full, vocab = tmp_path / "kp", tmp_path / "kp_trainval_vocab"
    ids, y, a, b = _cohort()
    _scores(full / "colistin" / "lr" / "eval_scores.npz", ids, y, a)
    _scores(vocab / "colistin" / "colistin" / "lr" / "eval_scores.npz", ids, y, b)
    assert mod.run(full, vocab, None, n_boot=10) == 0
    assert "0 separate from zero" in capsys.readouterr().out


def test_regenerated_summary_wins_over_the_combine_phases_own(tmp_path):
    """<drug>/<drug>_gwas_summary.json beats <drug>/gwas/<drug>_gwas_summary.json.

    Kp ertapenem has both and they disagree: the gwas/ copy predates the combine-phase fix and left
    pheno_var at the 0.249 default, while the regenerated copy computed it. The write-ups quote the
    regenerated one, so reading gwas/ would put stale numbers in the table of record.
    """
    from bac_pyseer.ast_gwas.compare_vocab_arms import resolve_summary
    d = tmp_path / "ertapenem"
    (d / "gwas").mkdir(parents=True)
    _summary(d / "gwas" / "ertapenem_gwas_summary.json", pheno_var=0.249, n_variants=1)
    _summary(d / "ertapenem_gwas_summary.json", pheno_var=0.2247794905842714, n_variants=3_371_827)
    got = resolve_summary(d, "ertapenem")
    assert got.parent == d
    assert json.loads(got.read_text())["n_variants"] == 3_371_827


def test_resolve_summary_falls_back_to_the_gwas_copy(tmp_path):
    from bac_pyseer.ast_gwas.compare_vocab_arms import resolve_summary
    d = tmp_path / "colistin"
    (d / "gwas").mkdir(parents=True)
    _summary(d / "gwas" / "colistin_gwas_summary.json")
    assert resolve_summary(d, "colistin").parent.name == "gwas"
    assert resolve_summary(tmp_path / "absent", "x") is None


def test_an_uncomputed_pheno_var_warns_rather_than_failing(tmp_path):
    """pheno_var_source=default means "never measured", not "the labels differ"."""
    from bac_pyseer.ast_gwas.compare_vocab_arms import gwas_row
    a = _summary(tmp_path / "a.json", pheno_var=0.249, pheno_var_source="default")
    b = _summary(tmp_path / "b.json", pheno_var=0.2247794905842714, pheno_var_source="computed:/x")
    row = gwas_row("ertapenem", a, b)          # must not raise
    assert row["pheno_var_uncomputed"] == "full_cohort"


def test_two_computed_but_differing_pheno_vars_still_fail(tmp_path):
    """The real label-mismatch case must stay fatal."""
    from bac_pyseer.ast_gwas.compare_vocab_arms import gwas_row
    a = _summary(tmp_path / "a.json", pheno_var=0.19, pheno_var_source="computed:/x")
    b = _summary(tmp_path / "b.json", pheno_var=0.25, pheno_var_source="computed:/y")
    with pytest.raises(SystemExit) as e:
        gwas_row("colistin", a, b)
    assert "both were computed" in str(e.value)
