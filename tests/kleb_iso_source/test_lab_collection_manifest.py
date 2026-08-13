"""Tests for the lab-collection manifest build.

The manifest is the join that decides which physical isolates get scored and which of those scores
are honest predictions rather than recall of training labels. The failure modes worth guarding are
all silent ones: a duplicate accession scored twice, a genome quietly dropped for want of an
embedding, or a memorised training genome presented as a prediction.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kleb_iso_source.build_lab_collection_manifest import (
    ID_COL,
    UNKNOWN_SL,
    attach_cohort_splits,
    attach_sublineage,
    build_exclusions,
    flag_availability,
    resolve_duplicates,
)


def _rows(accessions: list[str], **extra) -> pd.DataFrame:
    df = pd.DataFrame({ID_COL: accessions, "strain": [f"s{i}" for i in range(len(accessions))],
                       "LabID": [f"L{i}" for i in range(len(accessions))]})
    for k, v in extra.items():
        df[k] = v
    return df


def _write_metadata(tmp_path, rows):
    p = tmp_path / "metadata_v2.tsv"
    pd.DataFrame(rows).to_csv(p, sep="\t", index=False)
    return p


def _write_cohort(tmp_path, cohort, rows):
    d = tmp_path / "blood_faeces" / cohort / "kpsc_human"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / "binary_blood_vs_faeces_with_split.csv", index=False)
    return tmp_path


# --------------------------------------------------------------------------- sublineage join


def test_sublineage_join_labels_unmatched_as_unknown_not_dropped(tmp_path):
    df = _rows(["A", "B", "C"])
    meta = _write_metadata(tmp_path, [
        {"Sample": "A", "Sublineage": "SL258", "Clonal group": "CG258", "country_parsed": "UK"},
        {"Sample": "B", "Sublineage": "SL147", "Clonal group": "CG147", "country_parsed": "IT"},
    ])
    out = attach_sublineage(df, meta, min_coverage=0.5)
    assert len(out) == 3, "an unmatched genome must survive the join"
    assert out.set_index(ID_COL).loc["C", "Sublineage"] == UNKNOWN_SL
    assert out.set_index(ID_COL).loc["A", "Sublineage"] == "SL258"


def test_sublineage_join_fails_loudly_on_a_broken_key(tmp_path):
    """A key mismatch would otherwise look like 'no lineage data', a biology-shaped answer."""
    df = _rows(["A", "B", "C", "D"])
    meta = _write_metadata(tmp_path, [{"Sample": "zzz", "Sublineage": "SL258",
                                       "Clonal group": "CG258", "country_parsed": "UK"}])
    with pytest.raises(SystemExit, match="matched only"):
        attach_sublineage(df, meta, min_coverage=0.5)


# --------------------------------------------------------------------------- split provenance


def test_both_cohort_splits_are_recorded_and_label_is_shared(tmp_path):
    df = _rows(["A", "B", "C"])
    _write_cohort(tmp_path, "sampled_country_2_1_all", [
        {"Sample": "A", "blood_vs_faeces_label": 1, "train_val_eval": "train"},
        {"Sample": "B", "blood_vs_faeces_label": 0, "train_val_eval": "evaluate"},
    ])
    _write_cohort(tmp_path, "all_samples", [
        {"Sample": "A", "blood_vs_faeces_label": 1, "train_val_eval": "train"},
        {"Sample": "B", "blood_vs_faeces_label": 0, "train_val_eval": "train"},
    ])
    out = attach_cohort_splits(df, tmp_path).set_index(ID_COL)

    # B is held out under pooled but TRAINED ON under all_samples — the whole reason both are kept.
    assert out.loc["B", "pooled_split"] == "evaluate"
    assert out.loc["B", "all_samples_split"] == "train"
    assert out.loc["C", "pooled_split"] == "unseen"
    assert out.loc["A", "true_label"] == 1
    assert pd.isna(out.loc["C", "true_label"])


def test_conflicting_labels_across_cohorts_raise(tmp_path):
    df = _rows(["A"])
    _write_cohort(tmp_path, "sampled_country_2_1_all",
                  [{"Sample": "A", "blood_vs_faeces_label": 1, "train_val_eval": "train"}])
    _write_cohort(tmp_path, "all_samples",
                  [{"Sample": "A", "blood_vs_faeces_label": 0, "train_val_eval": "train"}])
    with pytest.raises(ValueError, match="conflicting labels"):
        attach_cohort_splits(df, tmp_path)


# --------------------------------------------------------------------------- duplicates


def test_duplicate_accession_scores_once_but_keeps_both_rows(tmp_path):
    df = _rows(["A", "A", "B"])
    df["has_embedding"] = [False, True, True]
    df["has_assembly"] = [False, True, True]
    out, dropped = resolve_duplicates(df)

    assert len(out) == 3, "both physical tubes stay in the table for the lab to disambiguate"
    assert out["is_scoring_row"].sum() == 2, "the genome is scored once, not twice"
    kept = out[out[ID_COL].eq("A") & out["is_scoring_row"]].iloc[0]
    assert kept["has_embedding"] and kept["has_assembly"], "the more complete duplicate is kept"
    assert out[out[ID_COL].eq("A")]["duplicate_accession"].all()
    assert len(dropped) == 1


# --------------------------------------------------------------------------- availability + exclusions


def test_missing_embedding_is_flagged_not_dropped(tmp_path):
    (tmp_path / "A_esm_embeddings.pt").write_bytes(b"x")
    df = _rows(["A", "B"])
    df["assembly_path"] = [str(tmp_path / "a.fa"), None]
    out = flag_availability(df, tmp_path, check_assemblies=False)
    assert len(out) == 2
    assert out.set_index(ID_COL).loc["A", "has_embedding"]
    assert not out.set_index(ID_COL).loc["B", "has_embedding"]


def test_exclusions_record_every_reason(tmp_path):
    df = _rows(["A", "B"])
    df["has_embedding"] = [True, False]
    df["has_assembly"] = [True, False]
    df["Sublineage"] = ["SL258", UNKNOWN_SL]
    df["duplicate_accession"] = False
    df["is_scoring_row"] = True

    excl = build_exclusions(df)
    assert list(excl[ID_COL]) == ["B"], "a fully-resolved genome is not an exclusion"
    reason = excl.iloc[0]["reason"]
    assert "no ESM embedding" in reason and "no assembly" in reason and "no Sublineage" in reason
