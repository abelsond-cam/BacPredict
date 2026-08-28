"""Tests for the per-drug leakage audit — every assertion behind the train+validate vocabulary claim.

Each case here is a way the rebuild can produce a full-cohort vocabulary while looking correct: a
reflist built without the split filter, a stale GGCAT build reused from another cohort's ``OUT_DIR``,
clusters carried over from the old run. None of them raise on their own, all of them yield a
plausible AUROC, and the only thing that separates them from a real result is an assertion that reads
the tool's own output. So the tests are written against the failure, not the success.
"""

from __future__ import annotations

import gzip
import json

import pandas as pd
import pytest

from bac_pyseer.ast_gwas.leakage_audit import (
    audit_clusters,
    audit_design,
    audit_mash,
    audit_reflist,
    audit_vocabulary,
    update_audit,
)

TRAIN = [f"tr{i}" for i in range(6)]
VALIDATE = [f"va{i}" for i in range(2)]
HOLDOUT = [f"ho{i}" for i in range(3)]


@pytest.fixture
def split_table(tmp_path):
    rows = [
        *({"Sample": s, "ast_label": float(i % 2), "split": "train"} for i, s in enumerate(TRAIN)),
        *({"Sample": s, "ast_label": float(i % 2), "split": "validate"} for i, s in enumerate(VALIDATE)),
        *({"Sample": s, "ast_label": float(i % 2), "split": "holdout"} for i, s in enumerate(HOLDOUT)),
    ]
    path = tmp_path / "drugx_split.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _reflist(tmp_path, samples, name="refs.tsv"):
    path = tmp_path / name
    path.write_text("".join(f"{s}\t/asm/{s}.fa.gz\n" for s in samples))
    return path


def _color_names(tmp_path, samples, name="color_names.jsonl"):
    path = tmp_path / name
    path.write_text("".join(
        json.dumps({"color_index": i, "color_name": s}) + "\n" for i, s in enumerate(samples)
    ))
    return path


# --------------------------------------------------------------------------------------------------
# reflist
# --------------------------------------------------------------------------------------------------
def test_a_trainval_reflist_passes_and_records_the_maf_floor(tmp_path, split_table):
    """The floor is recorded because it changes between runs and partly explains the AUROC delta."""
    payload = audit_reflist(_reflist(tmp_path, [*TRAIN, *VALIDATE]), split_table)
    assert payload["n_holdout_in_reflist"] == 0
    assert payload["n_reflist"] == 8
    assert payload["min_samples_floor"] == 1  # ceil(0.01 * 8)


def test_a_reflist_built_without_the_split_filter_is_caught(tmp_path, split_table):
    """The commonest way to get this wrong: forget --splits and build over the whole cohort."""
    with pytest.raises(SystemExit, match="holdout genome"):
        audit_reflist(_reflist(tmp_path, [*TRAIN, *VALIDATE, *HOLDOUT]), split_table)


def test_a_duplicated_genome_is_caught(tmp_path, split_table):
    """A doubled colour would silently double a genome's weight in the graph."""
    with pytest.raises(SystemExit, match="more than once"):
        audit_reflist(_reflist(tmp_path, [*TRAIN, *VALIDATE, TRAIN[0]]), split_table)


def test_trainval_genomes_without_an_assembly_are_recorded_not_rejected(tmp_path, split_table):
    """They cannot be coloured and cannot be tested, so they are a fact about the run, not an error."""
    payload = audit_reflist(_reflist(tmp_path, [*TRAIN[:4], *VALIDATE]), split_table)
    assert payload["n_trainval_without_assembly"] == 2


# --------------------------------------------------------------------------------------------------
# vocabulary — from GGCAT's own colour record
# --------------------------------------------------------------------------------------------------
def test_the_graph_colours_must_match_the_reflist(tmp_path, split_table):
    trainval = [*TRAIN, *VALIDATE]
    payload = audit_vocabulary(_color_names(tmp_path, trainval), _reflist(tmp_path, trainval), split_table)
    assert payload["n_holdout_coloured"] == 0
    assert payload["n_missing_from_graph"] == payload["n_extra_in_graph"] == 0


def test_a_stale_full_cohort_build_reused_from_the_wrong_out_dir_is_caught(tmp_path, split_table):
    """run_ggcat_unitigs.sh skips the build when the artifacts exist, so this fails silently."""
    trainval = [*TRAIN, *VALIDATE]
    stale = _color_names(tmp_path, [*trainval, *HOLDOUT])
    with pytest.raises(SystemExit, match="supplied colours"):
        audit_vocabulary(stale, _reflist(tmp_path, trainval), split_table)


def test_a_graph_missing_reflist_genomes_is_caught(tmp_path, split_table):
    """A build from a different, smaller cohort — the same trap in the other direction."""
    trainval = [*TRAIN, *VALIDATE]
    with pytest.raises(SystemExit, match="absent from the graph"):
        audit_vocabulary(_color_names(tmp_path, trainval[:5]), _reflist(tmp_path, trainval), split_table)


def test_the_matrix_head_is_scanned_for_holdout_carriers_independently(tmp_path, split_table):
    """Belt and braces: a matrix inherited from another build would carry holdout tokens."""
    trainval = [*TRAIN, *VALIDATE]
    matrix = tmp_path / "unitigs.pyseer.gz"
    with gzip.open(matrix, "wt") as fh:
        fh.write(f"ACGT | {TRAIN[0]}:1 {HOLDOUT[0]}:1\n")
    with pytest.raises(SystemExit, match="appear as carriers"):
        audit_vocabulary(_color_names(tmp_path, trainval), _reflist(tmp_path, trainval),
                         split_table, matrix_gz=matrix)


# --------------------------------------------------------------------------------------------------
# clusters and mash
# --------------------------------------------------------------------------------------------------
def test_clusters_must_cover_exactly_the_reflist(tmp_path):
    trainval = [*TRAIN, *VALIDATE]
    path = tmp_path / "lineage_clusters.tsv"
    path.write_text("".join(f"{s}\t{'sl1' if i < 5 else 'other'}\n" for i, s in enumerate(trainval)))
    payload = audit_clusters(path, _reflist(tmp_path, trainval))
    assert payload["n_clusters"] == 2 and payload["n_in_other"] == 3

    path.write_text("".join(f"{s}\tsl1\n" for s in [*trainval, *HOLDOUT]))
    with pytest.raises(SystemExit, match="does not cover the reflist"):
        audit_clusters(path, _reflist(tmp_path, trainval))


def test_a_fresh_sketch_must_equal_the_old_triangle_subset(tmp_path):
    """Subsetting is mathematically identical to re-sketching; the audit turns that into a number."""
    ids = [*TRAIN, *VALIDATE]
    frame = pd.DataFrame(1.0, index=ids, columns=ids)
    fresh, ref = tmp_path / "fresh.tsv", tmp_path / "ref.tsv"
    frame.to_csv(fresh, sep="\t")
    frame.to_csv(ref, sep="\t")
    assert audit_mash(fresh, ref)["max_abs_diff"] == 0.0

    perturbed = frame.copy()
    perturbed.iloc[0, 1] = 0.5
    perturbed.to_csv(fresh, sep="\t")
    assert audit_mash(fresh, ref)["max_abs_diff"] == pytest.approx(0.5)


def test_sections_accumulate_so_a_partial_audit_is_visibly_partial(tmp_path, split_table):
    """A missing section must read as 'not checked', which needs one file rather than several."""
    audit = tmp_path / "leakage_audit.json"
    update_audit(audit, "reflist", audit_reflist(_reflist(tmp_path, [*TRAIN, *VALIDATE]), split_table))
    update_audit(audit, "clusters", {"n_clusters": 3})
    written = json.loads(audit.read_text())
    assert set(written) == {"reflist", "clusters"}
    assert "vocabulary" not in written


def test_the_design_stage_refuses_a_merge_whose_scanner_was_never_checked(tmp_path):
    """Zero mismatches out of zero comparisons is not a passed gate, and must not be filed as one."""
    manifest = tmp_path / "merge_manifest.json"
    manifest.write_text(json.dumps({
        "rows_from_ggcat": 100, "rows_from_scanner": 40, "n_features": 500, "nnz": 9000,
        "verification": {"n_shared": 0, "n_mismatch_cells": None},
        "holdout_coverage": {"checked": True, "ratio": 0.8},
        "shard_completeness": {"n_shards": 8, "shard_files": ["a"] * 8},
    }))
    with pytest.raises(SystemExit, match="never checked"):
        audit_design(manifest)


def test_the_design_stage_carries_the_merge_gates_into_the_audit_file(tmp_path):
    """One file should account for the whole run, not send a reader hunting through manifests."""
    manifest = tmp_path / "merge_manifest.json"
    manifest.write_text(json.dumps({
        "rows_from_ggcat": 100, "rows_from_scanner": 40, "n_features": 500, "nnz": 9000,
        "verification": {"n_shared": 100, "n_mismatch_cells": 0},
        "holdout_coverage": {"checked": True, "ratio": 0.81},
        "shard_completeness": {"n_shards": 8, "shard_files": ["a"] * 8},
    }))
    payload = audit_design(manifest)
    assert payload["verification"]["n_mismatch_cells"] == 0
    assert payload["holdout_coverage"]["ratio"] == 0.81
    assert payload["rows_from_scanner"] == 40
    assert payload["shard_completeness"]["n_shards"] == 8


def test_the_design_stage_refuses_a_merge_that_cannot_show_its_scan_was_complete(tmp_path):
    """A manifest predating the shard gate cannot prove the scan was whole.

    An incomplete strided scan thins the holdout evenly instead of truncating it, so the design that
    results looks entirely healthy — the scanner check passes, the carrier ratio passes, and the
    read-out reports a clean AUROC on a fraction of the genomes. Absence of the record is therefore
    not evidence the scan passed, and must not be filed as though it were.
    """
    manifest = tmp_path / "merge_manifest.json"
    manifest.write_text(json.dumps({
        "rows_from_ggcat": 100, "rows_from_scanner": 40, "n_features": 500, "nnz": 9000,
        "verification": {"n_shared": 100, "n_mismatch_cells": 0},
        "holdout_coverage": {"checked": True, "ratio": 0.81},
    }))
    with pytest.raises(SystemExit, match="shard_completeness"):
        audit_design(manifest)
