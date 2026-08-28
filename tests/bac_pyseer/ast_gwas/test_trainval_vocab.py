"""Tests for the leakage-free rebuild: a train+validate vocabulary, and a holdout scored from sequence.

The rebuild removes the last path by which holdout genomes shaped the model — they no longer supply
colours to the GGCAT graph. That has a consequence the pipeline was never built for: the unitig matrix
then contains **no holdout carriers at all**, so asking it for holdout rows returns zeros rather than an
error. A logistic regression fitted on that still trains, still scores, and still reports a
well-formed AUROC of about 0.5, with nothing anywhere naming the cause.

So the tests here are mostly about failing loudly. Each one pins a step where the wrong answer would
otherwise be indistinguishable from a real result: a reflist that quietly kept its holdout genomes, a
scanner that disagrees with the operator which produced the training rows, and a holdout that was
never really scored.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from bac_pyseer.ast_gwas.resolve_ast_assemblies import cohort_samples
from bac_pyseer.ast_gwas.unitig_design_matrix import check_holdout_coverage
from bac_pyseer.ast_gwas.unitig_kmer_presence import (
    _main_cli,
    score_samples,
    verify_against_ggcat,
)
from bac_pyseer.kleb_iso_source.unitig_placement import _revcomp

TRAIN = ["s_tr0", "s_tr1", "s_tr2", "s_tr3"]
VALIDATE = ["s_va0", "s_va1"]
HOLDOUT = ["s_ho0", "s_ho1", "s_ho2"]


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def _rand(rng, n: int) -> str:
    return "".join(rng.choice(list("ACGT"), n))


@pytest.fixture
def split_table(tmp_path):
    """A ``<drug>_split.csv`` in the shape ``load_splits`` requires, floats included."""
    rows = [
        *({"Sample": s, "ast_label": float(i % 2), "split": "train"} for i, s in enumerate(TRAIN)),
        *({"Sample": s, "ast_label": float(i % 2), "split": "validate"} for i, s in enumerate(VALIDATE)),
        *({"Sample": s, "ast_label": float(i % 2), "split": "holdout"} for i, s in enumerate(HOLDOUT)),
    ]
    path = tmp_path / "drugx_split.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def cohort(tmp_path, rng, split_table):
    """Assemblies, features, a reflist, and the GGCAT design over train+validate only.

    Every genome is given a random backbone plus whichever features it is meant to carry, so presence
    is a property of the sequence and the scanner has to recover it rather than be told it.
    """
    feats = [_rand(rng, 60) for _ in range(5)]
    # feature f is carried by sample i iff carried[i][f]; the holdout carries plenty, as a real one would
    carried = {
        "s_tr0": [0, 1], "s_tr1": [0, 2], "s_tr2": [1, 3, 4], "s_tr3": [0, 1, 2],
        "s_va0": [2, 3], "s_va1": [0, 4],
        "s_ho0": [0, 1, 2], "s_ho1": [1, 3], "s_ho2": [0, 2, 4],
    }
    asm_dir = tmp_path / "asm"
    asm_dir.mkdir()
    reflist = tmp_path / "refs.tsv"
    lines = []
    for sample, idxs in carried.items():
        # one feature reverse-complemented, so strand handling is exercised on real inputs
        pieces = [_revcomp(feats[j]) if j % 2 else feats[j] for j in idxs]
        seq = _rand(rng, 300) + "".join(p + _rand(rng, 300) for p in pieces)
        path = asm_dir / f"{sample}.fa.gz"
        with gzip.open(path, "wt") as fh:
            fh.write(f">{sample}_c1\n{seq}\n")
        lines.append(f"{sample}\t{path}\n")
    reflist.write_text("".join(lines))

    id_map = pd.DataFrame({
        "unitig_idx": np.arange(len(feats)),
        "variant": feats,
        "af": np.linspace(0.2, 0.8, len(feats)),
    })
    design = tmp_path / "design"
    design.mkdir()
    id_map.to_csv(design / "id_map.tsv", sep="\t", index=False)
    trainval = [*TRAIN, *VALIDATE]
    (design / "samples.txt").write_text("".join(f"{s}\n" for s in trainval))
    dense = np.zeros((len(trainval), len(feats)), dtype=np.int8)
    for i, s in enumerate(trainval):
        dense[i, carried[s]] = 1
    sparse.save_npz(design / "presence.npz", sparse.csr_matrix(dense))
    return {"design": design, "reflist": reflist, "feats": feats, "carried": carried, "id_map": id_map}


# --------------------------------------------------------------------------------------------------
# the reflist: no holdout genome may enter the vocabulary
# --------------------------------------------------------------------------------------------------
def test_trainval_reflist_excludes_every_holdout_genome(split_table):
    """The assertion the whole rebuild rests on — GGCAT never sees a holdout assembly."""
    samples = cohort_samples("kp", split_table=split_table, splits=("train", "validate"))
    assert samples == [*TRAIN, *VALIDATE]
    assert not set(samples) & set(HOLDOUT)


def test_restricting_splits_without_a_split_table_is_rejected_not_ignored(tmp_path):
    """Silently returning the whole cohort would build a full-cohort vocabulary under a clean name."""
    with pytest.raises(SystemExit, match="needs --split-table"):
        cohort_samples("kp", ast_sheet=tmp_path / "nonexistent.csv", splits=("train", "validate"))


def test_an_unknown_split_name_is_rejected(split_table):
    """A typo must not silently select nothing."""
    with pytest.raises(SystemExit, match="unknown split"):
        cohort_samples("kp", split_table=split_table, splits=("train", "valdiate"))


# --------------------------------------------------------------------------------------------------
# the scanner
# --------------------------------------------------------------------------------------------------
def test_score_samples_recovers_carriage_in_id_map_column_order(cohort):
    """Column j must be id_map row j — that is what lets a coefficient be traced to its GWAS row."""
    samples = [*TRAIN, *VALIDATE, *HOLDOUT]
    path_of = dict(
        line.split("\t") for line in cohort["reflist"].read_text().splitlines()
    )
    scored = score_samples(cohort["feats"], samples, path_of, progress_every=0).toarray()
    expected = np.zeros_like(scored)
    for i, s in enumerate(samples):
        expected[i, cohort["carried"][s]] = 1
    np.testing.assert_array_equal(scored, expected)


def test_verify_reports_zero_mismatches_when_the_two_operators_agree(cohort):
    """The self-verifying property: scanning train+validate must reproduce GGCAT's own colouring."""
    trainval = [*TRAIN, *VALIDATE]
    path_of = dict(line.split("\t") for line in cohort["reflist"].read_text().splitlines())
    scan = score_samples(cohort["feats"], trainval, path_of, progress_every=0)
    ggcat = sparse.load_npz(cohort["design"] / "presence.npz").tocsr()
    report = verify_against_ggcat(scan, trainval, ggcat, trainval)
    assert report["n_shared"] == len(trainval)
    assert report["n_mismatch_cells"] == 0
    assert report["scanner_present"] == report["ggcat_present"] > 0


def test_verify_counts_a_disagreement_rather_than_averaging_it_away(cohort):
    """One flipped cell must surface as one mismatch, with the genome named."""
    trainval = [*TRAIN, *VALIDATE]
    ggcat = sparse.load_npz(cohort["design"] / "presence.npz").tolil()
    scan = ggcat.copy()
    scan[1, 4] = 1 - scan[1, 4]
    report = verify_against_ggcat(scan.tocsr(), trainval, ggcat.tocsr(), trainval)
    assert report["n_mismatch_cells"] == 1
    assert report["n_mismatch_genomes"] == 1
    assert report["examples"][0]["sample"] == trainval[1]


def test_verify_records_a_skipped_comparison_instead_of_passing_silently(cohort):
    """A holdout-only scan shares no genome with the design, so it can check nothing — and must say so.

    Zero mismatches out of zero comparisons is indistinguishable from a passed gate, which is why the
    shared-genome count is reported alongside it and the mismatch count is ``None`` rather than 0.
    """
    ggcat = sparse.load_npz(cohort["design"] / "presence.npz").tocsr()
    holdout_only_scan = sparse.csr_matrix((len(HOLDOUT), ggcat.shape[1]), dtype=np.int8)
    report = verify_against_ggcat(holdout_only_scan, HOLDOUT, ggcat, [*TRAIN, *VALIDATE])
    assert report["n_shared"] == 0
    assert report["n_mismatch_cells"] is None


# --------------------------------------------------------------------------------------------------
# the all-zero-holdout guard
# --------------------------------------------------------------------------------------------------
def _coverage_fixture(n_train: int, n_hold: int, hold_cols: int, n_cols: int = 6):
    """``(matrix, ids, train_ids, holdout_ids)`` where train carries everything and holdout ``hold_cols``."""
    train = [f"t{i}" for i in range(n_train)]
    hold = [f"h{i}" for i in range(n_hold)]
    dense = np.zeros((n_train + n_hold, n_cols), dtype=np.int8)
    dense[:n_train, :] = 1
    dense[n_train:, :hold_cols] = 1
    return sparse.csr_matrix(dense), [*train, *hold], train, hold


def test_an_unscored_holdout_fails_the_guard_rather_than_scoring_auroc_half():
    """All-zero holdout rows are the one failure that produces a clean, publishable, wrong number."""
    matrix, ids, train, hold = _coverage_fixture(100, 40, hold_cols=0)
    with pytest.raises(SystemExit, match="never really scored"):
        check_holdout_coverage(matrix, ids, train, hold)


def test_a_holdout_carrying_its_share_passes_the_guard():
    """The guard is a floor against nothing, not a test of the out-of-vocabulary effect."""
    matrix, ids, train, hold = _coverage_fixture(100, 40, hold_cols=4)  # 4/6 of the train rate
    stats = check_holdout_coverage(matrix, ids, train, hold)
    assert stats["checked"] and stats["ratio"] == pytest.approx(4 / 6)
    assert stats["n_holdout_all_zero"] == 0


def test_too_few_holdout_genomes_records_an_unchecked_gate_rather_than_a_pass():
    """A mean over a handful of genomes swings on one carrier, so it must not be read as evidence."""
    matrix, ids, train, hold = _coverage_fixture(10, 5, hold_cols=0)
    stats = check_holdout_coverage(matrix, ids, train, hold)  # would fail on ratio alone
    assert stats["checked"] is False
    assert stats["ratio"] == 0.0 and stats["n_holdout_all_zero"] == 5


# --------------------------------------------------------------------------------------------------
# score -> merge, end to end through the CLI
# --------------------------------------------------------------------------------------------------
def test_scan_and_merge_builds_a_design_with_ggcat_trainval_and_scanned_holdout(cohort, split_table, tmp_path):
    """The deliverable: one matrix, train+validate rows from GGCAT, holdout rows from sequence."""
    shard_dir = tmp_path / "scan"
    for i in range(2):  # two shards, to exercise the reassembly
        _main_cli([
            "score", "--id-map", str(cohort["design"] / "id_map.tsv"),
            "--split-table", str(split_table), "--reflist", str(cohort["reflist"]),
            "--shard-index", str(i), "--n-shards", "2", "--progress-every", "0",
            "--out", str(shard_dir / f"scan_{i:02d}.npz"),
        ])
    out = tmp_path / "design_merged"
    _main_cli([
        "merge", "--design-dir", str(cohort["design"]), "--shard-dir", str(shard_dir),
        "--split-table", str(split_table), "--out-dir", str(out),
    ])

    manifest = json.loads((out / "merge_manifest.json").read_text())
    assert manifest["rows_from_ggcat"] == len(TRAIN) + len(VALIDATE)
    assert manifest["rows_from_scanner"] == len(HOLDOUT)
    assert manifest["verification"]["n_mismatch_cells"] == 0
    assert manifest["verification"]["n_shared"] == len(TRAIN) + len(VALIDATE)

    samples = (out / "samples.txt").read_text().split()
    assert samples == [*TRAIN, *VALIDATE, *HOLDOUT]
    matrix = sparse.load_npz(out / "presence.npz").toarray()
    expected = np.zeros_like(matrix)
    for i, s in enumerate(samples):
        expected[i, cohort["carried"][s]] = 1
    np.testing.assert_array_equal(matrix, expected)
    # the column order carried over from the GGCAT design, unchanged
    pd.testing.assert_frame_equal(
        pd.read_csv(out / "id_map.tsv", sep="\t"), cohort["id_map"], check_dtype=False
    )


def test_merge_refuses_a_scan_that_disagrees_with_the_training_rows(cohort, split_table, tmp_path):
    """A scanner disagreeing with GGCAT means the holdout is scored by a different rule — stop."""
    shard_dir = tmp_path / "scan"
    _main_cli([
        "score", "--id-map", str(cohort["design"] / "id_map.tsv"),
        "--split-table", str(split_table), "--reflist", str(cohort["reflist"]),
        "--progress-every", "0", "--out", str(shard_dir / "scan_00.npz"),
    ])
    corrupt = sparse.load_npz(shard_dir / "scan_00.npz").tolil()
    corrupt[0, 3] = 1 - corrupt[0, 3]
    sparse.save_npz(shard_dir / "scan_00.npz", corrupt.tocsr())
    with pytest.raises(SystemExit, match="disagrees with GGCAT"):
        _main_cli([
            "merge", "--design-dir", str(cohort["design"]), "--shard-dir", str(shard_dir),
            "--split-table", str(split_table), "--out-dir", str(tmp_path / "merged"),
        ])


def test_one_scan_serves_a_dedup_design_because_columns_align_by_sequence(cohort, split_table, tmp_path):
    """The LD control keeps a subset of the features, so re-scanning for it would be pure waste.

    Matching by position would silently mis-assign every column here — the dedup id_map holds
    features 0, 2 and 4 under indices 0, 1 and 2 — and the resulting matrix would still be
    well-formed. Matching by sequence is what makes the reuse safe.
    """
    shard_dir = tmp_path / "scan"
    _main_cli([
        "score", "--id-map", str(cohort["design"] / "id_map.tsv"),
        "--split-table", str(split_table), "--reflist", str(cohort["reflist"]),
        "--progress-every", "0", "--out", str(shard_dir / "scan_00.npz"),
    ])

    keep = [0, 2, 4]
    dedup = tmp_path / "design_dedup"
    dedup.mkdir()
    sub = cohort["id_map"].iloc[keep].reset_index(drop=True)
    sub["unitig_idx"] = np.arange(len(keep))
    sub.to_csv(dedup / "id_map.tsv", sep="\t", index=False)
    trainval = [*TRAIN, *VALIDATE]
    (dedup / "samples.txt").write_text("".join(f"{s}\n" for s in trainval))
    full = sparse.load_npz(cohort["design"] / "presence.npz").tocsr()
    sparse.save_npz(dedup / "presence.npz", full[:, keep])

    out = tmp_path / "dedup_merged"
    _main_cli([
        "merge", "--design-dir", str(dedup), "--shard-dir", str(shard_dir),
        "--split-table", str(split_table), "--out-dir", str(out),
    ])
    manifest = json.loads((out / "merge_manifest.json").read_text())
    assert manifest["n_features"] == len(keep)
    assert manifest["n_scan_features"] == len(cohort["feats"])
    assert manifest["verification"]["n_mismatch_cells"] == 0

    matrix = sparse.load_npz(out / "presence.npz").toarray()
    samples = (out / "samples.txt").read_text().split()
    expected = np.zeros_like(matrix)
    for i, s in enumerate(samples):
        for col, feat in enumerate(keep):
            expected[i, col] = feat in cohort["carried"][s]
    np.testing.assert_array_equal(matrix, expected)


def test_merge_refuses_a_scan_that_never_covered_the_design_features(cohort, split_table, tmp_path):
    """A scan from another drug would align on nothing — that must stop, not produce zeros."""
    shard_dir = tmp_path / "scan"
    other = tmp_path / "other_id_map.tsv"
    pd.DataFrame({"unitig_idx": [0], "variant": ["A" * 60]}).to_csv(other, sep="\t", index=False)
    _main_cli([
        "score", "--id-map", str(other), "--split-table", str(split_table),
        "--reflist", str(cohort["reflist"]), "--progress-every", "0",
        "--out", str(shard_dir / "scan_00.npz"),
    ])
    with pytest.raises(SystemExit, match="never scanned"):
        _main_cli([
            "merge", "--design-dir", str(cohort["design"]), "--shard-dir", str(shard_dir),
            "--split-table", str(split_table), "--out-dir", str(tmp_path / "merged"),
        ])


def _shard(dirpath, index, n_shards, name="scan"):
    """One scan shard: the .npz plus the .scan.json sidecar the completeness check reads."""
    import json as _json
    dirpath.mkdir(parents=True, exist_ok=True)
    npz = dirpath / f"{name}_{index:02d}.npz"
    npz.write_bytes(b"")
    npz.with_suffix(".scan.json").write_text(
        _json.dumps({"shard_index": index, "n_shards": n_shards, "id_map": "x"})
    )
    return npz


def test_assert_shards_complete_accepts_a_full_set(tmp_path):
    from bac_pyseer.ast_gwas.unitig_kmer_presence import assert_shards_complete
    shards = [_shard(tmp_path, i, 8) for i in range(8)]
    info = assert_shards_complete(sorted(shards))
    assert info["n_shards"] == 8
    assert len(info["shard_files"]) == 8


def test_a_single_surviving_shard_is_refused(tmp_path):
    """The 2026-08-28 failure: all 8 array tasks wrote one file, so 7/8 of the scan vanished.

    Because the scan strides the sample list, the survivor is an even sample of the holdout rather
    than a prefix — every downstream gate passes and the read-out reports a clean AUROC on an
    eighth of the genomes. Only a completeness check catches it.
    """
    from bac_pyseer.ast_gwas.unitig_kmer_presence import assert_shards_complete
    with pytest.raises(SystemExit) as e:
        assert_shards_complete([_shard(tmp_path, 0, 8)])
    msg = str(e.value)
    assert "1/8 shards present" in msg
    assert "[1, 2, 3, 4, 5, 6, 7]" in msg


def test_a_shard_without_its_sidecar_is_refused(tmp_path):
    from bac_pyseer.ast_gwas.unitig_kmer_presence import assert_shards_complete
    good = _shard(tmp_path, 0, 2)
    orphan = tmp_path / "scan_01.npz"
    orphan.write_bytes(b"")
    with pytest.raises(SystemExit) as e:
        assert_shards_complete([good, orphan])
    assert "no .scan.json sidecar" in str(e.value)


def test_shards_from_two_different_runs_are_refused(tmp_path):
    from bac_pyseer.ast_gwas.unitig_kmer_presence import assert_shards_complete
    a = _shard(tmp_path, 0, 8)
    b = _shard(tmp_path, 1, 16)
    with pytest.raises(SystemExit) as e:
        assert_shards_complete([a, b])
    assert "disagree on n_shards" in str(e.value)


def test_no_shards_at_all_is_refused(tmp_path):
    from bac_pyseer.ast_gwas.unitig_kmer_presence import assert_shards_complete
    with pytest.raises(SystemExit) as e:
        assert_shards_complete([])
    assert "no scan shards" in str(e.value)
