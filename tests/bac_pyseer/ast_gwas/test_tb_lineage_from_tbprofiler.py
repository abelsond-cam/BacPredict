"""TB lineage clusters from TB-Profiler calls — the comparator partition and the permutation strata.

Every sample in the reflist must get a cluster, because pyseer needs one for each phenotyped genome;
the question is only whether it gets a *named* one. The failure worth guarding is a bad join, which
looks exactly like "this cohort has no lineages" — TB-Profiler assigns a lineage to essentially
every genome it processes, so a low coverage figure means the ids do not match, not that the biology
is unclear.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bac_pyseer.ast_gwas.sublineage_from_metadata import OTHER
from bac_pyseer.ast_gwas.tb_lineage_from_tbprofiler import load_lineages, load_reflist, run


def _cohort(tmp_path: Path, counts: dict[str, int], *, unlabelled: int = 0):
    """A reflist plus a matching tbprofiler_lineage.csv, from {sub_lineage: n} counts."""
    rows, ids = [], []
    for label, n in counts.items():
        for i in range(n):
            sid = f"SAMEA{label.replace('.', '_')}_{i:04d}"
            ids.append(sid)
            rows.append({"Sample": sid, "main_lineage": label.split(".")[0], "sub_lineage": label})
    for i in range(unlabelled):
        ids.append(f"SAMEA_unlab_{i:04d}")   # in the reflist, absent from TB-Profiler's output
    reflist = tmp_path / "reflist.tsv"
    reflist.write_text("".join(f"{s}\t/data/{s}.fa.gz\n" for s in ids))
    csv = tmp_path / "tbprofiler_lineage.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return reflist, csv, ids


def test_every_reflist_sample_gets_a_cluster(tmp_path: Path) -> None:
    """pyseer needs a cluster per phenotyped genome; an unlabelled one joins `other`, never vanishes."""
    reflist, csv, ids = _cohort(tmp_path, {"lineage2.2.1": 150, "lineage4.3": 120}, unlabelled=30)
    out = tmp_path / "clusters.tsv"
    manifest = run(reflist=reflist, lineage_csv=csv, out_tsv=out, min_size=100)
    written = dict(line.split("\t") for line in out.read_text().splitlines())
    assert list(written) == ids
    assert manifest["n_samples"] == 300
    assert manifest["n_in_other"] == 30
    assert written["SAMEA_unlab_0000"] == OTHER


def test_rare_lineages_collapse_into_other(tmp_path: Path) -> None:
    """Sizes are counted over the cohort being tested — a lineage common worldwide but rare here
    cannot support a within-lineage permutation."""
    reflist, csv, _ = _cohort(tmp_path, {"lineage2.2.1": 150, "lineage1.1.1": 20, "lineage3": 5})
    manifest = run(reflist=reflist, lineage_csv=csv, out_tsv=tmp_path / "c.tsv", min_size=100)
    assert manifest["n_clusters"] == 1
    assert manifest["n_in_other"] == 25
    assert manifest["n_distinct_labels_in_cohort"] == 3, "all three were seen, two were collapsed"


def test_main_lineage_gives_the_coarser_bracket(tmp_path: Path) -> None:
    """The permutation null runs at two resolutions; sub_lineage is fine, main_lineage is coarse."""
    counts = {"lineage2.2.1": 60, "lineage2.1": 60, "lineage4.3": 120}
    reflist, csv, _ = _cohort(tmp_path, counts)
    fine = run(reflist=reflist, lineage_csv=csv, out_tsv=tmp_path / "f.tsv",
               min_size=100, cluster_source="sub_lineage")
    coarse = run(reflist=reflist, lineage_csv=csv, out_tsv=tmp_path / "c.tsv",
                 min_size=100, cluster_source="main_lineage")
    # At sub_lineage the two lineage2 branches are 60 each and both collapse; at main_lineage they
    # merge into one cluster of 120 that survives.
    assert fine["n_clusters"] == 1 and fine["n_in_other"] == 120
    assert coarse["n_clusters"] == 2 and coarse["n_in_other"] == 0


def test_the_manifest_separates_the_two_coverage_numbers(tmp_path: Path) -> None:
    """Label coverage and named-cluster coverage are different, and get conflated."""
    reflist, csv, _ = _cohort(tmp_path, {"lineage2.2.1": 150, "lineage1.1.1": 50}, unlabelled=50)
    out = tmp_path / "c.tsv"
    run(reflist=reflist, lineage_csv=csv, out_tsv=out, min_size=100)
    m = json.loads(out.with_suffix(".manifest.json").read_text())
    assert m["label_coverage"] == pytest.approx(200 / 250)      # has any call
    assert m["named_cluster_coverage"] == pytest.approx(150 / 250)  # survived min_size
    assert m["cluster_type"] == "sub_lineage"
    assert m["source"] == "tbprofiler"


def test_a_bad_join_is_a_hard_error_not_a_lineage_free_cohort(tmp_path: Path) -> None:
    """The failure mode this guard exists for: right file, wrong id column, plausible-looking output."""
    reflist, csv, _ = _cohort(tmp_path, {"lineage2.2.1": 150})
    reflist.write_text("".join(f"OTHER_ID_{i}\t/data/x.fa.gz\n" for i in range(150)))
    with pytest.raises(SystemExit) as e:
        run(reflist=reflist, lineage_csv=csv, out_tsv=tmp_path / "c.tsv", min_coverage=0.5)
    assert "join problem" in str(e.value)


def test_uncalled_lineages_are_dropped_rather_than_clustered(tmp_path: Path) -> None:
    """TB-Profiler emits an empty lineage for a genome with no informative barcode SNP. That is a
    real outcome, and it must not become a cluster of its own."""
    csv = tmp_path / "l.csv"
    pd.DataFrame({
        "Sample": ["A", "B", "C", "D"],
        "main_lineage": ["lineage2", "", "lineage2", None],
        "sub_lineage": ["lineage2.2.1", "NA", "lineage2.2.1", None],
    }).to_csv(csv, index=False)
    assert load_lineages(csv) == {"A": "lineage2.2.1", "C": "lineage2.2.1"}


def test_a_missing_column_names_the_producer(tmp_path: Path) -> None:
    csv = tmp_path / "wrong.csv"
    pd.DataFrame({"Sample": ["A"], "lineage": ["lineage2"]}).to_csv(csv, index=False)
    with pytest.raises(SystemExit) as e:
        load_lineages(csv)
    assert "parse_tbprofiler_calls" in str(e.value)


def test_an_unknown_cluster_source_is_refused(tmp_path: Path) -> None:
    csv = tmp_path / "l.csv"
    pd.DataFrame({"Sample": ["A"], "main_lineage": ["l2"], "sub_lineage": ["l2.2"]}).to_csv(csv, index=False)
    with pytest.raises(SystemExit):
        load_lineages(csv, cluster_source="Sublineage")


def test_reflist_accepts_both_shapes(tmp_path: Path) -> None:
    """Sample<TAB>path is what resolve_ast_assemblies emits; a bare id list is what a human writes."""
    tabbed = tmp_path / "a.tsv"
    tabbed.write_text("S1\t/data/S1.fa.gz\nS2\t/data/S2.fa.gz\n")
    bare = tmp_path / "b.txt"
    bare.write_text("S1\nS2\n")
    assert load_reflist(tabbed) == load_reflist(bare) == ["S1", "S2"]


def test_an_empty_reflist_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "e.tsv"
    empty.write_text("\n\n")
    with pytest.raises(SystemExit):
        load_reflist(empty)
