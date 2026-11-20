"""check_cohort_store: a cohort's protein and ESM-C stores, checked against its annotation table."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import pyarrow.parquet as papq
import pytest
import torch

import bacpredict.engine.embedding.check_cohort_store as ccs

T0 = 1_700_000_000  # GFF3 written, then the parquet (+100 s), then the .pt (+200 s)


class Store:
    """A tiny cohort on disk: contigs, GFF3, parquet and .pt per genome, plus the annotation table."""

    def __init__(self, root: Path):
        self.root = root
        for d in ("contigs", "bakta", "protein_sequences", "esm"):
            (root / d).mkdir()
        self.prot, self.esm = root / "protein_sequences", root / "esm"
        self.rows: list[dict] = []

    def add(self, s: str, contigs=(("MA", "MB"), ("MC",)), n_pseudo=1, pt=True, dtype=torch.bfloat16):
        fna = self.root / "contigs" / f"{s}.fna.gz"
        fna.write_bytes(f">{s}.contig00001\nACGT\n".encode())
        gff = self.root / "bakta" / f"{s}.bakta.gff3.gz"
        gff.write_bytes(b"##gff-version 3\n")
        n = sum(len(c) for c in contigs)
        pq = self.prot / f"{s}_protein_sequences.parquet"
        # the extractor's layout: one row, per-contig lists
        pd.DataFrame([{"sample_id": s, "contig_idx": list(range(len(contigs))),
                       "protein_sequence": [list(c) for c in contigs],
                       "gene_name": [[f"g{i}" for i in range(len(c))] for c in contigs]}]).to_parquet(pq)
        self.rows.append({"Sample": s, "sr_assembly_file": str(fna), "sr_gff_file": str(gff), "origin": "bakrep",
                          "n_cds": n + n_pseudo, "n_pseudo": n_pseudo,
                          "contigs_md5": hashlib.md5(fna.read_bytes()).hexdigest()})
        os.utime(gff, (T0, T0))
        os.utime(pq, (T0 + 100, T0 + 100))
        if pt:
            self.write_pt(s, min(n, 9000), dtype=dtype)
        return self

    def write_pt(self, s: str, n: int, dtype=torch.bfloat16, width=960, nan=False):
        pe = torch.zeros(1, n, width, dtype=dtype)
        if nan:
            pe[0, 0, 0] = float("nan")
        path = self.esm / f"{s}_esm_embeddings.pt"
        torch.save({"protein_embeddings": pe, "contig_ids": torch.zeros(1, n), "attention_mask": torch.ones(1, n)},
                   path)
        os.utime(path, (T0 + 200, T0 + 200))

    @property
    def cohort(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def check(self, samples=None, esm=True, **kw) -> ccs.StoreCheck:
        samples = samples if samples is not None else sorted(r["Sample"] for r in self.rows)
        return ccs.check_cohort_store(self.cohort, self.prot, samples, esm_dir=self.esm if esm else None, **kw)


def _rewrite(path: Path, content: bytes, mtime: int) -> None:
    path.write_bytes(content)
    os.utime(path, (mtime, mtime))


@pytest.fixture
def store(tmp_path) -> Store:
    st = Store(tmp_path)
    for s in ("SAMN1", "SAMN2", "SAMN3", "SAMN4"):
        st.add(s)
    return st


def test_a_complete_store_passes(store):
    r = store.check()
    assert r.ok, r.problems
    assert (r.with_parquet, r.with_embedding, r.count_checked, r.md5_checked) == (4, 4, 4, 4)
    assert r.n_proteins == dict.fromkeys(("SAMN1", "SAMN2", "SAMN3", "SAMN4"), 3)
    assert len(r.sampled) == 4 and all(line.endswith(" ok") for line in r.sampled)


def test_the_process_pool_gives_the_same_answer(store):
    (store.prot / "SAMN3_protein_sequences.parquet").unlink()
    serial = store.check(workers=1, batch_size=1)
    pooled = store.check(workers=2, batch_size=1)
    assert pooled.problems == serial.problems == {ccs.NO_PARQUET: ["SAMN3"]}
    assert pooled.n_proteins == serial.n_proteins


def test_protein_count_equals_the_to_pylist_count(store):
    pq = store.prot / "SAMN1_protein_sequences.parquet"
    slow = sum(len(x) for x in papq.read_table(pq, columns=["protein_sequence"]).column(0).to_pylist()[0])
    assert ccs.protein_count(pq) == slow == 3


def test_proteins_only_skips_the_esm_store(store):
    for f in store.esm.iterdir():
        f.unlink()
    r = store.check(esm=False)
    assert r.ok and r.with_embedding == 0 and r.sampled == []


@pytest.mark.parametrize("break_it, problem", [
    (lambda st: (st.prot / "SAMN2_protein_sequences.parquet").unlink(), ccs.NO_PARQUET),
    (lambda st: _rewrite(st.prot / "SAMN2_protein_sequences.parquet", b"not a parquet", T0 + 100),
     ccs.UNREADABLE_PARQUET),
    (lambda st: os.utime(st.root / "bakta" / "SAMN2.bakta.gff3.gz", (T0 + 150, T0 + 150)), ccs.PARQUET_OLDER),
    (lambda st: (st.root / "bakta" / "SAMN2.bakta.gff3.gz").unlink(), ccs.GFF_MISSING),
    (lambda st: st.rows[1].update(n_pseudo=0), ccs.COUNT_MISMATCH),  # the GFF3 says one more protein
    (lambda st: (st.root / "contigs" / "SAMN2.fna.gz").write_bytes(b">rebuilt\nTTTT\n"), ccs.CONTIGS_CHANGED),
    (lambda st: (st.root / "contigs" / "SAMN2.fna.gz").unlink(), ccs.CONTIGS_UNREADABLE),
    (lambda st: (st.esm / "SAMN2_esm_embeddings.pt").unlink(), ccs.NO_EMBEDDING),
    (lambda st: (st.esm / "SAMN2_esm_embeddings.pt").write_bytes(b"truncated"), ccs.UNREADABLE_EMBEDDING),
    (lambda st: os.utime(st.esm / "SAMN2_esm_embeddings.pt", (T0 + 50, T0 + 50)), ccs.EMBEDDING_OLDER),
    (lambda st: st.write_pt("SAMN2", 3, dtype=torch.float32), ccs.SAMPLED_MISMATCH),
    (lambda st: st.write_pt("SAMN2", 3, width=480), ccs.SAMPLED_MISMATCH),
    (lambda st: st.write_pt("SAMN2", 2), ccs.SAMPLED_MISMATCH),
    (lambda st: st.write_pt("SAMN2", 3, nan=True), ccs.SAMPLED_MISMATCH),
], ids=["no-parquet", "bad-parquet", "parquet-stale", "no-gff", "count", "contigs-changed", "no-contigs", "no-pt",
        "bad-pt", "pt-stale", "pt-dtype", "pt-width", "pt-proteins", "pt-nan"])
def test_each_problem_is_found_and_named(store, break_it, problem):
    break_it(store)
    r = store.check(workers=2, batch_size=1)
    assert not r.ok
    assert r.problems == {problem: ["SAMN2"]}


def test_the_embedding_cap_is_min_of_proteins_and_max_proteins(tmp_path):
    st = Store(tmp_path).add("SAMN9", contigs=(("M",) * 5,), pt=False)
    st.write_pt("SAMN9", 4)
    assert st.check(max_proteins=4).ok
    assert st.check(max_proteins=9000).problems == {ccs.SAMPLED_MISMATCH: ["SAMN9"]}


def test_cohort_level_problems(store):
    store.rows.append(dict(store.rows[0]))  # SAMN1 twice
    r = store.check(samples=["SAMN1", "SAMN2", "SAMX"], qc_samples={"SAMN1", "SAMN2", "SAMN3", "SAMN4", "SAMN5"},
                    qc_name="embedding_input.csv")
    assert r.problems[ccs.DUPLICATE] == ["SAMN1"]
    assert r.problems[ccs.NOT_IN_COHORT] == ["SAMX"]
    assert r.problems["cohort differs from embedding_input.csv"] == ["SAMN5"]
    assert r.problems[ccs.NO_PARQUET] == ["SAMX"]


def test_an_empty_sample_set_fails(store):
    assert not store.check(samples=[]).ok


@pytest.mark.parametrize("col", ccs.REQUIRED_COLUMNS)
def test_a_missing_column_is_a_named_failure_not_a_skipped_check(store, col):
    samples = sorted(r["Sample"] for r in store.rows)
    r = ccs.check_cohort_store(store.cohort.drop(columns=col), store.prot, samples, esm_dir=store.esm)
    assert not r.ok
    assert r.problems[ccs.missing_column(col)] == samples
    extra = {ccs.CONTIGS_UNREADABLE: samples} if col == "sr_assembly_file" else {}  # nothing to hash
    assert r.problems == {ccs.missing_column(col): samples, **extra}


def test_presence_only_needs_no_annotation_columns(store):
    samples = sorted(r["Sample"] for r in store.rows)
    cohort = store.cohort[["Sample"]]
    assert ccs.check_cohort_store(cohort, store.prot, samples, esm_dir=store.esm, presence_only=True).ok


def test_presence_only_reads_the_listings(store):
    (store.prot / "SAMN2_protein_sequences.parquet").unlink()
    (store.esm / "SAMN3_esm_embeddings.pt").unlink()
    (store.root / "contigs" / "SAMN4.fna.gz").unlink()  # not looked at
    r = store.check(presence_only=True)
    assert r.problems == {ccs.NO_PARQUET: ["SAMN2"], ccs.NO_EMBEDDING: ["SAMN3"]}
    assert store.check(presence_only=True, esm=False).problems == {ccs.NO_PARQUET: ["SAMN2"]}


def test_cli_scope_report_and_exit_status(store, tmp_path, capsys):
    cohort = tmp_path / "bakta_annotation.csv"
    store.cohort.to_csv(cohort, index=False)
    lst = tmp_path / "esm_samples.txt"
    lst.write_text("SAMN1\nSAMN2\nSAMN3\nSAMN4\n")
    (store.esm / "SAMN4_esm_embeddings.pt").unlink()
    base = ["--cohort", str(cohort), "--protein-dir", str(store.prot), "--esm-dir", str(store.esm), "--workers", "2"]

    # the first N of the list: SAMN4's missing .pt is outside it
    assert ccs.main([*base, "--samples-file", str(lst), "--n", "3"]) == 0
    assert "first 3 of esm_samples.txt" in capsys.readouterr().out

    report = tmp_path / "problems.csv"
    assert ccs.main([*base, "--report", str(report)]) == 1
    out = capsys.readouterr().out
    assert "FAIL no embedding: 1 ['SAMN4']" in out and out.rstrip().endswith("FAIL")
    assert pd.read_csv(report).to_dict("records") == [{"Sample": "SAMN4", "problem": ccs.NO_EMBEDDING}]


def test_cli_refuses_an_unsorted_list(store, tmp_path):
    cohort = tmp_path / "bakta_annotation.csv"
    store.cohort.to_csv(cohort, index=False)
    lst = tmp_path / "esm_samples.txt"
    lst.write_text("SAMN2\nSAMN1\n")
    with pytest.raises(SystemExit):
        ccs.main(["--cohort", str(cohort), "--protein-dir", str(store.prot), "--samples-file", str(lst)])


def test_cli_pass_marker_holds_the_md5_of_the_cohort_table_and_only_on_pass(store, tmp_path):
    cohort = tmp_path / "bakta_annotation.csv"
    store.cohort.to_csv(cohort, index=False)
    report = tmp_path / "problems.csv"
    marker = tmp_path / "problems.csv.pass"
    base = ["--cohort", str(cohort), "--protein-dir", str(store.prot), "--workers", "1", "--report", str(report)]

    assert ccs.main(base) == 0
    assert marker.read_text() == hashlib.md5(cohort.read_bytes()).hexdigest() + "\n"

    # a failing run removes the marker an earlier run left beside the same report
    (store.prot / "SAMN2_protein_sequences.parquet").unlink()
    assert ccs.main(base) == 1
    assert not marker.exists()
