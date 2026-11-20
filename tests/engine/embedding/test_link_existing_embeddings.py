"""link_existing_embeddings: another store's ESM-C .pt linked in only where it is the embedding we would write."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import bacpredict.engine.embedding.link_existing_embeddings as lee

CONTIGS = (("MA", "MB", "MC"), ("MD",), ("ME", "MF"))  # 6 proteins over 3 contigs → ids 0 0 0 1 2 2


class Stores:
    """Our parquets, a source store of .pt (the Kp store's layout), and our target ESM store."""

    def __init__(self, root: Path):
        self.prot, self.src, self.esm = root / "protein_sequences", root / "kp_esm", root / "esm"
        for d in (self.prot, self.src, self.esm):
            d.mkdir()

    def parquet(self, s: str, contigs=CONTIGS) -> Stores:
        pd.DataFrame([{"sample_id": s, "contig_idx": list(range(len(contigs))),
                       "protein_sequence": [list(c) for c in contigs]}]).to_parquet(
            self.prot / f"{s}_protein_sequences.parquet")
        return self

    def source(self, s: str, contigs=CONTIGS, *, ids=None, dtype=torch.float32, width=960, mask=None,
               max_contigs: int = 1000) -> Stores:
        if ids is None:
            ids = [min(i, max_contigs - 1) for i, c in enumerate(contigs) for _ in c]
        n = len(ids)
        torch.save({"protein_embeddings": torch.randn(1, n, width).to(dtype),
                    "contig_ids": torch.tensor([ids], dtype=torch.float32),  # float32, as the Kp store holds them
                    "attention_mask": torch.ones(1, n) if mask is None else mask},
                   self.src / f"{s}_esm_embeddings.pt")
        return self

    def target(self, s: str) -> Path:
        return self.esm / f"{s}_esm_embeddings.pt"

    def run(self, samples, **kw) -> pd.DataFrame:
        return lee.link_existing_embeddings(samples, self.prot, self.src, self.esm, **kw)


def status(report: pd.DataFrame) -> dict[str, str]:
    return dict(zip(report["Sample"], report["status"], strict=True))


@pytest.fixture
def st(tmp_path) -> Stores:
    return Stores(tmp_path)


def test_an_aligned_source_is_linked_by_absolute_path(st):
    st.parquet("SAMN1").source("SAMN1")
    r = st.run(["SAMN1"])
    src = str(st.src.resolve() / "SAMN1_esm_embeddings.pt")
    assert r.to_dict("records") == [{"Sample": "SAMN1", "status": lee.LINKED, "n_ours": 6, "n_src": 6,
                                     "src_dtype": "float32", "detail": src}]
    link = st.target("SAMN1")
    assert link.is_symlink() and os.path.isabs(os.readlink(link))
    assert torch.equal(torch.load(link, weights_only=True)["protein_embeddings"],
                       torch.load(st.src / "SAMN1_esm_embeddings.pt", weights_only=True)["protein_embeddings"])


def test_a_relative_source_dir_still_gives_an_absolute_link(st, monkeypatch):
    st.parquet("SAMN1").source("SAMN1")
    monkeypatch.chdir(st.src.parent)
    lee.link_existing_embeddings(["SAMN1"], st.prot, Path(st.src.name), st.esm)
    assert os.readlink(st.target("SAMN1")) == str(st.src.resolve() / "SAMN1_esm_embeddings.pt")


@pytest.mark.parametrize("make, want", [
    # the Kp store's one measured misalignment: three selenoproteins missing
    (lambda st: st.parquet("S").source("S", contigs=(("MA", "MB"), ("MD",), ("ME",))), lee.COUNT_MISMATCH),
    (lambda st: st.parquet("S").source("S", ids=[0, 0, 1, 1, 2, 2]), lee.CONTIG_MISMATCH),  # same count, other split
    (lambda st: st.parquet("S").source("S", ids=[1, 1, 1, 2, 3, 3]), lee.CONTIG_MISMATCH),  # same runs, other ids
    (lambda st: st.parquet("S"), lee.ABSENT),
    (lambda st: (st.parquet("S"), (st.src / "S_esm_embeddings.pt").write_bytes(b"not a torch file")),
     lee.UNREADABLE),
    (lambda st: (st.parquet("S"), torch.save({"protein_embeddings": torch.zeros(1, 6, 960)},
                                             st.src / "S_esm_embeddings.pt")), lee.UNREADABLE),
    (lambda st: st.parquet("S").source("S", width=480), lee.FORMAT_MISMATCH),
    (lambda st: st.parquet("S").source("S", dtype=torch.float16), lee.FORMAT_MISMATCH),
    (lambda st: st.parquet("S").source("S", mask=torch.tensor([[1.0, 1, 1, 1, 1, 0]])), lee.FORMAT_MISMATCH),
    (lambda st: st.source("S"), lee.NO_PARQUET),
], ids=["count", "contig-split", "contig-ids", "absent", "unreadable", "missing-keys", "width", "dtype", "mask",
        "no-parquet"])
def test_each_outcome_is_named_and_links_nothing(st, make, want):
    make(st)
    r = st.run(["S"])
    assert status(r) == {"S": want}
    assert not os.path.lexists(st.target("S"))


def test_a_bf16_source_links_too(st):
    st.parquet("SAMN1").source("SAMN1", dtype=torch.bfloat16)
    assert st.run(["SAMN1"])["src_dtype"].tolist() == ["bfloat16"]
    assert st.target("SAMN1").is_symlink()


@pytest.mark.parametrize("existing", ["file", "link", "broken-link"])
def test_an_existing_target_is_never_touched(st, existing):
    st.parquet("SAMN1").source("SAMN1")
    t = st.target("SAMN1")
    if existing == "file":
        t.write_bytes(b"ours")
    elif existing == "link":
        os.symlink(st.src / "SAMN1_esm_embeddings.pt", t)
    else:
        os.symlink(st.src / "gone.pt", t)
    before = (t.is_symlink(), os.readlink(t) if t.is_symlink() else t.read_bytes())
    r = st.run(["SAMN1"])
    assert status(r) == {"SAMN1": lee.EXISTS}
    assert r["detail"].iloc[0] == ("file" if existing == "file" else f"link -> {before[1]}")
    assert (t.is_symlink(), os.readlink(t) if t.is_symlink() else t.read_bytes()) == before


def test_a_name_that_appears_mid_run_is_not_replaced(st, monkeypatch):
    st.parquet("SAMN1").source("SAMN1")
    real = lee._read_source

    def racing(pt):  # the array writes the .pt between the existence check and the link
        st.target("SAMN1").write_bytes(b"embedded meanwhile")
        return real(pt)

    monkeypatch.setattr(lee, "_read_source", racing)
    r = st.run(["SAMN1"])
    assert status(r) == {"SAMN1": lee.EXISTS} and r["detail"].iloc[0] == "appeared during this run"
    assert st.target("SAMN1").read_bytes() == b"embedded meanwhile"


def test_dry_run_decides_but_links_nothing(st):
    st.parquet("SAMN1").source("SAMN1")
    assert status(st.run(["SAMN1"], dry_run=True)) == {"SAMN1": lee.WOULD_LINK}
    assert not os.path.lexists(st.target("SAMN1"))


def test_the_protein_cap_is_what_our_embedder_would_write(st):
    st.parquet("SAMN1").source("SAMN1", ids=[0, 0, 0, 1])  # a source cut at 4 proteins
    assert status(st.run(["SAMN1"], max_proteins=9000, dry_run=True)) == {"SAMN1": lee.COUNT_MISMATCH}
    r = st.run(["SAMN1"], max_proteins=4)
    assert status(r) == {"SAMN1": lee.LINKED} and (r["n_ours"].iloc[0], r["n_src"].iloc[0]) == (6, 4)


def test_contig_ids_are_clamped_as_our_embedder_clamps_them(st):
    st.parquet("SAMN1").source("SAMN1", max_contigs=2)  # ids 0 0 0 1 1 1
    assert status(st.run(["SAMN1"], dry_run=True)) == {"SAMN1": lee.CONTIG_MISMATCH}
    assert status(st.run(["SAMN1"], max_contigs=2)) == {"SAMN1": lee.LINKED}


def test_expected_contig_ids():
    counts = np.array([3, 1, 2])
    assert lee.expected_contig_ids(counts, 9000, 1000).tolist() == [0, 0, 0, 1, 2, 2]
    assert lee.expected_contig_ids(counts, 5, 1000).tolist() == [0, 0, 0, 1, 2]
    assert lee.expected_contig_ids(counts, 9000, 2).tolist() == [0, 0, 0, 1, 1, 1]


def test_contig_protein_counts_reads_list_lengths(st):
    st.parquet("SAMN1")
    assert lee.contig_protein_counts(st.prot / "SAMN1_protein_sequences.parquet").tolist() == [3, 1, 2]


def test_the_source_is_opened_memory_mapped(st, monkeypatch):
    st.parquet("SAMN1").source("SAMN1")
    seen = {}
    real = torch.load

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(torch, "load", spy)
    st.run(["SAMN1"], dry_run=True)
    assert seen.get("mmap") is True and seen.get("weights_only") is True


def test_the_process_pool_gives_the_same_answer(st):
    samples = []
    for i in range(6):
        s = f"SAMN{i}"
        samples.append(s)
        st.parquet(s)
        if i % 3:
            st.source(s, ids=None if i % 2 else [0, 0, 1, 1, 2, 2])
    serial = st.run(samples, workers=1, batch_size=1, dry_run=True)
    pooled = st.run(samples, workers=2, batch_size=1, dry_run=True)
    pd.testing.assert_frame_equal(serial, pooled)
    assert status(serial) == {"SAMN0": lee.ABSENT, "SAMN1": lee.WOULD_LINK, "SAMN2": lee.CONTIG_MISMATCH,
                              "SAMN3": lee.ABSENT, "SAMN4": lee.CONTIG_MISMATCH, "SAMN5": lee.WOULD_LINK}


def test_cli_report_and_exit_status(st, tmp_path, capsys):
    st.parquet("SAMN1").source("SAMN1")
    st.parquet("SAMN2")
    st.source("SAMN3")  # no parquet: the stage ran before the proteins
    lst = tmp_path / "kp.txt"
    lst.write_text("SAMN1\nSAMN2\n")
    report = tmp_path / "linked.tsv"
    base = ["--protein-dir", str(st.prot), "--src-dir", str(st.src), "--esm-dir", str(st.esm), "--workers", "1",
            "--report", str(report)]

    assert lee.main(["--samples-file", str(lst), *base]) == 0
    out = capsys.readouterr().out
    assert "linked: 1" in out and "absent: 1" in out and "dtype of the linked: {'float32': 1}" in out
    got = pd.read_csv(report, sep="\t", dtype={"Sample": str})
    assert list(got.columns) == lee.REPORT_COLUMNS
    assert dict(zip(got["Sample"], got["status"], strict=True)) == {"SAMN1": lee.LINKED, "SAMN2": lee.ABSENT}

    # a rerun leaves the link alone; a genome without a parquet fails the run
    lst.write_text("SAMN1\nSAMN3\n")
    assert lee.main(["--samples-file", str(lst), *base]) == 1
    out = capsys.readouterr().out
    assert "exists: 1" in out and out.rstrip().endswith("FAIL: 1 no_parquet")
    assert not list(tmp_path.glob(".linked.tsv.*.part"))


def test_cli_refuses_one_store_as_both_ends(st, tmp_path):
    lst = tmp_path / "kp.txt"
    lst.write_text("SAMN1\n")
    with pytest.raises(SystemExit):
        lee.main(["--samples-file", str(lst), "--protein-dir", str(st.prot), "--src-dir", str(st.esm),
                  "--esm-dir", str(st.esm), "--report", str(tmp_path / "r.tsv")])
