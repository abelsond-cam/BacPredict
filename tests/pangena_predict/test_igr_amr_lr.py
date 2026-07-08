"""Unit smoke for the Stage-2c baclm promoter-IGR probe.

Fabricates tiny GFF + baclm ``.pt`` + AST fixtures where the promoter IGR row for a ``+``-strand and a
``−``-strand gene carries a label signal, then runs the real anchoring + learning-curve end-to-end.
Covers the only new logic: GFF parse, 5′-promoter row selection by strand, the far-flank audit, and
signal recovery through :func:`igr_amr_lr.run_igr_probe`.
"""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import torch

from pangena_predict.igr_amr_lr import (
    IgrTarget,
    _parse_gff,
    build_promoter_frames,
    run_igr_probe,
)

DIM = 960
# Genome layout on one contig "c1" (1-based inclusive):
#   featA CDS   100-200 (+)      <- far flank of geneP's promoter (CDS)
#   [promoter IGR 201-300]       <- geneP promoter (+ strand: low side, abuts geneP start 301)
#   geneP CDS   301-500 (+)
#   geneM CDS   501-700 (−)
#   [promoter IGR 701-900]       <- geneM promoter (− strand: high side, abuts geneM end 700)
#   rRNA        901-1000         <- far flank of geneM's promoter (RNA-abutting)
GFF_LINES = [
    "c1\tProdigal\tCDS\t100\t200\t.\t+\t0\tID=a;gene=featA",
    "c1\tProdigal\tCDS\t301\t500\t.\t+\t0\tID=p;gene=geneP",
    "c1\tProdigal\tCDS\t501\t700\t.\t-\t0\tID=m;gene=geneM",
    "c1\tBarrnap\trRNA\t901\t1000\t.\t+\t0\tID=r;gene=rrs",
]
# intergenic rows (seqid, start, end): row0 = geneP promoter, row1 = geneM promoter
IG_SEQID = ["c1", "c1"]
IG_START = [201, 701]
IG_END = [300, 900]


def _write_gff(path, gzip_it=True):
    text = "##gff-version 3\n" + "\n".join(GFF_LINES) + "\n##FASTA\n>c1\nACGT\n"
    if gzip_it:
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        path.write_text(text)


def _make_fixtures(root, n=40, signal=2.5):
    root.mkdir(parents=True, exist_ok=True)
    (root / "baclm").mkdir(exist_ok=True)
    (root / "gff").mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    rows, gff_map = [], {}
    for i in range(n):
        sid = f"s{i:03d}"
        label = i % 2
        ig = rng.normal(0, 1, (2, DIM)).astype(np.float32)
        ig[0, :20] += signal * label       # geneP promoter carries the signal
        torch.save(
            {"intergenic_embeddings": torch.from_numpy(ig).to(torch.bfloat16),
             "intergenic_seqid": IG_SEQID, "intergenic_start": IG_START, "intergenic_end": IG_END,
             "n_intergenic": 2},
            root / "baclm" / f"{sid}_baclm_embeddings.pt",
        )
        gpath = root / "gff" / f"{sid}.gff3.gz"
        _write_gff(gpath)
        gff_map[sid] = str(gpath)
        rows.append({"Sample": sid, "testdrug": label, "train_val_eval": "train"})
    ast = root / "ast.csv"
    pd.DataFrame(rows).to_csv(ast, index=False)
    return ast, gff_map


def test_parse_gff_reads_genes_and_strand(tmp_path):
    g = tmp_path / "one.gff3.gz"
    _write_gff(g)
    feats, genes = _parse_gff(g)
    assert genes["genep"][0] == ("c1", 301, 500, "+")
    assert genes["genem"][0] == ("c1", 501, 700, "-")
    assert len(feats["c1"]) == 4  # featA, geneP, geneM, rRNA all occupy


def test_promoter_anchoring_by_strand_and_audit(tmp_path):
    ast, gff_map = _make_fixtures(tmp_path, n=8)
    ids = sorted(gff_map)
    frames, audits = build_promoter_frames(ids, gff_map, tmp_path / "baclm",
                                          [("geneP", ()), ("geneM", ())])
    # + strand: geneP promoter is the low-side IGR (row0: 201-300), abuts gene start 301
    assert frames["geneP"].shape == (8, DIM)
    aP = audits["geneP"].iloc[0]
    assert (aP["igr_start"], aP["igr_end"], aP["strand"]) == (201, 300, "+")
    assert aP["boundary_gap"] == 0 and aP["far_flank_type"] == "cds"  # featA upstream
    # − strand: geneM promoter is the high-side IGR (row1: 701-900), abutting gene end 700
    aM = audits["geneM"].iloc[0]
    assert (aM["igr_start"], aM["igr_end"], aM["strand"]) == (701, 900, "-")
    assert aM["boundary_gap"] == 0 and aM["far_flank_type"] == "rna"  # rRNA on the outward side


def test_igr_probe_recovers_promoter_signal(tmp_path):
    ast, gff_map = _make_fixtures(tmp_path, n=40)
    ids = sorted(gff_map)
    frames, audits = build_promoter_frames(ids, gff_map, tmp_path / "baclm", [("geneP", ())])
    res = run_igr_probe(IgrTarget("geneP", "testdrug"), frames["geneP"], audits["geneP"], ast,
                        seeds=(1, 2), step=8, fine_until=1000)
    assert res.get("error") is None
    assert res["rungs"][-1]["igr"]["mean"] > 0.6  # promoter signal recovered
    assert res["audit"]["frac_cds_flanked"] == 1.0
