"""Unit smoke for the non-coding channel audit: run counting, windowing count, RNA fusion."""

from __future__ import annotations

import pandas as pd

from pangena_predict.audit_noncoding_regions import run_audit

# contig c1 length 6000; CDS 100-200 & 400-500; rRNA 250-350 (inside the middle gap).
# only-CDS-occupying runs (min_len 30): [1-99]=99, [201-399]=199 (has rRNA), [501-6000]=5500 (windowed).
GFF = (
    "##gff-version 3\n"
    "##sequence-region c1 1 6000\n"
    "c1\tProdigal\tCDS\t100\t200\t.\t+\t0\tID=a\n"
    "c1\tBarrnap\trRNA\t250\t350\t.\t+\t0\tID=r;gene=rrs\n"
    "c1\tProdigal\tCDS\t400\t500\t.\t+\t0\tID=b\n"
    "##FASTA\n>c1\nACGT\n"
)


def test_audit_counts_runs_windows_and_rna_fusion(tmp_path):
    gff = tmp_path / "s1.gff3"
    gff.write_text(GFF)
    csv = tmp_path / "in.csv"
    pd.DataFrame([{"Sample": "s1", "sr_gff_file": str(gff)}]).to_csv(csv, index=False)

    agg = run_audit(csv, n=None, workers=1, min_len=30)

    assert agg["n_genomes"] == 1
    assert agg["total_noncoding_runs"] == 3
    assert agg["runs_over_maxlen"] == 1            # the 5500 bp run
    assert agg["max_run_len_seen"] == 5500
    # windows: ceil(99/2048)+ceil(199/2048)+ceil(5500/2048) = 1+1+3 = 5 -> 2 extra from the long run
    assert agg["total_windows"] == 5
    assert agg["extra_windows_from_long_runs"] == 2
    assert agg["runs_containing_rna"] == 1         # the [201-399] run swallows the rRNA
    assert agg["total_rna_bodies"] == 1
    assert agg["rna_type_counts"] == {"rrna": 1}
    assert agg["rna_over_maxlen"] == 0
