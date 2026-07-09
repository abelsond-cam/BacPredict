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
    assert agg["mean_cds_per_genome"] == 2         # featA, featB
    assert agg["total_noncoding_runs"] == 3
    assert agg["runs_over_maxlen"] == 1            # the 5500 bp run
    assert agg["max_run_len_seen"] == 5500
    # the 5500 run has no RNA -> splitting yields one non-RNA piece still > window, zero RNA pieces
    assert agg["nonrna_pieces_over_maxlen"] == 1
    assert agg["rna_pieces_over_maxlen"] == 0
    # windows: ceil(99/2048)+ceil(199/2048)+ceil(5500/2048) = 1+1+3 = 5 -> 2 extra from the long run
    assert agg["total_windows"] == 5
    assert agg["extra_windows_from_long_runs"] == 2
    assert agg["runs_containing_rna"] == 1         # the [201-399] run swallows the rRNA
    assert agg["total_rna_bodies"] == 1
    assert agg["rna_over_maxlen"] == 0
    # rRNA 250-350 (101 bp) sits in run 201-399 (199 bp) -> ~98 bp of flanking IGR -> adjacent to IGR.
    rb = agg["feature_breakdown"]["rrna"]
    assert rb["total"] == 1
    assert rb["adjacent_to_igr"] == 1 and rb["solo_in_run"] == 0
    assert rb["adjacent_to_other_rna"] == 0
    assert agg["feature_type_counts"] == {"rrna": 1}


# contig c2: tRNA tightly CDS-flanked (little IGR) = "solo"; plus a CRISPR feature ("other" type).
GFF_SOLO = (
    "##gff-version 3\n"
    "##sequence-region c2 1 1000\n"
    "c2\tProdigal\tCDS\t1\t100\t.\t+\t0\tID=a\n"
    "c2\ttRNAscan\ttRNA\t105\t200\t.\t+\t0\tID=t;gene=trnA\n"
    "c2\tProdigal\tCDS\t205\t400\t.\t+\t0\tID=b\n"
    "c2\tPILER-CR\tCRISPR\t500\t700\t.\t+\t0\tID=c\n"
    "##FASTA\n>c2\nACGT\n"
)


def test_audit_solo_rna_and_other_feature_types(tmp_path):
    gff = tmp_path / "s2.gff3"
    gff.write_text(GFF_SOLO)
    csv = tmp_path / "in.csv"
    pd.DataFrame([{"Sample": "s2", "sr_gff_file": str(gff)}]).to_csv(csv, index=False)

    agg = run_audit(csv, n=None, workers=1, min_len=30)

    # tRNA 105-200 (96 bp) in run 100-204 (104 bp) -> ~8 bp IGR (<30) -> solo, not IGR-adjacent.
    rb = agg["feature_breakdown"]["trna"]
    assert rb["total"] == 1 and rb["solo_in_run"] == 1 and rb["adjacent_to_igr"] == 0
    # The CRISPR array is now in the fusion breakdown too (its own run, 400-1000 span) + feature tally.
    assert agg["feature_type_counts"].get("crispr") == 1
    cr = agg["feature_breakdown"]["crispr"]
    assert cr["total"] == 1 and cr["adjacent_to_other_rna"] is None  # non-RNA -> no adj-RNA


# contig c3: a run over the window whose rRNA body (rrl-like, 3101 bp) itself exceeds the window,
# with only tiny IGR flanks -> after splitting, the RNA piece STAYS over-window, non-RNA does not.
GFF_LONG_RRNA = (
    "##gff-version 3\n"
    "##sequence-region c3 1 5000\n"
    "c3\tProdigal\tCDS\t100\t200\t.\t+\t0\tID=a\n"
    "c3\tBarrnap\trRNA\t300\t3400\t.\t+\t0\tID=r;gene=rrl\n"
    "c3\tProdigal\tCDS\t3500\t3600\t.\t+\t0\tID=b\n"
    "##FASTA\n>c3\nACGT\n"
)


def test_split_keeps_long_rrna_over_window(tmp_path):
    gff = tmp_path / "s3.gff3"
    gff.write_text(GFF_LONG_RRNA)
    csv = tmp_path / "in.csv"
    pd.DataFrame([{"Sample": "s3", "sr_gff_file": str(gff)}]).to_csv(csv, index=False)

    agg = run_audit(csv, n=None, workers=1, min_len=30)

    assert agg["runs_over_maxlen"] == 1               # the 3299 bp run
    assert agg["rna_pieces_over_maxlen"] == 1         # the 3101 bp rrl body stays over-window
    assert agg["nonrna_pieces_over_maxlen"] == 0      # tiny IGR flanks (~99 bp)
    assert agg["rna_over_maxlen"] == 1                # rrl body counted as an over-window RNA
