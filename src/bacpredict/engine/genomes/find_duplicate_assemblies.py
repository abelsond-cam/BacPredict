"""Find genomes that are the same assembly deposited under more than one BioSample.

Two cheap stages, in order, because the second cannot be run over every pair:

1. **Fingerprint collision** — genomes sharing an exact ``(total_length, n_contigs, n_cds)`` triple in
   ``genome_qc.csv``. A table read; no sequence is touched.
2. **Sequence confirmation** — each candidate group's contigs are re-read and hashed *sequence only*, with the
   header dropped and each contig canonicalised against its reverse complement, so the comparison does not depend
   on contig order, orientation, line wrapping or the header text.

⛔ **All three fingerprint columns are load-bearing.** Measured on the KpSC cohort (PROJECT_STATE §3.6 *Caveats*):
the triple gave 3 groups, and all 3 were confirmed identical; dropping ``n_cds`` added SAMN06437650/SAMN06438666,
which share only 5 of their 142 contigs. ``n_cds`` separates them because both are ``portal_gff`` genomes carrying
their submitter's own annotation.

⛔ **A file checksum cannot do this job.** ``contigs_md5`` and ``atb_md5`` hash the gzipped file, and every contig
header embeds the BioSample (``>SAMD00061218.contig00001 … src=atb``), so two deposits of one assembly never share
a file hash — all 9,095 KpSC ``atb_md5`` values are distinct by construction (``build_genomes.py:179-184``).

**What this does and does not establish.** A confirmed group is the same assembled *sequence* under two accessions.
That is not the same claim as one biological isolate: it cannot tell a double deposit from an assembly linked to
the wrong BioSample, and a *pair of isolates* from one patient or one outbreak will not be caught here at all (nor
by ANI — skani scores a genome against itself at 99.99, and a fifth of random KpSC ST258 pairs clear 99.98). The
report is evidence for a decision, not the decision.

⚠ **Shared engine** — PsA, Enterobacterales and CABBAGE v2 all call this, so both experiments screen identically.
Its output is deliberately *not* a ``--groups`` file unless ``--groups-out`` is passed: handing a split builder a
groups list re-draws that arm's whole split (``build_amr_holdout_splits.py`` L396 → L430 → L311-313), so it is a
decision, not a by-product.

Example, one unit of a cohort::

    python -m bacpredict.engine.genomes.find_duplicate_assemblies \\
        --qc $AMR/processed/enterobacterales/genomes/genome_qc.csv \\
        --samples $AMR/processed/kpsc/splits/eligible_genomes.csv \\
        --out $AMR/processed/kpsc/duplicate_assemblies.csv
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path

import pandas as pd

#: The fingerprint. Every column is required; see the module docstring for why ``n_cds`` cannot be dropped.
FINGERPRINT = ("total_length", "n_contigs", "n_cds")

#: Reverse complement over the IUPAC alphabet, so a contig stored in either orientation hashes the same.
_COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVNacgtryswkmbdhvn", "TGCAYRSWMKVHDBNtgcayrswmkvhdbn")


def candidate_groups(qc: pd.DataFrame, samples: list[str] | None = None) -> pd.DataFrame:
    """Genomes sharing an exact :data:`FINGERPRINT` triple with at least one other genome.

    Parameters
    ----------
    qc
        ``genome_qc.csv``, with ``Sample``, ``contigs_path``, ``qc_pass`` and the :data:`FINGERPRINT` columns.
    samples
        Restrict to these ``Sample`` ids — a unit's eligible set, so a cohort can be screened per unit. ``None``
        screens every QC-passing row.

    Returns
    -------
    pandas.DataFrame
        ``Sample, group, contigs_path`` plus the fingerprint columns, sorted by group then Sample. ``group`` is
        ``dup<NN>``, numbered in fingerprint order. Empty with those columns when nothing collides.
    """
    missing = [c for c in (*FINGERPRINT, "Sample", "contigs_path") if c not in qc.columns]
    if missing:
        raise ValueError(f"genome_qc is missing {missing}")
    df = qc
    if "qc_pass" in df.columns:
        df = df[df["qc_pass"].astype(str).str.lower().isin(("true", "1"))]
    if samples is not None:
        want = set(samples)
        if absent := sorted(want - set(df["Sample"])):
            raise ValueError(f"{len(absent)} requested samples are not QC-passing rows of genome_qc, "
                             f"e.g. {absent[:5]}")
        df = df[df["Sample"].isin(want)]
    if not df["Sample"].is_unique:
        raise ValueError("genome_qc holds a Sample more than once")
    keys = list(FINGERPRINT)
    sized = df.groupby(keys, dropna=False)["Sample"].transform("size")
    hits = df[sized > 1].sort_values([*keys, "Sample"])
    groups = hits.groupby(keys, dropna=False, sort=False).ngroup()
    out = hits[["Sample", "contigs_path", *keys]].copy()
    out.insert(1, "group", [f"dup{n + 1:02d}" for n in groups])
    return out.sort_values(["group", "Sample"]).reset_index(drop=True)


def contig_hashes(path: str | Path) -> list[str]:
    """One md5 per contig of a (optionally gzipped) FASTA, over the sequence alone.

    The header is dropped, the sequence upper-cased and stripped of line breaks, and each contig canonicalised
    as ``min(sequence, reverse_complement)`` so orientation does not matter.

    Parameters
    ----------
    path
        The contigs file. ``.gz`` is read with :mod:`gzip`.

    Returns
    -------
    list of str
        md5 hex digests, in file order.
    """
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    out, cur = [], []

    def flush() -> None:
        if cur:
            seq = "".join(cur)
            out.append(hashlib.md5(min(seq, seq.translate(_COMPLEMENT)[::-1]).encode()).hexdigest())
            cur.clear()

    with opener(p, "rt") as fh:  # type: ignore[operator]
        for line in fh:
            if line.startswith(">"):
                flush()
            else:
                cur.append(line.strip().upper())
    flush()
    return out


def confirm_groups(candidates: pd.DataFrame) -> pd.DataFrame:
    """Re-read each candidate group's contigs and say whether its genomes are the same sequence.

    Parameters
    ----------
    candidates
        :func:`candidate_groups`' output.

    Returns
    -------
    pandas.DataFrame
        One row per genome: the candidate columns plus ``n_contig_seqs`` (distinct contig hashes in that genome),
        ``n_shared`` (hashes common to every genome of the group) and ``identical`` (the group's genomes have the
        same multiset of contig sequences). ``identical`` is the column a policy should read.
    """
    rows = []
    for group, g in candidates.groupby("group", sort=True):
        hashes = {s: contig_hashes(p) for s, p in zip(g["Sample"], g["contigs_path"], strict=True)}
        sets = [set(v) for v in hashes.values()]
        shared = set.intersection(*sets) if sets else set()
        counted = [sorted(v) for v in hashes.values()]
        identical = all(c == counted[0] for c in counted)
        for _, r in g.iterrows():
            rows.append({**r.to_dict(), "n_contig_seqs": len(set(hashes[r["Sample"]])),
                         "n_shared": len(shared), "identical": identical})
    cols = [*candidates.columns, "n_contig_seqs", "n_shared", "identical"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _read_samples(path: Path) -> list[str]:
    """``Sample`` ids from a one-per-line list or from any CSV carrying a ``Sample`` column."""
    text = path.read_text().splitlines()
    if not text:
        return []
    if "," in text[0] or text[0].strip() == "Sample":
        df = pd.read_csv(path, dtype=str)
        if "Sample" not in df.columns:
            raise ValueError(f"{path}: no Sample column")
        return df["Sample"].str.strip().tolist()
    return [line.strip() for line in text if line.strip()]


def main() -> None:
    """Screen one cohort or unit and write the report."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qc", type=Path, required=True, help="genome_qc.csv")
    ap.add_argument("--samples", type=Path, default=None,
                    help="restrict to these Samples (one per line, or a CSV with a Sample column)")
    ap.add_argument("--out", type=Path, required=True, help="the report CSV, one row per candidate genome")
    ap.add_argument("--groups-out", type=Path, default=None,
                    help="also write the CONFIRMED groups as a splits --groups file (Sample,group). Writing one "
                         "does not make it policy: passing it to the split builder re-draws that arm's whole split")
    args = ap.parse_args()

    qc = pd.read_csv(args.qc, dtype=str, low_memory=False)
    for c in FINGERPRINT:
        qc[c] = pd.to_numeric(qc[c], errors="coerce")
    samples = _read_samples(args.samples) if args.samples else None
    cand = candidate_groups(qc, samples)
    n_screened = len(samples) if samples is not None else int(
        qc["qc_pass"].astype(str).str.lower().isin(("true", "1")).sum() if "qc_pass" in qc.columns else len(qc))
    print(f"screened {n_screened} genomes; {cand['group'].nunique()} fingerprint groups, {len(cand)} genomes")
    report = confirm_groups(cand)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    if len(report):
        ok = report[report["identical"]]
        print(f"confirmed identical: {ok['group'].nunique()} groups, {len(ok)} genomes; "
              f"refuted: {report['group'].nunique() - ok['group'].nunique()} groups")
        for group, g in ok.groupby("group", sort=True):
            print(f"  {group}: {', '.join(g['Sample'])} "
                  f"({int(g['n_contigs'].iloc[0])} contigs, {int(g['total_length'].iloc[0])} bp)")
    print(f"report → {args.out}")
    if args.groups_out is not None:
        ok = report[report["identical"]] if len(report) else report
        ok[["Sample", "group"]].to_csv(args.groups_out, index=False)
        print(f"groups file → {args.groups_out} ({len(ok)} genomes). Passing it to the split builder re-draws "
              "the arm's whole split: that is a decision, not a by-product of this screen.")


if __name__ == "__main__":
    main()
