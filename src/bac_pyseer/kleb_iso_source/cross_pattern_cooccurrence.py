r"""Descriptive carrier co-occurrence BETWEEN the top unitig pattern groups (not a gate).

A pyseer ``pattern_group`` is a perfect-LD block: every unitig in it has the *identical*
presence/absence pattern, hence the identical carrier-sample set. This tool asks a purely
descriptive question about the *strongest* blood/faeces unitig signals: do the top pattern groups
travel together across genomes (shared clonal background / co-mobilised element), or are they
carried by disjoint sample sets (independent signals)? It quantifies that with the pairwise
**carrier Jaccard between distinct patterns** — never a reliability judgement, just a co-occurrence
description to hand to the biologists.

Two stages (the middle one is a heavy ``zcat | grep`` over the ~77 GB unitig matrix, so it is an
sbatch step wedged between two cheap python calls — see ``scripts/run_cross_pattern_cooccurrence.sh``):

* ``select`` — rank the pattern groups in the annotated unitig-hit table by
  ``af_weight × n_in_pattern × var_explained_pct`` and emit one **representative unitig sequence**
  per top group to ``seqs.txt`` (+ a small meta TSV keying sequence → pattern group). Because a
  group's members share one pattern, a single representative fully characterises its carrier set.
* ``jaccard`` — parse the ``grep``-pulled matrix lines (``<sequence> | Sample:1 …``) into per-pattern
  carrier sets and write the pairwise between-pattern Jaccard TSV.

``af_weight`` uses ``af`` when the table carries it, else ``invasive_af`` (the annotated table's
orientation-flipped frequency) — a rough "commonness" weight; the ranking only decides which
handful of patterns to describe, so its exact form is not load-bearing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_AF_CANDIDATES = ("af", "invasive_af")
_REQUIRED = ("variant", "pattern_group", "n_in_pattern", "var_explained_pct")


def _af_weight(hits: pd.DataFrame) -> pd.Series:
    """Return a per-row commonness weight from the first available af-like column (else 1.0)."""
    for col in _AF_CANDIDATES:
        if col in hits.columns:
            w = pd.to_numeric(hits[col], errors="coerce")
            print(f"ranking af_weight from column {col!r}", file=sys.stderr)
            return w.fillna(0.0)
    print("no af/invasive_af column found — af_weight=1.0 for all", file=sys.stderr)
    return pd.Series(1.0, index=hits.index)


def select(hits_path: Path, top: int, out_seqs: Path, out_meta: Path) -> pd.DataFrame:
    """Pick the top pattern groups and write their representative unitig sequences + meta.

    Parameters
    ----------
    hits_path : Path
        Annotated unitig-hit TSV (``blood_vs_faeces_unitig_hits_annotated.tsv``).
    top : int
        Number of distinct pattern groups to keep.
    out_seqs : Path
        One representative unitig sequence per top pattern group (input to ``grep -F -f``).
    out_meta : Path
        Meta TSV keying ``rep_variant`` → ``pattern_group`` (+ rank inputs) for the ``jaccard`` stage.

    Returns
    -------
    pandas.DataFrame
        The per-pattern meta table that was written.
    """
    hits = pd.read_csv(hits_path, sep="\t")
    missing = [c for c in _REQUIRED if c not in hits.columns]
    if missing:
        raise SystemExit(f"{hits_path} missing columns {missing} (has: {list(hits.columns)})")
    hits = hits.copy()
    hits["af_weight"] = _af_weight(hits)
    hits["var_explained_pct"] = pd.to_numeric(hits["var_explained_pct"], errors="coerce").fillna(0.0)
    hits["n_in_pattern"] = pd.to_numeric(hits["n_in_pattern"], errors="coerce").fillna(0).astype(int)
    hits["rank_score"] = hits["af_weight"] * hits["n_in_pattern"] * hits["var_explained_pct"]

    # one representative row per pattern group (members share the pattern → any row is representative)
    reps = (
        hits.sort_values("rank_score", ascending=False)
        .drop_duplicates("pattern_group", keep="first")
        .head(top)
        .reset_index(drop=True)
    )
    keep = [c for c in ("pattern_group", "variant", "af_weight", "n_in_pattern", "var_explained_pct",
                        "rank_score", "gene", "product") if c in reps.columns]
    meta = reps[keep].rename(columns={"variant": "rep_variant"})

    out_seqs.parent.mkdir(parents=True, exist_ok=True)
    out_seqs.write_text("\n".join(meta["rep_variant"].astype(str)) + "\n")
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out_meta, sep="\t", index=False)
    print(f"wrote {out_seqs} ({len(meta)} representative unitigs) and {out_meta}", file=sys.stderr)
    print(meta.to_string(index=False), file=sys.stderr)
    return meta


def _parse_carrier_lines(matrix_lines: Path) -> dict[str, set[str]]:
    """Parse ``<sequence> | Sample:1 …`` lines into ``{sequence: {carrier samples}}``."""
    carriers: dict[str, set[str]] = {}
    with open(matrix_lines) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or " | " not in line:
                continue
            seq, _, rest = line.partition(" | ")
            samples = {tok.rsplit(":", 1)[0] for tok in rest.split() if tok}
            carriers[seq.strip()] = samples
    return carriers


def jaccard(matrix_lines: Path, meta_path: Path, out_path: Path) -> pd.DataFrame:
    """Compute pairwise between-pattern carrier Jaccard from the grep-pulled matrix lines.

    Parameters
    ----------
    matrix_lines : Path
        Lines pulled from the unitig matrix by ``grep -F -f seqs.txt`` (``<sequence> | Sample:1 …``).
    meta_path : Path
        The ``select`` meta TSV (keys ``rep_variant`` → ``pattern_group``).
    out_path : Path
        Output tidy TSV (``pattern_i, pattern_j, n_i, n_j, n_shared, jaccard`` + gene labels if present).

    Returns
    -------
    pandas.DataFrame
        The pairwise Jaccard table that was written.
    """
    meta = pd.read_csv(meta_path, sep="\t")
    carriers = _parse_carrier_lines(matrix_lines)
    label = {}
    sets: dict[object, set[str]] = {}
    for _, r in meta.iterrows():
        seq = str(r["rep_variant"])
        pg = r["pattern_group"]
        if seq not in carriers:
            print(f"WARNING: representative unitig for pattern {pg} not found in matrix lines "
                  f"(seq prefix {seq[:30]}…) — skipped", file=sys.stderr)
            continue
        sets[pg] = carriers[seq]
        gene = str(r.get("gene", "")) if "gene" in meta.columns else ""
        label[pg] = gene

    pgs = list(sets)
    rows = []
    for i in range(len(pgs)):
        for j in range(i + 1, len(pgs)):
            a, b = sets[pgs[i]], sets[pgs[j]]
            inter = len(a & b)
            union = len(a | b)
            rows.append({
                "pattern_i": pgs[i], "pattern_j": pgs[j],
                "gene_i": label.get(pgs[i], ""), "gene_j": label.get(pgs[j], ""),
                "n_i": len(a), "n_j": len(b), "n_shared": inter,
                "jaccard": round(inter / union, 4) if union else float("nan"),
            })
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"wrote {out_path} ({len(df)} pattern pairs from {len(pgs)} patterns)", file=sys.stderr)
    if len(df):
        print(df.to_string(index=False), file=sys.stderr)
    return df


def main(argv: list[str] | None = None) -> None:
    """CLI entry point (subcommands ``select`` and ``jaccard``)."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("select", help="Rank pattern groups; emit representative sequences + meta.")
    s.add_argument("--hits", type=Path, required=True, help="Annotated unitig-hit TSV.")
    s.add_argument("--top", type=int, default=4, help="Number of top distinct pattern groups (default 4).")
    s.add_argument("--out-seqs", type=Path, required=True, help="Representative unitig sequences (grep input).")
    s.add_argument("--out-meta", type=Path, required=True, help="Meta TSV (rep_variant -> pattern_group).")

    j = sub.add_parser("jaccard", help="Pairwise between-pattern carrier Jaccard from grep output.")
    j.add_argument("--matrix-lines", type=Path, required=True, help="grep -F -f output from the unitig matrix.")
    j.add_argument("--meta", type=Path, required=True, help="The select-stage meta TSV.")
    j.add_argument("--out", type=Path, required=True, help="Output pairwise Jaccard TSV.")

    args = p.parse_args(argv)
    if args.mode == "select":
        select(args.hits, args.top, args.out_seqs, args.out_meta)
    else:
        jaccard(args.matrix_lines, args.meta, args.out)


if __name__ == "__main__":
    main()
