"""Per-drug assertions that a unitig vocabulary was built without its own holdout genomes.

Why a separate module
---------------------
The train+validate-vocabulary rebuild rests on one claim: **no holdout genome supplied a colour to
the GGCAT graph**, so the feature representation — not just the labels — was shaped by training
sequence alone. That claim is cheap to make and expensive to check after the fact, and every way of
getting it wrong is silent:

* a reflist built without ``--splits``, which yields the full cohort under a directory named
  ``trainval_vocab``;
* ``run_ggcat_unitigs.sh`` skipping the build because the graph artifacts already exist, so a wrong
  ``OUT_DIR`` returns the *full-cohort* vocabulary with no error anywhere;
* ``extract_hit_submatrix`` reusing a cached sub-matrix on path existence alone, with no provenance
  check.

None of those fail. They all produce a well-formed matrix and a plausible AUROC. So each stage gets
an assertion here, written into one ``leakage_audit.json`` per drug, and the assertion is against the
*tool's own output* wherever possible — ``color_names.jsonl`` is GGCAT's record of which genomes it
took colours from, which is stronger evidence than the reflist we believe we handed it.

Sections accumulate into the same file, so a partial audit is visibly partial rather than absent.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bac_pyseer.kleb_iso_source.ggcat_to_pyseer import load_color_names
from bacpredict.engine.splits.load_splits import load_splits

logger = logging.getLogger(__name__)


def _reflist_samples(path: Path) -> list[str]:
    """``Sample<TAB>path`` -> the sample column, order preserved."""
    return [ln.split("\t")[0] for ln in path.read_text().splitlines() if ln.strip()]


def update_audit(audit_path: Path, section: str, payload: dict) -> dict:
    """Merge one section into the drug's ``leakage_audit.json`` and return the whole file."""
    audit = json.loads(audit_path.read_text()) if audit_path.is_file() else {}
    audit[section] = payload
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2))
    return audit


def audit_reflist(reflist: Path, split_table: Path) -> dict:
    """Assert the reflist is exactly the train+validate genomes and touches no holdout genome."""
    samples = _reflist_samples(reflist)
    _, train_ids, validate_ids, holdout_ids = load_splits(split_table)
    trainval, holdout = set(train_ids) | set(validate_ids), set(holdout_ids)
    leaked = sorted(set(samples) & holdout)
    outside = sorted(set(samples) - trainval)
    payload = {
        "reflist": str(reflist), "split_table": str(split_table),
        "n_reflist": len(samples), "n_unique": len(set(samples)),
        "n_train": len(train_ids), "n_validate": len(validate_ids), "n_holdout": len(holdout_ids),
        "n_holdout_in_reflist": len(leaked),
        "n_outside_trainval": len(outside),
        # Genomes in train+validate with no assembly. Expected small and non-zero; they cannot be
        # coloured, so they are absent from the vocabulary and from the GWAS alike.
        "n_trainval_without_assembly": len(trainval - set(samples)),
        "min_samples_floor": math.ceil(0.01 * len(samples)),
    }
    if leaked:
        raise SystemExit(
            f"{len(leaked)} holdout genome(s) are in {reflist} — the vocabulary would be built over "
            f"the test set. e.g. {leaked[:5]}"
        )
    if outside:
        raise SystemExit(f"{len(outside)} reflist genome(s) are in neither train nor validate, e.g. {outside[:5]}")
    if len(set(samples)) != len(samples):
        raise SystemExit(f"{reflist} lists a sample more than once")
    return payload


def audit_vocabulary(color_names: Path, reflist: Path, split_table: Path,
                     matrix_gz: Path | None = None, sample_lines: int = 2000) -> dict:
    """Assert GGCAT's own colour record matches the reflist, from the tool's output rather than ours.

    ``color_names.jsonl`` is written by ``ggcat dump-colors`` and names every genome that supplied a
    colour to the graph. Checking it — rather than the reflist we believe we passed — is what closes
    the silent-reuse trap: a stale build in the wrong ``OUT_DIR`` is reused without complaint, and its
    colour names are the only artifact that still remembers which cohort it was actually built from.
    """
    # The converter's own reader, so the audit agrees with what actually built the matrix — including
    # its gap check on the colour indices.
    names = load_color_names(color_names)

    expected = set(_reflist_samples(reflist))
    _, _, _, holdout_ids = load_splits(split_table)
    holdout = set(holdout_ids)
    missing, extra = sorted(expected - set(names)), sorted(set(names) - expected)
    leaked = sorted(set(names) & holdout)

    payload = {
        "color_names": str(color_names), "n_colors": len(names), "n_reflist": len(expected),
        "n_missing_from_graph": len(missing), "n_extra_in_graph": len(extra),
        "n_holdout_coloured": len(leaked),
    }
    if leaked:
        raise SystemExit(
            f"{len(leaked)} holdout genome(s) supplied colours to {color_names} — this graph was built "
            f"over the test set. e.g. {leaked[:5]}"
        )
    if missing or extra:
        raise SystemExit(
            f"{color_names} does not match {reflist}: {len(missing)} reflist genomes absent from the "
            f"graph, {len(extra)} graph colours not in the reflist. A stale build from another cohort "
            f"is the usual cause — check OUT_DIR. e.g. missing={missing[:3]} extra={extra[:3]}"
        )

    if matrix_gz is not None and matrix_gz.is_file():
        # Independent of the colour names: read the head of the matrix and confirm no holdout id
        # appears as a carrier. Cheap, and it catches a matrix inherited from a different build.
        seen: set[str] = set()
        with gzip.open(matrix_gz, "rt") as fh:
            for i, line in enumerate(fh):
                if i >= sample_lines:
                    break
                _, _, rest = line.partition(" | ")
                seen.update(tok.rpartition(":")[0] for tok in rest.split())
        matrix_leak = sorted(seen & holdout)
        payload |= {"matrix_lines_scanned": min(sample_lines, i + 1),
                    "matrix_carriers_seen": len(seen), "n_holdout_in_matrix_head": len(matrix_leak)}
        if matrix_leak:
            raise SystemExit(f"holdout genomes appear as carriers in {matrix_gz}, e.g. {matrix_leak[:5]}")
    return payload


def audit_clusters(clusters_tsv: Path, reflist: Path) -> dict:
    """Assert the lineage clusters cover exactly the reflist, and record the shape for comparison."""
    df = pd.read_csv(clusters_tsv, sep="\t", header=None, names=["Sample", "cluster"], dtype=str)
    expected = set(_reflist_samples(reflist))
    missing, extra = sorted(expected - set(df["Sample"])), sorted(set(df["Sample"]) - expected)
    sizes = df["cluster"].value_counts()
    payload = {
        "clusters_tsv": str(clusters_tsv), "n_rows": int(len(df)),
        "n_clusters": int(sizes.size), "n_in_other": int(sizes.get("other", 0)),
        "n_missing": len(missing), "n_extra": len(extra),
        "largest_clusters": {str(k): int(v) for k, v in sizes.head(5).items()},
    }
    if missing or extra:
        raise SystemExit(
            f"{clusters_tsv} does not cover the reflist: {len(missing)} missing, {len(extra)} extra. "
            f"The permutation null and --lineage would then describe a different cohort from the GWAS."
        )
    return payload


def audit_mash(fresh: Path, reference: Path) -> dict:
    """Assert a freshly-sketched similarity matrix equals the old cohort triangle's subset, exactly.

    Subsetting a mash triangle is mathematically identical to re-sketching, because each cell is a
    distance between one pair of genomes and no cohort-wide statistic enters. Re-sketching anyway
    costs a minute and turns that argument into a number an auditor can read off a file.
    """
    a = pd.read_csv(fresh, sep="\t", index_col=0)
    b = pd.read_csv(reference, sep="\t", index_col=0)
    shared = sorted(set(a.index) & set(b.index))
    if not shared:
        raise SystemExit(f"{fresh} and {reference} share no genome — wrong pair of files?")
    diff = np.abs(a.loc[shared, shared].to_numpy(float) - b.loc[shared, shared].to_numpy(float))
    payload = {
        "fresh": str(fresh), "reference": str(reference), "n_shared": len(shared),
        "n_fresh": int(len(a)), "n_reference": int(len(b)),
        "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
    }
    return payload


def audit_design(merge_manifest: Path) -> dict:
    """Carry the merge's own gates into the drug's audit file.

    The merge already asserts them and refuses to write a design that fails; recording them here is
    what makes the audit file a complete account of the run rather than a partial one, so a reader
    does not have to know which gates live in which manifest.
    """
    merged = json.loads(merge_manifest.read_text())
    keep = ("rows_from_ggcat", "rows_from_scanner", "n_features", "nnz",
            "verification", "holdout_coverage")
    payload = {"merge_manifest": str(merge_manifest), **{k: merged[k] for k in keep if k in merged}}
    verification = payload.get("verification", {})
    if not verification.get("n_shared"):
        raise SystemExit(
            f"{merge_manifest}: the scanner was never checked against GGCAT's colouring (n_shared=0). "
            f"Re-run the scan with --splits train,validate,holdout so the two operators overlap."
        )
    return payload


def main(argv: list[str] | None = None) -> None:
    """CLI entry point — one subcommand per audited stage, all writing the same JSON."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audit-json", type=Path, required=True, help="Per-drug leakage_audit.json (merged into).")
    sub = p.add_subparsers(dest="stage", required=True)

    r = sub.add_parser("reflist", help="The reflist is train+validate and touches no holdout genome.")
    r.add_argument("--reflist", type=Path, required=True)
    r.add_argument("--split-table", type=Path, required=True)

    v = sub.add_parser("vocabulary", help="GGCAT's own colour record matches the reflist.")
    v.add_argument("--color-names", type=Path, required=True, help="color_names.jsonl from ggcat dump-colors")
    v.add_argument("--reflist", type=Path, required=True)
    v.add_argument("--split-table", type=Path, required=True)
    v.add_argument("--matrix-gz", type=Path, default=None, help="Also scan the matrix head for holdout carriers.")
    v.add_argument("--sample-lines", type=int, default=2000)

    c = sub.add_parser("clusters", help="The lineage clusters cover exactly the reflist.")
    c.add_argument("--clusters-tsv", type=Path, required=True)
    c.add_argument("--reflist", type=Path, required=True)

    d = sub.add_parser("design", help="Record the merged design's gates in the drug's audit file.")
    d.add_argument("--merge-manifest", type=Path, required=True)

    m = sub.add_parser("mash", help="A fresh similarity matrix equals the old triangle's subset.")
    m.add_argument("--fresh", type=Path, required=True)
    m.add_argument("--reference", type=Path, required=True)
    m.add_argument("--max-abs-diff", type=float, default=0.0)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.stage == "reflist":
        payload = audit_reflist(args.reflist, args.split_table)
    elif args.stage == "vocabulary":
        payload = audit_vocabulary(args.color_names, args.reflist, args.split_table,
                                   args.matrix_gz, args.sample_lines)
    elif args.stage == "clusters":
        payload = audit_clusters(args.clusters_tsv, args.reflist)
    elif args.stage == "design":
        payload = audit_design(args.merge_manifest)
    else:
        payload = audit_mash(args.fresh, args.reference)
        if payload["max_abs_diff"] > args.max_abs_diff:
            update_audit(args.audit_json, args.stage, payload)
            raise SystemExit(
                f"fresh and reference similarity differ by {payload['max_abs_diff']:.3e} > "
                f"{args.max_abs_diff}. Subsetting a triangle should be identical to re-sketching, so "
                f"a non-zero difference means the two files are not the same cohort or the same metric."
            )

    update_audit(args.audit_json, args.stage, payload)
    print(json.dumps({args.stage: payload}, indent=2))


if __name__ == "__main__":
    main()
