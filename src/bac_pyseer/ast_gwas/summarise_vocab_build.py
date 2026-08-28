"""Collect the per-drug ``trainval_vocab`` build audits into one table, and say whether it passed.

The rebuild's whole claim is that no holdout genome shaped any drug's unitig vocabulary. That claim
is asserted per drug, per stage, in 22 separate ``leakage_audit.json`` files — which is the right
place for it to be enforced but a poor place to read it from. This module reduces those 22 files plus
the artifacts beside them to one row per drug, and returns non-zero if any row is not clean.

**A missing section counts as UNCHECKED, never as passed.** That distinction is the entire point:
each stage's audit is written only when that stage actually ran, so an absent ``vocabulary`` block
means the assertion never executed. Treating absence as success would let a drug whose GGCAT job died
report a clean build — exactly the failure mode the audit exists to prevent. ``status`` is therefore
one of ``ok`` / ``FAIL`` / ``unchecked``, and only ``ok`` is a pass.

Sizes are collected in the same pass because C3 asks for them against the plan's extrapolation
(``size(N) ~ 27 GB x N/7,080``, which the measured builds undershoot substantially — the matrix is
dominated by carrier tokens, and a smaller cohort shortens every carrier list as well as thinning the
feature set).

Usage
-----
``python -m bac_pyseer.ast_gwas.summarise_vocab_build --vocab-root <root> --out-tsv <path>``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# (section, key, expected) — every assertion the build stages are supposed to have enforced. Restated
# here so the summary is a genuine second read of the audit files rather than a reformatting of them:
# if a stage recorded a payload but the enforcing branch was never reached, this still catches it.
CHECKS: tuple[tuple[str, str, int], ...] = (
    ("reflist", "n_holdout_in_reflist", 0),
    ("reflist", "n_outside_trainval", 0),
    ("vocabulary", "n_holdout_coloured", 0),
    ("vocabulary", "n_missing_from_graph", 0),
    ("vocabulary", "n_extra_in_graph", 0),
    ("vocabulary", "n_holdout_in_matrix_head", 0),
    ("clusters", "n_missing", 0),
    ("clusters", "n_extra", 0),
)
REQUIRED_SECTIONS = ("reflist", "vocabulary", "clusters")


def _size(path: Path) -> int:
    """Bytes, or 0 when absent — never raise, so one missing artifact cannot hide the other 21 rows."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def summarise_drug(vocab_root: Path, drug: str) -> dict[str, object]:
    """Reduce one drug's audit file and artifacts to a single row.

    Parameters
    ----------
    vocab_root
        The ``<organism>_trainval_vocab`` directory holding one subdirectory per drug.
    drug
        Drug name, i.e. the subdirectory name.

    Returns
    -------
    dict
        One row. ``status`` is ``"ok"`` only when every required section is present *and* every
        check in :data:`CHECKS` holds; ``"unchecked"`` when a section is missing; ``"FAIL"`` when a
        present section violates a check. ``notes`` names what went wrong.
    """
    drug_dir = vocab_root / drug
    audit_path = drug_dir / "leakage_audit.json"
    row: dict[str, object] = {"drug": drug, "status": "unchecked", "notes": ""}

    if not audit_path.exists():
        row["notes"] = "no leakage_audit.json — the build never started"
        return row
    audit = json.loads(audit_path.read_text())

    reflist, vocab, clusters = (audit.get(s) or {} for s in REQUIRED_SECTIONS)
    row.update(
        n_reflist=reflist.get("n_reflist"),
        n_train=reflist.get("n_train"),
        n_validate=reflist.get("n_validate"),
        n_holdout=reflist.get("n_holdout"),
        min_samples_floor=reflist.get("min_samples_floor"),
        n_colors=vocab.get("n_colors"),
        n_holdout_coloured=vocab.get("n_holdout_coloured"),
        n_clusters=clusters.get("n_clusters"),
        n_in_other=clusters.get("n_in_other"),
        matrix_bytes=_size(drug_dir / "unitigs" / "unitigs.pyseer.gz"),
        colormap_bytes=_size(drug_dir / "unitigs" / "colormap_ranges.csv"),
        fasta_bytes=_size(drug_dir / "unitigs" / "unitigs_ggcat.fa.gz"),
        triangle_bytes=_size(drug_dir / "structure" / "mash_triangle.txt"),
    )

    missing = [s for s in REQUIRED_SECTIONS if s not in audit]
    failures = [
        f"{sec}.{key}={audit[sec][key]} (expected {want})"
        for sec, key, want in CHECKS
        if sec in audit and key in audit[sec] and audit[sec][key] != want
    ]
    # n_colors is GGCAT's own record of which genomes supplied a colour. If it disagrees with the
    # reflist we handed it, the graph on disk was built from a different cohort — the silent-reuse
    # trap that a fresh per-drug OUT_DIR is supposed to prevent and this is the backstop for.
    if vocab and reflist and vocab.get("n_colors") != reflist.get("n_reflist"):
        failures.append(f"n_colors={vocab.get('n_colors')} != n_reflist={reflist.get('n_reflist')}")
    if not row["matrix_bytes"] and "vocabulary" in audit:
        failures.append("unitigs.pyseer.gz is absent or empty")

    if failures:
        row["status"], row["notes"] = "FAIL", "; ".join(failures)
    elif missing:
        row["status"], row["notes"] = "unchecked", f"no {'/'.join(missing)} section — that stage never ran"
    else:
        row["status"] = "ok"
    return row


def run(vocab_root: Path, out_tsv: Path | None, drugs: list[str] | None = None) -> int:
    """Summarise every drug under ``vocab_root`` → printed table, optional TSV, process exit code."""
    names = sorted(drugs or [d.name for d in vocab_root.iterdir() if d.is_dir()])
    if not names:
        raise SystemExit(f"no drug directories under {vocab_root}")
    rows = [summarise_drug(vocab_root, d) for d in names]

    gib = 1024**3
    header = f"{'drug':<32} {'status':<9} {'n_ref':>6} {'floor':>5} {'colours':>7} {'hold':>4} {'clus':>4} {'matrix':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        matrix_bytes = int(r.get("matrix_bytes") or 0)
        cells = [
            f"{r['drug']:<32}",
            f"{r['status']:<9}",
            f"{r.get('n_reflist') or '-':>6}",
            f"{r.get('min_samples_floor') or '-':>5}",
            f"{r.get('n_colors') or '-':>7}",
            # 0 is the passing value here, so `or '-'` would print the pass as a blank.
            f"{r['n_holdout_coloured'] if r.get('n_holdout_coloured') is not None else '-':>4}",
            f"{r.get('n_clusters') or '-':>4}",
            f"{f'{matrix_bytes / gib:.2f}G' if matrix_bytes else '-':>8}",
        ]
        print(" ".join(cells))
        if r["notes"]:
            print(f"{'':<32} └─ {r['notes']}")

    ok = [r for r in rows if r["status"] == "ok"]
    total_bytes = sum(int(r.get("matrix_bytes") or 0) + int(r.get("colormap_bytes") or 0) for r in rows)
    print(
        f"\n{len(ok)}/{len(rows)} clean · "
        f"{sum(1 for r in rows if r['status'] == 'FAIL')} FAIL · "
        f"{sum(1 for r in rows if r['status'] == 'unchecked')} unchecked"
    )
    print(f"matrix + colormap on disk: {total_bytes / gib:.1f} GiB across {len(rows)} drugs")
    if ok:
        n = [int(r["n_reflist"]) for r in ok if r.get("n_reflist")]
        print(f"reflists: {min(n)}–{max(n)} genomes (mean {sum(n) / len(n):.0f}) of the 7,080-genome cohort")

    if out_tsv:
        cols = list(rows[0])
        for r in rows:
            cols += [c for c in r if c not in cols]
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        with out_tsv.open("w") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
        print(f"wrote {out_tsv}")

    return 0 if len(ok) == len(rows) else 1


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vocab-root", type=Path, required=True, help="<organism>_trainval_vocab directory")
    p.add_argument("--out-tsv", type=Path, default=None, help="also write the table here")
    p.add_argument("--drug", action="append", dest="drugs", default=None, help="restrict to these drugs")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    sys.exit(run(args.vocab_root, args.out_tsv, args.drugs))


if __name__ == "__main__":
    main()
