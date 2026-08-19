"""Turn MiST result JSONs into a sublineage table, with the match quality kept alongside.

MiST always names a nearest profile, so every genome comes back with *a* sublineage. What separates
a fact from a guess is ``nb_matches`` — how many of the 629 loci the matched reference profile
actually agreed on — and that is why it travels as a first-class column rather than being collapsed
into a pass/fail at parse time.

**Output-format versions.** MiST <= 1.2 wrote a single ``profile`` object; 1.3 writes a ``profiles``
list so it can report several equally good STs. Both are read here. A parser that knows only one
shape returns nothing for the other, which is indistinguishable from a genome that failed to type —
the archived results on CSD3 are the old shape and any new run is the new one, so this matters.

**The gate.** ``--max-loci-mismatched`` defaults to 30, which is the ``max_missing`` the scheme
itself declares for its LIN codes. It is also where the evidence sits: scoring archived calls whose
exact profile is absent from the public table, nearest-profile sublineage agreed with the full
database on 312/312 at this cut and stayed at 100% out to 60, while every disagreement lay beyond
that (median 442 loci mismatched). Genomes failing the gate keep their row, flagged — dropping a
label is the caller's decision, not the parser's.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

N_LOCI = 629
#: BIGSdb declares max_missing=30 for this scheme's LIN codes.
DEFAULT_MAX_MISMATCHED = 30

COLUMNS = (
    "Sample", "Sublineage", "LINcode", "Clonal group", "Phylogroup", "scgST",
    "nb_matches", "pct_match", "loci_mismatched", "n_equivalent_profiles", "passes_gate",
)


def matched_profiles(data: dict) -> list[dict]:
    """Return the matched reference profiles, across both MiST output formats."""
    if data.get("profile"):
        return [data["profile"]]
    return list(data.get("profiles") or [])


def sample_id(data: dict, path: Path) -> str:
    """Recover the sample id, preferring what MiST recorded over the filename."""
    recorded = ((data.get("metadata") or {}).get("input") or {}).get("sample_id")
    if recorded:
        return str(recorded)
    name = path.name
    for suffix in (".json", ".gz", ".fa", ".fasta", ".fna"):
        while name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def parse_one(path: Path, *, max_mismatched: int) -> dict | None:
    """Parse a single result JSON into one output row, or ``None`` if it typed nothing."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("unreadable JSON, skipping: %s", path)
        return None

    profiles = matched_profiles(data)
    if not profiles:
        return None
    profile = profiles[0]
    metadata = dict(profile.get("metadata") or [])
    nb_matches = profile.get("nb_matches")
    n_mismatched = N_LOCI - nb_matches if nb_matches is not None else None

    return {
        "Sample": sample_id(data, path),
        "Sublineage": metadata.get("Sublineage", ""),
        "LINcode": metadata.get("LINcode", ""),
        "Clonal group": metadata.get("Clonal group", ""),
        "Phylogroup": metadata.get("Phylogroup", ""),
        "scgST": metadata.get("scgST", profile.get("name", "")),
        "nb_matches": nb_matches,
        "pct_match": round(profile["pct_match"], 4) if profile.get("pct_match") is not None else None,
        "loci_mismatched": n_mismatched,
        "n_equivalent_profiles": len(profiles),
        "passes_gate": bool(n_mismatched is not None and n_mismatched <= max_mismatched),
    }


def run(*, json_dirs: list[Path], out_tsv: Path, max_mismatched: int) -> None:
    """Parse every result JSON under ``json_dirs`` and write the sublineage table."""
    rows: dict[str, dict] = {}
    n_files = n_untyped = n_duplicate = 0
    for d in json_dirs:
        if not d.is_dir():
            logger.warning("results dir does not exist, skipping: %s", d)
            continue
        for path in sorted(d.glob("*.json")):
            n_files += 1
            row = parse_one(path, max_mismatched=max_mismatched)
            if row is None:
                n_untyped += 1
                continue
            if row["Sample"] in rows:
                n_duplicate += 1
                continue
            rows[row["Sample"]] = row

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(rows))

    n_pass = sum(1 for r in rows.values() if r["passes_gate"])
    n_tied = sum(1 for r in rows.values() if r["n_equivalent_profiles"] > 1)
    manifest = {
        "out_tsv": str(out_tsv),
        "json_dirs": [str(d) for d in json_dirs],
        "max_loci_mismatched": max_mismatched,
        "n_json_files": n_files,
        "n_untyped": n_untyped,
        "n_duplicate_samples": n_duplicate,
        "n_rows": len(rows),
        "n_passes_gate": n_pass,
        "n_fails_gate": len(rows) - n_pass,
        "n_with_tied_profiles": n_tied,
        "note": (
            "Sublineage is read verbatim from the matched reference profile's LIN metadata. It is "
            "not derived from ST. Rows failing the gate are retained and flagged, not deleted."
        ),
    }
    out_tsv.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    logger.info("%d JSONs -> %d rows (%d untyped, %d duplicate samples)",
                n_files, len(rows), n_untyped, n_duplicate)
    logger.info("gate (<=%d loci mismatched): %d pass, %d fail; %d had tied profiles",
                max_mismatched, n_pass, len(rows) - n_pass, n_tied)
    logger.info("wrote %s", out_tsv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json-dir", type=Path, action="append", required=True,
                   help="Directory of MiST result JSONs. Repeatable.")
    p.add_argument("--out-tsv", type=Path, required=True)
    p.add_argument("--max-loci-mismatched", type=int, default=DEFAULT_MAX_MISMATCHED,
                   help=f"Gate on match quality. Default {DEFAULT_MAX_MISMATCHED} (the scheme's own max_missing).")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    run(json_dirs=args.json_dir, out_tsv=args.out_tsv, max_mismatched=args.max_loci_mismatched)


if __name__ == "__main__":
    main()
