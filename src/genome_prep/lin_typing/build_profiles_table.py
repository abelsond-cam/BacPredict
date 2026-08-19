"""Assemble the ``profiles.tsv`` that gives a MiST index its LIN codes.

``mist index --profiles`` copies a profile table into the index, and ``mist call`` reads it to turn
an allele vector into a ``scgST`` plus its ``LINcode`` / ``Phylogroup`` / ``Sublineage`` /
``Clonal group``. Without it MiST still calls alleles but can name no sublineage at all.

Two sources are supported, and **which one you used is recorded in the sidecar manifest** because it
changes what an imperfect match means:

``--public-tsv``
    The scheme's ``profiles_csv`` endpoint. Unauthenticated it is capped at profiles submitted on or
    before 2024-12-31 — for this cohort that is roughly half the profiles actually needed, so the cap
    is not cosmetic.
``--archived-json-dir``
    Previously computed MiST results. ``profile.alleles`` in a result JSON is the **matched reference
    profile**, not the query's own calls (verified: on an 87.9%-match call the two differ at exactly
    the unmatched loci), so archived results are legitimate reference rows carrying LIN assignments
    made against the full database at the time they were run.

Rows are keyed by ``scgST`` and the public table always wins a collision.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

#: Non-locus columns, in the order the scheme's ``profiles_csv`` emits them.
KEY_COLUMN = "scgST"
METADATA_COLUMNS = ("LINcode", "Phylogroup", "Sublineage", "Clonal group")


def read_public(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Read the scheme profile table, returning ``(header, {scgST: row})``."""
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if header[0] != KEY_COLUMN:
            raise SystemExit(f"{path}: expected first column {KEY_COLUMN!r}, got {header[0]!r}")
        rows = {row[0]: row for row in reader if row}
    logger.info("%s: %d profiles, %d columns", path, len(rows), len(header))
    return header, rows


def rows_from_archived(
    json_dirs: list[Path], header: list[str], *, min_pct_match: float = 0.0
) -> dict[str, list[str]]:
    """Harvest reference-profile rows from archived MiST result JSONs.

    Parameters
    ----------
    json_dirs
        Directories of MiST ``--out-json`` results.
    header
        Column order to emit, taken from the public table so the two concatenate cleanly.
    min_pct_match
        Skip archived calls below this match percentage. The reference profile itself is exact
        regardless, so the default keeps everything; raise it only to be conservative.
    """
    locus_columns = [c for c in header if c != KEY_COLUMN and c not in METADATA_COLUMNS]
    harvested: dict[str, list[str]] = {}
    n_seen = n_no_profile = n_low = n_incomplete = 0

    for d in json_dirs:
        if not d.is_dir():
            logger.warning("archived-json dir does not exist, skipping: %s", d)
            continue
        for path in d.glob("*.json"):
            n_seen += 1
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                logger.warning("unreadable JSON, skipping: %s", path)
                continue
            profile = data.get("profile") or {}
            scgst = profile.get("name")
            alleles = profile.get("alleles") or {}
            metadata = dict(profile.get("metadata") or [])
            if not scgst or not alleles:
                n_no_profile += 1
                continue
            if (profile.get("pct_match") or 0.0) < min_pct_match:
                n_low += 1
                continue
            if scgst in harvested:
                continue
            missing = [c for c in locus_columns if c not in alleles]
            if missing:
                n_incomplete += 1
                continue
            row = []
            for column in header:
                if column == KEY_COLUMN:
                    row.append(str(scgst))
                elif column in METADATA_COLUMNS:
                    row.append(str(metadata.get(column, "")))
                else:
                    row.append(str(alleles[column]))
            harvested[scgst] = row

    logger.info(
        "archived: %d JSONs -> %d distinct profiles (%d no profile, %d below min-pct-match, "
        "%d missing loci)", n_seen, len(harvested), n_no_profile, n_low, n_incomplete,
    )
    return harvested


def run(
    *, public_tsv: Path, out_tsv: Path, archived_json_dirs: list[Path] | None = None,
    min_pct_match: float = 0.0,
) -> None:
    """Write the merged profile table and its manifest."""
    header, rows = read_public(public_tsv)
    n_public = len(rows)

    archived = rows_from_archived(archived_json_dirs or [], header, min_pct_match=min_pct_match)
    n_new = 0
    for scgst, row in archived.items():
        if scgst not in rows:
            rows[scgst] = row
            n_new += 1

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_tsv.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows[k] for k in sorted(rows, key=lambda s: (len(s), s)))

    manifest = {
        "out_tsv": str(out_tsv),
        "public_tsv": str(public_tsv),
        "archived_json_dirs": [str(d) for d in (archived_json_dirs or [])],
        "min_pct_match": min_pct_match,
        "n_public": n_public,
        "n_archived_distinct": len(archived),
        "n_archived_added": n_new,
        "n_total": len(rows),
        "columns": len(header),
        "note": (
            "Archived rows are reference profiles recovered from prior MiST results, not new LIN "
            "assignments. A public table downloaded without authentication is capped at profiles "
            "submitted on or before 2024-12-31."
        ),
    }
    out_tsv.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("wrote %s: %d profiles (%d public + %d recovered)", out_tsv, len(rows), n_public, n_new)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--public-tsv", type=Path, required=True, help="Scheme profiles_csv download.")
    p.add_argument("--out-tsv", type=Path, required=True)
    p.add_argument("--archived-json-dir", type=Path, action="append", default=[],
                   help="Directory of prior MiST result JSONs to harvest reference profiles from. Repeatable.")
    p.add_argument("--min-pct-match", type=float, default=0.0,
                   help="Skip archived calls below this pct_match. Default 0 (keep all).")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    run(public_tsv=args.public_tsv, out_tsv=args.out_tsv,
        archived_json_dirs=args.archived_json_dir, min_pct_match=args.min_pct_match)


if __name__ == "__main__":
    main()
