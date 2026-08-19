"""Build the MiST work list: cohort genomes that still have no sublineage.

A genome is skipped when it already carries a ``Sublineage`` in ``metadata_v2`` (through the primary
``Sample`` key *or* one of the fallback id columns), or when a MiST result JSON for it already
exists. Everything else needs LIN-typing.

The two skip reasons are counted separately in the manifest. A sudden jump in either is the signal
that the join or the salvaged-results inventory changed underneath us.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from bac_pyseer.ast_gwas.sublineage_from_metadata import (
    DEFAULT_SAMPLE_COLUMN,
    DEFAULT_SUBLINEAGE_COLUMN,
    load_sublineages,
)

logger = logging.getLogger(__name__)

#: Stripped from a result-JSON basename to recover the sample id. MiST names its output after the
#: input file, so a gzipped assembly becomes ``<Sample>.fa.gz.json``.
_JSON_SUFFIXES = (".json", ".gz", ".fa", ".fasta", ".fna")


def sample_id_from_json(path: Path) -> str:
    """Recover the sample id from a MiST result-JSON filename.

    Parameters
    ----------
    path
        Path to a result JSON, e.g. ``SAMEA4780982.fa.gz.json``.

    Returns
    -------
    str
        The sample id, e.g. ``SAMEA4780982``.
    """
    name = path.name
    changed = True
    while changed:
        changed = False
        for suffix in _JSON_SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
    return name


def already_called(dirs: list[Path]) -> set[str]:
    """Collect sample ids that already have a MiST result JSON in any of ``dirs``."""
    done: set[str] = set()
    for d in dirs:
        if not d.is_dir():
            logger.warning("existing-results dir does not exist, skipping: %s", d)
            continue
        found = {sample_id_from_json(p) for p in d.glob("*.json")}
        logger.info("%s: %d result JSONs", d, len(found))
        done |= found
    return done


def read_reflist(reflist: Path) -> list[tuple[str, str]]:
    """Read a ``Sample<TAB>path`` reflist, preserving order and rejecting malformed rows."""
    rows: list[tuple[str, str]] = []
    for n, line in enumerate(reflist.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2 or not parts[1].strip():
            raise SystemExit(f"{reflist}:{n}: expected 'Sample<TAB>path', got {line!r}")
        rows.append((parts[0].strip(), parts[1].strip()))
    if not rows:
        raise SystemExit(f"{reflist} yielded no samples")
    return rows


def run(
    *, reflist: Path, metadata_tsv: Path, out_tsv: Path,
    existing_json_dirs: list[Path] | None = None,
    sample_column: str = DEFAULT_SAMPLE_COLUMN,
    sublineage_column: str = DEFAULT_SUBLINEAGE_COLUMN,
) -> None:
    """Write the work list and its manifest."""
    rows = read_reflist(reflist)
    sublineage_of, recovered = load_sublineages(
        metadata_tsv, sample_column=sample_column, sublineage_column=sublineage_column
    )
    done = already_called(existing_json_dirs or [])

    work: list[tuple[str, str]] = []
    n_have_label = n_have_json = 0
    missing_assembly: list[str] = []
    for sample, path in rows:
        if sample in sublineage_of:
            n_have_label += 1
            continue
        if sample in done:
            n_have_json += 1
            continue
        if not Path(path).exists():
            missing_assembly.append(sample)
            continue
        work.append((sample, path))

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.write_text("".join(f"{s}\t{p}\n" for s, p in work))

    manifest = {
        "reflist": str(reflist),
        "metadata_tsv": str(metadata_tsv),
        "existing_json_dirs": [str(d) for d in (existing_json_dirs or [])],
        "n_cohort": len(rows),
        "n_skipped_have_sublineage": n_have_label,
        "n_skipped_already_called": n_have_json,
        "n_missing_assembly": len(missing_assembly),
        "missing_assembly": sorted(missing_assembly)[:50],
        "n_work": len(work),
        "recovered_by_fallback_column": {c: len(v) for c, v in recovered.items()},
        "sublineage_column": sublineage_column,
        "note": (
            "Sublineage comes from Pasteur BIGSdb LIN-typing. It is not derived from ST and these "
            "genomes are not 'missing an ST' — they are missing a LIN-typed sublineage."
        ),
    }
    manifest_path = out_tsv.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    logger.info(
        "cohort %d -> work %d (skipped: %d already labelled, %d already called, %d no assembly)",
        len(rows), len(work), n_have_label, n_have_json, len(missing_assembly),
    )
    logger.info("wrote %s and %s", out_tsv, manifest_path)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reflist", type=Path, required=True, help="Sample<TAB>path reflist for the cohort.")
    p.add_argument("--metadata-tsv", type=Path, required=True, help="metadata_v2_all_samples_and_columns.tsv")
    p.add_argument("--out-tsv", type=Path, required=True, help="Output Sample<TAB>path work list.")
    p.add_argument("--existing-json-dir", type=Path, action="append", default=[],
                   help="Directory of MiST result JSONs to treat as already done. Repeatable.")
    p.add_argument("--sample-column", default=DEFAULT_SAMPLE_COLUMN)
    p.add_argument("--sublineage-column", default=DEFAULT_SUBLINEAGE_COLUMN)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    run(
        reflist=args.reflist, metadata_tsv=args.metadata_tsv, out_tsv=args.out_tsv,
        existing_json_dirs=args.existing_json_dir, sample_column=args.sample_column,
        sublineage_column=args.sublineage_column,
    )


if __name__ == "__main__":
    main()
