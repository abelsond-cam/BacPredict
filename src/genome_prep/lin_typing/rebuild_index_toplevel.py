"""Regenerate the top-level files of a MiST index from its surviving per-locus directories.

A MiST index is 629 per-locus directories plus four small files at its root. On the shared CSD3
copy the root files were deleted while every locus directory survived, which makes ``mist call``
fail at ``FileNotFoundError: .../loci.txt`` before it does any work.

Three of the four are derivable from what is still on disk, exactly as ``mist index`` builds them:

``loci.txt``
    One locus name per line.
``loci_repr.fasta`` (+ its minimap2 index)
    The per-locus ``<locus>-clustered.fasta`` files concatenated — the seed-alignment target.
``profiles.tsv``
    **Not** derivable. It comes from the scheme and is installed separately; see
    ``build_profiles_table.py``. Without it MiST calls alleles but names no sublineage.

Rebuilding is cheap; re-downloading and re-clustering the scheme is not, which is the whole point.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mist.app.utils import minimap2utils, sequenceutils

logger = logging.getLogger(__name__)

LOCI_TXT = "loci.txt"
LOCI_REPR = "loci_repr.fasta"
PROFILES = "profiles.tsv"


def discover_loci(db_dir: Path) -> list[str]:
    """Return the locus names of ``db_dir``, sorted, requiring each to look like a MiST locus."""
    loci = []
    for child in sorted(db_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "mist_db.json").exists():
            logger.warning("%s has no mist_db.json — not a locus dir, skipping", child.name)
            continue
        if not (child / f"{child.name}-clustered.fasta").exists():
            raise SystemExit(f"{child}: missing {child.name}-clustered.fasta, cannot rebuild")
        loci.append(child.name)
    if not loci:
        raise SystemExit(f"{db_dir}: no locus directories found")
    return loci


def run(*, db_dir: Path, profiles_tsv: Path | None = None, force: bool = False) -> None:
    """Rebuild ``loci.txt`` and ``loci_repr.fasta``, and install ``profiles.tsv`` if given."""
    db_dir = db_dir.expanduser().resolve()
    loci = discover_loci(db_dir)
    logger.info("%s: %d loci", db_dir, len(loci))

    path_repr = db_dir / LOCI_REPR
    if path_repr.exists() and not force:
        logger.info("%s already exists, leaving it (pass --force to rebuild)", path_repr)
    else:
        nb_seqs = sequenceutils.merge_fasta_files(
            paths_fasta=[db_dir / locus / f"{locus}-clustered.fasta" for locus in loci],
            path_out=path_repr,
        )
        minimap2utils.create_index(path_repr)
        logger.info("wrote %s (%d representative sequences) and its minimap2 index", path_repr, nb_seqs)

    (db_dir / LOCI_TXT).write_text("".join(f"{locus}\n" for locus in loci))
    logger.info("wrote %s", db_dir / LOCI_TXT)

    if profiles_tsv is not None:
        target = db_dir / PROFILES
        target.write_bytes(profiles_tsv.read_bytes())
        n_rows = sum(1 for _ in target.open()) - 1
        logger.info("installed %s -> %s (%d profiles)", profiles_tsv, target, n_rows)
    elif not (db_dir / PROFILES).exists():
        logger.warning(
            "no %s in %s — mist call will succeed but every genome will come back without a "
            "sublineage. Build one with build_profiles_table.py.", PROFILES, db_dir,
        )

    manifest = {
        "db_dir": str(db_dir),
        "n_loci": len(loci),
        "profiles_installed_from": str(profiles_tsv) if profiles_tsv else None,
        "has_profiles": (db_dir / PROFILES).exists(),
        "note": "Top-level index files rebuilt from surviving per-locus dirs; allele data untouched.",
    }
    (db_dir / "rebuild.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-dir", type=Path, required=True, help="MiST index directory to repair.")
    p.add_argument("--profiles-tsv", type=Path, default=None, help="Profile table to install into the index.")
    p.add_argument("--force", action="store_true", help="Rebuild loci_repr.fasta even if it exists.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    run(db_dir=args.db_dir, profiles_tsv=args.profiles_tsv, force=args.force)


if __name__ == "__main__":
    main()
