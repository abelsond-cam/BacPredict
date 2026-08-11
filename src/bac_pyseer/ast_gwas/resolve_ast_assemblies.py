"""Resolve an AST cohort to ``Sample<TAB>assembly_path`` — the input GGCAT and mash both take.

:mod:`bac_pyseer.kleb_iso_source.resolve_assembly_paths` does this for the isolation-source cohort,
but it is hard-wired to the CSD3 ``metadata_v2`` sheet and its ``sr_assembly_file`` /
``lr_assembly_file`` / ``kpsc_final_list`` columns. The AST cohorts need none of that: their
assemblies are flat, BioSample-keyed files (``raw/kleb_ast/assemblies/<Sample>.fa.gz`` and
``raw/tb/assemblies/<Sample>.fa.gz``), so resolution is a filename join and does not depend on a
cluster-specific metadata TSV.

The cohort universe is the AST sheet (``binary_ast_with_split.csv``), which is already pruned to
samples with embeddings — so the unitig set is built over exactly the genomes the Bacformer arm
could also have used. Restrict further with ``--split-table`` to build over one drug's labelled
subset instead, though the intended use is one build per organism over the whole cohort, reused by
every drug.

Note the raw-directory naming asymmetry on disk: ``raw/kleb_ast/`` but ``raw/tb/``, while the
processed dirs are consistently ``processed/train_{kleb,tb}_ast/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from bacpredict.engine.config import organism as organism_config
from bacpredict.engine.config import resolve_data_root
from bacpredict.engine.splits.load_splits import load_splits

logger = logging.getLogger(__name__)

# raw/<dir>/assemblies/<Sample><suffix> — the raw dir name does not match the processed task name.
RAW_SUBDIR = {"kp": "kleb_ast", "tb": "tb"}
ASSEMBLY_SUFFIXES = (".fa.gz", ".fna.gz", ".fa", ".fna", ".fasta.gz", ".fasta")


def assemblies_dir(organism_key: str, data_root: Path | str | None = None) -> Path:
    """``<root>/raw/<kleb_ast|tb>/assemblies`` for an organism key."""
    if organism_key not in RAW_SUBDIR:
        raise SystemExit(f"unknown organism {organism_key!r}; expected one of {sorted(RAW_SUBDIR)}")
    return resolve_data_root(data_root) / "raw" / RAW_SUBDIR[organism_key] / "assemblies"


def cohort_samples(
    organism_key: str, *, ast_sheet: Path | None = None, split_table: Path | None = None,
    data_root: Path | str | None = None,
) -> list[str]:
    """The sample universe: one drug's split table if given, else the whole AST sheet."""
    if split_table is not None:
        label_map, train_ids, validate_ids, holdout_ids = load_splits(split_table)
        return [*train_ids, *validate_ids, *holdout_ids]
    sheet = ast_sheet if ast_sheet is not None else organism_config(organism_key).store_paths().ast_sheet
    df = pd.read_csv(sheet, low_memory=False)
    for col in ("Sample", "phenotype-BioSample_ID"):
        if col in df.columns:
            return df[col].dropna().astype(str).drop_duplicates().tolist()
    raise SystemExit(f"{sheet} has neither 'Sample' nor 'phenotype-BioSample_ID' (has {list(df.columns)[:10]})")


def resolve(
    samples: list[str], asm_dir: Path, *, check_exists: bool = True
) -> tuple[list[tuple[str, Path]], list[str]]:
    """Join samples to ``<asm_dir>/<Sample><suffix>`` → ``(resolved pairs, unresolved samples)``."""
    resolved: list[tuple[str, Path]] = []
    missing: list[str] = []
    for sample in samples:
        for suffix in ASSEMBLY_SUFFIXES:
            candidate = asm_dir / f"{sample}{suffix}"
            if candidate.is_file():
                resolved.append((sample, candidate))
                break
        else:
            if check_exists:
                missing.append(sample)
            else:
                resolved.append((sample, asm_dir / f"{sample}{ASSEMBLY_SUFFIXES[0]}"))
    return resolved, missing


def run(
    *, organism_key: str, out_tsv: Path, ast_sheet: Path | None = None, split_table: Path | None = None,
    asm_dir: Path | None = None, data_root: Path | str | None = None, check_exists: bool = True,
) -> dict[str, object]:
    """Write the ``Sample<TAB>path`` reflist plus a manifest recording what could not be resolved."""
    samples = cohort_samples(
        organism_key, ast_sheet=ast_sheet, split_table=split_table, data_root=data_root
    )
    directory = asm_dir if asm_dir is not None else assemblies_dir(organism_key, data_root)
    resolved, missing = resolve(samples, directory, check_exists=check_exists)
    if not resolved:
        raise SystemExit(f"no assemblies resolved for {len(samples)} samples under {directory}")

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.write_text("".join(f"{s}\t{p}\n" for s, p in resolved))
    if missing:
        out_tsv.with_name(f"{out_tsv.stem}.missing.txt").write_text("".join(f"{s}\n" for s in missing))

    manifest = {
        "organism": organism_key,
        "assemblies_dir": str(directory),
        "source": str(split_table or ast_sheet or organism_config(organism_key).store_paths().ast_sheet),
        "n_cohort": len(samples),
        "n_resolved": len(resolved),
        "n_missing": len(missing),
        "output": str(out_tsv),
    }
    out_tsv.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("resolved %d/%d assemblies -> %s (%d missing)", len(resolved), len(samples), out_tsv, len(missing))
    return manifest


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--organism", choices=sorted(RAW_SUBDIR), required=True)
    p.add_argument("--out-tsv", type=Path, required=True, help="Output Sample<TAB>path reflist.")
    p.add_argument("--ast-sheet", type=Path, default=None, help="Override the AST sheet defining the cohort.")
    p.add_argument("--split-table", type=Path, default=None,
                   help="Restrict to one drug's labelled samples instead of the whole cohort.")
    p.add_argument("--assemblies-dir", type=Path, default=None, help="Override raw/<organism>/assemblies.")
    p.add_argument("--data-root", default=None, help="Override the resolved data root.")
    p.add_argument("--no-check-exists", action="store_true",
                   help="Emit paths without stat()ing them (default is to check and report misses).")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(json.dumps(run(
        organism_key=args.organism, out_tsv=args.out_tsv, ast_sheet=args.ast_sheet,
        split_table=args.split_table, asm_dir=args.assemblies_dir, data_root=args.data_root,
        check_exists=not args.no_check_exists,
    ), indent=2))


if __name__ == "__main__":
    main()
