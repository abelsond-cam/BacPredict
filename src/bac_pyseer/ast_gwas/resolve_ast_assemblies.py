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

**The flat-directory assumption is Isambard's, and does not hold for Kp on CSD3.** There,
``raw/kleb_ast/assemblies`` does not exist at all: ``raw/assemblies`` is the whole-*Klebsiella*
store keyed by **GCA accession** (``GCA_900451215.1_44310_G01_genomic.fna.gz``), so a filename join
against BioSample ids resolves nothing, and the AST genomes themselves live sharded across
``seb/assemblies_2/klebsiella_pneumoniae__NN/<BioSample>.fa.gz`` batch directories. CSD3 ships the
join already made, as ``raw/assemblies_file_list.tsv`` — a ``Sample<TAB>path`` TSV in exactly the
format this module *emits*. ``--file-list`` consumes it, so that layout needs a filter rather than
a second resolution strategy. TB is unaffected: its assemblies are flat and BioSample-keyed on both
clusters, and the default path is already correct.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from bacpredict.engine.config import organism as organism_config
from bacpredict.engine.config import resolve_data_root
from bacpredict.engine.splits.load_splits import load_splits

logger = logging.getLogger(__name__)

# raw/<dir>/assemblies/<Sample><suffix> — the raw dir name does not match the processed task name.
RAW_SUBDIR = {"kp": "kleb_ast", "tb": "tb"}
ASSEMBLY_SUFFIXES = (".fa.gz", ".fna.gz", ".fa", ".fna", ".fasta.gz", ".fasta")
# The slices of a <drug>_split.csv, in the canonical row order every downstream module assumes.
SPLIT_SLICES = ("train", "validate", "holdout")


def assemblies_dir(organism_key: str, data_root: Path | str | None = None) -> Path:
    """``<root>/raw/<kleb_ast|tb>/assemblies`` for an organism key."""
    if organism_key not in RAW_SUBDIR:
        raise SystemExit(f"unknown organism {organism_key!r}; expected one of {sorted(RAW_SUBDIR)}")
    return resolve_data_root(data_root) / "raw" / RAW_SUBDIR[organism_key] / "assemblies"


def cohort_samples(
    organism_key: str, *, ast_sheet: Path | None = None, split_table: Path | None = None,
    data_root: Path | str | None = None, splits: Sequence[str] = SPLIT_SLICES,
) -> list[str]:
    """The sample universe: one drug's split table if given, else the whole AST sheet.

    ``splits`` selects which slices of that split table to take, in canonical order. The default is
    all three — the cohort every drug shares. Passing ``("train", "validate")`` yields a reflist that
    **no holdout genome can enter**, which is what makes a leakage-free unitig vocabulary possible:
    GGCAT then only ever sees genomes the model is allowed to learn from, so the feature
    *representation* is shaped by training sequence alone.

    Restricting the splits without a ``--split-table`` is rejected rather than ignored. The AST sheet
    carries no per-drug split assignment, so honouring the flag is impossible there — and silently
    returning the whole cohort would build a full-cohort vocabulary under a name asserting it is
    train+validate only, which is the exact contamination the flag exists to prevent.
    """
    if split_table is not None:
        unknown = [s for s in splits if s not in SPLIT_SLICES]
        if unknown:
            raise SystemExit(f"unknown split(s) {unknown}; expected a subset of {list(SPLIT_SLICES)}")
        if not splits:
            raise SystemExit("--splits selected nothing")
        label_map, train_ids, validate_ids, holdout_ids = load_splits(split_table)
        by_split = {"train": train_ids, "validate": validate_ids, "holdout": holdout_ids}
        return [sample for name in SPLIT_SLICES if name in splits for sample in by_split[name]]
    if set(splits) != set(SPLIT_SLICES):
        raise SystemExit("--splits needs --split-table; the AST sheet has no split assignment to filter on")
    sheet = ast_sheet if ast_sheet is not None else organism_config(organism_key).store_paths().ast_sheet
    df = pd.read_csv(sheet, low_memory=False)
    for col in ("Sample", "phenotype-BioSample_ID"):
        if col in df.columns:
            return df[col].dropna().astype(str).drop_duplicates().tolist()
    raise SystemExit(f"{sheet} has neither 'Sample' nor 'phenotype-BioSample_ID' (has {list(df.columns)[:10]})")


def load_file_list(path: Path) -> dict[str, Path]:
    """Read a ``Sample<TAB>path`` TSV into a mapping, tolerating a header row.

    For stores that are not one flat directory — see the module docstring on CSD3's Kp layout.
    Later rows win, matching the behaviour of re-running a resolution and appending.
    """
    mapping: dict[str, Path] = {}
    with path.open() as handle:
        for lineno, line in enumerate(handle):
            row = line.rstrip("\n").split("\t")
            if len(row) < 2 or not row[0]:
                continue
            if lineno == 0 and row[0] in {"Sample", "sample", "samples"}:
                continue  # header
            mapping[row[0]] = Path(row[1])
    if not mapping:
        raise SystemExit(f"{path} yielded no Sample<TAB>path rows")
    return mapping


def resolve_via_file_list(
    samples: list[str], mapping: dict[str, Path], *, check_exists: bool = True
) -> tuple[list[tuple[str, Path]], list[str]]:
    """Join samples through a pre-built mapping → ``(resolved pairs, unresolved samples)``.

    A sample absent from the mapping, or present but pointing at a file that does not exist (a
    broken symlink into a shared store is the realistic failure), counts as missing.
    """
    resolved: list[tuple[str, Path]] = []
    missing: list[str] = []
    for sample in samples:
        candidate = mapping.get(sample)
        if candidate is None or (check_exists and not candidate.is_file()):
            missing.append(sample)
        else:
            resolved.append((sample, candidate))
    return resolved, missing


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
    asm_dir: Path | None = None, file_list: Path | None = None,
    data_root: Path | str | None = None, check_exists: bool = True,
    splits: Sequence[str] = SPLIT_SLICES,
) -> dict[str, object]:
    """Write the ``Sample<TAB>path`` reflist plus a manifest recording what could not be resolved."""
    samples = cohort_samples(
        organism_key, ast_sheet=ast_sheet, split_table=split_table, data_root=data_root, splits=splits
    )
    if file_list is not None:
        source_desc = str(file_list)
        resolved, missing = resolve_via_file_list(
            samples, load_file_list(file_list), check_exists=check_exists
        )
    else:
        directory = asm_dir if asm_dir is not None else assemblies_dir(organism_key, data_root)
        source_desc = str(directory)
        resolved, missing = resolve(samples, directory, check_exists=check_exists)
    if not resolved:
        raise SystemExit(f"no assemblies resolved for {len(samples)} samples under {source_desc}")

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.write_text("".join(f"{s}\t{p}\n" for s, p in resolved))
    if missing:
        out_tsv.with_name(f"{out_tsv.stem}.missing.txt").write_text("".join(f"{s}\n" for s in missing))

    manifest = {
        "organism": organism_key,
        "assemblies_dir": source_desc,
        "resolution": "file_list" if file_list is not None else "directory_scan",
        "source": str(split_table or ast_sheet or organism_config(organism_key).store_paths().ast_sheet),
        "splits": list(splits),
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
    p.add_argument("--splits", default=",".join(SPLIT_SLICES),
                   help="Comma-separated split slices to keep (needs --split-table). "
                        "'train,validate' builds a vocabulary no holdout genome can enter.")
    p.add_argument("--assemblies-dir", type=Path, default=None, help="Override raw/<organism>/assemblies.")
    p.add_argument("--file-list", type=Path, default=None,
                   help="Resolve through a Sample<TAB>path TSV instead of scanning a directory "
                        "(CSD3 Kp: raw/assemblies_file_list.tsv). Takes precedence over --assemblies-dir.")
    p.add_argument("--data-root", default=None, help="Override the resolved data root.")
    p.add_argument("--no-check-exists", action="store_true",
                   help="Emit paths without stat()ing them (default is to check and report misses).")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(json.dumps(run(
        organism_key=args.organism, out_tsv=args.out_tsv, ast_sheet=args.ast_sheet,
        split_table=args.split_table, asm_dir=args.assemblies_dir, file_list=args.file_list,
        data_root=args.data_root, check_exists=not args.no_check_exists,
        splits=tuple(s.strip() for s in args.splits.split(",") if s.strip()),
    ), indent=2))


if __name__ == "__main__":
    main()
