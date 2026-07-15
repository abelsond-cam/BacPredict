"""Isambard glue: build the assembly/GFF path sheet that :mod:`annotate_amr_sidecar` needs.

On CSD3 the AMR-annotation worklist is drawn from ``metadata_v2``'s ``sr_assembly_file`` /
``sr_gff_file`` columns. That table is **not** staged on Isambard, but the raw inputs are — the Kp AST
cohort's assemblies live flat under ``raw/kleb_ast/assemblies/{Sample}.fa.gz`` and the Bakta GFFs under
a sharded tree ``raw/kleb_ast/gff/<shard>/{Sample}/{Sample}.bakta.gff3.gz``. This walks that layout once
(a single ``listdir`` per shard, not a recursive crawl) and writes a minimal 3-column TSV
(``Sample, sr_assembly_file, sr_gff_file`` with **absolute** paths) that ``annotate_amr_sidecar
--metadata`` consumes unchanged — the sidecar module's ``_resolve`` leaves absolute paths as-is, so
``--path-root`` is irrelevant.

A sample is emitted only when **both** its assembly and its GFF exist on disk; drops are counted. The
resulting sheet doubles as the record of exactly which genomes were offered to the annotator.

Isambard-only; when CSD3 returns, use ``metadata_v2`` directly instead.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_gff_index(gff_root: Path, gff_suffix: str) -> dict[str, Path]:
    """Map ``Sample -> {shard}/{Sample}/{Sample}{gff_suffix}`` by one listdir per shard (bounded)."""
    index: dict[str, Path] = {}
    for shard in sorted(p for p in gff_root.iterdir() if p.is_dir()):
        for sample_dir in shard.iterdir():
            if not sample_dir.is_dir():
                continue
            gff = sample_dir / f"{sample_dir.name}{gff_suffix}"
            if gff.exists():
                index[sample_dir.name] = gff
    logger.info("gff index: %d samples across %d shards under %s", len(index), len(list(gff_root.iterdir())), gff_root)
    return index


def build_sheet(
    ast_sheet: Path,
    assembly_dir: Path,
    gff_root: Path,
    *,
    assembly_suffix: str = ".fa.gz",
    gff_suffix: str = ".bakta.gff3.gz",
) -> pd.DataFrame:
    """``Sample, sr_assembly_file, sr_gff_file`` (absolute) for cohort samples with both inputs present."""
    samples = sorted(pd.read_csv(ast_sheet, usecols=["Sample"])["Sample"].astype(str).unique())
    gff_index = build_gff_index(gff_root, gff_suffix)

    rows, drops = [], {"no_assembly": 0, "no_gff": 0}
    for s in samples:
        asm = assembly_dir / f"{s}{assembly_suffix}"
        gff = gff_index.get(s)
        if not asm.exists():
            drops["no_assembly"] += 1
            continue
        if gff is None:
            drops["no_gff"] += 1
            continue
        rows.append({"Sample": s, "sr_assembly_file": str(asm), "sr_gff_file": str(gff)})
    logger.info("path sheet: %d/%d cohort samples have both inputs (drops: %s)", len(rows), len(samples), drops)
    return pd.DataFrame(rows, columns=["Sample", "sr_assembly_file", "sr_gff_file"])


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ast-sheet", type=Path, required=True,
                   help="binary_ast_with_split.csv — defines the cohort (Sample column).")
    p.add_argument("--assembly-dir", type=Path, required=True,
                   help="Dir of flat {Sample}.fa.gz assemblies (raw/kleb_ast/assemblies).")
    p.add_argument("--gff-root", type=Path, required=True,
                   help="Sharded GFF root {shard}/{Sample}/{Sample}.bakta.gff3.gz (raw/kleb_ast/gff).")
    p.add_argument("--out", type=Path, required=True, help="Output TSV path (the --metadata sheet).")
    p.add_argument("--assembly-suffix", type=str, default=".fa.gz")
    p.add_argument("--gff-suffix", type=str, default=".bakta.gff3.gz")
    args = p.parse_args()

    sheet = build_sheet(args.ast_sheet, args.assembly_dir, args.gff_root,
                        assembly_suffix=args.assembly_suffix, gff_suffix=args.gff_suffix)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(args.out, sep="\t", index=False)
    logger.info("wrote %d-row AMR path sheet -> %s", len(sheet), args.out)


if __name__ == "__main__":
    main()
