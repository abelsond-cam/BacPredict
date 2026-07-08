"""Reliable Kleborate-style AMR labels for the Kp AST cohort — a flat-index sidecar.

Bakta under-annotates acquired AMR genes for several classes, so keying our per-gene analysis on
the Bakta ``gene_name`` mislabels / drops carriers. This driver re-identifies AMR genes the way
Kleborate does — ``minimap2`` of the vendored CARD acquired-allele refs (+ the chromosomal QRDR /
OmpK / MgrB-PmrB refs) against each genome assembly (see
:func:`tl.embed.extract_proteins_from_gff_fna.annotate_amr_calls`) — and writes, **per sample**, a
small ``{Sample}_amr.parquet`` whose rows are the AMR calls keyed by **flat protein index** (the row
into ``{Sample}_esm_embeddings.pt``). No protein parquet is rewritten and **no genome is re-embedded**.

Correctness by construction + a guard:

- The AMR alignment runs against the *same* ``sr_assembly_file`` / ``sr_gff_file`` (from
  ``metadata_v2``) that built the existing protein parquet, so the hit→CDS overlap join uses the
  builder's own in-memory contig/coords — no contig-name↔``contig_idx`` reconstruction.
- Before writing, the re-extracted protein flat list is checked against the existing
  ``{Sample}_protein_sequences.parquet`` (length + sampled sequences). A mismatch means the
  ``flat_index`` would not align with the embedding rows, so the sample is **skipped and counted**
  rather than written with a wrong index.

Each sidecar row: ``Sample, flat_index, seqid, amr_allele, amr_gene_family, amr_class,
amr_drug_classes, amr_pct_id, amr_pct_cov, amr_source, amr_flags, bakta_gene_name, protein_id,
protein_sequence``. ``flat_index = -1`` marks an acquired CARD call with **no** overlapping CDS — the
Bakta-missed-the-gene case — kept so the miss can be quantified (those rows carry no embedding).

CPU-only; shells out to ``minimap2`` (provide via the BacPredict pixi env or PATH). Run as a chunked
SLURM array (``--start/--count``) over the cohort; see ``scripts/annotate_amr_parquet.sh``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd

from pangena_predict.locate_gene import flatten_proteins
from tl.embed.extract_proteins_from_gff_fna import extract_proteins_from_gff_fna

RDS_ROOT = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw")
# metadata_v2 stores sr_assembly_file / sr_gff_file *relative to the project_k root* (e.g.
# "seb/assemblies_2/.../X.fa.gz", "david/raw/.../X.bakta.gff3.gz"); resolve them against this.
PROJECT_K_ROOT = RDS_ROOT
DEFAULT_AST_SHEET = RDS_ROOT / "david" / "processed" / "train_kleb_ast" / "binary_ast_with_split.csv"
DEFAULT_METADATA = RDS_ROOT / "david" / "final" / "metadata_v2_all_samples_and_columns.tsv"
DEFAULT_PROTEIN_DIR = RDS_ROOT / "david" / "processed" / "klebsiella_protein_sequences"
DEFAULT_OUT_DIR = RDS_ROOT / "david" / "processed" / "train_kleb_ast" / "amr_annotation"
# Vendored Kleborate KpSC AMR refs live in the sibling BacHGT repo (read-only).
DEFAULT_AMR_REF_DIR = Path(
    "/home/dca36/workspace/BacHGT/src/bac_kleborate/refs/kleb_amr/inputs"
)

_SIDECAR_COLUMNS = [
    "Sample", "flat_index", "seqid", "amr_allele", "amr_gene_family", "amr_class",
    "amr_drug_classes", "amr_pct_id", "amr_pct_cov", "amr_source", "amr_flags",
    "bakta_gene_name", "bakta_locus_tag", "protein_id", "protein_sequence",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _resolve(path: str, root: Path) -> str:
    """Absolute path: leave absolute paths as-is, resolve relative ones against ``root``."""
    return path if Path(path).is_absolute() else str(root / path)


def build_worklist(
    ast_sheet: Path, metadata: Path, protein_dir: Path, path_root: Path
) -> pd.DataFrame:
    """The AST-cohort samples joined to their ``sr_assembly_file`` / ``sr_gff_file`` paths.

    Keeps only samples that (a) appear in the AST sheet, (b) have both assembly + GFF paths in
    ``metadata_v2``, and (c) already have a protein parquet (so the flat-order guard can run and
    an embedding exists). Relative assembly/GFF paths are resolved against ``path_root`` (the
    project_k root). Returns ``Sample, sr_assembly_file, sr_gff_file`` with absolute paths.
    """
    ast = pd.read_csv(ast_sheet, usecols=["Sample"])
    ast["Sample"] = ast["Sample"].astype(str)
    samples = set(ast["Sample"].unique())

    meta = pd.read_csv(metadata, sep="\t", usecols=["Sample", "sr_assembly_file", "sr_gff_file"],
                       low_memory=False)
    meta["Sample"] = meta["Sample"].astype(str)
    meta = meta[meta["Sample"].isin(samples)].dropna(subset=["sr_assembly_file", "sr_gff_file"])
    meta = meta.drop_duplicates(subset=["Sample"], keep="first").reset_index(drop=True)
    for col in ("sr_assembly_file", "sr_gff_file"):
        meta[col] = meta[col].astype(str).apply(lambda p: _resolve(p, path_root))

    has_parquet = meta["Sample"].apply(
        lambda s: (protein_dir / f"{s}_protein_sequences.parquet").exists()
    )
    n_no_parquet = int((~has_parquet).sum())
    if n_no_parquet:
        logger.warning("worklist: %d cohort samples have no protein parquet (skipped)", n_no_parquet)
    meta = meta[has_parquet].reset_index(drop=True)
    logger.info("worklist: %d samples to annotate (of %d AST-cohort samples)", len(meta), len(samples))
    return meta


def _flat_order_matches(reextracted_flat: list[dict], existing_parquet: Path) -> bool:
    """True if the re-extracted protein flat list aligns with the existing parquet (count + samples)."""
    existing = flatten_proteins(pd.read_parquet(existing_parquet))
    if len(existing) != len(reextracted_flat):
        return False
    n = len(existing)
    for i in {0, n // 2, n - 1}:
        if existing[i]["protein_sequence"] != reextracted_flat[i]["protein_sequence"]:
            return False
    return True


def annotate_one(
    sample: str,
    sr_assembly_file: str,
    sr_gff_file: str,
    *,
    protein_dir: Path,
    amr_ref_dir: Path,
    out_dir: Path,
    minimap2_bin: str,
    threads: int,
    skip_existing: bool,
) -> tuple[str, str, int]:
    """Annotate one genome and write its sidecar. Returns ``(sample, status, n_calls)``.

    ``status`` is ``ok`` / ``exists`` / ``misaligned`` / ``no_calls`` / ``error: ...``.
    """
    out_path = out_dir / f"{sample}_amr.parquet"
    if skip_existing and out_path.exists():
        return sample, "exists", 0
    try:
        out = extract_proteins_from_gff_fna(
            sr_gff_file, sr_assembly_file, keep_internal_stop=True,
            annotate_amr=True, amr_ref_dir=amr_ref_dir,
            minimap2_bin=minimap2_bin, amr_threads=threads,
        )
        flat = flatten_proteins(pd.Series(out))
        if not _flat_order_matches(flat, protein_dir / f"{sample}_protein_sequences.parquet"):
            return sample, "misaligned", 0

        by_index = {r["flat_index"]: r for r in flat}
        rows = []
        for call in out.get("amr_calls", []):
            fi = call["flat_index"]
            prot = by_index.get(fi) if fi >= 0 else None
            rows.append({
                "Sample": sample,
                "flat_index": fi,
                "seqid": call.get("seqid"),
                "amr_allele": call.get("amr_allele"),
                "amr_gene_family": call.get("amr_gene_family"),
                "amr_class": call.get("amr_class"),
                "amr_drug_classes": call.get("amr_drug_classes"),
                "amr_pct_id": call.get("amr_pct_id"),
                "amr_pct_cov": call.get("amr_pct_cov"),
                "amr_source": call.get("amr_source"),
                "amr_flags": call.get("amr_flags"),
                "bakta_gene_name": prot["gene_name"] if prot else None,
                "bakta_locus_tag": prot["protein_name"] if prot else None,
                "protein_id": prot["protein_id"] if prot else None,
                "protein_sequence": prot["protein_sequence"] if prot else None,
            })
        df = pd.DataFrame(rows, columns=_SIDECAR_COLUMNS)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, engine="pyarrow", compression="snappy")
        return sample, ("ok" if rows else "no_calls"), len(rows)
    except Exception as e:  # noqa: BLE001 — worker isolates per-sample failures
        logger.error("%s: %s", sample, e)
        return sample, f"error: {e}", 0


def _worker(args_tuple: tuple) -> tuple[str, str, int]:
    """Pool adapter for :func:`annotate_one`."""
    sample, asm, gff, kw = args_tuple
    return annotate_one(sample, asm, gff, **kw)


def run(
    *,
    ast_sheet: Path,
    metadata: Path,
    protein_dir: Path,
    amr_ref_dir: Path,
    out_dir: Path,
    minimap2_bin: str,
    threads: int,
    workers: int,
    skip_existing: bool,
    start: int,
    count: int | None,
    samples: list[str] | None,
    path_root: Path = PROJECT_K_ROOT,
    dry_run: bool = False,
) -> None:
    """Build the worklist, annotate each genome in parallel, write per-sample sidecars."""
    work = build_worklist(ast_sheet, metadata, protein_dir, path_root)
    if dry_run:
        logger.info("dry-run: worklist size = %d (chunk it with --start/--count for the array)", len(work))
        return
    if samples:
        work = work[work["Sample"].isin(samples)].reset_index(drop=True)
        logger.info("restricted to %d explicit sample(s)", len(work))
    if start or count is not None:
        end = start + count if count is not None else len(work)
        work = work.iloc[start:end].reset_index(drop=True)
        logger.info("chunk [%d:%s] -> %d samples", start, end, len(work))
    if work.empty:
        logger.warning("nothing to do")
        return

    kw = {"protein_dir": protein_dir, "amr_ref_dir": amr_ref_dir, "out_dir": out_dir,
          "minimap2_bin": minimap2_bin, "threads": threads, "skip_existing": skip_existing}
    tasks = [(r["Sample"], str(r["sr_assembly_file"]), str(r["sr_gff_file"]), kw)
             for _, r in work.iterrows()]

    n_workers = max(1, min(workers, cpu_count(), len(tasks)))
    logger.info("annotating %d genomes with %d workers", len(tasks), n_workers)
    t0 = time.time()
    status_counts: dict[str, int] = {}
    done = 0
    with Pool(processes=n_workers) as pool:
        for _sample, status, _n_calls in pool.imap_unordered(_worker, tasks):
            key = status.split(":")[0]
            status_counts[key] = status_counts.get(key, 0) + 1
            done += 1
            if done % 200 == 0:
                logger.info("progress %d/%d — status so far: %s", done, len(tasks), status_counts)
    dt = time.time() - t0
    logger.info("DONE %d genomes in %.1fs — status: %s", len(tasks), dt, status_counts)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ast-sheet", type=Path, default=DEFAULT_AST_SHEET)
    p.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    p.add_argument("--protein-dir", type=Path, default=DEFAULT_PROTEIN_DIR)
    p.add_argument("--amr-ref-dir", type=Path, default=DEFAULT_AMR_REF_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--path-root", type=Path, default=PROJECT_K_ROOT,
                   help="Root for resolving relative sr_assembly_file/sr_gff_file (project_k root).")
    p.add_argument("--minimap2-bin", type=str, default="minimap2")
    p.add_argument("--threads", type=int, default=4, help="minimap2 threads per genome.")
    p.add_argument("--workers", type=int, default=16, help="parallel genome workers.")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--start", type=int, default=0, help="chunk start index into the worklist.")
    p.add_argument("--count", type=int, default=None, help="chunk size (default: to the end).")
    p.add_argument("--samples", type=str, nargs="+", default=None,
                   help="Explicit sample IDs (smoke); overrides chunking selection within them.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build + log the worklist size, then exit (use to size the SLURM array).")
    args = p.parse_args()
    run(
        ast_sheet=args.ast_sheet, metadata=args.metadata, protein_dir=args.protein_dir,
        amr_ref_dir=args.amr_ref_dir, out_dir=args.out_dir, minimap2_bin=args.minimap2_bin,
        threads=args.threads, workers=args.workers, skip_existing=args.skip_existing,
        start=args.start, count=args.count, samples=args.samples, path_root=args.path_root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
