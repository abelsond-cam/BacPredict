"""Mash/ANI kinship for ``pyseer --lmm --similarity``, built once per organism and subset per drug.

Mash distance is a MinHash estimate of ``1 - ANI`` over a genome's whole k-mer content, so a kinship
built from it captures the population structure the LMM should absorb as a random effect. The
isolation-source GWAS used a core-SNP kinship in production and recorded mash as *"a trade-off, not
a fix"* for that phenotype — the worry being that a whole-genome kinship also absorbs accessory
content, which for an HGT-driven signal is the very thing you are trying to detect. For AMR that
trade-off is live too (plasmid content correlates with both kinship and resistance), which is why
the within-lineage permutation null is a required check, not an optional one.

There is no `mash sketch`/`mash triangle` sbatch anywhere in the repo — the invasion cohort's
triangle was built by hand — so this module owns that step and makes it reproducible.

**The per-drug subset is the point.** A cohort-wide similarity matrix is n², and TB's cohort is
~38k genomes: ~11 GB as float64 in memory and far larger as the TSV pyseer parses. But pyseer only
ever uses the phenotyped samples, and per-drug counts are much smaller (Kp ertapenem ~2.1k,
colistin ~1.4k; TB ethionamide ~10k). So the expensive part — sketching and the triangle — is done
once per organism, and each drug gets a small square matrix cut from it. Only TB rifampin
(~29k train+validate) approaches the cohort-wide size.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bac_pyseer.ast_gwas.build_ast_phenotype import SAMPLE_COL
from bac_pyseer.kleb_iso_source.mash_dist_to_kinship import parse_triangle, to_similarity

logger = logging.getLogger(__name__)

DEFAULT_SKETCH_SIZE = 10_000  # mash -s; larger than mash's default 1000 for finer distances
DEFAULT_KMER = 21  # mash -k


def sketch(
    reflist: Path, out_msh: Path, *, sketch_size: int = DEFAULT_SKETCH_SIZE,
    kmer: int = DEFAULT_KMER, threads: int = 8, mash_bin: str = "mash",
) -> Path:
    """``mash sketch`` every assembly in a ``Sample<TAB>path`` reflist into one sketch file.

    Sample ids come from the file *basename* downstream (that is how
    :func:`~bac_pyseer.kleb_iso_source.mash_dist_to_kinship.parse_triangle` recovers them), which is
    exactly right here because the AST assemblies are named ``<BioSample>.fa.gz``.
    """
    if shutil.which(mash_bin) is None:
        raise SystemExit(f"{mash_bin!r} not on PATH — run under the bac_pyseer pixi env")
    paths = [line.split("\t")[1] for line in reflist.read_text().splitlines() if "\t" in line]
    if not paths:
        raise SystemExit(f"{reflist} has no Sample<TAB>path rows")
    out_msh.parent.mkdir(parents=True, exist_ok=True)
    # mash appends .msh itself, so hand it the stem.
    stem = out_msh.with_suffix("") if out_msh.suffix == ".msh" else out_msh
    listing = out_msh.parent / f"{out_msh.name}.paths.txt"
    listing.write_text("".join(f"{p}\n" for p in paths))
    subprocess.run(
        [mash_bin, "sketch", "-p", str(threads), "-s", str(sketch_size), "-k", str(kmer),
         "-o", str(stem), "-l", str(listing)],
        check=True,
    )
    listing.unlink(missing_ok=True)
    return stem.with_suffix(".msh")


def triangle(msh: Path, out_txt: Path, *, threads: int = 8, mash_bin: str = "mash") -> Path:
    """``mash triangle`` → lower-triangular PHYLIP distances."""
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w") as fh:
        subprocess.run([mash_bin, "triangle", "-p", str(threads), str(msh)], stdout=fh, check=True)
    return out_txt


def similarity_for_samples(
    triangle_path: Path, samples: list[str], *, double_center: bool = False
) -> pd.DataFrame:
    """Cut a cohort-wide mash triangle down to ``samples`` → the square TSV pyseer ``--similarity`` takes.

    Raises
    ------
    SystemExit
        If any requested sample is absent from the triangle — pyseer would otherwise silently drop
        those genomes from the GWAS, changing the cohort without saying so.
    """
    names, d = parse_triangle(triangle_path)
    index_of = {n: i for i, n in enumerate(names)}
    missing = [s for s in samples if s not in index_of]
    if missing:
        raise SystemExit(
            f"{len(missing)} phenotyped sample(s) absent from {triangle_path}, e.g. {missing[:5]} — "
            "re-sketch so the kinship covers every genome the GWAS will test"
        )
    idx = [index_of[s] for s in samples]
    sub = to_similarity(d[np.ix_(idx, idx)], double_center)
    return pd.DataFrame(sub, index=samples, columns=samples)


def distances_for_samples(triangle_path: Path, samples: list[str]) -> pd.DataFrame:
    """The same subset as :func:`similarity_for_samples`, but as raw distances.

    ``pyseer --lineage`` also wants ``--distances`` (it reports per-hit lineage effects off it). The
    iso-source runs passed the Jaccard matrix from the variant pipeline; the AMR cohorts have no
    such matrix, and the mash distances we already computed serve the same purpose.
    """
    names, d = parse_triangle(triangle_path)
    index_of = {n: i for i, n in enumerate(names)}
    missing = [s for s in samples if s not in index_of]
    if missing:
        raise SystemExit(f"{len(missing)} sample(s) absent from {triangle_path}, e.g. {missing[:5]}")
    idx = [index_of[s] for s in samples]
    return pd.DataFrame(d[np.ix_(idx, idx)], index=samples, columns=samples)


def run(
    *, triangle_path: Path, out_tsv: Path, phenotype_tsv: Path | None = None,
    double_center: bool = False, distances_tsv: Path | None = None,
) -> dict[str, object]:
    """Write the similarity matrix for a phenotype's samples (or the whole triangle if none given)."""
    if phenotype_tsv is not None:
        samples = pd.read_csv(phenotype_tsv, sep="\t")[SAMPLE_COL].astype(str).tolist()
    else:
        samples, _ = parse_triangle(triangle_path)
    frame = similarity_for_samples(triangle_path, samples, double_center=double_center)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_tsv, sep="\t")
    if distances_tsv is not None:
        distances_for_samples(triangle_path, samples).to_csv(distances_tsv, sep="\t")

    manifest = {
        "triangle": str(triangle_path),
        "phenotype_tsv": str(phenotype_tsv) if phenotype_tsv else None,
        "output": str(out_tsv),
        "distances_output": str(distances_tsv) if distances_tsv else None,
        "n_samples": len(samples),
        "double_center": double_center,
        "kinship": "mash (whole-genome k-mer, ~1-ANI)",
        "note": "validate with the within-lineage permutation null before trusting the hits",
    }
    out_tsv.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %s: %d x %d similarity matrix", out_tsv, len(samples), len(samples))
    return manifest


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    s = sub.add_parser("sketch", help="mash sketch + triangle over a cohort reflist (once per organism).")
    s.add_argument("--reflist", type=Path, required=True, help="Sample<TAB>assembly_path reflist.")
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--sketch-size", type=int, default=DEFAULT_SKETCH_SIZE)
    s.add_argument("--kmer", type=int, default=DEFAULT_KMER)
    s.add_argument("--threads", type=int, default=8)

    k = sub.add_parser("kinship", help="Cut a triangle down to one phenotype's samples.")
    k.add_argument("--triangle", type=Path, required=True)
    k.add_argument("--out-tsv", type=Path, required=True)
    k.add_argument("--phenotype-tsv", type=Path, default=None,
                   help="Restrict to these samples (the same file passed to pyseer --phenotypes).")
    k.add_argument("--double-center", action="store_true", help="Emit the MDS Gram (PSD) instead of 1-D.")
    k.add_argument("--distances-tsv", type=Path, default=None,
                   help="Also write the raw distance matrix for the same samples — pyseer --lineage "
                        "needs a --distances file, and the AMR cohorts have no Jaccard matrix.")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.stage == "sketch":
        msh = sketch(args.reflist, args.out_dir / "mash_sketch.msh", sketch_size=args.sketch_size,
                     kmer=args.kmer, threads=args.threads)
        tri = triangle(msh, args.out_dir / "mash_triangle.txt", threads=args.threads)
        print(json.dumps({"sketch": str(msh), "triangle": str(tri)}, indent=2))
    else:
        print(json.dumps(run(
            triangle_path=args.triangle, out_tsv=args.out_tsv,
            phenotype_tsv=args.phenotype_tsv, double_center=args.double_center,
            distances_tsv=args.distances_tsv,
        ), indent=2))


if __name__ == "__main__":
    main()
