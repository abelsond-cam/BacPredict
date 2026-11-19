r"""Join per-protein attribution scores to gene identity and to the drug's catalogue.

The ablation stage knows only that row *i* of a genome moved the logit by some amount. This stage
says *which gene* row *i* is, and whether that gene is a known determinant for the drug — the two
facts every downstream statistic and figure needs.

Gene identity comes from the per-sample ``{Sample}_protein_sequences.parquet`` via
:func:`bacpredict.engine.gene_lr.locate_gene.flatten_proteins`, which walks contigs in ``contig_idx``
order so its flat index is the embedding row. Note what ``gene_name`` actually is: Bakta's gene
symbol **or, where Bakta found none, that genome's own locus tag** (``BOLBDL_00005``). A locus tag is
genome-specific, so unnamed genes cannot be pooled across genomes at all — which is exactly the gap
the nuna gene families are meant to fill, and why the output marks them.

Catalogue annotation differs by organism, so each is a small adapter:

- **Kp** — the AMR sidecar ``{Sample}_amr.parquet`` carries ``flat_index`` directly, verified against
  the alignment when it was written, so the join is exact rather than by name. It brings
  ``amr_gene_family``, ``amr_class`` and crucially ``amr_source`` (**acquired vs chromosomal**), which
  is the mechanism stratification every AMR report here must carry. Rows with ``flat_index == -1``
  are CARD hits where Bakta called no CDS: there is no embedding row to attribute, so they are
  excluded **and counted**, never dropped silently.
- **TB** — ``tbprofiler_variants.parquet`` is joined on gene symbol **per genome**, so the output says
  which genes carry a drug-associated variant *in this genome* rather than which genes are on a
  static list. Two limits are structural: an unnamed or fragmented CDS cannot be joined at all, and a
  non-coding determinant (``rrs``, the ``inhA`` promoter) has no protein row, so this method cannot
  see it in either organism.

Examples
--------
::

    python -m bacpredict.engine.explain.annotate_attributions \
        --organism kp --drug ciprofloxacin \
        --attribution-dir "$B/train_kleb_ast/attribution/ciprofloxacin" \
        --cohort-csv "$B/train_kleb_ast/attribution/attribution_cohort_ciprofloxacin.csv" \
        --out-parquet "$B/train_kleb_ast/attribution/ciprofloxacin_attributions_annotated.parquet"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bacpredict.engine.config import organism as resolve_organism
from bacpredict.engine.config import store_paths
from bacpredict.engine.explain.gene_attribution import delta_probability
from bacpredict.engine.gene_lr.locate_gene import flatten_proteins

logger = logging.getLogger(__name__)

# Variant-effect names that place a change outside the coding sequence. Such a variant cannot be
# visible to a per-protein ablation, because the protein's embedding is wild-type.
NON_CODING_VARIANT_PATTERN = r"upstream|downstream|intergenic|non_coding|intron|5_prime|3_prime"

# Columns every organism's output carries, whatever its catalogue adapter adds.
BASE_COLUMNS = ["Sample", "flat_index", "gene_name", "is_named_gene", "attribution_logit",
                "attribution_prob", "base_logit", "ast_label"]


def gene_names_for_sample(sample: str, parquet_dir: Path, suffix: str) -> pd.DataFrame:
    """Return ``flat_index`` → ``gene_name`` for one sample, in embedding row order.

    Parameters
    ----------
    sample : str
        Sample ID.
    parquet_dir : pathlib.Path
        Directory of per-sample protein parquets.
    suffix : str
        Parquet filename suffix.

    Returns
    -------
    pandas.DataFrame
        Columns ``flat_index`` and ``gene_name``.

    Raises
    ------
    FileNotFoundError
        If the sample has no protein parquet.
    """
    path = parquet_dir / f"{sample}{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"No protein parquet for {sample} at {path}")
    records = flatten_proteins(pd.read_parquet(path))
    return pd.DataFrame({
        "flat_index": np.arange(len(records), dtype=np.int32),
        "gene_name": [r["gene_name"] for r in records],
    })


def looks_like_a_locus_tag(gene_name: pd.Series) -> pd.Series:
    """Flag names that are Bakta locus tags rather than gene symbols.

    Bakta writes ``<PREFIX>_<digits>`` when it has no symbol for a CDS. Such a name is unique to the
    genome that produced it, so pooling on it across genomes is meaningless.

    Parameters
    ----------
    gene_name : pandas.Series
        The ``gene_name`` column.

    Returns
    -------
    pandas.Series
        Boolean, True where the value looks like a locus tag (or is missing).
    """
    return gene_name.isna() | gene_name.astype(str).str.fullmatch(r"[A-Z]{4,}_\d+")


def kp_determinant_sets(drug: str) -> tuple[set[str], set[str]]:
    """Resolve the CARD gene-families and alleles causal for ``drug`` — the Kp app seam.

    The engine is organism-agnostic, so the CARD drug→mechanism map stays in
    :mod:`bacpredict.apps.kleb.card_label` and is pulled in here lazily rather than imported at
    module scope. That keeps the dependency pointing app→engine everywhere that matters, and mirrors
    the ``calls_fn`` seam ``per_gene_lr_from_annotation`` already uses. Inject a different resolver
    through :func:`annotate_cohort` to override it.

    Parameters
    ----------
    drug : str
        Lowercase AST drug name, e.g. ``"ciprofloxacin"``.

    Returns
    -------
    families, alleles : set of str
        Strings that match the sidecar's ``amr_gene_family`` / ``amr_allele`` exactly.
    """
    from bacpredict.apps.kleb.card_label import causal_genes_for_drug

    return causal_genes_for_drug(drug, grain="family"), causal_genes_for_drug(drug, grain="allele")


def annotate_kp(
    frame: pd.DataFrame,
    sample: str,
    task_root: Path,
    drug: str,
    determinant_sets: tuple[set[str], set[str]] | None = None,
) -> pd.DataFrame:
    """Attach CARD/Kleborate sidecar columns to one Kp genome's attributions.

    ``is_drug_determinant`` is **membership in the drug's CARD causal set**, not a text match. The
    sidecar's ``amr_class`` and ``amr_drug_classes`` hold class-level values (``"Flq"``,
    ``"fluoroquinolone antibiotic"``) and never the drug's own name, so asking whether they contain
    ``"ciprofloxacin"`` is False for every genome — including for gyrA, parC and qnr, the very genes
    the ciprofloxacin analysis exists to find.

    Parameters
    ----------
    frame : pandas.DataFrame
        One genome's attributions, carrying ``flat_index``.
    sample : str
        Sample ID.
    task_root : pathlib.Path
        ``<data root>/processed/train_kleb_ast``.
    drug : str
        Drug name.
    determinant_sets : tuple of (set of str, set of str), optional
        ``(families, alleles)`` causal for the drug; resolved via :func:`kp_determinant_sets` when
        omitted.

    Returns
    -------
    pandas.DataFrame
        ``frame`` plus ``amr_allele``, ``amr_gene_family``, ``amr_class``, ``amr_source``,
        ``annotation_status``, ``is_catalogue_determinant`` and ``is_drug_determinant``.
    """
    families, alleles = determinant_sets if determinant_sets is not None else kp_determinant_sets(drug)
    path = task_root / "amr_annotation" / f"{sample}_amr.parquet"
    keep = ["flat_index", "amr_allele", "amr_gene_family", "amr_class", "amr_source"]
    if not path.exists():
        # Fail visibly: without this column a genome that was never annotated is indistinguishable
        # from one that genuinely carries no determinant, which would quietly dilute every statistic.
        logger.warning("No AMR sidecar for %s — marking its rows annotation_status='no_sidecar'", sample)
        calls = pd.DataFrame({c: pd.Series(dtype="object") for c in keep})
        status = "no_sidecar"
    else:
        calls = pd.read_parquet(path)[keep]
        # flat_index == -1 is a CARD hit with no Bakta CDS, so no embedding row exists to attribute.
        n_unmapped = int((calls["flat_index"] < 0).sum())
        if n_unmapped:
            logger.info("%s: %d CARD hit(s) have no CDS (flat_index=-1) and cannot be attributed",
                        sample, n_unmapped)
        calls = calls[calls["flat_index"] >= 0].drop_duplicates(subset="flat_index", keep="first")
        status = "annotated"

    out = frame.merge(calls, on="flat_index", how="left")
    out["annotation_status"] = status
    out["is_catalogue_determinant"] = out["amr_gene_family"].notna()
    out["is_drug_determinant"] = (
        out["amr_gene_family"].isin(families) | out["amr_allele"].isin(alleles)
    )
    return out


def load_tb_variant_genes(task_root: Path, drug: str) -> pd.DataFrame:
    """Return the (Sample, gene_name) pairs carrying a variant TB-Profiler associates with ``drug``.

    Parameters
    ----------
    task_root : pathlib.Path
        ``<data root>/processed/train_tb_ast``.
    drug : str
        Drug name as it appears in the ``drugs`` column (e.g. ``"rifampin"``).

    Returns
    -------
    pandas.DataFrame
        Columns ``Sample``, ``gene_name``, ``variant_changes`` (semicolon-joined), and
        ``is_drug_determinant`` (always True — absence of a row means no variant).

    Raises
    ------
    FileNotFoundError
        If the variants parquet is absent.
    """
    path = task_root / "tbprofiler" / "tbprofiler_variants.parquet"
    if not path.exists():
        raise FileNotFoundError(f"TB-Profiler variants not found at {path}")
    variants = pd.read_parquet(path)
    hit = variants[variants["drugs"].fillna("").str.contains(drug, case=False, regex=False)]

    # A promoter or other non-coding variant changes expression, not the protein. Its gene's protein
    # row carries a wild-type embedding, so marking that row a determinant would claim the model saw
    # something that is not in its input. (For rifampin every call is coding — missense plus inframe
    # indels in rpoB/rpoC — so this filter is a no-op there and matters for drugs like isoniazid,
    # whose inhA promoter variants would otherwise mark wild-type InhA.)
    non_coding = hit["type"].fillna("").str.contains(NON_CODING_VARIANT_PATTERN, case=False, regex=True)
    if int(non_coding.sum()):
        logger.info("TB-Profiler: dropping %d non-coding %s call(s) (%s) — no protein row carries them",
                    int(non_coding.sum()), drug,
                    ", ".join(sorted(hit.loc[non_coding, "type"].dropna().unique())[:5]))
    hit = hit[~non_coding]
    logger.info("TB-Profiler: %d coding variant calls associated with %s across %d genomes",
                len(hit), drug, hit["Sample"].nunique())
    grouped = (
        hit.groupby(["Sample", "gene_name"])["change"]
        .agg(lambda s: ";".join(sorted(set(map(str, s)))))
        .reset_index()
        .rename(columns={"change": "variant_changes"})
    )
    grouped["is_drug_determinant"] = True
    return grouped


def annotate_tb(frame: pd.DataFrame, sample: str, variant_genes: pd.DataFrame) -> pd.DataFrame:
    """Attach TB-Profiler per-genome variant status to one TB genome's attributions."""
    mine = variant_genes.loc[variant_genes["Sample"] == sample, ["gene_name", "variant_changes",
                                                                 "is_drug_determinant"]]
    out = frame.merge(mine, on="gene_name", how="left")
    # `.eq(True)` rather than fillna+astype: the merge leaves NaN for a gene this genome has no
    # variant in, and fillna on an object column is deprecated and downcasts noisily.
    out["is_drug_determinant"] = out["is_drug_determinant"].eq(True)
    out["is_catalogue_determinant"] = out["is_drug_determinant"]
    return out


def _attribution_method(attribution_path: Path) -> str:
    """Read the method a per-genome attribution parquet was written by.

    The writer stamps ``bacpredict_method`` into the parquet's key-value metadata. Files written
    before that stamp existed are ablation, which is what the default has always been.

    Parameters
    ----------
    attribution_path : pathlib.Path
        One ``{Sample}_attribution.parquet``.

    Returns
    -------
    str
        ``"ablation"``, ``"integrated_gradients"`` or ``"input_x_gradient"``.
    """
    import pyarrow.parquet as pq

    meta = pq.read_schema(attribution_path).metadata or {}
    return meta.get(b"bacpredict_method", b"ablation").decode() or "ablation"


def annotate_cohort(
    attribution_dir: Path,
    cohort: pd.DataFrame,
    organism_key: str,
    drug: str,
    *,
    data_root: Path | None = None,
    determinant_sets: tuple[set[str], set[str]] | None = None,
) -> pd.DataFrame:
    """Annotate every genome's attributions and return one long frame.

    Parameters
    ----------
    attribution_dir : pathlib.Path
        Directory of ``{Sample}_attribution.parquet`` files.
    cohort : pandas.DataFrame
        The cohort table (``Sample``, ``ast_label``, …).
    organism_key : str
        ``"tb"`` or ``"kp"``.
    drug : str
        Drug name.
    data_root : pathlib.Path, optional
        Override the resolved data root.
    determinant_sets : tuple of (set of str, set of str), optional
        Kp only — ``(families, alleles)`` causal for the drug, overriding :func:`kp_determinant_sets`.

    Returns
    -------
    pandas.DataFrame
        One row per (genome, protein), carrying gene identity, both attribution scales, the genome's
        label, and the organism's catalogue columns.
    """
    task_root = resolve_organism(organism_key).data_root(data_root)
    paths = store_paths(organism_key)
    labels = dict(zip(cohort["Sample"].astype(str), cohort["ast_label"], strict=False))
    variant_genes = load_tb_variant_genes(task_root, drug) if organism_key == "tb" else None
    # Resolved once, not per genome: reading the vendored CARD table 877 times would dominate runtime.
    kp_sets = None
    if organism_key == "kp":
        kp_sets = determinant_sets if determinant_sets is not None else kp_determinant_sets(drug)
        logger.info("CARD causal set for %s: %d families, %d alleles", drug, len(kp_sets[0]), len(kp_sets[1]))

    parts: list[pd.DataFrame] = []
    missing = 0
    for sample in cohort["Sample"].astype(str):
        attribution_path = attribution_dir / f"{sample}_attribution.parquet"
        if not attribution_path.exists():
            missing += 1
            continue
        frame = pd.read_parquet(attribution_path)
        genes = gene_names_for_sample(sample, Path(paths.parquet_dir), paths.parquet_suffix)
        if len(genes) != len(frame):
            raise ValueError(
                f"{sample}: {len(frame)} attributed proteins but {len(genes)} in the parquet. The "
                "flat index does not line up; refusing to guess which gene is which."
            )
        frame = frame.merge(genes, on="flat_index", how="left")
        frame["is_named_gene"] = ~looks_like_a_locus_tag(frame["gene_name"])
        frame["ast_label"] = labels.get(sample, -1)
        # delta_probability is only meaningful for ablation, where a score IS the counterfactual
        # logit shift from one perturbation. A gradient method returns a *share of the total gap*
        # f(x) - f(baseline), so sigmoid(base) - sigmoid(base - a_i) is not any probability shift.
        # Writing it anyway produced a column that looked authoritative and meant nothing.
        written_method = _attribution_method(attribution_path)
        if written_method == "ablation":
            frame["attribution_prob"] = delta_probability(frame["base_logit"], frame["attribution_logit"])
        else:
            frame["attribution_prob"] = np.nan
        frame["attribution_method"] = written_method

        frame = (annotate_tb(frame, sample, variant_genes) if organism_key == "tb"
                 else annotate_kp(frame, sample, task_root, drug, determinant_sets=kp_sets))
        parts.append(frame)
        if len(parts) % 100 == 0:
            logger.info("annotated %d genomes", len(parts))

    if missing:
        logger.warning("%d of %d cohort genomes have no attribution output yet", missing, len(cohort))
    if not parts:
        raise ValueError(f"No attribution parquets found in {attribution_dir}")

    out = pd.concat(parts, ignore_index=True)
    ordered = BASE_COLUMNS + [c for c in out.columns if c not in BASE_COLUMNS]
    return out[ordered]


def log_summary(frame: pd.DataFrame, drug: str) -> None:
    """Log what was annotated — coverage, catalogue hits, and the unnamed-gene share."""
    n_genomes = frame["Sample"].nunique()
    logger.info("Annotated %d genomes, %d gene rows", n_genomes, len(frame))
    logger.info("  named genes:        %.1f%% (the rest are genome-specific Bakta locus tags)",
                100 * frame["is_named_gene"].mean())
    if "is_catalogue_determinant" in frame:
        logger.info("  catalogue hits:     %d rows across %d genomes",
                    int(frame["is_catalogue_determinant"].sum()),
                    int(frame.loc[frame["is_catalogue_determinant"], "Sample"].nunique()))
    if "is_drug_determinant" in frame:
        n_drug = int(frame["is_drug_determinant"].sum())
        logger.info("  %s determinants:  %d rows across %d genomes", drug, n_drug,
                    int(frame.loc[frame["is_drug_determinant"], "Sample"].nunique()))
    if "amr_source" in frame:
        for source, count in frame["amr_source"].value_counts().items():
            logger.info("  mechanism %-12s %d rows", str(source), int(count))


def main() -> None:
    """CLI entry point — annotate a cohort's attributions and write one long parquet."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--organism", required=True, choices=("tb", "kp"), help="Organism key.")
    parser.add_argument("--drug", required=True, help="Drug name.")
    parser.add_argument("--attribution-dir", type=Path, required=True, help="Dir of per-genome attributions.")
    parser.add_argument("--cohort-csv", type=Path, required=True, help="The pinned cohort table.")
    parser.add_argument("--out-parquet", type=Path, required=True, help="Where to write the annotated frame.")
    parser.add_argument("--data-root", type=Path, default=None, help="Override the data root.")
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort_csv, low_memory=False)
    cohort["Sample"] = cohort["Sample"].astype(str)
    frame = annotate_cohort(args.attribution_dir, cohort, args.organism, args.drug, data_root=args.data_root)
    log_summary(frame, args.drug)

    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out_parquet, index=False)
    logger.info("Wrote %d rows to %s", len(frame), args.out_parquet)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
