r"""Turn per-genome attribution scores into per-gene statistics, with the lineage control.

Three questions, in the order they must be asked:

1. **Which genes does the model attend to?** Per genome, attributions are normalised to a z-score of
   their absolute value, so a genome the model is very sure about and one it is unsure about
   contribute on the same scale. A gene's statistic is then the mean of its z across the genomes that
   carry it, tested against 0. Normalising within genome makes 0 the natural null — the background is
   the rest of that genome — which also sidesteps the defect in the published implementation, where
   a gene was pooled into the background it was being tested against. **This ranking works and is
   unchanged**: on the TB rifampicin cohort it put ``rpoB`` first at mean z 38.2 with ``katG`` and
   ``rpoA`` at zero.

2. **Does it attend to that gene *because of resistance*?** Among genomes carrying the gene, compare
   its attribution in resistant versus susceptible genomes. A determinant should separate; a gene
   the model merely always reads should not.

3. **Does that survive lineage?** The same resistant-versus-susceptible test, repeated inside each
   lineage. This is the control that decides what the whole analysis means. Resistance is near-fixed
   in the dominant clonal lineages, so a gene can rank highly simply by marking a lineage in which
   every genome is resistant. A determinant's signal survives stratification; a lineage marker's
   collapses. Nothing here licenses a causal reading without it.

Why the contrast is run on the **signed** score, and the ranking on ``|.|``
--------------------------------------------------------------------------
These are two different statistics and only one of them needed changing.

The ranking (question 1) asks *how much* of the model's output a gene moves, where direction is
irrelevant and ``|.|`` is right. The contrast (questions 2 and 3) asks *which way* it moves the
output, and there taking the absolute value destroys the signal: measured on the TB rifampicin
cohort, wild-type ``rpoB`` carries a large **negative** attribution (median -10.10) and mutant
``rpoB`` a positive one (+3.85), so ``|.|`` ranks the wild-type genomes *higher* and the
resistant-minus-susceptible contrast comes out backwards (delta -11.41, failing the lineage control)
on a gene the signed score separates at p = 2e-12.

The signed score is used **raw, not per-genome normalised**, because the cohort-mean IG baseline
already gives every genome a shared origin: ``f(baseline)`` measured -2.7569, -2.7571, -2.7586 and
-2.7607 across genomes of different size, label and lineage, so every genome's attributions sum to
its logit plus ~2.758 against the same reference and are comparable as they stand. Subtracting a
per-genome mean would remove that comparability rather than create it. The ``|.|`` ranking keeps its
z-score because it has no such shared origin -- a confident genome's attributions are simply larger.

Groups are split on the **AST label**, never on variant carriage. Carriage is known only where a
catalogue exists; the label is what is known for any phenotype, which is the setting this method has
to work in.

**Multi-copy genes are collapsed to one value per genome before any test.** Two annotated copies of
a gene in one genome are not two independent observations, and counting rows made ``n_genomes``
read 190 for ``rpoC`` across 98 genomes. Copies are summed by default: integrated-gradient
attributions are additive, so a gene's contribution to the logit is the sum over its copies.

**Only named genes are pooled.** Where Bakta found no symbol, ``gene_name`` is that genome's own
locus tag, which is meaningless across genomes; those rows are counted and excluded rather than
quietly treated as thousands of singleton genes. The fraction of attribution mass sitting in unnamed
genes is reported, because that is the share only an annotation-free gene family (nuna) can reach.

**The ``|.|`` normalisation is compositional, by construction.** Scaling within a genome means a gene
that takes a large share of the attribution raises that genome's mean and standard deviation, which
pushes every *other* gene's z down. The artifact shrinks as ``1/n_genes``, so across the ~5,390
genes of a real genome it is negligible — but it points the opposite way to the true signal, so on a
small gene set an inert gene can look significant in reverse. Never run these tests over a
hand-picked handful of genes without re-reading this paragraph. The raw signed score is not
compositional and does not carry this caveat.

Guards against the defects in the published implementation: p-values stay floating point end to end
(they were written to CSV at two significant figures and re-read), survival functions are used
instead of ``1 - cdf`` so a strong signal does not underflow to exactly zero, genes below an
occurrence floor are dropped rather than allowed to top the table on n=1, and every gene is excluded
from its own background.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

DEFAULT_MIN_GENOMES = 20
DEFAULT_MIN_PER_CLASS = 5
#: How the copies of a multi-copy gene are combined into one value per genome. ``sum`` is the
#: default because IG attributions are additive, so the sum is the gene's total contribution.
MULTI_COPY_CHOICES = ("sum", "max", "mean")
#: Columns carried through a copy collapse unchanged — one value per (Sample, gene) by definition.
_CARRIED = ("ast_label", "lineage", "is_named_gene", "base_logit", "genome_has_spread")


def normalise_within_genome(frame: pd.DataFrame, score_column: str = "attribution_logit") -> pd.DataFrame:
    """Add the signed score, its absolute value, and the per-genome z of that absolute value.

    A genome whose attributions have zero variance carries no information about which of its genes
    mattered. Those are **flagged and excluded, never zero-filled**: the count is the saturation
    diagnostic that told us bf16 was unusable, and silently turning them into zeros would hide a
    recurrence.

    Parameters
    ----------
    frame : pandas.DataFrame
        Long frame with ``Sample`` and ``score_column``.
    score_column : str, default "attribution_logit"
        Which score to use.

    Returns
    -------
    pandas.DataFrame
        ``frame`` plus ``signed_score``, ``abs_score``, ``z_abs`` and ``genome_has_spread``.
    """
    out = frame.copy()
    out["signed_score"] = out[score_column]
    out["abs_score"] = out[score_column].abs()
    grouped = out.groupby("Sample")["abs_score"]
    mean = grouped.transform("mean")
    sd = grouped.transform("std")
    out["genome_has_spread"] = sd.notna() & (sd > 0)
    out["z_abs"] = np.where(out["genome_has_spread"], (out["abs_score"] - mean) / sd.replace(0, np.nan), np.nan)

    dead = out.loc[~out["genome_has_spread"], "Sample"].nunique()
    if dead:
        logger.warning("%d of %d genomes have zero attribution spread and are excluded from every test",
                       dead, out["Sample"].nunique())
    return out


def collapse_gene_copies(frame: pd.DataFrame, *, how: str = "sum") -> pd.DataFrame:
    """Reduce a gene's annotated copies within one genome to a single value.

    Two copies of a gene in one genome are one observation, not two. Leaving them separate inflates
    every per-gene ``n`` (``rpoC`` read 190 across 98 genomes) and breaks the independence the
    rank tests assume.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`normalise_within_genome`.
    how : {"sum", "max", "mean"}, default "sum"
        ``sum`` is the default because integrated-gradient attributions are additive: the gene's
        contribution to the logit is the sum over its copies. ``max`` treats the strongest copy as
        the gene; ``mean`` treats copies as repeated measurements of one thing.

    Returns
    -------
    pandas.DataFrame
        One row per ``(Sample, gene_name)``, plus ``n_copies``.
    """
    if how not in MULTI_COPY_CHOICES:
        raise ValueError(f"how must be one of {MULTI_COPY_CHOICES}, got {how!r}")
    if not frame.duplicated(["Sample", "gene_name"]).any():
        out = frame.copy()
        out["n_copies"] = 1
        return out

    carried = [c for c in _CARRIED if c in frame.columns]
    agg: dict[str, str] = {"signed_score": how, "abs_score": how, "z_abs": how}
    # max of |.| and max of the signed score are different rows; for "max" keep the signed score of
    # the largest-magnitude copy so sign and magnitude still describe the same protein.
    if how == "max":
        agg = {"abs_score": "max", "z_abs": "max"}
    grouped = frame.groupby(["Sample", "gene_name"], sort=False)
    out = grouped.agg({**agg, **{c: "first" for c in carried}}).reset_index()
    if how == "max":
        peak = frame.loc[frame.groupby(["Sample", "gene_name"])["abs_score"].idxmax(),
                         ["Sample", "gene_name", "signed_score"]]
        out = out.merge(peak, on=["Sample", "gene_name"], how="left")
    out["n_copies"] = grouped.size().reset_index(drop=True)
    collapsed = int((out["n_copies"] > 1).sum())
    logger.info("collapsed %d (genome, gene) pairs with >1 annotated copy using how=%r", collapsed, how)
    return out


def _bh(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, NaN-safe."""
    out = np.full(len(pvalues), np.nan)
    ok = np.isfinite(pvalues)
    if ok.any():
        out[ok] = multipletests(pvalues[ok], method="fdr_bh")[1]
    return out


def rank_genes(frame: pd.DataFrame, *, min_genomes: int = DEFAULT_MIN_GENOMES,
               multi_copy: str = "sum") -> pd.DataFrame:
    """Rank named genes by mean within-genome normalised |attribution|.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`normalise_within_genome`, carrying ``gene_name`` and ``is_named_gene``.
    min_genomes : int, default 20
        Genes seen in fewer **genomes** are dropped. Without a floor, a gene appearing once tops the
        table on a single extreme value — the published implementation ranked a singleton at z=75.8.
    multi_copy : {"sum", "max", "mean"}, default "sum"
        How a gene's copies within one genome are combined; see :func:`collapse_gene_copies`.

    Returns
    -------
    pandas.DataFrame
        One row per gene: ``n_genomes``, ``mean_z``, ``median_z``, ``median_n_copies``,
        ``p_value``, ``p_adj``, sorted by ``mean_z``.
    """
    usable = collapse_gene_copies(frame[frame["is_named_gene"] & frame["genome_has_spread"]],
                                  how=multi_copy)
    rows = []
    for gene, group in usable.groupby("gene_name"):
        valid = group[np.isfinite(group["z_abs"])]
        z = valid["z_abs"].to_numpy()
        if valid["Sample"].nunique() < min_genomes:
            continue
        # One-sided: is this gene attended to more than its genome's average gene? The within-genome
        # normalisation already makes 0 the background, so the gene is not in its own comparison set.
        result = stats.wilcoxon(z, alternative="greater", zero_method="zsplit") if np.any(z != 0) else None
        rows.append({
            "gene_name": gene, "n_genomes": int(valid["Sample"].nunique()), "mean_z": float(z.mean()),
            "median_z": float(np.median(z)),
            "median_n_copies": float(valid["n_copies"].median()),
            "p_value": float(result.pvalue) if result is not None else np.nan,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        logger.warning("No gene met the min_genomes=%d floor", min_genomes)
        return out
    out["p_adj"] = _bh(out["p_value"].to_numpy())
    return out.sort_values("mean_z", ascending=False).reset_index(drop=True)


def resistant_vs_susceptible(
    frame: pd.DataFrame,
    *,
    value_column: str = "signed_score",
    min_per_class: int = DEFAULT_MIN_PER_CLASS,
    stratum: str | None = None,
    multi_copy: str = "sum",
) -> pd.DataFrame:
    """Per gene, compare attribution between resistant and susceptible genomes.

    The split is on ``ast_label`` — the phenotype call — and never on variant carriage, because the
    label is the only thing known for a phenotype with no catalogue, which is where this method has
    to work.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`normalise_within_genome`, carrying ``ast_label``.
    value_column : str, default "signed_score"
        ``signed_score`` (the default, and the one that works — see the module docstring) or
        ``z_abs`` for the magnitude-only comparison.
    min_per_class : int, default 5
        A gene needs at least this many genomes on each side.
    stratum : str, optional
        Column to stratify by (e.g. ``"lineage"``). When given, the test is run **within** each
        value and the result carries a ``stratum`` column. This is the lineage control.
    multi_copy : {"sum", "max", "mean"}, default "sum"
        How a gene's copies within one genome are combined; see :func:`collapse_gene_copies`.

    Returns
    -------
    pandas.DataFrame
        ``gene_name``, optional ``stratum``, ``value_column``, ``n_resistant``, ``n_susceptible``,
        means and medians per class, ``delta`` (resistant minus susceptible, on means),
        ``delta_median``, ``rank_biserial``, ``disjoint``, ``p_value``, ``p_adj``.
    """
    usable = collapse_gene_copies(frame[frame["is_named_gene"] & frame["genome_has_spread"]],
                                  how=multi_copy)
    keys = ["gene_name"] if stratum is None else [stratum, "gene_name"]
    rows = []
    for key, group in usable.groupby(keys):
        res = group.loc[group["ast_label"] == 1, value_column].dropna().to_numpy()
        sus = group.loc[group["ast_label"] == 0, value_column].dropna().to_numpy()
        if len(res) < min_per_class or len(sus) < min_per_class:
            continue
        # Two-sided: a determinant the model reads *less* in resistant genomes is as notable as more.
        u = stats.mannwhitneyu(res, sus, alternative="two-sided")
        row = {
            "gene_name": key[-1] if isinstance(key, tuple) else key,
            "value_column": value_column,
            "n_resistant": len(res), "n_susceptible": len(sus),
            "mean_resistant": float(res.mean()), "mean_susceptible": float(sus.mean()),
            "median_resistant": float(np.median(res)), "median_susceptible": float(np.median(sus)),
            "delta": float(res.mean() - sus.mean()),
            "delta_median": float(np.median(res) - np.median(sus)),
            # Rank-biserial from U: +1 means every resistant genome outranks every susceptible one.
            "rank_biserial": float(2.0 * u.statistic / (len(res) * len(sus)) - 1.0),
            # Completely non-overlapping distributions are worth naming; a p-value alone does not
            # say whether the two groups merely shifted or separated entirely.
            "disjoint": bool(res.min() > sus.max() or sus.min() > res.max()),
            "p_value": float(u.pvalue),
        }
        if stratum is not None:
            row[stratum] = key[0]
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        logger.warning("No gene had >=%d genomes in both classes%s", min_per_class,
                       f" within any {stratum}" if stratum else "")
        return out
    out["p_adj"] = _bh(out["p_value"].to_numpy())
    return out.sort_values("p_value").reset_index(drop=True)


def summarise_lineage_control(overall: pd.DataFrame, within: pd.DataFrame, stratum: str) -> pd.DataFrame:
    """Join the pooled contrast to its within-lineage repeats — the table the conclusion rests on.

    A gene that separates resistant from susceptible overall **and** keeps doing so inside lineages
    is behaving like a determinant. One that separates only overall is consistent with marking a
    lineage in which resistance happens to be fixed, which is not the same claim and must not be
    reported as one.

    Parameters
    ----------
    overall : pandas.DataFrame
        :func:`resistant_vs_susceptible` with ``stratum=None``.
    within : pandas.DataFrame
        :func:`resistant_vs_susceptible` with a stratum.
    stratum : str
        The stratifying column's name.

    Returns
    -------
    pandas.DataFrame
        Per gene: the pooled effect, how many strata it was testable in, in how many it stayed
        significant and same-signed, how many it was merely same-signed in, whether any stratum
        separated the two classes completely, and the median within-stratum effect.
    """
    if overall.empty or within.empty:
        return pd.DataFrame()
    agg = []
    for gene, group in within.groupby("gene_name"):
        pooled = overall.loc[overall["gene_name"] == gene]
        if pooled.empty:
            continue
        pooled_delta = float(pooled["delta"].iloc[0])
        same_sign = np.sign(group["delta"]) == np.sign(pooled_delta)
        significant = (group["p_adj"] < 0.05) & same_sign
        agg.append({
            "gene_name": gene,
            "pooled_delta": pooled_delta,
            "pooled_p_adj": float(pooled["p_adj"].iloc[0]),
            "n_strata_tested": len(group),
            "n_strata_same_sign": int(same_sign.sum()),
            "n_strata_significant": int(significant.sum()),
            "n_strata_disjoint": int((group["disjoint"] & same_sign).sum()) if "disjoint" in group else 0,
            "median_within_delta": float(group["delta"].median()),
            "survives_stratification": bool(significant.any()),
        })
    out = pd.DataFrame(agg)
    return out.sort_values("pooled_delta", ascending=False).reset_index(drop=True) if not out.empty else out


def report_unnamed_share(frame: pd.DataFrame) -> float:
    """Log and return the share of total |attribution| sitting in genes Bakta could not name."""
    total = frame["abs_score"].sum()
    if total <= 0:
        return float("nan")
    share = float(frame.loc[~frame["is_named_gene"], "abs_score"].sum() / total)
    logger.info("%.1f%% of total |attribution| sits in unnamed genes (genome-specific locus tags); "
                "only an annotation-free gene family can pool those", 100 * share)
    return share


def main() -> None:
    """CLI entry point — compute the gene tables and the lineage control from an annotated frame."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotated-parquet", type=Path, required=True,
                        help="Output of engine.explain.annotate_attributions.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where the tables are written.")
    parser.add_argument("--lineage-csv", type=Path, default=None,
                        help="CSV with Sample + a lineage column, for the control.")
    parser.add_argument("--lineage-column", default="lineage", help="Lineage column name.")
    parser.add_argument("--score-column", default="attribution_logit",
                        help="attribution_logit (default) or attribution_prob.")
    parser.add_argument("--min-genomes", type=int, default=DEFAULT_MIN_GENOMES)
    parser.add_argument("--min-per-class", type=int, default=DEFAULT_MIN_PER_CLASS)
    parser.add_argument("--multi-copy", choices=MULTI_COPY_CHOICES, default="sum",
                        help="How a gene's copies within one genome are combined (default: sum).")
    args = parser.parse_args()

    frame = pd.read_parquet(args.annotated_parquet)
    logger.info("Loaded %d rows over %d genomes", len(frame), frame["Sample"].nunique())
    frame = normalise_within_genome(frame, args.score_column)
    report_unnamed_share(frame)

    if args.lineage_csv is not None:
        lineage = pd.read_csv(args.lineage_csv).drop_duplicates("Sample")
        mapping = lineage.set_index("Sample")[args.lineage_column]
        frame["lineage"] = frame["Sample"].map(mapping).fillna("unassigned")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ranked = rank_genes(frame, min_genomes=args.min_genomes, multi_copy=args.multi_copy)
    ranked.to_csv(args.out_dir / "gene_attribution_ranking.csv", index=False)
    logger.info("Top genes by mean normalised |attribution|: %s",
                ", ".join(ranked["gene_name"].head(10)) if not ranked.empty else "none")

    # The contrast of record is on the signed score. The |.| version is written beside it because
    # the difference between the two is itself a finding, and a reader must be able to check it.
    overall = resistant_vs_susceptible(frame, value_column="signed_score",
                                       min_per_class=args.min_per_class, multi_copy=args.multi_copy)
    overall.to_csv(args.out_dir / "gene_resistant_vs_susceptible_signed.csv", index=False)
    magnitude = resistant_vs_susceptible(frame, value_column="z_abs",
                                         min_per_class=args.min_per_class, multi_copy=args.multi_copy)
    magnitude.to_csv(args.out_dir / "gene_resistant_vs_susceptible_abs_z.csv", index=False)

    if "lineage" in frame.columns:
        within = resistant_vs_susceptible(frame, value_column="signed_score",
                                          min_per_class=args.min_per_class, stratum="lineage",
                                          multi_copy=args.multi_copy)
        within.to_csv(args.out_dir / "gene_resistant_vs_susceptible_within_lineage_signed.csv", index=False)
        control = summarise_lineage_control(overall, within, "lineage")
        control.to_csv(args.out_dir / "lineage_control_summary.csv", index=False)
        if not control.empty:
            survived = int(control["survives_stratification"].sum())
            logger.info("Lineage control: %d of %d testable genes keep a same-signed significant "
                        "effect inside at least one lineage", survived, len(control))
    else:
        logger.warning("No lineage column — the control that licenses a causal reading was NOT run")

    logger.info("Wrote tables to %s", args.out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
