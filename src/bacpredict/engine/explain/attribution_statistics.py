r"""Turn per-genome attribution scores into per-gene statistics, with the lineage control.

Three questions, in the order they must be asked:

1. **Which genes does the model attend to?** Per genome, attributions are normalised to a z-score of
   their absolute value, so a genome the model is very sure about and one it is unsure about
   contribute on the same scale. A gene's statistic is then the mean of its z across the genomes that
   carry it, tested against 0. Normalising within genome makes 0 the natural null — the background is
   the rest of that genome — which also sidesteps the defect in the published implementation, where
   a gene was pooled into the background it was being tested against.

2. **Does it attend to that gene *because of resistance*?** Among genomes carrying the gene, compare
   its normalised attribution in resistant versus susceptible genomes. A determinant should separate;
   a gene the model merely always reads should not.

3. **Does that survive lineage?** The same resistant-versus-susceptible test, repeated inside each
   lineage. This is the control that decides what the whole analysis means. Resistance is near-fixed
   in the dominant clonal lineages, so a gene can rank highly simply by marking a lineage in which
   every genome is resistant. A determinant's signal survives stratification; a lineage marker's
   collapses. Nothing here licenses a causal reading without it.

**Only named genes are pooled.** Where Bakta found no symbol, ``gene_name`` is that genome's own
locus tag, which is meaningless across genomes; those rows are counted and excluded rather than
quietly treated as thousands of singleton genes. The fraction of attribution mass sitting in unnamed
genes is reported, because that is the share only an annotation-free gene family (nuna) can reach.

**The normalisation is compositional, by construction.** Scaling within a genome means a gene that
takes a large share of the attribution raises that genome's mean and standard deviation, which
pushes every *other* gene's z down. The artifact shrinks as ``1/n_genes``, so across the ~5,390
genes of a real genome it is negligible — but it points the opposite way to the true signal, so on a
small gene set an inert gene can look significant in reverse. Never run these tests over a
hand-picked handful of genes without re-reading this paragraph.

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


def normalise_within_genome(frame: pd.DataFrame, score_column: str = "attribution_logit") -> pd.DataFrame:
    """Add ``abs_score`` and its per-genome z-score, flagging genomes with no spread.

    A genome whose attributions have zero variance carries no information about which of its genes
    mattered. Those are **flagged and excluded, never zero-filled**: the count is the saturation
    diagnostic that told us bf16 was unusable, and silently turning them into zeros would hide a
    recurrence.

    Parameters
    ----------
    frame : pandas.DataFrame
        Long frame with ``Sample`` and ``score_column``.
    score_column : str, default "attribution_logit"
        Which score to normalise.

    Returns
    -------
    pandas.DataFrame
        ``frame`` plus ``abs_score``, ``z_abs`` and ``genome_has_spread``.
    """
    out = frame.copy()
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


def _bh(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values, NaN-safe."""
    out = np.full(len(pvalues), np.nan)
    ok = np.isfinite(pvalues)
    if ok.any():
        out[ok] = multipletests(pvalues[ok], method="fdr_bh")[1]
    return out


def rank_genes(frame: pd.DataFrame, *, min_genomes: int = DEFAULT_MIN_GENOMES) -> pd.DataFrame:
    """Rank named genes by mean within-genome normalised |attribution| (the primary statistic).

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`normalise_within_genome`, carrying ``gene_name`` and ``is_named_gene``.
    min_genomes : int, default 20
        Genes seen in fewer genomes are dropped. Without a floor, a gene appearing once tops the
        table on a single extreme value — the published implementation ranked a singleton at z=75.8.

    Returns
    -------
    pandas.DataFrame
        One row per gene: ``n_genomes``, ``mean_z``, ``p_value``, ``p_adj``, sorted by ``mean_z``.
    """
    usable = frame[frame["is_named_gene"] & frame["genome_has_spread"]]
    rows = []
    for gene, group in usable.groupby("gene_name"):
        z = group["z_abs"].to_numpy()
        z = z[np.isfinite(z)]
        if len(z) < min_genomes:
            continue
        # One-sided: is this gene attended to more than its genome's average gene? The within-genome
        # normalisation already makes 0 the background, so the gene is not in its own comparison set.
        result = stats.wilcoxon(z, alternative="greater", zero_method="zsplit") if np.any(z != 0) else None
        rows.append({
            "gene_name": gene, "n_genomes": len(z), "mean_z": float(z.mean()),
            "median_z": float(np.median(z)),
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
    min_per_class: int = DEFAULT_MIN_PER_CLASS,
    stratum: str | None = None,
) -> pd.DataFrame:
    """Per gene, compare normalised attribution between resistant and susceptible genomes.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`normalise_within_genome`, carrying ``ast_label``.
    min_per_class : int, default 5
        A gene needs at least this many genomes on each side.
    stratum : str, optional
        Column to stratify by (e.g. ``"lineage"``). When given, the test is run **within** each
        value and the result carries a ``stratum`` column. This is the lineage control.

    Returns
    -------
    pandas.DataFrame
        ``gene_name``, optional ``stratum``, ``n_resistant``, ``n_susceptible``, ``mean_z_resistant``,
        ``mean_z_susceptible``, ``delta_z``, ``rank_biserial``, ``p_value``, ``p_adj``.
    """
    usable = frame[frame["is_named_gene"] & frame["genome_has_spread"]]
    keys = ["gene_name"] if stratum is None else [stratum, "gene_name"]
    rows = []
    for key, group in usable.groupby(keys):
        res = group.loc[group["ast_label"] == 1, "z_abs"].dropna().to_numpy()
        sus = group.loc[group["ast_label"] == 0, "z_abs"].dropna().to_numpy()
        if len(res) < min_per_class or len(sus) < min_per_class:
            continue
        # Two-sided: a determinant the model reads *less* in resistant genomes is as notable as more.
        u = stats.mannwhitneyu(res, sus, alternative="two-sided")
        row = {
            "gene_name": key[-1] if isinstance(key, tuple) else key,
            "n_resistant": len(res), "n_susceptible": len(sus),
            "mean_z_resistant": float(res.mean()), "mean_z_susceptible": float(sus.mean()),
            "delta_z": float(res.mean() - sus.mean()),
            # Rank-biserial from U: +1 means every resistant genome outranks every susceptible one.
            "rank_biserial": float(2.0 * u.statistic / (len(res) * len(sus)) - 1.0),
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
        significant and same-signed, and the median within-stratum effect.
    """
    if overall.empty or within.empty:
        return pd.DataFrame()
    agg = []
    for gene, group in within.groupby("gene_name"):
        pooled = overall.loc[overall["gene_name"] == gene]
        if pooled.empty:
            continue
        pooled_delta = float(pooled["delta_z"].iloc[0])
        same_sign = np.sign(group["delta_z"]) == np.sign(pooled_delta)
        agg.append({
            "gene_name": gene,
            "pooled_delta_z": pooled_delta,
            "pooled_p_adj": float(pooled["p_adj"].iloc[0]),
            "n_strata_tested": len(group),
            "n_strata_significant": int(((group["p_adj"] < 0.05) & same_sign).sum()),
            "median_within_delta_z": float(group["delta_z"].median()),
            "survives_stratification": bool(((group["p_adj"] < 0.05) & same_sign).any()),
        })
    out = pd.DataFrame(agg)
    return out.sort_values("pooled_delta_z", ascending=False).reset_index(drop=True) if not out.empty else out


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
    ranked = rank_genes(frame, min_genomes=args.min_genomes)
    ranked.to_csv(args.out_dir / "gene_attribution_ranking.csv", index=False)
    logger.info("Top genes by mean normalised |attribution|: %s",
                ", ".join(ranked["gene_name"].head(10)) if not ranked.empty else "none")

    overall = resistant_vs_susceptible(frame, min_per_class=args.min_per_class)
    overall.to_csv(args.out_dir / "gene_resistant_vs_susceptible.csv", index=False)

    if "lineage" in frame.columns:
        within = resistant_vs_susceptible(frame, min_per_class=args.min_per_class, stratum="lineage")
        within.to_csv(args.out_dir / "gene_resistant_vs_susceptible_within_lineage.csv", index=False)
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
