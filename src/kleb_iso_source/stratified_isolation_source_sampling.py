#!/usr/bin/env python3
"""
Stratified sampling of bacterial isolates across different isolation sources.

METHODOLOGY
-----------
This script performs stratified random sampling to balance sample sizes between two
isolation source categories (e.g., blood vs faeces, blood vs respiratory, etc.) while
maintaining geographic representation and study type distribution.

Sampling Strategy:
1. INPUT FILTERING
   - Resolve isolation source categories from `--isolation-sources <token1> <token2>`
     using strict word-based matching against `isolation_source_category`
     (case-insensitive; punctuation ignored).
   - Each token must resolve to exactly one unique isolation source category; if a
     token matches zero categories or multiple categories, the script aborts and lists
     the matched categories (and their sample counts).
   - The script prints an `Isolation Source Token Mapping` block early in the log
     for provenance (token -> matched `isolation_source_category`).
   - Filter to host_category == "human"
   - Optional filter to study_setting containing "Hospital"

2. STUDY TYPE STRATIFICATION
   - Split samples into three independent threads based on amr_study field:
     * AMR thread: amr_study contains "amr" or "AMR" (including control samples)
     * Surveillance thread: amr_study contains "surveillance" (but not amr)
     * NA thread: amr_study is missing/null
   - Each thread is stratified independently to preserve study design integrity

3. GEOGRAPHIC STRATIFICATION (within each study thread)
   - Process at country level (country_parsed field)
   - For each country with samples from both isolation sources:
     a. Calculate current ratio (larger_source : smaller_source)
     b. If ratio within acceptable bounds (1/target_ratio to target_ratio):
        - Keep all samples from both sources
     c. If ratio outside bounds:
        - Keep ALL samples from smaller source
        - Randomly sample from larger source to achieve target ratio
        - Sample size = min(target_ratio × smaller_count, larger_count)
   - Countries with only one isolation source are excluded

4. RATIO BOUNDS
   - For target ratio R, acceptable range is [1/R, R]
   - Example: R=2.0 accepts ratios from 0.5 to 2.0
   - This symmetric bound ensures balance regardless of which source is larger
   - `--ratio <= 0` (e.g., `0` or `-1`) disables ratio-based country downsampling
     and keeps all eligible country samples.

5. RANDOMIZATION
   - Random sampling uses fixed seed (random_state=42) for reproducibility
   - Within-country sampling preserves metadata relationships

Output:
- Stratified dataset maintains country representation
- Each country contributes samples from both isolation sources (when available)
- Study type distributions are preserved within geographic strata
- Final ratio approximates target ratio across all countries

Statistical Properties:
- Sampling is stratified by country AND study type
- No weighting applied (uniform geographic representation)
- Smaller isolation source determines maximum sample size per country
- Variance in country contribution reflects natural geographic distribution

Use Cases:
- Compare clinical outcomes between isolation sources
- Build balanced training sets for machine learning
- Control for geographic and study design confounders
- Analyze isolation-source-specific genomic features

"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from kleb_iso_source.isolation_source_cli_parsing import validate_and_resolve_tokens

# Constants
DEFAULT_RATIO = 2.0
TEST_RATIOS = [1.0, 2.0, 2.5, 3.0]
DEFAULT_OUTPUT_FILE = "stratified_selected_isolation_source_metadata.tsv"
_PROCESSED = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed")
DEFAULT_OUTPUT_BASE_DIR = _PROCESSED / "train_on_sr_mags"


def setup_logging(log_file: str = "stratify_isolation_source_sampling.log"):
    """Configure logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
    )


def _slugify_token(token: str) -> str:
    """Convert token to a safe lowercase path fragment."""
    slug = re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_")
    return slug or "unknown"


def _log_category_breakdown(
    df: pd.DataFrame,
    category_col: str,
    header: str,
    source_categories: list[str],
    source_labels: list[str],
    top_n: int = None,
    exclude_vals: list[str] = None,
    include_isolation_breakdown: bool = True,
) -> None:
    """
    Log category counts. Optionally include isolation source sub-breakdown per category.

    Args:
        df: DataFrame with category_col and isolation_source_category
        category_col: Column to break down
        header: Header line for the report
        source_categories: List of the two isolation source categories being compared (matched categories)
        source_labels: List of the two isolation source labels to display in logs (user tokens)
        top_n: If set, show only top N categories by total
        exclude_vals: If set, exclude these values from the breakdown
        include_isolation_breakdown: If False, log only category: total (no per-source lines)
    """
    if category_col not in df.columns:
        logging.info(f"  {header}: column '{category_col}' not found")
        return
    crosstab = pd.crosstab(df[category_col], df["isolation_source_category"])
    if exclude_vals:
        crosstab = crosstab[~crosstab.index.isin(exclude_vals)]
    totals = crosstab.sum(axis=1)
    totals = totals.sort_values(ascending=False)
    if top_n:
        totals = totals.head(top_n)
    logging.info(f"  {header}:")
    for cat_val in totals.index:
        label = "NA" if pd.isna(cat_val) else str(cat_val)
        row = crosstab.loc[cat_val]
        n_source1 = int(row.get(source_categories[0], 0)) if isinstance(row, pd.Series) else 0
        n_source2 = int(row.get(source_categories[1], 0)) if isinstance(row, pd.Series) else 0
        total = n_source1 + n_source2
        if include_isolation_breakdown:
            logging.info(f"    {label}:")
            logging.info(f"      Total: {total:,}")
            logging.info(f"      {source_labels[0]}: {n_source1:,}")
            logging.info(f"      {source_labels[1]}: {n_source2:,}")
        else:
            logging.info(f"    {label}: {total:,}")


def _log_final_country_table(
    df_init: pd.DataFrame,
    df_final: pd.DataFrame,
    source_categories: list[str],
    source_labels: list[str],
) -> None:
    """Log final table by country using user labels.

    Country | Source1 (Init -> Sample) | Source2 (Init -> Sample) | Sampled ratio
    """
    if df_final.empty or "country_parsed" not in df_final.columns:
        return
    if "country_parsed" not in df_init.columns or "isolation_source_category" not in df_init.columns:
        return

    iso_col = "isolation_source_category"
    source1, source2 = source_categories
    source1_label, source2_label = source_labels

    def _counts_by_country(df: pd.DataFrame) -> pd.DataFrame:
        ctab = pd.crosstab(df["country_parsed"], df[iso_col])
        return ctab

    init_ct = _counts_by_country(df_init)
    final_ct = _counts_by_country(df_final)

    all_countries = sorted(set(init_ct.index) | set(final_ct.index), key=lambda x: str(x) if pd.notna(x) else "")
    rows = []
    for country in all_countries:
        if pd.isna(country):
            continue
        init_row = init_ct.loc[country] if country in init_ct.index else pd.Series()
        final_row = final_ct.loc[country] if country in final_ct.index else pd.Series()
        i_s1 = int(init_row.get(source1, 0))
        i_s2 = int(init_row.get(source2, 0))
        s_s1 = int(final_row.get(source1, 0))
        s_s2 = int(final_row.get(source2, 0))

        # Calculate ratio (larger/smaller) for display
        if s_s1 > 0 and s_s2 > 0:
            ratio_val = max(s_s1, s_s2) / min(s_s1, s_s2)
            ratio_str = f"{ratio_val:.2f}"
        else:
            ratio_str = "-"

        rows.append((country, i_s1, s_s1, i_s2, s_s2, ratio_str))

    rows.sort(key=lambda r: r[2] + r[4], reverse=True)

    logging.info(f"\n{'=' * 80}")
    logging.info("FINAL SAMPLES BY COUNTRY")
    logging.info(f"{'=' * 80}")

    # Dynamic header based on isolation source names
    source1_short = source1_label[:20]
    source2_short = source2_label[:20]
    logging.info(
        f"{'Country':<30} "
        f"{source1_short + ' (Init -> Sample)':<30} "
        f"{source2_short + ' (Init -> Sample)':<30} "
        f"{'Sampled ratio':<15}"
    )
    logging.info("-" * 105)
    for country, i_s1, s_s1, i_s2, s_s2, ratio_str in rows:
        source1_str = f"{i_s1:,} -> {s_s1:,}"
        source2_str = f"{i_s2:,} -> {s_s2:,}"
        logging.info(f"{str(country):<30} {source1_str:<30} {source2_str:<30} {ratio_str:<15}")
    logging.info("")


def calculate_ratio_bounds(ratio: float) -> tuple[float, float]:
    """
    Calculate acceptable ratio bounds.

    Args:
        ratio: Target ratio (e.g., 2.0)

    Returns
    -------
        Tuple of (lower_bound, upper_bound) where acceptable ratios are
        between 1/ratio and ratio (e.g., for ratio=2: 0.5 to 2.0)
    """
    if ratio <= 0:
        return (0.0, float("inf"))
    return (1.0 / ratio, ratio)


def load_and_filter_data(
    metadata_file: str,
    source_categories: list[str],
    source_labels: list[str],
    filter_by_study_setting: bool = False,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load metadata and apply initial filters.

    Does NOT filter by amr_study - that split is done in main for dual-thread stratification.

    Args:
        metadata_file: Path to the metadata TSV file
        source_categories: List of two matched isolation source categories to keep
        source_labels: List of two user token labels to display in logs
        filter_by_study_setting: If True, filter to study_setting contains "Hospital"

    Returns
    -------
        Tuple of (filtered_dataframe, filter_counts_dict)
    """
    logging.info(f"Loading metadata from: {metadata_file}")

    df = pd.read_csv(metadata_file, sep="\t", low_memory=False)
    initial_count = len(df)
    logging.info(f"Initial dataset size: {initial_count:,} samples")

    filter_counts = {"initial": initial_count}

    # Filter 1: Isolation source category
    logging.info(f"\nFiltering to isolation sources (tokens): {source_labels}")
    _log_category_breakdown(
        df,
        "isolation_source_category",
        "isolation_source_category (pre-filter, top 6)",
        source_categories,
        source_labels,
        top_n=6,
        include_isolation_breakdown=False,
    )
    df = df[df["isolation_source_category"].isin(source_categories)]
    filter_counts["after_isolation_source"] = len(df)
    logging.info(f"After isolation source filter: {len(df):,} samples")

    # Filter 2: Host category == "human"
    logging.info('\nFiltering to host_category == "human"')
    _log_category_breakdown(
        df,
        "host_category",
        "host_category (pre-filter, top 5 other)",
        source_categories,
        source_labels,
        top_n=5,
        exclude_vals=["human"],
    )
    df = df[df["host_category"] == "human"]
    filter_counts["after_host_category"] = len(df)
    logging.info(f"After host_category filter: {len(df):,} samples")

    # Filter 3: Study setting (conditional)
    logging.info("\nStudy setting:")
    _log_category_breakdown(
        df, "study_setting", "study_setting (pre-filter, all categories)", source_categories, source_labels
    )
    if filter_by_study_setting:
        df = df[df["study_setting"].str.contains("Hospital", case=False, na=False)]
        filter_counts["after_study_setting"] = len(df)
        logging.info(f"After study_setting filter: {len(df):,} samples (filter applied)")
    else:
        filter_counts["after_study_setting"] = len(df)
        logging.info(
            f"Study setting filter: NOT APPLIED (--filter-by-study-setting not set). All {len(df):,} samples retained."
        )

    # AMR study routing (no filter - report only; split done in main)
    logging.info("\namr_study routing (samples assigned to AMR, Surveillance, NA threads):")
    _log_category_breakdown(df, "amr_study", "amr_study (all categories)", source_categories, source_labels)
    mask_amr = df["amr_study"].str.contains("amr", case=False, na=False)
    mask_surveillance = df["amr_study"].str.contains("surveillance", case=False, na=False) & ~mask_amr
    mask_na = df["amr_study"].isna()
    n_amr = mask_amr.sum()
    n_surveillance = mask_surveillance.sum()
    n_na = mask_na.sum()
    n_unassigned = len(df) - n_amr - n_surveillance - n_na
    logging.info(f"  AMR thread (amr/AMR plus control): {n_amr:,} samples")
    logging.info(f"  Surveillance thread: {n_surveillance:,} samples")
    logging.info(f"  NA thread: {n_na:,} samples")
    if n_unassigned > 0:
        logging.info(f"  Unassigned (excluded from stratification): {n_unassigned:,} samples")

    # Summary of isolation sources in retained df
    logging.info("\nBreakdown by isolation source (after pre-amr filters):")
    source_counts = df["isolation_source_category"].value_counts()
    for label, category in zip(source_labels, source_categories, strict=True):
        logging.info(f"  {label}: {int(source_counts.get(category, 0)):,}")

    return df, filter_counts


def sample_to_ratio(
    df: pd.DataFrame, ratio: float, isolation_sources: list[str], isolation_col: str = "isolation_source_category"
) -> pd.DataFrame:
    """Keep all samples from the smaller group and sample from the larger group.

    Target: larger_sampled = ratio * smaller_count (capped at available).

    E.g. 67 source1, 436 source2 with ratio=3: keep all 67 source1, sample 3*67=201 source2.

    Args:
        df: DataFrame subset to sample
        ratio: Target ratio (larger_group : smaller_group)
        isolation_sources: List of two isolation source categories
        isolation_col: Column name for isolation source

    Returns
    -------
        Sampled DataFrame
    """
    source1, source2 = isolation_sources
    counts = df[isolation_col].value_counts()
    n_source1 = counts.get(source1, 0)
    n_source2 = counts.get(source2, 0)

    if n_source1 == 0 or n_source2 == 0:
        return df

    # Keep ALL of the smaller group; sample min(ratio * smaller, larger) from the larger
    if n_source1 < n_source2:
        n_smaller, n_larger = n_source1, n_source2
        smaller_val, larger_val = source1, source2
    else:
        n_smaller, n_larger = n_source2, n_source1
        smaller_val, larger_val = source2, source1

    target_larger = min(int(ratio * n_smaller), n_larger)
    smaller_df = df[df[isolation_col] == smaller_val]
    larger_df = df[df[isolation_col] == larger_val].sample(n=target_larger, random_state=42)
    return pd.concat([smaller_df, larger_df])


def stratify_by_country(
    df: pd.DataFrame,
    ratio: float,
    isolation_sources: list[str],
    thread_label: str = "",
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply country-level stratification.

    All countries with both isolation sources are processed; none are deferred to region.

    Args:
        df: Input dataframe
        ratio: Target ratio
        isolation_sources: List of two isolation source categories
        thread_label: Optional label for log context (e.g. "AMR", "Surveillance")

    Returns
    -------
        Tuple of (stratified_dataframe, sampling_log)
    """
    source1, source2 = isolation_sources
    lower_bound, upper_bound = calculate_ratio_bounds(ratio)
    ratio_sampling_disabled = ratio <= 0

    stratified_samples = []
    sampling_log = []

    title = f"COUNTRY-LEVEL STRATIFICATION{f' [{thread_label}]' if thread_label else ''}"
    logging.info(f"\n{'=' * 80}")
    logging.info(title)
    logging.info(f"{'=' * 80}")
    if ratio_sampling_disabled:
        logging.info(f"Target ratio: {ratio} (ratio-based country sampling disabled; keeping all eligible samples)")
    else:
        logging.info(f"Target ratio: {ratio} (acceptable range: {lower_bound:.2f} to {upper_bound:.2f})")
    logging.info("")

    for country in df["country_parsed"].unique():
        if pd.isna(country):
            continue

        country_df = df[df["country_parsed"] == country]

        counts = country_df["isolation_source_category"].value_counts()
        n_source1 = counts.get(source1, 0)
        n_source2 = counts.get(source2, 0)

        if ratio_sampling_disabled:
            # In no-ratio mode, keep all countries, including one-sided countries.
            stratified_samples.append(country_df)
            action = "accepted_all_no_ratio"
            sampled_source1 = n_source1
            sampled_source2 = n_source2
            if n_source1 > 0 and n_source2 > 0:
                current_ratio = max(n_source1, n_source2) / min(n_source1, n_source2)
            elif n_source1 > 0 or n_source2 > 0:
                current_ratio = float("inf")
            else:
                current_ratio = 0.0
        else:
            if n_source1 == 0 or n_source2 == 0:
                continue

            # Calculate ratio as larger/smaller for comparison
            current_ratio = max(n_source1, n_source2) / min(n_source1, n_source2)

            if lower_bound <= current_ratio <= upper_bound:
                stratified_samples.append(country_df)
                action = "accepted_all"
                sampled_source1 = n_source1
                sampled_source2 = n_source2
            else:
                sampled_df = sample_to_ratio(country_df, ratio, isolation_sources)
                stratified_samples.append(sampled_df)
                action = "country_sampled"
                sampled_counts = sampled_df["isolation_source_category"].value_counts()
                sampled_source1 = sampled_counts.get(source1, 0)
                sampled_source2 = sampled_counts.get(source2, 0)

        sampling_log.append(
            {
                "location": country,
                "location_type": "country",
                "initial_source1": n_source1,
                "sampled_source1": sampled_source1,
                "initial_source2": n_source2,
                "sampled_source2": sampled_source2,
                "initial_ratio": current_ratio,
                "action": action,
            }
        )

    stratified_df = pd.concat(stratified_samples, ignore_index=True) if stratified_samples else pd.DataFrame()

    logging.info(f"Countries processed: {len(sampling_log)}")
    logging.info(f"Samples stratified at country level: {len(stratified_df):,}")

    return stratified_df, sampling_log


def stratify_by_location(
    df: pd.DataFrame,
    ratio: float,
    isolation_sources: list[str],
    source_labels: list[str],
    thread_label: str = "",
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Main stratification function - country-level only.

    Args:
        df: Input dataframe
        ratio: Target ratio
        isolation_sources: List of two isolation source categories (matched categories)
        source_labels: List of two source labels to display in logs (user tokens)
        thread_label: Optional label for log context (e.g. "AMR", "Surveillance")

    Returns
    -------
        Tuple of (final_stratified_dataframe, sampling_log)
    """
    final_df, complete_log = stratify_by_country(df, ratio, isolation_sources, thread_label=thread_label)

    source1, source2 = isolation_sources
    summary_title = f"FINAL STRATIFICATION SUMMARY{f' [{thread_label}]' if thread_label else ''}"
    logging.info(f"\n{'=' * 80}")
    logging.info(summary_title)
    logging.info(f"{'=' * 80}")
    logging.info(f"Total samples after stratification: {len(final_df):,}")

    # Final counts by isolation source
    if not final_df.empty:
        final_counts = final_df["isolation_source_category"].value_counts()
        logging.info("\nFinal breakdown by isolation source:")
        for label, category in zip(source_labels, isolation_sources, strict=True):
            logging.info(f"  {label}: {int(final_counts.get(category, 0)):,}")

        n_source1 = final_counts.get(source1, 0)
        n_source2 = final_counts.get(source2, 0)
        if n_source1 > 0 and n_source2 > 0:
            final_ratio = max(n_source1, n_source2) / min(n_source1, n_source2)
            logging.info(f"\nFinal ratio (larger:smaller): {final_ratio:.2f}")

    return final_df, complete_log


def create_detailed_report(
    sampling_log: list[dict],
    ratio: float,
    isolation_sources: list[str],
    thread_label: str = "",
    source_labels: list[str] = None,
):
    """
    Create detailed country breakdown report.

    Args:
        sampling_log: List of sampling decisions
        ratio: The ratio used
        isolation_sources: List of two isolation source categories
        thread_label: Optional label for log context (e.g. "AMR", "Surveillance")
    """
    if source_labels is None:
        source_labels = isolation_sources
    source1_label, source2_label = source_labels

    title = f"DETAILED BREAKDOWN FOR RATIO = {ratio}{f' [{thread_label}]' if thread_label else ''}"
    logging.info(f"\n{'=' * 80}")
    logging.info(title)
    logging.info(f"{'=' * 80}")
    logging.info("")

    # Sort by total initial samples (descending)
    sorted_log = sorted(sampling_log, key=lambda x: x["initial_source1"] + x["initial_source2"], reverse=True)

    # Header - shorten source names if needed
    source1_short = source1_label[:15]
    source2_short = source2_label[:15]
    logging.info(f"{'Location':<30} {'Type':<10} {source1_short:<20} {source2_short:<20} {'Ratio':<10} {'Action':<20}")
    logging.info(f"{'':^30} {'':^10} {'Init → Sample':<20} {'Init → Sample':<20} {'(L/S)':<10} {'':<20}")
    logging.info("-" * 125)

    # Detail rows
    for entry in sorted_log:
        location = entry["location"][:28]  # Truncate if too long
        loc_type = entry["location_type"]

        source1_str = f"{entry['initial_source1']:,} → {entry['sampled_source1']:,}"
        source2_str = f"{entry['initial_source2']:,} → {entry['sampled_source2']:,}"
        ratio_str = f"{entry['initial_ratio']:.2f}"
        action = entry["action"].replace("_", " ").title()

        logging.info(f"{location:<30} {loc_type:<10} {source1_str:<20} {source2_str:<20} {ratio_str:<10} {action:<20}")

    logging.info("")


def test_multiple_ratios(
    df: pd.DataFrame,
    test_ratios: list[float],
    isolation_sources: list[str],
    source_labels: list[str],
    filter_counts: dict[str, int],
):
    """
    Test stratification with multiple ratios.

    Args:
        df: Filtered dataframe
        test_ratios: List of ratios to test
        isolation_sources: List of two isolation source categories
        filter_counts: Dictionary of filter stage counts
    """
    logging.info(f"\n{'=' * 80}")
    logging.info("TESTING MULTIPLE RATIOS")
    logging.info(f"{'=' * 80}")
    logging.info("")

    results = {}

    for ratio in test_ratios:
        logging.info(f"\n{'-' * 80}")
        logging.info(f"RATIO: {ratio}")
        logging.info(f"{'-' * 80}")

        stratified_df, sampling_log = stratify_by_location(df.copy(), ratio, isolation_sources, source_labels)

        # Store results
        results[ratio] = {
            "stratified_df": stratified_df,
            "sampling_log": sampling_log,
            "final_count": len(stratified_df),
        }

        # Create detailed report for default ratio
        if ratio == DEFAULT_RATIO:
            create_detailed_report(sampling_log, ratio, isolation_sources, source_labels=source_labels)

    # Summary table across all ratios
    source1_cat, source2_cat = isolation_sources
    source1_label, source2_label = source_labels
    logging.info(f"\n{'=' * 80}")
    logging.info("SUMMARY ACROSS ALL RATIOS")
    logging.info(f"{'=' * 80}")
    logging.info("")

    source1_short = source1_label[:15]
    source2_short = source2_label[:15]
    logging.info(f"{'Ratio':<10} {'Final Count':<15} {source1_short:<20} {source2_short:<20} {'Ratio':<15}")
    logging.info("-" * 80)

    for ratio in test_ratios:
        stratified_df = results[ratio]["stratified_df"]
        if not stratified_df.empty:
            counts = stratified_df["isolation_source_category"].value_counts()
            n_source1 = counts.get(source1_cat, 0)
            n_source2 = counts.get(source2_cat, 0)
            if n_source1 > 0 and n_source2 > 0:
                final_ratio = max(n_source1, n_source2) / min(n_source1, n_source2)
            else:
                final_ratio = 0

            logging.info(
                f"{ratio:<10} {len(stratified_df):<15,} {n_source1:<20,} {n_source2:<20,} {final_ratio:<15.2f}"
            )

    logging.info("")

    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Stratified sampling of bacterial isolates across different isolation sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Blood vs faeces/rectal swabs
  %(prog)s --isolation-sources blood faeces --metadata-file data.tsv \
    --output-file train_blood_vs_faeces/stratified_selected_isolation_source_metadata.tsv

  # Blood vs respiratory
  %(prog)s --isolation-sources blood respiratory --metadata-file data.tsv \
    --output-file train_blood_vs_respiratory/stratified_selected_isolation_source_metadata.tsv

  # Blood vs urine
  %(prog)s --isolation-sources blood urine --metadata-file data.tsv --ratio 2.5
        """,
    )
    parser.add_argument(
        "--isolation-sources",
        nargs=2,
        required=True,
        metavar=("TOKEN1", "TOKEN2"),
        help='Two isolation source tokens to compare (e.g., "blood" "faeces"). '
        "Tokens are matched as words within isolation_source_category field. "
        "Each token must match exactly one category.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO,
        help=(
            f'Target ratio as a single number (larger:smaller), e.g. 4 for ~4:1; not "4:1". '
            f"Use 0 or -1 to disable ratio-based country sampling (default: {DEFAULT_RATIO})"
        ),
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/final/metadata_v2_all_samples_and_columns.tsv",
        help="Path to metadata TSV file",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="stratify_isolation_source_sampling.log",
        help="Log file name (saved in the same directory as --output-file)",
    )
    parser.add_argument(
        "--filter-by-study-setting",
        action="store_true",
        help='Filter to study_setting contains "Hospital" (default: no filter, all samples retained)',
    )
    parser.add_argument(
        "--test-all-ratios", action="store_true", help="Test all predefined ratios instead of just the specified one"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Path to output TSV file. If omitted, writes to "
            f"train_on_sr_mags/training_<token1>_<token2>/{DEFAULT_OUTPUT_FILE}"
        ),
    )

    args = parser.parse_args()

    token1, token2 = args.isolation_sources
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        base = DEFAULT_OUTPUT_BASE_DIR
        output_dir = base / f"training_{_slugify_token(token1)}_{_slugify_token(token2)}"
        output_path = output_dir / DEFAULT_OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.parent / Path(args.log_file).name

    # Setup logging
    setup_logging(str(log_path))

    logging.info("=" * 80)
    logging.info("STRATIFIED ISOLATION SOURCE SAMPLING")
    logging.info("=" * 80)
    logging.info("")

    try:
        # Load initial data to resolve tokens
        logging.info("Loading metadata to resolve isolation source tokens...")
        df_raw = pd.read_csv(args.metadata_file, sep="\t", low_memory=False)

        # Validate and resolve tokens
        logging.info(f"Resolving tokens: '{token1}' and '{token2}'")
        try:
            category1, category2 = validate_and_resolve_tokens(df_raw, token1, token2)
            source_labels = [token1, token2]
            isolation_sources = [category1, category2]

            logging.info("Isolation Source Token Mapping:")
            logging.info(f"  Token '{token1}' → matched category '{category1}'")
            logging.info(f"  Token '{token2}' → matched category '{category2}'")
            logging.info("")
        except ValueError as e:
            logging.error(f"\nERROR: {e}")
            return

        # Now load and filter with resolved categories
        df, filter_counts = load_and_filter_data(
            args.metadata_file,
            isolation_sources,
            source_labels,
            filter_by_study_setting=args.filter_by_study_setting,
        )

        if df.empty:
            logging.error("No data remaining after filtering!")
            return

        # Split by amr_study: AMR, Surveillance, NA (three separate threads)
        mask_amr = df["amr_study"].str.contains("amr", case=False, na=False)
        mask_surveillance = df["amr_study"].str.contains("surveillance", case=False, na=False) & ~mask_amr
        mask_na = df["amr_study"].isna()
        df_amr = df[mask_amr].copy()
        df_surveillance = df[mask_surveillance].copy()
        df_na = df[mask_na].copy()

        logging.info(f"\n{'=' * 80}")
        logging.info("THREE-THREAD STRATIFICATION (AMR + Surveillance + NA)")
        logging.info(f"{'=' * 80}")
        logging.info(f"AMR thread (amr/AMR plus control): {len(df_amr):,} samples")
        logging.info(f"Surveillance thread: {len(df_surveillance):,} samples")
        logging.info(f"NA thread: {len(df_na):,} samples")
        logging.info("")

        # Test multiple ratios or single ratio
        if args.test_all_ratios:
            results = test_multiple_ratios(
                df_amr,
                TEST_RATIOS,
                isolation_sources,
                source_labels,
                filter_counts,
            )
            final_df = results[DEFAULT_RATIO]["stratified_df"]
            # test_multiple_ratios does not support multi-thread; use AMR only for --test-all-ratios
            if not df_surveillance.empty or not df_na.empty:
                logging.info(
                    "Note: --test-all-ratios runs on AMR thread only. Surveillance and NA samples not included."
                )
        else:
            logging.info(f"\nUsing single ratio: {args.ratio}")

            stratified_amr = pd.DataFrame()
            log_amr = []
            stratified_surveillance = pd.DataFrame()
            log_surveillance = []

            if not df_amr.empty:
                stratified_amr, log_amr = stratify_by_location(
                    df_amr,
                    args.ratio,
                    isolation_sources,
                    source_labels,
                    thread_label="AMR",
                )
                create_detailed_report(
                    log_amr,
                    args.ratio,
                    isolation_sources,
                    "AMR",
                    source_labels=source_labels,
                )

            if not df_surveillance.empty:
                stratified_surveillance, log_surveillance = stratify_by_location(
                    df_surveillance,
                    args.ratio,
                    isolation_sources,
                    source_labels,
                    thread_label="Surveillance",
                )
                create_detailed_report(
                    log_surveillance,
                    args.ratio,
                    isolation_sources,
                    "Surveillance",
                    source_labels=source_labels,
                )

            stratified_na = pd.DataFrame()
            if not df_na.empty:
                stratified_na, log_na = stratify_by_location(
                    df_na,
                    args.ratio,
                    isolation_sources,
                    source_labels,
                    thread_label="NA",
                )
                create_detailed_report(
                    log_na,
                    args.ratio,
                    isolation_sources,
                    "NA",
                    source_labels=source_labels,
                )

            # Combine
            parts = [p for p in [stratified_amr, stratified_surveillance, stratified_na] if not p.empty]
            final_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

            # Combined summary
            logging.info(f"\n{'=' * 80}")
            logging.info("COMBINED STRATIFICATION SUMMARY")
            logging.info(f"{'=' * 80}")
            if not stratified_amr.empty:
                amr_counts = stratified_amr["isolation_source_category"].value_counts()
                n_source1 = amr_counts.get(isolation_sources[0], 0)
                n_source2 = amr_counts.get(isolation_sources[1], 0)
                logging.info(
                    f"AMR thread: {len(stratified_amr):,} samples "
                    f"({source_labels[0]}: {n_source1:,}, {source_labels[1]}: {n_source2:,})"
                )
            if not stratified_surveillance.empty:
                surv_counts = stratified_surveillance["isolation_source_category"].value_counts()
                n_source1 = surv_counts.get(isolation_sources[0], 0)
                n_source2 = surv_counts.get(isolation_sources[1], 0)
                logging.info(
                    f"Surveillance thread: {len(stratified_surveillance):,} samples "
                    f"({source_labels[0]}: {n_source1:,}, {source_labels[1]}: {n_source2:,})"
                )
            if not stratified_na.empty:
                na_counts = stratified_na["isolation_source_category"].value_counts()
                n_source1 = na_counts.get(isolation_sources[0], 0)
                n_source2 = na_counts.get(isolation_sources[1], 0)
                logging.info(
                    f"NA thread: {len(stratified_na):,} samples "
                    f"({source_labels[0]}: {n_source1:,}, {source_labels[1]}: {n_source2:,})"
                )
            if not final_df.empty:
                final_counts = final_df["isolation_source_category"].value_counts()
                n_source1 = final_counts.get(isolation_sources[0], 0)
                n_source2 = final_counts.get(isolation_sources[1], 0)
                logging.info(
                    f"Combined total: {len(final_df):,} samples "
                    f"({source_labels[0]}: {n_source1:,}, {source_labels[1]}: {n_source2:,})"
                )

            # Final table by country
            df_init_parts = [p for p in [df_amr, df_surveillance, df_na] if not p.empty]
            df_init = pd.concat(df_init_parts, ignore_index=True) if df_init_parts else pd.DataFrame()
            _log_final_country_table(df_init, final_df, isolation_sources, source_labels)

        # Save TSV output (always writes if there are samples)
        if not final_df.empty:
            final_df.to_csv(output_path, index=False, sep="\t")
            logging.info(f"\nStratified samples saved to: {output_path}")
            logging.info(f"Log file saved to: {log_path}")
            logging.info("\nSaved outputs:")
            logging.info(f"  Metadata TSV: {output_path}")
            logging.info(f"  Log file: {log_path}")
        else:
            logging.info("\nNo output file saved because no samples remained after stratification.")
            logging.info(f"Log file saved to: {log_path}")

        logging.info("\n" + "=" * 80)
        logging.info("STRATIFICATION COMPLETE")
        logging.info("=" * 80)

    except Exception as e:
        logging.error(f"\nError during stratification: {e}")
        import traceback

        logging.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
