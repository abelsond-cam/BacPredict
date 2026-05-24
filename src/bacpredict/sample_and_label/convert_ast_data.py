import pathlib
import re
import numpy as np
import pandas as pd

BASE_DIR = pathlib.Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david")
RAW_SUB = "raw"
PROCESSED_SUB = "processed"
VIS_SUB = "results_visualisations"
FILE_LEAF = "klebsiella_ebi_amr_records_20260216.csv"

DEFAULT_INPUT = BASE_DIR / RAW_SUB / FILE_LEAF
DEFAULT_OUTPUT_DIR = BASE_DIR / PROCESSED_SUB
DEFAULT_VIZ_DIR = BASE_DIR / VIS_SUB


def read_ast_data(mic_data_path=None):
    """Read AST data from file path.
    
    Parameters
    ----------
    mic_data_path : Path or str, optional
        Path to the AST data CSV file. If None, uses default path.
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing AST data
    """
    if mic_data_path is None:
        mic_data_path = DEFAULT_INPUT
    return pd.read_csv(mic_data_path)


def parse_number_section(number_section):
    """Parse the 'number' part of a MIC string (e.g. '32', '.25', '64/4').
    - Takes anything up to '/' if present and discards the rest.
    - If the number starts with '.', prepends '0' for clarity.
    Returns float or None on invalid input.
    """
    if not isinstance(number_section, str) or not number_section.strip():
        return None
    # 1. Take only the part before '/' if present
    num_str = number_section.split("/")[0].strip()
    if not num_str:
        return None
    # 2. Leading dot: treat as 0.xxx
    if num_str.startswith(".") and (len(num_str) == 1 or num_str[1].isdigit() or num_str[1] == "."):
        num_str = "0" + num_str
    try:
        return float(num_str)
    except ValueError:
        return None

# Regex for the number section only (used to validate before parsing).
#   ^           = start of string
#   \d*         = zero or more digits
#   \.?         = optional literal dot (\. is dot, ? means 0 or 1)
#   \d+         = one or more digits (at least one digit somewhere)
#   (?:/\d+.*)? = optional non-capturing group (?: ... )? ; \/ = slash ; \d+ = digits ; .* = rest
#   $           = end of string
_number_section_pattern = re.compile(r"^\d*\.?\d+(?:/\d+.*)?$")

VALID_OPERATORS = {">", "<", ">=", "<=", "=", "=="}
BRACKETED_OPERATOR = ">,<"  # "MIC strictly between X and Y", e.g. ">,< 1,2 mg/L"

def convert_ebi_mic_data(ast_data, ebi_mic_column="phenotype-gen_measurement"):
    """Convert EBI MIC data to log-scale MIC values with adjustments for inequality operators.

    Parses MIC by splitting on space: operator, then number section, then optional units.
    Creates log-transformed MIC with adjustments for censored data:
    - For "<" values: log_mic is reduced by 1 (equivalent to dividing MIC by 10)
    - For ">" values: log_mic is increased by 1 (equivalent to multiplying MIC by 10)
    - For other operators (=, ==, <=, >=): log_mic uses the exact value
    - For ">,< X,Y" (bracketed MIC, e.g. ">,< 1,2 mg/L" meaning 1 < MIC < 2):
      MIC_meaning is set to "~", MIC_value is the geometric mean sqrt(X*Y), and
      log_mic = log10(sqrt(X*Y)) with no censoring adjustment (this is a point
      estimate on the log scale, not a censored bound).

    Parameters
    ----------
    ast_data : pandas.DataFrame
        DataFrame containing AST (Antimicrobial Susceptibility Testing) data
    ebi_mic_column : str, default="phenotype-gen_measurement"
        Column name containing MIC values in format "operator number_section units"
        (space-separated; number_section can be e.g. "32", ".25", "64/4", or
        "X,Y" when the operator is ">,<").

    Returns
    -------
    tuple of (pandas.DataFrame, list of dict)
        - DataFrame with added columns MIC_meaning, MIC_value, log_mic (NA where unparsable)
        - List of unparsable results: [{"index": idx, "raw": str, "reason": str}, ...]
    """
    nan_mask = ast_data[ebi_mic_column].isna()
    n_nan = nan_mask.sum()
    if n_nan > 0:
        print(f"Warning: Found {n_nan} NaN values in {ebi_mic_column}, these will be excluded from parsing")

    ast_data["MIC_meaning"] = pd.NA
    ast_data["MIC_value"] = pd.NA
    ast_data["log_mic"] = pd.NA

    valid_data = ast_data[~nan_mask].copy()
    if len(valid_data) == 0:
        print("Warning: No valid (non-NaN) values to parse")
        return ast_data, []

    unparsable = []
    mic_meaning = []
    mic_value = []
    log_mic = []

    for idx, row in valid_data.iterrows():
        raw = row[ebi_mic_column]
        parts = raw.strip().split()
        if len(parts) < 2:
            unparsable.append({"index": idx, "raw": raw, "reason": "fewer than 2 space-separated parts"})
            mic_meaning.append(pd.NA)
            mic_value.append(np.nan)
            log_mic.append(np.nan)
            continue
        operator, number_section = parts[0], parts[1]
        if operator == BRACKETED_OPERATOR:
            # Bracketed MIC: ">,< X,Y unit" means X < MIC < Y. Use geometric mean
            # (log midpoint) as the point estimate; no censoring adjustment.
            bounds = number_section.split(",")
            if len(bounds) != 2:
                unparsable.append({"index": idx, "raw": raw, "reason": f"bracketed MIC needs 2 bounds, got {number_section!r}"})
                mic_meaning.append(pd.NA)
                mic_value.append(np.nan)
                log_mic.append(np.nan)
                continue
            try:
                lower, upper = float(bounds[0]), float(bounds[1])
            except ValueError:
                unparsable.append({"index": idx, "raw": raw, "reason": f"bracketed MIC bounds not numeric: {number_section!r}"})
                mic_meaning.append(pd.NA)
                mic_value.append(np.nan)
                log_mic.append(np.nan)
                continue
            if lower <= 0 or upper <= 0:
                unparsable.append({"index": idx, "raw": raw, "reason": f"bracketed MIC bounds must be positive: {number_section!r}"})
                mic_meaning.append(pd.NA)
                mic_value.append(np.nan)
                log_mic.append(np.nan)
                continue
            geo_mean = (lower * upper) ** 0.5
            mic_meaning.append("~")
            mic_value.append(geo_mean)
            log_mic.append(np.log10(geo_mean))
            continue
        if operator not in VALID_OPERATORS:
            unparsable.append({"index": idx, "raw": raw, "reason": f"invalid operator {operator!r}"})
            mic_meaning.append(pd.NA)
            mic_value.append(np.nan)
            log_mic.append(np.nan)
            continue
        if not _number_section_pattern.match(number_section):
            unparsable.append({"index": idx, "raw": raw, "reason": f"number section does not match expected pattern: {number_section!r}"})
            mic_meaning.append(pd.NA)
            mic_value.append(np.nan)
            log_mic.append(np.nan)
            continue
        value = parse_number_section(number_section)
        if value is None:
            unparsable.append({"index": idx, "raw": raw, "reason": f"parse_number_section failed for {number_section!r}"})
            mic_meaning.append(pd.NA)
            mic_value.append(np.nan)
            log_mic.append(np.nan)
            continue
        mic_meaning.append(operator)
        mic_value.append(value)
        log_mic.append(np.log10(value))

    valid_data["MIC_meaning"] = mic_meaning
    valid_data["MIC_value"] = mic_value
    valid_data["log_mic"] = np.array(log_mic, dtype=float)
    valid_data.loc[valid_data["MIC_meaning"] == "<", "log_mic"] -= 1
    valid_data.loc[valid_data["MIC_meaning"] == ">", "log_mic"] += 1

    ast_data.loc[~nan_mask, "MIC_meaning"] = valid_data["MIC_meaning"]
    ast_data.loc[~nan_mask, "MIC_value"] = valid_data["MIC_value"]
    ast_data.loc[~nan_mask, "log_mic"] = valid_data["log_mic"]

    n_ok = (~valid_data["MIC_value"].isna()).sum()
    print(f"\nSuccessfully parsed {n_ok} MIC values")
    if unparsable:
        print(f"Unparsable: {len(unparsable)} rows (returned in second value for inspection)")
    print("Operator distribution:")
    print(ast_data["MIC_meaning"].value_counts())

    return ast_data, unparsable


def antibiogram_by_species(resistance_df, figsize=(14, 8), title=None):
    """
    Create an antibiogram showing antibiotic resistance percentages by species.

    Parameters
    ----------
    resistance_df : pandas.DataFrame
        DataFrame with resistance statistics from tabulate_antibiotic_resistance_per_species
    figsize : tuple, optional
        Figure size (width, height), by default (14, 8)
    title : str, optional
        Custom title for the plot, by default None

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Check if DataFrame is empty
    if len(resistance_df) == 0:
        print("No data to plot")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get unique species and antibiotics
    species_list = resistance_df["Species"].unique()
    antibiotics_list = resistance_df["Antibiotic"].unique()

    # Set up plot parameters
    n_species = len(species_list)
    n_antibiotics = len(antibiotics_list)
    bar_width = 0.8 / n_antibiotics

    # Set up colors for different antibiotics
    colors = sns.color_palette("husl", n_antibiotics)

    # First add all antibiotics to legend with full opacity
    for i, antibiotic in enumerate(antibiotics_list):
        # Add an invisible bar for each antibiotic with full opacity for the legend
        ax.bar(0, 0, width=0, color=colors[i], label=antibiotic)

    # Plot bars for each antibiotic
    for i, antibiotic in enumerate(antibiotics_list):
        antibiotic_data = resistance_df[resistance_df["Antibiotic"] == antibiotic]

        x_positions = []
        resistance_values = []
        alpha_values = []  # Store transparency values based on Percent_Tested

        for j, species in enumerate(species_list):
            species_by_antibiotic_data = antibiotic_data[antibiotic_data["Species"] == species]

            if len(species_by_antibiotic_data) > 0:
                x_positions.append(j + i * bar_width - (n_antibiotics - 1) * bar_width / 2)
                resistance_values.append(species_by_antibiotic_data["Percent_Resistance"].values[0])
                # Check if we have enough samples (both total and tested) to display data
                total_samples = species_by_antibiotic_data["Total_Genomes"].values[0]
                unique_tested = species_by_antibiotic_data["Total_Tests"].values[0]

                if total_samples < 100 or unique_tested < 100:
                    # Not enough samples, set alpha to 0 (don't display)
                    alpha = 0
                else:
                    # Calculate alpha (transparency) based on Percent_Tested (0-100%)
                    alpha = min(1.0, species_by_antibiotic_data["Percent_Tested"].values[0] / 100)

                alpha_values.append(alpha)
            else:
                x_positions.append(j + i * bar_width - (n_antibiotics - 1) * bar_width / 2)
                resistance_values.append(0)
                alpha_values.append(0)  # Don't plot bars for species with less than 100 samples

        # Plot bars for this antibiotic with varying transparency
        for k, (x, y, alpha) in enumerate(zip(x_positions, resistance_values, alpha_values, strict=True)):
            # Always plot a bar, but use a small value (0.5) for zero values to make them visible as a tiny stub
            plot_height = max(0.5, y) if alpha > 0 else 0  # Only plot if alpha > 0 (enough samples)

            if alpha > 0:  # Only plot if we have enough samples
                ax.bar(
                    x,
                    plot_height,
                    width=bar_width,
                    color=colors[i],
                    alpha=alpha,
                    label="",  # No label here since we added them separately above
                    edgecolor="black",
                    linewidth=0.5,
                )

                # Data labels removed to reduce clutter

    # Set x-axis ticks and labels with sample size
    ax.set_xticks(range(n_species))

    # Create species labels with sample size
    species_labels = []
    for species in species_list:
        # Get the Total_BioSamples for this species (same for all antibiotics)
        species_data = resistance_df[resistance_df["Species"] == species]
        if len(species_data) > 0:
            total_samples = int(species_data["Total_Genomes"].iloc[0])
            species_labels.append(f"{species}\n(n={total_samples:,})")
        else:
            species_labels.append(species)

    ax.set_xticklabels(species_labels, rotation=45, ha="right")

    # Set y-axis limits and label
    ax.set_ylim(0, 100)
    ax.set_ylabel("Resistance Percentage (%)")

    # Add title
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Antibiogram: Resistance Percentage by Species")

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=True))
    ax.legend(by_label.values(), by_label.keys(), title="Antibiotics", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add grid lines for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Add notes about transparency and filtering
    plt.figtext(
        0.5,
        0.01,
        "Notes: 1) Bar transparency indicates percentage of isolates tested (100% tested = solid)\n"
        "2) Data with fewer than 100 total samples or 100 tested samples are not displayed",
        ha="center",
        fontsize=8,
        style="italic",
    )

    plt.tight_layout(rect=(0, 0.03, 1, 0.98))  # Adjust layout to make room for the note
    plt.show()

    return fig


def convert_resistance_to_binary(ast_data, resistance_column="phenotype-resistance_phenotype"):
    """Convert resistance phenotype to binary values.
    
    Parameters
    ----------
    ast_data : pandas.DataFrame
        DataFrame containing AST data
    resistance_column : str, default="phenotype-resistance_phenotype"
        Column name containing resistance phenotype values
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with added binary_resistance column (1=resistant, 0=susceptible, NaN=intermediate)
    """
    print(f"\n=== Converting Resistance Phenotype to Binary ===")
    print(f"Distribution of {resistance_column}:")
    print(ast_data[resistance_column].value_counts(dropna=False))
    
    # Create binary resistance column
    ast_data['binary_resistance'] = pd.NA
    ast_data.loc[ast_data[resistance_column].str.lower() == 'resistant', 'binary_resistance'] = 1
    ast_data.loc[ast_data[resistance_column].str.lower() == 'susceptible', 'binary_resistance'] = 0
    # intermediate stays as NaN
    
    print(f"\nBinary resistance distribution:")
    print(f"  Resistant (1): {(ast_data['binary_resistance'] == 1).sum()}")
    print(f"  Susceptible (0): {(ast_data['binary_resistance'] == 0).sum()}")
    print(f"  Intermediate/NaN: {ast_data['binary_resistance'].isna().sum()}")
    
    return ast_data


def filter_antibiotics_by_count(ast_data, antibiotic_column="phenotype-antibiotic_name", min_count=1000, top_n=None):
    """Filter antibiotics by minimum measurement count, or to the top-N most-tested.

    Parameters
    ----------
    ast_data : pandas.DataFrame
        DataFrame containing AST data
    antibiotic_column : str, default="phenotype-antibiotic_name"
        Column name containing antibiotic names
    min_count : int, default=1000
        Minimum number of measurements required to keep an antibiotic.
        Ignored when ``top_n`` is supplied.
    top_n : int, optional
        If provided, keep only the top-N most-tested antibiotics. Overrides ``min_count``.

    Returns
    -------
    tuple
        (filtered_dataframe, list_of_kept_antibiotics)
    """
    if top_n is not None:
        print(f"\n=== Filtering Antibiotics: top {top_n} most-tested ===")
    else:
        print(f"\n=== Filtering Antibiotics by Count (min={min_count}) ===")

    antibiotic_counts = ast_data[antibiotic_column].value_counts()

    print(f"Total unique antibiotics: {len(antibiotic_counts)}")
    print(f"Antibiotic counts range: {antibiotic_counts.min()} to {antibiotic_counts.max()}")

    if top_n is not None:
        kept_series = antibiotic_counts.head(top_n)
        kept_antibiotics = kept_series.index.tolist()
        dropped_with_count = antibiotic_counts.iloc[top_n:]
        min_kept = int(kept_series.min()) if len(kept_series) > 0 else 0
        print(f"Antibiotics kept (top {top_n}): {len(kept_antibiotics)} (min count in kept set: {min_kept:,})")
        print(f"Antibiotics dropped: {len(dropped_with_count)}")
    else:
        kept_antibiotics = antibiotic_counts[antibiotic_counts >= min_count].index.tolist()
        dropped_with_count = antibiotic_counts[antibiotic_counts < min_count]
        print(f"Antibiotics kept (>= {min_count}): {len(kept_antibiotics)}")
        print(f"Antibiotics dropped (< {min_count}): {len(dropped_with_count)}")

    # Print all dropped antibiotics with count > 500, as "antibiotic (n=...)"
    dropped_gt_500 = dropped_with_count[dropped_with_count > 500]
    if len(dropped_gt_500) > 0:
        print(f"\nDropped antibiotics with n > 500:")
        for ab, count in dropped_gt_500.items():
            print(f"  {ab} (n={count:,})")

    # Filter dataframe
    filtered_data = ast_data[ast_data[antibiotic_column].isin(kept_antibiotics)].copy()

    print(f"\nRows before filter: {len(ast_data)}")
    print(f"Rows after filter: {len(filtered_data)}")

    return filtered_data, kept_antibiotics


def compute_antibiotic_testing_stats(ast_data, antibiotics, resistance_column="phenotype-resistance_phenotype"):
    """Compute per-antibiotic resistance/susceptibility statistics.

    Parameters
    ----------
    ast_data : pandas.DataFrame
        Full AST dataframe (after any filtering).
    antibiotics : list of str
        Antibiotic names to include.
    resistance_column : str, default="phenotype-resistance_phenotype"
        Column containing categorical resistance labels.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per antibiotic and columns:
        - antibiotic
        - n_total
        - n_resistant
        - n_susceptible
        - n_intermediate
        - n_nan
        - resistance_pct
        - susceptible_pct
    """
    stats = []
    for antibiotic in antibiotics:
        ab_data = ast_data[ast_data["phenotype-antibiotic_name"] == antibiotic]
        total_tests = len(ab_data)

        labels = ab_data[resistance_column].astype("string").str.lower()
        n_resistant = (labels == "resistant").sum()
        n_susceptible = (labels == "susceptible").sum()
        n_intermediate = (labels == "intermediate").sum()

        n_labelled = n_resistant + n_susceptible + n_intermediate
        n_nan = total_tests - n_labelled

        if n_labelled > 0:
            resistance_pct = (n_resistant / n_labelled) * 100
            susceptible_pct = (n_susceptible / n_labelled) * 100
        else:
            resistance_pct = 0.0
            susceptible_pct = 0.0

        stats.append(
            {
                "antibiotic": antibiotic,
                "n_total": total_tests,
                "n_resistant": n_resistant,
                "n_susceptible": n_susceptible,
                "n_intermediate": n_intermediate,
                "n_nan": n_nan,
                "resistance_pct": resistance_pct,
                "susceptible_pct": susceptible_pct,
            }
        )

    return pd.DataFrame(stats)


def create_metadata_table(ast_data):
    """Create metadata table with one row per sample.
    
    Parameters
    ----------
    ast_data : pandas.DataFrame
        DataFrame containing AST data
        
    Returns
    -------
    pandas.DataFrame
        Metadata table with one row per sample_accession
    """
    print(f"\n=== Creating Metadata Table ===")
    
    metadata_columns = {
        "sample_accession": "phenotype-BioSample_ID",
        "sample_name": "phenotype-assembly_ID",
        "sra_accession": "phenotype-SRA_accession",
        "collection_year": "phenotype-collection_year",
        "host": "phenotype-host",
        "species": "phenotype-species",
        "isolation_source": "phenotype-isolation_source",
        "isolation_source_category": "phenotype-isolation_source_category",
        "isolation_latitude": "phenotype-isolation_latitude",
        "isolation_longitude": "phenotype-isolation_longitude",
        "country": "phenotype-country",
        "region": "phenotype-geographical_region",
        "subregion": "phenotype-geographical_subregion",
    }
    
    # Select relevant columns and rename
    metadata_df = ast_data[list(metadata_columns.values())].copy()
    metadata_df.columns = list(metadata_columns.keys())
    
    # Group by sample_accession and take first value
    metadata_df = metadata_df.groupby('sample_accession').first().reset_index()
    
    print(f"Total unique samples: {len(metadata_df)}")
    print(f"Metadata columns: {', '.join(metadata_df.columns)}")
    
    return metadata_df


def create_antibiotic_pivot_tables(ast_data, value_column, index_col="phenotype-BioSample_ID", 
                                   antibiotic_col="phenotype-antibiotic_name"):
    """Create pivot table with samples as rows and antibiotics as columns.
    
    Parameters
    ----------
    ast_data : pandas.DataFrame
        DataFrame containing AST data
    value_column : str
        Column name to use as values (e.g., 'binary_resistance' or 'log_mic')
    index_col : str, default="phenotype-BioSample_ID"
        Column to use as index (sample identifier)
    antibiotic_col : str, default="phenotype-antibiotic_name"
        Column to use as columns (antibiotic names)
        
    Returns
    -------
    pandas.DataFrame
        Pivot table with samples as rows, antibiotics as columns
    """
    print(f"\n=== Creating Pivot Table for {value_column} ===")
    
    # Create pivot table, taking mean if there are multiple measurements
    pivot_df = ast_data.pivot_table(
        index=index_col,
        columns=antibiotic_col,
        values=value_column,
        aggfunc='mean'
    )
    
    print(f"Pivot table shape: {pivot_df.shape[0]} samples x {pivot_df.shape[1]} antibiotics")
    
    # Calculate sparsity
    total_cells = pivot_df.shape[0] * pivot_df.shape[1]
    nan_cells = pivot_df.isna().sum().sum()
    sparsity = (nan_cells / total_cells) * 100
    
    print(f"Sparsity: {sparsity:.1f}% NaN values")
    print(f"Non-NaN values: {total_cells - nan_cells:,} / {total_cells:,}")
    
    return pivot_df


def create_klebsiella_antibiogram(antibiotic_stats_df, output_path, figsize=(14, 8), min_samples=2000):
    """Create antibiogram visualization showing resistance rates per antibiotic.

    Bars are coloured by total sample count (linear plasma, fixed 0..45,000
    range) and annotated above each column with ``R=<resistant count>``.
    Total tests is shown via the colour bar, not duplicated in the
    annotation. Antibiotics with fewer than ``min_samples`` tests are
    dropped from the figure.

    Parameters
    ----------
    antibiotic_stats_df : pandas.DataFrame
        Per-antibiotic statistics from compute_antibiotic_testing_stats
    output_path : Path or str
        Path to save the output figure
    figsize : tuple, default=(14, 8)
        Figure size (width, height)
    min_samples : int, default=2000
        Minimum number of tests required to display an antibiotic.

    Returns
    -------
    matplotlib.figure.Figure or None
        The generated figure object, or None if no antibiotics meet the threshold
    """
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    print(f"\n=== Creating Antibiogram (min_samples={min_samples}) ===")

    # Filter to antibiotics with at least min_samples tests
    stats_df = antibiotic_stats_df.copy()
    stats_df = stats_df[stats_df["n_total"] >= min_samples].sort_values("resistance_pct", ascending=False)

    if stats_df.empty:
        print(f"No antibiotics with at least {min_samples} samples; skipping antibiogram.")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Colour bars by total sample count (linear scale, fixed 0..45000 range).
    # Plasma gives more visual differentiation at the low end than viridis.
    x_pos = list(range(len(stats_df)))
    n_total = stats_df["n_total"].to_numpy()
    n_resistant = stats_df["n_resistant"].to_numpy()
    cmap = plt.get_cmap("plasma")
    norm = mcolors.Normalize(vmin=0, vmax=45000)
    colors = [cmap(norm(n)) for n in n_total]
    bars = ax.bar(x_pos, stats_df["resistance_pct"], color=colors, edgecolor="black", linewidth=0.5)

    # Annotate each bar with R=<resistant count>. Total tests is shown via the colour bar.
    for bar, n_r in zip(bars, n_resistant, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"R={n_r:,}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # Set labels and title
    ax.set_xlabel("Antibiotic", fontsize=12, fontweight="bold")
    ax.set_ylabel("Resistance Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_title("Antibiogram - Resistance Rates by Antibiotic", fontsize=14, fontweight="bold")

    # X-axis labels: antibiotic names only
    labels = [row["antibiotic"] for _, row in stats_df.iterrows()]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

    # Y range: 0..105 to leave headroom for the R= label
    ax.set_ylim(0, 105)

    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Add colour legend for sample count
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Total tests", fontsize=10)

    # Add note explaining denominators
    note_text = (
        f"Note: only antibiotics with >= {min_samples:,} tests shown. "
        "Bar colour = total tests (linear, 0-45,000); R = resistant count."
    )
    plt.figtext(0.5, 0.02, note_text, ha="center", fontsize=9, style="italic")

    plt.tight_layout(rect=(0, 0.04, 1, 0.98))

    # Save figure
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Antibiogram saved to: {output_path}")

    plt.show()

    return fig


def print_antibiotic_stats_table(antibiotic_stats_df, title_suffix=""):
    """Print the per-antibiotic statistics table (same format as pipeline report)."""
    if title_suffix:
        print(f"\n--- ANTIBIOTIC TESTING STATISTICS {title_suffix} ---")
    print(f"Total antibiotics included: {len(antibiotic_stats_df)}")
    print("\nPer-antibiotic statistics:")
    print(
        f"{'Antibiotic':<25} {'Total Tests':<12} {'Resistant':<11} "
        f"{'Susceptible':<11} {'Intermediate':<13} {'NaN/Other':<11} {'Resistance %':<15}"
    )
    print("-" * 115)
    for row in antibiotic_stats_df.sort_values("antibiotic").itertuples(index=False):
        print(
            f"{row.antibiotic:<25} "
            f"{row.n_total:<12,} "
            f"{row.n_resistant:<11,} "
            f"{row.n_susceptible:<11,} "
            f"{row.n_intermediate:<13,} "
            f"{row.n_nan:<11,} "
            f"{row.resistance_pct:<15.1f}"
        )


def process_klebsiella_ast_data(
    input_file=None,
    min_antibiotic_count=1000,
    reporting_size=None,
    top_n=None,
    output_dir=None,
    viz_dir=None,
    antibiogram_min_samples=2000,
):
    """Main processing function for EBI AST data (organism-agnostic).

    This function performs the complete pipeline:
    1. Load AST data
    2. Convert resistance phenotype to binary
    3. Convert MIC values to log scale
    4. Filter antibiotics (by min count, or to top-N most-tested if ``top_n`` is set)
    5. Create metadata table
    6. Create pivot tables (binary and log MIC)
    7. Print comprehensive statistics
    8. Save outputs
    9. Generate antibiogram visualization
    If reporting_size is set, repeats stats table and antibiogram for antibiotics with count >= reporting_size.

    Parameters
    ----------
    input_file : Path or str, optional
        Path to input CSV file. If None, uses ``DEFAULT_INPUT``.
    min_antibiotic_count : int, default=1000
        Minimum number of measurements required to keep an antibiotic.
        Ignored when ``top_n`` is supplied.
    reporting_size : int, optional
        If set, repeat the antibiotic stats table and antibiogram using this as the
        minimum test count; antibiogram is saved as antibiogram_n_{reporting_size}.png
    top_n : int, optional
        If set, keep only the top-N most-tested antibiotics (overrides ``min_antibiotic_count``).
    output_dir : Path or str, optional
        Directory for CSV outputs (metadata, binary_ast, regression_log_mic).
        Defaults to ``DEFAULT_OUTPUT_DIR``.
    viz_dir : Path or str, optional
        Directory for antibiogram PNG outputs. Defaults to ``DEFAULT_VIZ_DIR``.
    antibiogram_min_samples : int, default=2000
        Minimum tests for an antibiotic to be displayed in the antibiogram.
        Does not affect the saved pivot tables or stats CSV — only the figure.

    Returns
    -------
    dict
        Dictionary containing all output dataframes and paths
    """
    output_dir = pathlib.Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    viz_dir = pathlib.Path(viz_dir) if viz_dir is not None else DEFAULT_VIZ_DIR
    print("=" * 80)
    print("KLEBSIELLA AST DATA PROCESSING PIPELINE")
    print("=" * 80)
    
    # Step 1: Load data
    print("\n[STEP 1] Loading AST data...")
    ast_data = read_ast_data(input_file)
    print(f"Loaded {len(ast_data)} rows")
    
    # Step 2: Convert resistance to binary
    print("\n[STEP 2] Converting resistance phenotype to binary...")
    ast_data = convert_resistance_to_binary(ast_data)

    # Step 3: Convert MIC values to log scale (once)
    print("\n[STEP 3] Converting MIC values to log scale...")
    ast_data, unparsable = convert_ebi_mic_data(ast_data)
    if unparsable:
        print(f"\nUnparsable MIC rows ({len(unparsable)}):")
        for u in unparsable:
            print(f"  index={u['index']!r} raw={u['raw']!r} reason={u['reason']}")
    
    # Step 4: Filter antibiotics
    print("\n[STEP 4] Filtering antibiotics...")
    ast_data, kept_antibiotics = filter_antibiotics_by_count(
        ast_data,
        min_count=min_antibiotic_count,
        top_n=top_n,
    )
    
    # Step 5: Create metadata table
    print("\n[STEP 5] Creating metadata table...")
    metadata_df = create_metadata_table(ast_data)
    
    # Step 6: Create pivot tables
    print("\n[STEP 6] Creating pivot tables...")
    binary_ast_df = create_antibiotic_pivot_tables(ast_data, 'binary_resistance')
    regression_log_mic_df = create_antibiotic_pivot_tables(ast_data, 'log_mic')
    
    # Step 7: Print comprehensive statistics
    print("\n" + "=" * 80)
    print("COMPREHENSIVE STATISTICS")
    print("=" * 80)
    
    # Sample Statistics
    print("\n--- SAMPLE STATISTICS ---")
    print(f"Total unique samples: {len(metadata_df)}")
    if 'collection_year' in metadata_df.columns:
        valid_years = metadata_df['collection_year'].dropna()
        if len(valid_years) > 0:
            print(f"Collection year range: {valid_years.min():.0f} - {valid_years.max():.0f}")
            print(f"Samples with collection year: {len(valid_years)} ({len(valid_years)/len(metadata_df)*100:.1f}%)")
    
    # Host Information
    print("\n--- HOST INFORMATION ---")
    host_counts = metadata_df['host'].value_counts()
    print("Top 10 hosts:")
    for i, (host, count) in enumerate(host_counts.head(10).items(), 1):
        pct = (count / len(metadata_df)) * 100
        print(f"  {i:2d}. {host}: {count:,} ({pct:.1f}%)")
    nan_hosts = metadata_df['host'].isna().sum()
    print(f"NaN host values: {nan_hosts} ({nan_hosts/len(metadata_df)*100:.1f}%)")
    
    # Species Distribution
    if 'species' in metadata_df.columns:
        print("\n--- SPECIES DISTRIBUTION ---")
        species_counts = metadata_df['species'].value_counts()
        print("Species:")
        for i, (species, count) in enumerate(species_counts.items(), 1):
            pct = (count / len(metadata_df)) * 100
            print(f"  {i:2d}. {species}: {count:,} ({pct:.1f}%)")
        nan_species = metadata_df['species'].isna().sum()
        print(f"NaN species values: {nan_species} ({nan_species/len(metadata_df)*100:.1f}%)")

    # Isolation Source Category
    print("\n--- ISOLATION SOURCE CATEGORY ---")
    iso_source_cat = metadata_df['isolation_source_category'].value_counts()
    print("Top 10 isolation source categories:")
    for i, (cat, count) in enumerate(iso_source_cat.head(10).items(), 1):
        pct = (count / len(metadata_df)) * 100
        print(f"  {i:2d}. {cat}: {count:,} ({pct:.1f}%)")
    nan_iso = metadata_df['isolation_source_category'].isna().sum()
    print(f"NaN values: {nan_iso} ({nan_iso/len(metadata_df)*100:.1f}%)")
    
    # Country Distribution
    print("\n--- COUNTRY DISTRIBUTION ---")
    country_counts = metadata_df['country'].value_counts()
    print("Top 10 countries:")
    for i, (country, count) in enumerate(country_counts.head(10).items(), 1):
        pct = (count / len(metadata_df)) * 100
        print(f"  {i:2d}. {country}: {count:,} ({pct:.1f}%)")
    nan_countries = metadata_df['country'].isna().sum()
    print(f"NaN country values: {nan_countries} ({nan_countries/len(metadata_df)*100:.1f}%)")
    
    # Geographic Regions
    print("\n--- GEOGRAPHIC REGIONS ---")
    region_counts = metadata_df['region'].value_counts()
    print("Top 10 regions:")
    for i, (region, count) in enumerate(region_counts.head(10).items(), 1):
        pct = (count / len(metadata_df)) * 100
        print(f"  {i:2d}. {region}: {count:,} ({pct:.1f}%)")
    nan_regions = metadata_df['region'].isna().sum()
    print(f"NaN region values: {nan_regions} ({nan_regions/len(metadata_df)*100:.1f}%)")
    
    print("\nTop 10 subregions:")
    subregion_counts = metadata_df['subregion'].value_counts()
    for i, (subregion, count) in enumerate(subregion_counts.head(10).items(), 1):
        pct = (count / len(metadata_df)) * 100
        print(f"  {i:2d}. {subregion}: {count:,} ({pct:.1f}%)")
    nan_subregions = metadata_df['subregion'].isna().sum()
    print(f"NaN subregion values: {nan_subregions} ({nan_subregions/len(metadata_df)*100:.1f}%)")
    
    # Antibiotic Testing Statistics
    print("\n--- ANTIBIOTIC TESTING STATISTICS ---")
    antibiotic_stats_df = compute_antibiotic_testing_stats(ast_data, kept_antibiotics)
    print_antibiotic_stats_table(antibiotic_stats_df)

    # Pivot Table Summaries
    print("\n--- PIVOT TABLE SUMMARIES ---")
    print(f"\nBinary AST Table:")
    print(f"  Dimensions: {binary_ast_df.shape[0]:,} samples × {binary_ast_df.shape[1]} antibiotics")
    total_cells = binary_ast_df.shape[0] * binary_ast_df.shape[1]
    nan_cells = binary_ast_df.isna().sum().sum()
    print(f"  Sparsity: {(nan_cells/total_cells)*100:.1f}% NaN")
    print(f"  Non-NaN values: {total_cells - nan_cells:,} / {total_cells:,}")
    
    print(f"\nRegression log_mic Table:")
    print(f"  Dimensions: {regression_log_mic_df.shape[0]:,} samples × {regression_log_mic_df.shape[1]} antibiotics")
    total_cells_reg = regression_log_mic_df.shape[0] * regression_log_mic_df.shape[1]
    nan_cells_reg = regression_log_mic_df.isna().sum().sum()
    print(f"  Sparsity: {(nan_cells_reg/total_cells_reg)*100:.1f}% NaN")
    print(f"  Non-NaN values: {total_cells_reg - nan_cells_reg:,} / {total_cells_reg:,}")
    
    # Step 8: Save outputs
    print("\n" + "=" * 80)
    print("[STEP 8] Saving output files...")
    print("=" * 80)
    
    # Ensure directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Save metadata
    metadata_path = output_dir / "ebi_parsed_ast_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"✓ Metadata saved: {metadata_path}")

    # Save binary AST
    binary_path = output_dir / "binary_ast.csv"
    binary_ast_df.to_csv(binary_path)
    print(f"✓ Binary AST saved: {binary_path}")

    # Save regression log_mic
    regression_path = output_dir / "regression_log_mic.csv"
    regression_log_mic_df.to_csv(regression_path)
    print(f"✓ Regression log_mic saved: {regression_path}")

    # Save per-antibiotic stats (n_total, n_resistant, n_susceptible, n_intermediate,
    # n_nan, resistance_pct, susceptible_pct) for the kept antibiotic set.
    stats_path = output_dir / "antibiotic_testing_stats.csv"
    antibiotic_stats_df.to_csv(stats_path, index=False)
    print(f"✓ Antibiotic stats saved: {stats_path}")

    # Step 9: Generate antibiogram (filtered to antibiogram_min_samples)
    print(f"\n[STEP 9] Generating antibiogram (min_samples={antibiogram_min_samples})...")
    antibiogram_path = viz_dir / "antibiogram.png"
    fig = create_klebsiella_antibiogram(
        antibiotic_stats_df, antibiogram_path, min_samples=antibiogram_min_samples
    )

    paths = {
        'metadata': metadata_path,
        'binary_ast': binary_path,
        'regression_log_mic': regression_path,
        'antibiotic_stats': stats_path,
        'antibiogram': antibiogram_path,
    }

    # Optional: repeat stats table and antibiogram with reporting_size threshold
    if reporting_size is not None:
        print("\n" + "=" * 80)
        print(f"REPORTING SUBSET (n >= {reporting_size})")
        print("=" * 80)
        ast_data_reporting, kept_reporting = filter_antibiotics_by_count(
            ast_data, min_count=reporting_size
        )
        antibiotic_stats_reporting = compute_antibiotic_testing_stats(
            ast_data_reporting, kept_reporting
        )
        print_antibiotic_stats_table(
            antibiotic_stats_reporting,
            title_suffix=f"(n >= {reporting_size})"
        )
        antibiogram_reporting_path = viz_dir / f"antibiogram_n_{reporting_size}.png"
        print(f"\nGenerating antibiogram for n >= {reporting_size}...")
        create_klebsiella_antibiogram(
            antibiotic_stats_reporting,
            antibiogram_reporting_path,
            min_samples=0,
        )
        paths['antibiogram_n'] = antibiogram_reporting_path

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)

    # Return all outputs
    return {
        'metadata': metadata_df,
        'binary_ast': binary_ast_df,
        'regression_log_mic': regression_log_mic_df,
        'kept_antibiotics': kept_antibiotics,
        'paths': paths
    }


def main():
    """Load AST data and run the EBI AST processing pipeline from the command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Process EBI AST data: print reports and save tables.")
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=None,
        help=f"Path to AST CSV file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--min-antibiotic-count",
        type=int,
        default=1000,
        help="Minimum number of measurements to keep an antibiotic (default: 1000). Ignored if --top-n is set.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        metavar="N",
        help="Keep only the top-N most-tested antibiotics. Overrides --min-antibiotic-count.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help=f"Directory for CSV outputs (metadata, binary_ast, regression_log_mic). Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--viz-dir",
        type=pathlib.Path,
        default=None,
        help=f"Directory for antibiogram PNGs. Default: {DEFAULT_VIZ_DIR}",
    )
    parser.add_argument(
        "--reporting-size",
        type=int,
        default=None,
        metavar="N",
        help="If set, repeat stats table and antibiogram for antibiotics with >= N tests; "
        "save antibiogram as antibiogram_n_N.png",
    )
    parser.add_argument(
        "--antibiogram-min-samples",
        type=int,
        default=2000,
        metavar="N",
        help="Minimum tests for an antibiotic to be displayed in the antibiogram figure "
        "(default: 2000). Does not affect saved pivot tables or stats CSV.",
    )
    args = parser.parse_args()

    process_klebsiella_ast_data(
        input_file=args.input,
        min_antibiotic_count=args.min_antibiotic_count,
        reporting_size=args.reporting_size,
        top_n=args.top_n,
        output_dir=args.output_dir,
        viz_dir=args.viz_dir,
        antibiogram_min_samples=args.antibiogram_min_samples,
    )


if __name__ == "__main__":
    main()
