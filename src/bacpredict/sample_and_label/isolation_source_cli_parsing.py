"""Shared CLI parsing helpers for isolation-source pair workflows."""

from __future__ import annotations

import re
from typing import Set, Tuple

import pandas as pd


def resolve_isolation_column(df: pd.DataFrame) -> str:
    """Resolve the isolation source category column name."""
    if "isolation_source_category" in df.columns:
        return "isolation_source_category"
    if "phenotype-isolation_source_category" in df.columns:
        return "phenotype-isolation_source_category"
    raise ValueError(
        "Expected 'isolation_source_category' or 'phenotype-isolation_source_category' "
        f"in columns, got {list(df.columns)}"
    )


def find_matching_categories(
    df: pd.DataFrame,
    token: str,
    isolation_col: str = "isolation_source_category",
) -> Set[str]:
    """Find all isolation-source categories that contain a token as a whole word."""
    if isolation_col not in df.columns:
        return set()

    categories = df[isolation_col].dropna().unique()
    token_lower = token.strip().lower()

    matches: Set[str] = set()
    for category in categories:
        words = re.findall(r"[a-z0-9]+", str(category).lower())
        if token_lower in words:
            matches.add(str(category))

    return matches


def resolve_isolation_source_token(
    df: pd.DataFrame,
    token: str,
    isolation_col: str = "isolation_source_category",
) -> str:
    """Resolve a CLI token to exactly one isolation source category."""
    token_norm = token.strip().lower()
    if not token_norm:
        raise ValueError("Isolation source token cannot be empty.")
    if isolation_col not in df.columns:
        raise ValueError(f"Expected column '{isolation_col}' in metadata.")

    matches = find_matching_categories(df, token_norm, isolation_col=isolation_col)
    counts = df[isolation_col].value_counts(dropna=True).to_dict()

    if len(matches) == 0:
        available = sorted(df[isolation_col].dropna().astype(str).unique().tolist())
        preview = ", ".join(available[:25]) + ("..." if len(available) > 25 else "")
        raise ValueError(
            f"Token '{token}' did not match any isolation source category.\n"
            "Matched words are compared against alphanumeric tokens in the category.\n"
            f"Available categories (preview): {preview}"
        )

    if len(matches) > 1:
        matched_lines = [
            f"  - {category}: {counts.get(category, 0):,} samples"
            for category in sorted(matches)
        ]
        raise ValueError(
            f"Token '{token}' matched multiple isolation source categories; expected exactly 1.\n"
            f"Matched categories:\n{chr(10).join(matched_lines)}"
        )

    return next(iter(matches))


def validate_and_resolve_tokens(
    df: pd.DataFrame,
    token1: str,
    token2: str,
    isolation_col: str = "isolation_source_category",
) -> Tuple[str, str]:
    """Validate two tokens and resolve them to two different category values."""
    category1 = resolve_isolation_source_token(df, token1, isolation_col=isolation_col)
    category2 = resolve_isolation_source_token(df, token2, isolation_col=isolation_col)

    if category1 == category2:
        raise ValueError(
            f"Both tokens match the same category: '{category1}'. "
            "Please provide two different isolation source tokens."
        )

    return category1, category2


def slugify_isolation_source_token(token: str) -> str:
    """Convert a CLI isolation-source token to a safe lowercase path fragment (matches prepare script)."""
    slug = re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_")
    return slug or "unknown"


def sanitize_pair_name(token1: str, token2: str) -> str:
    """Create deterministic lowercase slug for an isolation-source token pair."""
    left = re.sub(r"[^a-z0-9]+", "_", token1.strip().lower()).strip("_")
    right = re.sub(r"[^a-z0-9]+", "_", token2.strip().lower()).strip("_")
    if not left or not right:
        raise ValueError("Could not derive pair name from provided isolation source tokens.")
    return f"{left}_vs_{right}"
