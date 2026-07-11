"""Shared plotting labels — the drug display-name map used across the ladder / cause / ranking plots.

The AST data columns use US / short spellings (``rifampin``); titles and the per-drug visualisation
dirs want the INN spelling (``rifampicin``). This one map was copy-pasted into every plot module.
"""
from __future__ import annotations

# AST column name -> display spelling for titles / visualisation dirs. Drugs not listed pass through.
DRUG_DISPLAY = {"rifampin": "rifampicin"}


def display_name(drug: str) -> str:
    """The proper drug name used in titles and the per-drug visualisation dir (``rifampin`` → ``rifampicin``)."""
    return DRUG_DISPLAY.get(drug, drug)
