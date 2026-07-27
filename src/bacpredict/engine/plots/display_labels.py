"""Shared plotting labels — the drug display-name map used across the ladder / cause / ranking plots.

The AST data columns use US / short spellings (``rifampin``); titles and the per-drug visualisation
dirs want the INN spelling (``rifampicin``). This one map was copy-pasted into every plot module.
"""
from __future__ import annotations

# AST column name -> display spelling for titles / visualisation dirs. Drugs not listed pass through.
DRUG_DISPLAY = {"rifampin": "rifampicin"}

# Non-coding region anchor-gene -> the determinant it regulates, when the two differ. The mabA(fabG1)-inhA
# operon promoter is anchored 5′ of ``fabG1`` but is the ``inhA`` promoter clinically, so a region label
# should read by the determinant the catalogue names. (Source of truth for the causal plot's join synonyms.)
PROMOTER_GENE_TO_DETERMINANT = {"fabg1": "inhA"}

# per-unit region type token -> its short display; the per-unit key is ``<type>:<name>`` (e.g. ``rrna:rrs``).
_UNIT_TYPE_DISPLAY = {"rrna": "rRNA", "trna": "tRNA", "ncrna": "ncRNA", "tmrna": "tmRNA"}


def display_name(drug: str) -> str:
    """The proper drug name used in titles and the per-drug visualisation dir (``rifampin`` → ``rifampicin``)."""
    return DRUG_DISPLAY.get(drug, drug)


def region_label(block: str) -> str:
    """Terse human label for a non-coding region block key — shared by the ladder + causal plots.

    * ``upstream:<gene>`` → "``<determinant>`` promoter" (mapping the anchor gene back to the determinant it
      regulates via :data:`PROMOTER_GENE_TO_DETERMINANT`, e.g. ``upstream:fabg1`` → "inhA promoter").
    * ``between:<a>→<b>`` (convergent flank-pair fallback) → the bare pair ``a→b``.
    * per-unit ``<type>:<name>`` → "``<name> <type>``" for RNA bodies (``rrna:rrs`` → "rrs rRNA"), else the
      key before the colon (``oric:origin of replication`` → "oric").
    * a bare gene or an already-merged convergent pair (``mura→ogt``) passes through unchanged.
    """
    b = str(block).strip()
    low = b.lower()
    if low.startswith("upstream:"):
        gene = b.split(":", 1)[1].strip()
        return f"{PROMOTER_GENE_TO_DETERMINANT.get(gene.lower(), gene)} promoter"
    if low.startswith("between:"):
        return b.split(":", 1)[1].strip()
    if ":" in b:
        typ, name = (x.strip() for x in b.split(":", 1))
        disp = _UNIT_TYPE_DISPLAY.get(typ.lower())
        return f"{name} {disp}" if disp else typ
    return b
