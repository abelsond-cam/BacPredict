"""CARD-merged AMR labels + per-drug causal-gene lookup — the label-migration core.

Phases 1–2 attached an authoritative **CARD allele / gene-family** label to every AMR protein in the
``{Sample}_amr.parquet`` sidecars (:mod:`kleb_ast.annotate_amr_sidecar`), where CARD is the gold standard
and Bakta the historical (lossy) labeller. Downstream we migrate to a single **"CARD" label**:

    CARD family/allele where a minimap call qualifies, else the Bakta ``gene_name``.

That keeps the gene universe complete (non-AMR genes stay Bakta-named) while making every AMR protein's
identity reliable. Two helpers implement the migration, both pure lookups (no forward pass, no I/O beyond the
vendored CARD table):

- :func:`merged_label` — per ``flat_index``, the merged CARD-else-Bakta label (+ which source named it),
  at gene-**family** or **allele** grain. Consumers build their gene universe from this.
- :func:`causal_genes_for_drug` — the set of CARD gene-families/alleles **known to be causal** for a drug,
  for cross-hatching the per-gene plots. NARROWER than the inclusive ceiling map
  (:data:`kleb_ast.kleborate_determinant_lr.DRUG_COLUMNS`): the ceiling deliberately includes every plausible
  determinant, but the causal hatch wants the mechanistically-appropriate ones (carbapenems →
  carbapenemases + porin loss, *not* every β-lactamase). Built data-driven from
  ``CARD_AMR_clustered.csv`` (class / ``bla_class`` subgroup) plus the chromosomal mechanisms — no regex on
  gene names.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical vendored CARD clustering table (BacHGT bac_kleborate refs). HPC checkout first, local-dev second.
_CARD_CSV_CANDIDATES = (
    Path("/home/dca36/workspace/BacHGT/src/bac_kleborate/refs/kleb_amr/inputs/CARD_AMR_clustered.csv"),
    Path.home() / "developer" / "BacHGT" / "src" / "bac_kleborate" / "refs" / "kleb_amr" / "inputs"
    / "CARD_AMR_clustered.csv",
)


def _default_card_csv() -> Path:
    """First existing canonical CARD CSV path (HPC then local dev), else the HPC default."""
    for cand in _CARD_CSV_CANDIDATES:
        if cand.exists():
            return cand
    return _CARD_CSV_CANDIDATES[0]


DEFAULT_CARD_CSV = _default_card_csv()

# Chromosomal display gene-families — mirror bacpredict.engine.embedding.extract_proteins_from_gff_fna._CHROMOSOMAL_GENE_META.
# These are the exact amr_gene_family / amr_allele strings a chromosomal sidecar call carries (the same
# string at both grains, since the extractor sets allele == family for chromosomal refs).
_PORINS = ("OmpK35", "OmpK36")
_FLQ_CHR = ("GyrA", "ParC")
_COL_CHR = ("MgrB", "PmrB")

# β-lactamase subgroups (CARD ``bla_class``) by drug mechanism.
_CARB = ("Bla_Carb",)                                          # carbapenemases (KPC/NDM/OXA-48/VIM/IMP…)
_ESBL = ("Bla_ESBL", "Bla_ESBL_inhR", "Bla_Carb")             # ESBLs (+ carbapenemases co-confer ceph R)
_BLA_BROAD = ("Bla", "Bla_inhR", "Bla_ESBL", "Bla_ESBL_inhR", "Bla_Carb", "Bla_chr")  # penicillinases → carb

# Drug → causal mechanism spec. Pulls CARD families/alleles by ``class`` and/or ``bla_class`` subgroup, plus
# the chromosomal display gene-families. ``cr_fq`` adds the aac(6')-Ib-cr fluoroquinolone-acetylating allele
# (an AGly-class gene that cross-confers FQ resistance — precise only at allele grain). Keys are the
# lowercase AST drug columns; the set matches DRUG_COLUMNS (cross-checked in tests).
_DRUG_CAUSAL: dict[str, dict] = {
    # carbapenems: carbapenemases + porin loss
    "ertapenem": {"bla_class": _CARB, "chromosomal": _PORINS},
    "imipenem": {"bla_class": _CARB, "chromosomal": _PORINS},
    "meropenem": {"bla_class": _CARB, "chromosomal": _PORINS},
    # extended-spectrum cephalosporins + monobactam: ESBLs (+ carbapenemases) + porin loss
    "cefotaxime": {"bla_class": _ESBL, "chromosomal": _PORINS},
    "ceftriaxone": {"bla_class": _ESBL, "chromosomal": _PORINS},
    "ceftazidime": {"bla_class": _ESBL, "chromosomal": _PORINS},
    "cefepime": {"bla_class": _ESBL, "chromosomal": _PORINS},
    "aztreonam": {"bla_class": _ESBL, "chromosomal": _PORINS},
    # cephamycin + early cephalosporins + penicillin/combos: broad β-lactamases + porin loss
    "cefoxitin": {"bla_class": _BLA_BROAD, "chromosomal": _PORINS},
    "cefazolin": {"bla_class": _BLA_BROAD, "chromosomal": _PORINS},
    "cefuroxime": {"bla_class": _BLA_BROAD, "chromosomal": _PORINS},
    "ampicillin-sulbactam": {"bla_class": _BLA_BROAD, "chromosomal": _PORINS},
    "piperacillin-tazobactam": {"bla_class": _BLA_BROAD, "chromosomal": _PORINS},
    # fluoroquinolones: Flq-class acquired (qnr/qepA/…) + GyrA/ParC QRDR (+ aac(6')-Ib-cr at allele grain)
    "ciprofloxacin": {"card_class": ("Flq",), "chromosomal": _FLQ_CHR, "cr_fq": True},
    "levofloxacin": {"card_class": ("Flq",), "chromosomal": _FLQ_CHR, "cr_fq": True},
    # aminoglycosides: all AGly-class modifying enzymes (per-drug AME specificity not modelled — see docstring)
    "gentamicin": {"card_class": ("AGly",)},
    "amikacin": {"card_class": ("AGly",)},
    "tobramycin": {"card_class": ("AGly",)},
    # other classes
    "colistin": {"card_class": ("Col",), "chromosomal": _COL_CHR},
    "tetracycline": {"card_class": ("Tet",)},
    "azithromycin": {"card_class": ("MLS",)},
    "trimethoprim-sulfamethoxazole": {"card_class": ("Tmt", "Sul")},
}


# Drug → INCLUSIVE determinant spec (the one-hot ceiling universe — *broader* than the causal hatch). Where
# _DRUG_CAUSAL narrows β-lactams to the drug-appropriate subgroup (carbapenems → carbapenemases), the
# ceiling should weigh every plausibly-relevant determinant (all Bla subgroups + porins) and let the LR
# decide — under-including only lowers the ceiling. Mirrors the inclusive scope of DRUG_COLUMNS.
_BLA_ALL_INCL = ("Bla", "Bla_inhR", "Bla_ESBL", "Bla_ESBL_inhR", "Bla_Carb", "Bla_chr")
_DRUG_DETERMINANT: dict[str, dict] = {
    **{d: {"bla_class": _BLA_ALL_INCL, "chromosomal": _PORINS}
       for d in ("ertapenem", "imipenem", "meropenem", "cefotaxime", "ceftriaxone", "ceftazidime",
                 "cefepime", "aztreonam", "cefoxitin", "cefazolin", "cefuroxime",
                 "ampicillin-sulbactam", "piperacillin-tazobactam")},
    "ciprofloxacin": {"card_class": ("Flq",), "chromosomal": _FLQ_CHR, "cr_fq": True},
    "levofloxacin": {"card_class": ("Flq",), "chromosomal": _FLQ_CHR, "cr_fq": True},
    "gentamicin": {"card_class": ("AGly",)},
    "amikacin": {"card_class": ("AGly",)},
    "tobramycin": {"card_class": ("AGly",)},
    "colistin": {"card_class": ("Col",), "chromosomal": _COL_CHR},
    "tetracycline": {"card_class": ("Tet",)},
    "azithromycin": {"card_class": ("MLS",)},
    "trimethoprim-sulfamethoxazole": {"card_class": ("Tmt", "Sul")},
}


@lru_cache(maxsize=4)
def _load_card(card_csv: str) -> pd.DataFrame:
    """Read CARD_AMR_clustered.csv as all-string columns (empty bla_class stays ``""``)."""
    return pd.read_csv(card_csv, dtype=str).fillna("")


def _genes_for_spec(spec: dict, card: pd.DataFrame, col: str, grain: str) -> set[str]:
    """Resolve a causal/determinant spec dict to a set of CARD ``col`` values + chromosomal display names."""
    out: set[str] = set()
    classes = spec.get("card_class", ())
    if classes:
        out |= set(card.loc[card["class"].isin(classes), col]) - {""}
    blacs = spec.get("bla_class", ())
    if blacs:
        out |= set(card.loc[card["bla_class"].isin(blacs), col]) - {""}
    if spec.get("cr_fq") and grain == "allele":
        cr = card["allele"].str.lower().str.contains("ib-cr", na=False)
        out |= set(card.loc[cr, "allele"]) - {""}
    out |= set(spec.get("chromosomal", ()))
    return out


def determinant_genes_for_drug(
    drug: str, *, grain: str = "family", card_csv: Path | str | None = None
) -> set[str]:
    """Inclusive CARD gene/allele set for the drug's resistance classes — the one-hot **ceiling** universe.

    Broader than :func:`causal_genes_for_drug`: β-lactams span every ``bla_class`` subgroup (a ceiling
    should weigh all plausible determinants, like the inclusive ``DRUG_COLUMNS``). Used to scope the CARD
    one-hot bars + ``__ALL_CARD__`` ceiling and the ladder's gene universe.
    """
    if grain not in ("family", "allele"):
        raise ValueError(f"grain must be 'family' or 'allele', got {grain!r}")
    spec = _DRUG_DETERMINANT.get(drug)
    if spec is None:
        raise ValueError(f"no determinant spec for drug {drug!r} (known: {sorted(_DRUG_DETERMINANT)})")
    card = _load_card(str(card_csv or DEFAULT_CARD_CSV))
    return _genes_for_spec(spec, card, "gene" if grain == "family" else "allele", grain)


def causal_genes_for_drug(
    drug: str, *, grain: str = "family", card_csv: Path | str | None = None
) -> set[str]:
    """Set of CARD gene-families (or alleles) known to be causal for ``drug``, for the per-gene hatch.

    Parameters
    ----------
    drug
        Lowercase AST drug column (e.g. ``"meropenem"``, ``"ciprofloxacin"``).
    grain
        ``"family"`` → CARD ``gene`` families; ``"allele"`` → CARD ``allele`` strings. Chromosomal display
        names (GyrA/ParC/OmpK35/OmpK36/MgrB/PmrB) are returned identically at both grains.
    card_csv
        Override the vendored ``CARD_AMR_clustered.csv`` path (defaults to :data:`DEFAULT_CARD_CSV`).

    Returns
    -------
    set[str]
        Causal CARD families/alleles + chromosomal mechanisms. These strings match the sidecar
        ``amr_gene_family`` / ``amr_allele`` exactly (same provenance), so :func:`merged_label` output can be
        membership-tested against this set directly.

    Notes
    -----
    NARROWER than ``DRUG_COLUMNS`` (the inclusive catalogue ceiling): β-lactams are restricted to the
    mechanistically-appropriate ``bla_class`` subgroup (carbapenems → carbapenemases, not every β-lactamase).
    Aminoglycoside AME-vs-drug specificity is *not* modelled (all AGly enzymes count for every
    aminoglycoside) — the programme axis is HGT-vs-chromosomal, for which class-level is faithful.
    """
    if grain not in ("family", "allele"):
        raise ValueError(f"grain must be 'family' or 'allele', got {grain!r}")
    spec = _DRUG_CAUSAL.get(drug)
    if spec is None:
        raise ValueError(f"no causal-mechanism spec for drug {drug!r} (known: {sorted(_DRUG_CAUSAL)})")
    card = _load_card(str(card_csv or DEFAULT_CARD_CSV))
    return _genes_for_spec(spec, card, "gene" if grain == "family" else "allele", grain)


def merged_label(
    bakta_gene_names: list, calls_df: pd.DataFrame | None, *, grain: str = "family"
) -> pd.DataFrame:
    """Per ``flat_index`` merged label: CARD family/allele where an AMR call lands, else Bakta gene_name.

    Parameters
    ----------
    bakta_gene_names
        The genome's per-protein Bakta ``gene_name`` values **in flat order** (index = ``flat_index``), i.e.
        ``[r["gene_name"] for r in flatten_proteins(parquet)]``. ``None``/``NaN``/``""`` → unnamed.
    calls_df
        The genome's ``{Sample}_amr.parquet`` sidecar (or ``None``). Rows with ``amr_source`` in
        ``{"acquired", "chromosomal"}`` and a valid ``flat_index`` override the Bakta name at that index.
    grain
        ``"family"`` reads ``amr_gene_family``; ``"allele"`` reads ``amr_allele``.

    Returns
    -------
    pandas.DataFrame
        Columns ``flat_index``, ``label`` (str or ``None`` when neither source names the protein),
        ``source`` (``"card"`` / ``"bakta"`` / ``None``). One row per protein, in flat order.
    """
    if grain not in ("family", "allele"):
        raise ValueError(f"grain must be 'family' or 'allele', got {grain!r}")
    n = len(bakta_gene_names)
    label_col = "amr_gene_family" if grain == "family" else "amr_allele"

    card_by_fi: dict[int, str] = {}
    if calls_df is not None and len(calls_df) and label_col in calls_df.columns:
        sub = calls_df[calls_df["amr_source"].astype(str).isin(("acquired", "chromosomal"))]
        for fi, lab in zip(sub["flat_index"].to_numpy(), sub[label_col].to_numpy(), strict=True):
            idx = int(fi)
            s = "" if lab is None else str(lab)
            if 0 <= idx < n and s and s.lower() != "nan":
                card_by_fi[idx] = s

    labels: list[str | None] = []
    sources: list[str | None] = []
    for i, gn in enumerate(bakta_gene_names):
        if i in card_by_fi:
            labels.append(card_by_fi[i])
            sources.append("card")
            continue
        g = "" if gn is None else str(gn)
        if g and g.lower() != "nan":
            labels.append(g)
            sources.append("bakta")
        else:
            labels.append(None)
            sources.append(None)
    return pd.DataFrame({"flat_index": list(range(n)), "label": labels, "source": sources})
