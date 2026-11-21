"""Which labelled species each split-builder unit of the Enterobacterales cohort holds.

A unit is only a ``--species NAME`` in :mod:`bacpredict.engine.splits.build_amr_holdout_splits`. **The builder does
not filter by species**: a unit's eligible-genome CSV is its entire membership (``build_amr_holdout_splits.py:390``),
and the NAME is a label that also seeds the unit's shuffle. So the membership rule lives here, in one place, and
every arm reads it — the single-species arms, the cross-species arm, and CABBAGE v2 (§3.7). ``--assert-same-as``
compares arms row by row, so a unit that is defined differently in two arms silently fails the comparison it exists
to guarantee.

⛔ **A unit names its species. It never prefix-matches the genus.** *K. aerogenes* is not in the *K. pneumoniae*
species complex, and the two label sources spell it differently — the EBI portal 2026-07 cohort says
``Enterobacter aerogenes`` (28 genomes), CABBAGE v2 says ``Klebsiella aerogenes`` (424) — so
``species.startswith("Klebsiella")`` would put one organism in two different units depending only on which sheet a
row came from. Both spellings are in ``ENTEROBACTER`` below.

The rule reads the **submitter's species label**, not the skani ``species_call``. ANI against our draft reference
panel cannot adjudicate the *E. coli* / *Shigella* boundary — 5,199 of the 21,699 genomes labelled *E. coli* are
called *S. boydii*, and 8 of the 12 labelled *S. boydii* come back *S. sonnei* — because *Shigella* sits
phylogenetically inside *E. coli*. PROJECT_STATE §3.6 *Caveats*.

Subgroups follow David's rule (PROJECT_STATE §6, 2026-11-21): *E. coli* and *Shigella* are split, smaller species
are not.
"""

from __future__ import annotations

import pandas as pd

#: The *K. pneumoniae* species complex. ``K. aerogenes`` is NOT a member and belongs to :data:`ENTEROBACTER`.
KPSC = (
    "Klebsiella pneumoniae",
    "Klebsiella variicola",
    "Klebsiella quasipneumoniae",
    "Klebsiella africana",
    "Klebsiella tropica",
)

#: *Enterobacter*, plus *K. aerogenes* under both the old and the current name.
ENTEROBACTER = ("Klebsiella aerogenes", "Enterobacter aerogenes")

#: ⚠ *S. boydii*, *S. dysenteriae* and a bare ``Shigella`` label are too few for a unit of their own (12, 6 and 3 in
#: the 2026-07 cohort) and sit genomically inside *E. coli*, so they join the ``ecoli`` unit. Their STs come from the
#: same ``ecoli_achtman_4`` scheme, so the ``st_key`` namespace is already shared. **Confirm with David before the
#: cross-species arm is built**: the alternative is to leave them out of every unit, which drops 21 labelled genomes.
SHIGELLA_WITH_ECOLI = ("Shigella boydii", "Shigella dysenteriae", "Shigella")

#: unit name -> a predicate over the submitter's species label. Order matters only for readability; the units are
#: disjoint, which :func:`assign_units` asserts (the builder refuses a genome eligible in two units anyway).
UNITS: dict[str, tuple[str, ...]] = {
    "kpsc": KPSC,
    "ecoli": ("Escherichia coli", *SHIGELLA_WITH_ECOLI),
    "ssonnei": ("Shigella sonnei",),
    "sflexneri": ("Shigella flexneri",),
    "enterobacter": ENTEROBACTER,
}

#: Genera taken whole, for species that are not named individually above. *Salmonella* carries serovar-style labels
#: (``Salmonella Infantis``, ``Salmonella sp.``) that no explicit list would catch, and it has no subgroup rule.
UNIT_GENERA: dict[str, tuple[str, ...]] = {
    "salmonella": ("Salmonella",),
    "enterobacter": ("Enterobacter",),
}

#: ⚠ Genera with no unit yet. They are Enterobacterales and the cross-species arm is "PsA + all Enterobacterales",
#: so they need a decision before that arm is built — one unit each, one pooled unit, or left out. They do not
#: affect any single-species arm. Counts in the 2026-07 QC-pass cohort: Serratia 329, Proteus 104, Morganella 60,
#: Providencia 25.
UNDECIDED_GENERA = ("Serratia", "Proteus", "Morganella", "Providencia")


def assign_units(species: pd.Series) -> pd.Series:
    """Unit name per genome, ``NA`` where the species belongs to no unit.

    Parameters
    ----------
    species
        The submitter's species label, one row per genome (``genome_qc.csv`` / ``cohort.csv`` ``species``).

    Returns
    -------
    pandas.Series
        The unit name, indexed as ``species``, with ``pandas.NA`` for a species in no unit.

    Raises
    ------
    ValueError
        If a species would fall in more than one unit, which would make the units overlap.
    """
    s = species.fillna("").astype(str).str.strip()
    genus = s.str.split().str[0]
    out = pd.Series(pd.NA, index=species.index, dtype="object")
    for unit, names in UNITS.items():
        hit = s.isin(names)
        clash = hit & out.notna()
        if clash.any():
            raise ValueError(f"{unit}: {sorted(set(s[clash]))} already in {sorted(set(out[clash]))}")
        out = out.mask(hit, unit)
    for unit, genera in UNIT_GENERA.items():
        hit = genus.isin(genera) & out.isna()
        out = out.mask(hit, unit)
    return out


def unit_samples(qc: pd.DataFrame, unit: str) -> list[str]:
    """The ``Sample`` ids of ``unit``'s QC-passing genomes, sorted.

    Parameters
    ----------
    qc
        ``genome_qc.csv``, with ``Sample``, ``species`` and ``qc_pass`` columns.
    unit
        A key of :data:`UNITS` or :data:`UNIT_GENERA`.

    Returns
    -------
    list of str
        Sorted sample ids.
    """
    if unit not in UNITS and unit not in UNIT_GENERA:
        raise ValueError(f"unknown unit {unit!r}; expected one of {sorted({*UNITS, *UNIT_GENERA})}")
    passing = qc[qc["qc_pass"].astype(str).str.lower().isin(("true", "1"))]
    return sorted(passing.loc[assign_units(passing["species"]).eq(unit), "Sample"])
