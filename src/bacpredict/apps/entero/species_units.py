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

**What a unit is FOR** (PROJECT_STATE §6, 2026-11-22, and the reason most of the open questions here closed): a
unit exists so an ST can be held out of it, and an ST is held out only where the unit has ≥ 1,000 genomes and the
ST ≥ 50 — see :data:`HOLDOUT_ST_MIN_UNIT`. Below that threshold a unit's boundaries change only how the random
holdout is drawn, never what the arm can measure. So the remaining question for every small genus is not *which*
unit but *whether* it is in one at all: a species named nowhere in this module is absent from the cross-species
arm, silently.
"""

from __future__ import annotations

import pandas as pd

#: A unit carries a held-out ST only at or above this many genomes, and the ST itself only at or above
#: :data:`HOLDOUT_ST_MIN_ST`. David, PROJECT_STATE §6 (2026-11-22): "we only hold out an ST if there will be at
#: least 50 samples. We need at least 1000 samples from the species to make that even a concern."
#:
#: ⚠ Nothing here enforces it — ``--holdout-st`` is passed per arm by the training scripts, and the builder does not
#: know a unit's size rule. It is recorded here because this module is where a unit is defined, and the rule is what
#: decides whether a unit's *membership* matters beyond the random draw. Units that qualify today:
#: ``kpsc`` 9,370 genomes with ``klebsiella:37`` at 211, and PsA 3,677 with ``paeruginosa:155`` at 82
#: (PROJECT_STATE §3.6). Below the threshold, grouping changes only how the random holdout is drawn — which is not
#: nothing, since a unit's NAME seeds its shuffle, but it is no longer a question of what the arm can measure.
HOLDOUT_ST_MIN_UNIT = 1000
HOLDOUT_ST_MIN_ST = 50

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

#: *S. boydii*, *S. dysenteriae* and a bare ``Shigella`` label are too few for a unit of their own (12, 6 and 3 in
#: the 2026-07 cohort; 32 labelled in v2) and sit genomically inside *E. coli*, so they join the ``ecoli`` unit.
#: Their STs come from the same ``ecoli_achtman_4`` scheme, so the ``st_key`` namespace is already shared — a
#: *Shigella* that happens to be ``ecoli``'s held-out ST is therefore held out with it, which is what we want.
#: Settled by :data:`HOLDOUT_ST_MIN_UNIT`: at 32 they could never carry a holdout ST of their own, so the only
#: live alternative was to leave them in no unit at all, which drops labelled genomes for nothing.
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

#: Genera in no unit. **No longer a correctness question** — see :data:`HOLDOUT_ST_MIN_UNIT` above. None of them is
#: within a factor of two of the 1,000-genome threshold (the largest, *Serratia*, has 553 labelled genomes in v2's
#: ``cohort.csv``; PROJECT_STATE §6, 2026-11-22), so none can carry a held-out ST, and which unit holds them changes
#: only how the random holdout is drawn. They do not affect any single-species arm. Counts as 2026-07 QC-pass +
#: CABBAGE v2 new (§3.7): Serratia 329 + 224, Providencia 25 + 338, Morganella 60 + 201, Proteus 104 + 153.
#: 518 genomes here, ~1,435 across both cohorts.
UNDECIDED_GENERA = ("Serratia", "Proteus", "Morganella", "Providencia")

#: ⚠ Genera that only CABBAGE v2 has, so reading this rule against the 2026-07 cohort alone can never show them.
#: They are listed for exactly that reason: unlisted, they fall to ``NA`` silently, which is the failure this
#: module exists to prevent. v2 labelled counts: Citrobacter 399, Yersinia 296, Kluyvera 12, Pluralibacter 9,
#: Hafnia 8, Leclercia 3, Cronobacter 2, Erwinia 1, Pantoea 1. The largest, *Citrobacter*, is well under
#: :data:`HOLDOUT_ST_MIN_UNIT`, so none of them can carry a held-out ST either.
UNDECIDED_GENERA_V2 = ("Citrobacter", "Yersinia", "Kluyvera", "Pluralibacter", "Hafnia", "Leclercia",
                       "Cronobacter", "Erwinia", "Pantoea")

#: ⚠ The *K. oxytoca* complex is Klebsiella but NOT KpSC, so it falls in no unit and would be dropped from a
#: cross-species arm. The 2026-07 cohort has none; **CABBAGE v2 (§3.7) does** — it pins a `koxytoca` MLST scheme.
#: v2 labelled counts, all new: K. oxytoca 173, K. michiganensis 131, K. grimontii 15, K. pasteurii 11 (332 by v2's
#: ``cohort.csv``). At 332 it is below :data:`HOLDOUT_ST_MIN_UNIT`, so it needs no holdout ST of its own; it still
#: needs *some* unit, or those genomes are simply absent. Listed, not assigned, so the gap stays visible.
UNDECIDED_KOSC = ("Klebsiella oxytoca", "Klebsiella michiganensis", "Klebsiella grimontii", "Klebsiella pasteurii")

#: Everything still in no unit, for a caller that wants to report what an arm would drop. **This is now the only
#: consequence that matters**: a genome in no unit is not excluded from the cross-species arm, it is invisible to
#: it — the builder takes a unit's membership as its eligible set (``build_amr_holdout_splits.py:390``), so a
#: species named nowhere here is silently absent rather than reported. Every group below is under
#: :data:`HOLDOUT_ST_MIN_UNIT`, so assigning them is about inclusion, not about what any arm can measure.
UNDECIDED = (*UNDECIDED_GENERA, *UNDECIDED_GENERA_V2, *UNDECIDED_KOSC)


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
