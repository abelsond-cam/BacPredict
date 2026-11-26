"""AST labels for the PsA ten drugs at each phylogenetic tier, balanced to PsA's own rate.

Scoping only — it counts labels, it does not build a cohort. The rule is the one of record:

* labels come from ``engine/ast_labels`` unchanged (EUCAST v15 from the MIC where a breakpoint row
  exists, the source's own R/S where there is no MIC, censored MICs that straddle R dropped);
* every species is then thinned to PsA's own %R for that drug by dropping at random from whichever
  class is in excess (decision of record, 2026-11-25). Only the *count* is needed here, which is
  ``min(n_R / p, n_S / (1 - p))``, so nothing is actually sampled.

Two columns per (species, drug), because the breakpoint table reaches only Enterobacterales and
Pseudomonas today:

``n_today``    what the pipeline would label right now. Outside those two taxa an MIC resolves to no
               breakpoint row and the test is dropped — it is NOT rescued by the source's call — so
               these species keep only their bare R/S calls and disk tests.
``n_ceiling``  what they would label if the breakpoint table were extended to the added taxa. The
               MIC-derived call is unknown until those rows exist, so the source's own call stands in
               for it (PROJECT_STATE 3.7: source call vs our EUCAST v15 re-call agree on 95.5-99.7 %
               of tests that carry both). An estimate, not a count.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import bacpredict
from bacpredict.engine.ast_labels.ast_tests import SOURCE_CALL_TO_BINARY, call_tests, read_portal_tests
from bacpredict.engine.ast_labels.breakpoints import DEFAULT_BREAKPOINTS, load_breakpoints

AMR = Path(os.environ.get("BACPREDICT_AMR_ROOT", "/rds/project/rds-4k08a2yyQLw/david/amr_prediction"))
PHENOTYPE = AMR / "processed/ebi_amr_portal/cabbage_v2_20261121/phenotype.parquet"
OUT = AMR / "processed/ebi_amr_portal/scoping_wider_phylogeny_20261126"

DRUGS = ["ciprofloxacin", "ceftazidime", "meropenem", "tobramycin", "amikacin", "imipenem",
         "piperacillin-tazobactam", "aztreonam", "levofloxacin", "cefepime"]

# Target resistance rate per drug: PsA's own, from the decision of record of 2026-11-25
# (PROJECT_STATE 3.8, the v2 balanced split tables' train rows). PsA is the reference and is never thinned.
TARGET_PCT_R = {"meropenem": 43.5, "imipenem": 64.2, "piperacillin-tazobactam": 52.7, "amikacin": 21.8,
                "ceftazidime": 48.4, "ciprofloxacin": 61.8, "cefepime": 48.8, "aztreonam": 44.1,
                "tobramycin": 35.0, "levofloxacin": 56.2}

PA = "Pseudomonas aeruginosa"

# genus -> (family, order, class, phylum). NCBI ranks; Moraxellaceae sits in Pseudomonadales, so
# Acinetobacter enters at P. aeruginosa's OWN order, not at class.
_T = {
    "Enterobacteriaceae": ("Enterobacterales", "Gammaproteobacteria", "Pseudomonadota"),
    "Yersiniaceae": ("Enterobacterales", "Gammaproteobacteria", "Pseudomonadota"),
    "Morganellaceae": ("Enterobacterales", "Gammaproteobacteria", "Pseudomonadota"),
    "Hafniaceae": ("Enterobacterales", "Gammaproteobacteria", "Pseudomonadota"),
    "Erwiniaceae": ("Enterobacterales", "Gammaproteobacteria", "Pseudomonadota"),
    "Pectobacteriaceae": ("Enterobacterales", "Gammaproteobacteria", "Pseudomonadota"),
    "Budviciaceae": ("Enterobacterales", "Gammaproteobacteria", "Pseudomonadota"),
    "Pseudomonadaceae": ("Pseudomonadales", "Gammaproteobacteria", "Pseudomonadota"),
    "Moraxellaceae": ("Pseudomonadales", "Gammaproteobacteria", "Pseudomonadota"),
    "Xanthomonadaceae": ("Xanthomonadales", "Gammaproteobacteria", "Pseudomonadota"),
    "Vibrionaceae": ("Vibrionales", "Gammaproteobacteria", "Pseudomonadota"),
    "Aeromonadaceae": ("Aeromonadales", "Gammaproteobacteria", "Pseudomonadota"),
    "Shewanellaceae": ("Alteromonadales", "Gammaproteobacteria", "Pseudomonadota"),
    "Pasteurellaceae": ("Pasteurellales", "Gammaproteobacteria", "Pseudomonadota"),
    "Legionellaceae": ("Legionellales", "Gammaproteobacteria", "Pseudomonadota"),
    "Francisellaceae": ("Thiotrichales", "Gammaproteobacteria", "Pseudomonadota"),
    "Burkholderiaceae": ("Burkholderiales", "Betaproteobacteria", "Pseudomonadota"),
    "Alcaligenaceae": ("Burkholderiales", "Betaproteobacteria", "Pseudomonadota"),
    "Neisseriaceae": ("Neisseriales", "Betaproteobacteria", "Pseudomonadota"),
    "Brucellaceae": ("Hyphomicrobiales", "Alphaproteobacteria", "Pseudomonadota"),
    "Campylobacteraceae": ("Campylobacterales", "Epsilonproteobacteria", "Campylobacterota"),
    "Helicobacteraceae": ("Campylobacterales", "Epsilonproteobacteria", "Campylobacterota"),
    "Staphylococcaceae": ("Bacillales", "Bacilli", "Bacillota"),
    "Listeriaceae": ("Bacillales", "Bacilli", "Bacillota"),
    "Bacillaceae": ("Bacillales", "Bacilli", "Bacillota"),
    "Streptococcaceae": ("Lactobacillales", "Bacilli", "Bacillota"),
    "Enterococcaceae": ("Lactobacillales", "Bacilli", "Bacillota"),
    "Lactobacillaceae": ("Lactobacillales", "Bacilli", "Bacillota"),
    "Peptostreptococcaceae": ("Eubacteriales", "Clostridia", "Bacillota"),
    "Clostridiaceae": ("Eubacteriales", "Clostridia", "Bacillota"),
    "Erysipelotrichaceae": ("Erysipelotrichales", "Bacilli", "Bacillota"),
    "Mycobacteriaceae": ("Mycobacteriales", "Actinomycetes", "Actinomycetota"),
    "Corynebacteriaceae": ("Mycobacteriales", "Actinomycetes", "Actinomycetota"),
    "Nocardiaceae": ("Mycobacteriales", "Actinomycetes", "Actinomycetota"),
    "Bifidobacteriaceae": ("Bifidobacteriales", "Actinomycetes", "Actinomycetota"),
    "Actinomycetaceae": ("Actinomycetales", "Actinomycetes", "Actinomycetota"),
    "Flavobacteriaceae": ("Flavobacteriales", "Flavobacteriia", "Bacteroidota"),
    "Bacteroidaceae": ("Bacteroidales", "Bacteroidia", "Bacteroidota"),
    "Prevotellaceae": ("Bacteroidales", "Bacteroidia", "Bacteroidota"),
    "Mycoplasmataceae": ("Mycoplasmatales", "Mollicutes", "Mycoplasmatota"),
    "Brachyspiraceae": ("Brachyspirales", "Spirochaetia", "Spirochaetota"),
    "Fusobacteriaceae": ("Fusobacteriales", "Fusobacteriia", "Fusobacteriota"),
    "NON_BACTERIAL": ("NON_BACTERIAL", "NON_BACTERIAL", "NON_BACTERIAL"),
}
_GENUS_FAMILY = {
    **dict.fromkeys(("Escherichia", "Shigella", "Salmonella", "Klebsiella", "Enterobacter", "Citrobacter", "Raoultella", "Cronobacter", "Kluyvera", "Leclercia", "Kosakonia", "Pluralibacter", "Lelliottia", "Atlantibacter", "Phytobacter", "Siccibacter", "Plesiomonas", "Klebsiella/Raoultella", "Enterobacteriaceae", "Enterobacterales", "Superficieibacter", "Metakosakonia", "Izhakiella", "Buttiauxella"), "Enterobacteriaceae"),
    **dict.fromkeys(("Serratia", "Yersinia", "Rouxiella", "Ewingella", "Rahnella"), "Yersiniaceae"),
    **dict.fromkeys(("Proteus", "Morganella", "Providencia", "Photorhabdus", "Xenorhabdus"), "Morganellaceae"),
    **dict.fromkeys(("Hafnia", "Edwardsiella", "Obesumbacterium"), "Hafniaceae"),
    **dict.fromkeys(("Pantoea", "Erwinia", "Mixta", "Tatumella"), "Erwiniaceae"),
    **dict.fromkeys(("Pectobacterium", "Dickeya", "Brenneria", "Sodalis"), "Pectobacteriaceae"),
    **dict.fromkeys(("Budvicia", "Limnobaculum"), "Budviciaceae"),
    **dict.fromkeys(("Pseudomonas", "Azotobacter", "Stutzerimonas", "Halopseudomonas"), "Pseudomonadaceae"),
    **dict.fromkeys(("Acinetobacter", "Moraxella", "Psychrobacter"), "Moraxellaceae"),
    **dict.fromkeys(("Stenotrophomonas", "Xanthomonas", "Lysobacter", "Luteimonas"), "Xanthomonadaceae"),
    **dict.fromkeys(("Vibrio", "Photobacterium", "Grimontia"), "Vibrionaceae"),
    **dict.fromkeys(("Aeromonas", "Oceanimonas"), "Aeromonadaceae"),
    **dict.fromkeys(("Shewanella",), "Shewanellaceae"),
    **dict.fromkeys(("Haemophilus", "Glaesserella", "Actinobacillus", "Pasteurella", "Mannheimia", "Avibacterium", "Histophilus", "Frederiksenia", "Parahaemophilus", "Bibersteinia", "Gallibacterium", "Rodentibacter", "Necropsobacter", "Aggregatibacter", "Pasteurellaceae"), "Pasteurellaceae"),
    **dict.fromkeys(("Legionella",), "Legionellaceae"),
    **dict.fromkeys(("Francisella",), "Francisellaceae"),
    **dict.fromkeys(("Burkholderia", "Pandoraea", "Ralstonia", "Cupriavidus", "Paraburkholderia", "Caballeronia"), "Burkholderiaceae"),
    **dict.fromkeys(("Bordetella", "Achromobacter", "Kerstersia", "Alcaligenes", "Taylorella"), "Alcaligenaceae"),
    **dict.fromkeys(("Neisseria", "Eikenella", "Kingella", "Laribacter", "Chromobacterium", "Simonsiella"), "Neisseriaceae"),
    **dict.fromkeys(("Brucella", "Ochrobactrum"), "Brucellaceae"),
    **dict.fromkeys(("Campylobacter", "Aliarcobacter", "Arcobacter", "Malaciobacter"), "Campylobacteraceae"),
    **dict.fromkeys(("Helicobacter",), "Helicobacteraceae"),
    **dict.fromkeys(("Staphylococcus", "Mammaliicoccus", "Macrococcus"), "Staphylococcaceae"),
    **dict.fromkeys(("Listeria",), "Listeriaceae"),
    **dict.fromkeys(("Bacillus", "Priestia", "Peribacillus", "Lysinibacillus"), "Bacillaceae"),
    **dict.fromkeys(("Streptococcus", "Lactococcus"), "Streptococcaceae"),
    **dict.fromkeys(("Enterococcus", "Vagococcus", "Tetragenococcus"), "Enterococcaceae"),
    **dict.fromkeys(("Lactobacillus", "Limosilactobacillus", "Lacticaseibacillus", "Lactiplantibacillus", "Ligilactobacillus", "Levilactobacillus", "Pediococcus", "Leuconostoc", "Weissella", "Companilactobacillus", "Latilactobacillus", "Loigolactobacillus"), "Lactobacillaceae"),
    **dict.fromkeys(("Clostridioides", "Paeniclostridium", "Paraclostridium"), "Peptostreptococcaceae"),
    **dict.fromkeys(("Clostridium",), "Clostridiaceae"),
    **dict.fromkeys(("Erysipelothrix",), "Erysipelotrichaceae"),
    **dict.fromkeys(("Mycobacterium", "Mycobacteroides", "Mycolicibacterium", "Mycolicibacter", "Mycolicibacillus"), "Mycobacteriaceae"),
    **dict.fromkeys(("Corynebacterium",), "Corynebacteriaceae"),
    **dict.fromkeys(("Nocardia", "Rhodococcus"), "Nocardiaceae"),
    **dict.fromkeys(("Bifidobacterium", "Gardnerella"), "Bifidobacteriaceae"),
    **dict.fromkeys(("Actinomyces", "Trueperella", "Arcanobacterium"), "Actinomycetaceae"),
    **dict.fromkeys(("Riemerella", "Ornithobacterium", "Elizabethkingia", "Chryseobacterium", "Empedobacter", "Myroides", "Flavobacterium", "Weeksella", "Capnocytophaga", "Bergeyella"), "Flavobacteriaceae"),
    **dict.fromkeys(("Bacteroides", "Phocaeicola"), "Bacteroidaceae"),
    **dict.fromkeys(("Prevotella",), "Prevotellaceae"),
    **dict.fromkeys(("Mycoplasmopsis", "Mycoplasmoides", "Mycoplasma", "Ureaplasma", "Metamycoplasma", "Mesomycoplasma", "Malacoplasma"), "Mycoplasmataceae"),
    **dict.fromkeys(("Brachyspira",), "Brachyspiraceae"),
    **dict.fromkeys(("Fusobacterium",), "Fusobacteriaceae"),
    **dict.fromkeys(("Candida", "Candidozyma", "Aspergillus", "Cryptococcus", "Saccharomyces", "Nakaseomyces", "Pichia", "Homo", "human", "Macaca", "Sus", "Bos", "Gallus"), "NON_BACTERIAL"),
}
# species strings that name no organism at all, and genera that are not bacteria. Both are kept in
# the output under tier "X excluded" rather than dropped, so the exclusion is visible and countable
# (the portal carries 755 "metagenome" and 299 "Homo sapiens" labels for these ten drugs).
NON_TAXON = {"mixed", "metagenome", "bacterium", "uncultured", "unidentified", "synthetic", "mixed sample"}
JUNK_GENERA = frozenset({"NON_TAXON", "Homo", "human", "Macaca", "Sus", "Bos", "Gallus", "sediment",
                         "Candida", "Candidozyma", "Aspergillus", "Cryptococcus", "Saccharomyces"})

# Each tier is what that widening ADDS, relative to the arm now training (Enterobacteriaceae).
# Moraxellaceae sits inside Pseudomonadales, so Acinetobacter enters at P. aeruginosa's own order
# rather than at class — which is why tier 3 is the one that carries the ten-drug panel.
TIER_ORDER = [
    "0 P. aeruginosa (reference)",
    "1 family: Enterobacteriaceae",
    "2 genus: other Pseudomonas",
    "3 order: rest of Enterobacterales + Moraxellaceae",
    "4 class: rest of Gammaproteobacteria",
    "5 phylum: rest of Pseudomonadota",
    "6 domain: all other bacteria",
    "X excluded: not a bacterial taxon",
]


def tier_of(species: str, genus: str, family: str, order: str, klass: str, phylum: str) -> str:
    """Which widening first brings this species in. See :data:`TIER_ORDER`."""
    if genus in JUNK_GENERA or family == "NON_BACTERIAL":
        return TIER_ORDER[7]
    if species == PA:
        return TIER_ORDER[0]
    if family == "Enterobacteriaceae":
        return TIER_ORDER[1]
    if genus == "Pseudomonas":
        return TIER_ORDER[2]
    if order in ("Enterobacterales", "Pseudomonadales"):
        return TIER_ORDER[3]
    if klass == "Gammaproteobacteria":
        return TIER_ORDER[4]
    if phylum == "Pseudomonadota":
        return TIER_ORDER[5]
    return TIER_ORDER[6]


def balanced_total(n_r: np.ndarray, n_s: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Genomes kept when a unit is thinned at random to a resistance rate of ``p`` (floor, elementwise).

    A unit with no R or no S collapses to 0 — that is the rule working, not a gap (meropenem
    *Salmonella* 14,362 -> 37 in the arm now training).
    """
    n_r, n_s, p = (np.asarray(x, dtype=float) for x in (n_r, n_s, p))
    usable = (p > 0) & (p < 1) & (n_r > 0) & (n_s > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        total = np.minimum(n_r / p, n_s / (1 - p))
    return np.where(usable, np.floor(np.nan_to_num(total, nan=0.0, posinf=0.0)), 0).astype(int)


def md5(path: Path) -> str:
    """The file's md5, so a table can be tied to the exact input it was built from."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    """Count, balance and write the four tables."""
    OUT.mkdir(parents=True, exist_ok=True)
    bp = load_breakpoints()
    tests = read_portal_tests(PHENOTYPE, drugs=DRUGS, source="cabbage_v2")
    print(f"tests for the ten drugs: {len(tests):,} rows, {tests.Sample.nunique():,} BioSamples")

    called = call_tests(tests, bp, unitless_mic_as_mg_l=True)
    y_src = called["source_call"].astype("string").str.strip().str.lower().map(SOURCE_CALL_TO_BINARY).astype(float)
    called["y_ceiling"] = called["y_test"].where(called["y_test"].notna(), y_src)
    called["is_no_bp"] = called["basis"].eq("no_breakpoint")
    called["has_route"] = called["assembly_phenotype"].notna() | called["reads_accession"].notna()
    print(called["basis"].value_counts().to_string())

    g = called.groupby(["Sample", "drug"], sort=False)
    lab = g.agg(y=("y_test", "mean"), y_ceiling=("y_ceiling", "mean"),
                n_no_bp=("is_no_bp", "sum"), has_route=("has_route", "max")).reset_index()
    lab["species"] = lab["Sample"].map(called.groupby("Sample")["species"].first())

    sp = lab["species"].astype("string").fillna("")
    genus = sp.str.split().str[0]
    lab["genus"] = genus.where(~sp.str.lower().isin(NON_TAXON), "NON_TAXON")
    lab["family"] = lab["genus"].map(_GENUS_FAMILY).fillna("UNMAPPED")
    lab["order"] = lab["family"].map(lambda f: _T.get(f, ("UNMAPPED",) * 3)[0])
    lab["class"] = lab["family"].map(lambda f: _T.get(f, ("UNMAPPED",) * 3)[1])
    lab["phylum"] = lab["family"].map(lambda f: _T.get(f, ("UNMAPPED",) * 3)[2])

    unmapped = lab[lab["family"].eq("UNMAPPED")].groupby("genus").size().sort_values(ascending=False)
    unmapped.rename("n_genome_drug_pairs").to_csv(OUT / "unmapped_genera.csv")
    print(f"unmapped genera: {len(unmapped)}  ({int(unmapped.sum()):,} genome x drug pairs)")
    print(unmapped.head(60).to_string())

    lab["tier"] = [tier_of(s, g, f, o, c, p) for s, g, f, o, c, p
                   in zip(lab.species, lab.genus, lab.family, lab["order"], lab["class"], lab.phylum,
                          strict=True)]

    rows = []
    for col, tag in (("y", "today"), ("y_ceiling", "ceiling")):
        clean = lab[lab[col].isin([0.0, 1.0])]
        agg = clean.groupby(["tier", "phylum", "class", "order", "family", "genus", "species", "drug"]).agg(
            n_labelled=(col, "size"), n_R=(col, "sum"), n_route=("has_route", "sum")).reset_index()
        agg["n_R"] = agg["n_R"].astype(int)
        agg["n_S"] = agg["n_labelled"] - agg["n_R"]
        agg["variant"] = tag
        rows.append(agg)
    out = pd.concat(rows, ignore_index=True)
    out["pct_R"] = (100 * out["n_R"] / out["n_labelled"]).round(1)
    out["target_pct_R"] = out["drug"].map(TARGET_PCT_R)
    out["n_balanced"] = balanced_total(out["n_R"].to_numpy(), out["n_S"].to_numpy(),
                                       out["target_pct_R"].to_numpy() / 100)
    # P. aeruginosa is the reference and is never thinned
    out.loc[out["species"].eq(PA), "n_balanced"] = out.loc[out["species"].eq(PA), "n_labelled"]
    out = out.sort_values(["variant", "drug", "n_balanced"], ascending=[True, True, False])
    out.to_csv(OUT / "labels_by_species_drug.csv", index=False)

    by_species = out.groupby(["variant", "tier", "phylum", "class", "order", "family", "species"]).agg(
        n_drugs=("drug", "nunique"), n_labelled=("n_labelled", "sum"), n_R=("n_R", "sum"),
        n_balanced=("n_balanced", "sum"), n_route=("n_route", "sum")).reset_index()
    by_species = by_species.sort_values(["variant", "n_balanced"], ascending=[True, False])
    by_species.to_csv(OUT / "labels_by_species.csv", index=False)

    # distinct genomes, not labels — what a build and an ESM-C pass would actually cost
    keys = ["tier", "phylum", "class", "order", "family", "genus", "species"]
    gen = []
    for col, tag in (("y", "today"), ("y_ceiling", "ceiling")):
        clean = lab[lab[col].isin([0.0, 1.0])]
        gg = clean.groupby(keys)["Sample"].nunique().rename("n_genomes").to_frame()
        gg["n_genomes_with_route"] = clean[clean["has_route"]].groupby(keys)["Sample"].nunique()
        gg = gg.fillna(0).astype(int).reset_index()
        gg["variant"] = tag
        gen.append(gg)
    genomes = pd.concat(gen, ignore_index=True).sort_values(["variant", "n_genomes"], ascending=[True, False])
    genomes.to_csv(OUT / "genomes_by_species.csv", index=False)

    by_genus = out.groupby(["variant", "tier", "phylum", "class", "order", "family", "genus"]).agg(
        n_species=("species", "nunique"), n_drugs=("drug", "nunique"), n_labelled=("n_labelled", "sum"),
        n_R=("n_R", "sum"), n_balanced=("n_balanced", "sum"), n_route=("n_route", "sum")).reset_index()
    by_genus.sort_values(["variant", "n_balanced"], ascending=[True, False]).to_csv(
        OUT / "labels_by_genus.csv", index=False)

    by_tier = out.groupby(["variant", "tier", "drug"]).agg(
        n_species=("species", "nunique"), n_labelled=("n_labelled", "sum"),
        n_balanced=("n_balanced", "sum")).reset_index()
    by_tier.to_csv(OUT / "labels_by_tier_drug.csv", index=False)

    tier_totals = genomes.groupby(["variant", "tier"])["n_genomes"].sum().rename("n_genomes")
    headline = by_tier.groupby(["variant", "tier"]).agg(
        n_labelled=("n_labelled", "sum"), n_balanced=("n_balanced", "sum")).join(tier_totals)
    # distinct species per tier — NOT the per-drug maximum, which undercounts a tier whose species
    # are tested on different drugs
    headline.insert(0, "n_species", out.groupby(["variant", "tier"])["species"].nunique())
    headline["balanced_per_genome"] = (headline.n_balanced / headline.n_genomes).round(2)
    headline.reset_index().to_csv(OUT / "labels_by_tier.csv", index=False)

    (OUT / "manifest.json").write_text(json.dumps({
        "phenotype": str(PHENOTYPE), "phenotype_md5": md5(PHENOTYPE),
        "breakpoints_md5": md5(Path(DEFAULT_BREAKPOINTS)), "drugs": DRUGS, "target_pct_R": TARGET_PCT_R,
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                  cwd=Path(bacpredict.__file__).parent).stdout.strip() or "unknown",
        "n_test_rows": int(len(called)), "n_biosamples": int(called.Sample.nunique()),
    }, indent=2))

    for tag in ("today", "ceiling"):
        t = headline.loc[tag].reindex(TIER_ORDER).dropna(how="all")
        print(f"\n=== {tag} ===\n" + t.to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
