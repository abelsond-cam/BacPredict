"""Measure what an incomplete profile table costs in *sublineage* terms.

Downloaded without authentication, the scheme's profile table stops at 2024-12-31, and for this
cohort roughly half the exactly-matching profiles fall after that date. That sounds fatal, but it
only matters if losing the exact profile also loses the **sublineage** — LIN codes are hierarchical,
so a near neighbour often carries the same one.

This answers that empirically and without running MiST: for archived calls whose ``scgST`` is absent
from the table, find the profile with the most matching loci — which is what MiST's own
``ProfileQuery`` selects — and compare its ``Sublineage`` with the one the archived call recorded
against the complete database.

The output is the evidence for whether authenticated access is required or merely tidier.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

KEY_COLUMN = "scgST"
METADATA_COLUMNS = ("LINcode", "Phylogroup", "Sublineage", "Clonal group")


def load_profiles(path: Path) -> tuple[list[str], np.ndarray, list[dict[str, str]], dict[str, int]]:
    """Load the profile table into an integer matrix plus per-row metadata.

    Returns ``(locus_columns, matrix, metadata_rows, vocabulary)``. Alleles are interned to ints so
    that matching is a vectorised equality count rather than string comparison.
    """
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        loci = [c for c in header if c != KEY_COLUMN and c not in METADATA_COLUMNS]
        idx = {c: i for i, c in enumerate(header)}
        vocab: dict[str, int] = {}
        rows, meta = [], []
        for row in reader:
            if not row:
                continue
            encoded = np.empty(len(loci), dtype=np.int32)
            for j, locus in enumerate(loci):
                value = row[idx[locus]]
                encoded[j] = vocab.setdefault(value, len(vocab))
            rows.append(encoded)
            meta.append({c: row[idx[c]] for c in (KEY_COLUMN, *METADATA_COLUMNS) if c in idx})
    matrix = np.vstack(rows)
    logger.info("%s: %d profiles x %d loci (%d distinct allele tokens)", path, *matrix.shape, len(vocab))
    return loci, matrix, meta, vocab


def archived_profile(data: dict) -> dict | None:
    """Return the matched reference profile from a MiST result, across output-format versions.

    MiST <= 1.2 wrote a single ``profile`` object; 1.3 writes a ``profiles`` list so that multiple
    equally good STs can be reported. A parser that knows only one of these silently returns nothing
    for the other, which looks exactly like a genome that failed to type.
    """
    if "profile" in data and data["profile"]:
        return data["profile"]
    profiles = data.get("profiles") or []
    return profiles[0] if profiles else None


def run(*, profiles_tsv: Path, archived_json_dir: Path, out_json: Path, sample_size: int, seed: int) -> None:
    """Score nearest-profile sublineage agreement for archived calls missing from the table."""
    loci, matrix, meta, vocab = load_profiles(profiles_tsv)
    known_st = {m[KEY_COLUMN] for m in meta}
    locus_index = {locus: j for j, locus in enumerate(loci)}

    files = sorted(archived_json_dir.glob("*.json"))
    random.Random(seed).shuffle(files)
    logger.info("%d archived results available", len(files))

    n_exact = n_absent = 0
    agree = disagree = 0
    ties = 0
    examples: list[dict] = []
    dist_when_disagree: list[int] = []
    dist_when_agree: list[int] = []
    scored: list[tuple[int, bool]] = []

    for path in files:
        if n_absent >= sample_size:
            break
        profile = archived_profile(json.loads(path.read_text()))
        if not profile:
            continue
        md = dict(profile.get("metadata") or [])
        scgst, truth = md.get("scgST"), md.get("Sublineage")
        if not scgst or not truth:
            continue
        if scgst in known_st:
            n_exact += 1
            continue
        n_absent += 1

        query = np.full(len(loci), -1, dtype=np.int32)
        for locus, allele in (profile.get("alleles") or {}).items():
            j = locus_index.get(locus)
            if j is not None:
                query[j] = vocab.get(str(allele), -1)
        matches = (matrix == query).sum(axis=1)
        best = int(matches.max())
        winners = np.flatnonzero(matches == best)
        if len(winners) > 1:
            ties += 1
        predicted = meta[int(winners[0])].get("Sublineage", "")
        n_mismatched_loci = len(loci) - best
        scored.append((n_mismatched_loci, predicted == truth))
        if predicted == truth:
            agree += 1
            dist_when_agree.append(n_mismatched_loci)
        else:
            disagree += 1
            dist_when_disagree.append(n_mismatched_loci)
            if len(examples) < 10:
                examples.append({
                    "file": path.name, "scgST": scgst, "truth": truth, "predicted": predicted,
                    "loci_mismatched": n_mismatched_loci, "n_tied": len(winners),
                })

    n = agree + disagree
    # BIGSdb declares max_missing=30 for this scheme's LIN codes, so 30 is a principled cut rather
    # than a tuned one. The others bracket it.
    by_threshold = []
    for cut in (0, 5, 10, 30, 60, 629):
        kept = [ok for miss, ok in scored if miss <= cut]
        by_threshold.append({
            "max_loci_mismatched": cut,
            "n_retained": len(kept),
            "pct_of_evaluated_retained": round(100 * len(kept) / n, 1) if n else None,
            "n_agree": sum(kept),
            "pct_agree": round(100 * sum(kept) / len(kept), 2) if kept else None,
        })

    result = {
        "profiles_tsv": str(profiles_tsv),
        "archived_json_dir": str(archived_json_dir),
        "seed": seed,
        "n_scanned_with_exact_profile": n_exact,
        "n_evaluated_without_exact_profile": n,
        "n_sublineage_agree": agree,
        "n_sublineage_disagree": disagree,
        "pct_sublineage_agree": round(100 * agree / n, 2) if n else None,
        "n_ties": ties,
        "median_loci_mismatched_when_agree": int(np.median(dist_when_agree)) if dist_when_agree else None,
        "median_loci_mismatched_when_disagree": int(np.median(dist_when_disagree)) if dist_when_disagree else None,
        "agreement_by_mismatch_threshold": by_threshold,
        "disagreement_examples": examples,
        "top_disagreements": Counter(
            f"{e['truth']} -> {e['predicted']}" for e in examples
        ).most_common(5),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")

    logger.info("scanned: %d had an exact profile, %d did not", n_exact, n)
    if n:
        logger.info("of those without one, nearest-profile sublineage agreed %d/%d (%.1f%%)",
                    agree, n, 100 * agree / n)
        for row in by_threshold:
            logger.info("  <=%3d loci mismatched: %3d retained (%5s%%), %s%% agree",
                        row["max_loci_mismatched"], row["n_retained"],
                        row["pct_of_evaluated_retained"], row["pct_agree"])
    logger.info("wrote %s", out_json)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profiles-tsv", type=Path, required=True)
    p.add_argument("--archived-json-dir", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--sample-size", type=int, default=200, help="Archived calls lacking an exact profile to score.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    run(profiles_tsv=args.profiles_tsv, archived_json_dir=args.archived_json_dir,
        out_json=args.out_json, sample_size=args.sample_size, seed=args.seed)


if __name__ == "__main__":
    main()
