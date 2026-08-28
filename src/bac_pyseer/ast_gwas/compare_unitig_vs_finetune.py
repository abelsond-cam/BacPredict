"""Does the unitig GWAS baseline beat the Bacformer fine-tune, across drugs?

Per-drug AUROCs have been compared before; what has not been done is asking whether the *aggregate*
pattern is real. Two summaries, and they answer different questions:

* **A global average** — the mean and median of per-drug ``unitig − fine-tune``. Sensitive to
  magnitude, but averaging AUROCs across drugs of very different difficulty and prevalence is a
  crude summary: a drug at 0.85 and one at 0.99 do not contribute comparable headroom.
* **A sign test** — how many of the drugs the unitig arm wins, against a null of 50/50. Ignores
  magnitude entirely, which is exactly why it is worth having alongside the average: it cannot be
  swung by one large delta.

**⚠ The drugs are not independent, and the sign test's p-value is therefore optimistic.** They are
scored on overlapping genome sets, their resistance phenotypes are correlated, and single
determinants drive several drugs at once — one carbapenemase moves ertapenem, imipenem and
meropenem together. The effective number of independent comparisons is smaller than the drug count,
so treat a borderline p-value as suggestive rather than decisive. The exact binomial is reported
because it is the test that was asked for; the caveat travels with it.

**No per-drug confidence intervals against the fine-tune are available here.** Fine-tune training
saves aggregate metrics only — no per-sample scores — so a paired bootstrap needs
``engine.finetune.evaluate`` re-run per drug to emit ``eval_scores.npz``. Until then this module
compares point estimates, and says so. ``bac_pyseer.ast_gwas.collect_comparison`` does the paired CI
where those files exist.

**Fine-tune numbers are read from each checkpoint's own ``results.json``**, never from a summary
panel — reading a panel is how colistin came to be quoted as 0.8072 when it is 0.9094.

Usage
-----
``python -m bac_pyseer.ast_gwas.compare_unitig_vs_finetune --ft-root <models/finetune>
--full-root <pyseer_ast/kp> --vocab-root <pyseer_ast/kp_trainval_vocab> --out-csv <path>``
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from math import comb
from pathlib import Path
from statistics import median

from bac_pyseer.ast_gwas.summarise_vocab_build import drug_dirs


def exact_binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided exact binomial p-value for ``k`` successes in ``n`` trials.

    Implemented directly rather than pulling in a dependency for one call: the point-mass method
    (sum every outcome no more likely than the observed one) is the standard definition and is exact
    for the symmetric ``p = 0.5`` case this module uses.
    """
    if n == 0:
        return float("nan")
    probs = [comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(n + 1)]
    observed = probs[k]
    # 1e-12 slack: at p=0.5 the distribution is symmetric, and exact equality of two float
    # probabilities that are mathematically identical is not guaranteed.
    return min(1.0, sum(q for q in probs if q <= observed * (1 + 1e-12)))


def read_finetune(ft_root: Path, species: str = "klebsiella_pneumoniae") -> dict[str, dict]:
    """Every fine-tune checkpoint's own ``results.json`` → ``{drug: {...}}``."""
    out: dict[str, dict] = {}
    for d in sorted(ft_root.glob(f"{species}_*")):
        results = d / "results.json"
        if not results.is_file():
            continue
        payload = json.loads(results.read_text())
        drug = re.sub(rf"^{species}_(.+?)_lr_.*$", r"\1", d.name)
        metrics = payload.get("metrics") or {}
        out[drug] = {
            "ft_auroc": metrics.get("auroc"),
            "ft_auprc": metrics.get("auprc"),
            "ft_n_evaluate": payload.get("n_evaluate") or (payload.get("split") or {}).get("n_evaluate"),
            "ft_checkpoint": d.name,
        }
    return out


def read_unitig(root: Path, drug: str, *, nested: bool) -> dict:
    """One arm's unitig ``results.json``. ``nested`` selects the rebuild's ``<drug>/<drug>/`` layout."""
    base = root / drug / drug if nested else root / drug
    results = base / "lr" / "results.json"
    if not results.is_file():
        return {}
    payload = json.loads(results.read_text())
    metrics = payload.get("metrics") or {}
    # n_evaluate lives under split{} in this schema. Read only the top level and it comes back None,
    # which then compares unequal to the FT's real count and flags every drug as a mismatch — a
    # check that is not failing but vacuous, which is worse, because it looks like it ran.
    n_eval = payload.get("n_evaluate")
    if n_eval is None:
        n_eval = (payload.get("split") or {}).get("n_evaluate")
    return {"auroc": metrics.get("auroc"), "auprc": metrics.get("auprc"), "n_evaluate": n_eval}


def build_rows(ft_root: Path, full_root: Path, vocab_root: Path) -> list[dict]:
    """One row per drug carrying the fine-tune and both unitig arms."""
    ft = read_finetune(ft_root)
    rows = []
    for drug in drug_dirs(vocab_root):
        f = read_unitig(full_root, drug, nested=False)
        t = read_unitig(vocab_root, drug, nested=True)
        if drug not in ft or not t.get("auroc"):
            continue
        row = {"drug": drug, **ft[drug],
               "unitig_full_cohort_auroc": f.get("auroc"),
               "unitig_trainval_vocab_auroc": t.get("auroc"),
               "unitig_n_evaluate": t.get("n_evaluate")}
        # Both arms must have scored the same holdout or the delta is not a comparison. This is a
        # convention across pipelines rather than an enforced join, so it is checked, not assumed.
        row["same_holdout_n"] = (row["ft_n_evaluate"] == t.get("n_evaluate"))
        for key, val in (("full_cohort", f.get("auroc")), ("trainval_vocab", t.get("auroc"))):
            row[f"delta_{key}_minus_ft"] = (val - ft[drug]["ft_auroc"]) if val is not None else None
        rows.append(row)
    return rows


def summarise(rows: list[dict], key: str, label: str) -> dict:
    """Global average and sign test for one unitig arm against the fine-tune."""
    deltas = [r[f"delta_{key}_minus_ft"] for r in rows if r.get(f"delta_{key}_minus_ft") is not None]
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    ties = len(deltas) - wins - losses
    # Ties are excluded from the sign test rather than split, which is the conventional handling and
    # the conservative one — an exact AUROC tie is evidence for neither arm.
    n = wins + losses
    return {
        "arm": label, "n_drugs": len(deltas), "mean_delta": sum(deltas) / len(deltas),
        "median_delta": median(deltas), "min_delta": min(deltas), "max_delta": max(deltas),
        "wins": wins, "losses": losses, "ties": ties,
        "binomial_p": exact_binomial_two_sided(wins, n),
    }


def run(ft_root: Path, full_root: Path, vocab_root: Path, out_csv: Path | None) -> int:
    """Print the per-drug table and both aggregate summaries."""
    rows = build_rows(ft_root, full_root, vocab_root)
    if not rows:
        raise SystemExit("no drug had both a fine-tune results.json and a trainval unitig read-out")

    hdr = f"{'drug':<30} {'BacF FT':>8} {'uni full':>9} {'uni tv':>8} {'tv−FT':>8} {'full−FT':>8}  n"
    print(hdr)
    print("-" * (len(hdr) + 4))
    for r in sorted(rows, key=lambda x: x["delta_trainval_vocab_minus_ft"] or 0):
        flag = "" if r["same_holdout_n"] else "  ⚠ holdout n differs"
        print(
            f"{r['drug']:<30} {r['ft_auroc']:>8.4f} "
            f"{r['unitig_full_cohort_auroc'] or float('nan'):>9.4f} "
            f"{r['unitig_trainval_vocab_auroc']:>8.4f} "
            f"{r['delta_trainval_vocab_minus_ft']:>+8.4f} "
            f"{r['delta_full_cohort_minus_ft'] or float('nan'):>+8.4f}  {r['unitig_n_evaluate']}{flag}"
        )

    print()
    for key, label in (("trainval_vocab", "unitig (leakage-free) vs BacFormer FT"),
                       ("full_cohort", "unitig (full-cohort vocab) vs BacFormer FT")):
        s = summarise(rows, key, label)
        print(f"{s['arm']}")
        print(f"   mean delta   {s['mean_delta']:+.4f}   median {s['median_delta']:+.4f}   "
              f"range {s['min_delta']:+.4f} to {s['max_delta']:+.4f}")
        tie_note = f" ({s['ties']} tie)" if s["ties"] else ""
        print(f"   unitig wins  {s['wins']}/{s['wins'] + s['losses']}{tie_note}"
              f"   exact binomial p = {s['binomial_p']:.4f}")
        print()

    n_mismatch = sum(1 for r in rows if not r["same_holdout_n"])
    print(f"{len(rows)} drugs. Holdout size agrees between FT and unitig on "
          f"{len(rows) - n_mismatch}/{len(rows)}.")
    print("⚠ The drugs are NOT independent — overlapping genomes, correlated phenotypes, and single")
    print("  determinants driving several drugs at once — so the binomial p is optimistic. And these")
    print("  are point estimates: no per-drug CI against the FT exists until eval_scores.npz is")
    print("  regenerated for each fine-tune.")

    if out_csv:
        cols: list[str] = []
        for r in rows:
            cols += [c for c in r if c not in cols]
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out_csv}")
    return 0


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ft-root", type=Path, required=True, help="processed/train_<org>_ast/models/finetune")
    p.add_argument("--full-root", type=Path, required=True)
    p.add_argument("--vocab-root", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    sys.exit(run(args.ft_root, args.full_root, args.vocab_root, args.out_csv))


if __name__ == "__main__":
    main()
