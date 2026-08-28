"""Pair the full-cohort and train+validate-vocabulary unitig arms, drug by drug.

The rebuild exists to answer one question: **does the unitig result survive when the vocabulary
never saw the holdout?** That is a paired question. Both arms score the *same* holdout genomes —
the ``<drug>_split.csv`` tables are untouched by the rebuild — so the two AUROCs are not independent
samples, and resampling them separately would widen the interval by pretending each model met a
different set of easy and hard genomes. :func:`~bac_pyseer.kleb_iso_source.unitig_presence_model.paired_delta_ci`
is reused rather than reimplemented, for the same reason ``collect_comparison`` uses it.

**The holdout identity is asserted, not assumed.** Both arms are supposed to resolve their holdout
through ``engine.splits.load_splits`` against the same table, but "supposed to" is exactly what
failed in the 2026-07 read-out leak. So this refuses to compare two arms whose holdout **id sets**
differ — set equality, not matching counts, because two different 282-genome holdouts have matching
counts and nothing else. Labels are checked after alignment too: the same genome must carry the same
phenotype in both arms, or the pairing is meaningless whatever the ids say.

**Direction: ``delta = full_cohort - trainval_vocab``.** Positive means the old arm scored higher,
i.e. it was flattered by a vocabulary that had seen the test genomes. That framing is deliberate —
the delta is itself a result, a measurement of how much a unitig GWAS baseline flatters itself, and
it is worth reporting whichever way it lands.

**Three things move between the arms and only one is leakage**, so the table carries all three rather
than inviting the whole delta to be read as the leak:

* *representation advantage* — the vocabulary saw holdout sequence (what the rebuild removes);
* *out-of-vocabulary penalty* — a trainval feature is monochromatic within train+validate by
  construction, so a holdout genome carrying it partially scores 0 where a full build would have
  split the feature and given partial credit. Deployment-realistic, and it penalises the new arm;
* *MAF floor* — ``MIN_SAMP`` was a flat 71 (1% of 7,080) and is now 1% of each drug's own reflist,
  i.e. 12–40. The old floor was an effective ~4% MAF cutoff while pyseer was told ``--min-af 0.01``,
  so the old run was under-powered for rare unitigs. Correcting it points the **opposite** way and
  will tend to *raise* the new arm, partially masking the leak.

``min_samples``, ``n_unitigs``, ``n_patterns`` and the Bonferroni threshold are carried for both arms
so the reader can see the floor change rather than take the delta at face value.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from bac_pyseer.kleb_iso_source.unitig_presence_model import paired_delta_ci

logger = logging.getLogger(__name__)


def load_arm(scores_npz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one arm's holdout scores → ``(sample_ids, y_true, y_prob)`` sorted by sample id.

    Sorting here is what makes the caller's pairing positional-safe: the two arms write their rows
    in whatever order ``load_splits`` yielded, and those orders need not match.
    """
    with np.load(scores_npz, allow_pickle=False) as z:
        ids = np.asarray(z["sample_ids"], dtype=str)
        y_true = np.asarray(z["y_true"]).astype(int)
        y_prob = np.asarray(z["y_prob"]).astype(float)
    order = np.argsort(ids)
    return ids[order], y_true[order], y_prob[order]


def _gwas_facts(results_json: Path, audit_json: Path | None) -> dict[str, object]:
    """Pull the confound columns for one arm — the floor, the feature count, the threshold."""
    facts: dict[str, object] = {}
    if results_json.is_file():
        payload = json.loads(results_json.read_text())
        extra = payload.get("extra") or {}
        facts["n_unitigs"] = extra.get("n_unitigs")
        facts["C"] = extra.get("C")
        summary = extra.get("gwas_summary") or {}
        for key in ("n_patterns", "n_tested", "threshold", "lambda", "genomic_inflation"):
            if key in summary:
                facts[key] = summary[key]
    if audit_json and audit_json.is_file():
        audit = json.loads(audit_json.read_text())
        facts["min_samples"] = (audit.get("reflist") or {}).get("min_samples_floor")
        facts["n_reflist"] = (audit.get("reflist") or {}).get("n_reflist")
    return facts


def compare_drug(
    drug: str,
    full_scores: Path,
    trainval_scores: Path,
    *,
    full_results: Path | None = None,
    trainval_results: Path | None = None,
    trainval_audit: Path | None = None,
    n_boot: int = 2000,
    seed: int = 1,
) -> dict[str, object]:
    """Compare one drug's two arms → a row with both AUROCs, the paired delta and its CI.

    Raises
    ------
    SystemExit
        If the two arms' holdout id sets differ, or if the same genome carries different labels in
        the two arms. Either means the arms are not scoring the same thing and no delta computed
        from them would mean anything.
    """
    ids_a, y_a, p_a = load_arm(full_scores)
    ids_b, y_b, p_b = load_arm(trainval_scores)

    set_a, set_b = set(ids_a.tolist()), set(ids_b.tolist())
    if set_a != set_b:
        only_a, only_b = sorted(set_a - set_b), sorted(set_b - set_a)
        raise SystemExit(
            f"{drug}: the two arms scored different holdout genomes — {len(only_a)} only in "
            f"full_cohort (e.g. {only_a[:3]}), {len(only_b)} only in trainval_vocab "
            f"(e.g. {only_b[:3]}). Both arms must resolve the holdout through the same "
            f"<drug>_split.csv via engine.splits.load_splits; a paired delta across different "
            f"genome sets is meaningless. Counts alone would not have caught this "
            f"({len(ids_a)} vs {len(ids_b)})."
        )
    if not np.array_equal(y_a, y_b):
        n_diff = int((y_a != y_b).sum())
        raise SystemExit(
            f"{drug}: {n_diff} genome(s) carry a different label in the two arms despite identical "
            f"ids. The label CSV or the split table diverged between runs; the pairing is invalid."
        )

    row: dict[str, object] = {
        "drug": drug,
        "n_holdout": int(len(ids_a)),
        "n_resistant": int(y_a.sum()),
        "full_cohort_auroc": float(roc_auc_score(y_a, p_a)),
        "trainval_vocab_auroc": float(roc_auc_score(y_b, p_b)),
    }
    # a = full_cohort, b = trainval_vocab, so delta > 0 means the old arm scored higher.
    row.update(paired_delta_ci(y_a, p_a, p_b, n_boot=n_boot, seed=seed))
    for prefix, results, audit in (
        ("full", full_results, None),
        ("trainval", trainval_results, trainval_audit),
    ):
        if results is not None:
            row.update({f"{prefix}_{k}": v for k, v in _gwas_facts(results, audit).items()})
    return row


def run(
    full_root: Path,
    vocab_root: Path,
    out_csv: Path | None,
    drugs: list[str] | None = None,
    *,
    arm: str = "lr",
    n_boot: int = 2000,
    seed: int = 1,
) -> int:
    """Compare every drug present in both roots → printed table, optional CSV, exit code."""
    names = sorted(drugs or [d.name for d in vocab_root.iterdir() if d.is_dir()])
    rows, skipped = [], []
    for drug in names:
        # The rebuild's per-drug OUT_DIR makes run_drug.sh's DRUG_DIR <vocab>/<drug>/<drug>.
        full_scores = full_root / drug / arm / "eval_scores.npz"
        tv_scores = vocab_root / drug / drug / arm / "eval_scores.npz"
        if not (full_scores.is_file() and tv_scores.is_file()):
            missing = "full_cohort" if not full_scores.is_file() else "trainval_vocab"
            skipped.append(f"{drug} (no {arm} scores for {missing})")
            continue
        rows.append(
            compare_drug(
                drug, full_scores, tv_scores,
                full_results=full_root / drug / arm / "results.json",
                trainval_results=vocab_root / drug / drug / arm / "results.json",
                trainval_audit=vocab_root / drug / "leakage_audit.json",
                n_boot=n_boot, seed=seed,
            )
        )

    if not rows:
        raise SystemExit(f"no drug had both arms' {arm}/eval_scores.npz — nothing to compare")

    hdr = f"{'drug':<30} {'full':>7} {'trainval':>8} {'delta':>8} {'95% CI':>19} {'sep':>4} {'floor':>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        floor = f"{r.get('trainval_min_samples', '?')} vs 71"
        print(
            f"{r['drug']:<30} {r['full_cohort_auroc']:>7.4f} {r['trainval_vocab_auroc']:>8.4f} "
            f"{r['delta']:>+8.4f} [{r['ci_lo']:>+7.4f},{r['ci_hi']:>+7.4f}] "
            f"{'yes' if r['separates_from_zero'] else 'no':>4} {floor:>11}"
        )
    for note in skipped:
        print(f"  skipped: {note}")

    deltas = np.array([r["delta"] for r in rows])
    sep = [r for r in rows if r["separates_from_zero"]]
    print(
        f"\n{len(rows)} drug(s) compared · median delta {np.median(deltas):+.4f} · "
        f"mean {deltas.mean():+.4f} · {len(sep)} separate from zero · "
        f"{int((deltas > 0).sum())} favour full_cohort, {int((deltas < 0).sum())} favour trainval_vocab"
    )
    print(
        "delta = full_cohort - trainval_vocab, so positive means the old arm scored higher. It is "
        "NOT all leakage: the MIN_SAMP rebase (71 -> per-drug) points the other way. See the module "
        "docstring before attributing any of it."
    )

    if out_csv:
        cols: list[str] = []
        for r in rows:
            cols += [c for c in r if c not in cols]
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv}")
    return 0


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full-root", type=Path, required=True, help="processed/pyseer_ast/<organism>")
    p.add_argument("--vocab-root", type=Path, required=True, help="processed/pyseer_ast/<organism>_trainval_vocab")
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--drug", action="append", dest="drugs", default=None)
    p.add_argument("--arm", default="lr", choices=["lr", "lr_dedup"], help="lr_dedup is the LD control")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.exit(run(args.full_root, args.vocab_root, args.out_csv, args.drugs,
                 arm=args.arm, n_boot=args.n_boot, seed=args.seed))


if __name__ == "__main__":
    main()
