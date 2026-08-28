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

from bac_pyseer.ast_gwas.summarise_vocab_build import drug_dirs
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


GWAS_KEYS = (
    "n_variants", "n_unique_patterns", "bonferroni_threshold", "n_significant",
    "genomic_inflation_lambda", "pheno_var",
)


def _gwas_facts(
    results_json: Path, audit_json: Path | None, summary_json: Path | None = None
) -> dict[str, object]:
    """Pull the confound columns for one arm — the floor, the feature count, the threshold.

    The GWAS summary is read from its own file when given, and only falls back to the copy embedded
    in ``results.json``. That embedded copy is unreliable: ``unitig_lr`` treats ``--gwas-summary`` as
    optional, so the read-out scripts' wrong path dropped it silently, and the full-cohort arm has it
    for ertapenem but not for ceftazidime or colistin. The summary file itself is always written.
    """
    facts: dict[str, object] = {}
    summary: dict = {}
    if summary_json is not None and summary_json.is_file():
        summary = json.loads(summary_json.read_text())
    if results_json.is_file():
        payload = json.loads(results_json.read_text())
        extra = payload.get("extra") or {}
        facts["n_unitigs"] = extra.get("n_unitigs")
        facts["C"] = extra.get("C")
        summary = summary or (extra.get("gwas_summary") or {})
    # These are the keys the summary actually uses. Guessing shorter names (n_patterns, threshold,
    # lambda) silently yields empty columns that read as a missing measurement rather than a bug.
    for key in GWAS_KEYS:
        if key in summary:
            facts[key] = summary[key]
    if audit_json and audit_json.is_file():
        audit = json.loads(audit_json.read_text())
        facts["min_samples"] = (audit.get("reflist") or {}).get("min_samples_floor")
        facts["n_reflist"] = (audit.get("reflist") or {}).get("n_reflist")
    return facts


def resolve_summary(drug_dir: Path, drug: str) -> Path | None:
    """The drug's GWAS summary, preferring the regenerated copy over the combine phase's own.

    Same precedence ``run_readout.sh`` uses for the hit table, and for the same reason: the combine
    phase writes a summary under ``gwas/``, but on any run predating the combine-phase fix it left
    ``pheno_var`` at the 0.249 default. The regenerated copy one level up was written with
    ``--phenotype-tsv`` and carries the real value. For Kp ertapenem — the pilot drug, and the only
    one with both — the two disagree on pheno_var, n_variants, n_patterns, n_significant and lambda,
    and it is the regenerated one the write-ups quote.
    """
    for candidate in (drug_dir / f"{drug}_gwas_summary.json", drug_dir / "gwas" / f"{drug}_gwas_summary.json"):
        if candidate.is_file():
            return candidate
    return None


def gwas_row(drug: str, full_summary: Path, trainval_summary: Path) -> dict[str, object]:
    """Both arms' GWAS-level facts for one drug → one row.

    This is the C4 table, and it is deliberately separate from the read-out comparison: it exists
    before any LR has run, and it is where the **MAF-floor confound** becomes visible. ``MIN_SAMP``
    fell from a flat 71 to 1% of each drug's own reflist, so the new run tests unitigs the old one
    could not — a difference that is not leakage and that pushes the delta the *other* way.

    ``pheno_var`` is carried as a control: the rebuild does not touch the phenotype, so it must be
    identical across the arms. If it is not, the two runs are not on the same labels and nothing
    downstream compares.
    """
    a = json.loads(full_summary.read_text())
    b = json.loads(trainval_summary.read_text())
    row: dict[str, object] = {"drug": drug}
    for k in GWAS_KEYS:
        row[f"full_{k}"] = a.get(k)
        row[f"trainval_{k}"] = b.get(k)
    pv_a, pv_b = a.get("pheno_var"), b.get("pheno_var")
    # A summary that never computed pheno_var says so in pheno_var_source. Treat that as "not
    # measured", not as "the labels differ" — keying on the value alone would both misdiagnose it
    # and miss the case where a stale default happens to coincide with a real value (tetracycline's
    # computed pheno_var is 0.2496, a whisker from the 0.249 default).
    uncomputed = [
        arm for arm, d in (("full_cohort", a), ("trainval_vocab", b))
        if str(d.get("pheno_var_source", "")).startswith("default")
    ]
    row["pheno_var_uncomputed"] = ",".join(uncomputed)
    if uncomputed:
        logger.warning(
            "%s: %s never computed pheno_var (source=default), so the arms cannot be compared on it. "
            "This does NOT indicate different labels; it is a stale summary field.", drug, uncomputed,
        )
    elif pv_a is not None and pv_b is not None and abs(pv_a - pv_b) > 1e-9:
        raise SystemExit(
            f"{drug}: pheno_var differs between arms ({pv_a} vs {pv_b}), and both were computed. "
            f"The rebuild changes the vocabulary, never the phenotype — so the two runs are not on "
            f"the same labels and no comparison between them is meaningful."
        )
    for k in ("n_variants", "n_unique_patterns", "n_significant"):
        if a.get(k) and b.get(k):
            row[f"ratio_{k}"] = round(b[k] / a[k], 4)
    return row


def compare_drug(
    drug: str,
    full_scores: Path,
    trainval_scores: Path,
    *,
    full_results: Path | None = None,
    trainval_results: Path | None = None,
    trainval_audit: Path | None = None,
    full_summary: Path | None = None,
    trainval_summary: Path | None = None,
    n_boot: int = 2000,
    seed: int = 1,
    max_holdout_mismatch_frac: float = 0.02,
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
    only_a, only_b = sorted(set_a - set_b), sorted(set_b - set_a)
    shared = sorted(set_a & set_b)
    # A small, one-sided difference is expected and benign: ~0.16% of the comparator's design rows
    # are genomes present in the split table but absent from assembly_refs.txt, so they never became
    # a GGCAT colour and carry an ENTIRELY ZERO feature row — the full-cohort arm scores them from
    # the intercept alone, while the rebuild's scanner cannot score them at all (it needs an
    # assembly). Pairing on the intersection is the honest response; silently intersecting is not,
    # so the counts are recorded and a large divergence is still fatal.
    worst = max(len(only_a), len(only_b)) / max(len(set_a | set_b), 1)
    if worst > max_holdout_mismatch_frac:
        raise SystemExit(
            f"{drug}: the two arms scored substantially different holdout genomes — {len(only_a)} "
            f"only in full_cohort (e.g. {only_a[:3]}), {len(only_b)} only in trainval_vocab "
            f"(e.g. {only_b[:3]}); {worst:.1%} > {max_holdout_mismatch_frac:.1%}. Both arms must "
            f"resolve the holdout through the same <drug>_split.csv via engine.splits.load_splits. "
            f"Counts alone would not have caught this ({len(ids_a)} vs {len(ids_b)})."
        )
    keep_a = np.isin(ids_a, shared)
    keep_b = np.isin(ids_b, shared)
    ids_a, y_a, p_a = ids_a[keep_a], y_a[keep_a], p_a[keep_a]
    ids_b, y_b, p_b = ids_b[keep_b], y_b[keep_b], p_b[keep_b]
    if not np.array_equal(y_a, y_b):
        n_diff = int((y_a != y_b).sum())
        raise SystemExit(
            f"{drug}: {n_diff} genome(s) carry a different label in the two arms despite identical "
            f"ids. The label CSV or the split table diverged between runs; the pairing is invalid."
        )

    row: dict[str, object] = {
        "drug": drug,
        "n_holdout": int(len(ids_a)),
        "n_only_full_cohort": len(only_a),
        "n_only_trainval_vocab": len(only_b),
        "unpaired_examples": ";".join((only_a + only_b)[:5]),
        "n_resistant": int(y_a.sum()),
        "full_cohort_auroc": float(roc_auc_score(y_a, p_a)),
        "trainval_vocab_auroc": float(roc_auc_score(y_b, p_b)),
    }
    # a = full_cohort, b = trainval_vocab, so delta > 0 means the old arm scored higher.
    row.update(paired_delta_ci(y_a, p_a, p_b, n_boot=n_boot, seed=seed))
    for prefix, results, audit, summary in (
        ("full", full_results, None, full_summary),
        ("trainval", trainval_results, trainval_audit, trainval_summary),
    ):
        if results is not None or summary is not None:
            facts = _gwas_facts(results or Path("/nonexistent"), audit, summary)
            row.update({f"{prefix}_{k}": v for k, v in facts.items()})
    return row


def run_gwas(
    full_root: Path, vocab_root: Path, out_csv: Path | None, drugs: list[str] | None = None
) -> int:
    """The C4 table: both arms' pattern counts, thresholds and lambda, per drug."""
    names = sorted(drugs or drug_dirs(vocab_root))
    rows, skipped = [], []
    for drug in names:
        a = resolve_summary(full_root / drug, drug)
        b = resolve_summary(vocab_root / drug / drug, drug)
        if a is None or b is None:
            skipped.append(f"{drug} (no summary for {'full_cohort' if a is None else 'trainval_vocab'})")
            continue
        rows.append(gwas_row(drug, a, b))
    if not rows:
        raise SystemExit("no drug had both arms' gwas summary — nothing to compare")

    hdr = (f"{'drug':<30} {'variants full':>13} {'trainval':>10} {'ratio':>6} "
           f"{'patterns full':>13} {'trainval':>10} {'ratio':>6} {'sig full':>9} {'trainval':>9} "
           f"{'lam full':>8} {'lam tv':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['drug']:<30} {r['full_n_variants'] or 0:>13,} {r['trainval_n_variants'] or 0:>10,} "
            f"{r.get('ratio_n_variants', float('nan')):>6.2f} "
            f"{r['full_n_unique_patterns'] or 0:>13,} {r['trainval_n_unique_patterns'] or 0:>10,} "
            f"{r.get('ratio_n_unique_patterns', float('nan')):>6.2f} "
            f"{r['full_n_significant'] or 0:>9,} {r['trainval_n_significant'] or 0:>9,} "
            f"{r['full_genomic_inflation_lambda'] or float('nan'):>8.3f} "
            f"{r['trainval_genomic_inflation_lambda'] or float('nan'):>7.3f}"
        )
    for note in skipped:
        print(f"  skipped: {note}")
    print(f"\n{len(rows)} drug(s) · pheno_var identical across arms in all of them (asserted)")
    print(
        "ratio = trainval / full_cohort. A pattern ratio near 1 with a variant ratio well below it "
        "means the old vocabulary carried more LD-redundant unitigs collapsing onto the same "
        "presence pattern — the Bonferroni burden is set by patterns, so it barely moves."
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
    names = sorted(drugs or drug_dirs(vocab_root))
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
                full_summary=full_root / drug / "gwas" / f"{drug}_gwas_summary.json",
                trainval_summary=vocab_root / drug / drug / "gwas" / f"{drug}_gwas_summary.json",
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
            f"{'yes' if r.get('separates_from_zero') else 'no':>4} {floor:>11}"
        )
    for note in skipped:
        print(f"  skipped: {note}")

    deltas = np.array([r["delta"] for r in rows])
    # .get: paired_delta_ci omits the key entirely when every resample was
    # single-class, which a small imbalanced holdout can produce. A bare KeyError
    # there would lose the other 21 drugs' comparison along with it.
    sep = [r for r in rows if r.get("separates_from_zero")]
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
    p.add_argument("--stage", default="readout", choices=["readout", "gwas"],
                   help="gwas = the C4 table (patterns/threshold/lambda, no LR needed); "
                        "readout = the C6 paired AUROC comparison")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.stage == "gwas":
        sys.exit(run_gwas(args.full_root, args.vocab_root, args.out_csv, args.drugs))
    sys.exit(run(args.full_root, args.vocab_root, args.out_csv, args.drugs,
                 arm=args.arm, n_boot=args.n_boot, seed=args.seed))


if __name__ == "__main__":
    main()
