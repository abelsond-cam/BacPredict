r"""Blood↔respiratory directional-concordance over the union of Bonferroni hits (the replication test).

The strongest single piece of evidence that the variant-axis invasion signal is real is that it
**replicates in direction across two independent invasive cohorts**. Both contrasts share faeces as the
control and use the invasive niche as the case, so a variant's β sign is directly comparable: ``β>0`` ⇒
the ALT allele is the invasion allele in *both*. This script makes the test **symmetric** — it takes the
**union of every Bonferroni-significant variant in blood OR respiratory**, looks each up in *both* full
``.assoc`` files, and asks how often the two β agree in sign.

Outputs (1) a per-variant table (``blood_resp_concordance_union.tsv``) and (2) summary statistics printed
to stderr: the concordant fraction, a binomial sign-test p-value (vs 0.5) computed over **independent
patterns** (one representative per perfect-LD clonal block, so clonal inflation doesn't drive the count),
and the r² of β_blood vs β_resp over the variants tested in both cohorts.

The two ``.assoc`` live on RDS (``…/pyseer_iso_source/{blood_faeces,faeces_respiratory}/…/gwas_lmm/``);
run on the HPC login node, or pass ``--blood-assoc/--resp-assoc`` to local copies. Annotation
(``display_name``/``consequence``/``lineage``) is merged from the per-contrast hit tables in the repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_RDS = Path("/home/dca36/rds/rds-floto-bacterial-4k08a2yyQLw/david/processed/pyseer_iso_source")
_DOCS = Path("src/bac_pyseer/docs/visualise")


def _load_assoc(path: Path) -> pd.DataFrame:
    """Load a pyseer ``.assoc`` → ``variant, af, beta, p`` (numeric), one row per variant."""
    a = pd.read_csv(path, sep="\t", usecols=["variant", "af", "lrt-pvalue", "beta"], dtype={"variant": str})
    a["af"] = pd.to_numeric(a["af"], errors="coerce")
    a["beta"] = pd.to_numeric(a["beta"], errors="coerce")
    a["p"] = pd.to_numeric(a["lrt-pvalue"], errors="coerce")
    return a[["variant", "af", "beta", "p"]].drop_duplicates("variant")


def _pattern_ids(df: pd.DataFrame, beta: str, p: str, prefix: str) -> pd.Series:
    """Perfect-LD clonal-block id: identical (β, p) ⇒ one pattern (mirrors pyseer_postprocess)."""
    key = prefix + "|" + df[beta].round(6).astype(str) + "|" + df[p].map(lambda x: f"{x:.3e}")
    return key.map({k: i for i, k in enumerate(key.drop_duplicates())}).astype(str)


def _annotation(blood_hits: Path, resp_hits: Path) -> pd.DataFrame:
    """Merge ``display_name``/``consequence`` per variant from the two hit tables (blood preferred)."""
    cols = ["variant", "display_name", "consequence"]
    frames = []
    for path in (blood_hits, resp_hits):
        if path.exists():
            h = pd.read_csv(path, sep="\t", dtype={"variant": str})
            frames.append(h[[c for c in cols if c in h.columns]])
    if not frames:
        return pd.DataFrame(columns=cols).set_index("variant")
    ann = pd.concat(frames, ignore_index=True).drop_duplicates("variant")
    return ann.set_index("variant")


def build(
    blood_assoc: Path, resp_assoc: Path, blood_bonf: float, resp_bonf: float,
    blood_hits: Path, resp_hits: Path, pheno_var: float,
) -> tuple[pd.DataFrame, dict]:
    """Assemble the union concordance table + summary stats."""
    blood = _load_assoc(blood_assoc).set_index("variant")
    resp = _load_assoc(resp_assoc).set_index("variant")

    sig_blood = blood.index[blood["p"] < blood_bonf]
    sig_resp = resp.index[resp["p"] < resp_bonf]
    union = sig_blood.union(sig_resp)

    df = pd.DataFrame(index=union)
    df.index.name = "variant"
    df["blood_beta"], df["blood_p"], df["blood_af"] = blood["beta"], blood["p"], blood["af"]
    df["resp_beta"], df["resp_p"], df["resp_af"] = resp["beta"], resp["p"], resp["af"]
    df["blood_sig"] = df.index.isin(sig_blood)
    df["resp_sig"] = df.index.isin(sig_resp)

    # invasion orientation: β>0 ⇒ ALT is the invasion allele (same in both contrasts: case = invasion).
    # Take the sign/af from the niche where the variant is genome-wide significant (blood preferred).
    lead_beta = np.where(df["blood_sig"], df["blood_beta"], df["resp_beta"])
    lead_af = np.where(df["blood_sig"], df["blood_af"], df["resp_af"])
    df["invasion_allele"] = np.where(lead_beta > 0, "ALT", "REF")
    df["invasive_af"] = np.where(lead_beta > 0, lead_af, 1 - lead_af)
    df["blood_ve"] = df["blood_af"] * (1 - df["blood_af"]) * df["blood_beta"] ** 2 / pheno_var * 100
    df["resp_ve"] = df["resp_af"] * (1 - df["resp_af"]) * df["resp_beta"] ** 2 / pheno_var * 100

    df["both_tested"] = df["blood_beta"].notna() & df["resp_beta"].notna()
    conc = (np.sign(df["blood_beta"]) == np.sign(df["resp_beta"]))
    df["concordant"] = pd.array(conc.to_numpy(), dtype="boolean")
    df.loc[~df["both_tested"], "concordant"] = pd.NA

    # independent-pattern id: blood pattern for blood-sig variants, else resp pattern (disjoint prefixes).
    pid = pd.Series(index=df.index, dtype=object)
    bsig = df.index[df["blood_sig"]]
    pid.loc[bsig] = _pattern_ids(df.loc[bsig], "blood_beta", "blood_p", "B").values
    ronly = df.index[df["resp_sig"] & ~df["blood_sig"]]
    pid.loc[ronly] = _pattern_ids(df.loc[ronly], "resp_beta", "resp_p", "R").values
    df["pattern_id"] = pid

    ann = _annotation(blood_hits, resp_hits)
    df = df.join(ann)

    # ---- stats ----
    bt = df[df["both_tested"]]
    r = float(np.corrcoef(bt["blood_beta"], bt["resp_beta"])[0, 1]) if len(bt) > 1 else float("nan")
    # one representative per independent pattern (max |blood β|) for the binomial sign-test
    rep = bt.assign(_ab=bt["blood_beta"].abs().fillna(bt["resp_beta"].abs())).sort_values(
        "_ab", ascending=False).drop_duplicates("pattern_id")
    k = int(rep["concordant"].sum())
    n = int(len(rep))
    binom = stats.binomtest(k, n, 0.5, alternative="greater").pvalue if n else float("nan")
    stats_d = {
        "n_union_variants": int(len(df)),
        "n_blood_sig": int(df["blood_sig"].sum()), "n_resp_sig": int(df["resp_sig"].sum()),
        "n_sig_both_niches": int((df["blood_sig"] & df["resp_sig"]).sum()),
        "n_both_tested": int(len(bt)),
        "n_concordant_variants": int(bt["concordant"].sum()), "n_variants_for_frac": int(len(bt)),
        "n_independent_patterns": n, "n_concordant_patterns": k,
        "binomial_sign_test_p": float(binom), "beta_pearson_r2": r * r if r == r else float("nan"),
        "beta_pearson_r": r,
    }
    df = df.sort_values(["blood_ve", "resp_ve"], ascending=False)
    return df, stats_d


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blood-assoc", type=Path,
                   default=_RDS / "blood_faeces/sampled_country_2_1_all/gwas_lmm/blood_vs_faeces.assoc")
    p.add_argument("--resp-assoc", type=Path,
                   default=_RDS / "faeces_respiratory/sampled_country_2_1_all/gwas_lmm/respiratory_vs_faeces.assoc")
    p.add_argument("--blood-bonf", type=float, default=1.4162259843478706e-07)
    p.add_argument("--resp-bonf", type=float, default=1.7582850391042593e-07)
    p.add_argument("--blood-hits", type=Path, default=_DOCS / "lmm_model/blood_vs_faeces_hits_annotated.tsv")
    p.add_argument("--resp-hits", type=Path,
                   default=_DOCS / "faeces_resp_lmm_model/respiratory_vs_faeces_hits_annotated.tsv")
    p.add_argument("--pheno-var", type=float, default=0.249)
    p.add_argument("--out", type=Path, default=_DOCS / "faeces_resp_lmm_model/blood_resp_concordance_union.tsv")
    args = p.parse_args(argv)

    df, s = build(args.blood_assoc, args.resp_assoc, args.blood_bonf, args.resp_bonf,
                   args.blood_hits, args.resp_hits, args.pheno_var)
    cols = ["display_name", "consequence", "invasion_allele", "invasive_af", "blood_sig", "resp_sig",
            "blood_beta", "blood_p", "blood_ve", "resp_beta", "resp_p", "resp_ve",
            "both_tested", "concordant", "pattern_id"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df[[c for c in cols if c in df.columns]].to_csv(args.out, sep="\t")
    print(f"wrote {args.out} ({len(df)} union variants)", file=sys.stderr)
    for k, v in s.items():
        print(f"  {k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
