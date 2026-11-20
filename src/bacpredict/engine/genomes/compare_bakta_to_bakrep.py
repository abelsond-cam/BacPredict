r"""Does our Bakta 1.9.4 reproduce BakRep's, and does our protein extractor reproduce Bakta's own proteins?

The gate before any genome BakRep lacks is annotated by us. For each roster genome, from
``<gate>/ours/<S>.bakta.{gff3,faa,json}.gz`` (our run on our contigs) and ``<gate>/bakrep/<S>.bakta.*.gz``
(BakRep's files), against the contigs our run was given:

- **G1 contigs** — BakRep's embedded ``##FASTA`` equals our contigs, name for name and base for base;
- **G2 release** — both GFF3s carry ``# Software: v1.9.4`` / ``# Database: v5.1, full``;
- **G3 gene calls** — the CDS sets (seqid, start, end, strand) are equal, with equal phase and pseudo flags;
- **G4 embedding input** — ``extract_proteins_from_gff_fna`` gives the same per-contig protein lists from our GFF3
  as from BakRep's (what ESM-C would see is identical);
- **G5 extractor** — those proteins equal Bakta's own ``.faa`` (ours and BakRep's), residue for residue.

Reported, not gated: non-CDS features, attribute differences on shared features (product names, Dbxrefs, locus
tags) and AMRFinderPlus calls — annotation text, which no embedding reads. Genomes BakRep lacks (``group`` ==
``atb_bakta_fail``) get G2 and G5 on our run only.

Writes ``<gate>/bakta_gate_summary.csv`` (one row per genome) and ``<gate>/bakta_gate_diffs.csv`` (up to 20
examples per genome and kind), prints the gate, and exits 1 if any gated check fails.

    python -m bacpredict.engine.genomes.compare_bakta_to_bakrep --gate-dir G
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from bacpredict.engine.embedding.extract_proteins_from_gff_fna import extract_proteins_from_gff_fna
from bacpredict.engine.genomes.bakta import BAKTA_DATABASE, BAKTA_SOFTWARE, gff3_release
from bacpredict.engine.genomes.build_genomes import _open, read_fasta

GFF_COLS = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
KEY = ["type", "seqid", "start", "end", "strand"]
MAX_EXAMPLES = 20


def read_gff3(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Feature lines (``GFF_COLS``) and the ``##FASTA`` section (``{first word: sequence}``) of a GFF3."""
    rows, fasta, name, seq, in_fasta = [], {}, None, [], False
    with _open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if in_fasta:
                if line.startswith(">"):
                    if name is not None:
                        fasta[name] = "".join(seq)
                    name, seq = line[1:].split()[0], []
                elif line:
                    seq.append(line.strip())
            elif line.startswith("##FASTA"):
                in_fasta = True
            elif line and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) == 9:
                    rows.append(parts)
    if name is not None:
        fasta[name] = "".join(seq)
    df = pd.DataFrame(rows, columns=GFF_COLS)
    df[["start", "end"]] = df[["start", "end"]].astype(int)
    return df, fasta


def attrs(field: str) -> dict[str, str]:
    """A GFF3 attribute column as a dict."""
    return dict(kv.split("=", 1) for kv in field.split(";") if "=" in kv)


def read_faa(path: Path) -> dict[str, str]:
    """``{locus tag: protein}`` from a Bakta ``.faa`` (header ``>LOCUS product``)."""
    return {h.split()[0]: s for h, s in read_fasta(path)}


def extracted(gff: Path, contigs: Path) -> dict:
    """The protein extractor's output: per-contig lists, exactly what the protein parquet stores."""
    return extract_proteins_from_gff_fna(gff, contigs)


def flat(ext: dict) -> list[tuple[str, str]]:
    """``[(locus tag, protein)]`` in extractor order."""
    return [(n, s) for names, seqs in zip(ext["protein_name"], ext["protein_sequence"], strict=True)
            for n, s in zip(names, seqs, strict=True)]


def amr_calls(path: Path) -> set[tuple]:
    """AMRFinderPlus calls in a Bakta JSON: ``(contig, start, stop, strand, gene, product)``."""
    with gzip.open(path, "rt") as fh:
        feats = json.load(fh)["features"]
    return {(f["contig"], f["start"], f["stop"], f["strand"], e.get("gene"), e.get("product"))
            for f in feats for e in f.get("expert", []) if "amrfinder" in str(e.get("type", "")).lower()}


def pseudo_tags(gff: pd.DataFrame) -> set[str]:
    """Locus tags of the CDS a GFF3 marks ``pseudo=True``."""
    cds = gff[(gff["type"] == "CDS") & gff["attributes"].str.contains("pseudo=True", regex=False)]
    return {attrs(a).get("locus_tag") for a in cds["attributes"]}


def faa_check(ext: dict, faa: dict[str, str], pseudo: set[str], tag: str, sample: str, diffs: list) -> dict:
    """Compare extracted proteins with a Bakta ``.faa`` by locus tag.

    Bakta 1.9.4 writes pseudogenes to its ``.faa`` (``io/fasta.py::write_faa`` filters on feature type only) while
    the extractor drops them by design, so a pseudogene absent from the extraction is expected and counted on its
    own; any other ``.faa`` protein the extractor did not produce is a failure, and so is a pseudogene it did.
    """
    f = dict(flat(ext))
    shared = (f.keys() & faa.keys()) - pseudo
    bad = sorted(k for k in shared if f[k] != faa[k])
    only_ext = sorted(f.keys() - faa.keys())
    only_faa = sorted(faa.keys() - f.keys() - pseudo)
    pseudo_extracted = sorted(pseudo & f.keys())
    for k in pseudo_extracted[:MAX_EXAMPLES]:
        diffs.append({"sample": sample, "kind": f"pseudogene_extracted_{tag}", "detail": k})
    for k in bad[:MAX_EXAMPLES]:
        a, b = f[k], faa[k]
        i = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        diffs.append({"sample": sample, "kind": f"protein_vs_faa_{tag}", "detail":
                      f"{k}: len {len(a)} vs {len(b)}, first difference at {i}: {a[i:i + 5]!r} vs {b[i:i + 5]!r}"})
    for k in only_ext[:MAX_EXAMPLES]:
        diffs.append({"sample": sample, "kind": f"extracted_not_in_faa_{tag}", "detail": k})
    for k in only_faa[:MAX_EXAMPLES]:
        diffs.append({"sample": sample, "kind": f"faa_not_extracted_{tag}", "detail": k})
    return {f"prot_{tag}_n": len(f), f"faa_{tag}_n": len(faa), f"faa_{tag}_pseudo_n": len(pseudo & faa.keys()),
            f"prot_{tag}_vs_faa_diff": len(bad), f"prot_{tag}_not_in_faa": len(only_ext) + len(pseudo_extracted),
            f"faa_{tag}_not_extracted": len(only_faa)}


def compare_one(row: pd.Series, gate: Path) -> tuple[dict, list[dict]]:
    """Every check for one roster genome."""
    s, contigs = row["Sample"], Path(row["contigs_path"])
    ours = {ext: gate / "ours" / f"{s}.bakta.{ext}.gz" for ext in ("gff3", "faa", "json")}
    rep = {ext: gate / "bakrep" / f"{s}.bakta.{ext}.gz" for ext in ("gff3", "faa", "json")}
    out: dict = {"Sample": s, "group": row["group"], "species": row["species"]}
    diffs: list[dict] = []
    run = gate / "ours" / f"{s}.bakta.run.json"
    if not (ours["gff3"].exists() and run.exists()):
        out["status"] = "no run"
        return out, diffs
    out.update({k: v for k, v in json.loads(run.read_text()).items() if k in ("seconds", "max_rss_gb", "threads")})
    out["status"] = "ok"
    our_contigs = {h.split()[0]: seq for h, seq in read_fasta(contigs)}
    out["n_contigs"] = len(our_contigs)
    out["release_ours"] = " / ".join(map(str, gff3_release(ours["gff3"])))
    g_ours, fa_ours = read_gff3(ours["gff3"])
    out["ours_fasta_equal_contigs"] = fa_ours == our_contigs
    ext_ours = extracted(ours["gff3"], contigs)
    out.update(faa_check(ext_ours, read_faa(ours["faa"]), pseudo_tags(g_ours), "ours", s, diffs))
    cds_o = g_ours[g_ours["type"] == "CDS"]
    out["cds_ours"] = len(cds_o)
    out["pseudo_ours"] = int(cds_o["attributes"].str.contains("pseudo=True", regex=False).sum())
    if not rep["gff3"].exists():
        return out, diffs

    # G1, G2
    g_rep, fa_rep = read_gff3(rep["gff3"])
    out["release_bakrep"] = " / ".join(map(str, gff3_release(rep["gff3"])))
    out["contigs_equal_bakrep"] = fa_rep == our_contigs
    if not out["contigs_equal_bakrep"]:
        only_rep, only_ours = sorted(fa_rep.keys() - our_contigs.keys()), sorted(our_contigs.keys() - fa_rep.keys())
        seq_diff = sorted(k for k in fa_rep.keys() & our_contigs.keys() if fa_rep[k] != our_contigs[k])
        diffs.append({"sample": s, "kind": "contigs", "detail": f"names only in BakRep {len(only_rep)} "
                      f"{only_rep[:3]}, only ours {len(only_ours)} {only_ours[:3]}, sequence differs {len(seq_diff)}"})

    # G3 and the reported feature comparison
    m = g_ours.merge(g_rep, on=KEY, how="outer", suffixes=("_ours", "_rep"), indicator=True)
    for side, col in (("left_only", "only_ours"), ("right_only", "only_bakrep")):
        sub = m[m["_merge"] == side]
        out[f"cds_{col}"] = int((sub["type"] == "CDS").sum())
        out[f"other_{col}"] = int((sub["type"] != "CDS").sum())
        for _, r in sub.head(MAX_EXAMPLES).iterrows():
            diffs.append({"sample": s, "kind": f"feature_{col}",
                          "detail": f"{r['type']} {r['seqid']}:{r['start']}-{r['end']}{r['strand']}"})
    both = m[m["_merge"] == "both"]
    cds_both = both[both["type"] == "CDS"]
    out["cds_bakrep"] = int((g_rep["type"] == "CDS").sum())
    out["cds_phase_diff"] = int((cds_both["phase_ours"] != cds_both["phase_rep"]).sum())
    out["cds_pseudo_diff"] = int((cds_both["attributes_ours"].str.contains("pseudo=True", regex=False)
                                  != cds_both["attributes_rep"].str.contains("pseudo=True", regex=False)).sum())
    differ = both[both["attributes_ours"] != both["attributes_rep"]]
    out["attr_diff_features"] = len(differ)
    keys: Counter = Counter()
    for _, r in differ.iterrows():
        a, b = attrs(r["attributes_ours"]), attrs(r["attributes_rep"])
        ks = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
        keys.update(ks)
        if len(diffs) < 10 * MAX_EXAMPLES:
            diffs.append({"sample": s, "kind": "attributes", "detail": f"{r['type']} {r['seqid']}:{r['start']}: "
                          + "; ".join(f"{k}: {a.get(k)!r} vs {b.get(k)!r}" for k in ks)[:400]})
    out["attr_diff_keys"] = ";".join(f"{k}={n}" for k, n in keys.most_common(8))
    lt_o = cds_both["attributes_ours"].map(lambda x: attrs(x).get("locus_tag"))
    lt_r = cds_both["attributes_rep"].map(lambda x: attrs(x).get("locus_tag"))
    out["locus_tag_diff"] = int((lt_o != lt_r).sum())

    # G4, G5
    ext_rep = extracted(rep["gff3"], contigs)
    out.update(faa_check(ext_rep, read_faa(rep["faa"]), pseudo_tags(g_rep), "bakrep", s, diffs))
    out["embedding_input_equal"] = ext_ours["protein_sequence"] == ext_rep["protein_sequence"]
    if not out["embedding_input_equal"]:
        a, b = [p for _, p in flat(ext_ours)], [p for _, p in flat(ext_rep)]
        diffs.append({"sample": s, "kind": "embedding_input", "detail": f"{len(a)} vs {len(b)} proteins, "
                      f"{len(set(a) ^ set(b))} distinct sequences in only one"})

    # AMRFinderPlus (reported)
    if ours["json"].exists() and rep["json"].exists():
        a, b = amr_calls(ours["json"]), amr_calls(rep["json"])
        out.update(amr_ours=len(a), amr_bakrep=len(b), amr_diff=len(a ^ b))
        for x in sorted(a - b, key=str)[:MAX_EXAMPLES]:
            diffs.append({"sample": s, "kind": "amr_only_ours", "detail": str(x)})
        for x in sorted(b - a, key=str)[:MAX_EXAMPLES]:
            diffs.append({"sample": s, "kind": "amr_only_bakrep", "detail": str(x)})
    return out, diffs


def gate(summary: pd.DataFrame) -> list[tuple[str, int, int]]:
    """``(check, n failing, n checked)`` for each gated check."""
    held = summary[summary["group"] != "atb_bakta_fail"]
    record = f"{BAKTA_SOFTWARE} / {BAKTA_DATABASE}"
    checks = [("run finished", summary["status"] != "ok", summary)]
    ok = summary[summary["status"] == "ok"]
    ok_held = held[held["status"] == "ok"]
    checks += [
        ("G1 BakRep contigs = ours", ok_held["contigs_equal_bakrep"] != True, ok_held),  # noqa: E712 (NaN-safe)
        ("G2 release of record (ours)", ok["release_ours"] != record, ok),
        ("G2 release of record (BakRep)", ok_held["release_bakrep"] != record, ok_held),
        ("G3 CDS sets equal", (ok_held["cds_only_ours"] + ok_held["cds_only_bakrep"]) > 0, ok_held),
        ("G3 CDS phase + pseudo equal", (ok_held["cds_phase_diff"] + ok_held["cds_pseudo_diff"]) > 0, ok_held),
        ("G4 embedding input equal", ok_held["embedding_input_equal"] != True, ok_held),  # noqa: E712
        ("G5 extractor = Bakta .faa (ours)",
         ok[["prot_ours_vs_faa_diff", "prot_ours_not_in_faa", "faa_ours_not_extracted"]].sum(axis=1) > 0, ok),
        ("G5 extractor = Bakta .faa (BakRep)",
         ok_held[["prot_bakrep_vs_faa_diff", "prot_bakrep_not_in_faa", "faa_bakrep_not_extracted"]].sum(axis=1) > 0,
         ok_held),
    ]
    return [(name, int(mask.sum()), len(frame)) for name, mask, frame in checks]


def main() -> None:
    """Compare every roster genome, write the summary and diffs, print the gate."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gate-dir", type=Path, required=True, help="holds roster.csv, ours/, bakrep/")
    a = ap.parse_args()
    roster = pd.read_csv(a.gate_dir / "roster.csv", dtype=str)
    rows, diffs = [], []
    for _, r in roster.iterrows():
        o, d = compare_one(r, a.gate_dir)
        rows.append(o)
        diffs += d
    summary = pd.DataFrame(rows)
    summary.to_csv(a.gate_dir / "bakta_gate_summary.csv", index=False)
    pd.DataFrame(diffs, columns=["sample", "kind", "detail"]).to_csv(a.gate_dir / "bakta_gate_diffs.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60, "display.max_colwidth", 60):
        print(summary.to_string(index=False))
    print()
    failed = False
    for name, n_bad, n in gate(summary):
        failed |= n_bad > 0
        print(f"{'FAIL' if n_bad else 'pass'}  {name}: {n - n_bad}/{n}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
