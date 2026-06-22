"""Extract protein sequences from a Bakta/NCBI GFF3 + sibling FASTA pair.

Bacformer's `preprocess_genome_assembly` GFF code path returns annotation
metadata only (no translations). For samples with `.gff(.gff3)(.gz)` annotations
+ separate `.fna(.gz)` assemblies, we splice CDS regions from the FASTA and
translate them with the bacterial codon table.

Output shape matches the keys the downstream parquet writer expects.

Optional AMR annotation (``annotate_amr=True``)
----------------------------------------------
Bakta under-annotates acquired AMR genes for several classes, so the GFF
``gene_name`` is an unreliable way to decide which proteins are which AMR gene.
With ``annotate_amr=True`` the extractor additionally ``minimap2``-aligns
Kleborate's vendored AMR references (CARD acquired alleles + the chromosomal
QRDR / OmpK / MgrB-PmrB refs) against the *same* FASTA it is already splicing
CDS from, applies Kleborate's thresholds (acquired ≥90% identity / ≥80% query
coverage), and attaches per protein — **by direct in-builder contig+coordinate
overlap** — the authoritative allele-level label. Because minimap runs against
the exact assembly the CDS coordinates come from, the hit→CDS join is exact (no
contig-name↔``contig_idx`` reconstruction). The result rides alongside the
protein flat list under the ``amr_calls`` key (a flat list, one entry per
AMR-matched protein), so the feature is **default-off, additive, and repeatable
for any bacterium**. Non-AMR builds need no ``minimap2`` binary.
"""

from __future__ import annotations

import csv
import gzip
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from Bio import SeqIO
from Bio.Seq import Seq

logger = logging.getLogger(__name__)

_GFF_SUFFIXES = (".gff", ".gff3", ".gff.gz", ".gff3.gz")
_GBFF_SUFFIXES = (".gbff", ".gbff.gz")

# Bacterial initiator codons (table 11). BioPython's default translate() renders
# GTG/TTG as V/L because it treats every codon as an internal codon. CDSs whose
# nucleotide sequence begins with any of these should have their first amino acid
# rewritten as M to match initiator-codon biology and the BakRep/GBFF cohort,
# whose translations already start with M.
_BACTERIAL_INITIATORS = frozenset({"ATG", "GTG", "TTG", "CTG", "ATT", "ATC", "ATA"})


def is_gff_path(path: str | Path) -> bool:
    """Return True if `path` ends with a GFF/GFF3 extension (gzipped or not)."""
    s = str(path).lower()
    return s.endswith(_GFF_SUFFIXES)


def is_gbff_path(path: str | Path) -> bool:
    """Return True if `path` ends with a GenBank flat-file extension (gzipped or not)."""
    s = str(path).lower()
    return s.endswith(_GBFF_SUFFIXES)


def _open_text(path: Path):
    """Open a text file transparently, decompressing if needed."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def _load_fna(fna_path: Path) -> dict[str, Seq]:
    """Load contigs from a (gzipped) FASTA into a dict[seqid, Seq]."""
    with _open_text(fna_path) as handle:
        return {record.id: record.seq for record in SeqIO.parse(handle, "fasta")}


def _parse_gff_attributes(field: str) -> dict[str, str]:
    """Parse a GFF3 attributes column into a dict (last value wins on duplicate keys)."""
    out: dict[str, str] = {}
    for attr in field.split(";"):
        if "=" in attr:
            key, value = attr.split("=", 1)
            out[key.strip()] = value.strip()
    return out


# --------------------------------------------------------------------------- #
# AMR annotation (Kleborate-style minimap2, optional)                          #
# --------------------------------------------------------------------------- #

# Kleborate's acquired-gene call (resMinimap): a CARD allele is present at
# ≥90% nucleotide identity AND ≥80% query coverage. The chromosomal references
# (gyrA/parC, ompK35/36, mgrB/pmrB) *carry* the resistance variants, so they are
# located with a permissive identity floor — we only need gene identity, the AA
# string carries any point mutation into the embedding downstream.
_AMR_ACQUIRED_MIN_IDENT = 0.90
_AMR_ACQUIRED_MIN_COV = 0.80
_AMR_CHROMOSOMAL_MIN_IDENT = 0.80
_AMR_CHROMOSOMAL_MIN_COV = 0.80

# Reference filenames inside the vendored Kleborate KpSC AMR inputs dir.
_AMR_CARD_GLOB = "CARD_v*.fasta"            # acquired alleles (CARD)
_AMR_CHROMOSOMAL_FILES = ("QRDR_120.fasta", "OmpK.fasta", "MgrB_and_PmrB.fasta")
_AMR_CARD_LABELS_CSV = "CARD_AMR_clustered.csv"

# header (lower-case) -> (display gene/allele, Kleborate class, drug-class description)
_CHROMOSOMAL_GENE_META: dict[str, tuple[str, str, str]] = {
    "gyra": ("GyrA", "Flq", "fluoroquinolone antibiotic"),
    "parc": ("ParC", "Flq", "fluoroquinolone antibiotic"),
    "ompk35": ("OmpK35", "Bla_porin", "beta-lactam (porin)"),
    "ompk36": ("OmpK36", "Bla_porin", "beta-lactam (porin)"),
    "mgrb": ("MgrB", "Col", "colistin (peptide antibiotic)"),
    "pmrb": ("PmrB", "Col", "colistin (peptide antibiotic)"),
}


def _iter_fasta_records(path: Path):
    """Yield ``(header_no_gt, sequence_lines_joined)`` from a (gzipped) FASTA."""
    name: str | None = None
    chunks: list[str] = []
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
    if name is not None:
        yield name, "".join(chunks)


def _load_card_labels(csv_path: Path) -> dict[str, dict[str, str]]:
    """``seqID -> {class, gene(family), allele, bla_class, CARD_class}`` from CARD_AMR_clustered.csv."""
    labels: dict[str, dict[str, str]] = {}
    if not csv_path.exists():
        return labels
    with csv_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            labels[str(row["seqID"])] = {
                "class": row.get("class", "") or "",
                "gene": row.get("gene", "") or "",
                "allele": row.get("allele", "") or "",
                "bla_class": row.get("bla_class", "") or "",
                "CARD_class": row.get("CARD_class", "") or "",
            }
    return labels


def _build_amr_query_fasta(amr_ref_dir: Path, out_fasta: Path) -> int:
    """Concatenate the CARD + chromosomal refs into one query, tagging headers ``ACQ|``/``CHR|``.

    Returns the number of sequences written. The ``ACQ|``/``CHR|`` prefix routes PAF parsing
    (acquired vs chromosomal thresholds + label source) without a side table.
    """
    n = 0
    card_fastas = sorted(amr_ref_dir.glob(_AMR_CARD_GLOB))
    if not card_fastas:
        raise FileNotFoundError(f"no CARD FASTA matching {_AMR_CARD_GLOB!r} under {amr_ref_dir}")
    with out_fasta.open("w") as out:
        for name, seq in _iter_fasta_records(card_fastas[0]):
            out.write(f">ACQ|{name}\n{seq}\n")
            n += 1
        for fname in _AMR_CHROMOSOMAL_FILES:
            ref = amr_ref_dir / fname
            if not ref.exists():
                logger.warning("AMR chromosomal ref missing (skipping): %s", ref)
                continue
            for name, seq in _iter_fasta_records(ref):
                out.write(f">CHR|{name}\n{seq}\n")
                n += 1
    return n


def _resolve_card_label(name: str, card_labels: dict[str, dict[str, str]]) -> dict[str, str]:
    """Parse a CARD header (``clusterid__Family_Class__allele__seqID``) → label fields.

    Authoritative fields come from ``card_labels`` keyed by seqID; the header is the fallback.
    ``amr_class`` is refined with ``bla_class`` for beta-lactamases (e.g. ``Bla_Carb``, ``Bla_ESBL``).
    """
    parts = name.split("__")
    hdr_family = parts[1].rsplit("_", 1)[0] if len(parts) > 1 and "_" in parts[1] else (parts[1] if len(parts) > 1 else "")
    hdr_class = parts[1].rsplit("_", 1)[-1] if len(parts) > 1 and "_" in parts[1] else ""
    hdr_allele = parts[2] if len(parts) > 2 else name
    seqid = parts[3] if len(parts) > 3 else ""

    lab = card_labels.get(seqid, {})
    allele = lab.get("allele") or hdr_allele
    family = lab.get("gene") or hdr_family
    klass = lab.get("class") or hdr_class
    bla_class = lab.get("bla_class") or ""
    drug = lab.get("CARD_class") or ""
    if klass == "Bla" and bla_class and bla_class not in ("NA", "Bla"):
        klass = bla_class
    return {
        "amr_allele": allele,
        "amr_gene_family": family,
        "amr_class": klass,
        "amr_drug_classes": drug,
    }


def _parse_amr_paf(paf_path: Path, card_labels: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    """Parse a PAF of AMR refs (query) vs assembly (target) → qualifying hit dicts.

    Each dict carries the assembly contig + 0-based half-open target interval (``tname``,
    ``tstart``, ``tend``), identity/coverage, and the resolved label fields. Acquired (``ACQ|``)
    hits use the 90/80 bar; chromosomal (``CHR|``) hits the permissive 80/80 bar.
    """
    hits: list[dict[str, Any]] = []
    with paf_path.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 11:
                continue
            qname = parts[0]
            qlen, qstart, qend = int(parts[1]), int(parts[2]), int(parts[3])
            tname = parts[5]
            tstart, tend = int(parts[7]), int(parts[8])
            nmatch, alnlen = int(parts[9]), int(parts[10])
            if alnlen == 0 or qlen == 0:
                continue
            ident = nmatch / alnlen
            cov = (qend - qstart) / qlen

            source, _, raw = qname.partition("|")
            if source == "ACQ":
                if ident < _AMR_ACQUIRED_MIN_IDENT or cov < _AMR_ACQUIRED_MIN_COV:
                    continue
                label = _resolve_card_label(raw, card_labels)
                src = "acquired"
            elif source == "CHR":
                if ident < _AMR_CHROMOSOMAL_MIN_IDENT or cov < _AMR_CHROMOSOMAL_MIN_COV:
                    continue
                disp, klass, drug = _CHROMOSOMAL_GENE_META.get(
                    raw.lower(), (raw, "", "")
                )
                label = {"amr_allele": disp, "amr_gene_family": disp,
                         "amr_class": klass, "amr_drug_classes": drug}
                src = "chromosomal"
            else:
                continue

            flags = [src]
            if cov < 0.90:
                flags.append("partial")
            hits.append({
                "tname": tname, "tstart": tstart, "tend": tend,
                "amr_pct_id": round(100.0 * ident, 2), "amr_pct_cov": round(100.0 * cov, 2),
                "amr_source": src, "amr_flags": ";".join(flags),
                "_score": ident * cov, **label,
            })
    return hits


def _run_minimap2(minimap2_bin: str, target: Path, query: Path, out_paf: Path, *, threads: int) -> None:
    """Run ``minimap2 -cx asm10 --secondary=no`` (target=assembly, query=refs) → PAF."""
    cmd = [minimap2_bin, "-cx", "asm10", "--secondary=no", "-t", str(threads), str(target), str(query)]
    with out_paf.open("w") as fh:
        res = subprocess.run(cmd, stdout=fh, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0:
        raise RuntimeError(
            f"minimap2 failed (rc={res.returncode}): {res.stderr.decode('utf-8', errors='replace')}"
        )


def annotate_amr_calls(
    fna_path: str | Path,
    flat_cds: list[dict[str, Any]],
    amr_ref_dir: str | Path,
    *,
    minimap2_bin: str = "minimap2",
    threads: int = 4,
) -> list[dict[str, Any]]:
    """Align Kleborate AMR refs vs the assembly and attach allele labels to overlapping CDS.

    Parameters
    ----------
    fna_path
        The genome FASTA the CDS in ``flat_cds`` were spliced from (minimap target).
    flat_cds
        Flat per-protein records (in embedding/flat-index order), each with ``flat_index``,
        ``seqid`` (the contig name), 1-based inclusive ``start``/``end``.
    amr_ref_dir
        Directory of vendored Kleborate KpSC AMR refs (CARD FASTA + chromosomal FASTAs +
        ``CARD_AMR_clustered.csv``).
    minimap2_bin, threads
        minimap2 binary (must be on PATH / explicit) and thread count.

    Returns
    -------
    list of dict
        One entry per AMR call. Calls that overlap a CDS carry that protein's ``flat_index``
        (≥0); CARD calls with **no** overlapping CDS — the Bakta-missed-the-gene case — are
        kept with ``flat_index = -1`` so the miss can be quantified downstream. Best hit per
        CDS is kept (highest identity×coverage). Each dict has ``flat_index, seqid, tstart,
        tend, amr_allele, amr_gene_family, amr_class, amr_drug_classes, amr_pct_id,
        amr_pct_cov, amr_source, amr_flags``.
    """
    fna_path = Path(fna_path)
    amr_ref_dir = Path(amr_ref_dir)
    if shutil.which(minimap2_bin) is None and not Path(minimap2_bin).is_file():
        raise FileNotFoundError(f"minimap2 not found: {minimap2_bin!r}; provide it on PATH or pass minimap2_bin")
    card_labels = _load_card_labels(amr_ref_dir / _AMR_CARD_LABELS_CSV)

    with tempfile.TemporaryDirectory(prefix="amr_minimap_") as tmpdir:
        tmp = Path(tmpdir)
        query = tmp / "amr_refs.fa"
        n_ref = _build_amr_query_fasta(amr_ref_dir, query)
        paf = tmp / "amr.paf"
        _run_minimap2(minimap2_bin, fna_path, query, paf, threads=threads)
        hits = _parse_amr_paf(paf, card_labels)
    logger.info("AMR annotation: %d ref seqs, %d qualifying hits over %s", n_ref, len(hits), fna_path.name)
    return _assign_hits_to_cds(hits, flat_cds)


def _assign_hits_to_cds(
    hits: list[dict[str, Any]], flat_cds: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Join AMR ref hits to overlapping CDS by contig + coordinate; keep the best hit per CDS.

    ``hits`` are the qualifying PAF hits from :func:`_parse_amr_paf` (assembly = target,
    0-based half-open ``tstart``/``tend``). ``flat_cds`` are flat per-protein records with
    1-based inclusive ``start``/``end``. Acquired hits with no overlapping CDS — the
    Bakta-missed-the-gene case — are kept with ``flat_index = -1``; chromosomal misses are
    dropped (those genes are core and always called). Pure (no I/O) so it is unit-testable.
    """
    cds_by_contig: dict[str, list[dict[str, Any]]] = {}
    for rec in flat_cds:
        cds_by_contig.setdefault(rec["seqid"], []).append(rec)

    best_for_flat: dict[int, dict[str, Any]] = {}      # flat_index -> best call
    orphans: list[dict[str, Any]] = []                 # acquired hits with no overlapping CDS
    for h in hits:
        best_rec, best_ov = None, 0
        for rec in cds_by_contig.get(h["tname"], ()):
            ov = min(int(rec["end"]), h["tend"]) - max(int(rec["start"]) - 1, h["tstart"])
            if ov > best_ov:
                best_rec, best_ov = rec, ov
        call = {k: v for k, v in h.items() if k != "_score"}
        call["seqid"] = h["tname"]
        if best_rec is None:
            if h["amr_source"] == "acquired":
                call["flat_index"] = -1
                orphans.append(call)
            continue
        fi = int(best_rec["flat_index"])
        call["flat_index"] = fi
        if fi not in best_for_flat or h["_score"] > best_for_flat[fi]["_score"]:
            best_for_flat[fi] = {**call, "_score": h["_score"]}

    calls = [{k: v for k, v in c.items() if k != "_score"} for c in best_for_flat.values()]
    calls.extend(orphans)
    calls.sort(key=lambda c: (c["flat_index"] < 0, c["flat_index"]))
    if orphans:
        logger.info("AMR annotation: %d acquired call(s) had no overlapping CDS (Bakta miss)", len(orphans))
    return calls


def extract_proteins_from_gff_fna(
    gff_path: str | Path,
    fna_path: str | Path,
    *,
    translation_table: int = 11,
    keep_internal_stop: bool = False,
    annotate_amr: bool = False,
    amr_ref_dir: str | Path | None = None,
    minimap2_bin: str = "minimap2",
    amr_threads: int = 4,
) -> dict[str, Any]:
    """Extract protein sequences for one genome from a GFF + FASTA pair.

    Parameters
    ----------
    gff_path : str or Path
        Path to a GFF3 file (plain or gzipped). Bakta-annotated files are
        preferred; NCBI PGAP GFFs also work.
    fna_path : str or Path
        Path to the genome FASTA (plain or gzipped) whose contig IDs match the
        GFF `seqid` column.
    translation_table : int, default 11
        NCBI translation table (11 = bacterial/archaeal/plant plastid).
    keep_internal_stop : bool, default False
        Keep CDS whose translation has an internal stop (the ``*`` is retained in
        the sequence) instead of dropping them. Off by default. Set this only to
        reproduce a *historical* protein order — e.g. regenerating the Kp
        protein-sequence parquets so their flat index still aligns 1:1 with
        embeddings generated before the internal-stop skip was added.
    annotate_amr : bool, default False
        Also ``minimap2``-align the vendored Kleborate AMR references against
        ``fna_path`` and attach allele-level labels to overlapping CDS (see the
        module docstring). Requires ``amr_ref_dir`` and a ``minimap2`` binary;
        adds an ``amr_calls`` key to the returned dict. Default-off — non-AMR
        builds are unchanged and need no minimap2.
    amr_ref_dir : str or Path, optional
        Directory of vendored Kleborate KpSC AMR refs. Required iff
        ``annotate_amr`` is set.
    minimap2_bin : str, default "minimap2"
        minimap2 binary (on PATH or an explicit path). Only used when
        ``annotate_amr`` is set.
    amr_threads : int, default 4
        minimap2 thread count for the AMR alignment.

    Returns
    -------
    dict
        Keys mirror the structure produced by `preprocess_genome_assembly` for
        GBFF inputs (contig-grouped lists), at minimum containing
        ``protein_sequence`` (list of amino-acid strings). When ``annotate_amr``
        is set, additionally carries ``amr_calls`` — a flat list (one entry per
        AMR-matched protein) as returned by :func:`annotate_amr_calls`.
    """
    gff_path = Path(gff_path)
    fna_path = Path(fna_path)
    if annotate_amr and amr_ref_dir is None:
        raise ValueError("annotate_amr=True requires amr_ref_dir")

    contigs = _load_fna(fna_path)
    if not contigs:
        raise ValueError(f"No contigs parsed from FASTA: {fna_path}")

    # Per-contig accumulators (keyed by seqid encounter order for stable contig_idx).
    contig_order: list[str] = []
    per_contig: dict[str, dict[str, list]] = {}

    n_skipped_pseudo = 0
    n_skipped_internal_stop = 0
    n_skipped_missing_contig = 0

    with _open_text(gff_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                # Bakta appends a ##FASTA block at EOF; stop parsing features there.
                if line.startswith("##FASTA"):
                    break
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "CDS":
                continue

            seqid = parts[0]
            try:
                start = int(parts[3])
                end = int(parts[4])
            except ValueError:
                continue
            strand = parts[6]
            phase_raw = parts[7]
            try:
                phase = int(phase_raw) if phase_raw and phase_raw != "." else 0
            except ValueError:
                phase = 0
            attrs = _parse_gff_attributes(parts[8])

            if attrs.get("pseudo", "").lower() in {"true", "1"}:
                n_skipped_pseudo += 1
                continue

            contig_seq = contigs.get(seqid)
            if contig_seq is None:
                n_skipped_missing_contig += 1
                continue

            nt = contig_seq[start - 1 : end]
            if strand == "-":
                nt = nt.reverse_complement()

            # Phase column: drop this many bases from the start of the CDS to reach
            # the next codon boundary. Non-zero phase means the CDS is partial at
            # the 5' end (gene continues before this region), so there is no real
            # start codon to promote.
            if phase:
                nt = nt[phase:]
            # Trim trailing bases so the sequence is a clean multiple of 3
            # (prevents BiopythonWarning: "Partial codon").
            remainder = len(nt) % 3
            if remainder:
                nt = nt[:-remainder]
            if len(nt) < 3:
                continue

            is_partial_5p = phase != 0 or attrs.get("partial", "").lower() in {"true", "1", "10", "11"}

            protein = str(nt.translate(table=translation_table, to_stop=False))
            if protein.endswith("*"):
                protein = protein[:-1]
            if "*" in protein and not keep_internal_stop:
                n_skipped_internal_stop += 1
                continue
            if not protein:
                continue
            # Promote alternative start codons to M only when the CDS isn't
            # 5'-partial (a partial CDS does not begin at a real start codon).
            if (
                not is_partial_5p
                and len(nt) >= 3
                and str(nt[:3]).upper() in _BACTERIAL_INITIATORS
                and protein[0] != "M"
            ):
                protein = "M" + protein[1:]

            if seqid not in per_contig:
                contig_order.append(seqid)
                per_contig[seqid] = {
                    "gene_name": [],
                    "protein_name": [],
                    "start": [],
                    "end": [],
                    "protein_id": [],
                    "protein_sequence": [],
                }
            bucket = per_contig[seqid]
            locus_tag = attrs.get("locus_tag") or attrs.get("ID")
            bucket["gene_name"].append(attrs.get("gene") or locus_tag)
            bucket["protein_name"].append(locus_tag)
            bucket["start"].append(start)
            bucket["end"].append(end)
            bucket["protein_id"].append(attrs.get("protein_id"))
            bucket["protein_sequence"].append(protein)

    if n_skipped_pseudo or n_skipped_internal_stop or n_skipped_missing_contig:
        logger.info(
            "Skipped CDS records: pseudo=%d, internal_stop=%d, missing_contig=%d",
            n_skipped_pseudo,
            n_skipped_internal_stop,
            n_skipped_missing_contig,
        )

    # Flatten per-contig lists into the genome-level shape produced by
    # `preprocess_genome_assembly` for GBFF (lists-of-lists across contigs).
    contig_idx = list(range(len(contig_order)))
    out: dict[str, Any] = {
        "contig_idx": contig_idx,
        "gene_name": [per_contig[c]["gene_name"] for c in contig_order],
        "protein_name": [per_contig[c]["protein_name"] for c in contig_order],
        "start": [per_contig[c]["start"] for c in contig_order],
        "end": [per_contig[c]["end"] for c in contig_order],
        "protein_id": [per_contig[c]["protein_id"] for c in contig_order],
        "protein_sequence": [per_contig[c]["protein_sequence"] for c in contig_order],
    }

    if annotate_amr:
        # Flat per-protein records in the SAME order `locate_gene.flatten_proteins`
        # produces (contig_order, then per-contig append order) so flat_index lines
        # up 1:1 with the ESM-C embedding rows. Carry the contig name (seqid) — the
        # parquet only stores the integer contig_idx, so the overlap join must run
        # here, against the in-memory assembly the CDS were spliced from.
        flat_cds: list[dict[str, Any]] = []
        fi = 0
        for seqid in contig_order:
            bucket = per_contig[seqid]
            for k in range(len(bucket["protein_sequence"])):
                flat_cds.append({
                    "flat_index": fi, "seqid": seqid,
                    "start": bucket["start"][k], "end": bucket["end"][k],
                })
                fi += 1
        out["amr_calls"] = annotate_amr_calls(
            fna_path, flat_cds, amr_ref_dir, minimap2_bin=minimap2_bin, threads=amr_threads
        )
    return out
