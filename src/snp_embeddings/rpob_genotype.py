"""Call the rpoB RRDR genotype per isolate, straight from the assembled protein.

Step 1 reads the resistance allele from the *same* amino-acid sequence ESM-C
embedded — the rpoB protein in each sample's ``*_protein_sequences.parquet`` —
not from a variant caller. For every isolate we:

1. locate rpoB (:func:`snp_embeddings.locate_gene.locate_gene`),
2. globally align it to the H37Rv reference (UniProt P9WGY9, bundled reference),
3. read off the observed amino acid at each RRDR codon (Mtb numbering).

Genotype provenance & ground truth
----------------------------------
**Reference.** *M. tuberculosis* H37Rv — genome NCBI RefSeq ``NC_000962.3``
(GenBank ``AL123456.3``); rpoB = locus ``Rv0667``; protein UniProt ``P9WGY9``
(1,178 aa), downloaded from UniProt and bundled as
``reference_gene/rpoB_H37Rv.faa``. Resistance framing = WHO 2nd-edition
catalogue (the same catalogue TB-Profiler's ``tbdb`` uses).

**Numbering.** The standard *M. tuberculosis* RRDR codon numbers (D435, S441,
H445, S450 ...) are **+6 vs** UniProt P9WGY9 positions. Rather than hard-code the
offset we anchor on the conserved RRDR core motif ``DQNNPLSGLTHKRR`` — whose
leading ``D`` is codon 435 — and **assert** the wild-type residue at every panel
codon. Swap in a different reference and a wrong offset fails loudly.

**Locating rpoB — no minimap.** We use the existing **Bakta** annotation already
in each ``{Sample}_protein_sequences.parquet`` (``gene=rpoB`` CDS, table-11
translation) — i.e. the *exact* rpoB protein ESM-C embedded. This is the crux:
Step 1 (genotype), Steps 2/2b (embeddings) and Step 3a (LLR) are all derived from
the *same molecule*, so the ceiling-vs-embedding comparison is internally
consistent. The assembly→H37Rv mapping is **not** re-done by us (TB-Profiler does
its own mapping in the fast-follow validation track).

**rpoB-copy QC.** :func:`build_genotype_table` keeps only genomes with **exactly
one** annotated rpoB. Genomes with **0** rpoB hits and those with **>1** copies
are counted, printed to the terminal, written to ``rpob_copy_qc.log`` (Sample IDs
+ counts), and **excluded** from the test.

**Aligning + calling the allele.** Global pairwise alignment (Biopython
``PairwiseAligner``, BLOSUM62) of the annotated rpoB protein to the H37Rv
reference; RRDR codons read off the aligned columns; WT identity asserted at each
panel codon.

**Why not TB-Profiler as primary.** TB-Profiler genotypes the *assembly* (a
separate derivation), whereas this diagnostic needs the allele of the *exact
embedded protein* — so sequence-derived is primary for internal consistency.
TB-Profiler ``--fasta`` (``bioconda::tb-profiler``; repos ``jodyphelan/TBProfiler``
+ ``jodyphelan/tbdb``) is a planned fast-follow to validate these calls
(concordance %), supply lineage, and give the WHO-catalogue calls reviewers
expect — not blocking Steps 1–3.

The phenotype label is the ``rifampin`` column (US spelling) of the TB
``binary_ast.csv``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices

from snp_embeddings.locate_gene import flatten_proteins

logger = logging.getLogger(__name__)

REFERENCE_RPOB_H37RV = Path(__file__).parent / "reference_gene" / "rpoB_H37Rv.faa"

# Conserved RRDR core; the leading D is Mtb codon 435. Used only on the (WT)
# reference to anchor the numbering — never on samples, which may be mutated here.
_RRDR_ANCHOR_MOTIF = "DQNNPLSGLTHKRR"
_ANCHOR_FIRST_CODON = 435

# RRDR window reported per isolate (Mtb codon numbering, inclusive).
RRDR_FIRST_CODON = 426
RRDR_LAST_CODON = 452

# Canonical high-confidence RIF-R panel: (Mtb codon, wild-type AA, resistant AA).
# Wild-type residues are asserted against the reference at load time.
RRDR_PANEL = (
    (435, "D", "V"),
    (441, "S", "L"),
    (445, "H", "Y"),
    (450, "S", "L"),
)

RIFAMPIN_COLUMN = "rifampin"


def load_reference(reference: str | Path = REFERENCE_RPOB_H37RV) -> str:
    """Load the H37Rv rpoB reference amino-acid sequence (UniProt P9WGY9)."""
    record = next(SeqIO.parse(str(reference), "fasta"))
    return str(record.seq)


def reference_codon_index(reference: str) -> int:
    """Return the 0-based reference index of Mtb codon 435 via the anchor motif.

    Raises
    ------
    ValueError
        If the anchor motif is absent or not unique in the reference.
    """
    first = reference.find(_RRDR_ANCHOR_MOTIF)
    if first == -1:
        raise ValueError(f"RRDR anchor motif {_RRDR_ANCHOR_MOTIF!r} not found in reference")
    if reference.find(_RRDR_ANCHOR_MOTIF, first + 1) != -1:
        raise ValueError(f"RRDR anchor motif {_RRDR_ANCHOR_MOTIF!r} is not unique in reference")
    return first


def ref_index_for_codon(reference: str, codon: int) -> int:
    """0-based reference string index for an Mtb codon number."""
    anchor0 = reference_codon_index(reference)
    return anchor0 + (codon - _ANCHOR_FIRST_CODON)


def assert_reference_panel(reference: str) -> None:
    """Assert each panel codon carries its expected wild-type residue in the reference."""
    for codon, wt, _alt in RRDR_PANEL:
        idx = ref_index_for_codon(reference, codon)
        if not 0 <= idx < len(reference):
            raise ValueError(f"codon {codon} maps outside the reference (idx={idx})")
        if reference[idx] != wt:
            raise ValueError(
                f"reference residue at Mtb codon {codon} is {reference[idx]!r}, expected wild-type {wt!r}"
            )


def _build_aligner() -> PairwiseAligner:
    """Global protein aligner (BLOSUM62) tuned for near-identical rpoB sequences."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    return aligner


def _ref_to_query_index(alignment, ref_idx: int) -> int | None:
    """Map a reference (target) index to the aligned query index, or None if gapped."""
    target_blocks, query_blocks = alignment.aligned
    for (t_start, t_end), (q_start, _q_end) in zip(target_blocks, query_blocks, strict=False):
        if t_start <= ref_idx < t_end:
            return int(q_start + (ref_idx - t_start))
    return None


def genotype_rrdr(
    sample_seq: str,
    reference: str,
    aligner: PairwiseAligner | None = None,
    *,
    first_codon: int = RRDR_FIRST_CODON,
    last_codon: int = RRDR_LAST_CODON,
) -> dict[int, str]:
    """Read the observed amino acid at each RRDR codon for one sample.

    Parameters
    ----------
    sample_seq : str
        The sample's rpoB amino-acid sequence.
    reference : str
        H37Rv rpoB reference sequence.
    aligner : PairwiseAligner, optional
        Reused aligner; one is built if not supplied.
    first_codon, last_codon : int
        Inclusive RRDR window in Mtb codon numbering.

    Returns
    -------
    dict[int, str]
        ``{codon: observed_amino_acid}``. A codon deleted in the sample (aligned
        to a gap) maps to ``"-"``.
    """
    aligner = aligner or _build_aligner()
    alignment = aligner.align(reference, sample_seq)[0]
    observed: dict[int, str] = {}
    for codon in range(first_codon, last_codon + 1):
        ref_idx = ref_index_for_codon(reference, codon)
        q_idx = _ref_to_query_index(alignment, ref_idx)
        observed[codon] = sample_seq[q_idx] if q_idx is not None else "-"
    return observed


def sample_codon_positions(
    sample_seq: str,
    reference: str,
    codons: list[int],
    aligner: PairwiseAligner | None = None,
) -> dict[int, int | None]:
    """Map Mtb codon numbers to 0-based positions in the *sample* sequence.

    Used by the masked-marginal predictor, which must mask each panel codon in
    the sample's own rpoB sequence (the string ESM-C embedded).

    Returns
    -------
    dict[int, int | None]
        ``{codon: sample_index}``; ``None`` where the codon aligns to a gap.
    """
    aligner = aligner or _build_aligner()
    alignment = aligner.align(reference, sample_seq)[0]
    return {codon: _ref_to_query_index(alignment, ref_index_for_codon(reference, codon)) for codon in codons}


def build_genotype_table(
    sample_ids: list[str],
    parquet_dir: str | Path,
    reference: str | None = None,
    *,
    parquet_suffix: str = "_protein_sequences.parquet",
    qc_log_path: str | Path = "rpob_copy_qc.log",
) -> pd.DataFrame:
    """Build the per-isolate RRDR genotype table, keeping single-copy rpoB genomes only.

    Each TB genome should carry exactly one annotated rpoB. Genomes with **0**
    rpoB hits and those with **>1** copies are counted, printed to the terminal,
    written to ``qc_log_path`` (Sample IDs + copy counts), and **excluded** —
    only the single-copy genomes are genotyped and returned. Genomes with no
    parquet on disk are counted separately (also excluded).

    Parameters
    ----------
    sample_ids : list of str
        Samples to genotype.
    parquet_dir : str or Path
        Directory of ``{sample_id}{parquet_suffix}`` files.
    reference : str, optional
        H37Rv rpoB reference; loaded from the bundled reference if not supplied.
    parquet_suffix : str
        Filename suffix for the protein-sequence parquets.
    qc_log_path : str or Path
        Where to write the excluded-genome QC log (0-copy and >1-copy Sample IDs).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``Sample`` (single-copy rpoB genomes only). Columns:
        ``rpob_flat_index`` (int, the embedding index for predictor 2),
        ``n_proteins`` (flat protein count — guards the embedding-store row count
        against the parquet), ``rpob_sequence`` (str), one column per RRDR codon
        named ``codon_{n}`` holding the observed amino acid, and
        ``n_rrdr_substitutions`` (count of codons differing from wild-type).
    """
    parquet_dir = Path(parquet_dir)
    reference = reference if reference is not None else load_reference()
    assert_reference_panel(reference)
    aligner = _build_aligner()

    codons = list(range(RRDR_FIRST_CODON, RRDR_LAST_CODON + 1))
    wt_by_codon = {codon: reference[ref_index_for_codon(reference, codon)] for codon in codons}

    import gc
    import resource

    import pyarrow

    # Only the columns the genotype + flat-index recovery need — reading the full
    # nested schema (start/end/protein_id/protein_name) ~triples per-read memory.
    needed_cols = ["contig_idx", "gene_name", "protein_sequence"]

    rows: list[dict] = []
    no_parquet: list[str] = []          # sample absent on disk
    zero_copy: list[str] = []           # parquet present, no rpoB annotation
    multi_copy: list[tuple[str, int]] = []  # parquet present, >1 rpoB annotations
    for i, sample_id in enumerate(sample_ids):
        parquet_path = parquet_dir / f"{sample_id}{parquet_suffix}"
        if not parquet_path.exists():
            no_parquet.append(sample_id)
            continue
        # Read + flatten once so we capture the rpoB hit and the total protein count
        # (the latter guards the embedding-store row count in predictor 2).
        df_p = pd.read_parquet(parquet_path, columns=needed_cols)
        records = flatten_proteins(df_p)
        rpob = [r for r in records if r["gene_name"] is not None and str(r["gene_name"]).lower() == "rpob"]
        if len(rpob) == 0:
            zero_copy.append(sample_id)
        elif len(rpob) > 1:
            multi_copy.append((sample_id, len(rpob)))
        else:
            hit = rpob[0]
            observed = genotype_rrdr(hit["protein_sequence"], reference, aligner)
            row = {
                "Sample": sample_id,
                "rpob_flat_index": hit["flat_index"],
                "n_proteins": len(records),
                "rpob_sequence": hit["protein_sequence"],
            }
            n_subs = 0
            for codon in codons:
                aa = observed[codon]
                row[f"codon_{codon}"] = aa
                if aa not in ("-", wt_by_codon[codon]):
                    n_subs += 1
            row["n_rrdr_substitutions"] = n_subs
            rows.append(row)
        del df_p, records
        # pyarrow holds freed read buffers in its own pool; over ~38k reads that
        # climbs into tens of GB. Return it to the OS periodically.
        if (i + 1) % 2000 == 0:
            gc.collect()
            pyarrow.default_memory_pool().release_unused()
            rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
            logger.info("genotyped %d/%d (peak RSS %.1f GB)", i + 1, len(sample_ids), rss_gb)

    _report_rpob_copy_qc(
        total=len(sample_ids),
        kept=len(rows),
        no_parquet=no_parquet,
        zero_copy=zero_copy,
        multi_copy=multi_copy,
        qc_log_path=qc_log_path,
    )

    return pd.DataFrame(rows).set_index("Sample")


def _report_rpob_copy_qc(
    *,
    total: int,
    kept: int,
    no_parquet: list[str],
    zero_copy: list[str],
    multi_copy: list[tuple[str, int]],
    qc_log_path: str | Path,
) -> None:
    """Print the rpoB-copy QC summary and write the excluded Sample IDs to a log."""
    summary = (
        f"rpoB-copy QC: {total} samples requested | {kept} kept (exactly one rpoB) | "
        f"excluded: {len(no_parquet)} no-parquet, {len(zero_copy)} zero-copy, "
        f"{len(multi_copy)} multi-copy (>1 rpoB)"
    )
    logger.warning(summary)
    print(summary, flush=True)

    qc_log_path = Path(qc_log_path)
    with qc_log_path.open("w") as fh:
        fh.write(summary + "\n\n")
        fh.write(f"# no_parquet ({len(no_parquet)}): sample absent from parquet dir\n")
        for s in no_parquet:
            fh.write(f"no_parquet\t{s}\n")
        fh.write(f"\n# zero_copy ({len(zero_copy)}): parquet present, no rpoB annotation\n")
        for s in zero_copy:
            fh.write(f"zero_copy\t{s}\n")
        fh.write(f"\n# multi_copy ({len(multi_copy)}): parquet present, >1 rpoB annotation\n")
        for s, n in multi_copy:
            fh.write(f"multi_copy\t{s}\t{n}\n")
    logger.info("rpoB-copy QC log written to %s", qc_log_path)


def join_rifampin_label(
    genotype: pd.DataFrame,
    ast_csv: str | Path,
    *,
    sample_column: str = "Sample",
    label_column: str = RIFAMPIN_COLUMN,
) -> pd.DataFrame:
    """Join the binary ``rifampin`` AST label onto the genotype table.

    Parameters
    ----------
    genotype : pandas.DataFrame
        Output of :func:`build_genotype_table` (indexed by Sample).
    ast_csv : str or Path
        TB ``binary_ast.csv`` with a Sample column and a ``rifampin`` column.
    sample_column : str
        Sample-id column name in the AST CSV.
    label_column : str
        Phenotype column (default ``rifampin``).

    Returns
    -------
    pandas.DataFrame
        ``genotype`` with a ``rifampin`` column added; rows with no label are
        kept (``NaN``) so callers decide how to drop them.
    """
    ast = pd.read_csv(ast_csv)
    if sample_column not in ast.columns:
        raise ValueError(f"AST CSV missing sample column {sample_column!r}; has {list(ast.columns)[:10]}")
    if label_column not in ast.columns:
        raise ValueError(f"AST CSV missing label column {label_column!r}; has {list(ast.columns)[:20]}")
    labels = ast[[sample_column, label_column]].drop_duplicates(subset=sample_column).set_index(sample_column)
    return genotype.join(labels[label_column])
