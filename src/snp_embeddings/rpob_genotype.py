"""Call the rpoB RRDR genotype per isolate, straight from the assembled protein.

Stage 1.1 reads the resistance allele from the *same* amino-acid sequence ESM-C
embedded — the rpoB protein in each sample's ``*_protein_sequences.parquet`` —
not from a variant caller. For every isolate we:

1. locate rpoB (:func:`snp_embeddings.locate_gene.locate_gene`),
2. globally align it to the H37Rv reference (UniProt P9WGY9, bundled fixture),
3. read off the observed amino acid at each RRDR codon (Mtb numbering).

**Numbering.** The standard *M. tuberculosis* RRDR codon numbers (D435, S441,
H445, S450 ...) are offset from UniProt P9WGY9 positions. Rather than hard-code
the offset we anchor on the conserved core motif ``DQNNPLSGLTHKRR`` — whose
leading ``D`` is codon 435 — and **assert** the wild-type residue at every panel
codon. Swap in a different reference and a wrong offset fails loudly.

The phenotype label is the ``rifampin`` column (US spelling) of the TB
``binary_ast.csv``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Align import PairwiseAligner, substitution_matrices

from snp_embeddings.locate_gene import GeneHit, locate_gene

logger = logging.getLogger(__name__)

FIXTURE_RPOB_H37RV = Path(__file__).parent / "fixtures" / "rpoB_H37Rv.faa"

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


def load_reference(fixture: str | Path = FIXTURE_RPOB_H37RV) -> str:
    """Load the H37Rv rpoB reference amino-acid sequence from the fixture FASTA."""
    record = next(SeqIO.parse(str(fixture), "fasta"))
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


def _rpob_hit(parquet_path: Path) -> GeneHit | None:
    """Return the single best rpoB hit for a sample parquet, or None."""
    hits = locate_gene(parquet_path, "rpoB")
    if not hits:
        return None
    # Single-copy gene; if duplicated, take the longest sequence (most complete).
    return max(hits, key=lambda h: len(h.sequence))


def build_genotype_table(
    sample_ids: list[str],
    parquet_dir: str | Path,
    reference: str | None = None,
    *,
    parquet_suffix: str = "_protein_sequences.parquet",
) -> pd.DataFrame:
    """Build the per-isolate RRDR genotype table.

    Parameters
    ----------
    sample_ids : list of str
        Samples to genotype.
    parquet_dir : str or Path
        Directory of ``{sample_id}{parquet_suffix}`` files.
    reference : str, optional
        H37Rv rpoB reference; loaded from the fixture if not supplied.
    parquet_suffix : str
        Filename suffix for the protein-sequence parquets.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``Sample``. Columns: ``rpob_flat_index`` (int, the embedding
        index for predictor 3), ``rpob_sequence`` (str), one column per RRDR
        codon named ``codon_{n}`` holding the observed amino acid, and
        ``n_rrdr_substitutions`` (count of codons differing from wild-type).
    """
    parquet_dir = Path(parquet_dir)
    reference = reference if reference is not None else load_reference()
    assert_reference_panel(reference)
    aligner = _build_aligner()

    codons = list(range(RRDR_FIRST_CODON, RRDR_LAST_CODON + 1))
    wt_by_codon = {codon: reference[ref_index_for_codon(reference, codon)] for codon in codons}

    rows: list[dict] = []
    n_missing = 0
    for sample_id in sample_ids:
        parquet_path = parquet_dir / f"{sample_id}{parquet_suffix}"
        if not parquet_path.exists():
            n_missing += 1
            continue
        hit = _rpob_hit(parquet_path)
        if hit is None:
            n_missing += 1
            continue
        observed = genotype_rrdr(hit.sequence, reference, aligner)
        row = {"Sample": sample_id, "rpob_flat_index": hit.flat_index, "rpob_sequence": hit.sequence}
        n_subs = 0
        for codon in codons:
            aa = observed[codon]
            row[f"codon_{codon}"] = aa
            if aa not in ("-", wt_by_codon[codon]):
                n_subs += 1
        row["n_rrdr_substitutions"] = n_subs
        rows.append(row)

    if n_missing:
        logger.warning("rpoB genotype: %d/%d samples had no parquet or no rpoB hit", n_missing, len(sample_ids))

    return pd.DataFrame(rows).set_index("Sample")


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
