r"""The sample list an ESM-C array embeds: one BioSample per non-blank line, sorted and unique.

The protein and ESM-C stores are flat and shared across species (``processed/{protein_sequences,esm}``), so an
array cannot take "every parquet in the directory" as its work list: it would embed every species at once, and
its slices would shift whenever another species wrote a parquet mid-array. A cohort's driver writes this list
from its annotation table instead; ``generate_embeddings --samples-file`` slices it, and
``esm_embeddings_array.sh`` counts it with awk to cut the slices. The three must agree on what the file says, so
the parsing rule is spelled out here once:

- lines are split on ``\n`` only (awk's record separator), and each is stripped of ASCII whitespace
  (awk's ``[[:space:]]``); a line left empty is skipped;
- the samples must be strictly increasing (sorted, no duplicate), so a list has one canonical form and two
  drivers writing it from the same cohort write the same bytes.
"""

from __future__ import annotations

import string
from pathlib import Path


def parse_samples(text: str) -> list[str]:
    """The non-blank lines of ``text``, stripped of ASCII whitespace (no sortedness check)."""
    return [s for s in (line.strip(string.whitespace) for line in text.split("\n")) if s]


def read_samples_file(path: Path) -> list[str]:
    """Read a sample list and require it to be non-empty, sorted and unique.

    Parameters
    ----------
    path : Path
        One sample per line; blank lines are ignored.

    Returns
    -------
    list of str
        The samples, in file order (which is sorted order).

    Raises
    ------
    ValueError
        If the file lists no sample, or a sample is not strictly greater than the one before it.
    """
    samples = parse_samples(Path(path).read_text())
    if not samples:
        raise ValueError(f"{path} lists no samples")
    for prev, cur in zip(samples, samples[1:], strict=False):
        if cur <= prev:
            what = "a duplicate" if cur == prev else "out of order"
            raise ValueError(f"{path} must be sorted and unique: {cur!r} after {prev!r} is {what}")
    return samples
