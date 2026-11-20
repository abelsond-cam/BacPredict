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

The file is read once, as bytes: an md5 given to :func:`read_samples_file` is checked against exactly the bytes it
then parses, so a list rewritten after the array's own md5 check is still caught (and ``\r`` is never a line break,
as for awk).
"""

from __future__ import annotations

import hashlib
import string
from pathlib import Path


def parse_samples(text: str) -> list[str]:
    """The non-blank lines of ``text``, stripped of ASCII whitespace (no sortedness check)."""
    return [s for s in (line.strip(string.whitespace) for line in text.split("\n")) if s]


def read_samples_file(path: Path, md5: str | None = None) -> list[str]:
    """Read a sample list and require it to be non-empty, sorted and unique (and, given ``md5``, unchanged).

    Parameters
    ----------
    path : Path
        One sample per line; blank lines are ignored. Decoded as UTF-8.
    md5 : str, optional
        The hex md5 the file must have (what the array was submitted with). Checked against the same bytes that
        are parsed, so there is no gap between the check and the read.

    Returns
    -------
    list of str
        The samples, in file order (which is sorted order).

    Raises
    ------
    ValueError
        If the file's md5 is not ``md5``, it is not UTF-8, it lists no sample, or a sample is not strictly greater
        than the one before it.
    """
    data = Path(path).read_bytes()
    if md5 is not None and (got := hashlib.md5(data).hexdigest()) != md5:
        raise ValueError(f"{path} changed since submit: md5 {got}, submitted with {md5}")
    samples = parse_samples(data.decode("utf-8"))
    if not samples:
        raise ValueError(f"{path} lists no samples")
    for prev, cur in zip(samples, samples[1:], strict=False):
        if cur <= prev:
            what = "a duplicate" if cur == prev else "out of order"
            raise ValueError(f"{path} must be sorted and unique: {cur!r} after {prev!r} is {what}")
    return samples
