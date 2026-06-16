"""QC: variant frequency spectrum + per-position allele-frequency view for one cohort.

Two figures derived from the pyseer presence data, as a sanity-check gate before the GWAS:

1. **Frequency histogram** — a log10-binned histogram of each locus's frequency (the
   fraction of samples carrying the ALT). Shows how the variant mass splits across the
   spectrum and, in particular, how many loci fall below the 1% threshold that the GWAS
   presence matrix drops.
2. **Per-position scatter** — per-locus allele frequency (% of samples) against position
   along the single reference contig ``NC_009648``, restricted to the GWAS loci (>=1%).
   Unadjusted for population structure — purely a view of where common variation sits.

Per-locus frequencies come from one of three sources (cheapest first):

* ``--rtab`` — a single streaming pass over the post-filter presence Rtab gives the
  ``>=1%`` loci's frequencies with no cache rebuild (the ``<1%`` count is read from the
  ``collation_manifest.json`` for annotation). This is the light, default path.
* ``--from-npz`` — a saved ``prefilter_locus_spectrum.npz`` (written by the reduce), the
  full spectrum incl. the ``<1%`` loci.
* ``--cohort-csv`` + ``--cache-dir`` — recompute the full pre-filter spectrum from the
  per-sample locus cache via :func:`build_presence_and_distances.build_presence_matrix`
  (himem; only needed when no saved spectrum exists and the ``<1%`` shape is wanted).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from math import ceil, log10
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bac_pyseer.kleb_iso_source.build_presence_and_distances import (
    _present_samples,
    build_presence_matrix,
    parse_positions,
)

DEFAULT_CONTIG = "NC_009648"
DEFAULT_CONTIG_LEN = 5_315_120  # NC_009648 (K. pneumoniae MGH 78578), bp

# Frequency-band edges (as fractions of the cohort) for the annotation table.
_BAND_EDGES = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0 + 1e-9]
_BAND_LABELS = ["<0.1%", "0.1-1%", "1-10%", "10-50%", ">=50%"]


def frequency_bands(freq: np.ndarray, n_samples: int) -> dict[str, int]:
    """Count loci per frequency band.

    Parameters
    ----------
    freq
        Per-locus sample counts (number of samples carrying each variant).
    n_samples
        Cohort size (denominator for the frequency fraction).

    Returns
    -------
    dict
        Ordered mapping ``band_label -> n_loci`` over :data:`_BAND_LABELS`.
    """
    frac = freq / n_samples
    counts = np.histogram(frac, bins=_BAND_EDGES)[0]
    return dict(zip(_BAND_LABELS, (int(c) for c in counts), strict=True))


def _annotation_text(
    freq: np.ndarray, n_samples: int, min_count: int,
    *, dropped_lt_min: int | None = None, total_loci: int | None = None,
) -> str:
    """Build the headline + band-table text drawn on the histogram.

    When ``dropped_lt_min`` is given (Rtab mode, ``freq`` holds only the >=1% loci), the
    ``<1%`` count comes from that value (the manifest) and only the >=1% bands are shown.
    Otherwise ``freq`` is the full pre-filter spectrum and every band is measured.
    """
    bands = frequency_bands(freq, n_samples)
    if dropped_lt_min is None:
        total = freq.size
        n_lt = int((freq < min_count).sum())
        n_ge = total - n_lt
        lt_label = "<1% (dropped)"
        band_items = list(bands.items())
    else:
        total = int(total_loci) if total_loci is not None else int(dropped_lt_min) + freq.size
        n_lt = int(dropped_lt_min)
        n_ge = freq.size
        lt_label = "<1% (manifest)"
        band_items = [(k, v) for k, v in bands.items() if k in ("1-10%", "10-50%", ">=50%")]
    lines = [
        f"total loci: {total:,}",
        f"{lt_label}: {n_lt:,} ({n_lt / total:.1%})",
        f">=1% (GWAS): {n_ge:,} ({n_ge / total:.1%})",
        "—",
        *[f"{lab}: {n:,}" for lab, n in band_items],
    ]
    return "\n".join(lines)


def compute_spectrum(
    *, cohort_csv: Path, cache_dir: Path, label_col: str, min_freq: float, n_jobs: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Rebuild the pre-filter per-locus frequency spectrum from the per-sample cache.

    Mirrors the reduce's present-sample selection so the spectrum matches the GWAS inputs.

    Returns
    -------
    tuple
        ``(pos, freq, n_samples, min_count)`` — POS and sample-count per locus, the cohort
        size, and the >=``min_freq`` count threshold.
    """
    cohort = pd.read_csv(cohort_csv, usecols=["Sample", label_col], low_memory=False)
    cohort["Sample"] = cohort["Sample"].astype(str)
    cohort = cohort.dropna(subset=[label_col]).drop_duplicates(subset=["Sample"])
    present, paths, missing = _present_samples(cohort["Sample"].tolist(), cache_dir)
    logging.info("cohort labelled=%d  with cache=%d  missing=%d", len(cohort), len(present), len(missing))
    if not present:
        raise SystemExit("No cohort samples have a cache file — run extract_sample_loci first.")

    x, keys = build_presence_matrix(paths, n_jobs)
    freq = np.asarray(x.sum(axis=0)).ravel().astype(np.int64)
    pos = parse_positions(keys)
    min_count = max(1, ceil(min_freq * len(present)))
    logging.info("loci=%d  n_samples=%d  min_count(>=1%%)=%d", freq.size, len(present), min_count)
    return pos, freq, len(present), min_count


def load_spectrum(npz_path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load a saved ``prefilter_locus_spectrum.npz`` → ``(pos, freq, n_samples, min_count)``."""
    d = np.load(npz_path)
    return d["pos"], d["freq"], int(d["n_samples"]), int(d["min_count"])


def spectrum_from_rtab(rtab_path: Path, contig: str) -> tuple[np.ndarray, np.ndarray, int]:
    r"""Stream the post-filter presence Rtab → ``(pos, freq, n_samples)`` for the >=1% loci.

    One sequential pass, low memory, no cache rebuild. Each data row is
    ``"<contig>_<POS>_<REF>_<ALT>\\t0\\t1\\t..."``; the variant's sample count is the number
    of ``"1"`` characters after the first tab (cells are 0/1, tabs/0s contain no ``"1"``),
    and POS is parsed from the variant id after stripping the ``"<contig>_"`` prefix.

    Parameters
    ----------
    rtab_path
        Path to ``variant_by_loci_presence.Rtab``.
    contig
        Reference contig name prefixing each variant id (e.g. ``"NC_009648"``).

    Returns
    -------
    tuple
        ``(pos, freq, n_samples)`` — POS and sample-count per locus, and the cohort size.
    """
    prefix = f"{contig}_"
    pos_list: list[int] = []
    freq_list: list[int] = []
    with open(rtab_path) as fh:
        header = fh.readline().rstrip("\n")
        n_samples = header.count("\t")  # 'variant' col + N sample cols → N tabs
        for line in fh:
            vid, _, rest = line.partition("\t")
            freq_list.append(rest.count("1"))
            tail = vid[len(prefix):] if vid.startswith(prefix) else vid
            pos_list.append(int(tail.split("_", 1)[0]))
    return np.array(pos_list, dtype=np.int64), np.array(freq_list, dtype=np.int64), int(n_samples)


def load_manifest_counts(manifest_path: Path) -> tuple[int, int, int]:
    """Read ``(n_loci_prefilter, n_loci_postfilter, min_freq_count)`` from a collation manifest."""
    m = json.loads(Path(manifest_path).read_text())
    return int(m["n_loci_prefilter"]), int(m["n_loci_postfilter"]), int(m["min_freq_count"])


def plot_histogram(
    freq: np.ndarray, n_samples: int, min_count: int, out_path: Path, n_bins: int = 60,
    *, x_min: float | None = None, dropped_lt_min: int | None = None, total_loci: int | None = None,
) -> None:
    """Render the log10-binned per-locus frequency histogram with a 1% line + band table.

    ``x_min`` sets the left bin edge (default ``1/n_samples``); pass ``min_freq`` to start
    the axis at 1% in Rtab (``>=1%`` only) mode. ``dropped_lt_min`` / ``total_loci`` annotate
    the ``<1%`` count from the manifest when ``freq`` holds only the >=1% loci.
    """
    frac = freq / n_samples
    lo = x_min if x_min is not None else 1.0 / n_samples
    bins = np.logspace(log10(lo), 0.0, n_bins + 1)
    scope = ">=1% loci (<1% from manifest)" if dropped_lt_min is not None else "pre-filter"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(frac, bins=bins, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(0.01, color="crimson", ls="--", lw=1.3, label="1% GWAS filter")
    ax.set_xlabel("per-locus frequency (fraction of samples with the variant)")
    ax.set_ylabel("number of variant loci")
    ax.set_title(f"Variant frequency spectrum — {n_samples:,} samples, {freq.size:,} loci ({scope})")
    ax.text(
        0.015, 0.97,
        _annotation_text(freq, n_samples, min_count, dropped_lt_min=dropped_lt_min, total_loci=total_loci),
        transform=ax.transAxes, va="top", ha="left", fontsize=8.5, family="monospace",
        bbox={"boxstyle": "round", "fc": "white", "ec": "0.7", "alpha": 0.9},
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("wrote %s", out_path)


def plot_position_scatter(
    pos: np.ndarray, freq: np.ndarray, n_samples: int, min_count: int, out_path: Path,
    contig: str = DEFAULT_CONTIG, contig_len: int = DEFAULT_CONTIG_LEN,
) -> None:
    """Render per-locus allele frequency (%) vs position for the GWAS loci (>=``min_count``)."""
    keep = freq >= min_count
    x = pos[keep]
    y = 100.0 * freq[keep] / n_samples
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.scatter(x, y, s=2, alpha=0.15, color="navy", linewidths=0, rasterized=True)
    ax.set_xlim(0, contig_len)
    ax.set_ylim(0, 100)
    ax.set_xlabel(f"position on {contig} (bp)")
    ax.set_ylabel("% of samples with variant")
    ax.set_title(
        f"Per-locus allele frequency along {contig} — {int(keep.sum()):,} GWAS loci (>=1%)\n"
        "unadjusted for population structure; chromosomal loci only"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("wrote %s (%d points)", out_path, int(keep.sum()))


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rtab", type=Path, help="Post-filter Rtab — stream for >=1%% freq (light, no rebuild).")
    p.add_argument("--manifest", type=Path, help="collation_manifest.json (for the <1%% count in --rtab mode).")
    p.add_argument("--cohort-csv", type=Path, help="Cohort/split CSV (Sample + label) for the full-spectrum rebuild.")
    p.add_argument("--cache-dir", type=Path, help="Per-sample locus cache dir for the full-spectrum rebuild.")
    p.add_argument("--label-col", default="blood_vs_faeces_label")
    p.add_argument("--min-freq", type=float, default=0.01, help="GWAS prevalence threshold (drawn as the 1%% line).")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--spectrum-npz", type=Path, required=True, help="Output npz (or input when --from-npz).")
    p.add_argument("--from-npz", action="store_true", help="Load --spectrum-npz (saved full pre-filter spectrum).")
    p.add_argument("--out-fig-dir", type=Path, required=True, help="Directory for the two PNGs.")
    p.add_argument("--contig", default=DEFAULT_CONTIG)
    p.add_argument("--contig-len", type=int, default=DEFAULT_CONTIG_LEN)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    dropped_lt_min: int | None = None
    total_loci: int | None = None

    if args.rtab is not None:
        # Light default path: >=1% frequencies straight from the Rtab; <1% count from manifest.
        if args.manifest is None:
            p.error("--manifest is required with --rtab (for the <1% count annotation)")
        pos, freq, n_samples = spectrum_from_rtab(args.rtab, args.contig)
        n_pre, n_post, min_count = load_manifest_counts(args.manifest)
        dropped_lt_min, total_loci = n_pre - n_post, n_pre
        if freq.size != n_post:
            logging.warning("Rtab rows (%d) != manifest postfilter (%d)", freq.size, n_post)
        logging.info("Rtab spectrum: n=%d, >=1%% loci=%d, <1%% (manifest)=%d", n_samples, freq.size, dropped_lt_min)
    elif args.from_npz:
        pos, freq, n_samples, min_count = load_spectrum(args.spectrum_npz)
        logging.info("loaded spectrum from %s (loci=%d, n=%d)", args.spectrum_npz, freq.size, n_samples)
    else:
        if args.cohort_csv is None or args.cache_dir is None:
            p.error("one of --rtab, --from-npz, or (--cohort-csv + --cache-dir) is required")
        pos, freq, n_samples, min_count = compute_spectrum(
            cohort_csv=args.cohort_csv, cache_dir=args.cache_dir,
            label_col=args.label_col, min_freq=args.min_freq, n_jobs=args.n_jobs,
        )

    args.spectrum_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.spectrum_npz, pos=pos, freq=freq, n_samples=np.int64(n_samples), min_count=np.int64(min_count),
    )
    logging.info("wrote %s", args.spectrum_npz)

    plot_histogram(
        freq, n_samples, min_count, args.out_fig_dir / "variant_frequency_spectrum.png",
        x_min=(args.min_freq if args.rtab is not None else None),
        dropped_lt_min=dropped_lt_min, total_loci=total_loci,
    )
    plot_position_scatter(
        pos, freq, n_samples, min_count, args.out_fig_dir / "variant_frequency_by_position.png",
        contig=args.contig, contig_len=args.contig_len,
    )


if __name__ == "__main__":
    main()
