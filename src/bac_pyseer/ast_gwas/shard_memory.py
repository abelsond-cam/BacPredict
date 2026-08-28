"""Predict a unitig-LMM shard's peak memory, and refuse to submit a job that cannot hold it.

Why this exists
---------------
The sharded LMM chain reserves thousands of core-hours before the first shard reports anything, and
its failure mode is an **OOM hours in**. That is the expensive way to discover a memory estimate was
wrong. This module makes the estimate explicit, testable, and checkable *before* submission — and
records real measurements so the estimate improves instead of being re-guessed.

What the calibration actually supports
--------------------------------------
One sweep exists, from the invasion GWAS (``run_unitig_lmm_sharded.sh:9-11``, 2026-06-24): at
**n = 13,602 genomes, cpu = 8**, shards of 10k / 50k / 100k unitigs peaked at **9 / 21 / 26 GB**.

* **Shard size is well determined.** ``peak ≈ 7.383·ln(u) − 59.0`` fits the two outer points and then
  predicts the *held-out* middle one at 20.9 GB against 21 measured. This is the quantitative form of
  the script's own remark that "memory is sub-linear in shard size".
* **Cohort size is not determined at all.** Every point is at one n. There is no second cohort, and
  no ``sacct MaxRSS`` for a unitig shard is recorded anywhere in the repo.

And the folk model is refuted by its own calibration. "Peak ≈ cpu × n²" implies one n×n rotation
matrix per worker; at the anchor that is 8 × 1.48 GB = **11.8 GB**, which is *more than the 9 GB
measured in total* at u = 10k. So pyseer does not hold a private copy per worker — plausibly fork
copy-on-write — and any extrapolation resting on that model is unsound.

So this module does not pretend to a point estimate. It returns a **bracket**, honestly wide where
the data is silent:

* ``low``  — peak grows **linearly** in n, and not at all in cpu (fully shared rotation).
* ``high`` — peak grows **quadratically** in n and linearly in cpu (fully private rotation).

At TB's rifampin (n = 28,508, cpu = 8, u = 100k) that is roughly **54–114 GB** against the current
``--mem=128G`` default. So the default clears the *upper* bound — but by 1.12x, where the invasion
run enjoyed ~5x, and the bound itself is an extrapolation from a single cohort. That is the whole
finding: not "128G is too small", but "nobody can currently say", which is why a measurement is
needed rather than an argument. Feed observations in and the bracket collapses onto a fitted
exponent.

Usage
-----
``python -m bac_pyseer.ast_gwas.shard_memory --n 28508 --cpu 8 --unitigs-per-shard 100000
--mem-gb 128`` — exits non-zero if the allocation does not clear the bracket by ``--margin``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

# The invasion sweep. n and cpu are fixed across it, which is the whole limitation.
ANCHOR_N = 13_602
ANCHOR_CPU = 8
ANCHOR_SWEEP: tuple[tuple[int, float], ...] = ((10_000, 9.0), (50_000, 21.0), (100_000, 26.0))

# peak(u) = _SLOPE * ln(u) + _INTERCEPT, at the anchor cohort. Derived from the outer two points of
# ANCHOR_SWEEP; the middle point is deliberately NOT used, so it stays a check rather than a fit.
_SLOPE = (ANCHOR_SWEEP[-1][1] - ANCHOR_SWEEP[0][1]) / (
    math.log(ANCHOR_SWEEP[-1][0]) - math.log(ANCHOR_SWEEP[0][0])
)
_INTERCEPT = ANCHOR_SWEEP[0][1] - _SLOPE * math.log(ANCHOR_SWEEP[0][0])

DEFAULT_MARGIN = 1.25
# CSD3 icelake-himem sells memory by the core; a --mem above cpus x this silently inflates the
# allocation (and the reservation SLURM checks against AssocGrpCPUMinutesLimit).
MB_PER_CORE = 6_760


@dataclass(frozen=True)
class Observation:
    """One measured shard peak. ``max_rss_gb`` is what ``sacct -o MaxRSS`` actually reported."""

    n: int
    cpu: int
    unitigs_per_shard: int
    max_rss_gb: float
    source: str = ""

    def as_dict(self) -> dict:
        """JSON-serialisable form."""
        return {
            "n": self.n, "cpu": self.cpu, "unitigs_per_shard": self.unitigs_per_shard,
            "max_rss_gb": self.max_rss_gb, "source": self.source,
        }


def anchor_peak_gb(unitigs_per_shard: int) -> float:
    """Peak at the anchor cohort (n=13,602, cpu=8) for a given shard size, in GB."""
    if unitigs_per_shard <= 0:
        raise ValueError("unitigs_per_shard must be positive")
    return _SLOPE * math.log(unitigs_per_shard) + _INTERCEPT


def _fit_exponent(observations: list[Observation], unitigs_per_shard: int) -> float | None:
    """Fit the exponent p in ``peak ∝ n^p`` from observations, or ``None`` if they cannot pin it.

    Needs at least two observations at genuinely different n. Each is first divided by the anchor's
    shard-size term so that points taken at different shard sizes are comparable, then a
    least-squares line is fitted in log-log space.
    """
    pts = []
    for o in observations:
        base = anchor_peak_gb(o.unitigs_per_shard)
        if base <= 0 or o.n <= 0 or o.max_rss_gb <= 0:
            continue
        pts.append((math.log(o.n / ANCHOR_N), math.log(o.max_rss_gb / base)))
    if len({round(x, 9) for x, _ in pts}) < 2:
        return None
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    denom = sum((x - mx) ** 2 for x, _ in pts)
    if denom <= 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in pts) / denom
    # Clamp to the physically defensible band: sub-linear or super-quadratic scaling would be an
    # artefact of noise in two points, not a discovery.
    return min(2.0, max(1.0, slope))


def estimate_shard_peak(
    n: int, cpu: int, unitigs_per_shard: int, observations: list[Observation] | None = None
) -> tuple[float, float]:
    """Bracket a shard's peak memory in GB as ``(low, high)``.

    With no observations the bracket spans the two defensible n-scalings (linear with a shared
    rotation matrix, quadratic with a private one per worker). Given observations spanning more than
    one cohort size, the fitted exponent replaces both ends and the bracket collapses to a ±15% band
    around it — narrow, but still a band, because a fit through a handful of points is not a promise.
    """
    if n <= 0 or cpu <= 0:
        raise ValueError("n and cpu must be positive")
    base = anchor_peak_gb(unitigs_per_shard)
    ratio = n / ANCHOR_N

    fitted = _fit_exponent(observations or [], unitigs_per_shard)
    if fitted is not None:
        centre = base * ratio**fitted
        return centre * 0.85, centre * 1.15

    low = base * ratio  # rotation shared across workers; cpu does not enter
    high = base * ratio**2 * (cpu / ANCHOR_CPU)  # a private copy per worker
    return low, max(low, high)


def cores_for_mem(mem_gb: float) -> int:
    """Cores CSD3 will actually allocate for a ``--mem`` request, given ``MaxMemPerCPU``.

    Requesting ``--mem=128G`` at ``--cpus-per-task=8`` does not get 8 cores: SLURM raises the
    allocation to cover the memory. That inflated number is what the reservation is computed against,
    so a reservation printed from the requested cores understates the truth.

    SLURM's ``G`` suffix is **GiB**, not GB, and at this ratio the difference is a whole core
    (128 GiB / 6,760 MB = 20 cores; the decimal reading gives 19).
    """
    return max(1, math.ceil(mem_gb * 1024 / MB_PER_CORE))


def check_allocation(
    mem_gb: float, n: int, cpu: int, unitigs_per_shard: int, *,
    margin: float = DEFAULT_MARGIN, observations: list[Observation] | None = None,
) -> tuple[bool, str]:
    """Does ``mem_gb`` clear the estimated peak by ``margin``? Returns ``(ok, human-readable why)``.

    The *upper* end of the bracket is what must be cleared. Sizing to the optimistic end is how a
    job passes its own check and then dies four hours in.
    """
    low, high = estimate_shard_peak(n, cpu, unitigs_per_shard, observations)
    need = high * margin
    ok = mem_gb >= need
    fitted = _fit_exponent(observations or [], unitigs_per_shard)
    basis = (
        f"fitted n^{fitted:.2f} from {len(observations or [])} measurement(s)"
        if fitted is not None
        else "UNMEASURED at this cohort size — bracket spans linear to quadratic in n"
    )
    verdict = "OK" if ok else "INSUFFICIENT"
    return ok, (
        f"[{verdict}] n={n:,} cpu={cpu} unitigs/shard={unitigs_per_shard:,}\n"
        f"  estimated peak  {low:.1f}-{high:.1f} GB   ({basis})\n"
        f"  need >= {need:.1f} GB at margin {margin:g}x; requested --mem={mem_gb:g}G\n"
        f"  note: {mem_gb:g}G allocates {cores_for_mem(mem_gb)} cores at "
        f"MaxMemPerCPU={MB_PER_CORE} MB, which is what the reservation is billed against"
    )


def load_observations(path: Path | None) -> list[Observation]:
    """Read a calibration file (JSON list). A missing file is empty, not an error."""
    if path is None or not path.is_file():
        return []
    payload = json.loads(path.read_text() or "[]")
    return [
        Observation(
            n=int(r["n"]), cpu=int(r["cpu"]), unitigs_per_shard=int(r["unitigs_per_shard"]),
            max_rss_gb=float(r["max_rss_gb"]), source=str(r.get("source", "")),
        )
        for r in payload
    ]


def record_observation(path: Path, obs: Observation) -> list[Observation]:
    """Append a measurement and rewrite the calibration file. Returns every observation held."""
    existing = load_observations(path)
    existing.append(obs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([o.as_dict() for o in existing], indent=2) + "\n")
    return existing


def main(argv: list[str] | None = None) -> None:
    """CLI entry point. Exits 1 when the allocation does not clear the estimate."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, required=True, help="train+validate genomes in this drug's GWAS")
    p.add_argument("--cpu", type=int, default=8, help="pyseer --cpu (workers), not --cpus-per-task")
    p.add_argument("--unitigs-per-shard", type=int, required=True)
    p.add_argument("--mem-gb", type=float, default=None, help="the --mem being requested; omit to only report")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    p.add_argument("--calibration", type=Path, default=None, help="JSON of prior measurements")
    p.add_argument("--record-max-rss-gb", type=float, default=None, help="append a measurement and exit")
    p.add_argument("--source", default="", help="provenance for --record-max-rss-gb, e.g. a job id")
    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.record_max_rss_gb is not None:
        if args.calibration is None:
            raise SystemExit("--record-max-rss-gb needs --calibration")
        held = record_observation(args.calibration, Observation(
            n=args.n, cpu=args.cpu, unitigs_per_shard=args.unitigs_per_shard,
            max_rss_gb=args.record_max_rss_gb, source=args.source,
        ))
        print(f"recorded {args.record_max_rss_gb:.1f} GB at n={args.n:,}; "
              f"{len(held)} observation(s) in {args.calibration}")
        return

    obs = load_observations(args.calibration)
    if args.mem_gb is None:
        low, high = estimate_shard_peak(args.n, args.cpu, args.unitigs_per_shard, obs)
        print(f"{low:.1f} {high:.1f}")
        return
    ok, message = check_allocation(
        args.mem_gb, args.n, args.cpu, args.unitigs_per_shard, margin=args.margin, observations=obs
    )
    print(message, file=sys.stderr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
