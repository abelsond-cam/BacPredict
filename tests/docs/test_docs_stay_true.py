"""Guard the documentation invariants that a state audit had to fix by hand.

Every check here corresponds to a defect found in the 2026-08 audit, and each one failed silently
for weeks before it was found. Docs have no test suite by default, so a wrong path or a stale number
in a `CLAUDE.md` is discovered only when it misleads someone — which is exactly what happened: four
fine-tune AUROCs were quoted from a checked-in summary panel, one of them wrong by 0.10, while the
two documents presenting themselves as authoritative described a repo layout that had not existed
for a month.

These are cheap, exact checks. They cannot tell whether prose is *true*; they can tell whether it
points at things that exist, and whether the artifacts it must not use have come back.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Directories whose whole point is to preserve superseded material. Their contents are expected to
# name dead paths and stale numbers, so linting them would defeat the purpose of keeping them.
EXEMPT_DIRS = ("docs/_archive/", "docs/_retired/", "docs/_parked/", "visualisations/_superseded/")

# A markdown link target that looks like a repo path: (src/...), (tests/...), (docs/...).
_LINK = re.compile(r"\[[^\]]*\]\((?!https?:|mailto:|#)([^)#]+)")


def _tracked_markdown() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    return [REPO / p for p in out if not any(e in p for e in EXEMPT_DIRS)]


def _is_exempt(path: Path) -> bool:
    rel = path.relative_to(REPO).as_posix()
    return any(e in rel for e in EXEMPT_DIRS)


@pytest.mark.parametrize("doc", _tracked_markdown(), ids=lambda p: p.relative_to(REPO).as_posix())
def test_doc_links_resolve(doc: Path) -> None:
    """Every relative link in a tracked doc points at something that exists.

    This single check catches every dead-path defect the audit found: `src/tl/`, `src/tb_ast/`,
    `src/kleb_ast/`, `src/pangena_predict/`, `src/predict_hgt/`, the `snp_vs_esm_prediction.py`
    that was deleted, and two `../../../CLAUDE.md` links that had been off by one directory since
    the consolidation.
    """
    broken = []
    for match in _LINK.finditer(doc.read_text(encoding="utf-8")):
        target = match.group(1).strip()
        # `~`-rooted references point outside the repo (the ~/.claude cluster docs) — not ours to check.
        if not target or target.startswith(("~", "<", "$")):
            continue
        if not (doc.parent / target).resolve().exists():
            broken.append(target)
    assert not broken, f"{doc.relative_to(REPO)} links to paths that do not exist: {broken}"


def test_convention_docs_carry_no_results() -> None:
    """The root CLAUDE.md states conventions and must not state results.

    It is auto-loaded into every session, so a number in it is read constantly and updated never.
    That is how `~7 pp` for the bf16 gap — itself an invalid val-peak-against-eval-holdout
    comparison — survived in the most-read file in the repo. Numbers belong in PROJECT_STATE.md,
    where they carry the artifact they came from.
    """
    text = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    # Strip fenced code blocks: a threshold or a version in an example command is not a result.
    prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    prose = re.sub(r"`[^`]*`", "", prose)
    offenders = re.findall(r"(?<![\d.])0\.\d{3,}", prose)
    assert not offenders, (
        f"root CLAUDE.md contains what look like result values: {sorted(set(offenders))}. "
        "Numbers of record belong in PROJECT_STATE.md §3, named against their artifact."
    )


def test_no_pre_migration_ceiling_outside_superseded() -> None:
    """No live ceiling CSV may use the retired k-fold-probe estimator.

    Kp's catalogue ceiling was migrated onto the deployment-holdout scorer; TB's was not, so the two
    are a different estimator on a different evaluation set. Both schemas carry a `mut_auroc_sd`
    column, so its presence proves nothing — the marker is its *value*: the holdout scorer fits once
    and reports exactly 0.0, the k-fold probe reports the spread across folds.

    The live panels must therefore either be all-zero SD, or say `provisional` about themselves.
    """
    vis = REPO / "src/bacpredict/visualisations"
    for csv in vis.rglob("catalogue_ceiling_panel.csv"):
        if _is_exempt(csv):
            continue
        lines = csv.read_text(encoding="utf-8").strip().splitlines()
        header = lines[0].split(",")
        assert "ceiling_status" in header, f"{csv.relative_to(REPO)} has no ceiling_status column"
        sd_i, status_i = header.index("ceiling_auroc_sd"), header.index("ceiling_status")
        for line in lines[1:]:
            row = line.split(",")
            if float(row[sd_i]) != 0.0:
                assert row[status_i] == "provisional", (
                    f"{csv.relative_to(REPO)}: {row[0]} has a non-zero ceiling_auroc_sd "
                    f"({row[sd_i]}), so it came from the k-fold probe, but is not marked provisional"
                )


def test_summary_panels_stay_quarantined() -> None:
    """`amr_summary_panel.csv` must not reappear outside `_superseded/`.

    That file is the physical origin of the wrong fine-tune numbers — its `ft_auroc` column reads
    colistin 0.8072 against a real 0.9094. Regenerating one into the live tree would silently make
    it the first thing a grep finds again.
    """
    vis = REPO / "src/bacpredict/visualisations"
    live = [p.relative_to(REPO).as_posix() for p in vis.rglob("*amr_summary_panel.csv") if not _is_exempt(p)]
    assert not live, (
        f"summary panels found outside _superseded/: {live}. These carry a stale ft_auroc; "
        "the ceiling belongs in catalogue_ceiling_panel.csv and FT numbers come from results.json."
    )


def test_project_state_stamp_is_parseable_and_not_far_behind() -> None:
    """PROJECT_STATE.md carries a stamp, and the stamp is a real ancestor of HEAD.

    A state file with no verification date is a state file nobody can trust. Bounding how far HEAD
    may run ahead turns "we forgot to update it" from an invisible drift into a failing test.
    """
    text = (REPO / "PROJECT_STATE.md").read_text(encoding="utf-8")
    m = re.search(r"Last verified:\s*(\d{4}-\d{2}-\d{2})\s*@\s*`([0-9a-f]{7,40})`", text)
    assert m, "PROJECT_STATE.md must carry a `Last verified: <YYYY-MM-DD> @ `<sha>`` stamp"
    sha = m.group(2)

    known = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=REPO, capture_output=True
    )
    if known.returncode != 0:
        pytest.skip(f"stamped commit {sha} not present in this checkout (shallow clone or rebase)")

    ahead = subprocess.run(
        ["git", "rev-list", "--count", f"{sha}..HEAD"], cwd=REPO, capture_output=True, text=True
    )
    if ahead.returncode != 0:
        pytest.skip("cannot count commits since the stamp in this checkout")
    n = int(ahead.stdout.strip() or 0)
    assert n <= 25, (
        f"PROJECT_STATE.md was last verified {n} commits ago (at {sha}). Re-verify it and move the "
        "stamp — it is meant to be updated in the same commit as the work that changes a fact."
    )
