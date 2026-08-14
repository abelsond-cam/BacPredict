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

import csv as csv_module
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


# Top-level packages retired in the 2026-07-11 consolidation. Naming one is not itself a defect —
# the dead-path tables have to name them — but a doc that mentions one without anywhere saying it is
# gone is telling a reader to use it.
RETIRED_PACKAGES = ("tl", "tb_ast", "kleb_ast", "pangena_predict", "predict_hgt", "admixture")
_RETIRED = re.compile(rf"src/({'|'.join(RETIRED_PACKAGES)})/")
_MARKS_THEM_DEAD = re.compile(
    r"retired|dead|no longer exist|stopped existing|superseded|consolidat|DELETED|not exist",
    re.IGNORECASE,
)


@pytest.mark.parametrize("doc", _tracked_markdown(), ids=lambda p: p.relative_to(REPO).as_posix())
def test_retired_packages_are_only_named_as_retired(doc: Path) -> None:
    """A doc naming a retired package must somewhere say the package is gone.

    `test_doc_links_resolve` only inspects link *targets*, so it misses the two ways a dead path
    actually reaches a reader: inside a code span or table cell (README described the whole AMR
    pipeline as `src/kleb_ast/...` for a month after it stopped existing), and as a link *label*
    whose target happens to be live (`docs/results_schema.md` rendered `src/tl/train/metrics.py`
    while pointing at the real file, so it passed while still telling the reader the wrong path).
    """
    text = doc.read_text(encoding="utf-8")
    hits = sorted(set(_RETIRED.findall(text)))
    if not hits:
        return
    assert _MARKS_THEM_DEAD.search(text), (
        f"{doc.relative_to(REPO)} names retired package(s) {hits} but never says they are gone. "
        "Either repoint to the engine/apps path or add a banner marking them retired — see "
        "PROJECT_STATE.md §2."
    )


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
    """No live ceiling row may claim to be current unless it declares the deployment-holdout scorer.

    Kp's catalogue ceiling was migrated onto the deployment-holdout scorer (it fits on `train` and
    scores on `holdout` via `load_splits` — see `apps/kleb/card_determinant_lr.py`); TB's was not, so
    TB is a different estimator on a different evaluation set and its rows must say `provisional`.

    **The check is on `ceiling_estimator`, deliberately.** An earlier version of this test inferred
    the estimator from `ceiling_auroc_sd == 0.0`, on the theory that fitting once gives no spread.
    That is a weaker check in two ways, both of which matter:

    * It answers "was it fit once?", not "was it fit on the *deployment holdout*?" — and the
      pre-fix leaky read-out path also fit once, on the wrong split, so it would report 0.0 and pass.
    * Zero spread is not exclusive to the holdout scorer. The k-fold probe reports exactly 0.0 for a
      determinant whose AUROC happens to be identical across folds (`tbprofiler_gene_lr_isoniazid.csv`,
      row `inhA/coding`, is a live example).

    `ceiling_estimator` states the answer outright, so read it rather than inferring it. The SD is
    still checked, but only as a corroborating signal on rows claiming to be current.
    """
    vis = REPO / "src/bacpredict/visualisations"
    for csv in vis.rglob("catalogue_ceiling_panel.csv"):
        if _is_exempt(csv):
            continue
        with csv.open(newline="", encoding="utf-8") as fh:
            rows = list(csv_module.DictReader(fh))
        assert rows, f"{csv.relative_to(REPO)} has no data rows"
        for col in ("ceiling_estimator", "ceiling_status", "ceiling_auroc_sd"):
            assert col in rows[0], f"{csv.relative_to(REPO)} has no {col} column"
        for row in rows:
            drug, estimator, status = row["drug"], row["ceiling_estimator"], row["ceiling_status"]
            if status != "provisional":
                assert estimator == "deployment_holdout", (
                    f"{csv.relative_to(REPO)}: {drug} claims status '{status}' but its estimator is "
                    f"'{estimator}'. Only deployment_holdout rows may be quoted as current; anything "
                    "else is a different estimator on a different evaluation set."
                )
                sd = float(row["ceiling_auroc_sd"] or 0.0)
                assert sd == 0.0, (
                    f"{csv.relative_to(REPO)}: {drug} claims the deployment-holdout scorer, which "
                    f"fits once, but reports a non-zero spread ({sd}). One of the two is wrong."
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
