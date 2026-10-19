"""Record what every stored Phase 2 result was produced with (#138).

Reads the C1-C20 results and the B0 baseline results and groups them by the
fields that decide whether they are comparable and reproducible: source
commit and tree state, evaluation protocol, V1/V2 annotation digest (#136),
D1 known-pair count (#137), candidate-file hash, library versions and model
revisions.  Each source commit is resolved against the local repository:
whether it exists, whether ``develop`` contains it, and which branches or
tags do.  Results that cannot be combined with the current ones are flagged.

Commit reachability depends on the local clone's refs, so that column can
differ between machines; every other cell is read from the result files.

Usage
-----
python scripts/results_inventory.py \\
    --results_dirs data/results/ data/results_ngram_baseline/ \\
    --output artifacts/issue-138/results_inventory.md
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kgsemembed.evaluation.metrics import EVALUATION_PROTOCOL_VERSION  # noqa: E402
from kgsemembed.verbalisation.base import annotation_predicates_digest  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_D1_KNOWN_PAIRS = 726
_ANNOTATION_COMPONENTS = frozenset({"V1", "V2"})
_ANNOTATION_DATASETS = frozenset({"D1", "D2"})
_GIT_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class ResultRecord:
    """
    The provenance fields of one stored result file.

    Attributes
    ----------
    tree : str
        Results directory name, e.g. ``"results"``.
    condition_id, dataset_id, pair_name, strategy : str
        Result identity.
    model_key : Optional[str]
        ``None`` for B0.
    hf_revision : str
        Model revision, ``"unknown"`` for B0.
    git_commit : str
        Full source commit.
    git_dirty : Optional[bool]
        Tree state at run time.
    protocol : Optional[int]
        ``evaluation_protocol``.
    annotation_digest : Optional[str]
        ``annotation_predicates_sha256``, absent before #136.
    n_train_refs : Optional[int]
        Known pairs in the population record.
    candidates_sha256 : str
        Candidate-file hash.
    versions : str
        Library versions as canonical JSON.
    """

    tree: str
    condition_id: str
    dataset_id: str
    pair_name: str
    strategy: str
    model_key: Optional[str]
    hf_revision: str
    git_commit: str
    git_dirty: Optional[bool]
    protocol: Optional[int]
    annotation_digest: Optional[str]
    n_train_refs: Optional[int]
    candidates_sha256: str
    versions: str


@dataclass(frozen=True)
class CommitInfo:
    """
    A source commit as the local repository sees it.

    Attributes
    ----------
    commit : str
        Full hash.
    exists : bool
        The object is present in the local clone.
    date, subject : str
        Committer date and subject line (empty when missing).
    on_develop : bool
        ``develop`` contains the commit.
    refs : Tuple[str, ...]
        Local branches, remote-tracking branches and tags containing it.
    """

    commit: str
    exists: bool
    date: str
    subject: str
    on_develop: bool
    refs: Tuple[str, ...]


def _record(tree: str, payload: dict) -> ResultRecord:
    population = payload.get("population") or {}
    return ResultRecord(
        tree=tree, condition_id=str(payload["condition_id"]), dataset_id=str(payload["dataset_id"]),
        pair_name=str(payload["pair_name"]), strategy=str(payload.get("strategy")),
        model_key=payload.get("model_key"), hf_revision=str(payload.get("hf_revision")),
        git_commit=str(payload.get("git_commit")), git_dirty=payload.get("git_dirty"),
        protocol=payload.get("evaluation_protocol"),
        annotation_digest=payload.get("annotation_predicates_sha256"),
        n_train_refs=population.get("n_train_refs"),
        candidates_sha256=str(payload.get("candidates_sha256")),
        versions=json.dumps(payload.get("versions"), sort_keys=True),
    )


def load_records(results_dir: Path) -> List[ResultRecord]:
    """
    Read every ``*_results.json`` file under one results directory.

    Parameters
    ----------
    results_dir : Path
        Directory laid out as ``{condition}/{dataset}/{pair}_results.json``.

    Returns
    -------
    List[ResultRecord]
        One record per file, sorted by path.
    """
    paths = sorted(results_dir.glob("*/*/*_results.json"))
    tree = results_dir.resolve().name
    return [_record(tree, json.loads(path.read_text(encoding="utf-8"))) for path in paths]


def reads_annotations(record: ResultRecord) -> bool:
    """Return whether the result's text depends on the #136 annotation predicates."""
    components = set(record.strategy.split("+"))
    return record.dataset_id in _ANNOTATION_DATASETS and bool(components & _ANNOTATION_COMPONENTS)


def digest_state(record: ResultRecord, current: str) -> str:
    """
    Classify a result's annotation digest against the current one.

    Parameters
    ----------
    record : ResultRecord
        Result to classify.
    current : str
        :func:`annotation_predicates_digest` of the checked-out code.

    Returns
    -------
    str
        ``"current"``, ``"other"`` or ``"absent"``.
    """
    if record.annotation_digest is None:
        return "absent"
    return "current" if record.annotation_digest == current else "other"


def flags(record: ResultRecord, current_digest: str, pair_hashes: Dict[Tuple[str, str], set]) -> List[str]:
    """
    List the reasons a result cannot be combined with current results.

    Parameters
    ----------
    record : ResultRecord
        Result to check.
    current_digest : str
        Current annotation-predicate digest.
    pair_hashes : Dict[Tuple[str, str], set]
        Candidate hashes seen per ``(dataset_id, pair_name)``.

    Returns
    -------
    List[str]
        Empty when the result is current.
    """
    reasons = []
    if record.git_dirty is not False:
        reasons.append("dirty or unknown tree")
    if record.protocol != EVALUATION_PROTOCOL_VERSION:
        reasons.append(f"protocol {record.protocol}")
    if record.dataset_id == "D1" and record.n_train_refs != _D1_KNOWN_PAIRS:
        reasons.append("D1 before the known-pair fix (#137)")
    if reads_annotations(record) and digest_state(record, current_digest) != "current":
        reasons.append("V1/V2 text before #136")
    if len(pair_hashes.get((record.dataset_id, record.pair_name), ())) > 1:
        reasons.append("candidate hash differs within the pair")
    return reasons


def candidate_hashes(records: Sequence[ResultRecord]) -> Dict[Tuple[str, str], set]:
    """Collect the candidate hashes recorded for each pair."""
    hashes: Dict[Tuple[str, str], set] = {}
    for record in records:
        hashes.setdefault((record.dataset_id, record.pair_name), set()).add(record.candidates_sha256)
    return hashes


# ---------------------------------------------------------------------------
# Git
# ---------------------------------------------------------------------------


def _git(*args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True,
            timeout=_GIT_TIMEOUT_SECONDS, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def commit_info(commit: str) -> CommitInfo:
    """
    Resolve a source commit against the local repository.

    Parameters
    ----------
    commit : str
        Full commit hash from a result file.

    Returns
    -------
    CommitInfo
        Existence, date, subject, ``develop`` ancestry and containing refs.
    """
    shown = _git("show", "-s", "--format=%cs%x00%s", commit)
    if shown is None:
        return CommitInfo(commit, False, "", "", False, ())
    date, subject = shown.split("\x00", 1)
    on_develop = _git("merge-base", "--is-ancestor", commit, "develop") is not None
    listed = _git(
        "for-each-ref", "--contains", commit, "--format=%(refname:short)",
        "refs/heads", "refs/remotes", "refs/tags",
    )
    refs = tuple(sorted(ref for ref in (listed or "").splitlines() if ref and ref != "origin/HEAD"))
    return CommitInfo(commit, True, date, subject, on_develop, refs)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in rows] + [""]


def _condition_key(condition_id: str) -> Tuple[str, int]:
    digits = "".join(ch for ch in condition_id if ch.isdigit())
    return condition_id.rstrip("0123456789"), int(digits) if digits else 0


def _group_key(record: ResultRecord, current: str) -> Tuple:
    return (
        record.tree, record.dataset_id, record.git_commit[:7], str(record.git_dirty),
        str(record.protocol), digest_state(record, current), str(record.n_train_refs),
    )


def group_section(records: Sequence[ResultRecord], current: str) -> List[str]:
    """
    Render one row per provenance group, listing its conditions.

    Parameters
    ----------
    records : Sequence[ResultRecord]
        Every loaded result.
    current : str
        Current annotation-predicate digest.

    Returns
    -------
    List[str]
        Markdown lines.
    """
    groups: Dict[Tuple, List[ResultRecord]] = {}
    for record in records:
        groups.setdefault(_group_key(record, current), []).append(record)
    rows = []
    for key, members in sorted(groups.items()):
        conditions = sorted({m.condition_id for m in members}, key=_condition_key)
        rows.append([*key, str(len(members)), ", ".join(conditions)])
    header = ["Tree", "Column", "Commit", "Dirty", "Protocol", "Annotation digest", "n_train_refs", "Files", "Conditions"]
    return ["## 1. Provenance groups", "", *_table(header, rows)]


def commit_section(infos: Sequence[CommitInfo], counts: Counter) -> List[str]:
    """
    Render every source commit with its reachability.

    Parameters
    ----------
    infos : Sequence[CommitInfo]
        One entry per distinct commit.
    counts : Counter
        Result files per full commit hash.

    Returns
    -------
    List[str]
        Markdown lines.
    """
    rows = [
        [f"`{i.commit[:7]}`", i.date, i.subject.replace("|", "\\|"), str(counts[i.commit]),
         "yes" if i.on_develop else "no", ", ".join(i.refs) if i.refs else ("**none**" if i.exists else "**missing**")]
        for i in sorted(infos, key=lambda info: (info.date, info.commit))
    ]
    header = ["Commit", "Date", "Subject", "Files", "On develop", "Local refs containing it"]
    return ["## 2. Source commits", "", *_table(header, rows)]


def environment_section(records: Sequence[ResultRecord]) -> List[str]:
    """
    Render the distinct library-version sets and model revisions.

    Parameters
    ----------
    records : Sequence[ResultRecord]
        Every loaded result.

    Returns
    -------
    List[str]
        Markdown lines.
    """
    versions = Counter(record.versions for record in records)
    models = Counter((str(record.model_key), record.hf_revision) for record in records)
    lines = ["## 3. Environment", ""]
    lines += _table(["Library versions", "Files"], [[f"`{v}`", str(n)] for v, n in sorted(versions.items())])
    lines += _table(["Model", "Revision", "Files"], [[m, f"`{r}`", str(n)] for (m, r), n in sorted(models.items())])
    return lines


def flag_section(records: Sequence[ResultRecord], current: str) -> List[str]:
    """
    Render every result that cannot be combined with current results.

    Parameters
    ----------
    records : Sequence[ResultRecord]
        Every loaded result.
    current : str
        Current annotation-predicate digest.

    Returns
    -------
    List[str]
        Markdown lines; states explicitly when nothing is flagged.
    """
    hashes = candidate_hashes(records)
    rows = [
        [r.tree, r.condition_id, r.dataset_id, r.pair_name, "; ".join(reasons)]
        for r in records if (reasons := flags(r, current, hashes))
    ]
    lines = ["## 4. Flagged results", ""]
    if not rows:
        return lines + ["None: every result is from a clean tree, current protocol, current D1 "
                        "population, current V1/V2 annotation predicates where they apply, and one "
                        "candidate hash per pair.", ""]
    return lines + _table(["Tree", "Condition", "Column", "Pair", "Reasons"], rows)


def _preamble(records: Sequence[ResultRecord], current: str) -> List[str]:
    trees = Counter(record.tree for record in records)
    return [
        "# Results inventory",
        "",
        "Generated by `scripts/results_inventory.py` from the stored result files: "
        + ", ".join(f"`{tree}/` {n}" for tree, n in sorted(trees.items())) + ".",
        "",
        f"- Current evaluation protocol: {EVALUATION_PROTOCOL_VERSION}. "
        f"Current annotation-predicate digest: `{current[:12]}…`.",
        "- **Annotation digest** matters only where V1 or V2 text is read on D1 or D2; elsewhere the",
        "  #136 change left the text byte-identical, so `absent` is expected.",
        f"- **n_train_refs** must be {_D1_KNOWN_PAIRS} on D1 (#137 known pairs); D5 records its training links.",
        "- **Local refs** are read from this clone, so they can differ on another machine. Squash merges",
        "  mean no result commit is an ancestor of `develop`; a commit with no ref may still be",
        "  reachable on GitHub through `refs/pull/<n>/head`.",
        "",
    ]


def build_report(records: Sequence[ResultRecord], infos: Sequence[CommitInfo]) -> List[str]:
    """
    Assemble the full inventory.

    Parameters
    ----------
    records : Sequence[ResultRecord]
        Every loaded result.
    infos : Sequence[CommitInfo]
        One entry per distinct source commit.

    Returns
    -------
    List[str]
        Markdown lines.
    """
    current = annotation_predicates_digest()
    counts = Counter(record.git_commit for record in records)
    return [
        *_preamble(records, current), *group_section(records, current), *commit_section(infos, counts),
        *environment_section(records), *flag_section(records, current),
    ]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inventory the provenance of stored results.")
    parser.add_argument(
        "--results_dirs", nargs="+", default=["data/results/", "data/results_ngram_baseline/"],
        help="Results directories to read.",
    )
    parser.add_argument("--output", default="artifacts/issue-138/results_inventory.md", help="Output path.")
    return parser


def main() -> None:
    """Write the results inventory."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    records = [record for root in args.results_dirs for record in load_records(Path(root))]
    infos = [commit_info(commit) for commit in sorted({record.git_commit for record in records})]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(build_report(records, infos)), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
