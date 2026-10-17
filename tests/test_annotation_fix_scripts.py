"""Tests for the #136 inventory and before/after comparison scripts.

Dumps and result files are written to a temporary directory, so nothing reads
the repository's ``data/`` or loads a dataset or model.
"""

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.annotation_fix_report as report  # noqa: E402
import scripts.annotation_inventory as inventory  # noqa: E402
from kgsemembed.verbalisation.base import annotation_predicates_digest  # noqa: E402


def _dump(texts: dict) -> dict:
    """Build a dump whose records are derived from ``{pair: {side: {uri: (v1, v2)}}}``."""
    pairs = {
        pair: {side: {uri: inventory.text_record(list(t)) for uri, t in uris.items()} for side, uris in sides.items()}
        for pair, sides in texts.items()
    }
    return {"code": {"git_commit": "abc"}, "pairs": pairs}


def test_compare_counts_changed_texts_and_sums_d3_pairs() -> None:
    old = _dump({
        "D3/a-b": {"source": {"s": ("x", "Label: x.")}, "candidate": {"t": ("y", "Label: y.")}},
        "D3/a-c": {"source": {"s": ("x", "Label: x.")}, "candidate": {"t": ("z", "Label: z.")}},
    })
    new = _dump({
        "D3/a-b": {"source": {"s": ("x ; w", "Label: x.")}, "candidate": {"t": ("y", "Label: y.")}},
        "D3/a-c": {"source": {"s": ("x", "Label: x.")}, "candidate": {"t": ("z", "Label: z.")}},
    })
    totals = inventory.compare_counts(old, new)
    assert totals[("D3", "source", "V1")] == [2, 1, 2, 4]
    assert totals[("D3", "source", "V2")][1] == 0
    assert totals[("D3", "candidate", "V1")][:2] == [2, 0]


def test_compare_refuses_different_entities() -> None:
    old = _dump({"D2/p": {"source": {"s": ("x", "x")}, "candidate": {}}})
    new = _dump({"D2/p": {"source": {"other": ("x", "x")}, "candidate": {}}})
    with pytest.raises(ValueError, match="different entities"):
        inventory.compare_counts(old, new)


def test_affected_conditions_are_the_v1_v2_strategies_on_d1_d2() -> None:
    ids = {c.condition_id for c in report.affected_conditions()}
    assert ids == {"C1", "C2", "C3", "C6", "C10", "C11", "C13", "C15", "C16", "C17", "C18", "C19", "C20"}


def _write(results_dir: Path, condition_id: str, dataset_id: str, mrr: float, **extra) -> None:
    path = results_dir / condition_id / dataset_id / f"{report._PAIRS[dataset_id]}_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"condition_id": condition_id, "dataset_id": dataset_id, "git_commit": "0b99aef0",
               "git_dirty": False, "metrics": {"mrr": mrr, "recall_at_1": mrr, "f1": mrr}, **extra}
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_comparison_rows_report_differences_and_missing_cells(tmp_path: Path) -> None:
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    _write(old_dir, "C1", "D2", 0.8)
    _write(new_dir, "C1", "D2", 0.9, annotation_predicates_sha256=annotation_predicates_digest())
    rows = report.comparison_rows(old_dir, new_dir)
    assert rows["rows"] == [
        "| C1 | V1 | M1 | D2 | 0.8000 → 0.9000 (+0.1000) | 0.8000 → 0.9000 (+0.1000) "
        "| 0.8000 → 0.9000 (+0.1000) |"
    ]
    assert "C1/D1" in rows["missing"]


def test_comparison_refuses_a_rerun_with_another_annotation_digest(tmp_path: Path) -> None:
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    _write(old_dir, "C1", "D2", 0.8)
    _write(new_dir, "C1", "D2", 0.9, annotation_predicates_sha256="0" * 64)
    with pytest.raises(ValueError, match="annotation digest differs"):
        report.comparison_rows(old_dir, new_dir)
