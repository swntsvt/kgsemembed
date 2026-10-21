"""Tests for the #145 V8 inventory and before/after comparison scripts.

Dumps and result files are written to a temporary directory, so nothing reads
the repository's ``data/`` or loads a dataset or model.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.v8_fix_report as report  # noqa: E402
import scripts.v8_inventory as inventory  # noqa: E402
from kgsemembed.verbalisation.v8 import v8_exclusions_digest  # noqa: E402

_FMA_SYNONYM = "http://purl.org/sig/ont/fma/synonym"
_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
_COMMENT = "http://www.w3.org/2000/01/rdf-schema#comment"


def _dump(texts: dict) -> dict:
    """Build a dump from ``{pair: {side: {uri: text}}}``."""
    pairs = {
        pair: {side: {uri: inventory.text_record(t) for uri, t in uris.items()} for side, uris in sides.items()}
        for pair, sides in texts.items()
    }
    return {"code": {"git_commit": "abc"}, "pairs": pairs}


def test_compare_counts_changed_texts_and_sums_d3_pairs() -> None:
    old = _dump({
        "D3/a-b": {"source": {"s": "Outgoing relationships: synonym."}, "candidate": {"t": "x"}},
        "D3/a-c": {"source": {"s": "x"}, "candidate": {"t": "y"}},
    })
    new = _dump({
        "D3/a-b": {"source": {"s": "Outgoing relationships: none."}, "candidate": {"t": "x"}},
        "D3/a-c": {"source": {"s": "x"}, "candidate": {"t": "y"}},
    })
    totals = inventory.compare_counts(old, new)
    assert totals[("D3", "source")] == [2, 1, 4, 4]
    assert totals[("D3", "candidate")][:2] == [2, 0]


def test_compare_refuses_different_entities() -> None:
    old = _dump({"D2/p": {"source": {"s": "x"}, "candidate": {}}})
    new = _dump({"D2/p": {"source": {"other": "x"}, "candidate": {}}})
    with pytest.raises(ValueError, match="different entities"):
        inventory.compare_counts(old, new)


def test_classification_marks_new_and_earlier_exclusions() -> None:
    counts = Counter({
        ("D1", "candidate", "out", _FMA_SYNONYM): 5,
        ("D1", "candidate", "out", _SUBCLASS): 7,
        ("D3", "source", "out", _COMMENT): 2,
    })
    lines = inventory.classification_lines(counts, declared={_FMA_SYNONYM})
    text = "\n".join(lines)
    assert f"| D1 | candidate | out | `{_FMA_SYNONYM}` | roles_v1_annotation | kept | 5 |" in text
    assert f"| D3 | source | out | `{_COMMENT}` | roles_v1_annotation | excluded | 2 |" in text
    assert "| D1 | candidate | 1 |" in text
    assert "Declared `owl:AnnotationProperty` but kept by V8: none." in text


def test_classification_lists_a_declared_annotation_property_v8_keeps() -> None:
    counts = Counter({("D1", "source", "out", _SUBCLASS): 1})
    text = "\n".join(inventory.classification_lines(counts, declared={_SUBCLASS}))
    assert f"kept by V8: `{_SUBCLASS}`." in text


def test_affected_conditions_are_the_v8_strategies() -> None:
    ids = {c.condition_id for c in report.affected_conditions()}
    assert ids == {"C9", "C10", "C11", "C13", "C15", "C16", "C18", "C19"}


def test_condition_columns_skip_d3_and_expand_d4() -> None:
    c9 = next(c for c in report.affected_conditions() if c.condition_id == "C9")
    assert report._condition_columns(c9) == ["D1", "D2", "D4_schema", "D4_instance", "D5"]


def _write(results_dir: Path, condition_id: str, column: str, value: float, **extra) -> None:
    path = results_dir / condition_id / column / f"{column.lower()}_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "condition_id": condition_id, "dataset_id": column, "pair_name": column.lower(),
        "git_commit": "8b7b3eeb", "git_dirty": False, "evaluation_protocol": 2,
        "metrics": {"mrr": value, "hits_at_1": value, "f1": value}, **extra,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _rerun(results_dir: Path, condition_id: str, column: str, value: float) -> None:
    _write(results_dir, condition_id, column, value, v8_exclusions_sha256=v8_exclusions_digest())


def test_comparison_rows_report_differences_missing_and_reproduction(tmp_path: Path) -> None:
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    _write(old_dir, "C11", "D2", 0.8)
    _rerun(new_dir, "C11", "D2", 0.9)
    _write(old_dir, "C9", "D4_schema", 0.5)
    _rerun(new_dir, "C9", "D4_schema", 0.6)
    rows = report.comparison_rows(old_dir, new_dir, counts={})
    assert "| C11 | V2+V8 | M4 | D2 | 0.8000 → 0.9000 (+0.1000) | 0.8000 → 0.9000 (+0.1000) " \
           "| 0.8000 → 0.9000 (+0.1000) |" in rows["rows"]
    assert "C11/D1" in rows["missing"]
    assert rows["not_reproduced"] == ["C9/D4_schema"]


def test_comparison_refuses_a_rerun_with_another_v8_digest(tmp_path: Path) -> None:
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    _write(old_dir, "C11", "D2", 0.8)
    _write(new_dir, "C11", "D2", 0.9, v8_exclusions_sha256="0" * 64)
    with pytest.raises(ValueError, match="V8 exclusion digest differs"):
        report.comparison_rows(old_dir, new_dir, counts={})


def test_comparison_refuses_a_dirty_rerun(tmp_path: Path) -> None:
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    _write(old_dir, "C11", "D2", 0.8)
    _write(new_dir, "C11", "D2", 0.9, v8_exclusions_sha256=v8_exclusions_digest(), git_dirty=True)
    with pytest.raises(ValueError, match="dirty tree"):
        report.comparison_rows(old_dir, new_dir, counts={})


def test_c20_c10_rows_use_stored_d3_and_reruns_elsewhere(tmp_path: Path) -> None:
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    for column in ("D2", "D3"):
        _write(old_dir, "C20", column, 0.8)
        _write(old_dir, "C10", column, 0.7)
    _rerun(new_dir, "C10", "D2", 0.85)
    rows = report.c20_c10_rows(old_dir, new_dir, counts={})
    assert rows[1].startswith("| D2 | -0.1000 → 0.0500 (+0.1500) |")
    assert rows[2].startswith("| D3 | -0.1000 → -0.1000 (+0.0000) |")
    assert rows[0] == "| D1 | — | — | — |"


def test_merged_view_replaces_only_rerun_columns(tmp_path: Path) -> None:
    old_dir, new_dir = tmp_path / "old", tmp_path / "new"
    for column in ("D2", "D3"):
        _write(old_dir, "C20", column, 0.8)
        _write(old_dir, "C10", column, 0.7)
    _rerun(new_dir, "C10", "D2", 0.85)
    view = report.merged_view(old_dir, new_dir, tmp_path / "view")
    read = lambda c, col: json.loads((view / c / col / f"{col.lower()}_results.json").read_text())["metrics"]["mrr"]
    assert (read("C10", "D2"), read("C10", "D3"), read("C20", "D2")) == (0.85, 0.7, 0.8)
    assert json.loads((old_dir / "C10" / "D2" / "d2_results.json").read_text())["metrics"]["mrr"] == 0.7
