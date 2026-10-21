"""Tests for the Section 4.3 entity-type table script.

Result files are written to a temporary directory and read back through the
real ``load_all_results``, so filtering, weighting and rendering are exercised
end to end without touching the repository's ``data/results``.
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.generate_entity_type_table as script  # noqa: E402
from kgsemembed.pipeline.conditions import get_condition  # noqa: E402
from scripts.generate_entity_type_table import build_table_rows  # noqa: E402

_METRIC_NAMES = (
    "precision", "recall", "threshold", "mrr", "recall_at_1", "recall_at_5", "recall_at_10"
)


def _metrics(f1: float, n_refs: Optional[int] = None) -> dict:
    metrics = {"f1": f1, **{name: 0.5 for name in _METRIC_NAMES}}
    if n_refs is not None:
        metrics["n_refs"] = n_refs
    return metrics


def _write_result(
    results_dir: Path,
    condition_id: str,
    pair_name: str,
    per_entity_type: dict,
    dataset_id: str = "D3",
) -> None:
    """Write one result file whose buckets map entity type to ``(f1, n_refs)``."""
    condition = get_condition(condition_id)
    payload = {
        "condition_id": condition_id,
        "dataset_id": dataset_id,
        "pair_name": pair_name,
        "strategy": condition.strategy_name,
        "model_key": condition.model_key,
        "model_id": f"registry/{condition.model_key}",
        "metrics": _metrics(0.5),
        "per_entity_type": {
            etype: _metrics(f1, n_refs) for etype, (f1, n_refs) in per_entity_type.items()
        },
        "n_source_entities": 100,
        "n_candidates_per_entity": 20,
    }
    directory = results_dir / condition_id / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{pair_name}_results.json").write_text(json.dumps(payload))


def _run_cli(argv: List[str]) -> None:
    original = sys.argv
    sys.argv = ["generate_entity_type_table.py", *argv]
    try:
        script.main()
    finally:
        sys.argv = original


def _cells(row: str) -> List[str]:
    return [cell.strip() for cell in row.removesuffix("\\\\").split("&")]


def test_f1_is_weighted_by_reference_count(tmp_path: Path) -> None:
    """A pair with three times the references counts three times as much."""
    _write_result(tmp_path, "C1", "a", {"class": (0.8, 3), "predicate": (0.2, 4)})
    _write_result(tmp_path, "C1", "b", {"class": (0.4, 9), "predicate": (0.6, 4)})
    [row] = build_table_rows(str(tmp_path))
    assert _cells(row) == ["C1", "M1", "0.5000", "0.4000", "+0.1000", "2"]


def test_model_key_comes_from_condition_registry(tmp_path: Path) -> None:
    """The model column reflects the registered condition, here C13 on M5."""
    _write_result(tmp_path, "C13", "a", {"class": (0.1, 5), "predicate": (0.3, 5)})
    [row] = build_table_rows(str(tmp_path))
    assert _cells(row)[:2] == ["C13", "M5"]
    assert _cells(row)[4] == "-0.2000"


def test_rows_follow_paper_order(tmp_path: Path) -> None:
    """Rows are emitted in the fixed table order, not file or numeric order."""
    for condition_id in ("C18", "C10", "C2"):
        _write_result(tmp_path, condition_id, "a", {"class": (0.5, 5), "predicate": (0.5, 5)})
    rows = build_table_rows(str(tmp_path))
    assert [_cells(row)[0] for row in rows] == ["C2", "C10", "C18"]


def test_buckets_below_three_refs_are_excluded(tmp_path: Path) -> None:
    """A two-reference bucket neither shifts the mean nor counts as a pair."""
    _write_result(tmp_path, "C1", "a", {"class": (0.9, 3), "predicate": (0.3, 3)})
    _write_result(tmp_path, "C1", "b", {"class": (0.0, 2), "predicate": (0.0, 2)})
    [row] = build_table_rows(str(tmp_path))
    assert _cells(row)[2:] == ["0.9000", "0.3000", "+0.6000", "1"]


def test_condition_missing_an_entity_type_is_omitted(tmp_path: Path) -> None:
    """Without predicate data a condition cannot report a gap and is dropped."""
    _write_result(tmp_path, "C1", "a", {"class": (0.9, 5)})
    _write_result(tmp_path, "C2", "a", {"class": (0.5, 5), "predicate": (0.5, 5)})
    rows = build_table_rows(str(tmp_path))
    assert [_cells(row)[0] for row in rows] == ["C2"]


def test_other_datasets_are_ignored(tmp_path: Path) -> None:
    """D4_schema also reports buckets but does not belong in the D3 table."""
    _write_result(tmp_path, "C1", "a", {"class": (0.5, 5), "predicate": (0.5, 5)})
    _write_result(
        tmp_path, "C1", "d4", {"class": (0.0, 50), "predicate": (0.0, 50)}, dataset_id="D4_schema"
    )
    [row] = build_table_rows(str(tmp_path))
    assert _cells(row)[2:4] == ["0.5000", "0.5000"]


def test_conditions_outside_the_table_are_ignored(tmp_path: Path) -> None:
    """A registered condition not in the paper table yields no row."""
    _write_result(tmp_path, "C5", "a", {"class": (0.5, 5), "predicate": (0.5, 5)})
    assert build_table_rows(str(tmp_path)) == []


def test_malformed_result_file_is_skipped(tmp_path: Path) -> None:
    """An unreadable file is skipped rather than aborting the table."""
    _write_result(tmp_path, "C1", "a", {"class": (0.5, 5), "predicate": (0.5, 5)})
    (tmp_path / "C1" / "D3" / "broken_results.json").write_text("{not json")
    assert len(build_table_rows(str(tmp_path))) == 1


def test_empty_results_dir_yields_no_rows(tmp_path: Path) -> None:
    """With no results the table is empty rather than an error."""
    assert build_table_rows(str(tmp_path)) == []


def test_cli_prints_header_then_rows(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The CLI prints the paste marker followed by one LaTeX row per condition."""
    _write_result(tmp_path, "C1", "a", {"class": (0.75, 4), "predicate": (0.25, 4)})
    _run_cli(["--results_dir", str(tmp_path)])
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "% Paste into Table~\\ref{tab:entity_type}",
        "C1 & M1 & 0.7500 & 0.2500 & +0.5000 & 1 \\\\",
    ]
