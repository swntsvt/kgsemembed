"""Tests for the Section 4.3 entity-type table script and its shared summary (#166).

Result files are written to a temporary directory and read back through the
real ``load_all_results``, so filtering, aggregation and rendering are
exercised end to end without touching the repository's ``data/results``.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.generate_entity_type_table as script  # noqa: E402
from kgsemembed.evaluation import entity_types, load_all_results  # noqa: E402
from kgsemembed.evaluation.aggregator import generate_markdown_report  # noqa: E402
from kgsemembed.pipeline.conditions import get_condition  # noqa: E402
from scripts.generate_entity_type_table import build_table  # noqa: E402

_METRIC_NAMES = (
    "precision", "recall", "threshold", "recall_at_1", "recall_at_5", "recall_at_10"
)
Bucket = Tuple[float, int, Tuple[int, int, int]]


def _bucket(value: float, n_refs: int, counts: Tuple[int, int, int] = (1, 1, 1)) -> dict:
    """Metrics of one bucket: MRR = ``value``, Hits@1 = value / 2, F1 = value / 4."""
    return {
        **{name: 0.5 for name in _METRIC_NAMES},
        "mrr": value, "hits_at_1": value / 2, "f1": value / 4, "n_refs": n_refs,
        **dict(zip(entity_types.COUNT_FIELDS, counts)),
    }


def _write_result(
    results_dir: Path, condition_id: str, pair_name: str, buckets: Dict[str, Bucket],
    dataset_id: str = "D3",
) -> None:
    """Write one result whose ``per_entity_type`` holds ``buckets``."""
    condition = get_condition(condition_id)
    payload = {
        "condition_id": condition_id, "dataset_id": dataset_id, "pair_name": pair_name,
        "strategy": condition.strategy_name, "model_key": condition.model_key,
        "model_id": f"registry/{condition.model_key}",
        "metrics": {**{name: 0.5 for name in _METRIC_NAMES}, "mrr": 0.5, "f1": 0.5},
        "per_entity_type": {etype: _bucket(*spec) for etype, spec in buckets.items()},
        "n_source_entities": 100, "n_candidates_per_entity": 20,
    }
    directory = results_dir / condition_id / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{pair_name}_results.json").write_text(json.dumps(payload))


def _summary(results_dir: Path):
    return entity_types.entity_type_summary(
        load_all_results(str(results_dir), include_entity_type_breakdown=True)
    )


def _rows(results_dir: Path) -> List[List[str]]:
    """Return the cells of every condition row of the LaTeX table."""
    lines = build_table(str(results_dir))
    return [
        [cell.strip() for cell in line.removesuffix(r"\\").split("&")]
        for line in lines if re.match(r"C\d+ &", line)
    ]


def test_values_are_unweighted_means_over_pairs(tmp_path: Path) -> None:
    """D3's convention: each pair counts once whatever its reference count."""
    _write_result(tmp_path, "C1", "a", {"class": (0.8, 3), "predicate": (0.2, 4)})
    _write_result(tmp_path, "C1", "b", {"class": (0.4, 9), "predicate": (0.6, 4)})
    row = _summary(tmp_path).iloc[0]
    assert row["class_mrr"] == pytest.approx(0.6)
    assert row["class_hits_at_1"] == pytest.approx(0.3)
    assert row["class_f1"] == pytest.approx(0.15)
    assert (row["class_n_pairs"], row["class_n_refs"]) == (2, 12)
    assert row["mrr_gap"] == pytest.approx(0.6 - 0.4)


def test_micro_f1_pools_counts_over_pairs(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "a", {"class": (0.8, 4, (3, 1, 1)), "predicate": (0.2, 4, (0, 2, 4))})
    _write_result(tmp_path, "C1", "b", {"class": (0.4, 4, (1, 3, 3)), "predicate": (0.6, 4, (2, 0, 2))})
    row = _summary(tmp_path).iloc[0]
    assert row["class_f1_micro"] == pytest.approx(2 * 4 / (2 * 4 + 4 + 4))
    assert row["predicate_f1_micro"] == pytest.approx(2 * 2 / (2 * 2 + 2 + 6))
    assert entity_types.micro_f1(0, 0, 0) == 0.0


def test_buckets_below_three_refs_are_excluded(tmp_path: Path) -> None:
    """A two-reference bucket neither shifts the mean nor counts as a pair."""
    _write_result(tmp_path, "C1", "a", {"class": (0.9, 3), "predicate": (0.3, 3)})
    _write_result(tmp_path, "C1", "b", {"class": (0.0, 2), "predicate": (0.0, 2)})
    row = _summary(tmp_path).iloc[0]
    assert (row["class_mrr"], row["class_n_pairs"], row["class_n_refs"]) == (pytest.approx(0.9), 1, 3)


def test_condition_run_on_d3_only(tmp_path: Path) -> None:
    _write_result(tmp_path, "C4", "a", {"class": (0.5, 5), "predicate": (0.25, 5)})
    assert list(_summary(tmp_path)["condition_id"]) == ["C4"]
    assert [row[0] for row in _rows(tmp_path)] == ["C4"]


def test_condition_run_on_d4_schema_only_has_no_row(tmp_path: Path) -> None:
    """C5 reports buckets on D4_schema only; the D3 analysis leaves it out."""
    _write_result(tmp_path, "C5", "d4", {"class": (0.5, 12), "predicate": (0.9, 32)},
                  dataset_id="D4_schema")
    assert _summary(tmp_path).empty
    assert build_table(str(tmp_path)) == []


def test_condition_run_on_both_uses_d3_alone(tmp_path: Path) -> None:
    """C1's D4_schema buckets never enter its D3 values or counts."""
    _write_result(tmp_path, "C1", "a", {"class": (0.5, 5), "predicate": (0.25, 5)})
    _write_result(tmp_path, "C1", "d4", {"class": (0.0, 12), "predicate": (0.0, 32)},
                  dataset_id="D4_schema")
    row = _summary(tmp_path).iloc[0]
    assert (row["class_mrr"], row["class_n_refs"], row["predicate_n_refs"]) == (0.5, 5, 5)


def test_condition_missing_an_entity_type_is_omitted_from_the_table(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "a", {"class": (0.9, 5)})
    _write_result(tmp_path, "C2", "a", {"class": (0.5, 5), "predicate": (0.5, 5)})
    assert list(_summary(tmp_path)["condition_id"]) == ["C1", "C2"]
    assert [row[0] for row in _rows(tmp_path)] == ["C2"]


def test_rows_follow_registry_order(tmp_path: Path) -> None:
    for condition_id in ("C20", "C10", "C2"):
        _write_result(tmp_path, condition_id, "a", {"class": (0.5, 5), "predicate": (0.5, 5)})
    assert [row[0] for row in _rows(tmp_path)] == ["C2", "C10", "C20"]


def test_table_row_cells_and_caption_counts(tmp_path: Path) -> None:
    _write_result(tmp_path, "C13", "a", {"class": (0.8, 6, (2, 2, 4)), "predicate": (0.4, 4, (1, 1, 3))})
    lines = build_table(str(tmp_path))
    assert _rows(tmp_path) == [[
        "C13", "V2+V8", "M5", "0.8000", "0.4000", "0.4000", "0.2000", "0.2000", "0.1000",
        "0.4000", "0.3333",
    ]]
    caption = next(line for line in lines if line.startswith(r"\caption"))
    assert "class values rest on 1 pairs and 6 test references" in caption
    assert "predicate values on 1 pairs and 4." in caption
    assert r"\label{tab:entity_type}" in lines


def test_count_text_names_a_range_when_conditions_disagree() -> None:
    import pandas as pd

    assert entity_types.count_text(pd.Series([8, 8, float("nan")])) == "8"
    assert entity_types.count_text(pd.Series([7, 8])) == "7–8"


def test_report_and_script_print_identical_values(tmp_path: Path) -> None:
    """Every value of the LaTeX table appears in the report row of its condition."""
    results_dir = tmp_path / "results"
    _write_result(results_dir, "C1", "a", {"class": (0.123456, 5, (1, 2, 3)), "predicate": (0.98765, 4)})
    _write_result(results_dir, "C1", "b", {"class": (0.33335, 3), "predicate": (0.11115, 3)})
    _write_result(results_dir, "C1", "d4", {"class": (0.9, 12), "predicate": (0.9, 32)},
                  dataset_id="D4_schema")
    _write_result(results_dir, "C3", "a", {"class": (0.45, 5), "predicate": (0.55, 5, (2, 0, 1))})
    df = load_all_results(str(results_dir), include_entity_type_breakdown=True)
    generate_markdown_report(df, None, str(tmp_path / "report.md"))
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    section = text.split("# Entity-Type Analysis", 1)[1].split("\n# ", 1)[0]
    report_rows = {
        line.split(" | ")[0].lstrip("| "): line.strip("| ").split(" | ")
        for line in section.splitlines() if line.startswith("| C")
    }
    for cells in _rows(results_dir):
        report = report_rows[cells[0]]
        assert [report[i] for i in (5, 11, 6, 12, 7, 13, 8, 14)] == cells[3:]


def test_malformed_result_file_is_skipped(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "a", {"class": (0.5, 5), "predicate": (0.5, 5)})
    (tmp_path / "C1" / "D3" / "broken_results.json").write_text("{not json")
    assert len(_rows(tmp_path)) == 1


def test_empty_results_dir_yields_no_table(tmp_path: Path) -> None:
    assert build_table(str(tmp_path)) == []


def test_cli_prints_the_table(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _write_result(tmp_path, "C1", "a", {"class": (0.75, 4), "predicate": (0.25, 4)})
    original = sys.argv
    sys.argv = ["generate_entity_type_table.py", "--results_dir", str(tmp_path)]
    try:
        script.main()
    finally:
        sys.argv = original
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == r"\begin{table*}[t]" and lines[-1] == r"\end{table*}"
    assert any(line.startswith("C1 & V1 & M1 & 0.7500 & 0.2500") for line in lines)
