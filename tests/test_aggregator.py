"""Tests for the experiment results aggregator and Markdown report generator.

All tests operate on synthetic result files written into a temporary directory;
no real experiment outputs, embedding models, or repository datasets are used.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import kgsemembed.evaluation as evaluation
from kgsemembed.evaluation.aggregator import (
    _RESULT_COLUMNS,
    _SUMMARY_COLUMNS,
    _classify_entity_type,
    build_condition_summary_table,
    build_dataset_breakdown_table,
    generate_markdown_report,
    load_all_results,
)
from kgsemembed.pipeline.conditions import EXPERIMENT_CONDITIONS, get_condition

# The column contracts stated verbatim in the GitHub issue; asserting against
# these literals (rather than the module constants) guards the public schema.
_EXPECTED_RESULT_COLUMNS = [
    "condition_id",
    "dataset_id",
    "pair_name",
    "strategy",
    "model_key",
    "model_id",
    "f1",
    "precision",
    "recall",
    "threshold",
    "mrr",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "n_source_entities",
    "n_candidates_per_entity",
    "ablation_group",
]
_EXPECTED_SUMMARY_COLUMNS = [
    "condition_id",
    "strategy",
    "model_key",
    "mean_f1",
    "std_f1",
    "mean_mrr",
    "mean_recall_at_1",
    "mean_recall_at_5",
    "mean_recall_at_10",
    "n_pairs",
]

_METRIC_KEYS = (
    "f1",
    "precision",
    "recall",
    "threshold",
    "mrr",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
)


# ---------------------------------------------------------------------------
# Synthetic result fixtures
# ---------------------------------------------------------------------------
def _metrics(f1: float) -> Dict[str, float]:
    return {key: f1 for key in _METRIC_KEYS}


def _write_result(
    results_dir: Path,
    condition_id: str,
    dataset_id: str,
    pair_name: str,
    f1: float,
    *,
    overrides: Optional[dict] = None,
    drop: Optional[str] = None,
) -> Path:
    directory = results_dir / condition_id / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    condition = get_condition(condition_id)
    payload = {
        "condition_id": condition_id,
        "dataset_id": dataset_id,
        "pair_name": pair_name,
        "strategy": condition.strategy_name,
        "model_key": condition.model_key,
        "model_id": f"registry/{condition.model_key}",
        "metrics": _metrics(f1),
        "n_source_entities": 100,
        "n_candidates_per_entity": 20,
    }
    if overrides:
        payload.update(overrides)
    if drop:
        payload.pop(drop, None)
    path = directory / f"{pair_name}_results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _pair_name(dataset_id: str, index: int) -> str:
    if dataset_id == "D3":
        return f"{dataset_id}_class_p{index}"
    if dataset_id == "D4":
        return f"{dataset_id}_instance_p{index}" if index % 2 else f"{dataset_id}_predicate_p{index}"
    return f"{dataset_id}_p{index}"


def _write_all_conditions(results_dir: Path, pairs_per_dataset: int = 6) -> None:
    for index, condition in enumerate(EXPERIMENT_CONDITIONS):
        base = 0.5 + index * 0.001
        for dataset_id in condition.datasets:
            for pair_index in range(pairs_per_dataset):
                _write_result(
                    results_dir,
                    condition.condition_id,
                    dataset_id,
                    _pair_name(dataset_id, pair_index),
                    round(base + pair_index * 0.0001, 6),
                )


@pytest.fixture()
def full_results_dir(tmp_path: Path) -> Path:
    results_dir = tmp_path / "results"
    _write_all_conditions(results_dir)
    return results_dir


def _synthetic_stats() -> List[dict]:
    return [
        {
            "condition_a": "C1",
            "condition_b": "C2",
            "n_pairs": 5,
            "p_value": 0.0321,
            "delta_f1": 0.0187,
            "corrected_significant": True,
        }
    ]


# ---------------------------------------------------------------------------
# Package exports (regression)
# ---------------------------------------------------------------------------
def test_package_exports() -> None:
    expected = {
        "load_all_results",
        "build_condition_summary_table",
        "build_dataset_breakdown_table",
        "generate_markdown_report",
    }
    assert expected.issubset(set(evaluation.__all__))
    for name in expected:
        assert hasattr(evaluation, name)


def test_existing_exports_unchanged() -> None:
    for name in ("compute_all_metrics", "run_group_comparisons", "export_stats_table"):
        assert hasattr(evaluation, name)


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------
def test_module_constants_match_specification() -> None:
    assert list(_RESULT_COLUMNS) == _EXPECTED_RESULT_COLUMNS
    assert list(_SUMMARY_COLUMNS) == _EXPECTED_SUMMARY_COLUMNS


def test_recursive_discovery_and_schema(full_results_dir: Path) -> None:
    df = load_all_results(str(full_results_dir))
    assert list(df.columns) == _EXPECTED_RESULT_COLUMNS
    assert not df.empty
    assert set(df["condition_id"]) == {c.condition_id for c in EXPERIMENT_CONDITIONS}


def test_metrics_are_flattened(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "D1", "D1_p0", 0.75)
    df = load_all_results(str(tmp_path))
    assert df.loc[0, "f1"] == pytest.approx(0.75)
    assert df.loc[0, "model_id"] == "registry/M1"
    assert df.loc[0, "n_candidates_per_entity"] == 20


def test_malformed_json_skipped_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_result(tmp_path, "C1", "D1", "D1_p0", 0.5)
    bad_dir = tmp_path / "C2" / "D1"
    bad_dir.mkdir(parents=True)
    (bad_dir / "D1_p0_results.json").write_text("{not valid json", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        df = load_all_results(str(tmp_path))
    assert len(df) == 1
    assert df.loc[0, "condition_id"] == "C1"
    assert any("malformed" in message.lower() for message in caplog.messages)


def test_missing_required_field_skipped_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_result(tmp_path, "C1", "D1", "D1_p0", 0.5)
    _write_result(tmp_path, "C2", "D1", "D1_p0", 0.5, drop="strategy")
    with caplog.at_level(logging.WARNING):
        df = load_all_results(str(tmp_path))
    assert list(df["condition_id"]) == ["C1"]
    assert any("missing required fields" in message.lower() for message in caplog.messages)


def test_missing_metric_skipped(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "D1", "D1_p0", 0.5)
    _write_result(
        tmp_path, "C2", "D1", "D1_p1", 0.5, overrides={"metrics": {"precision": 0.5}}
    )
    df = load_all_results(str(tmp_path))
    assert list(df["condition_id"]) == ["C1"]


def test_unknown_condition_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_result(tmp_path, "C1", "D1", "D1_p0", 0.5)
    _write_result(tmp_path, "C1", "D1", "D1_p1", 0.5, overrides={"condition_id": "C999"})
    with caplog.at_level(logging.WARNING):
        df = load_all_results(str(tmp_path))
    assert list(df["condition_id"]) == ["C1"]
    assert any("unknown condition" in message.lower() for message in caplog.messages)


def test_ablation_group_derived_from_registry(full_results_dir: Path) -> None:
    df = load_all_results(str(full_results_dir))
    for condition in EXPERIMENT_CONDITIONS:
        rows = df[df["condition_id"] == condition.condition_id]
        assert set(rows["ablation_group"]) == {condition.ablation_group}


def test_empty_directory_returns_schema(tmp_path: Path) -> None:
    empty = tmp_path / "results"
    empty.mkdir()
    df = load_all_results(str(empty))
    assert df.empty
    assert list(df.columns) == list(_RESULT_COLUMNS)


def test_missing_directory_returns_schema(tmp_path: Path) -> None:
    df = load_all_results(str(tmp_path / "does_not_exist"))
    assert df.empty
    assert list(df.columns) == list(_RESULT_COLUMNS)


def test_duplicate_combination_kept_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_result(tmp_path, "C1", "D1", "D1_p0", 0.4)
    nested = tmp_path / "C1" / "D1" / "nested"
    nested.mkdir(parents=True)
    payload = json.loads(
        (tmp_path / "C1" / "D1" / "D1_p0_results.json").read_text(encoding="utf-8")
    )
    (nested / "D1_p0_results.json").write_text(json.dumps(payload), encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        df = load_all_results(str(tmp_path))
    assert len(df) == 1
    assert any("duplicate" in message.lower() for message in caplog.messages)


# ---------------------------------------------------------------------------
# Condition summary
# ---------------------------------------------------------------------------
def test_condition_summary_row_count(full_results_dir: Path) -> None:
    df = load_all_results(str(full_results_dir))
    summary = build_condition_summary_table(df)
    assert list(summary.columns) == _EXPECTED_SUMMARY_COLUMNS
    assert len(summary) == len(EXPERIMENT_CONDITIONS) == 18


def test_condition_summary_sorted_descending(full_results_dir: Path) -> None:
    df = load_all_results(str(full_results_dir))
    summary = build_condition_summary_table(df)
    assert list(summary["mean_f1"]) == sorted(summary["mean_f1"], reverse=True)


def test_condition_summary_tie_breaks_on_condition_id(tmp_path: Path) -> None:
    for condition_id in ("C3", "C1", "C2"):
        for pair_index in range(3):
            _write_result(tmp_path, condition_id, "D1", f"D1_p{pair_index}", 0.6)
    summary = build_condition_summary_table(load_all_results(str(tmp_path)))
    assert list(summary["condition_id"]) == ["C1", "C2", "C3"]


def test_condition_summary_rounding_and_n_pairs(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "D1", "D1_p0", 0.123456)
    _write_result(tmp_path, "C1", "D1", "D1_p1", 0.234567)
    _write_result(tmp_path, "C1", "D2", "D2_p0", 0.345678)
    summary = build_condition_summary_table(load_all_results(str(tmp_path)))
    assert summary.loc[0, "n_pairs"] == 3
    for column in ("mean_f1", "std_f1", "mean_mrr"):
        value = summary.loc[0, column]
        assert round(value, 4) == value


def test_condition_summary_empty_frame() -> None:
    summary = build_condition_summary_table(
        pd.DataFrame(columns=list(_RESULT_COLUMNS))
    )
    assert summary.empty
    assert list(summary.columns) == list(_SUMMARY_COLUMNS)


# ---------------------------------------------------------------------------
# Dataset breakdown
# ---------------------------------------------------------------------------
def test_dataset_breakdown_columns_and_order(full_results_dir: Path) -> None:
    df = load_all_results(str(full_results_dir))
    breakdown = build_dataset_breakdown_table(df, "f1")
    assert list(breakdown.columns) == ["D1", "D2", "D3", "D4", "D5"]
    assert list(breakdown.index) == sorted(breakdown.index)


def test_dataset_breakdown_rounding(full_results_dir: Path) -> None:
    df = load_all_results(str(full_results_dir))
    breakdown = build_dataset_breakdown_table(df, "f1")
    for value in breakdown.to_numpy().ravel():
        if pd.notna(value):
            assert round(float(value), 4) == value


def test_dataset_breakdown_unsupported_metric(full_results_dir: Path) -> None:
    df = load_all_results(str(full_results_dir))
    with pytest.raises(ValueError, match="Unknown metric"):
        build_dataset_breakdown_table(df, "does_not_exist")


def test_dataset_breakdown_alternate_metric(full_results_dir: Path) -> None:
    df = load_all_results(str(full_results_dir))
    breakdown = build_dataset_breakdown_table(df, "mrr")
    assert list(breakdown.columns) == ["D1", "D2", "D3", "D4", "D5"]


def test_dataset_breakdown_empty_frame_returns_empty() -> None:
    empty = pd.DataFrame(columns=list(_RESULT_COLUMNS))
    breakdown = build_dataset_breakdown_table(empty, "f1")
    assert breakdown.empty


def test_dataset_breakdown_empty_frame_still_validates_metric() -> None:
    empty = pd.DataFrame(columns=list(_RESULT_COLUMNS))
    with pytest.raises(ValueError, match="Unknown metric"):
        build_dataset_breakdown_table(empty, "nope")


# ---------------------------------------------------------------------------
# Entity-type classification
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "pair_name, expected",
    [
        ("d3_classes_align", "Classes"),
        ("d4_predicate_map", "Predicates"),
        ("d4_property_links", "Predicates"),
        ("d4_relation_pairs", "Predicates"),
        ("d4_instance_pairs", "Instances"),
        ("d1_snomed_fma", None),
    ],
)
def test_classify_entity_type(pair_name: str, expected: Optional[str]) -> None:
    assert _classify_entity_type(pair_name) == expected


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _headings(text: str) -> List[str]:
    return [line for line in text.splitlines() if line.startswith("# ")]


def test_report_created_with_parent_dirs(full_results_dir: Path, tmp_path: Path) -> None:
    df = load_all_results(str(full_results_dir))
    output = tmp_path / "nested" / "deep" / "report.md"
    generate_markdown_report(df, _synthetic_stats(), str(output))
    assert output.exists()
    assert "| condition_id |" in output.read_text(encoding="utf-8")


def test_report_seven_sections_with_stats(full_results_dir: Path, tmp_path: Path) -> None:
    df = load_all_results(str(full_results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, _synthetic_stats(), str(output))
    headings = _headings(output.read_text(encoding="utf-8"))
    assert headings == [
        "# Executive Summary",
        "# Condition Summary Table",
        "# Per-Dataset Breakdown",
        "# Entity-Type Analysis",
        "# Ablation Group Analysis",
        "# Statistical Significance",
        "# PPAS Ablation",
    ]


def test_report_six_sections_without_stats(full_results_dir: Path, tmp_path: Path) -> None:
    df = load_all_results(str(full_results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    headings = _headings(output.read_text(encoding="utf-8"))
    assert len(headings) == 6
    assert "# Statistical Significance" not in headings


def test_report_is_deterministic(full_results_dir: Path, tmp_path: Path) -> None:
    df = load_all_results(str(full_results_dir))
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    generate_markdown_report(df, _synthetic_stats(), str(first))
    generate_markdown_report(df, _synthetic_stats(), str(second))
    assert first.read_bytes() == second.read_bytes()


def test_report_entity_type_labels(full_results_dir: Path, tmp_path: Path) -> None:
    df = load_all_results(str(full_results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    text = output.read_text(encoding="utf-8")
    assert "Classes" in text
    assert "Predicates" in text
    assert "Instances" in text


def test_report_handles_empty_results(tmp_path: Path) -> None:
    df = load_all_results(str(tmp_path / "missing"))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    text = output.read_text(encoding="utf-8")
    assert "# Executive Summary" in text
    assert "_No results available._" in text


def test_report_empty_stats_list_includes_section(full_results_dir: Path, tmp_path: Path) -> None:
    df = load_all_results(str(full_results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, [], str(output))
    text = output.read_text(encoding="utf-8")
    assert "# Statistical Significance" in text
    assert "_No statistical comparison results available._" in text


def test_report_entity_type_missing_subtask_note(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    for pair_index in range(6):
        _write_result(results_dir, "C1", "D3", f"D3_generic_p{pair_index}", 0.5)
    df = load_all_results(str(results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    text = output.read_text(encoding="utf-8")
    assert "_No entity-type sub-task data available for D3._" in text


def test_report_per_dataset_missing_note(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    for pair_index in range(6):
        _write_result(results_dir, "C1", "D1", f"D1_p{pair_index}", 0.5)
    df = load_all_results(str(results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    text = output.read_text(encoding="utf-8")
    assert "_No results available for D5._" in text


def test_report_ppas_values_from_results(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    for pair_index in range(6):
        _write_result(results_dir, "C5", "D5", f"D5_p{pair_index}", 0.40)
        _write_result(results_dir, "C14", "D5", f"D5_p{pair_index}", 0.55)
    df = load_all_results(str(results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    text = output.read_text(encoding="utf-8")
    assert "| C5 vs C14 | D5 | 0.4000 | 0.5500 | 0.1500 |" in text
    assert "| C10 vs C15 | D1 | N/A | N/A | N/A |" in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _run_cli(argv: List[str]) -> None:
    from scripts.generate_report import main

    original = sys.argv
    sys.argv = ["generate_report.py", *argv]
    try:
        main()
    finally:
        sys.argv = original


def test_cli_end_to_end(full_results_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    output = tmp_path / "report.md"
    _run_cli(["--results_dir", str(full_results_dir), "--output", str(output)])
    assert output.exists()
    assert "| condition_id |" in output.read_text(encoding="utf-8")
    assert str(output) in capsys.readouterr().out


def test_cli_group_failure_is_resilient(
    full_results_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.generate_report as script

    def fake_run_group_comparisons(group: str, results_dir: str) -> List[dict]:
        if group == "A":
            return _synthetic_stats()
        raise ValueError(f"insufficient data for group {group}")

    monkeypatch.setattr(script, "run_group_comparisons", fake_run_group_comparisons)
    output = tmp_path / "report.md"
    _run_cli(["--results_dir", str(full_results_dir), "--output", str(output)])
    text = output.read_text(encoding="utf-8")
    assert "# Statistical Significance" in text
    assert "| C1 | C2 |" in text
