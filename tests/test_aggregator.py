"""Tests for the experiment results aggregator and Markdown report generator.

All tests operate on synthetic result files written into a temporary directory;
no real experiment outputs, embedding models, or repository datasets are used.
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import kgsemembed.evaluation as evaluation
from kgsemembed.evaluation import export_stats_table
from kgsemembed.evaluation.aggregator import (
    _RESULT_COLUMNS,
    _STATS_COLUMNS,
    _SUMMARY_COLUMNS,
    _stats_row,
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


def _synthetic_stats(
    effect_size_r: Optional[float] = 0.7333,
    wilcoxon_warning: Optional[str] = None,
) -> List[dict]:
    return [
        {
            "condition_a": "C1",
            "condition_b": "C2",
            "n_pairs": 5,
            "p_value": 0.0321,
            "delta_f1": 0.0187,
            "corrected_significant": True,
            "effect_size_r": effect_size_r,
            "wilcoxon_warning": wilcoxon_warning,
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
# Per-entity-type breakdown rows
# ---------------------------------------------------------------------------
def _entity_type_metrics(f1: float, n_refs: int) -> Dict[str, float]:
    return {**_metrics(f1), "n_refs": n_refs}


def _write_mixed_result(results_dir: Path, condition_id: str = "C1") -> Path:
    return _write_result(
        results_dir,
        condition_id,
        "D3",
        "cmt-edas",
        0.17,
        overrides={
            "per_entity_type": {
                "class": _entity_type_metrics(0.21, 18),
                "predicate": _entity_type_metrics(0.08, 6),
            }
        },
    )


def test_breakdown_disabled_by_default(tmp_path: Path) -> None:
    _write_mixed_result(tmp_path)
    df = load_all_results(str(tmp_path))
    assert list(df.columns) == _EXPECTED_RESULT_COLUMNS
    assert len(df) == 1
    assert df.loc[0, "f1"] == pytest.approx(0.17)


def test_breakdown_widens_the_schema(tmp_path: Path) -> None:
    _write_mixed_result(tmp_path)
    df = load_all_results(str(tmp_path), include_entity_type_breakdown=True)
    assert list(df.columns) == _EXPECTED_RESULT_COLUMNS + ["entity_type", "n_refs"]


def test_breakdown_adds_one_row_per_entity_type(tmp_path: Path) -> None:
    _write_mixed_result(tmp_path)
    df = load_all_results(str(tmp_path), include_entity_type_breakdown=True)

    assert list(df["entity_type"]) == ["overall", "class", "predicate"]
    assert list(df["f1"]) == pytest.approx([0.17, 0.21, 0.08])
    assert list(df["n_refs"][1:]) == [18, 6]


def test_breakdown_rows_inherit_pair_identity(tmp_path: Path) -> None:
    _write_mixed_result(tmp_path)
    df = load_all_results(str(tmp_path), include_entity_type_breakdown=True)

    for column in ("condition_id", "dataset_id", "pair_name", "strategy", "ablation_group"):
        assert set(df[column]) == {df.loc[0, column]}


def test_breakdown_carries_every_metric(tmp_path: Path) -> None:
    _write_mixed_result(tmp_path)
    df = load_all_results(str(tmp_path), include_entity_type_breakdown=True)

    class_row = df[df["entity_type"] == "class"].iloc[0]
    for metric in _METRIC_KEYS:
        assert class_row[metric] == pytest.approx(0.21)


def test_breakdown_leaves_legacy_results_untouched(tmp_path: Path) -> None:
    """Results written before the breakdown existed must still load."""
    _write_result(tmp_path, "C1", "D1", "D1_p0", 0.5)
    df = load_all_results(str(tmp_path), include_entity_type_breakdown=True)

    assert len(df) == 1
    assert df.loc[0, "entity_type"] == "overall"
    assert pd.isna(df.loc[0, "n_refs"])


def test_breakdown_ignores_an_empty_per_entity_type(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "D1", "D1_p0", 0.5, overrides={"per_entity_type": {}})
    df = load_all_results(str(tmp_path), include_entity_type_breakdown=True)
    assert list(df["entity_type"]) == ["overall"]


def test_breakdown_ignores_a_malformed_per_entity_type(tmp_path: Path) -> None:
    _write_result(
        tmp_path, "C1", "D1", "D1_p0", 0.5, overrides={"per_entity_type": "class"}
    )
    df = load_all_results(str(tmp_path), include_entity_type_breakdown=True)
    assert list(df["entity_type"]) == ["overall"]


def test_breakdown_skips_a_non_mapping_bucket(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One malformed bucket must not abort the whole load."""
    _write_result(
        tmp_path,
        "C1",
        "D3",
        "cmt-edas",
        0.17,
        overrides={
            "per_entity_type": {
                "class": _entity_type_metrics(0.21, 18),
                "predicate": "not-a-mapping",
            }
        },
    )
    with caplog.at_level(logging.WARNING):
        df = load_all_results(str(tmp_path), include_entity_type_breakdown=True)

    assert list(df["entity_type"]) == ["overall", "class"]
    assert any("incomplete entity-type metrics" in message for message in caplog.messages)


def test_breakdown_skips_a_bucket_missing_a_metric(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_result(
        tmp_path,
        "C1",
        "D3",
        "cmt-edas",
        0.17,
        overrides={"per_entity_type": {"class": {"f1": 0.21, "n_refs": 18}}},
    )
    with caplog.at_level(logging.WARNING):
        df = load_all_results(str(tmp_path), include_entity_type_breakdown=True)

    assert list(df["entity_type"]) == ["overall"]
    assert any("incomplete entity-type metrics" in message for message in caplog.messages)


def test_breakdown_preserves_an_instance_bucket(tmp_path: Path) -> None:
    """Entity types are read from the result, not from a fixed class/predicate pair."""
    _write_result(
        tmp_path,
        "C1",
        "D3",
        "cmt-edas",
        0.17,
        overrides={"per_entity_type": {"instance": _entity_type_metrics(0.3, 7)}},
    )
    df = load_all_results(str(tmp_path), include_entity_type_breakdown=True)
    assert list(df["entity_type"]) == ["overall", "instance"]
    assert df.loc[1, "n_refs"] == 7


def test_breakdown_does_not_disturb_the_overall_row(tmp_path: Path) -> None:
    """The pair-level row must keep the top-level metrics, not a bucket's."""
    _write_mixed_result(tmp_path)
    df = load_all_results(str(tmp_path), include_entity_type_breakdown=True)

    overall = df[df["entity_type"] == "overall"].iloc[0]
    for metric in _METRIC_KEYS:
        assert overall[metric] == pytest.approx(0.17)
    assert overall["n_source_entities"] == 100


def test_breakdown_of_an_empty_directory_returns_the_wide_schema(tmp_path: Path) -> None:
    empty = tmp_path / "results"
    empty.mkdir()
    df = load_all_results(str(empty), include_entity_type_breakdown=True)
    assert df.empty
    assert list(df.columns) == _EXPECTED_RESULT_COLUMNS + ["entity_type", "n_refs"]


def test_breakdown_of_a_missing_directory_returns_the_wide_schema(tmp_path: Path) -> None:
    df = load_all_results(
        str(tmp_path / "does_not_exist"), include_entity_type_breakdown=True
    )
    assert df.empty
    assert list(df.columns) == _EXPECTED_RESULT_COLUMNS + ["entity_type", "n_refs"]


def test_breakdown_rows_do_not_reach_the_default_report_path(tmp_path: Path) -> None:
    """The report loads without the flag, so summaries stay one row per pair."""
    _write_mixed_result(tmp_path)
    summary = build_condition_summary_table(load_all_results(str(tmp_path)))
    assert list(summary["n_pairs"]) == [1]


# ---------------------------------------------------------------------------
# Condition summary
# ---------------------------------------------------------------------------
def test_condition_summary_row_count(full_results_dir: Path) -> None:
    df = load_all_results(str(full_results_dir))
    summary = build_condition_summary_table(df)
    assert list(summary.columns) == _EXPECTED_SUMMARY_COLUMNS
    assert len(summary) == len(EXPERIMENT_CONDITIONS) == 19


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


def test_report_entity_type_placeholder_without_breakdown(
    full_results_dir: Path, tmp_path: Path
) -> None:
    """A frame loaded without the breakdown still renders the section."""
    df = load_all_results(str(full_results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    text = output.read_text(encoding="utf-8")
    assert "# Entity-Type Analysis" in text
    assert "_No per-entity-type data available._" in text


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


def _per_dataset_section(text: str) -> str:
    """Return the body of the report's Per-Dataset Breakdown section."""
    return text.split("# Per-Dataset Breakdown", 1)[1].split("\n# ", 1)[0]


def _sub_table(section: str, dataset_id: str) -> List[str]:
    """Return the table rows rendered under a Per-Dataset Breakdown heading."""
    body = section.split(f"## {dataset_id}\n", 1)[1].split("\n## ", 1)[0]
    return [line for line in body.splitlines() if line.startswith("|")]


def _headings_of(section: str) -> List[str]:
    """Return the level-two headings of a report section."""
    return [line for line in section.splitlines() if line.startswith("## ")]


def _write_production_layout(results_dir: Path) -> None:
    """Write results using the dataset IDs the pipeline actually emits."""
    for dataset_id, f1 in (
        ("D1", 0.10),
        ("D2", 0.20),
        ("D3", 0.30),
        ("D4_schema", 0.40),
        ("D4_instance", 0.50),
        ("D5", 0.60),
    ):
        _write_result(results_dir, "C1", dataset_id, f"{dataset_id}_pair", f1)


def test_report_per_dataset_lists_only_datasets_present(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    for pair_index in range(6):
        _write_result(results_dir, "C1", "D1", f"D1_p{pair_index}", 0.5)
    df = load_all_results(str(results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    section = _per_dataset_section(output.read_text(encoding="utf-8"))
    assert _headings_of(section) == ["## D1"]
    assert "_No results available for" not in section


def test_report_per_dataset_production_dataset_layout(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_production_layout(results_dir)
    df = load_all_results(str(results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    section = _per_dataset_section(output.read_text(encoding="utf-8"))
    assert _headings_of(section) == [
        "## D1",
        "## D2",
        "## D3",
        "## D4_instance",
        "## D4_schema",
        "## D5",
    ]
    assert _sub_table(section, "D1")[2:] == ["| C1 | 0.1000 |"]
    assert _sub_table(section, "D2")[2:] == ["| C1 | 0.2000 |"]
    assert _sub_table(section, "D3")[2:] == ["| C1 | 0.3000 |"]
    assert _sub_table(section, "D4_schema")[2:] == ["| C1 | 0.4000 |"]
    assert _sub_table(section, "D4_instance")[2:] == ["| C1 | 0.5000 |"]
    assert _sub_table(section, "D5")[2:] == ["| C1 | 0.6000 |"]


def test_executive_summary_covers_d4_sub_datasets(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_production_layout(results_dir)
    df = load_all_results(str(results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    text = output.read_text(encoding="utf-8")
    body = text.split("# Executive Summary", 1)[1].split("\n# ", 1)[0]
    rows = [line for line in body.splitlines() if line.startswith("|")]
    datasets = [row.split(" | ")[0].lstrip("| ") for row in rows[2:]]
    assert datasets == ["D1", "D2", "D3", "D4_instance", "D4_schema", "D5"]


def test_report_per_dataset_renders_d4_sub_datasets(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    for pair_index in range(6):
        _write_result(results_dir, "C1", "D4_schema", f"D4_predicate_p{pair_index}", 0.6)
        _write_result(results_dir, "C1", "D4_instance", f"D4_instance_p{pair_index}", 0.7)
    df = load_all_results(str(results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    section = _per_dataset_section(output.read_text(encoding="utf-8"))
    assert "_No results available for D4._" not in section
    assert _sub_table(section, "D4_schema")[2:] == ["| C1 | 0.6000 |"]
    assert _sub_table(section, "D4_instance")[2:] == ["| C1 | 0.7000 |"]


def test_report_per_dataset_headings_are_sorted(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    for dataset_id in ("D5", "D4_schema", "D1", "D4_instance"):
        _write_result(results_dir, "C1", dataset_id, f"{dataset_id}_p0", 0.5)
    df = load_all_results(str(results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    section = _per_dataset_section(output.read_text(encoding="utf-8"))
    headings = [line for line in section.splitlines() if line.startswith("## ")]
    assert headings == ["## D1", "## D4_instance", "## D4_schema", "## D5"]


def test_report_per_dataset_rows_sorted_by_mean_f1(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    for condition_id, f1 in (("C1", 0.30), ("C2", 0.90), ("C3", 0.30)):
        _write_result(results_dir, condition_id, "D4_schema", "schema_pair", f1)
    df = load_all_results(str(results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    section = _per_dataset_section(output.read_text(encoding="utf-8"))
    assert _sub_table(section, "D4_schema")[2:] == [
        "| C2 | 0.9000 |",
        "| C1 | 0.3000 |",
        "| C3 | 0.3000 |",
    ]


def test_report_per_dataset_all_nan_metric_keeps_heading(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_result(results_dir, "C1", "D4_schema", "schema_pair", 0.5)
    _write_result(
        results_dir,
        "C1",
        "D4_instance",
        "instance_pair",
        0.5,
        overrides={"metrics": {key: float("nan") for key in _METRIC_KEYS}},
    )
    df = load_all_results(str(results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    section = _per_dataset_section(output.read_text(encoding="utf-8"))
    assert _headings_of(section) == ["## D4_instance", "## D4_schema"]
    assert "_No results available for D4_instance._" in section
    assert _sub_table(section, "D4_schema")[2:] == ["| C1 | 0.5000 |"]


def test_report_per_dataset_empty_results_note(tmp_path: Path) -> None:
    df = load_all_results(str(tmp_path / "missing"))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    section = _per_dataset_section(output.read_text(encoding="utf-8"))
    assert "_No results available._" in section
    assert _headings_of(section) == []


# ---------------------------------------------------------------------------
# Entity-Type Analysis section
# ---------------------------------------------------------------------------
_ENTITY_TYPE_HEADER = (
    "| Condition | Strategy | Model | Class F1 | Class n_refs "
    "| Predicate F1 | Predicate n_refs | Class vs Predicate gap |"
)


def _write_breakdown_result(
    results_dir: Path,
    condition_id: str,
    pair: tuple,
    buckets: Dict[str, Dict[str, float]],
) -> Path:
    """Write one result whose ``per_entity_type`` block holds ``buckets``."""
    dataset_id, pair_name = pair
    return _write_result(
        results_dir,
        condition_id,
        dataset_id,
        pair_name,
        0.17,
        overrides={"per_entity_type": buckets},
    )


def _entity_type_report(results_dir: Path, output: Path) -> str:
    """Render a report from the expanded frame and return its text."""
    df = load_all_results(str(results_dir), include_entity_type_breakdown=True)
    generate_markdown_report(df, None, str(output))
    return output.read_text(encoding="utf-8")


def _entity_type_section_text(text: str) -> str:
    """Return the body of the report's Entity-Type Analysis section."""
    return text.split("# Entity-Type Analysis", 1)[1].split("\n# ", 1)[0]


def _entity_type_rows(text: str) -> List[str]:
    """Return the table rows rendered in the Entity-Type Analysis section."""
    section = _entity_type_section_text(text)
    return [line for line in section.splitlines() if line.startswith("|")]


def _expected_row(condition_id: str, cells: str) -> str:
    """Compose the expected entity-type row for a registered condition."""
    condition = get_condition(condition_id)
    return f"| {condition_id} | {condition.strategy_name} | {condition.model_key} | {cells} |"


def _cmt_edas_buckets() -> Dict[str, Dict[str, float]]:
    return {
        "class": _entity_type_metrics(0.21, 18),
        "predicate": _entity_type_metrics(0.08, 6),
    }


def test_entity_type_section_between_breakdown_and_ablation(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_breakdown_result(results_dir, "C1", ("D3", "cmt-edas"), _cmt_edas_buckets())
    text = _entity_type_report(results_dir, tmp_path / "report.md")
    assert _headings(text) == [
        "# Executive Summary",
        "# Condition Summary Table",
        "# Per-Dataset Breakdown",
        "# Entity-Type Analysis",
        "# Ablation Group Analysis",
        "# PPAS Ablation",
    ]


def test_entity_type_table_columns_and_values(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_breakdown_result(results_dir, "C1", ("D3", "cmt-edas"), _cmt_edas_buckets())
    rows = _entity_type_rows(_entity_type_report(results_dir, tmp_path / "report.md"))
    assert rows[0] == _ENTITY_TYPE_HEADER
    assert rows[2:] == [_expected_row("C1", "0.2100 | 18 | 0.0800 | 6 | 0.1300")]


def test_entity_type_table_averages_over_pairs_and_datasets(tmp_path: Path) -> None:
    """One row per condition, aggregating every mixed pair it ran on."""
    results_dir = tmp_path / "results"
    _write_breakdown_result(
        results_dir,
        "C1",
        ("D3", "cmt-edas"),
        {"class": _entity_type_metrics(0.20, 10), "predicate": _entity_type_metrics(0.10, 4)},
    )
    _write_breakdown_result(
        results_dir,
        "C1",
        ("D4_schema", "memoryalpha-stexpanded"),
        {"class": _entity_type_metrics(0.40, 30), "predicate": _entity_type_metrics(0.20, 6)},
    )
    rows = _entity_type_rows(_entity_type_report(results_dir, tmp_path / "report.md"))
    assert rows[2:] == [_expected_row("C1", "0.3000 | 40 | 0.1500 | 10 | 0.1500")]


def test_entity_type_table_sorted_by_class_f1_descending(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    for condition_id, class_f1 in (("C1", 0.30), ("C2", 0.90), ("C3", 0.60)):
        _write_breakdown_result(
            results_dir,
            condition_id,
            ("D3", "cmt-edas"),
            {
                "class": _entity_type_metrics(class_f1, 18),
                "predicate": _entity_type_metrics(0.05, 6),
            },
        )
    rows = _entity_type_rows(_entity_type_report(results_dir, tmp_path / "report.md"))
    assert [row.split(" | ")[0].lstrip("| ") for row in rows[2:]] == ["C2", "C3", "C1"]


def test_entity_type_table_omits_conditions_without_breakdown(tmp_path: Path) -> None:
    """A condition that only ran on D1/D2 never reaches the entity-type table."""
    results_dir = tmp_path / "results"
    _write_breakdown_result(results_dir, "C1", ("D3", "cmt-edas"), _cmt_edas_buckets())
    for pair_index in range(3):
        _write_result(results_dir, "C11", "D1", f"D1_p{pair_index}", 0.5)
    text = _entity_type_report(results_dir, tmp_path / "report.md")
    section = _entity_type_section_text(text)
    assert "| C1 |" in section
    assert "| C11 |" not in section
    assert "# Condition Summary Table" in text


def test_entity_type_table_ignores_non_mixed_datasets(tmp_path: Path) -> None:
    """Only D3 and D4_schema contribute; a stray breakdown elsewhere is dropped."""
    results_dir = tmp_path / "results"
    _write_breakdown_result(
        results_dir, "C1", ("D4_instance", "memoryalpha-stexpanded"), _cmt_edas_buckets()
    )
    section = _entity_type_section_text(
        _entity_type_report(results_dir, tmp_path / "report.md")
    )
    assert "_No per-entity-type data available._" in section


def test_entity_type_table_drops_buckets_below_the_ref_threshold(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_breakdown_result(
        results_dir,
        "C1",
        ("D3", "cmt-edas"),
        {
            "class": _entity_type_metrics(0.21, 18),
            "predicate": _entity_type_metrics(0.08, 2),
        },
    )
    rows = _entity_type_rows(_entity_type_report(results_dir, tmp_path / "report.md"))
    assert rows[2:] == [_expected_row("C1", "0.2100 | 18 | N/A | N/A | N/A")]


def test_entity_type_finding_cites_computed_values(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_breakdown_result(results_dir, "C1", ("D3", "cmt-edas"), _cmt_edas_buckets())
    _write_breakdown_result(
        results_dir,
        "C1",
        ("D3", "cmt-conference"),
        {"class": _entity_type_metrics(0.85, 20), "predicate": _entity_type_metrics(0.0, 5)},
    )
    section = _entity_type_section_text(
        _entity_type_report(results_dir, tmp_path / "report.md")
    )
    assert "Predicates are the harder entity type" in section
    assert "mean class F1 is 0.5300" in section
    assert "mean predicate F1 of 0.0400" in section
    assert "gap is 0.4900 F1" in section
    assert "C1 on cmt-conference" in section
    assert "by 0.8500" in section


def test_entity_type_finding_names_classes_when_they_score_lower(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_breakdown_result(
        results_dir,
        "C1",
        ("D3", "cmt-edas"),
        {"class": _entity_type_metrics(0.10, 18), "predicate": _entity_type_metrics(0.60, 6)},
    )
    section = _entity_type_section_text(
        _entity_type_report(results_dir, tmp_path / "report.md")
    )
    assert "Classes are the harder entity type" in section
    assert "gap is -0.5000 F1" in section


def test_entity_type_finding_excludes_d4_schema_from_the_averages(tmp_path: Path) -> None:
    """One D4_schema pair must not outvote every Conference pair in the prose."""
    results_dir = tmp_path / "results"
    _write_breakdown_result(
        results_dir,
        "C1",
        ("D3", "cmt-edas"),
        {"class": _entity_type_metrics(0.30, 18), "predicate": _entity_type_metrics(0.10, 6)},
    )
    _write_breakdown_result(
        results_dir,
        "C5",
        ("D4_schema", "memoryalpha-stexpanded"),
        {"class": _entity_type_metrics(0.05, 12), "predicate": _entity_type_metrics(0.55, 32)},
    )
    text = _entity_type_report(results_dir, tmp_path / "report.md")
    section = _entity_type_section_text(text)
    assert _expected_row("C5", "0.0500 | 12 | 0.5500 | 32 | -0.5000") in section
    assert "Predicates are the harder entity type" in section
    assert "across the 1 condition reporting" in section
    assert "mean class F1 is 0.3000" in section
    assert "gap is 0.2000 F1 over 1 D3 pair." in section


def test_entity_type_finding_divergence_ignores_d4_schema(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_breakdown_result(
        results_dir,
        "C1",
        ("D3", "cmt-edas"),
        {"class": _entity_type_metrics(0.30, 18), "predicate": _entity_type_metrics(0.10, 6)},
    )
    _write_breakdown_result(
        results_dir,
        "C1",
        ("D4_schema", "memoryalpha-stexpanded"),
        {"class": _entity_type_metrics(0.90, 12), "predicate": _entity_type_metrics(0.05, 32)},
    )
    section = _entity_type_section_text(
        _entity_type_report(results_dir, tmp_path / "report.md")
    )
    assert "C1 on cmt-edas" in section
    assert "by 0.2000" in section
    assert "memoryalpha-stexpanded" not in section


def test_entity_type_finding_reports_absent_d3_data(tmp_path: Path) -> None:
    """A D4_schema-only run keeps its table row but states no finding."""
    results_dir = tmp_path / "results"
    _write_breakdown_result(
        results_dir, "C5", ("D4_schema", "memoryalpha-stexpanded"), _cmt_edas_buckets()
    )
    section = _entity_type_section_text(
        _entity_type_report(results_dir, tmp_path / "report.md")
    )
    assert _expected_row("C5", "0.2100 | 18 | 0.0800 | 6 | 0.1300") in section
    assert "No D3 pair reports a per-entity-type breakdown" in section
    assert "harder entity type" not in section


_FINDING_NUMBERS = re.compile(
    r"mean class F1 is (-?\d+\.\d{4}) against mean predicate F1 of (-?\d+\.\d{4})\. "
    r"The mean class-versus-predicate gap is (-?\d+\.\d{4}) F1"
)


def test_entity_type_finding_gap_matches_the_quoted_means(tmp_path: Path) -> None:
    """A condition missing one bucket must not skew one mean and not the other."""
    results_dir = tmp_path / "results"
    _write_breakdown_result(
        results_dir,
        "C1",
        ("D3", "cmt-edas"),
        {"class": _entity_type_metrics(0.90, 20), "predicate": _entity_type_metrics(0.10, 5)},
    )
    _write_breakdown_result(
        results_dir,
        "C2",
        ("D3", "cmt-conference"),
        {"class": _entity_type_metrics(0.10, 20)},
    )
    section = _entity_type_section_text(
        _entity_type_report(results_dir, tmp_path / "report.md")
    )
    assert _expected_row("C2", "0.1000 | 20 | N/A | N/A | N/A") in section
    mean_class, mean_predicate, gap = (float(v) for v in _FINDING_NUMBERS.search(section).groups())
    assert mean_class - mean_predicate == pytest.approx(gap)
    assert (mean_class, mean_predicate, gap) == (0.90, 0.10, pytest.approx(0.80))
    assert "across the 1 condition reporting both entity types" in section


def test_entity_type_finding_phrases_a_predicate_lead_positively(tmp_path: Path) -> None:
    """A predicate-led pair must not be reported as a negative class excess."""
    results_dir = tmp_path / "results"
    _write_breakdown_result(
        results_dir,
        "C1",
        ("D3", "cmt-edas"),
        {"class": _entity_type_metrics(0.10, 18), "predicate": _entity_type_metrics(0.60, 6)},
    )
    section = _entity_type_section_text(
        _entity_type_report(results_dir, tmp_path / "report.md")
    )
    assert "predicate F1 exceeds class F1 by 0.5000" in section
    assert "by -" not in section


def test_entity_type_finding_picks_the_widest_gap_by_magnitude(tmp_path: Path) -> None:
    """The cited pair is the strongest example whichever entity type leads it."""
    results_dir = tmp_path / "results"
    _write_breakdown_result(
        results_dir,
        "C1",
        ("D3", "cmt-edas"),
        {"class": _entity_type_metrics(0.90, 18), "predicate": _entity_type_metrics(0.10, 6)},
    )
    _write_breakdown_result(
        results_dir,
        "C1",
        ("D3", "cmt-conference"),
        {"class": _entity_type_metrics(0.05, 18), "predicate": _entity_type_metrics(0.95, 6)},
    )
    section = _entity_type_section_text(
        _entity_type_report(results_dir, tmp_path / "report.md")
    )
    assert "C1 on cmt-conference, where predicate F1 exceeds class F1 by 0.9000" in section


def test_entity_type_finding_claims_no_harder_type_on_a_zero_gap(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_breakdown_result(
        results_dir,
        "C1",
        ("D3", "cmt-edas"),
        {"class": _entity_type_metrics(0.0, 18), "predicate": _entity_type_metrics(0.0, 6)},
    )
    section = _entity_type_section_text(
        _entity_type_report(results_dir, tmp_path / "report.md")
    )
    assert "Neither entity type is harder" in section
    assert "No pair diverges" in section
    assert "harder entity type:" not in section


def test_entity_type_finding_without_any_complete_condition(tmp_path: Path) -> None:
    """Every D3 bucket is class-only, so no gap may be claimed."""
    results_dir = tmp_path / "results"
    _write_breakdown_result(
        results_dir, "C1", ("D3", "cmt-edas"), {"class": _entity_type_metrics(0.5, 18)}
    )
    section = _entity_type_section_text(
        _entity_type_report(results_dir, tmp_path / "report.md")
    )
    assert _expected_row("C1", "0.5000 | 18 | N/A | N/A | N/A") in section
    assert "No D3 condition reports both entity types" in section
    assert "harder entity type" not in section


def test_generate_markdown_report_never_loads_results_itself(
    full_results_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report renders the frame it is given; the caller loads exactly once."""
    import kgsemembed.evaluation.aggregator as aggregator

    calls: List[tuple] = []

    def counting_load(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append((args, kwargs))
        raise AssertionError("generate_markdown_report must not load results.")

    df = load_all_results(str(full_results_dir), include_entity_type_breakdown=True)
    monkeypatch.setattr(aggregator, "load_all_results", counting_load)
    generate_markdown_report(df, None, str(tmp_path / "report.md"))
    assert calls == []


def test_entity_type_section_placeholder_when_no_mixed_results(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    for pair_index in range(3):
        _write_result(results_dir, "C1", "D1", f"D1_p{pair_index}", 0.5)
    section = _entity_type_section_text(
        _entity_type_report(results_dir, tmp_path / "report.md")
    )
    assert "_No per-entity-type data available._" in section


def test_entity_type_section_tolerates_legacy_result_files(tmp_path: Path) -> None:
    """Files predating per_entity_type must not break the expanded report."""
    results_dir = tmp_path / "results"
    _write_result(results_dir, "C1", "D2", "D2_p0", 0.5)
    _write_result(results_dir, "C1", "D1", "D1_p0", 0.4, overrides={"per_entity_type": {}})
    _write_breakdown_result(results_dir, "C2", ("D3", "cmt-edas"), _cmt_edas_buckets())
    rows = _entity_type_rows(_entity_type_report(results_dir, tmp_path / "report.md"))
    assert rows[2:] == [_expected_row("C2", "0.2100 | 18 | 0.0800 | 6 | 0.1300")]


def _sections_except_entity_type(text: str) -> List[str]:
    """Split a report into sections, dropping the entity-type one."""
    blocks = text.split("\n# ")
    return [block for block in blocks if not block.startswith("Entity-Type Analysis")]


def test_expanded_frame_leaves_every_other_section_byte_identical(tmp_path: Path) -> None:
    """Loading with the breakdown must change one section and nothing else."""
    results_dir = tmp_path / "results"
    _write_production_layout(results_dir)
    _write_breakdown_result(results_dir, "C1", ("D3", "cmt-edas"), _cmt_edas_buckets())
    _write_breakdown_result(
        results_dir, "C1", ("D4_schema", "memoryalpha-stexpanded"), _cmt_edas_buckets()
    )
    narrow = tmp_path / "narrow.md"
    wide = tmp_path / "wide.md"
    generate_markdown_report(load_all_results(str(results_dir)), None, str(narrow))
    generate_markdown_report(
        load_all_results(str(results_dir), include_entity_type_breakdown=True),
        None,
        str(wide),
    )
    narrow_text = narrow.read_text(encoding="utf-8")
    wide_text = wide.read_text(encoding="utf-8")
    assert _sections_except_entity_type(narrow_text) == _sections_except_entity_type(wide_text)
    assert "_No per-entity-type data available._" in narrow_text
    assert "| Condition | Strategy | Model |" in wide_text


def test_breakdown_rows_do_not_inflate_the_other_sections(tmp_path: Path) -> None:
    """Entity-type rows are consumed by one section, not counted as pairs."""
    results_dir = tmp_path / "results"
    _write_breakdown_result(results_dir, "C1", ("D3", "cmt-edas"), _cmt_edas_buckets())
    text = _entity_type_report(results_dir, tmp_path / "report.md")
    per_dataset = _per_dataset_section(text)
    assert _sub_table(per_dataset, "D3")[2:] == ["| C1 | 0.1700 |"]
    summary = text.split("# Condition Summary Table", 1)[1].split("\n# ", 1)[0]
    assert summary.strip().splitlines()[-1].endswith("| 1 |")


def test_entity_type_report_is_deterministic(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    for condition_id in ("C2", "C1"):
        _write_breakdown_result(
            results_dir, condition_id, ("D3", "cmt-edas"), _cmt_edas_buckets()
        )
    first = _entity_type_report(results_dir, tmp_path / "first.md")
    second = _entity_type_report(results_dir, tmp_path / "second.md")
    assert first == second


def test_report_ppas_values_from_results(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    for pair_index in range(6):
        _write_result(results_dir, "C5", "D5", f"D5_p{pair_index}", 0.40)
        _write_result(results_dir, "C14", "D5", f"D5_p{pair_index}", 0.55)
    df = load_all_results(str(results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    text = output.read_text(encoding="utf-8")
    confounded = "Confounded (different models)"
    controlled = "Controlled ablation (same model)"
    varied = "Yes — V4+V6 invokes PPAS via should_apply_ppas()"
    unvaried = "No — V2+V8 does not invoke PPAS"
    unvaried_confounded = f"{unvaried} (confounded by model change)"
    assert (
        f"| C5 vs C14 | D5 | {confounded} | {varied} | 0.4000 | 0.5500 | 0.1500 |"
    ) in text
    assert (
        f"| C10 vs C15 | D1 | {confounded} | {unvaried_confounded} | N/A | N/A | N/A |"
    ) in text
    assert (
        f"| C10 vs C19 | D1 | {controlled} | {unvaried} | N/A | N/A | N/A |"
    ) in text


def test_report_ppas_controlled_ablation_row(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    for pair_index in range(6):
        _write_result(results_dir, "C10", "D1", f"D1_p{pair_index}", 0.09)
        _write_result(results_dir, "C19", "D1", f"D1_p{pair_index}", 0.15)
    df = load_all_results(str(results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    text = output.read_text(encoding="utf-8")
    assert (
        "| C10 vs C19 | D1 | Controlled ablation (same model) | "
        "No — V2+V8 does not invoke PPAS | 0.0900 | 0.1500 | 0.0600 |" in text
    )


def test_report_ppas_marks_only_the_v4_comparison_as_varying_ppas(
    tmp_path: Path,
) -> None:
    """Only the V4+V6 pair consults a token budget, so only it varies PPAS."""
    output = tmp_path / "report.md"
    generate_markdown_report(load_all_results(str(tmp_path)), None, str(output))
    section = output.read_text(encoding="utf-8").split("# PPAS Ablation", 1)[1]
    rows = [line for line in section.splitlines() if " vs " in line]
    assert [row.count("| Yes —") for row in rows] == [1, 0, 0]
    assert [row.count("| No —") for row in rows] == [0, 1, 1]


# ---------------------------------------------------------------------------
# Statistical section
# ---------------------------------------------------------------------------
def _stats_table_lines(text: str) -> List[str]:
    """Return the table rows of the report's Statistical Significance section."""
    body = text.split("# Statistical Significance", 1)[1].split("\n# ", 1)[0]
    return [line for line in body.splitlines() if line.startswith("|")]


def _cell_count(row: str) -> int:
    """Count table cells, ignoring pipes escaped for Markdown."""
    return len(re.split(r"(?<!\\)\|", row)) - 2


def _report_with_stats(tmp_path: Path, stats: List[dict]) -> List[str]:
    """Render a report over empty results and return its statistical table rows."""
    output = tmp_path / "report.md"
    empty = pd.DataFrame(columns=list(_RESULT_COLUMNS))
    generate_markdown_report(empty, stats, str(output))
    return _stats_table_lines(output.read_text(encoding="utf-8"))


def test_stats_columns_include_effect_size_and_warning() -> None:
    assert "effect_size_r" in _STATS_COLUMNS
    assert _STATS_COLUMNS[-1] == "Warning"


def test_stats_row_populates_every_declared_column() -> None:
    # Guards against a column being declared but never filled, which pandas
    # would silently render as an empty N/A column rather than failing.
    assert list(_stats_row(_synthetic_stats()[0])) == list(_STATS_COLUMNS)


def test_stats_columns_match_export_stats_table(tmp_path: Path) -> None:
    exported = tmp_path / "stats.md"
    export_stats_table(_synthetic_stats(), str(exported))
    export_header = exported.read_text(encoding="utf-8").splitlines()[0]
    report_header = _report_with_stats(tmp_path, _synthetic_stats())[0]
    assert report_header == export_header


def test_report_stats_row_carries_effect_size(tmp_path: Path) -> None:
    rows = _report_with_stats(tmp_path, _synthetic_stats(effect_size_r=0.7333))
    assert rows[-1] == "| C1 | C2 | 5 | 0.0321 | True | 0.0187 | 0.7333 |  |"


def test_report_stats_warning_is_displayed(tmp_path: Path) -> None:
    rows = _report_with_stats(
        tmp_path, _synthetic_stats(wilcoxon_warning="zero differences detected")
    )
    assert rows[-1].endswith("| zero differences detected |")


def test_report_stats_warning_is_empty_when_absent(tmp_path: Path) -> None:
    rows = _report_with_stats(tmp_path, _synthetic_stats(wilcoxon_warning=None))
    assert rows[-1].endswith("|  |")


def test_report_stats_warning_stays_in_one_cell(tmp_path: Path) -> None:
    rows = _report_with_stats(
        tmp_path, _synthetic_stats(wilcoxon_warning="ties |\nand zeros")
    )
    assert rows[-1].endswith("| ties \\| and zeros |")
    assert _cell_count(rows[-1]) == _cell_count(rows[0])


def test_report_stats_warning_object_does_not_leak(tmp_path: Path) -> None:
    rows = _report_with_stats(
        tmp_path, _synthetic_stats(wilcoxon_warning=RuntimeWarning("zero differences"))
    )
    assert rows[-1].endswith("| zero differences |")
    assert "RuntimeWarning" not in rows[-1]


def test_report_stats_absent_warning_is_not_rendered_as_none(tmp_path: Path) -> None:
    rows = _report_with_stats(tmp_path, _synthetic_stats(wilcoxon_warning=None))
    assert "None" not in rows[-1]
    assert "N/A" not in rows[-1]


def test_report_stats_long_warning_collapses_to_one_line(tmp_path: Path) -> None:
    message = ("exact distribution unavailable; " * 12).strip()
    rows = _report_with_stats(tmp_path, _synthetic_stats(wilcoxon_warning=message))
    assert len(rows) == 3
    assert rows[-1].endswith(f"| {message} |")


@pytest.mark.parametrize("effect_size", [0.0, 0.5, 1.0, -0.25])
def test_report_stats_renders_effect_size_values(tmp_path: Path, effect_size: float) -> None:
    rows = _report_with_stats(tmp_path, _synthetic_stats(effect_size_r=effect_size))
    assert rows[-1].split(" | ")[6] == f"{effect_size:.4f}"


def test_report_stats_tolerates_missing_effect_size(tmp_path: Path) -> None:
    stats = _synthetic_stats()
    del stats[0]["effect_size_r"]
    rows = _report_with_stats(tmp_path, stats)
    assert rows[-1] == "| C1 | C2 | 5 | 0.0321 | True | 0.0187 | N/A |  |"


def test_report_stats_preserves_existing_columns(tmp_path: Path) -> None:
    header = _report_with_stats(tmp_path, _synthetic_stats())[0]
    for column in ("Condition A", "Condition B", "n", "p-value", "Corrected sig.", "delta-F1"):
        assert column in header


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
