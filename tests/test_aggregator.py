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
from kgsemembed.utils.errors import DataError

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
    "hits_at_1",
    "ablation_group",
]
_EXPECTED_SUMMARY_COLUMNS = [
    "condition_id",
    "strategy",
    "model_key",
    "ablation_group",
    "mean_mrr",
    "std_mrr",
    "mean_hits_at_1",
    "mean_f1",
    "std_f1",
    "mean_recall_at_1",
    "mean_recall_at_5",
    "mean_recall_at_10",
    "n_columns",
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
    return {key: f1 for key in (*_METRIC_KEYS, "hits_at_1")}


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
            "n_units": 6,
            "n_pairs": 26,
            "pair_wins": 20,
            "pair_ties": 2,
            "pair_losses": 4,
            "p_value": 0.0321,
            "min_attainable_p": 0.03125,
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
def _entity_type_metrics(f1: float, n_refs: int, counts: tuple = (1, 1, 1)) -> Dict[str, float]:
    return {**_metrics(f1), "n_refs": n_refs, **dict(zip(("tp", "fp", "fn"), counts))}


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
    assert list(df.columns) == _EXPECTED_RESULT_COLUMNS + ["entity_type", "n_refs", "tp", "fp", "fn"]


def test_breakdown_adds_one_row_per_entity_type(tmp_path: Path) -> None:
    _write_mixed_result(tmp_path)
    df = load_all_results(str(tmp_path), include_entity_type_breakdown=True)

    assert list(df["entity_type"]) == ["overall", "class", "predicate"]
    assert list(df["f1"]) == pytest.approx([0.17, 0.21, 0.08])
    assert list(df["n_refs"][1:]) == [18, 6]
    assert list(df["tp"][1:]) == [1, 1] and pd.isna(df.loc[0, "tp"])


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
    assert list(df.columns) == _EXPECTED_RESULT_COLUMNS + ["entity_type", "n_refs", "tp", "fp", "fn"]


def test_breakdown_of_a_missing_directory_returns_the_wide_schema(tmp_path: Path) -> None:
    df = load_all_results(
        str(tmp_path / "does_not_exist"), include_entity_type_breakdown=True
    )
    assert df.empty
    assert list(df.columns) == _EXPECTED_RESULT_COLUMNS + ["entity_type", "n_refs", "tp", "fp", "fn"]


def test_breakdown_rows_do_not_reach_the_default_report_path(tmp_path: Path) -> None:
    """The report loads without the flag, so summaries stay one row per pair."""
    _write_mixed_result(tmp_path)
    summary = build_condition_summary_table(load_all_results(str(tmp_path)))
    assert list(summary["n_columns"]) == [1]


# ---------------------------------------------------------------------------
# Condition summary
# ---------------------------------------------------------------------------
def test_condition_summary_row_count(full_results_dir: Path) -> None:
    df = load_all_results(str(full_results_dir))
    summary = build_condition_summary_table(df)
    assert list(summary.columns) == _EXPECTED_SUMMARY_COLUMNS
    assert len(summary) == len(EXPERIMENT_CONDITIONS) == 19


def test_condition_summary_lists_conditions_in_registry_order(full_results_dir: Path) -> None:
    df = load_all_results(str(full_results_dir))
    summary = build_condition_summary_table(df)
    assert list(summary["condition_id"]) == [c.condition_id for c in EXPERIMENT_CONDITIONS]


def test_condition_summary_is_not_ranked_by_score(tmp_path: Path) -> None:
    for condition_id, f1 in (("C3", 0.9), ("C1", 0.1), ("C2", 0.5)):
        _write_result(tmp_path, condition_id, "D1", "D1_p0", f1)
    summary = build_condition_summary_table(load_all_results(str(tmp_path)))
    assert list(summary["condition_id"]) == ["C1", "C2", "C3"]


def test_condition_summary_weights_each_dataset_column_once(tmp_path: Path) -> None:
    """D3's pairs are averaged into one column, so they cannot outweigh D1."""
    for pair_index in range(21):
        _write_result(tmp_path, "C1", "D3", f"D3_p{pair_index}", 0.2)
    _write_result(tmp_path, "C1", "D1", "D1_p0", 0.8)
    summary = build_condition_summary_table(load_all_results(str(tmp_path)))
    assert summary.loc[0, "n_columns"] == 2
    assert summary.loc[0, "mean_f1"] == pytest.approx(0.5)
    assert summary.loc[0, "mean_mrr"] == pytest.approx(0.5)
    assert summary.loc[0, "std_f1"] == pytest.approx(0.3)


def test_condition_summary_keeps_d4_columns_separate(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "D4_schema", "schema_pair", 0.9)
    _write_result(tmp_path, "C1", "D4_instance", "instance_pair", 0.5)
    summary = build_condition_summary_table(load_all_results(str(tmp_path)))
    assert summary.loc[0, "n_columns"] == 2
    assert summary.loc[0, "mean_f1"] == pytest.approx(0.7)


def test_condition_summary_rounding_and_n_columns(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "D1", "D1_p0", 0.123456)
    _write_result(tmp_path, "C1", "D1", "D1_p1", 0.234567)
    _write_result(tmp_path, "C1", "D2", "D2_p0", 0.345678)
    summary = build_condition_summary_table(load_all_results(str(tmp_path)))
    assert summary.loc[0, "n_columns"] == 2
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
        "# Cross-Dataset Consistency (Wilcoxon)",
        "# PPAS Ablation",
        "# Sanity Checks",
    ]


def _group_block(report: str, group: str) -> str:
    return report.split(f"## Group {group}\n", 1)[1].split("\n#", 1)[0]


def test_group_e_block_lists_c10_as_reference(full_results_dir: Path, tmp_path: Path) -> None:
    df = load_all_results(str(full_results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    block = _group_block(output.read_text(encoding="utf-8"), "E")
    assert "| C20 |" in block and "| C10 |" in block
    assert "_C10 (group B) shown for reference" in block


def test_groups_without_reference_rows_carry_no_note(
    full_results_dir: Path, tmp_path: Path
) -> None:
    df = load_all_results(str(full_results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    report = output.read_text(encoding="utf-8")
    for group in ("A", "B", "C", "D"):
        assert "shown for reference" not in _group_block(report, group)


def test_group_b_block_is_registry_membership_only(
    full_results_dir: Path, tmp_path: Path
) -> None:
    df = load_all_results(str(full_results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    block = _group_block(output.read_text(encoding="utf-8"), "B")
    assert "| C20 |" not in block and "| C1 |" not in block


def test_report_six_sections_without_stats(full_results_dir: Path, tmp_path: Path) -> None:
    df = load_all_results(str(full_results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    headings = _headings(output.read_text(encoding="utf-8"))
    assert len(headings) == 7
    assert "# Cross-Dataset Consistency (Wilcoxon)" not in headings


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
    assert "_No D3 per-entity-type data available._" in text


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
    assert "# Cross-Dataset Consistency (Wilcoxon)" in text
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
    assert _sub_table(section, "D1")[2:] == ["| C1 | 0.1000 | 0.1000 | 0.1000 |"]
    assert _sub_table(section, "D2")[2:] == ["| C1 | 0.2000 | 0.2000 | 0.2000 |"]
    assert _sub_table(section, "D3")[2:] == ["| C1 | 0.3000 | 0.3000 | 0.3000 |"]
    assert _sub_table(section, "D4_schema")[2:] == ["| C1 | 0.4000 | 0.4000 | 0.4000 |"]
    assert _sub_table(section, "D4_instance")[2:] == ["| C1 | 0.5000 | 0.5000 | 0.5000 |"]
    assert _sub_table(section, "D5")[2:] == ["| C1 | 0.6000 | 0.6000 | 0.6000 |"]


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
    assert _sub_table(section, "D4_schema")[2:] == ["| C1 | 0.6000 | 0.6000 | 0.6000 |"]
    assert _sub_table(section, "D4_instance")[2:] == ["| C1 | 0.7000 | 0.7000 | 0.7000 |"]


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


def test_report_per_dataset_rows_sorted_by_mrr(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    for condition_id, f1 in (("C1", 0.30), ("C2", 0.90), ("C3", 0.30)):
        _write_result(results_dir, condition_id, "D4_schema", "schema_pair", f1)
    df = load_all_results(str(results_dir))
    output = tmp_path / "report.md"
    generate_markdown_report(df, None, str(output))
    section = _per_dataset_section(output.read_text(encoding="utf-8"))
    assert _sub_table(section, "D4_schema")[2:] == [
        "| C2 | 0.9000 | 0.9000 | 0.9000 |",
        "| C1 | 0.3000 | 0.3000 | 0.3000 |",
        "| C3 | 0.3000 | 0.3000 | 0.3000 |",
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
    assert _sub_table(section, "D4_schema")[2:] == ["| C1 | 0.5000 | 0.5000 | 0.5000 |"]


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
    "| Condition | Strategy | Model | Class pairs | Class refs | Class MRR | Class Hits@1 "
    "| Class F1 | Class F1 (micro) | Predicate pairs | Predicate refs | Predicate MRR "
    "| Predicate Hits@1 | Predicate F1 | Predicate F1 (micro) | MRR gap | F1 gap |"
)
_MISSING_TYPE = " | ".join(["N/A"] * 6)


def _type_cells(n_pairs: int, n_refs: int, value: float, micro: float = 0.5) -> str:
    """Cells of one entity type whose MRR, Hits@1 and F1 all equal ``value``."""
    return f"{n_pairs} | {n_refs} | {value:.4f} | {value:.4f} | {value:.4f} | {micro:.4f}"


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
        "# Sanity Checks",
    ]


def test_entity_type_table_columns_and_values(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_breakdown_result(results_dir, "C1", ("D3", "cmt-edas"), _cmt_edas_buckets())
    rows = _entity_type_rows(_entity_type_report(results_dir, tmp_path / "report.md"))
    assert rows[0] == _ENTITY_TYPE_HEADER
    cells = f"{_type_cells(1, 18, 0.21)} | {_type_cells(1, 6, 0.08)} | 0.1300 | 0.1300"
    assert rows[2:] == [_expected_row("C1", cells)]


def test_entity_type_table_is_d3_only(tmp_path: Path) -> None:
    """D4_schema neither joins a D3 row nor gets a row of its own."""
    results_dir = tmp_path / "results"
    _write_breakdown_result(results_dir, "C1", ("D3", "cmt-edas"), _cmt_edas_buckets())
    _write_breakdown_result(
        results_dir, "C1", ("D4_schema", "memoryalpha-stexpanded"),
        {"class": _entity_type_metrics(0.90, 12), "predicate": _entity_type_metrics(0.90, 32)},
    )
    _write_breakdown_result(
        results_dir, "C5", ("D4_schema", "memoryalpha-stexpanded"), _cmt_edas_buckets()
    )
    text = _entity_type_report(results_dir, tmp_path / "report.md")
    section = _entity_type_section_text(text)
    assert [row.split(" | ")[0] for row in _entity_type_rows(text)[2:]] == ["| C1"]
    assert _type_cells(1, 18, 0.21) in section
    assert section.startswith("\n\n_D3 only: D4_schema")


def test_entity_type_table_lists_conditions_in_registry_order(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    for condition_id, class_f1 in (("C20", 0.30), ("C2", 0.90), ("C3", 0.60)):
        _write_breakdown_result(
            results_dir, condition_id, ("D3", "cmt-edas"),
            {"class": _entity_type_metrics(class_f1, 18), "predicate": _entity_type_metrics(0.05, 6)},
        )
    rows = _entity_type_rows(_entity_type_report(results_dir, tmp_path / "report.md"))
    assert [row.split(" | ")[0].lstrip("| ") for row in rows[2:]] == ["C2", "C3", "C20"]


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


def test_entity_type_table_drops_buckets_below_the_ref_threshold(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_breakdown_result(
        results_dir, "C1", ("D3", "cmt-edas"),
        {"class": _entity_type_metrics(0.21, 18), "predicate": _entity_type_metrics(0.08, 2)},
    )
    rows = _entity_type_rows(_entity_type_report(results_dir, tmp_path / "report.md"))
    expected = _expected_row("C1", f"{_type_cells(1, 18, 0.21)} | {_MISSING_TYPE} | N/A | N/A")
    assert rows[2:] == [expected]


def test_entity_type_finding_cites_mrr_f1_and_counts(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_breakdown_result(results_dir, "C1", ("D3", "cmt-edas"), _cmt_edas_buckets())
    _write_breakdown_result(
        results_dir, "C1", ("D3", "cmt-conference"),
        {"class": _entity_type_metrics(0.85, 20), "predicate": _entity_type_metrics(0.0, 5)},
    )
    section = _entity_type_section_text(_entity_type_report(results_dir, tmp_path / "report.md"))
    assert (
        "Across the 1 condition reporting both entity types on D3, mean class MRR is 0.5300 "
        "against 0.0400 for predicates (gap 0.4900); mean class F1 is 0.5300 against 0.0400 "
        "(gap 0.4900). Class values rest on 2 pairs and 38 test references, predicate values "
        "on 2 pairs and 11 test references."
    ) in section


def test_entity_type_finding_averages_only_complete_conditions(tmp_path: Path) -> None:
    """A condition missing one bucket must not skew one mean and not the other."""
    results_dir = tmp_path / "results"
    _write_breakdown_result(
        results_dir, "C1", ("D3", "cmt-edas"),
        {"class": _entity_type_metrics(0.90, 20), "predicate": _entity_type_metrics(0.10, 5)},
    )
    _write_breakdown_result(
        results_dir, "C2", ("D3", "cmt-conference"), {"class": _entity_type_metrics(0.10, 20)}
    )
    section = _entity_type_section_text(_entity_type_report(results_dir, tmp_path / "report.md"))
    expected = _expected_row("C2", f"{_type_cells(1, 20, 0.10)} | {_MISSING_TYPE} | N/A | N/A")
    assert expected in section
    assert "Across the 1 condition reporting both entity types" in section
    assert "mean class MRR is 0.9000 against 0.1000 for predicates (gap 0.8000)" in section


def test_entity_type_finding_without_any_complete_condition(tmp_path: Path) -> None:
    """Every D3 bucket is class-only, so no gap may be claimed."""
    results_dir = tmp_path / "results"
    _write_breakdown_result(
        results_dir, "C1", ("D3", "cmt-edas"), {"class": _entity_type_metrics(0.5, 18)}
    )
    section = _entity_type_section_text(_entity_type_report(results_dir, tmp_path / "report.md"))
    assert "No D3 condition reports both entity types" in section
    assert "mean class MRR" not in section


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
    assert "_No D3 per-entity-type data available._" in section


def test_entity_type_section_tolerates_legacy_result_files(tmp_path: Path) -> None:
    """Files predating per_entity_type must not break the expanded report."""
    results_dir = tmp_path / "results"
    _write_result(results_dir, "C1", "D2", "D2_p0", 0.5)
    _write_result(results_dir, "C1", "D1", "D1_p0", 0.4, overrides={"per_entity_type": {}})
    _write_breakdown_result(results_dir, "C2", ("D3", "cmt-edas"), _cmt_edas_buckets())
    rows = _entity_type_rows(_entity_type_report(results_dir, tmp_path / "report.md"))
    cells = f"{_type_cells(1, 18, 0.21)} | {_type_cells(1, 6, 0.08)} | 0.1300 | 0.1300"
    assert rows[2:] == [_expected_row("C2", cells)]


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
    assert "_No D3 per-entity-type data available._" in narrow_text
    assert "| Condition | Strategy | Model |" in wide_text


def test_breakdown_rows_do_not_inflate_the_other_sections(tmp_path: Path) -> None:
    """Entity-type rows are consumed by one section, not counted as pairs."""
    results_dir = tmp_path / "results"
    _write_breakdown_result(results_dir, "C1", ("D3", "cmt-edas"), _cmt_edas_buckets())
    text = _entity_type_report(results_dir, tmp_path / "report.md")
    per_dataset = _per_dataset_section(text)
    assert _sub_table(per_dataset, "D3")[2:] == ["| C1 | 0.1700 | 0.1700 | 0.1700 |"]
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
    assert (
        "| C5 vs C14 | D5 | Confounded (strategy V4 vs V4+V6, model M2 vs M3) | "
        "Not isolated — C5 can sample (V4, M2 budget); C14 cannot "
        "(ppas_effective false: M3 has no budget) | 0.4000 | 0.5500 | 0.1500 |"
    ) in text
    assert (
        "| C10 vs C15 | D1 | Confounded (model M2 vs M3) | "
        "No — V2+V8 does not invoke PPAS under either model | N/A | N/A | N/A |"
    ) in text


def test_report_ppas_section_no_longer_lists_c19(tmp_path: Path) -> None:
    output = tmp_path / "report.md"
    generate_markdown_report(load_all_results(str(tmp_path)), None, str(output))
    section = output.read_text(encoding="utf-8").split("# PPAS Ablation", 1)[1]
    section = section.split("\n# ", 1)[0]
    rows = [line for line in section.splitlines() if " vs " in line]
    assert [row.split(" | ")[0] for row in rows] == ["| C5 vs C14", "| C10 vs C15"]
    assert "Controlled" not in section and "| Yes —" not in section


def _sanity_report(tmp_path: Path, c19_f1: float) -> str:
    results_dir = tmp_path / "results"
    for pair_index in range(2):
        _write_result(results_dir, "C10", "D1", f"D1_p{pair_index}", 0.09)
        _write_result(results_dir, "C19", "D1", f"D1_p{pair_index}", c19_f1)
    output = tmp_path / "report.md"
    generate_markdown_report(load_all_results(str(results_dir)), None, str(output))
    return output.read_text(encoding="utf-8")


def test_sanity_section_confirms_c19_reproduces_c10(tmp_path: Path) -> None:
    text = _sanity_report(tmp_path, 0.09)
    assert (
        "| C19 vs C10 | D1 | Identical metrics: V2+V8 never consults a token budget | "
        "0.0900 | 0.0900 | 0.0900 | 0.0900 | Yes |"
    ) in text


def test_sanity_section_flags_a_c19_divergence(tmp_path: Path) -> None:
    text = _sanity_report(tmp_path, 0.15)
    assert text.rstrip().endswith("| 0.1500 | 0.0900 | 0.1500 | 0.0900 | No |")


def test_sanity_section_without_results(tmp_path: Path) -> None:
    output = tmp_path / "report.md"
    generate_markdown_report(load_all_results(str(tmp_path)), None, str(output))
    assert output.read_text(encoding="utf-8").rstrip().endswith("| N/A |")


def test_c19_stays_out_of_the_result_tables(full_results_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "report.md"
    generate_markdown_report(load_all_results(str(full_results_dir)), None, str(output))
    text = output.read_text(encoding="utf-8")
    before_sanity = text.split("# Sanity Checks", 1)[0]
    assert "| C19 |" not in before_sanity
    assert "| C19 vs C10 |" in text


def test_condition_summary_section_explains_it_is_not_a_ranking(
    full_results_dir: Path, tmp_path: Path
) -> None:
    output = tmp_path / "report.md"
    generate_markdown_report(load_all_results(str(full_results_dir)), None, str(output))
    section = output.read_text(encoding="utf-8").split("# Condition Summary Table", 1)[1]
    assert "not ranked" in section.split("\n# ", 1)[0]


# ---------------------------------------------------------------------------
# Statistical section
# ---------------------------------------------------------------------------
def _stats_table_lines(text: str) -> List[str]:
    """Return the table rows of the report's Cross-Dataset Consistency section."""
    body = text.split("# Cross-Dataset Consistency (Wilcoxon)", 1)[1].split("\n# ", 1)[0]
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
    assert rows[-1] == (
        "| C1 | C2 | f1 | 6 | 20/2/4 | 0.0321 | 0.0312 | True | 0.0187 | 0.7333 |  |"
    )


def test_report_stats_row_marks_untested_p_value(tmp_path: Path) -> None:
    stats = _synthetic_stats()
    stats[0]["p_value"] = None
    rows = _report_with_stats(tmp_path, stats)
    assert rows[-1].split(" | ")[5] == "N/A"


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
    assert rows[-1].split(" | ")[9] == f"{effect_size:.4f}"


def test_report_stats_tolerates_missing_effect_size(tmp_path: Path) -> None:
    stats = _synthetic_stats()
    del stats[0]["effect_size_r"]
    rows = _report_with_stats(tmp_path, stats)
    assert rows[-1] == "| C1 | C2 | f1 | 6 | 20/2/4 | 0.0321 | 0.0312 | True | 0.0187 | N/A |  |"


def test_report_stats_preserves_existing_columns(tmp_path: Path) -> None:
    header = _report_with_stats(tmp_path, _synthetic_stats())[0]
    for column in ("Condition A", "Condition B", "n", "p-value", "Corrected sig.", "delta"):
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

    def fake_run_group_comparisons(group: str, results_dir: str, metric: str = "f1") -> List[dict]:
        if group == "A":
            return _synthetic_stats()
        raise ValueError(f"insufficient data for group {group}")

    monkeypatch.setattr(script, "run_group_comparisons", fake_run_group_comparisons)
    output = tmp_path / "report.md"
    _run_cli(["--results_dir", str(full_results_dir), "--output", str(output)])
    text = output.read_text(encoding="utf-8")
    assert "# Cross-Dataset Consistency (Wilcoxon)" in text
    assert "| C1 | C2 |" in text


# ---------------------------------------------------------------------------
# Known-pair consistency (#137)
# ---------------------------------------------------------------------------
def _known(count: int) -> dict:
    return {"population": {"n_train_refs": count}}


def test_results_of_one_pair_with_different_known_pairs_are_refused(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "D1", "d1", 0.3, overrides=_known(726))
    _write_result(tmp_path, "C2", "D1", "d1", 0.3, overrides=_known(0))
    with pytest.raises(DataError, match="D1/d1 results mix known-pair counts"):
        load_all_results(str(tmp_path))


def test_matching_or_unrecorded_known_pairs_load(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "D1", "d1", 0.3, overrides=_known(726))
    _write_result(tmp_path, "C2", "D1", "d1", 0.3, overrides=_known(726))
    _write_result(tmp_path, "C3", "D1", "d1", 0.3)
    _write_result(tmp_path, "C1", "D2", "d2", 0.3, overrides=_known(0))
    assert len(load_all_results(str(tmp_path))) == 4


# ---------------------------------------------------------------------------
# MRR-first ordering and the per-dataset paired tests (#140)
# ---------------------------------------------------------------------------
def _write_scored(results_dir: Path, condition_id: str, mrr: float, f1: float) -> None:
    metrics = {**_metrics(f1), "mrr": mrr, "hits_at_1": mrr - 0.1}
    _write_result(results_dir, condition_id, "D1", "d1", f1, overrides={"metrics": metrics})


def _report_text(results_dir: Path, tmp_path: Path, **kwargs) -> str:
    output = tmp_path / "report.md"
    generate_markdown_report(load_all_results(str(results_dir)), output_path=str(output), **kwargs)
    return output.read_text(encoding="utf-8")


def test_per_dataset_breakdown_ranks_by_mrr_not_f1(tmp_path: Path) -> None:
    _write_scored(tmp_path, "C1", mrr=0.9, f1=0.2)
    _write_scored(tmp_path, "C2", mrr=0.5, f1=0.8)
    section = _per_dataset_section(_report_text(tmp_path, tmp_path))
    assert _sub_table(section, "D1")[0] == "| condition_id | mean_mrr | mean_hits_at_1 | mean_f1 |"
    assert _sub_table(section, "D1")[2:] == [
        "| C1 | 0.9000 | 0.8000 | 0.2000 |",
        "| C2 | 0.5000 | 0.4000 | 0.8000 |",
    ]


def test_executive_summary_names_the_best_condition_by_mrr(tmp_path: Path) -> None:
    _write_scored(tmp_path, "C1", mrr=0.9, f1=0.2)
    _write_scored(tmp_path, "C2", mrr=0.5, f1=0.8)
    text = _report_text(tmp_path, tmp_path)
    assert "| Dataset | Best Condition (MRR) | Strategy | Model | MRR | Hits@1 | F1 |" in text
    assert "| D1 | C1 | V1 | M1 | 0.9000 | 0.8000 | 0.2000 |" in text


def test_executive_summary_lists_every_tied_best_condition_in_registry_order(tmp_path: Path) -> None:
    for condition_id, mrr in (("C10", 0.9), ("C3", 0.9), ("C9", 0.90004), ("C1", 0.5)):
        _write_scored(tmp_path, condition_id, mrr=mrr, f1=0.2)
    text = _report_text(tmp_path, tmp_path)
    body = text.split("# Executive Summary", 1)[1].split("\n# ", 1)[0]
    rows = [line.split(" | ")[1] for line in body.splitlines() if line.startswith("| D1 |")]
    assert rows == ["C3", "C9", "C10"]
    assert "_Tied best MRR at four decimal places, one row per condition in registry order: D1 (C3, C9, C10)._" in body


def test_executive_summary_has_no_tie_note_without_a_tie(tmp_path: Path) -> None:
    _write_scored(tmp_path, "C1", mrr=0.9, f1=0.2)
    _write_scored(tmp_path, "C2", mrr=0.5, f1=0.8)
    assert "Tied best" not in _report_text(tmp_path, tmp_path)


def test_hits_at_1_of_an_older_result_is_derived_from_the_counts(tmp_path: Path) -> None:
    metrics = {key: 0.5 for key in _METRIC_KEYS}
    _write_result(tmp_path, "C1", "D1", "d1", 0.5,
                  overrides={"metrics": metrics, "population": {"n_test_refs": 12}})
    counts = {("D1", "d1"): {"n_test_refs": 12, "n_test_reference_sources": 10}}
    assert load_all_results(str(tmp_path))["hits_at_1"].isna().all()
    derived = load_all_results(str(tmp_path), counts=counts)["hits_at_1"].iloc[0]
    assert derived == pytest.approx(0.6)


def _paired_rows() -> List[dict]:
    tested = {
        "group": "E", "condition_a": "C20", "condition_b": "C10", "dataset_id": "D1",
        "tested": True, "value_a": 0.6, "value_b": 0.65, "delta": 0.05, "ci_low": 0.03,
        "ci_high": 0.07, "p_value": 0.0001, "p_holm": 0.0001, "family_size": 1,
        "significant": True, "ceiling": 0.886, "n_units": 4605, "n_pairs": 1,
    }
    rows = [{**tested, "metric": "mrr"}]
    undecided = {k: v for k, v in tested.items() if k not in ("p_holm", "family_size", "significant")}
    rows += [{**undecided, "metric": metric} for metric in ("hits_at_1", "f1")]
    untested = {"group": "D", "condition_a": "C5", "condition_b": "C14", "dataset_id": "D5",
                "tested": False, "reason": "C14/D5 has no per-source sidecar."}
    return rows + [{**untested, "metric": metric} for metric in ("mrr", "hits_at_1", "f1")]


def test_paired_section_renders_each_metric_and_the_untested_list(tmp_path: Path) -> None:
    text = _report_text(tmp_path, tmp_path, paired_results=_paired_rows())
    section = text.split("# Per-Dataset Paired Tests", 1)[1].split("\n# ", 1)[0]
    assert _headings_of(section) == ["## MRR (primary)", "## Hits@1 (primary)", "## F1 (secondary)"]
    decided = ("| E | C20 vs C10 | D1 | 0.6000 | 0.6500 | 0.0500 | [0.0300, 0.0700] | 0.0001 "
               "| 0.0001 | 1 | yes | 0.8860 | 4605 |")
    undecided = ("| E | C20 vs C10 | D1 | 0.6000 | 0.6500 | 0.0500 | [0.0300, 0.0700] | 0.0001 "
                 "| — | — | — | 0.8860 | 4605 |")
    assert section.count(decided) == 1 and section.count(undecided) == 2
    assert "--force_recompute" in section
    assert section.count("- C5 vs C14 on D5: C14/D5 has no per-source sidecar.") == 1


def test_paired_section_states_method_seed_and_replicates(tmp_path: Path) -> None:
    text = _report_text(tmp_path, tmp_path, paired_results=[])
    section = text.split("# Per-Dataset Paired Tests", 1)[1]
    assert "10,000 replicates" in section and "seed 42" in section and "Holm" in section


def test_statistical_sections_follow_the_group_analysis_in_order(tmp_path: Path) -> None:
    stats = [{**_synthetic_stats()[0], "metric": "mrr"}, *_synthetic_stats()]
    headings = _headings(_report_text(tmp_path, tmp_path, stats_results=stats, paired_results=[]))
    assert headings[4:] == [
        "# Ablation Group Analysis",
        "# Per-Dataset Paired Tests",
        "# Cross-Dataset Consistency (Wilcoxon)",
        "# PPAS Ablation",
        "# Sanity Checks",
    ]


def test_baseline_section_follows_the_paired_tests(tmp_path: Path) -> None:
    headings = _headings(_report_text(tmp_path, tmp_path, paired_results=[], baseline_results=[]))
    assert headings[5:7] == ["# Per-Dataset Paired Tests", "# Paired Tests Against B0"]
    assert "# Paired Tests Against B0" not in _report_text(tmp_path, tmp_path, paired_results=[])


def _baseline_rows() -> List[dict]:
    tested = {
        "group": "B0", "condition_a": "B0", "condition_b": "C1", "dataset_id": "D1",
        "tested": True, "value_a": 0.70, "value_b": 0.72, "delta": 0.02, "ci_low": 0.01,
        "ci_high": 0.03, "p_value": 0.002, "p_holm": 0.028, "family_size": 14,
        "significant": True, "ceiling": 0.886, "n_units": 4605, "n_pairs": 1,
    }
    undecided = {k: v for k, v in tested.items() if k not in ("p_holm", "family_size", "significant")}
    untested = {"group": "B0", "condition_a": "B0", "condition_b": "C2", "dataset_id": "D1",
                "tested": False, "reason": "C2/D1 has no per-source sidecar."}
    return ([{**tested, "metric": "mrr"}]
            + [{**undecided, "metric": m} for m in ("hits_at_1", "f1")]
            + [{**untested, "metric": m} for m in ("mrr", "hits_at_1", "f1")])


def test_baseline_section_renders_counts_tables_and_untested_cells(tmp_path: Path) -> None:
    text = _report_text(tmp_path, tmp_path, baseline_results=_baseline_rows())
    section = text.split("# Paired Tests Against B0", 1)[1].split("\n# ", 1)[0]
    assert _headings_of(section) == [
        "## MRR against B0 per column", "## MRR (primary)", "## Hits@1 (primary)", "## F1 (secondary)",
    ]
    assert "| D1 | 2 | 1 | 1 | 0 | 1 | 0 |" in section
    assert ("| B0 | B0 vs C1 | D1 | 0.7000 | 0.7200 | 0.0200 | [0.0100, 0.0300] | 0.0020 "
            "| 0.0280 | 14 | yes | 0.8860 | 4605 |") in section
    assert "- B0 vs C2 on D1: C2/D1 has no per-source sidecar." in section
    assert "Holm across every condition compared with B0 on one column" in section


def test_consistency_section_lists_mrr_before_f1_and_is_labelled(tmp_path: Path) -> None:
    stats = [*_synthetic_stats(), {**_synthetic_stats()[0], "metric": "mrr"}]
    text = _report_text(tmp_path, tmp_path, stats_results=stats)
    section = text.split("# Cross-Dataset Consistency (Wilcoxon)", 1)[1].split("\n# ", 1)[0]
    assert _headings_of(section) == ["## MRR", "## F1 (secondary)"]
    assert "consistency summary, not the primary test" in section


def test_missing_hits_at_1_is_warned_about_once(tmp_path: Path) -> None:
    metrics = {key: 0.5 for key in _METRIC_KEYS}
    for pair in ("a", "b"):
        _write_result(tmp_path, "C1", "D1", pair, 0.5, overrides={"metrics": metrics})
    logger = logging.getLogger("kgsemembed.evaluation.aggregator")
    records: List[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    logger.addHandler(handler)
    try:
        load_all_results(str(tmp_path))
    finally:
        logger.removeHandler(handler)
    messages = [r.getMessage() for r in records if "Hits@1" in r.getMessage()]
    assert len(messages) == 1 and "2 result(s)" in messages[0]


def test_cli_runs_without_the_ceilings_file(full_results_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "report.md"
    _run_cli(["--results_dir", str(full_results_dir), "--output", str(output),
              "--ceilings", str(tmp_path / "absent.json")])
    assert "# Per-Dataset Paired Tests" in output.read_text(encoding="utf-8")


def test_hits_at_1_mean_is_missing_when_any_pair_lacks_it(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "D3", "p0", 0.5)
    _write_result(tmp_path, "C1", "D3", "p1", 0.5,
                  overrides={"metrics": {key: 0.5 for key in _METRIC_KEYS}})
    summary = build_condition_summary_table(load_all_results(str(tmp_path)))
    assert pd.isna(summary.loc[0, "mean_hits_at_1"])
    assert summary.loc[0, "mean_mrr"] == pytest.approx(0.5)
    section = _per_dataset_section(_report_text(tmp_path, tmp_path))
    assert _sub_table(section, "D3")[2:] == ["| C1 | 0.5000 | N/A | 0.5000 |"]


def test_cli_survives_a_failing_paired_test_run(
    full_results_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.generate_report as script

    def broken(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(script, "run_paired_tests", broken)
    output = tmp_path / "report.md"
    _run_cli(["--results_dir", str(full_results_dir), "--output", str(output)])
    assert "# Sanity Checks" in output.read_text(encoding="utf-8")
