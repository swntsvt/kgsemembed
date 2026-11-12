"""Tests for the condition × column coverage table (#163)."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.coverage_table as coverage  # noqa: E402
from kgsemembed.pipeline.conditions import EXPERIMENT_CONDITIONS, get_condition  # noqa: E402


def _write(root: Path, condition_id: str, column: str, pairs: int = 1) -> None:
    directory = root / condition_id / column
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(pairs):
        (directory / f"p{index}{coverage.RESULT_SUFFIX}").write_text("{}", encoding="utf-8")


def test_registered_column_shows_its_pair_count(tmp_path: Path) -> None:
    _write(tmp_path, "C4", "D3", pairs=21)
    _write(tmp_path, "C4", "D4_instance")
    assert coverage.cell(tmp_path, get_condition("C4"), "D3") == "21"
    assert coverage.cell(tmp_path, get_condition("C4"), "D4_instance") == "1"


def test_registered_column_without_results_is_not_run(tmp_path: Path) -> None:
    assert coverage.cell(tmp_path, get_condition("C7"), "D5") == "not run"


def test_unregistered_column_carries_its_reason(tmp_path: Path) -> None:
    assert coverage.cell(tmp_path, get_condition("C17"), "D3") == "by design (a)"
    assert coverage.cell(tmp_path, get_condition("C15"), "D5") == "by design (c)"


def test_unjustified_gap_raises(tmp_path: Path) -> None:
    narrowed = replace(get_condition("C4"), datasets=["D1"])
    with pytest.raises(ValueError, match="C4 on D2"):
        coverage.cell(tmp_path, narrowed, "D2")


def test_every_registered_gap_is_justified(tmp_path: Path) -> None:
    lines = coverage.table_lines(tmp_path)
    assert len(lines) == 2 + len(EXPERIMENT_CONDITIONS)


def test_conditions_with_reasons_are_not_run_everywhere() -> None:
    full = {c.condition_id for c in EXPERIMENT_CONDITIONS if set(c.datasets) == {"D1", "D2", "D3", "D4", "D5"}}
    assert full.isdisjoint(coverage.BY_DESIGN)


def test_markdown_regenerates_byte_identically(tmp_path: Path) -> None:
    _write(tmp_path, "C1", "D1")
    first, second = (coverage.render_markdown(tmp_path) for _ in range(2))
    assert first == second and first.endswith(".\n")
    assert "| C1 | V1 | M1 | A | 1 | not run |" in first
    assert "- (d) C19:" in first and "**Coverage rule.**" in first


def test_main_writes_the_artifact(tmp_path: Path) -> None:
    output = tmp_path / "out" / "coverage.md"
    coverage.main(["--results_dir", str(tmp_path), "--output", str(output)])
    assert output.read_text(encoding="utf-8") == coverage.render_markdown(tmp_path)
