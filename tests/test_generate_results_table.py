"""Tests for the Section 4.1 main results table script.

Result files are written to a temporary directory and read back through the
real ``load_all_results``, so averaging, highlighting and rendering are
exercised end to end without touching the repository's ``data/results``.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterator, List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.generate_results_table as script  # noqa: E402
from kgsemembed.pipeline.conditions import get_condition  # noqa: E402
from scripts.generate_results_table import build_table  # noqa: E402

_METRIC_NAMES = (
    "precision", "recall", "threshold", "mrr", "recall_at_1", "recall_at_5", "recall_at_10"
)


def _write_result(
    results_dir: Path, condition_id: str, dataset_id: str, f1: float, pair_name: str = "p"
) -> None:
    condition = get_condition(condition_id)
    payload = {
        "condition_id": condition_id,
        "dataset_id": dataset_id,
        "pair_name": pair_name,
        "strategy": condition.strategy_name,
        "model_key": condition.model_key,
        "model_id": f"registry/{condition.model_key}",
        "metrics": {"f1": f1, **{name: 0.5 for name in _METRIC_NAMES}},
        "n_source_entities": 100,
        "n_candidates_per_entity": 20,
    }
    directory = results_dir / condition_id / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{pair_name}_results.json").write_text(json.dumps(payload))


def _rows(results_dir: Path) -> Dict[str, List[str]]:
    """Map each condition to its row cells, parsed from the built table."""
    rows = {}
    for line in build_table(str(results_dir)):
        cells = [cell.strip() for cell in line.removesuffix("\\\\").split("&")]
        if cells[0] in script._CONDITION_ORDER:
            rows[cells[0]] = cells
    return rows


@pytest.fixture
def captured_warnings() -> Iterator[List[logging.LogRecord]]:
    records: List[logging.LogRecord] = []
    handler = logging.Handler(level=logging.WARNING)
    handler.emit = records.append
    script._LOGGER.addHandler(handler)
    yield records
    script._LOGGER.removeHandler(handler)


def test_cell_is_mean_over_pairs(tmp_path: Path) -> None:
    """A D3 cell averages its pairs with equal weight."""
    _write_result(tmp_path, "C1", "D3", 0.2, "a")
    _write_result(tmp_path, "C1", "D3", 0.6, "b")
    assert _rows(tmp_path)["C1"][6] == r"\textbf{0.4000}"


def test_metadata_comes_from_condition_registry(tmp_path: Path) -> None:
    """Strategy, model and group columns reflect the registered condition."""
    assert _rows(tmp_path)["C13"][:4] == ["C13", "V2+V8", "M5", "C"]


def test_model_markers_match_caption(tmp_path: Path) -> None:
    """M2_uncapped carries the dagger and M4 the double dagger."""
    rows = _rows(tmp_path)
    assert rows["C19"][2] == r"M2$^\dagger$"
    assert rows["C17"][2] == r"M4$^\ddagger$"
    assert rows["C10"][2] == "M2"


def test_every_condition_appears_in_paper_order(tmp_path: Path) -> None:
    """All nineteen conditions are listed, in the fixed table order."""
    assert list(_rows(tmp_path)) == list(script._CONDITION_ORDER)


def test_missing_cells_render_as_dashes(tmp_path: Path) -> None:
    """A condition with no results has a dash in every value column."""
    assert _rows(tmp_path)["C14"][4:] == ["---"] * 7


def test_best_per_column_is_bold_and_ties_are_all_bold(tmp_path: Path) -> None:
    """Values equal at four decimal places are both highlighted."""
    _write_result(tmp_path, "C1", "D1", 0.50001)
    _write_result(tmp_path, "C2", "D1", 0.50004)
    _write_result(tmp_path, "C3", "D1", 0.3)
    rows = _rows(tmp_path)
    assert rows["C1"][4] == r"\textbf{0.5000}"
    assert rows["C2"][4] == r"\textbf{0.5000}"
    assert rows["C3"][4] == "0.3000"


def _write_d1_to_d4(results_dir: Path, condition_id: str, f1s: List[float]) -> None:
    for dataset_id, f1 in zip(("D1", "D2", "D3", "D4_schema", "D4_instance"), f1s):
        _write_result(results_dir, condition_id, dataset_id, f1)


def test_mean_column_averages_d1_to_d4_and_excludes_d5(tmp_path: Path) -> None:
    """With every D1-D4 cell present, the mean covers them and ignores D5."""
    _write_d1_to_d4(tmp_path, "C1", [0.1, 0.2, 0.3, 0.4, 0.5])
    _write_result(tmp_path, "C1", "D5", 0.9)
    assert _rows(tmp_path)["C1"][-1] == "0.3000"


def test_mean_column_is_dash_without_full_coverage(tmp_path: Path) -> None:
    """A condition missing any D1-D4 cell, such as C17, reports no mean."""
    _write_result(tmp_path, "C17", "D1", 0.4)
    _write_result(tmp_path, "C17", "D2", 0.6)
    assert _rows(tmp_path)["C17"][-1] == "---"


def test_mean_column_is_dash_when_one_d4_column_is_missing(tmp_path: Path) -> None:
    """Both D4 columns count, so lacking D4_instance withholds the mean."""
    _write_d1_to_d4(tmp_path, "C1", [0.1, 0.2, 0.3, 0.4])
    assert _rows(tmp_path)["C1"][-1] == "---"


def test_mean_column_is_dash_with_only_d5(tmp_path: Path) -> None:
    """A D5-only condition such as C14 has no mean to report."""
    _write_result(tmp_path, "C14", "D5", 0.06)
    assert _rows(tmp_path)["C14"][-2:] == [r"\textbf{0.0600}", "---"]


def test_rules_separate_ablation_groups(tmp_path: Path) -> None:
    """A rule appears before the first condition of each new group."""
    lines = build_table(str(tmp_path))
    for first_of_group in ("C9", "C4", "C14"):
        index = next(i for i, line in enumerate(lines) if line.startswith(f"{first_of_group} &"))
        assert lines[index - 1] == r"\hline"
    index = next(i for i, line in enumerate(lines) if line.startswith("C2 &"))
    assert lines[index - 1].startswith("C1 &")


def test_table_is_a_complete_environment(tmp_path: Path) -> None:
    """The output opens and closes the table and declares eleven columns."""
    lines = build_table(str(tmp_path))
    assert lines[0] == r"\begin{table*}[t]"
    assert lines[-1] == r"\end{table*}"
    assert r"\begin{tabular}{llllrrrrrrr}" in lines
    assert any(line.startswith(r"\multicolumn{11}{l}") for line in lines)


def test_caption_scopes_dashes_to_dataset_columns(tmp_path: Path) -> None:
    """A Mean dash means partial coverage, so the caption limits its dash note."""
    caption = next(line for line in build_table(str(tmp_path)) if line.startswith(r"\caption"))
    assert "Dashes in dataset columns indicate conditions not run" in caption


def test_partial_d3_pair_count_warns(
    tmp_path: Path, captured_warnings: List[logging.LogRecord]
) -> None:
    """A D3 cell over fewer than 21 pairs contradicts the caption and warns."""
    _write_result(tmp_path, "C1", "D3", 0.5)
    build_table(str(tmp_path))
    assert any("C1/D3 has 1 pairs" in record.getMessage() for record in captured_warnings)


def test_malformed_result_file_is_skipped(tmp_path: Path) -> None:
    """An unreadable file is skipped rather than aborting the table."""
    _write_result(tmp_path, "C1", "D1", 0.5)
    (tmp_path / "C1" / "D1" / "broken_results.json").write_text("{not json")
    assert _rows(tmp_path)["C1"][4] == r"\textbf{0.5000}"


def test_cli_prints_table(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The CLI prints the table read from the given results directory."""
    _write_result(tmp_path, "C1", "D1", 0.25)
    original = sys.argv
    sys.argv = ["generate_results_table.py", "--results_dir", str(tmp_path)]
    try:
        script.main()
    finally:
        sys.argv = original
    out = capsys.readouterr().out
    assert out.startswith(r"\begin{table*}[t]")
    assert r"C1 & V1 & M1 & A & \textbf{0.2500} & --- &" in out
