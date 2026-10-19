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
    results_dir: Path, condition_id: str, dataset_id: str, value: float, pair_name: str = "p",
    **extra: object,
) -> None:
    """Write a result whose F1 and MRR are ``value``; ``extra`` overrides top-level fields."""
    condition = get_condition(condition_id)
    payload = {
        "condition_id": condition_id,
        "dataset_id": dataset_id,
        "pair_name": pair_name,
        "strategy": condition.strategy_name,
        "model_key": condition.model_key,
        "model_id": f"registry/{condition.model_key}",
        "metrics": {**{name: 0.5 for name in _METRIC_NAMES}, "f1": value, "mrr": value},
        "counts": {"tp": 1, "fp": 1, "fn": 1},
        "population": {"n_test_refs": 4},
        "n_source_entities": 100,
        "n_candidates_per_entity": 20,
        **extra,
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
    """All twenty conditions are listed, in the fixed table order."""
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


def test_mean_note_sits_below_the_tabular(tmp_path: Path) -> None:
    """The note is a threeparttable note, not a row that widens the columns."""
    lines = build_table(str(tmp_path))
    assert not any(r"\multicolumn" in line for line in lines)
    assert lines[-7:] == [
        r"\hline",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        script._MEAN_NOTE,
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table*}",
    ]
    assert script._MEAN_NOTE.startswith(r"\item[*] Mean over D1--D4")


def test_caption_and_header_are_inside_threeparttable(tmp_path: Path) -> None:
    """The caption takes the table's width and the Mean head carries the mark."""
    lines = build_table(str(tmp_path))
    start = lines.index(r"\begin{threeparttable}")
    assert lines[start + 1].startswith(r"\caption{")
    assert any(r"\textbf{Mean\tnote{*}}" in line for line in lines)


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


# ---------------------------------------------------------------------------
# Primary metric, Hits@1, ceilings and F1 sensitivity (#139)
# ---------------------------------------------------------------------------
def _write_counts(path: Path, rows: List[dict]) -> str:
    defaults = {
        "n_val_refs": 2, "n_val_reference_sources": 2, "n_test_refs": 4,
        "n_test_reference_sources": 4, "n_test_multi_target_sources": 0,
        "n_test_sources_gold_listed": 3, "n_test_refs_listed": 3,
        "ranking_ceiling": 0.75, "candidate_recall_at_20": 0.75, "f1_recall_ceiling": 0.75,
    }
    path.write_text(json.dumps({"pairs": [{**defaults, **row} for row in rows]}))
    return str(path)


def _caption_and_label(lines: List[str]) -> List[str]:
    start = lines.index(r"\begin{threeparttable}")
    return lines[start + 1:start + 3]


def _annotation(lines: List[str], label: str) -> List[str]:
    line = next(line for line in lines if line.startswith(rf"\multicolumn{{4}}{{l}}{{{label}}}"))
    return [cell.strip() for cell in line.removesuffix("\\\\").split("&")][1:]


def test_primary_metric_is_the_default_main_table(tmp_path: Path) -> None:
    """MRR is the main table; F1 keeps a table of its own under another label."""
    caption, label = _caption_and_label(build_table(str(tmp_path)))
    assert label == r"\label{tab:main_results}"
    assert "re-ranking MRR" in caption and "primary metric" in caption
    assert _caption_and_label(build_table(str(tmp_path), "f1"))[1] == r"\label{tab:matching_f1}"
    assert script._build_arg_parser().parse_args([]).metric == "mrr"


def test_hits_at_1_is_derived_from_recall_at_1_for_one_to_many_pairs(tmp_path: Path) -> None:
    """Three references over two sources: Recall@1 1/3 is Hits@1 1/2."""
    results = tmp_path / "results"
    _write_result(results, "C1", "D1", 0.5, "d1", population={"n_test_refs": 3},
                  metrics={**{n: 0.5 for n in _METRIC_NAMES}, "f1": 0.5, "recall_at_1": 1 / 3})
    counts = _write_counts(tmp_path / "counts.json", [
        {"dataset_id": "D1", "pair_name": "d1", "n_test_refs": 3, "n_test_reference_sources": 2},
    ])
    lines = build_table(str(results), "hits_at_1", counts_path=counts)
    row = next(line for line in lines if line.startswith("C1 &"))
    assert r"\textbf{0.5000}" in row
    assert "Hits@1" in _caption_and_label(lines)[0]


def test_recorded_hits_at_1_is_used_as_is(tmp_path: Path) -> None:
    """A result that records hits_at_1 needs no counts file."""
    _write_result(tmp_path, "C1", "D1", 0.5,
                  metrics={**{n: 0.5 for n in _METRIC_NAMES}, "f1": 0.5, "hits_at_1": 0.625})
    row = next(line for line in build_table(str(tmp_path), "hits_at_1") if line.startswith("C1 &"))
    assert r"\textbf{0.6250}" in row


def test_hits_at_1_without_record_or_counts_raises(tmp_path: Path) -> None:
    _write_result(tmp_path, "C1", "D1", 0.5)
    with pytest.raises(ValueError, match="metric_ceilings"):
        build_table(str(tmp_path), "hits_at_1")


def test_ranking_table_has_a_ceiling_row(tmp_path: Path) -> None:
    """The ceiling row averages D3 pairs like the metric and fills the Mean column."""
    counts = _write_counts(tmp_path / "counts.json", [
        {"dataset_id": "D1", "pair_name": "a", "ranking_ceiling": 0.9},
        {"dataset_id": "D2", "pair_name": "a", "ranking_ceiling": 0.8},
        {"dataset_id": "D3", "pair_name": "a", "ranking_ceiling": 0.6},
        {"dataset_id": "D3", "pair_name": "b", "ranking_ceiling": 1.0},
        {"dataset_id": "D4_schema", "pair_name": "a", "ranking_ceiling": 0.7},
        {"dataset_id": "D4_instance", "pair_name": "a", "ranking_ceiling": 0.6},
        {"dataset_id": "D5", "pair_name": "a", "ranking_ceiling": 0.2},
    ])
    cells = _annotation(build_table(str(tmp_path / "r"), "mrr", counts_path=counts), "Top-20 ceiling")
    assert cells == ["0.9000", "0.8000", "0.8000", "0.7000", "0.6000", "0.2000", "0.7600"]


def test_f1_table_reports_d3_micro_beside_macro(tmp_path: Path) -> None:
    """Pooling TP/FP/FN differs from averaging per-pair F1 when pairs differ in size."""
    _write_result(tmp_path, "C1", "D3", 1.0, "a", counts={"tp": 1, "fp": 0, "fn": 0})
    _write_result(tmp_path, "C1", "D3", 0.0, "b", counts={"tp": 0, "fp": 9, "fn": 9})
    lines = build_table(str(tmp_path), "f1")
    assert r"\begin{tabular}{llllrrrrrrrr}" in lines
    assert any(r"\textbf{D3$_{\mu}$}" in line for line in lines)
    row = next(line for line in lines if line.startswith("C1 &"))
    cells = [cell.strip() for cell in row.removesuffix("\\\\").split("&")]
    assert cells[6:8] == [r"\textbf{0.5000}", r"\textbf{0.1000}"]


def test_f1_table_shows_reference_counts_and_recall_ceiling(tmp_path: Path) -> None:
    """D3 validation counts appear as their range over pairs and in total for D3 micro."""
    counts = _write_counts(tmp_path / "counts.json", [
        {"dataset_id": "D1", "pair_name": "a", "n_val_refs": 1304, "f1_recall_ceiling": 0.78},
        {"dataset_id": "D3", "pair_name": "a", "n_val_refs": 1, "n_test_refs": 4,
         "n_test_sources_gold_listed": 2, "f1_recall_ceiling": 0.5},
        {"dataset_id": "D3", "pair_name": "b", "n_val_refs": 5, "n_test_refs": 16,
         "n_test_sources_gold_listed": 16, "f1_recall_ceiling": 1.0},
    ])
    lines = build_table(str(tmp_path / "r"), "f1", counts_path=counts)
    assert _annotation(lines, "Validation refs")[:4] == ["1,304", "---", "1--5", "6"]
    assert _annotation(lines, "Test refs")[2:4] == ["4--16", "20"]
    assert _annotation(lines, "Recall ceiling")[:4] == ["0.7800", "---", "0.7500", "0.9000"]
