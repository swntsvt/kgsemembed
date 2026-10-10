"""Tests for the Figure 2 model comparison script.

Result files are written to a temporary directory, so loading, validation,
plotting and the printed summary are exercised without touching the
repository's ``data/results`` or ``figures``.
"""

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.generate_figure2_model_comparison import (  # noqa: E402
    BARS,
    build_figure,
    collect_scores,
    save_figure,
    summary_lines,
)

_PAIRS = {"D1": "d1_snomed_fma", "D2": "d2_anatomy"}


def _write_result(results_dir: Path, condition_id: str, dataset_id: str,
                  f1: float, strategy_model: tuple[str, str] | None = None) -> None:
    bar = next(b for b in BARS if b.condition_id == condition_id)
    strategy, model_key = strategy_model or (bar.strategy, bar.model_key)
    directory = results_dir / condition_id / dataset_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"strategy": strategy, "model_key": model_key, "metrics": {"f1": f1}}
    (directory / f"{_PAIRS[dataset_id]}_results.json").write_text(json.dumps(payload))


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    for i, bar in enumerate(BARS):
        _write_result(tmp_path, bar.condition_id, "D1", 0.1 * (i + 1))
        if bar.condition_id != "C15":
            _write_result(tmp_path, bar.condition_id, "D2", 0.05 * (i + 1))
    return tmp_path


def test_bars_hold_v2v8_constant_except_reference() -> None:
    assert [b.model_key for b in BARS[:-1]] == ["M1", "M2", "M3", "M4", "M5"]
    assert {b.strategy for b in BARS[:-1]} == {"V2+V8"}
    assert (BARS[-1].condition_id, BARS[-1].strategy, BARS[-1].model_key) == ("C17", "V1", "M4")


def test_collect_scores_aligns_values_with_bars(results_dir: Path) -> None:
    scores = collect_scores(results_dir)
    assert scores["D1"] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert scores["D2"][1] == pytest.approx(0.1)


def test_missing_result_is_none_not_zero(results_dir: Path) -> None:
    assert collect_scores(results_dir)["D2"][2] is None


def test_mismatched_strategy_raises(tmp_path: Path) -> None:
    _write_result(tmp_path, "C10", "D1", 0.3, strategy_model=("V2+V6", "M2"))
    with pytest.raises(ValueError, match="C10"):
        collect_scores(tmp_path)


def test_no_results_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        collect_scores(tmp_path)


def test_summary_marks_missing_values(results_dir: Path) -> None:
    lines = summary_lines(collect_scores(results_dir))
    c15_row = next(line for line in lines if line.startswith("C15"))
    assert c15_row.split() == ["C15", "V2+V8", "M3", "0.3000", "n/a"]
    assert len(lines) == len(BARS) + 2


def test_save_figure_writes_pdf_and_png(results_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "figures"
    pdf_path = save_figure(build_figure(collect_scores(results_dir)), output_dir)
    assert pdf_path == output_dir / "fig2_model_comparison.pdf"
    assert pdf_path.stat().st_size > 0
    assert (output_dir / "fig2_model_comparison.png").stat().st_size > 0
