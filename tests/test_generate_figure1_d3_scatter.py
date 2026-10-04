"""Tests for the Figure 1 D3 scatter script.

Result files are written to a temporary directory, so loading, pairing,
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

from scripts.generate_figure1_d3_scatter import (  # noqa: E402
    build_figure,
    pair_scores,
    save_figure,
    summary_lines,
)


def _write_result(results_dir: Path, condition_id: str, pair_name: str, f1: float) -> None:
    directory = results_dir / condition_id / "D3"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"pair_name": pair_name, "metrics": {"f1": f1}}
    (directory / f"{pair_name}_results.json").write_text(json.dumps(payload))


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    for pair, c3_f1, c10_f1 in [
        ("cmt-edas", 0.2, 0.4),
        ("confof-ekaw", 0.3, 0.1),
        ("ekaw-iasted", 0.15, 0.15),
    ]:
        _write_result(tmp_path, "C3", pair, c3_f1)
        _write_result(tmp_path, "C10", pair, c10_f1)
    _write_result(tmp_path, "C3", "iasted-sigkdd", 0.5)
    return tmp_path


def test_pair_scores_keeps_only_pairs_under_both_conditions(results_dir: Path) -> None:
    assert pair_scores(results_dir) == {
        "cmt-edas": (0.2, 0.4),
        "confof-ekaw": (0.3, 0.1),
        "ekaw-iasted": (0.15, 0.15),
    }


def test_pair_scores_raises_without_shared_pairs(tmp_path: Path) -> None:
    _write_result(tmp_path, "C3", "cmt-edas", 0.2)
    with pytest.raises(ValueError):
        pair_scores(tmp_path)


def test_summary_counts_wins_out_of_pairs_plotted(results_dir: Path) -> None:
    lines = summary_lines(pair_scores(results_dir))
    assert lines[-2] == "\nC10 wins: 1/3, C3 wins: 1/3"
    assert lines[-1] == "C10 wins on sparse pairs: 1/1"


def test_summary_stars_sparse_pairs_and_reports_ties(results_dir: Path) -> None:
    rows = summary_lines(pair_scores(results_dir))[2:5]
    assert rows[0].startswith("cmt-edas *")
    assert rows[1].split()[-1] == "C3"
    assert rows[2].split()[-1] == "tie"


def test_save_figure_writes_pdf_and_png(results_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "figures"
    pdf_path = save_figure(build_figure(pair_scores(results_dir)), output_dir)
    assert pdf_path == output_dir / "fig1_d3_scatter.pdf"
    assert pdf_path.stat().st_size > 0
    assert (output_dir / "fig1_d3_scatter.png").stat().st_size > 0
