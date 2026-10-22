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
    _LABEL_GAP,
    _LABEL_MIN_X,
    _LABEL_MIN_Y,
    _axis_limit,
    build_figure,
    label_positions,
    labelled_pairs,
    pair_scores,
    save_figure,
    summary_lines,
)


def _write_result(results_dir: Path, condition_id: str, pair_name: str, f1: float) -> None:
    directory = results_dir / condition_id / "D3"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"pair_name": pair_name, "metrics": {"f1": f1, "mrr": f1 / 2}}
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


def test_pair_scores_default_to_the_primary_metric(results_dir: Path) -> None:
    assert pair_scores(results_dir)["cmt-edas"] == (0.1, 0.2)
    assert summary_lines(pair_scores(results_dir))[0].split()[1:5] == ["C3", "MRR", "C10", "MRR"]


def test_metric_suffixes_the_file_of_a_secondary_metric(results_dir: Path, tmp_path: Path) -> None:
    scores = pair_scores(results_dir, "f1")
    pdf_path = save_figure(build_figure(scores, "f1"), tmp_path / "figures", "f1")
    assert pdf_path.name == "fig1_d3_scatter_f1.pdf"


def test_pair_scores_keeps_only_pairs_under_both_conditions(results_dir: Path) -> None:
    assert pair_scores(results_dir, "f1") == {
        "cmt-edas": (0.2, 0.4),
        "confof-ekaw": (0.3, 0.1),
        "ekaw-iasted": (0.15, 0.15),
    }


def test_pair_scores_raises_without_shared_pairs(tmp_path: Path) -> None:
    _write_result(tmp_path, "C3", "cmt-edas", 0.2)
    with pytest.raises(ValueError):
        pair_scores(tmp_path)


def test_summary_counts_wins_out_of_pairs_plotted(results_dir: Path) -> None:
    lines = summary_lines(pair_scores(results_dir, "f1"), "f1")
    assert lines[-2] == "\nC10 wins: 1/3, C3 wins: 1/3"
    assert lines[-1] == "C10 wins on CMT/EDAS pairs: 1/1"


def test_summary_stars_sparse_pairs_and_reports_ties(results_dir: Path) -> None:
    rows = summary_lines(pair_scores(results_dir, "f1"), "f1")[2:5]
    assert rows[0].startswith("cmt-edas *")
    assert rows[1].split()[-1] == "C3"
    assert rows[2].split()[-1] == "tie"


def test_save_figure_writes_pdf_and_png(results_dir: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "figures"
    pdf_path = save_figure(build_figure(pair_scores(results_dir)), output_dir)
    assert pdf_path == output_dir / "fig1_d3_scatter.pdf"
    assert pdf_path.stat().st_size > 0
    assert (output_dir / "fig1_d3_scatter.png").stat().st_size > 0


def test_labels_name_every_pair_where_c10_is_ahead() -> None:
    scores = {"a-b": (0.2, 0.4), "c-d": (0.3, 0.1), "e-f": (0.15, 0.15), "g-h": (0.5, 0.55)}
    assert labelled_pairs(scores) == ["a-b", "g-h"]


def test_label_positions_never_share_a_row() -> None:
    scores = {f"p{i}": (0.1, 0.3 + 0.001 * i) for i in range(6)}
    rows = sorted(y for _, y in label_positions(scores).values())
    gap = _LABEL_GAP * _axis_limit(scores)
    assert all(high - low >= gap - 1e-12 for low, high in zip(rows, rows[1:]))


def test_label_stack_is_lifted_above_the_axis_floor() -> None:
    scores = {f"p{i}": (0.02, 0.05 + 0.001 * i) for i in range(6)}
    scores["top"] = (0.1, 0.9)
    lim = _axis_limit(scores)
    positions = label_positions(scores)
    assert min(y for _, y in positions.values()) == pytest.approx(_LABEL_MIN_Y * lim)
    assert all(0 < x < lim and y < lim for x, y in positions.values())


def test_label_near_the_y_axis_stays_inside_it() -> None:
    scores = {"near": (0.05, 0.6), "far": (0.6, 0.9)}
    positions = label_positions(scores)
    assert positions["near"][0] == pytest.approx(_LABEL_MIN_X * _axis_limit(scores))
    assert positions["far"][0] < 0.6
