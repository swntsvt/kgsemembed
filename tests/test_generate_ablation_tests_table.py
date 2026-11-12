import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.generate_ablation_tests_table as table  # noqa: E402
from kgsemembed.evaluation.stats import expand_dataset_ids, registered_comparisons  # noqa: E402


def _rows(delta: float = 0.0123, significant: bool = True) -> list:
    rows = []
    for group, comparisons in registered_comparisons().items():
        for condition_a, condition_b, datasets in comparisons:
            for dataset_id in expand_dataset_ids(list(datasets)):
                keys = {"group": group, "condition_a": condition_a, "condition_b": condition_b,
                        "dataset_id": dataset_id}
                rows.append({**keys, "metric": "mrr", "tested": True, "delta": delta,
                             "significant": significant})
                rows.append({**keys, "metric": "f1", "tested": True, "delta": 9.0})
    return rows


@pytest.mark.parametrize("delta, significant, expected", [
    (0.0394, True, "$+0.039$$^{*}$"), (-0.0351, False, "$-0.035$"),
    (0.0, False, "$0.000$"), (-0.0004, True, "$0.000$$^{*}$"), (0.4683, True, "$+0.468$$^{*}$"),
])
def test_format_delta(delta, significant, expected):
    assert table.format_delta(delta, significant) == expected


def test_decision_cells_keep_only_mrr():
    cells = table.decision_cells(_rows())
    assert set(cells.values()) == {"$+0.012$$^{*}$"}
    assert ("A", "C1", "C2", "D4_schema") in cells


def test_decision_cells_refuse_an_untested_cell():
    rows = _rows()
    rows[0].update(tested=False, reason="no sidecar")
    with pytest.raises(ValueError, match="no sidecar"):
        table.decision_cells(rows)


def test_render_table_has_one_row_per_comparison_and_rules_between_groups():
    lines = table.render_table(table.decision_cells(_rows(significant=False))).splitlines()
    assert lines[0] == r"\begin{table*}[t]" and lines[-1] == r"\end{table*}"
    assert r"\label{tab:stats}" in lines
    body = [line for line in lines if line.count("&") == 8]
    assert len(body) == 1 + sum(len(c) for c in registered_comparisons().values())
    assert lines.count(r"\hline") == 2 + len(registered_comparisons())


def test_render_table_marks_uncovered_columns():
    text = table.render_table(table.decision_cells(_rows()))
    assert r"D & C5 & C14 & --- & --- & --- & --- & --- & $+0.012$$^{*}$ \\" in text
    assert r"A & C1 & C2 & " + " & ".join(["$+0.012$$^{*}$"] * 6) + r" \\" in text


def test_render_table_refuses_a_missing_covered_cell():
    cells = table.decision_cells(_rows())
    del cells[("E", "C20", "C10", "D5")]
    with pytest.raises(KeyError):
        table.render_table(cells)
