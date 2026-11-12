import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.rerun_comparison as rerun  # noqa: E402


def test_expected_change_names_each_correction() -> None:
    assert rerun.expected_change("C6", "D1") == "budget, M4 512"
    assert rerun.expected_change("C14", "D5") == "D5 suffix"
    assert rerun.expected_change("C4", "D5") == "budget, D5 suffix"
    assert rerun.expected_change("C14", "D4_instance") is None
    assert rerun.expected_change("C10", "D5") is None
    assert rerun.expected_change("B0", "D2") is None


def _payload(mrr: float, budget=None, over: int = 0) -> dict:
    role = {"n_over_budget": over, "n_truncated": 0}
    return {"metrics": {"mrr": mrr}, "counts": {"tp": 1}, "git_commit": "abc",
            "encoder_input": {"token_budget": budget, "source": role, "candidate": role}}


def _write(root: Path, condition: str, column: str, payload: dict, inputs: bool = True) -> None:
    folder = root / condition / column
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "p_results.json").write_text(json.dumps(payload))
    (folder / "p_entities.json").write_text(json.dumps({"inputs": {}} if inputs else {}))


def test_unaffected_change_fails_and_expected_change_passes(tmp_path) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    for condition in ("C10", "C4"):
        _write(old, condition, "D2", _payload(0.5))
    _write(new, "C10", "D2", _payload(0.6))
    _write(new, "C4", "D2", _payload(0.6, budget=420))
    rows, failures = rerun.compare(old, new)
    assert len(rows) == 2
    assert failures == ["('C10', 'D2', 'p'): must reproduce but differs in ['mrr']"]


def test_budgeted_result_over_budget_or_without_inputs_fails(tmp_path) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    _write(old, "C4", "D2", _payload(0.5))
    _write(new, "C4", "D2", _payload(0.5, budget=420, over=3), inputs=False)
    _, failures = rerun.compare(old, new)
    assert any("no per-entity inputs" in f for f in failures)
    assert any("over budget" in f for f in failures)


def test_c19_must_equal_c10() -> None:
    rows = [{"key": ("C10", "D1", "p"), "new": _payload(0.5)}, {"key": ("C19", "D1", "p"), "new": _payload(0.5)}]
    assert rerun.c19_problems(rows) == []
    rows[1]["new"] = _payload(0.4)
    assert rerun.c19_problems(rows) == ["C19 differs from C10 on ('D1', 'p')"]
    assert rerun.c19_problems(rows[:1]) == ["C19 = C10: no pair where both ran"]
