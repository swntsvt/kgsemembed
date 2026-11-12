"""Tests for the paired intervals behind Figures 1 and 2 (#149).

A stub cache stands in for the sidecars, so the subgroup, per-pair and
pairwise logic is exercised without result files.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kgsemembed.evaluation import paired_tests  # noqa: E402
from kgsemembed.utils.errors import DataError  # noqa: E402
from scripts import figure_claims  # noqa: E402
from scripts.figure_specs import DATASETS  # noqa: E402
from scripts.figure_claims import (  # noqa: E402
    _direction,
    add_holm,
    column_arrays,
    comparison_rows,
    figure1,
    figure2_conditions,
    pair_delta,
    pair_level_interval,
    render_markdown,
    subgroup_difference,
    subgroup_pairs,
)

_SETTINGS = paired_tests.TestSettings(n_replicates=200)
_SOURCES = [f"s{i}" for i in range(20)]
_FIG1 = {"condition_a": "C3", "condition_b": "C10", "dataset_id": "D3"}
_D1, _D2 = DATASETS


def _ranks(hit_share: float) -> dict:
    """Unit values of a pair where the first ``hit_share`` of sources rank gold first."""
    cut = int(hit_share * len(_SOURCES))
    return {s: (1.0 if i < cut else 0.25, 1.0) for i, s in enumerate(_SOURCES)}


def _f1_units(hit_share: float) -> dict:
    """F1 units of a pair: a TP for the first ``hit_share`` of sources, else a FN."""
    cut = int(hit_share * len(_SOURCES))
    return {s: (1.0, 0.0, 0.0) if i < cut else (0.0, 0.0, 1.0) for i, s in enumerate(_SOURCES)}


class StubCache:
    """Serve fixed sidecars and unit values for C3 and C10 on D3."""

    def __init__(self, shares: dict, gold: str = "t") -> None:
        self._shares, self._gold = shares, gold

    def sidecars(self, condition_id: str, dataset_id: str) -> dict:
        gold = self._gold if condition_id == "C10" else "t"
        records = [{"source": s, "gold_targets": [gold]} for s in _SOURCES]
        return {pair: {"test_sources": records, "setup": {}} for pair in self._shares}

    def values(self, condition_id: str, dataset_id: str, metric: str) -> dict:
        side = 0 if condition_id == "C3" else 1
        make = _f1_units if metric == "f1" else _ranks
        return {pair: make(shares[side]) for pair, shares in self._shares.items()}


@pytest.fixture
def cache() -> StubCache:
    return StubCache({"cmt-edas": (0.2, 0.8), "edas-ekaw": (0.3, 0.7),
                      "confof-ekaw": (0.8, 0.2), "ekaw-iasted": (0.7, 0.3)})


def test_subgroups_split_on_cmt_and_edas(cache: StubCache) -> None:
    arrays = column_arrays(cache, _FIG1)["mrr"]
    assert set(subgroup_pairs(arrays, "cmt_edas")) == {"cmt-edas", "edas-ekaw"}
    assert set(subgroup_pairs(arrays, "other")) == {"confof-ekaw", "ekaw-iasted"}
    assert len(subgroup_pairs(arrays, "all")) == 4


def test_pair_delta_is_b_minus_a(cache: StubCache) -> None:
    arrays = column_arrays(cache, _FIG1)["mrr"]
    assert pair_delta(arrays["cmt-edas"], "mrr") == pytest.approx(0.6 * 0.75)


def test_different_references_are_refused() -> None:
    with pytest.raises(DataError, match="Test references differ"):
        column_arrays(StubCache({"cmt-edas": (0.2, 0.8)}, gold="u"), _FIG1)


def test_figure1_separates_the_two_subgroups(cache: StubCache) -> None:
    result = figure1(cache, _SETTINGS)["mrr"]
    assert result["subgroups"]["cmt_edas"]["delta"] > 0 > result["subgroups"]["other"]["delta"]
    difference = result["difference"]
    assert difference["ci_sources"][0] > 0 and difference["ci_pairs"][0] > 0
    assert {row["favours"] for row in result["pairs"]} == {"A", "B"}


def test_pair_level_interval_of_constant_deltas_is_their_difference() -> None:
    low, high = pair_level_interval([0.3, 0.3], [0.1, 0.1, 0.1], _SETTINGS)
    assert low == pytest.approx(0.2) and high == pytest.approx(0.2)


def test_direction_names_the_favoured_condition() -> None:
    assert _direction({"ci_low": 0.1, "ci_high": 0.2}) == "B"
    assert _direction({"ci_low": -0.2, "ci_high": -0.1}) == "A"
    assert _direction({"ci_low": -0.1, "ci_high": 0.1}) == "none"


def test_holm_applies_to_mrr_rows_only() -> None:
    rows = add_holm([{"metric": "mrr", "tested": True, "p_value": 0.01},
                     {"metric": "mrr", "tested": True, "p_value": 0.04},
                     {"metric": "f1", "tested": True, "p_value": 0.01}])
    assert [row.get("p_holm") for row in rows] == [pytest.approx(0.02), pytest.approx(0.04), None]
    assert [row.get("family_size") for row in rows] == [2, 2, None]


def test_holm_is_withheld_when_the_family_is_incomplete() -> None:
    untested = {"dataset_id": "D1", "condition_a": "C18", "condition_b": "C13", "metric": "mrr",
                "tested": False, "reason": "stale"}
    tested = {"dataset_id": "D1", "condition_a": "C18", "condition_b": "C10", "metric": "mrr",
              "tested": True, "p_value": 0.01, "value_a": 0.6, "value_b": 0.7, "delta": 0.1,
              "ci_low": 0.05, "ci_high": 0.15}
    rows = add_holm([tested, untested])
    assert rows[0]["p_holm"] is None and rows[0]["family_size"] == 2
    assert "n/a (family incomplete)" in render_markdown({}, rows)


def _write_result(root: Path, condition: str, strategy: str, model_key: str,
                  name: str = "d2_anatomy") -> None:
    directory = root / condition / "D2"
    directory.mkdir(parents=True, exist_ok=True)
    result = {"strategy": strategy, "model_key": model_key}
    (directory / f"{name}_results.json").write_text(json.dumps(result))


def test_figure2_conditions_are_every_expected_bar_in_order(tmp_path: Path) -> None:
    _write_result(tmp_path, "C10", "V2+V8", "M2")
    assert figure2_conditions(tmp_path, _D1) == ["C18", "C10", "C15", "C11", "C13", "C17"]
    assert figure2_conditions(tmp_path, _D2) == ["C18", "C10", "C15", "C11", "C13", "C17"]


def test_figure2_conditions_refuse_a_mislabelled_result(tmp_path: Path) -> None:
    _write_result(tmp_path, "C11", "V1", "M4")
    with pytest.raises(ValueError, match="C11 result has strategy/model"):
        figure2_conditions(tmp_path, _D2)


def test_figure2_conditions_check_only_the_plotted_file(tmp_path: Path) -> None:
    _write_result(tmp_path, "C11", "V1", "M4", name="smoke_run")
    assert "C11" in figure2_conditions(tmp_path, _D2)


def test_a_missing_bar_withholds_holm(tmp_path: Path) -> None:
    rows = figure_claims.figure2(tmp_path, paired_tests.SidecarCache(tmp_path), _SETTINGS)
    mrr = [row for row in rows if row["metric"] == "mrr"]
    assert len(mrr) == 15 + 15 and all(not row["tested"] for row in mrr)
    assert all("No results for" in row["reason"] for row in mrr)


def test_pair_level_interval_needs_two_pairs_per_subgroup() -> None:
    assert pair_level_interval([0.3], [0.1, 0.2], _SETTINGS) is None
    assert figure_claims._pairs_ci(None) == "n/a (a subgroup has fewer than 2 pairs)"


def test_markdown_lists_every_metric_and_subgroup(cache: StubCache) -> None:
    text = render_markdown(figure1(cache, _SETTINGS), [])
    for label in ("| MRR | cmt_edas |", "| Hits@1 | other |", "| F1 | all |", "cmt_edas − other"):
        assert label in text
    assert np.isfinite(figure1(cache, _SETTINGS)["f1"]["difference"]["delta"])


def test_subgroup_bootstraps_are_independent() -> None:
    same = StubCache({"cmt-edas": (0.4, 0.6), "confof-ekaw": (0.4, 0.6)})
    arrays = column_arrays(same, _FIG1)["mrr"]
    low, high = subgroup_difference(arrays, "mrr", _SETTINGS)["ci_sources"]
    assert low < 0 < high


def test_empty_subgroup_is_refused(cache: StubCache) -> None:
    arrays = column_arrays(StubCache({"confof-ekaw": (0.4, 0.6)}), _FIG1)["mrr"]
    with pytest.raises(DataError, match="'cmt_edas' has no D3 pair"):
        subgroup_pairs(arrays, "cmt_edas")


def test_figure1_without_a_subgroup_is_not_tested() -> None:
    result = figure1(StubCache({"confof-ekaw": (0.4, 0.6)}), _SETTINGS)
    assert all(not block["tested"] and "has no D3 pair" in block["reason"] for block in result.values())
    row = next(line for line in render_markdown(result, []).splitlines() if "not tested" in line)
    assert row.count("|") == 9


def test_unmatched_references_make_a_figure2_comparison_untested() -> None:
    comparison = {"dataset_id": "D3", "condition_a": "C3", "condition_b": "C10"}
    rows = comparison_rows(StubCache({"cmt-edas": (0.2, 0.8)}, gold="u"), comparison, _SETTINGS)
    assert [row["metric"] for row in rows] == list(paired_tests.METRICS)
    assert all(not row["tested"] and "Test references differ" in row["reason"] for row in rows)
    assert all("p_holm" not in row for row in add_holm(rows))
    lines = render_markdown({}, add_holm(rows)).splitlines()
    row = next(line for line in lines if line.startswith("| D3 | MRR | C3 vs C10 | not tested: Test references differ"))
    header = next(line for line in lines if line.startswith("| Dataset |"))
    assert row.count("|") == header.count("|") == 10


def test_a_computation_error_is_not_reported_as_untested(cache: StubCache, monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise KeyError("ci_low")
    monkeypatch.setattr(figure_claims, "paired_test", broken)
    with pytest.raises(KeyError, match="ci_low"):
        figure1(cache, _SETTINGS)
    with pytest.raises(KeyError, match="ci_low"):
        comparison_rows(cache, _FIG1, _SETTINGS)


def test_figure2_conditions_refuse_a_result_without_strategy(tmp_path: Path) -> None:
    directory = tmp_path / "C18" / "D1"
    directory.mkdir(parents=True)
    (directory / "d1_snomed_fma_results.json").write_text("{}")
    with pytest.raises(ValueError, match="C18 result has strategy/model"):
        figure2_conditions(tmp_path, _D1)


def test_pair_delta_matches_paired_test(cache: StubCache) -> None:
    pair = column_arrays(cache, _FIG1)["f1"]["edas-ekaw"]
    assert pair_delta(pair, "f1") == pytest.approx(paired_tests.paired_test([pair], "f1", _SETTINGS)["delta"])


def test_importing_the_claims_does_not_load_matplotlib() -> None:
    code = "import sys; import scripts.figure_claims; print('matplotlib' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=_REPO_ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_untested_reason_cannot_break_the_table() -> None:
    row = {"dataset_id": "D1", "condition_a": "C18", "condition_b": "C10", "metric": "f1",
           "tested": False, "reason": "bad | path\nsecond line"}
    line = next(l for l in render_markdown({}, [row]).splitlines() if "not tested" in l)
    assert "bad \\| path second line" in line
    assert line.replace("\\|", "").count("|") == 10
