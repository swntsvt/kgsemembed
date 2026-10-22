"""Tests for the paired intervals behind Figures 1 and 2 (#149).

A stub cache stands in for the sidecars, so the subgroup, per-pair and
pairwise logic is exercised without result files.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kgsemembed.evaluation import paired_tests  # noqa: E402
from kgsemembed.utils.errors import DataError  # noqa: E402
from scripts.figure_claims import (  # noqa: E402
    _direction,
    add_holm,
    column_arrays,
    figure1,
    figure2_conditions,
    pair_delta,
    pair_level_interval,
    subgroup_difference,
    render_markdown,
    subgroup_pairs,
)

_SETTINGS = paired_tests.TestSettings(n_replicates=200)
_SOURCES = [f"s{i}" for i in range(20)]


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
    arrays = column_arrays(cache, ("C3", "C10"), "D3", "mrr")
    assert set(subgroup_pairs(arrays, "cmt_edas")) == {"cmt-edas", "edas-ekaw"}
    assert set(subgroup_pairs(arrays, "other")) == {"confof-ekaw", "ekaw-iasted"}
    assert len(subgroup_pairs(arrays, "all")) == 4


def test_pair_delta_is_b_minus_a(cache: StubCache) -> None:
    arrays = column_arrays(cache, ("C3", "C10"), "D3", "mrr")
    assert pair_delta(arrays["cmt-edas"], "mrr") == pytest.approx(0.6 * 0.75)


def test_different_references_are_refused() -> None:
    with pytest.raises(DataError, match="Test references differ"):
        column_arrays(StubCache({"cmt-edas": (0.2, 0.8)}, gold="u"), ("C3", "C10"), "D3", "mrr")


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
    rows = add_holm([{"metric": "mrr", "p_value": 0.01}, {"metric": "mrr", "p_value": 0.04},
                     {"metric": "f1", "p_value": 0.01}])
    assert [row.get("p_holm") for row in rows] == [pytest.approx(0.02), pytest.approx(0.04), None]


def test_figure2_conditions_follow_bar_order_and_skip_missing(tmp_path: Path) -> None:
    for condition in ("C17", "C10", "C18"):
        directory = tmp_path / condition / "D2"
        directory.mkdir(parents=True)
        (directory / "d2_anatomy_results.json").write_text("{}")
    assert figure2_conditions(tmp_path, "D2") == ["C18", "C10", "C17"]


def test_markdown_lists_every_metric_and_subgroup(cache: StubCache) -> None:
    text = render_markdown(figure1(cache, _SETTINGS), [])
    for label in ("| MRR | cmt_edas |", "| Hits@1 | other |", "| F1 | all |", "cmt_edas − other"):
        assert label in text
    assert np.isfinite(figure1(cache, _SETTINGS)["f1"]["difference"]["delta"])


def test_subgroup_bootstraps_are_independent() -> None:
    same = StubCache({"cmt-edas": (0.4, 0.6), "confof-ekaw": (0.4, 0.6)})
    arrays = column_arrays(same, ("C3", "C10"), "D3", "mrr")
    low, high = subgroup_difference(arrays, "mrr", _SETTINGS)["ci_sources"]
    assert low < 0 < high
