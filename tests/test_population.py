"""Tests for the validation/test evaluation populations."""

import zlib

import pytest
from rdflib import Graph

from kgsemembed.datasets import AlignmentPair
from kgsemembed.evaluation.population import (
    build_populations,
    draws_validation,
    restrict_ranked_lists,
    validation_share,
)


def _pair(source_entities, val_refs=(), test_refs=(), train_refs=()) -> AlignmentPair:
    return AlignmentPair(
        dataset_id="D1",
        pair_name="pair",
        source_graph=Graph(),
        target_graph=Graph(),
        source_entities=list(source_entities),
        val_refs=list(val_refs),
        test_refs=list(test_refs),
        train_refs=list(train_refs),
    )


def _unmatched(count: int) -> list:
    return [f"http://example.org/u{index}" for index in range(count)]


def test_reference_sources_go_to_their_own_slice() -> None:
    populations = build_populations(
        _pair(["v", "t"], val_refs=[("v", "x")], test_refs=[("t", "y")])
    )
    assert populations.val_sources == {"v"}
    assert populations.test_sources == {"t"}


def test_source_with_val_and_test_references_is_scored_on_test_only() -> None:
    populations = build_populations(
        _pair(["s"], val_refs=[("s", "x")], test_refs=[("s", "y")])
    )
    assert populations.test_sources == {"s"}
    assert "s" not in populations.val_sources


def test_training_sources_are_excluded_from_both_populations() -> None:
    populations = build_populations(
        _pair(["r", "v", "t"], val_refs=[("v", "x")], test_refs=[("t", "y")],
              train_refs=[("r", "z")])
    )
    assert "r" not in populations.val_sources | populations.test_sources


def test_every_unmatched_source_lands_in_exactly_one_population() -> None:
    unmatched = _unmatched(200)
    populations = build_populations(
        _pair(["v", "t", *unmatched], val_refs=[("v", "x")], test_refs=[("t", "y")])
    )
    assert populations.val_sources.isdisjoint(populations.test_sources)
    assert populations.val_sources | populations.test_sources == {"v", "t", *unmatched}
    assert populations.n_unmatched_val + populations.n_unmatched_test == 200


def test_unmatched_sources_follow_the_reference_source_ratio() -> None:
    val_refs = [(f"v{index}", "x") for index in range(20)]
    test_refs = [(f"t{index}", "y") for index in range(80)]
    unmatched = _unmatched(5000)
    sources = [s for s, _ in val_refs + test_refs] + unmatched
    populations = build_populations(_pair(sources, val_refs, test_refs))
    assert populations.n_unmatched_val / 5000 == pytest.approx(0.2, abs=0.02)


def test_assignment_is_independent_of_source_order() -> None:
    unmatched = _unmatched(100)
    forward = build_populations(_pair(unmatched, [("v", "x")], [("t", "y")]))
    backward = build_populations(_pair(unmatched[::-1], [("v", "x")], [("t", "y")]))
    assert forward == backward


def test_assignment_is_stable_crc32_rather_than_salted_hash() -> None:
    uri = "http://example.org/u7"
    expected = zlib.crc32(uri.encode("utf-8")) / 2**32 < 0.5
    assert draws_validation(uri, 0.5) is expected


@pytest.mark.parametrize("share,expected", [(0.0, False), (1.0, True)])
def test_extreme_shares_send_every_source_one_way(share: float, expected: bool) -> None:
    assert all(draws_validation(uri, share) is expected for uri in _unmatched(50))


def test_without_validation_references_unmatched_sources_all_go_to_test() -> None:
    populations = build_populations(_pair(_unmatched(30), test_refs=[("t", "y")]))
    assert populations.n_unmatched_val == 0
    assert populations.n_unmatched_test == 30


@pytest.mark.parametrize("n_val,n_test,expected", [(1, 4, 0.2), (0, 3, 0.0), (0, 0, 0.0)])
def test_validation_share(n_val: int, n_test: int, expected: float) -> None:
    assert validation_share(n_val, n_test) == expected


def test_restrict_ranked_lists_keeps_population_order_and_drops_empty_lists() -> None:
    ranked = [[("a", "x", 0.9)], [], [("b", "y", 0.8)], [("c", "z", 0.7)]]
    assert restrict_ranked_lists(ranked, frozenset({"c", "a"})) == [
        [("a", "x", 0.9)],
        [("c", "z", 0.7)],
    ]
