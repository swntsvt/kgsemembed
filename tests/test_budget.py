"""Tests for token budgets counted on the encoder input (#183)."""

import random
from pathlib import Path

import pytest
from rdflib import Graph, Literal, URIRef

from kgsemembed.embeddings.models import MODEL_REGISTRY, get_model_config
from kgsemembed.verbalisation import build_verbaliser
from kgsemembed.verbalisation.budget import (
    BUDGET_ENFORCEMENT,
    BudgetedVerbaliser,
    TokenBudget,
    budget_enforcement_digest,
    budget_tokens,
    encoder_input_counters,
    fit_items,
    fit_prefix,
    join_parts,
    strategy_is_budgeted,
    token_budget,
)
from kgsemembed.verbalisation.ppas import PPAS_BUDGETS

_ENT = URIRef("http://example.org/ont#Entity")
_DATA = Path(__file__).resolve().parents[1] / "data"


def _words(text: str) -> int:
    return len(text.split())


def _budget(tokens: int) -> TokenBudget:
    return TokenBudget(tokens, _words, _words)


class _Fixed(BudgetedVerbaliser.__mro__[1]):
    def __init__(self, text: str) -> None:
        self.text = text

    def verbalise(self, graph: Graph, entity_uri: URIRef, entity_type: str) -> str:
        return self.text


def test_join_parts_drops_empty_parts_and_strips() -> None:
    assert join_parts(["", "  a ", "\n", "b"]) == "a\nb"


def test_fits_counts_the_text_after_preceding() -> None:
    budget = _budget(3)
    assert budget.fits("c", "a b")
    assert not budget.fits("c d", "a b")


@pytest.mark.parametrize("tokens, expected", [(3, "one two three"), (2, "one two"), (0, "")])
def test_cut_keeps_the_longest_fitting_word_prefix(tokens: int, expected: str) -> None:
    budget = _budget(tokens)
    assert budget.cut("one two three") == expected


def test_cut_counts_only_texts_it_cuts() -> None:
    budget = _budget(2)
    budget.cut("a b")
    budget.cut("a b c d")
    assert (budget.n_cut, budget.tokens_cut) == (1, 2)


def test_fit_items_keeps_everything_that_fits() -> None:
    assert fit_items(["a", "b c"], " | ", _budget(4)) == ["a", "b c"]


def test_fit_items_skips_an_item_that_does_not_fit_and_continues() -> None:
    assert fit_items(["a", "b c d e", "f"], " | ", _budget(3)) == ["a", "f"]


def test_fit_items_uses_what_preceding_left() -> None:
    assert fit_items(["a", "b"], " | ", _budget(2), "x y") == []
    assert fit_items(["a", "b"], " | ", _budget(3), "x y") == ["a"]
    assert fit_items(["a", "b"], " | ", _budget(4), "x") == ["a", "b"]


def test_fit_items_backs_off_when_the_joined_text_counts_more() -> None:
    def joined_costs_more(text: str) -> int:
        return _words(text) + text.count("|")

    budget = TokenBudget(5, joined_costs_more, _words)
    kept = fit_items(["a", "b", "c", "d"], " | ", budget)
    assert budget.count(" | ".join(kept)) <= 5
    assert kept == ["a", "b"]


def test_fit_items_without_a_budget_keeps_everything() -> None:
    assert fit_items(["a"] * 50, " ", None) == ["a"] * 50


def test_fit_prefix_stops_at_the_first_item_that_does_not_fit() -> None:
    assert fit_prefix(["a", "b c d", "e"], " | ", _budget(3)) == ["a"]


def test_budgeted_verbaliser_cuts_overflow_and_counts_it() -> None:
    budget = _budget(2)
    verbaliser = BudgetedVerbaliser(_Fixed("one two three"), budget)
    assert verbaliser.verbalise(Graph(), _ENT, "class") == "one two"
    assert budget.n_cut == 1


def test_budget_is_the_ppas_budget_capped_at_the_input_limit() -> None:
    for key in MODEL_REGISTRY:
        expected = None if PPAS_BUDGETS[key] is None else min(PPAS_BUDGETS[key], get_model_config(key).max_tokens)
        assert budget_tokens(key) == expected
    assert budget_tokens("M4") == 128


@pytest.mark.parametrize(
    "strategy, model_key, expected",
    [("V2+V6", "M2", True), ("V5", "M2", True), ("V2+V7", "M4", True), ("V2+V8", "M2", False),
     ("V1", "M4", False), ("V4+V6", "M3", False), ("V2+V6", "M2_uncapped", False)],
)
def test_strategy_is_budgeted(strategy: str, model_key: str, expected: bool) -> None:
    assert strategy_is_budgeted(strategy, model_key) is expected


def test_enforcement_digest_is_pinned() -> None:
    assert BUDGET_ENFORCEMENT["version"] == 1
    assert budget_enforcement_digest() == budget_enforcement_digest()
    assert len(budget_enforcement_digest()) == 64


def test_encoder_input_includes_special_tokens() -> None:
    count, pieces = encoder_input_counters("M2", "candidate")
    assert count("") == 2
    assert count("heart") == pieces("heart") + 2


def test_m5_source_input_includes_the_instruction_prefix() -> None:
    source, _ = encoder_input_counters("M5", "source")
    candidate, _ = encoder_input_counters("M5", "candidate")
    assert source("heart") - candidate("heart") >= 10
    assert get_model_config("M5").query_prefix.startswith("Instruct:")


def _long_entity(n: int) -> Graph:
    g = Graph()
    for i in range(n):
        g.add((_ENT, URIRef(f"http://example.org/ont#property{i}"), Literal(f"value {i} " + "lorem ipsum " * 8)))
    return g


@pytest.mark.parametrize("strategy", ["V3", "V4", "V6", "V2+V6", "V6+V3"])
@pytest.mark.parametrize("role", ["source", "candidate"])
def test_budgeted_strategies_hold_the_budget_exactly(strategy: str, role: str) -> None:
    verbaliser = build_verbaliser(strategy, "M2", role)
    text = verbaliser.verbalise(_long_entity(60), _ENT, "instance")
    assert verbaliser.budget.count(text) <= 420
    assert verbaliser.budget.count(text) > 300


def test_composite_fills_components_in_order() -> None:
    verbaliser = build_verbaliser("V6+V3", "M2", "candidate")
    text = verbaliser.verbalise(_long_entity(60), _ENT, "instance")
    assert "\n" not in text
    assert " | " in text


def _sample_entities(pair, n: int) -> list[str]:
    return random.Random(183).sample(sorted(pair.source_entities), n)


@pytest.mark.parametrize("dataset_id, column", [("D4", "D4_instance"), ("D5", "D5")])
def test_representative_inputs_hold_the_budget(dataset_id: str, column: str) -> None:
    if not (_DATA / {"D4": "d4_kgtrack", "D5": "d5_openea"}[dataset_id]).exists():
        pytest.skip(f"dataset {dataset_id} missing under data/")
    from kgsemembed.datasets import load_dataset

    pair = next(p for p in load_dataset(dataset_id, str(_DATA)) if p.dataset_id == column)
    for strategy in ("V3", "V4", "V2+V6", "V6+V3", "V5"):
        verbaliser = build_verbaliser(strategy, "M2", "source")
        for uri in _sample_entities(pair, 200):
            text = verbaliser.verbalise(pair.source_graph, URIRef(uri), "instance")
            assert verbaliser.budget.count(text) <= 420, (strategy, uri)
