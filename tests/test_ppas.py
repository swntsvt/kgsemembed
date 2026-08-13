"""Tests for Predicate-Priority Adaptive Sampling (PPAS)."""

from unittest.mock import patch

from rdflib import Graph, Literal, URIRef

from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.v3 import TemplateNLVerbaliser
from kgsemembed.verbalisation.ppas import (
    CLASS_TIER_LIST,
    INSTANCE_TIER_LIST,
    PREDICATE_TIER_LIST,
    PPAS_BUDGETS,
    PPAS_TRIGGER_THRESHOLD,
    ppas_sample,
    should_apply_ppas,
    estimate_tokens,
)


def _verbalise(s: URIRef, p: URIRef, o: URIRef) -> str:
    """Return a predictable string of 2 words from the triple."""
    word = str(p).split("#")[-1]
    return f"{word} {word}"


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_hello_world() -> None:
    assert estimate_tokens("hello world") == 3


def test_estimate_tokens_single_word() -> None:
    # ceil(1 * 1.3) = 2
    assert estimate_tokens("hello") == 2


def test_estimate_tokens_empty_string() -> None:
    assert estimate_tokens("") == 0


def test_estimate_tokens_many_words() -> None:
    # 10 words -> ceil(10 * 1.3) = 13
    text = " ".join(["word"] * 10)
    assert estimate_tokens(text) == 13


def test_estimate_tokens_fractions_round_up() -> None:
    # 3 words -> ceil(3 * 1.3) = ceil(3.9) = 4
    assert estimate_tokens("a b c") == 4


# ---------------------------------------------------------------------------
# should_apply_ppas
# ---------------------------------------------------------------------------


def test_should_apply_m3_always_false() -> None:
    triples = [(URIRef("s"), URIRef("p"), URIRef("o"))] * 100
    assert should_apply_ppas(triples, "M3") is False


def test_should_apply_unknown_model_key() -> None:
    triples = [(URIRef("s"), URIRef("p"), URIRef("o"))] * 100
    assert should_apply_ppas(triples, "M99") is False


def test_should_apply_below_threshold() -> None:
    triples = [(URIRef("s"), URIRef("p"), URIRef("o"))] * 20
    assert should_apply_ppas(triples, "M1") is False


def test_should_apply_above_threshold() -> None:
    triples = [(URIRef("s"), URIRef("p"), URIRef("o"))] * 21
    assert should_apply_ppas(triples, "M1") is True


def test_should_apply_empty_triples() -> None:
    assert should_apply_ppas([], "M1") is False


# ---------------------------------------------------------------------------
# ppas_sample -- tier priority
# ---------------------------------------------------------------------------


def test_ppas_selects_higher_tier_first() -> None:
    """Triples in tier 0 should be selected before tier 1 when budget is tight."""
    tier0_pred = URIRef("http://example.org/ont#label_pred")
    tier1_pred = URIRef("http://example.org/ont#type_pred")

    s = URIRef("http://example.org/entity")
    o = URIRef("http://example.org/ont#obj")

    triples = [
        (s, tier0_pred, o),    # tier 0 -- should be selected first
        (s, tier1_pred, o),    # tier 1 -- lower priority
    ]

    tier_list = [
        [str(tier0_pred)],     # tier 0
        [str(tier1_pred)],     # tier 1
    ]

    # Each triple costs 3 tokens (2 words -> ceil(2*1.3)=3). Budget of 3 fits exactly one.
    result = ppas_sample(triples, tier_list, token_budget=3, verbalise_triple_fn=_verbalise)
    assert len(result) == 1
    assert result[0] == (s, tier0_pred, o)


def test_ppas_selects_multiple_tiers_when_budget_allows() -> None:
    """When budget allows, triples from multiple tiers are selected in order."""
    tier0_pred = URIRef("http://example.org/ont#a_pred")
    tier1_pred = URIRef("http://example.org/ont#b_pred")
    tier2_pred = URIRef("http://example.org/ont#c_pred")

    s = URIRef("http://example.org/entity")
    o = URIRef("http://example.org/ont#obj")

    triples = [
        (s, tier0_pred, o),
        (s, tier1_pred, o),
        (s, tier2_pred, o),
    ]

    tier_list = [
        [str(tier0_pred)],
        [str(tier1_pred)],
        [str(tier2_pred)],
    ]

    # Each triple costs 3 tokens; budget of 9 should select all three
    result = ppas_sample(triples, tier_list, token_budget=9, verbalise_triple_fn=_verbalise)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# ppas_sample -- oversized triple does NOT block others (CRITICAL RULE)
# ---------------------------------------------------------------------------


def test_oversized_triple_skipped_in_same_tier() -> None:
    """A triple that exceeds remaining budget is skipped; subsequent triples in the same tier are still considered."""
    s = URIRef("http://example.org/entity")
    small_pred = URIRef("http://example.org/ont#small_pred")
    big_pred = URIRef("http://example.org/ont#big_pred")
    o = URIRef("http://example.org/ont#obj")

    def _big_verbalise(_s: URIRef, p: URIRef, _o: URIRef) -> str:
        if "big" in str(p):
            return " ".join(["w"] * 10)    # 10 words -> 13 tokens
        return "x x"    # 2 words -> 3 tokens

    triples = [
        (s, big_pred, o),        # tier 0, 13 tokens -- too big for budget of 5
        (s, small_pred, o),      # tier 0, 3 tokens -- fits
    ]

    tier_list = [
        [str(big_pred), str(small_pred)],    # same tier
    ]

    result = ppas_sample(triples, tier_list, token_budget=5, verbalise_triple_fn=_big_verbalise)
    # big_pred triple is skipped, small_pred is selected
    assert len(result) == 1
    assert result[0] == (s, small_pred, o)


def test_oversized_triple_does_not_block_later_tiers() -> None:
    """When an oversized triple is in tier 0, tier 1 triples should still be considered."""
    s = URIRef("http://example.org/entity")
    small_pred = URIRef("http://example.org/ont#small_pred")
    big_pred = URIRef("http://example.org/ont#big_pred")
    o = URIRef("http://example.org/ont#obj")

    def _big_verbalise(_s: URIRef, p: URIRef, _o: URIRef) -> str:
        if "big" in str(p):
            return " ".join(["w"] * 10)    # 13 tokens
        return "x x"    # 3 tokens

    triples = [
        (s, big_pred, o),        # tier 0, too big
        (s, small_pred, o),      # tier 1, fits
    ]

    tier_list = [
        [str(big_pred)],     # tier 0
        [str(small_pred)],     # tier 1
    ]

    result = ppas_sample(triples, tier_list, token_budget=5, verbalise_triple_fn=_big_verbalise)
    assert len(result) == 1
    assert result[0] == (s, small_pred, o)


# ---------------------------------------------------------------------------
# ppas_sample -- budget never exceeded
# ---------------------------------------------------------------------------


def test_output_never_exceeds_budget() -> None:
    """Sum of estimate_tokens on output must never exceed token_budget."""
    preds = [URIRef(f"http://example.org/ont#pred_{i}") for i in range(10)]
    s = URIRef("http://example.org/entity")
    o = URIRef("http://example.org/ont#obj")

    triples = [(s, p, o) for p in preds]
    tier_list = [[str(p) for p in preds]]

    budget = 30
    result = ppas_sample(triples, tier_list, token_budget=budget, verbalise_triple_fn=_verbalise)

    total_tokens = sum(estimate_tokens(_verbalise(*t)) for t in result)
    assert total_tokens <= budget


def test_output_never_exceeds_budget_unpredictable_costs() -> None:
    """With varying costs, the total must still stay within budget."""

    def _varying_verbalise(_s: URIRef, p: URIRef, _o: URIRef) -> str:
        i = int(str(p).split("_")[-1]) % 5
        counts = [1, 5, 2, 10, 1]
        return " ".join(["w"] * counts[i])

    preds = [URIRef(f"http://example.org/ont#pred_{i}") for i in range(5)]
    s = URIRef("http://example.org/entity")
    o = URIRef("http://example.org/ont#obj")

    triples = [(s, p, o) for p in preds]
    tier_list = [[str(p) for p in preds]]

    budget = 15
    result = ppas_sample(triples, tier_list, token_budget=budget, verbalise_triple_fn=_varying_verbalise)

    total_tokens = sum(estimate_tokens(_varying_verbalise(*t)) for t in result)
    assert total_tokens <= budget


# ---------------------------------------------------------------------------
# ppas_sample -- edge cases
# ---------------------------------------------------------------------------


def test_ppas_empty_triples() -> None:
    result = ppas_sample([], [[str(URIRef("http://ex.org/#p"))]], token_budget=10, verbalise_triple_fn=_verbalise)
    assert result == []


def test_ppas_budget_zero() -> None:
    pred = URIRef("http://example.org/ont#pred")
    s = URIRef("http://example.org/entity")
    o = URIRef("http://example.org/ont#obj")
    triples = [(s, pred, o)]
    tier_list = [[str(pred)]]
    result = ppas_sample(triples, tier_list, token_budget=0, verbalise_triple_fn=_verbalise)
    assert result == []


def test_ppas_negative_budget() -> None:
    pred = URIRef("http://example.org/ont#pred")
    s = URIRef("http://example.org/entity")
    o = URIRef("http://example.org/ont#obj")
    triples = [(s, pred, o)]
    tier_list = [[str(pred)]]
    result = ppas_sample(triples, tier_list, token_budget=-5, verbalise_triple_fn=_verbalise)
    assert result == []


def test_ppas_all_triples_exceed_budget() -> None:
    def _big_verbalise(_s: URIRef, p: URIRef, _o: URIRef) -> str:
        return " ".join(["w"] * 10)    # 13 tokens

    pred = URIRef("http://example.org/ont#pred")
    s = URIRef("http://example.org/entity")
    o = URIRef("http://example.org/ont#obj")
    triples = [(s, pred, o)]
    tier_list = [[str(pred)]]
    # Budget is 5, each triple costs 13 -> nothing fits
    result = ppas_sample(triples, tier_list, token_budget=5, verbalise_triple_fn=_big_verbalise)
    assert result == []


def test_ppas_empty_tier_list() -> None:
    pred = URIRef("http://example.org/ont#pred")
    s = URIRef("http://example.org/entity")
    o = URIRef("http://example.org/ont#obj")
    triples = [(s, pred, o)]
    result = ppas_sample(triples, [], token_budget=10, verbalise_triple_fn=_verbalise)
    assert result == []


def test_ppas_empty_inner_tiers_handled() -> None:
    """Empty inner tier lists should be skipped without error."""
    pred = URIRef("http://example.org/ont#pred")
    s = URIRef("http://example.org/entity")
    o = URIRef("http://example.org/ont#obj")
    triples = [(s, pred, o)]
    tier_list = [[], [str(pred)], []]
    result = ppas_sample(triples, tier_list, token_budget=10, verbalise_triple_fn=_verbalise)
    assert len(result) == 1


def test_ppas_budget_exhausted_after_complete_tier() -> None:
    """When budget is exactly exhausted after a tier, the next tier is not processed."""
    pred0 = URIRef("http://example.org/ont#a_pred")
    pred1 = URIRef("http://example.org/ont#b_pred")

    s = URIRef("http://example.org/entity")
    o = URIRef("http://example.org/ont#obj")

    triples = [
        (s, pred0, o),
        (s, pred1, o),
    ]

    tier_list = [
        [str(pred0)],      # tier 0 -- 2 words = 3 tokens
        [str(pred1)],      # tier 1 -- 2 words = 3 tokens
    ]

    # Budget of 3: tier 0 uses it, tier 1 should be skipped
    result = ppas_sample(triples, tier_list, token_budget=3, verbalise_triple_fn=_verbalise)
    assert len(result) == 1
    assert result[0] == (s, pred0, o)


def test_ppas_selects_all_when_budget_sufficient() -> None:
    """When budget is large enough, all triples are selected."""
    preds = [URIRef(f"http://example.org/ont#pred_{i}") for i in range(5)]
    s = URIRef("http://example.org/entity")
    o = URIRef("http://example.org/ont#obj")

    triples = [(s, p, o) for p in preds]
    tier_list = [[str(p) for p in preds]]

    # Each triple costs 3 tokens; 5 triples = 15 tokens. Budget of 100 is plenty.
    result = ppas_sample(triples, tier_list, token_budget=100, verbalise_triple_fn=_verbalise)
    assert len(result) == 5


def test_ppas_ignores_predicates_not_in_any_tier() -> None:
    """Predicates not listed in any tier are silently skipped."""
    in_tier_pred = URIRef("http://example.org/ont#in_tier")
    not_in_tier_pred = URIRef("http://example.org/ont#orphan_pred")

    s = URIRef("http://example.org/entity")
    o = URIRef("http://example.org/ont#obj")

    triples = [
        (s, in_tier_pred, o),
        (s, not_in_tier_pred, o),
    ]

    tier_list = [[str(in_tier_pred)]]

    result = ppas_sample(triples, tier_list, token_budget=100, verbalise_triple_fn=_verbalise)
    assert len(result) == 1
    assert result[0] == (s, in_tier_pred, o)


# ---------------------------------------------------------------------------
# Tier list constants
# ---------------------------------------------------------------------------


def test_class_tier_list_structure() -> None:
    assert isinstance(CLASS_TIER_LIST, list)
    assert all(isinstance(tier, list) for tier in CLASS_TIER_LIST)
    # Tier 0 should have lexical predicates
    assert len(CLASS_TIER_LIST[0]) > 0
    # Tiers 3-5 should be empty (populated dynamically)
    assert CLASS_TIER_LIST[3] == []
    assert CLASS_TIER_LIST[4] == []
    assert CLASS_TIER_LIST[5] == []


def test_instance_tier_list_structure() -> None:
    assert isinstance(INSTANCE_TIER_LIST, list)
    assert all(isinstance(tier, list) for tier in INSTANCE_TIER_LIST)
    assert len(INSTANCE_TIER_LIST[0]) > 0
    assert INSTANCE_TIER_LIST[1] == []
    assert INSTANCE_TIER_LIST[2] == []


def test_predicate_tier_list_structure() -> None:
    assert isinstance(PREDICATE_TIER_LIST, list)
    assert all(isinstance(tier, list) for tier in PREDICATE_TIER_LIST)
    assert len(PREDICATE_TIER_LIST[0]) > 0


def test_ppas_budgets_contains_all_model_keys() -> None:
    for key in ("M1", "M2", "M3", "M4", "M5"):
        assert key in PPAS_BUDGETS


def test_ppas_trigger_threshold_is_positive_int() -> None:
    assert isinstance(PPAS_TRIGGER_THRESHOLD, int)
    assert PPAS_TRIGGER_THRESHOLD > 0


# ---------------------------------------------------------------------------
# Deterministic triple ordering feeding PPAS
# ---------------------------------------------------------------------------


class _OrderingVerbaliser(VerbaliserBase):
    """Concrete stub exposing the inherited triple collector."""

    def verbalise(self, graph: Graph, entity_uri: URIRef, entity_type: str) -> str:
        return ""


_ENTITY = URIRef("http://ex/entity")


def _unordered_graph() -> Graph:
    graph = Graph()
    for index in (7, 3, 11, 0, 5):
        graph.add((_ENTITY, URIRef(f"http://ex/p{index}"), Literal(f"v{index}")))
    return graph


def test_collected_triples_are_sorted_by_spo() -> None:
    triples = _OrderingVerbaliser()._collect_triples(_unordered_graph(), _ENTITY)
    keys = [(str(s), str(p), str(o)) for s, p, o in triples]
    assert keys == sorted(keys)


def test_collected_triples_ordering_is_stable_across_calls() -> None:
    graph = _unordered_graph()
    verbaliser = _OrderingVerbaliser()
    first = verbaliser._collect_triples(graph, _ENTITY)
    second = verbaliser._collect_triples(graph, _ENTITY)
    assert first == second


def test_collected_triples_ordering_independent_of_insertion_order() -> None:
    forward = Graph()
    reverse = Graph()
    predicates = [URIRef(f"http://ex/p{i}") for i in (7, 3, 11, 0, 5)]
    for pred in predicates:
        forward.add((_ENTITY, pred, Literal("v")))
    for pred in reversed(predicates):
        reverse.add((_ENTITY, pred, Literal("v")))

    verbaliser = _OrderingVerbaliser()
    assert verbaliser._collect_triples(forward, _ENTITY) == verbaliser._collect_triples(
        reverse, _ENTITY
    )


def test_ppas_selection_is_stable_for_sorted_triples() -> None:
    triples = _OrderingVerbaliser()._collect_triples(_unordered_graph(), _ENTITY)
    tier_list = [[f"http://ex/p{i}" for i in (0, 3, 5, 7, 11)]]
    first = ppas_sample(triples, tier_list, 100, _verbalise)
    second = ppas_sample(triples, tier_list, 100, _verbalise)
    assert first == second


def test_ppas_keeps_highest_priority_triples_under_tight_budget() -> None:
    triples = _OrderingVerbaliser()._collect_triples(_unordered_graph(), _ENTITY)
    tier_list = [["http://ex/p0"], [f"http://ex/p{i}" for i in (3, 5, 7, 11)]]
    selected = ppas_sample(triples, tier_list, 3, _verbalise)
    assert [str(p) for _, p, _ in selected] == ["http://ex/p0"]


def test_verbaliser_passes_sorted_triples_to_ppas() -> None:
    """The sorted collection, not raw graph order, is what PPAS receives."""
    graph = _unordered_graph()
    graph.add((_ENTITY, URIRef("http://www.w3.org/2000/01/rdf-schema#label"), Literal("Ent")))
    captured: list[list] = []

    def _capture(triples, tier_list, budget, fn):
        captured.append(list(triples))
        return triples

    with patch("kgsemembed.verbalisation.v3.ppas_sample", side_effect=_capture):
        TemplateNLVerbaliser(model_key="M2").verbalise(graph, _ENTITY, "instance")

    keys = [(str(s), str(p), str(o)) for s, p, o in captured[0]]
    assert keys == sorted(keys)
