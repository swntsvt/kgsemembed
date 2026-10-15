"""Tests for budgeted triple selection in the controlled PPAS ablation (#122)."""

import copy

import pytest
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

import kgsemembed.verbalisation.selection as selection
from kgsemembed.verbalisation.ppas import (
    CLASS_TIER_LIST,
    INSTANCE_TIER_LIST,
    PREDICATE_TIER_LIST,
    ppas_sample,
    untiered_predicates,
)
from kgsemembed.verbalisation.selection import (
    DETERMINISTIC,
    FULL,
    ITEM_SEPARATOR,
    PPAS,
    RANDOM,
    EligibleItem,
    PolicySelector,
    deterministic_order,
    join_items,
    make_token_counter,
    ppas_order,
    random_order,
    select_within_budget,
    tier_config_digest,
)

EX = "http://example.org/"
ENTITY = URIRef(f"{EX}e")
UNTIERED_A = URIRef(f"{EX}aaa")
UNTIERED_B = URIRef(f"{EX}bbb")


def count_words(text: str) -> int:
    """Count whitespace words plus two special tokens, like [CLS] ... [SEP]."""
    return len(text.split()) + 2


def _class_graph() -> Graph:
    """A class whose untiered predicates sort before its tiered ones."""
    graph = Graph()
    graph.add((ENTITY, UNTIERED_A, Literal("alpha one two three")))
    graph.add((ENTITY, UNTIERED_B, Literal("beta one two three")))
    graph.add((ENTITY, RDFS.label, Literal("heart")))
    graph.add((ENTITY, RDFS.subClassOf, URIRef(f"{EX}Organ")))
    graph.add((URIRef(f"{EX}Organ"), RDFS.label, Literal("organ")))
    return graph


def _selector(policy: str, budget=None, seed=None) -> PolicySelector:
    return PolicySelector(policy, count_words, budget, seed)


def _items(graph: Graph = None) -> list[EligibleItem]:
    return _selector(FULL).eligible_items(graph or _class_graph(), ENTITY)


def _many_items(n: int) -> list[EligibleItem]:
    """Untiered items of varied length, so budgets bind differently."""
    return [EligibleItem(f"{EX}p{i:02d}", f"p{i:02d}: " + "w " * (i % 5 + 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# Eligible items
# ---------------------------------------------------------------------------


def test_eligible_items_are_canonically_ordered_rendered_pairs():
    texts = [item.text for item in _items()]
    assert texts == [
        "aaa: alpha one two three",
        "bbb: beta one two three",
        "label: heart",
        "sub Class Of: organ",
    ]


def test_eligible_items_exclude_same_as_blank_nodes_and_reflexive_triples():
    graph = Graph()
    graph.add((ENTITY, OWL.sameAs, URIRef(f"{EX}other")))
    graph.add((ENTITY, UNTIERED_A, BNode()))
    graph.add((ENTITY, UNTIERED_B, ENTITY))
    assert _items(graph) == []


def test_eligible_items_strip_openea_datatype_suffix():
    graph = Graph()
    graph.add((ENTITY, UNTIERED_A, Literal('"1955-03-02"^^<http://www.w3.org/2001/XMLSchema#date>')))
    assert [item.text for item in _items(graph)] == ["aaa: 1955-03-02"]


def test_eligible_items_keep_duplicate_renderings_once():
    graph = Graph()
    graph.add((ENTITY, UNTIERED_A, URIRef(f"{EX}x1")))
    graph.add((ENTITY, UNTIERED_A, URIRef(f"{EX}x2")))
    graph.add((URIRef(f"{EX}x1"), RDFS.label, Literal("same")))
    graph.add((URIRef(f"{EX}x2"), RDFS.label, Literal("same")))
    assert [item.text for item in _items(graph)] == ["aaa: same"]


def test_entity_without_eligible_items_has_empty_text_under_every_policy():
    for policy, budget, seed in ((FULL, None, None), (PPAS, 5, None), (RANDOM, 5, 0)):
        result = _selector(policy, budget, seed).select([], str(ENTITY), "class")
        assert (result.text, result.sampling_invoked, result.n_eligible) == ("", False, 0)


# ---------------------------------------------------------------------------
# Budget enforcement shared by every policy
# ---------------------------------------------------------------------------


def test_unsampled_entity_text_is_identical_under_every_policy():
    items = _items()
    budget = count_words(join_items([i.text for i in items], list(range(len(items)))))
    texts = {
        _selector(policy, budget if policy != FULL else None, 0 if policy == RANDOM else None)
        .select(items, str(ENTITY), "class").text
        for policy in (FULL, PPAS, DETERMINISTIC, RANDOM)
    }
    assert len(texts) == 1


@pytest.mark.parametrize("budget", [3, 6, 9, 12, 20, 33])
@pytest.mark.parametrize("policy,seed", [(PPAS, None), (DETERMINISTIC, None), (RANDOM, 0), (RANDOM, 7)])
def test_capped_policies_respect_the_budget_on_the_final_text(policy, seed, budget):
    items = _many_items(25)
    result = _selector(policy, budget, seed).select(items, str(ENTITY), "instance")
    assert result.sampling_invoked
    assert result.n_tokens == count_words(result.text) <= budget


def test_selected_indices_are_exactly_the_items_in_the_text():
    items = _many_items(25)
    result = _selector(RANDOM, 20, 3).select(items, str(ENTITY), "instance")
    assert result.text == ITEM_SEPARATOR.join(items[i].text for i in result.selected)
    assert list(result.selected) == sorted(result.selected)


def test_oversized_item_does_not_block_shorter_items():
    items = [EligibleItem(f"{EX}a", "a: " + "long " * 50), EligibleItem(f"{EX}b", "b: short")]
    result = _selector(DETERMINISTIC, 10).select(items, str(ENTITY), "instance")
    assert result.selected == (1,)
    assert result.n_oversized == 1


def test_item_that_does_not_fit_is_skipped_and_later_items_still_considered():
    texts = ["a: x", "b: " + "y " * 6, "c: z"]
    chosen = select_within_budget(texts, [0, 1, 2], 10, count_words)
    assert chosen == [0, 2]


def test_budget_smaller_than_any_item_selects_nothing():
    result = _selector(PPAS, 3).select(_items(), str(ENTITY), "class")
    assert (result.text, result.selected, result.n_tokens) == ("", (), 2)


# ---------------------------------------------------------------------------
# Policy orderings
# ---------------------------------------------------------------------------


def test_ppas_selects_by_predicate_tier():
    result = _selector(PPAS, 10).select(_items(), str(ENTITY), "class")
    assert result.text == "label: heart | sub Class Of: organ"


def test_deterministic_selects_lexicographically_not_by_tier():
    result = _selector(DETERMINISTIC, 10).select(_items(), str(ENTITY), "class")
    assert result.text == "aaa: alpha one two three | label: heart"


def test_deterministic_order_never_consults_the_tier_lists(monkeypatch):
    def refuse(_entity_type):
        raise AssertionError("deterministic policy consulted PPAS tiers")

    monkeypatch.setattr(selection, "tier_list_for", refuse)
    result = _selector(DETERMINISTIC, 10).select(_items(), str(ENTITY), "class")
    assert result.text == "aaa: alpha one two three | label: heart"
    assert deterministic_order(_items()) == [0, 1, 2, 3]


def test_ppas_order_matches_existing_ppas_sample_ordering():
    graph = _class_graph()
    triples = PolicySelector(FULL, count_words)._collect_triples(graph, ENTITY)
    augmented = CLASS_TIER_LIST + [untiered_predicates(triples, CLASS_TIER_LIST)]
    expected = [str(t[1]) for t in ppas_sample(triples, augmented, 10**6, lambda *t: "x")]
    items = _items(graph)
    assert [items[i].predicate for i in ppas_order(items, CLASS_TIER_LIST)] == expected


def test_ppas_order_keeps_canonical_order_within_a_tier():
    items = [
        EligibleItem(str(RDFS.subClassOf), "subClassOf: b"),
        EligibleItem(str(RDFS.label), "label: a"),
        EligibleItem(str(RDFS.label), "label: c"),
    ]
    assert ppas_order(items, CLASS_TIER_LIST) == [1, 2, 0]


def test_ppas_uses_the_tier_list_of_the_entity_type():
    items = [EligibleItem(str(RDFS.subClassOf), "subClassOf: x"), EligibleItem(str(RDF.type), "type: y")]
    assert ppas_order(items, CLASS_TIER_LIST) == [0, 1]
    assert ppas_order(items, INSTANCE_TIER_LIST) == [1, 0]


def test_selection_does_not_mutate_the_tier_lists():
    before = copy.deepcopy((CLASS_TIER_LIST, INSTANCE_TIER_LIST, PREDICATE_TIER_LIST))
    for entity_type in ("class", "instance", "predicate"):
        _selector(PPAS, 9).select(_items(), str(ENTITY), entity_type)
    assert (CLASS_TIER_LIST, INSTANCE_TIER_LIST, PREDICATE_TIER_LIST) == before


# ---------------------------------------------------------------------------
# Random policy
# ---------------------------------------------------------------------------


def test_random_order_is_reproducible_for_a_fixed_seed():
    assert random_order(30, 4, str(ENTITY)) == random_order(30, 4, str(ENTITY))


def test_random_order_varies_across_seeds_and_entities():
    orders = {tuple(random_order(30, seed, str(ENTITY))) for seed in range(5)}
    assert len(orders) == 5
    assert random_order(30, 0, f"{EX}x") != random_order(30, 0, f"{EX}y")


def test_random_selection_is_independent_of_visiting_order():
    items = _many_items(25)
    selector = _selector(RANDOM, 20, 2)
    first = selector.select(items, f"{EX}x", "instance")
    selector.select(items, f"{EX}y", "instance")
    assert selector.select(items, f"{EX}x", "instance") == first


def test_random_order_is_a_permutation():
    assert sorted(random_order(17, 3, str(ENTITY))) == list(range(17))


# ---------------------------------------------------------------------------
# Validation, provenance and tokenizer wrapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy,budget,seed",
    [("bogus", 5, None), (FULL, 5, None), (PPAS, None, None), (PPAS, 5, 1), (RANDOM, 5, None), (PPAS, 0, None)],
)
def test_invalid_policy_configuration_raises(policy, budget, seed):
    with pytest.raises(ValueError):
        PolicySelector(policy, count_words, budget, seed)


def test_unsupported_entity_type_raises_for_ppas():
    with pytest.raises(ValueError):
        _selector(PPAS, 9).select(_items(), str(ENTITY), "mixed")


def test_tier_config_digest_is_stable_and_tracks_the_tiers(monkeypatch):
    digest = tier_config_digest()
    assert digest == tier_config_digest()
    monkeypatch.setitem(selection._TIER_LISTS, "class", [[f"{EX}changed"]])
    assert tier_config_digest() != digest


def test_token_counter_counts_special_tokens_without_truncation():
    calls = []

    def fake_tokenizer(text, **kwargs):
        calls.append(kwargs)
        return {"input_ids": [101] + [7] * len(text.split()) + [102]}

    assert make_token_counter(fake_tokenizer)("a b c") == 5
    assert calls == [{"add_special_tokens": True, "truncation": False, "verbose": False}]


@pytest.mark.slow
def test_budget_holds_under_the_real_bge_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
    count = make_token_counter(tokenizer)
    items = [EligibleItem(f"{EX}p{i}", f"release Date: 19{i:02d}-03-02 | Q{i}x{i}") for i in range(60)]
    for policy, seed in ((PPAS, None), (DETERMINISTIC, None), (RANDOM, 1)):
        result = PolicySelector(policy, count, 80, seed).select(items, str(ENTITY), "instance")
        encoded = tokenizer(result.text)["input_ids"]
        assert len(encoded) == result.n_tokens <= 80
        assert encoded[0] == tokenizer.cls_token_id and encoded[-1] == tokenizer.sep_token_id
