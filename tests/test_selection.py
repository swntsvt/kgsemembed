"""Tests for budgeted triple selection in the controlled PPAS ablation (#122, #128)."""

import copy

import pytest
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from kgsemembed.verbalisation.predicate_config import (
    PredicateVocabulary,
    expand_predicate,
    get_predicate_config,
)

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


# ---------------------------------------------------------------------------
# Controlled policies P1-P7 (#128): filtering and ordering are independent
# ---------------------------------------------------------------------------

FMA = "http://purl.org/sig/ont/fma/"
HEART = URIRef(f"{EX}heart")
BGE_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"

CONTROLLED = {
    "P1": ("all", "base", DETERMINISTIC),
    "P2": ("all", "base", PPAS),
    "P3": ("covered", "base", DETERMINISTIC),
    "P4": ("covered", "base", PPAS),
    "P5": ("all", "fma_extended", PPAS),
    "P6": ("covered", "fma_extended", DETERMINISTIC),
    "P7": ("covered", "fma_extended", PPAS),
}


def _fma_graph() -> Graph:
    """An FMA-style class; canonical order puts every fma: item before rdf/rdfs ones.

    Items, canonically: 0 FMAID, 1 author, 2 preferred name, 3 synonym cardiac
    pump, 4 synonym cor, 5 type, 6 label, 7 subClassOf.
    """
    graph = Graph()
    for local, value in (("author", "curator one"), ("FMAID", "7088"), ("preferred_name", "heart"),
                         ("synonym", "cor"), ("synonym", "cardiac pump")):
        graph.add((HEART, URIRef(f"{FMA}{local}"), Literal(value)))
    graph.add((HEART, RDF.type, OWL.Class))
    graph.add((HEART, RDFS.label, Literal("Heart")))
    graph.add((HEART, RDFS.subClassOf, URIRef(f"{EX}Organ")))
    graph.add((URIRef(f"{EX}Organ"), RDFS.label, Literal("organ")))
    return graph


def _fma_items() -> list[EligibleItem]:
    return _selector(FULL).eligible_items(_fma_graph(), HEART)


def _controlled(label: str, budget: int, count=count_words, vocabulary=None) -> PolicySelector:
    filter_mode, name, order_mode = CONTROLLED[label]
    return PolicySelector(
        order_mode=order_mode, count_tokens=count, budget=budget, filter_mode=filter_mode,
        predicate_config=vocabulary or get_predicate_config(name),
    )


def _select(label: str, budget: int, items=None):
    return _controlled(label, budget).select(items or _fma_items(), str(HEART), "class")


def _texts(selection) -> set[str]:
    return set(selection.text.split(ITEM_SEPARATOR)) if selection.text else set()


def test_fma_fixture_items_are_in_the_documented_canonical_order():
    predicates = [item.predicate for item in _fma_items()]
    assert predicates[:5] == [f"{FMA}FMAID", f"{FMA}author", f"{FMA}preferred_name",
                              f"{FMA}synonym", f"{FMA}synonym"]
    assert predicates[5:] == [str(RDF.type), str(RDFS.label), str(RDFS.subClassOf)]


def test_p1_and_p3_order_identically_when_given_the_same_eligible_items():
    covered = [i for i in _fma_items() if i.predicate in get_predicate_config("base").covered_predicates]
    for budget in (5, 6, 7, 9, 12):
        assert _select("P1", budget, covered).text == _select("P3", budget, covered).text


def test_p3_and_p4_share_the_eligible_set_but_order_it_differently():
    p3, p4 = _select("P3", 6), _select("P4", 6)
    assert (p3.n_eligible, p3.n_filtered_items) == (p4.n_eligible, p4.n_filtered_items) == (3, 5)
    assert _texts(p3) == {"type: Class"}
    assert _texts(p4) == {"label: Heart"}


def test_p3_excludes_predicates_outside_the_covered_vocabulary():
    p3 = _select("P3", 512)
    assert not p3.sampling_invoked
    assert _texts(p3) == {"type: Class", "label: Heart", "sub Class Of: organ"}


def test_p4_selects_only_from_the_items_p3_keeps():
    eligible = _texts(_select("P3", 512))
    for budget in (4, 6, 7, 9):
        assert _texts(_select("P4", budget)) <= eligible


@pytest.mark.parametrize("graph_fn,entity", [(_class_graph, ENTITY), (_fma_graph, HEART)])
@pytest.mark.parametrize("label,policy", [("P1", DETERMINISTIC), ("P2", PPAS)])
def test_p1_and_p2_reproduce_the_legacy_policies(graph_fn, entity, label, policy):
    items = _selector(FULL).eligible_items(graph_fn(), entity)
    for budget in (4, 6, 8, 12, 20):
        legacy = _selector(policy, budget).select(items, str(entity), "class")
        controlled = _controlled(label, budget).select(items, str(entity), "class")
        assert controlled.text == legacy.text
        assert controlled.selected == legacy.selected


def test_p5_uses_its_own_tiers_and_differs_from_p2():
    p2, p5 = _select("P2", 12), _select("P5", 12)
    assert _texts(p2) == {"type: Class", "label: Heart", "sub Class Of: organ"}
    assert _texts(p5) == {"preferred name: heart", "synonym: cardiac pump", "synonym: cor"}


def test_p5_order_follows_the_extended_tiers_with_fmaid_after_every_tier():
    selector = _controlled("P5", 12)
    assert selector._order(_fma_items(), str(HEART), "class") == [2, 3, 4, 6, 5, 7, 0, 1]
    base = _controlled("P2", 12)
    assert base._order(_fma_items(), str(HEART), "class") == [6, 5, 7, 0, 1, 2, 3, 4]


def test_a_misnamespaced_fma_vocabulary_selects_exactly_like_p2():
    extended = get_predicate_config("fma_extended")
    wrong = "http://purl.org/oasis/names/tc/ciq/xal/3.0#"
    moved = {t: [[p.replace(FMA, wrong) for p in tier] for tier in tiers]
             for t, tiers in extended.tier_lists.items()}
    covered = frozenset(p.replace(FMA, wrong) for p in extended.covered_predicates)
    misnamed = PredicateVocabulary("misnamed", "", covered, moved)
    selection = _controlled("P5", 12, vocabulary=misnamed).select(_fma_items(), str(HEART), "class")
    assert selection.text == _select("P2", 12).text


def test_p6_keeps_fma_lexical_and_identifier_items_and_drops_the_rest():
    p6 = _select("P6", 512)
    assert p6.n_filtered_items == 1
    assert "author: curator one" not in _texts(p6)
    assert {"FMAID: 7088", "preferred name: heart", "synonym: cor"} <= _texts(p6)


def test_p6_and_p7_share_the_eligible_set_and_differ_in_order():
    p6, p7 = _select("P6", 12), _select("P7", 12)
    assert p6.n_eligible == p7.n_eligible == 7
    assert "FMAID: 7088" in _texts(p6)
    assert "FMAID: 7088" not in _texts(p7)


def test_fmaid_is_kept_by_the_covered_filter_but_visited_last():
    selector = _controlled("P7", 512)
    filtered = [i for i in _fma_items() if i.predicate != f"{FMA}author"]
    order = selector._order(filtered, str(HEART), "class")
    assert filtered[order[-1]].predicate == f"{FMA}FMAID"


def test_item_predicates_match_prefix_expanded_vocabulary_uris():
    predicates = {item.predicate for item in _fma_items()}
    assert expand_predicate("fma:synonym") in predicates
    assert expand_predicate("rdfs:subClassOf") in predicates


def test_equal_priority_ties_keep_canonical_order_and_oversized_items_are_skipped():
    items = _fma_items()
    long_synonym = EligibleItem(f"{FMA}synonym", "synonym: " + "very " * 20 + "long")
    with_oversized = sorted(items + [long_synonym], key=lambda i: (i.predicate, i.text))
    first = _controlled("P5", 9).select(with_oversized, str(HEART), "class")
    again = _controlled("P5", 9).select(with_oversized, str(HEART), "class")
    assert first == again
    assert first.n_oversized == 1
    assert _texts(first) == {"preferred name: heart", "synonym: cardiac pump"}


def test_controlled_selections_record_their_modes_and_coverage():
    p5, p6 = _select("P5", 12), _select("P6", 12)
    assert (p5.filter_mode, p5.order_mode, p5.predicate_config_name) == ("all", PPAS, "fma_extended")
    assert (p6.filter_mode, p6.order_mode, p6.predicate_config_name) == ("covered", DETERMINISTIC, "fma_extended")
    assert p5.n_covered_items == p6.n_covered_items == 7
    assert _select("P1", 12).n_covered_items == 3
    assert p5.n_tiered_items == 6 and _select("P2", 12).n_tiered_items == 3


def test_unknown_filter_mode_raises():
    with pytest.raises(ValueError):
        PolicySelector(order_mode=PPAS, count_tokens=count_words, budget=9, filter_mode="some")


@pytest.mark.slow
def test_controlled_policies_hold_the_budget_under_the_real_bge_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5", revision=BGE_REVISION)
    count = make_token_counter(tokenizer)
    extra = [EligibleItem(f"{FMA}synonym", f"synonym: cardiac structure variant {i}") for i in range(30)]
    items = sorted(_fma_items() + extra, key=lambda i: (i.predicate, i.text))
    for label in CONTROLLED:
        result = _controlled(label, 48, count).select(items, str(HEART), "class")
        encoded = tokenizer(result.text)["input_ids"]
        assert result.sampling_invoked == (label not in ("P3", "P4"))
        assert len(encoded) == result.n_tokens <= 48
        assert encoded[0] == tokenizer.cls_token_id and encoded[-1] == tokenizer.sep_token_id
