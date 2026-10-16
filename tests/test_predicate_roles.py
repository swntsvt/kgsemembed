"""Tests for the predicate role map, role filter and core-first order (#131)."""

import pytest

from kgsemembed.pipeline.ablation_conditions import (
    ABLATION_CONDITIONS,
    ROLE_ARMS,
    ROLE_MATRIX,
    role_condition_id,
)
from kgsemembed.verbalisation.predicate_config import expand_predicate
from kgsemembed.verbalisation.predicate_roles import (
    CORE,
    NON_CORE_ROLES,
    PREFERRED_NAME,
    SYNONYM,
    PredicateRoleMap,
    RoleFilter,
    UnmappedPredicateError,
    get_role_map,
)
from kgsemembed.verbalisation.selection import (
    CORE_FIRST,
    FULL,
    EligibleItem,
    PolicySelector,
    core_first_order,
    filter_by_roles,
)

ROLES_V1 = get_role_map("roles_v1")
LABEL = expand_predicate("rdfs:label")
TYPE = expand_predicate("rdf:type")
PREF = expand_predicate("fma:preferred_name")
SYN = expand_predicate("fma:synonym")
FMAID = expand_predicate("fma:FMAID")

D1_D2_PREDICATES = [
    "rdfs:label", "rdf:type", "rdfs:subClassOf", "rdfs:comment", "owl:disjointWith",
    "skos:prefLabel", "skos:altLabel", "skos:definition",
    "fma:FMAID", "fma:preferred_name", "fma:synonym", "fma:non-English_equivalent", "fma:definition",
    "fma:RadLex_ID", "fma:Talairach", "fma:AAL", "fma:DK__Freesurfer_", "fma:Neurolex", "fma:JHU_DTI-81",
    "fma:homonym_for", "fma:CMA_label", "fma:JHU_White-Matter_Tractography_Atlas", "fma:homonym_of",
    "fma:fma0329122", "fma:Eponym", "fma:slot_synonym",
    "oboInOwl:hasRelatedSynonym", "oboInOwl:hasDefinition", "oboInOwl:hasAlternativeId", "oboInOwl:hasDbXref",
]
"""Every predicate the #131 audit found in D1 or D2 (both sides)."""


def _items(*predicates: str) -> list[EligibleItem]:
    return [EligibleItem(p, f"{p.rsplit('/', 1)[-1]}: value {k}") for k, p in enumerate(predicates)]


def _words(text: str) -> int:
    return len(text.split()) + 2


# ---------------------------------------------------------------------------
# Role map
# ---------------------------------------------------------------------------


def test_roles_v1_maps_every_audited_predicate():
    predicates = {expand_predicate(name) for name in D1_D2_PREDICATES}
    assert ROLES_V1.unmapped(predicates) == []


@pytest.mark.parametrize("name,role", [
    ("rdfs:label", "core"), ("rdf:type", "core"), ("rdfs:subClassOf", "core"),
    ("skos:prefLabel", "preferred_name"), ("fma:preferred_name", "preferred_name"),
    ("skos:altLabel", "synonym"), ("fma:synonym", "synonym"), ("oboInOwl:hasRelatedSynonym", "synonym"),
    ("fma:definition", "definition"), ("oboInOwl:hasDefinition", "definition"), ("rdfs:comment", "definition"),
    ("fma:FMAID", "identifier"), ("oboInOwl:hasDbXref", "identifier"),
    ("fma:non-English_equivalent", "non_english"), ("owl:disjointWith", "disjointness"),
    ("fma:Eponym", "other"),
])
def test_roles_v1_assigns_the_audited_role(name, role):
    assert ROLES_V1.role_of(expand_predicate(name)) == role


def test_unmapped_predicates_raise_instead_of_defaulting():
    with pytest.raises(UnmappedPredicateError):
        ROLES_V1.role_of(expand_predicate("fma:author"))


def test_role_map_hash_is_stable_and_derived():
    assert ROLES_V1.config_hash == get_role_map("roles_v1").config_hash
    other = PredicateRoleMap("roles_v1", {**ROLES_V1.roles, FMAID: "other"})
    assert other.config_hash != ROLES_V1.config_hash
    with pytest.raises(TypeError):
        PredicateRoleMap("x", {}, "forged")


def test_role_map_rejects_unknown_roles_and_prefixed_names():
    with pytest.raises(ValueError):
        PredicateRoleMap("x", {LABEL: "lexical"})
    with pytest.raises(ValueError):
        PredicateRoleMap("x", {"rdfs:label": CORE})


def test_role_filter_never_removes_core():
    with pytest.raises(ValueError):
        RoleFilter(ROLES_V1, frozenset({CORE}))


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------


def test_filter_by_roles_removes_exactly_the_excluded_roles():
    items = _items(FMAID, LABEL, PREF, SYN, TYPE)
    kept, removed = filter_by_roles(items, RoleFilter(ROLES_V1, frozenset({SYNONYM, PREFERRED_NAME})))
    assert [i.predicate for i in kept] == [FMAID, LABEL, TYPE] and removed == 2
    assert filter_by_roles(items, None) == (items, 0)


def test_role_filter_raises_on_an_unmapped_predicate_even_when_kept():
    with pytest.raises(UnmappedPredicateError):
        filter_by_roles(_items(LABEL, expand_predicate("fma:author")), RoleFilter(ROLES_V1))


def test_core_first_visits_core_items_before_annotations_that_sort_earlier():
    items = _items(FMAID, PREF, SYN, TYPE, LABEL)
    assert core_first_order(items, ROLES_V1) == [3, 4, 0, 1, 2]


def test_core_first_keeps_the_label_that_lexicographic_order_drops():
    items = sorted(_items(PREF, SYN, SYN, FMAID, LABEL, TYPE), key=lambda i: (i.predicate, i.text))
    budget = 2 + 2 * 4
    first = PolicySelector(order_mode=CORE_FIRST, count_tokens=_words, budget=budget,
                           role_filter=RoleFilter(ROLES_V1)).select(items, "e", "class")
    lexical = PolicySelector(order_mode="deterministic", count_tokens=_words, budget=budget).select(items, "e", "class")
    assert "label" in first.text and "label" not in lexical.text


def test_canonical_rendering_puts_fma_annotations_before_the_label_whatever_the_order():
    items = sorted(_items(SYN, LABEL, TYPE), key=lambda i: (i.predicate, i.text))
    for order in (FULL, CORE_FIRST):
        selection = PolicySelector(order_mode=order, count_tokens=_words, budget=None if order == FULL else 99,
                                   role_filter=RoleFilter(ROLES_V1)).select(items, "e", "class")
        assert selection.text.index("synonym") < selection.text.index("label")


def test_core_first_requires_a_role_map():
    with pytest.raises(ValueError):
        PolicySelector(order_mode=CORE_FIRST, count_tokens=_words, budget=48)


def test_full_order_with_a_role_filter_keeps_every_remaining_item():
    items = _items(FMAID, LABEL, SYN, TYPE)
    selection = PolicySelector(order_mode=FULL, count_tokens=_words,
                               role_filter=RoleFilter(ROLES_V1, frozenset({SYNONYM}))).select(items, "e", "class")
    assert not selection.sampling_invoked and selection.n_role_removed_items == 1
    assert "synonym" not in selection.text and selection.text.count(" | ") == 2


def test_entities_without_the_removed_role_keep_identical_text():
    items = _items(LABEL, TYPE)
    plain = PolicySelector(order_mode=FULL, count_tokens=_words).select(items, "e", "class")
    removed = PolicySelector(order_mode=FULL, count_tokens=_words,
                             role_filter=RoleFilter(ROLES_V1, frozenset({SYNONYM}))).select(items, "e", "class")
    assert plain.text == removed.text


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _role_runs() -> set:
    return {(d, c.token_budget, c.role_arm) for c in ABLATION_CONDITIONS if c.is_role_ablation for d in c.datasets}


def test_registry_holds_exactly_the_38_planned_runs():
    planned = {(d, b, a) for d, plan in ROLE_MATRIX.items() for b, arms in plan.items() for a in arms}
    assert _role_runs() == planned and len(planned) == 38


def test_role_arms_remove_the_planned_roles():
    assert ROLE_ARMS["A0"] == ()
    assert ROLE_ARMS["R-syn"] == (SYNONYM,)
    assert set(ROLE_ARMS["A-core"]) == set(NON_CORE_ROLES)
    assert set(NON_CORE_ROLES) - set(ROLE_ARMS["C+syn"]) == {SYNONYM}


def test_role_conditions_use_full_without_budget_and_core_first_with_one():
    for condition in (c for c in ABLATION_CONDITIONS if c.is_role_ablation):
        expected = FULL if condition.token_budget is None else CORE_FIRST
        assert condition.policy == condition.order_mode == expected
        assert (condition.filter_mode, condition.predicate_config, condition.policy_label) == ("all", "base", "")
        assert condition.role_map == "roles_v1"
        assert condition.condition_id == role_condition_id(condition.role_arm, condition.token_budget)


def test_d2_runs_no_arm_for_roles_it_lacks():
    d2 = {a for b, arms in ROLE_MATRIX["D2"].items() for a in arms}
    assert not d2 & {"R-pref", "R-nonEn", "C+pref", "C+id"}
