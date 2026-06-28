"""Tests for VerbaliserBase abstract base class and utility methods."""

from abc import ABC

import pytest
from rdflib import Graph, Literal, URIRef

from kgsemembed.verbalisation.base import (
    DEFINITION_PREDICATES,
    LABEL_PREDICATES,
    SYNONYM_PREDICATES,
    VerbaliserBase,
)


class _ConcreteVerbaliser(VerbaliserBase):
     """Minimal concrete subclass for testing (no-op verbalise)."""

     def verbalise(self, graph: Graph, entity_uri: URIRef, entity_type: str) -> str:
        return ""


# --- ABC enforcement ---


def test_verbaliser_base_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        VerbaliserBase()


def test_concrete_subclass_can_be_instantiated() -> None:
    v = _ConcreteVerbaliser()
    assert isinstance(v, VerbaliserBase)
    assert isinstance(v, ABC)


# --- get_label ---


def test_get_label_returns_english_over_french() -> None:
    g = Graph()
    ent = URIRef("http://example.org/ont#HeartDisease")
    g.add((ent, LABEL_PREDICATES[0], Literal("heart disease", lang="en")))
    g.add((ent, LABEL_PREDICATES[0], Literal("maladie cardiaque", lang="fr")))

    v = _ConcreteVerbaliser()
    assert v.get_label(g, ent) == "heart disease"


def test_get_label_returns_none_for_french_only() -> None:
    g = Graph()
    ent = URIRef("http://example.org/ont#Diabetes")
    g.add((ent, LABEL_PREDICATES[0], Literal("diabète", lang="fr")))

    v = _ConcreteVerbaliser()
    assert v.get_label(g, ent) is None


def test_get_label_returns_untagged_as_fallback() -> None:
    g = Graph()
    ent = URIRef("http://example.org/ont#Cancer")
    g.add((ent, LABEL_PREDICATES[0], Literal("cancer")))

    v = _ConcreteVerbaliser()
    assert v.get_label(g, ent) == "cancer"


def test_get_label_returns_none_for_unknown_entity() -> None:
    g = Graph()
    ent = URIRef("http://example.org/ont#Unknown")

    v = _ConcreteVerbaliser()
    assert v.get_label(g, ent) is None


def test_get_label_prefers_first_predicate_in_list() -> None:
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    g.add((ent, LABEL_PREDICATES[0], Literal("from rdfs:label")))
    g.add((ent, LABEL_PREDICATES[1], Literal("from skos:prefLabel")))

    v = _ConcreteVerbaliser()
    assert v.get_label(g, ent) == "from rdfs:label"


# --- get_local_name ---


def test_get_local_name_hash_fragment() -> None:
    v = _ConcreteVerbaliser()
    uri = URIRef("http://ex.org/ont#HeartDisease")
    assert v.get_local_name(uri) == "Heart Disease"


def test_get_local_name_slash_fragment() -> None:
    v = _ConcreteVerbaliser()
    uri = URIRef("http://ex.org/ont/HeartDisease")
    assert v.get_local_name(uri) == "Heart Disease"


def test_get_local_name_snake_case() -> None:
    v = _ConcreteVerbaliser()
    uri = URIRef("http://ex.org/ont#heart_disease")
    assert v.get_local_name(uri) == "heart disease"


def test_get_local_name_already_spaced() -> None:
    v = _ConcreteVerbaliser()
    uri = URIRef("http://ex.org/ont#Heart_Disease")
    assert v.get_local_name(uri) == "Heart Disease"


def test_get_local_name_simple() -> None:
    v = _ConcreteVerbaliser()
    uri = URIRef("http://ex.org/ont#Gene")
    assert v.get_local_name(uri) == "Gene"


def test_get_local_name_no_delimiter() -> None:
    v = _ConcreteVerbaliser()
    uri = URIRef("http://example.org/something")
    assert v.get_local_name(uri) == "something"


def test_get_local_name_empty_after_delimiter() -> None:
    v = _ConcreteVerbaliser()
    uri = URIRef("http://ex.org/ont#")
    assert v.get_local_name(uri) == ""


# --- get_label_or_local ---


def test_get_label_or_local_returns_label_when_available() -> None:
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    g.add((ent, LABEL_PREDICATES[0], Literal("test label")))

    v = _ConcreteVerbaliser()
    assert v.get_label_or_local(g, ent) == "test label"


def test_get_label_or_local_falls_back_to_local_name() -> None:
    g = Graph()
    ent = URIRef("http://example.org/ont#HeartDisease")

    v = _ConcreteVerbaliser()
    assert v.get_label_or_local(g, ent) == "Heart Disease"


def test_get_label_or_local_falls_back_for_empty_graph() -> None:
    g = Graph()
    ent = URIRef("http://example.org/ont#Unknown")

    v = _ConcreteVerbaliser()
    assert v.get_label_or_local(g, ent) == "Unknown"


# --- get_all_literals ---


def test_get_all_literals_returns_english_and_untagged() -> None:
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    g.add((ent, DEFINITION_PREDICATES[0], Literal("en definition")))
    g.add((ent, DEFINITION_PREDICATES[0], Literal("définition fr", lang="fr")))
    g.add((ent, DEFINITION_PREDICATES[0], Literal("untagged def")))

    v = _ConcreteVerbaliser()
    result = v.get_all_literals(g, ent, DEFINITION_PREDICATES)
    assert result == ["en definition", "untagged def"]


def test_get_all_literals_deduplicates_within_predicate() -> None:
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    pred = DEFINITION_PREDICATES[0]
    g.add((ent, pred, Literal("same", lang="en")))
    g.add((ent, pred, Literal("same", lang="en")))

    v = _ConcreteVerbaliser()
    result = v.get_all_literals(g, ent, [pred])
    assert result == ["same"]


def test_get_all_literals_deduplicates_across_predicates() -> None:
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    g.add((ent, DEFINITION_PREDICATES[0], Literal("shared", lang="en")))
    g.add((ent, DEFINITION_PREDICATES[1], Literal("shared", lang="en")))

    v = _ConcreteVerbaliser()
    result = v.get_all_literals(g, ent, DEFINITION_PREDICATES)
    assert result == ["shared"]


def test_get_all_literals_preserves_first_occurrence_order() -> None:
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    preds = [
        DEFINITION_PREDICATES[0],
        DEFINITION_PREDICATES[1],
    ]
    g.add((ent, preds[0], Literal("first")))
    g.add((ent, preds[1], Literal("second")))

    v = _ConcreteVerbaliser()
    result = v.get_all_literals(g, ent, preds)
    assert result == ["first", "second"]


def test_get_all_literals_empty_for_unknown_entity() -> None:
    g = Graph()
    ent = URIRef("http://example.org/ont#Unknown")

    v = _ConcreteVerbaliser()
    assert v.get_all_literals(g, ent, DEFINITION_PREDICATES) == []


def test_get_all_literals_filters_non_literal() -> None:
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    g.add((ent, DEFINITION_PREDICATES[0], URIRef("http://example.org/Other")))
    g.add((ent, DEFINITION_PREDICATES[0], Literal("real label")))

    v = _ConcreteVerbaliser()
    result = v.get_all_literals(g, ent, DEFINITION_PREDICATES)
    assert result == ["real label"]


# --- predicate constants ---


def test_label_predicates_are_urirefs() -> None:
    for p in LABEL_PREDICATES:
        assert isinstance(p, URIRef)


def test_definition_predicates_are_urirefs() -> None:
    for p in DEFINITION_PREDICATES:
        assert isinstance(p, URIRef)


def test_synonym_predicates_are_urirefs() -> None:
    for p in SYNONYM_PREDICATES:
        assert isinstance(p, URIRef)
