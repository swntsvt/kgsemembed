"""Tests for TemplateNLVerbaliser (V3)."""

import inspect

import pytest
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from kgsemembed.verbalisation import v3 as v3_module
from kgsemembed.verbalisation.ppas import PPAS_BUDGETS
from kgsemembed.verbalisation.v3 import TemplateNLVerbaliser

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_ENT = URIRef("http://example.org/ont#Heart")
_CLASS = URIRef("http://example.org/ont#Heart")
_PRED = URIRef("http://example.org/ont#authored_by")


def _label(g: Graph, uri: URIRef, text: str) -> None:
    g.add((uri, RDFS.label, Literal(text, lang="en")))


def _v(model_key: str = "M2") -> TemplateNLVerbaliser:
    return TemplateNLVerbaliser(model_key=model_key)


# ---------------------------------------------------------------------------
# Template coverage -- one independent test per supported template
# ---------------------------------------------------------------------------


def test_subclassof_template() -> None:
    g = Graph()
    organ = URIRef("http://example.org/ont#Organ")
    g.add((_CLASS, RDFS.subClassOf, organ))
    _label(g, organ, "Organ")

    assert _v().verbalise(g, _CLASS, "class") == "Heart is a subclass of Organ."


def test_type_template() -> None:
    g = Graph()
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Organ")))

    assert _v().verbalise(g, _ENT, "instance") == "Heart is a Organ."


def test_equivalent_class_template() -> None:
    g = Graph()
    other = URIRef("http://example.org/ont#CardiacMuscle")
    g.add((_CLASS, OWL.equivalentClass, other))
    _label(g, other, "Cardiac Muscle")

    assert _v().verbalise(g, _CLASS, "class") == "Heart is equivalent to Cardiac Muscle."


def test_disjoint_with_template() -> None:
    g = Graph()
    other = URIRef("http://example.org/ont#Bone")
    g.add((_CLASS, OWL.disjointWith, other))
    _label(g, other, "Bone")

    assert _v().verbalise(g, _CLASS, "class") == "Heart is disjoint from Bone."


def test_domain_template() -> None:
    g = Graph()
    person = URIRef("http://example.org/ont#Person")
    g.add((_PRED, RDFS.domain, person))
    _label(g, person, "Person")

    assert _v().verbalise(g, _PRED, "predicate") == "The domain of authored by is Person."


def test_range_template() -> None:
    g = Graph()
    doc = URIRef("http://example.org/ont#Document")
    g.add((_PRED, RDFS.range, doc))
    _label(g, doc, "Document")

    assert _v().verbalise(g, _PRED, "predicate") == "The range of authored by is Document."


def test_label_template() -> None:
    g = Graph()
    g.add((_ENT, RDFS.label, Literal("Heart", lang="en")))

    assert _v().verbalise(g, _ENT, "class") == "Heart is called Heart."


def test_comment_template() -> None:
    g = Graph()
    g.add((_CLASS, RDFS.comment, Literal("A hollow muscular organ", lang="en")))

    assert _v().verbalise(g, _CLASS, "class") == "Heart: A hollow muscular organ."


def test_inverse_of_template() -> None:
    g = Graph()
    other = URIRef("http://example.org/ont#authorOf")
    g.add((_PRED, OWL.inverseOf, other))
    _label(g, other, "author of")

    assert _v().verbalise(g, _PRED, "predicate") == "authored by is the inverse of author of."


def test_subproperty_of_template() -> None:
    g = Graph()
    parent = URIRef("http://example.org/ont#relatedTo")
    g.add((_PRED, RDFS.subPropertyOf, parent))
    _label(g, parent, "related to")

    assert _v().verbalise(g, _PRED, "predicate") == "authored by is a sub-property of related to."


def test_datatype_property_template() -> None:
    """A literal-valued predicate without a dedicated template uses 'has'."""
    g = Graph()
    mass = URIRef("http://example.org/ont#mass")
    g.add((_ENT, mass, Literal("300")))

    assert _v().verbalise(g, _ENT, "instance") == "Heart has mass 300."


def test_fallback_template_for_unknown_uri_predicate() -> None:
    """A URI-valued predicate without a dedicated template uses the fallback."""
    g = Graph()
    part_of = URIRef("http://example.org/ont#partOf")
    body = URIRef("http://example.org/ont#Body")
    g.add((_ENT, part_of, body))
    _label(g, body, "Body")

    assert _v().verbalise(g, _ENT, "instance") == "Heart part Of Body."


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------


def test_entity_without_label_uses_local_name() -> None:
    g = Graph()
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Organ")))

    assert _v().verbalise(g, _ENT, "instance") == "Heart is a Organ."


def test_uri_object_without_label_uses_local_name() -> None:
    g = Graph()
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#BodyPart")))

    assert _v().verbalise(g, _ENT, "instance") == "Heart is a Body Part."


def test_no_raw_uris_emitted() -> None:
    g = Graph()
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Organ")))

    assert "http://" not in _v().verbalise(g, _ENT, "instance")


# ---------------------------------------------------------------------------
# Entity-type validation
# ---------------------------------------------------------------------------


def test_unsupported_entity_type_raises_value_error() -> None:
    g = Graph()
    _label(g, _ENT, "Heart")

    with pytest.raises(ValueError, match="Unsupported entity type"):
        _v().verbalise(g, _ENT, "relationship")


def test_value_error_names_the_offending_type() -> None:
    g = Graph()

    with pytest.raises(ValueError, match="individual"):
        _v().verbalise(g, _ENT, "individual")


# ---------------------------------------------------------------------------
# PPAS integration
# ---------------------------------------------------------------------------


def test_correct_tier_list_selected_per_entity_type() -> None:
    """Each entity type resolves to its designated tier list."""
    from kgsemembed.verbalisation.ppas import (
        CLASS_TIER_LIST,
        INSTANCE_TIER_LIST,
        PREDICATE_TIER_LIST,
    )

    resolve = TemplateNLVerbaliser._tier_list_for
    assert resolve("class") is CLASS_TIER_LIST
    assert resolve("instance") is INSTANCE_TIER_LIST
    assert resolve("predicate") is PREDICATE_TIER_LIST


def test_ppas_applied_before_sentence_generation(monkeypatch) -> None:
    """ppas_sample is invoked with the correct tier list before rendering."""
    calls: list[tuple] = []
    real_ppas = v3_module.ppas_sample

    def spy(triples, tier_list, budget, fn):
        calls.append((tier_list, budget))
        return real_ppas(triples, tier_list, budget, fn)

    monkeypatch.setattr(v3_module, "ppas_sample", spy)

    g = Graph()
    _label(g, _ENT, "Heart")
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Organ")))

    _v().verbalise(g, _ENT, "instance")

    assert len(calls) == 1
    tier_list, budget = calls[0]
    # The instance tier list (with the appended untiered tier) drives selection.
    assert tier_list[0] == v3_module.INSTANCE_TIER_LIST[0]
    assert budget == PPAS_BUDGETS["M2"]


def test_rendering_uses_ppas_selection(monkeypatch) -> None:
    """Only triples returned by ppas_sample are rendered as sentences."""

    def only_type(triples, tier_list, budget, fn):
        return [triple for triple in triples if triple[1] == RDF.type]

    monkeypatch.setattr(v3_module, "ppas_sample", only_type)

    g = Graph()
    _label(g, _ENT, "Heart")
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Organ")))

    result = _v().verbalise(g, _ENT, "instance")

    assert result == "Heart is a Organ."
    assert "is called" not in result


def _many_literals(count: int) -> Graph:
    g = Graph()
    for i in range(count):
        g.add((_ENT, URIRef(f"http://example.org/ont#p{i}"), Literal(f"v{i}")))
    return g


def test_selected_triples_respect_budget(monkeypatch) -> None:
    """The triples ppas_sample returns fit within the model token budget."""
    from kgsemembed.verbalisation.ppas import estimate_tokens

    captured: list = []
    real_ppas = v3_module.ppas_sample

    def capture(triples, tier_list, budget, fn):
        selected = real_ppas(triples, tier_list, budget, fn)
        captured.append(selected)
        return selected

    monkeypatch.setattr(v3_module, "ppas_sample", capture)

    verbaliser = _v("M1")
    verbaliser.verbalise(_many_literals(200), _ENT, "instance")

    selected = captured[0]
    cost = sum(estimate_tokens(verbaliser._verbalise_triple(*t)) for t in selected)
    assert cost <= PPAS_BUDGETS["M1"]
    assert 0 < len(selected) < 200


def test_budgeted_model_respects_token_budget() -> None:
    g = _many_literals(200)
    result = _v("M1").verbalise(g, _ENT, "instance")
    sentences = result.split(". ")

    assert 0 < len(sentences) < 200


def test_m3_includes_all_selected_triples() -> None:
    assert PPAS_BUDGETS["M3"] is None
    g = _many_literals(200)

    result = _v("M3").verbalise(g, _ENT, "instance")

    assert result.count(".") == 200


def test_m3_output_exceeds_budgeted_output() -> None:
    g = _many_literals(200)
    m1 = _v("M1").verbalise(g, _ENT, "instance")
    m3 = _v("M3").verbalise(g, _ENT, "instance")

    assert m3.count(".") > m1.count(".")


def test_budget_retains_highest_priority_tier() -> None:
    """Under budget pressure, the tier-0 rdf:type survives."""
    g = _many_literals(200)
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Organ")))

    result = _v("M1").verbalise(g, _ENT, "instance")

    assert result.startswith("Heart is a Organ.")


# ---------------------------------------------------------------------------
# Skip rules
# ---------------------------------------------------------------------------


def test_blank_node_object_skipped() -> None:
    g = Graph()
    g.add((_ENT, URIRef("http://example.org/ont#p"), BNode()))
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Organ")))

    assert _v().verbalise(g, _ENT, "instance") == "Heart is a Organ."


def test_anonymous_subclass_expression_skipped() -> None:
    """rdfs:subClassOf pointing at a blank node produces no output."""
    g = Graph()
    g.add((_CLASS, RDFS.subClassOf, BNode()))
    g.add((_CLASS, RDF.type, URIRef("http://example.org/ont#Organ")))

    assert _v().verbalise(g, _CLASS, "class") == "Heart is a Organ."


def test_non_english_literal_skipped() -> None:
    g = Graph()
    g.add((_ENT, RDFS.label, Literal("Herz", lang="de")))
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Organ")))

    result = _v().verbalise(g, _ENT, "instance")

    assert "Herz" not in result
    assert result == "Heart is a Organ."


def test_non_english_literal_on_datatype_predicate_skipped() -> None:
    """Language filtering applies to non-templated predicates too."""
    g = Graph()
    g.add((_ENT, URIRef("http://example.org/ont#mass"), Literal("schwer", lang="de")))
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Organ")))

    result = _v().verbalise(g, _ENT, "instance")

    assert "schwer" not in result
    assert result == "Heart is a Organ."


def test_english_literal_retained() -> None:
    g = Graph()
    g.add((_ENT, RDFS.label, Literal("Heart", lang="en")))

    assert _v().verbalise(g, _ENT, "class") == "Heart is called Heart."


def test_untagged_literal_retained() -> None:
    g = Graph()
    _label(g, _ENT, "Heart")
    g.add((_ENT, URIRef("http://example.org/ont#mass"), Literal("300")))

    assert "Heart has mass 300." in _v().verbalise(g, _ENT, "instance")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _class_with_three_sentences() -> Graph:
    g = Graph()
    organ = URIRef("http://example.org/ont#Organ")
    _label(g, _CLASS, "Heart")
    g.add((_CLASS, RDFS.subClassOf, organ))
    g.add((_CLASS, RDFS.comment, Literal("A hollow muscular organ", lang="en")))
    _label(g, organ, "Organ")
    return g


def test_every_sentence_ends_with_period() -> None:
    result = _v().verbalise(_class_with_three_sentences(), _CLASS, "class")

    assert result.endswith(".")
    sentences = result.split(". ")
    assert len(sentences) == 3
    assert all(sentence for sentence in sentences)


def test_single_space_separator_no_double_or_trailing_space() -> None:
    result = _v().verbalise(_class_with_three_sentences(), _CLASS, "class")

    assert "  " not in result
    assert result == result.strip()
    assert ".  " not in result


def test_no_blank_sentences() -> None:
    g = Graph()
    _label(g, _ENT, "Heart")
    g.add((_ENT, RDFS.label, Literal("Herz", lang="de")))
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Organ")))

    result = _v().verbalise(g, _ENT, "instance")

    assert ".." not in result
    assert not result.startswith(" ")


def test_deterministic_output() -> None:
    g = _class_with_three_sentences()
    v = _v()

    assert v.verbalise(g, _CLASS, "class") == v.verbalise(g, _CLASS, "class")


def test_worked_example_sentences_present() -> None:
    """All three sentences from the issue's worked example are produced.

    The lexical tier (label, comment) precedes the structural tier
    (subClassOf), so the subclass sentence is rendered last.
    """
    result = _v().verbalise(_class_with_three_sentences(), _CLASS, "class")

    assert "Heart is called Heart." in result
    assert "Heart: A hollow muscular organ." in result
    assert result.endswith("Heart is a subclass of Organ.")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_graph_returns_empty_string() -> None:
    assert _v().verbalise(Graph(), _ENT, "instance") == ""


def test_minimal_entity_single_label() -> None:
    g = Graph()
    _label(g, _ENT, "Heart")

    assert _v().verbalise(g, _ENT, "instance") == "Heart is called Heart."


def test_multiple_triples_same_template() -> None:
    g = Graph()
    _label(g, _ENT, "Heart")
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Organ")))
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Muscle")))

    result = _v().verbalise(g, _ENT, "instance")

    assert "Heart is a Organ." in result
    assert "Heart is a Muscle." in result


def test_entity_with_many_triples_all_rendered_under_m3() -> None:
    g = _many_literals(50)
    _label(g, _ENT, "Heart")

    result = _v("M3").verbalise(g, _ENT, "instance")

    assert result.count(".") == 51


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_existing_verbalisers_unchanged() -> None:
    from kgsemembed.verbalisation.v1 import LabelVerbaliser
    from kgsemembed.verbalisation.v2 import AnnotationVerbaliser
    from kgsemembed.verbalisation.v4 import StructuredKVVerbaliser
    from kgsemembed.verbalisation.v6 import SchemaAwareVerbaliser

    g = Graph()
    _label(g, _ENT, "Heart")
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Organ")))
    _label(g, URIRef("http://example.org/ont#Organ"), "Organ")

    assert LabelVerbaliser().verbalise(g, _ENT, "instance") == "Heart"
    assert AnnotationVerbaliser().verbalise(g, _ENT, "instance") == "Label: Heart."
    assert "type: Organ" in StructuredKVVerbaliser().verbalise(g, _ENT, "instance")
    assert "type: Organ" in SchemaAwareVerbaliser().verbalise(g, _ENT, "instance")


def test_exports_correct() -> None:
    from kgsemembed.verbalisation import TemplateNLVerbaliser as PkgV3
    from kgsemembed.verbalisation.v3 import TemplateNLVerbaliser as ModV3

    assert PkgV3 is ModV3


def test_repository_conventions() -> None:
    sig = inspect.signature(TemplateNLVerbaliser.__init__)
    assert sig.parameters["model_key"].default == "M2"

    sig_v = inspect.signature(TemplateNLVerbaliser.verbalise)
    assert "graph" in sig_v.parameters
    assert "entity_uri" in sig_v.parameters
    assert "entity_type" in sig_v.parameters
