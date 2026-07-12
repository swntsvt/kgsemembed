"""Tests for SchemaAwareVerbaliser (V6)."""

import inspect

import pytest
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from kgsemembed.verbalisation.ppas import PPAS_BUDGETS
from kgsemembed.verbalisation.v6 import SchemaAwareVerbaliser


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_ENT = URIRef("http://example.org/ont#Aspirin")
_CLASS = URIRef("http://example.org/ont#Compound")
_PRED = URIRef("http://example.org/ont#authoredBy")

_CAS = URIRef("http://example.org/ont#CASNumber")
_FORMULA = URIRef("http://example.org/ont#MolecularFormula")


def _label(g: Graph, uri: URIRef, text: str) -> None:
    g.add((uri, RDFS.label, Literal(text, lang="en")))


# ---------------------------------------------------------------------------
# Entity-type behaviour: class
# ---------------------------------------------------------------------------


def test_class_uses_class_tier_ordering() -> None:
    """Class entities order triples by the class predicate tiers."""
    g = Graph()
    parent = URIRef("http://example.org/ont#Substance")
    equiv = URIRef("http://example.org/ont#ChemicalEntity")
    _label(g, _CLASS, "Compound")
    g.add((_CLASS, RDF.type, OWL.Class))
    g.add((_CLASS, RDFS.subClassOf, parent))
    g.add((_CLASS, OWL.equivalentClass, equiv))
    _label(g, parent, "Substance")
    _label(g, equiv, "Chemical Entity")

    result = SchemaAwareVerbaliser().verbalise(g, _CLASS, "class")

    assert "label: Compound" in result
    assert "equivalent Class: Chemical Entity" in result
    # Tier 0 (label) precedes Tier 1 (subClassOf) precedes Tier 2 (equivalence)
    assert result.index("label: Compound") < result.index("sub Class Of")
    assert result.index("sub Class Of") < result.index("equivalent Class")


def test_class_label_before_type() -> None:
    """Lexical tier (label) precedes the structural tier (rdf:type)."""
    g = Graph()
    _label(g, _CLASS, "Compound")
    g.add((_CLASS, RDF.type, OWL.Class))

    result = SchemaAwareVerbaliser().verbalise(g, _CLASS, "class")

    assert result.index("label: Compound") < result.index("type:")


# ---------------------------------------------------------------------------
# Entity-type behaviour: instance
# ---------------------------------------------------------------------------


def test_instance_type_precedes_datatype_properties() -> None:
    """rdf:type (tier 0) precedes untiered datatype properties."""
    g = Graph()
    g.add((_ENT, _CAS, Literal("50-78-2")))
    g.add((_ENT, _FORMULA, Literal("C9H8O4")))
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Drug")))
    _label(g, _CAS, "CAS number")
    _label(g, _FORMULA, "Molecular Formula")

    result = SchemaAwareVerbaliser().verbalise(g, _ENT, "instance")

    assert result.startswith("type:")
    assert result.index("type:") < result.index("CAS number:")
    assert result.index("type:") < result.index("Molecular Formula:")


def test_instance_ordering_follows_instance_tiers() -> None:
    """Instance ordering keeps tier-0 predicates ahead of untiered ones."""
    g = Graph()
    g.add((_ENT, _CAS, Literal("50-78-2")))
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Drug")))
    _label(g, _ENT, "Aspirin")
    _label(g, _CAS, "CAS number")

    result = SchemaAwareVerbaliser().verbalise(g, _ENT, "instance")
    pairs = result.split(" | ")

    tiered = {"type:", "label:"}
    assert any(pairs[0].startswith(p) for p in tiered)
    assert result.index("CAS number:") > result.index("type:")


# ---------------------------------------------------------------------------
# Entity-type behaviour: predicate
# ---------------------------------------------------------------------------


def test_predicate_uses_predicate_tier_ordering() -> None:
    """Predicate entities order triples by the predicate tiers."""
    g = Graph()
    domain = URIRef("http://example.org/ont#Person")
    rng = URIRef("http://example.org/ont#Document")
    parent = URIRef("http://example.org/ont#relatedTo")
    _label(g, _PRED, "authored by")
    g.add((_PRED, RDFS.domain, domain))
    g.add((_PRED, RDFS.range, rng))
    g.add((_PRED, RDFS.subPropertyOf, parent))
    _label(g, domain, "Person")
    _label(g, rng, "Document")
    _label(g, parent, "related to")

    result = SchemaAwareVerbaliser().verbalise(g, _PRED, "predicate")

    assert "domain: Person" in result
    assert "range: Document" in result
    # Tier 0 (label) < Tier 1 (domain/range) < Tier 2 (subPropertyOf)
    assert result.index("label: authored by") < result.index("domain: Person")
    assert result.index("range: Document") < result.index("sub Property Of")


# ---------------------------------------------------------------------------
# Entity-type validation
# ---------------------------------------------------------------------------


def test_unsupported_entity_type_raises_value_error() -> None:
    """An unsupported entity type raises a descriptive ValueError."""
    g = Graph()
    _label(g, _ENT, "Aspirin")
    v = SchemaAwareVerbaliser()

    with pytest.raises(ValueError, match="Unsupported entity type"):
        v.verbalise(g, _ENT, "relationship")


def test_value_error_names_the_offending_type() -> None:
    """The error message identifies the unsupported entity type."""
    g = Graph()
    v = SchemaAwareVerbaliser()

    with pytest.raises(ValueError, match="individual"):
        v.verbalise(g, _ENT, "individual")


# ---------------------------------------------------------------------------
# Token-budget behaviour
# ---------------------------------------------------------------------------


def _many_triples(count: int) -> Graph:
    g = Graph()
    for i in range(count):
        pred = URIRef(f"http://example.org/ont#prop_{i}")
        g.add((_ENT, pred, Literal(f"value_{i}")))
    return g


def test_budgeted_model_respects_token_budget() -> None:
    """A configured budget caps the number of serialised triples."""
    g = _many_triples(100)
    result = SchemaAwareVerbaliser(model_key="M1").verbalise(g, _ENT, "instance")
    pairs = result.split(" | ")

    assert 0 < len(pairs) < 100


def test_m3_includes_all_tier_ordered_triples() -> None:
    """M3 has no budget and retains every tier-ordered triple."""
    assert PPAS_BUDGETS["M3"] is None
    g = _many_triples(100)
    result = SchemaAwareVerbaliser(model_key="M3").verbalise(g, _ENT, "instance")
    pairs = result.split(" | ")

    assert len(pairs) == 100


def test_budget_retains_highest_priority_tier() -> None:
    """Under budget pressure, tier-0 predicates survive while untiered drop."""
    g = _many_triples(100)
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Drug")))

    result = SchemaAwareVerbaliser(model_key="M1").verbalise(g, _ENT, "instance")
    pairs = result.split(" | ")

    assert result.startswith("type: Drug")
    assert len(pairs) < 101


def test_selected_triples_never_exceed_budget() -> None:
    """The estimated cost of the retained triples stays within the budget.

    Uses separator-free predicate and object names so the rendered pair and
    the implementation's internal cost estimate share the same token count.
    """
    from kgsemembed.verbalisation.ppas import estimate_tokens

    g = Graph()
    for i in range(200):
        g.add((_ENT, URIRef(f"http://example.org/ont#p{i}"), Literal(f"v{i}")))
    result = SchemaAwareVerbaliser(model_key="M1").verbalise(g, _ENT, "instance")

    total = sum(estimate_tokens(pair) for pair in result.split(" | "))
    assert total <= PPAS_BUDGETS["M1"]


def test_m3_output_exceeds_budgeted_output() -> None:
    """M3 (no cap) keeps strictly more triples than a budgeted model."""
    g = _many_triples(100)
    m1 = SchemaAwareVerbaliser(model_key="M1").verbalise(g, _ENT, "instance")
    m3 = SchemaAwareVerbaliser(model_key="M3").verbalise(g, _ENT, "instance")

    assert len(m3.split(" | ")) > len(m1.split(" | "))


def test_m3_preserves_ordering() -> None:
    """M3 keeps tier ordering even without a budget cap."""
    g = Graph()
    g.add((_ENT, _CAS, Literal("50-78-2")))
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Drug")))
    _label(g, _CAS, "CAS number")

    result = SchemaAwareVerbaliser(model_key="M3").verbalise(g, _ENT, "instance")

    assert result.startswith("type:")


# ---------------------------------------------------------------------------
# Always-on tier selection (independent of triple count)
# ---------------------------------------------------------------------------


def _instance_with_type_and_props(prop_count: int) -> Graph:
    g = Graph()
    for i in range(prop_count):
        pred = URIRef(f"http://example.org/ont#prop_{i}")
        g.add((_ENT, pred, Literal(f"value_{i}")))
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Drug")))
    return g


@pytest.mark.parametrize("triple_count", [1, 5, 10, 100])
def test_tier_selection_applied_regardless_of_count(triple_count: int) -> None:
    """Tier ordering forces rdf:type first for any triple count."""
    g = _instance_with_type_and_props(triple_count - 1)
    result = SchemaAwareVerbaliser(model_key="M3").verbalise(g, _ENT, "instance")

    assert result.startswith("type:")


def test_tier_selection_applied_for_single_triple() -> None:
    """A single-triple entity is still tier-ordered, not bypassed."""
    g = Graph()
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Drug")))
    result = SchemaAwareVerbaliser().verbalise(g, _ENT, "instance")

    assert result == "type: Drug"


def test_tier_selection_applied_for_small_class() -> None:
    """A two-triple class entity still follows class tier ordering."""
    g = Graph()
    equiv = URIRef("http://example.org/ont#ChemicalEntity")
    _label(g, _CLASS, "Compound")
    g.add((_CLASS, OWL.equivalentClass, equiv))
    _label(g, equiv, "Chemical Entity")

    result = SchemaAwareVerbaliser().verbalise(g, _CLASS, "class")

    assert result.index("label: Compound") < result.index("equivalent Class")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_graph_returns_empty_string() -> None:
    """An entity with no triples yields an empty string."""
    result = SchemaAwareVerbaliser().verbalise(Graph(), _ENT, "instance")
    assert result == ""


def test_blank_node_object_excluded() -> None:
    """Blank-node objects are silently excluded without raising."""
    g = Graph()
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Drug")))
    g.add((_ENT, _FORMULA, BNode()))

    result = SchemaAwareVerbaliser().verbalise(g, _ENT, "instance")

    assert result == "type: Drug"


def test_owl_sameas_excluded() -> None:
    """owl:sameAs triples are excluded, matching V4 formatting."""
    g = Graph()
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Drug")))
    g.add((_ENT, OWL.sameAs, URIRef("http://dbpedia.org/resource/Aspirin")))

    result = SchemaAwareVerbaliser().verbalise(g, _ENT, "instance")

    assert "sameAs" not in result
    assert "dbpedia" not in result


def test_missing_predicate_label_falls_back_to_local_name() -> None:
    """A predicate without a label uses its camelCase-split local name."""
    g = Graph()
    g.add((_ENT, URIRef("http://example.org/ont#customProp"), Literal("v")))

    result = SchemaAwareVerbaliser().verbalise(g, _ENT, "instance")

    assert "custom Prop: v" in result


def test_missing_object_label_falls_back_to_local_name() -> None:
    """A URI object without a label uses its local name."""
    g = Graph()
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#SomeClass")))

    result = SchemaAwareVerbaliser().verbalise(g, _ENT, "instance")

    assert "type: Some Class" in result


def test_literal_and_uri_objects_both_serialised() -> None:
    """Literal values and URI-valued objects are both rendered."""
    g = Graph()
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Drug")))
    g.add((_ENT, _CAS, Literal("50-78-2")))
    _label(g, _CAS, "CAS number")

    result = SchemaAwareVerbaliser().verbalise(g, _ENT, "instance")

    assert "type: Drug" in result
    assert "CAS number: 50-78-2" in result


def test_multiple_values_for_same_predicate() -> None:
    """Repeated predicate values each produce a key-value pair."""
    g = Graph()
    a1 = URIRef("http://example.org/person/Alice")
    a2 = URIRef("http://example.org/person/Bob")
    g.add((_ENT, _PRED, a1))
    g.add((_ENT, _PRED, a2))
    _label(g, a1, "Alice")
    _label(g, a2, "Bob")

    result = SchemaAwareVerbaliser().verbalise(g, _ENT, "instance")

    assert result.count("authored By:") == 2
    assert "Alice" in result and "Bob" in result


def test_repeated_rdf_types() -> None:
    """Multiple rdf:type declarations each appear with the 'type' key."""
    g = Graph()
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Drug")))
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Compound")))

    result = SchemaAwareVerbaliser().verbalise(g, _ENT, "instance")

    assert result.count("type:") == 2


def test_minimal_entity_with_single_label() -> None:
    """An entity with only a label produces a single key-value pair."""
    g = Graph()
    _label(g, _ENT, "Minimal")

    result = SchemaAwareVerbaliser().verbalise(g, _ENT, "instance")

    assert result == "label: Minimal"


# ---------------------------------------------------------------------------
# Formatting parity with V4
# ---------------------------------------------------------------------------


def test_pipe_separator_format() -> None:
    """Output uses ' | ' between key-value pairs, matching V4."""
    g = Graph()
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Drug")))
    g.add((_ENT, _CAS, Literal("50-78-2")))
    _label(g, _CAS, "CAS number")

    result = SchemaAwareVerbaliser().verbalise(g, _ENT, "instance")
    pairs = result.split(" | ")

    assert len(pairs) == 2
    for pair in pairs:
        assert ": " in pair


def test_deterministic_output() -> None:
    """The same input always produces the same output."""
    g = Graph()
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Drug")))
    g.add((_ENT, _CAS, Literal("50-78-2")))
    _label(g, _CAS, "CAS number")
    v = SchemaAwareVerbaliser()

    assert v.verbalise(g, _ENT, "instance") == v.verbalise(g, _ENT, "instance")


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_v4_behaviour_unchanged() -> None:
    """V4 still serialises instance entities exactly as before."""
    from kgsemembed.verbalisation.v4 import StructuredKVVerbaliser

    g = Graph()
    g.add((_ENT, RDF.type, URIRef("http://example.org/ont#Drug")))
    _label(g, URIRef("http://example.org/ont#Drug"), "Drug")

    assert StructuredKVVerbaliser().verbalise(g, _ENT, "instance") == "type: Drug"


def test_existing_verbalisers_unaffected() -> None:
    """V1 and V2 continue to behave as before."""
    from kgsemembed.verbalisation.v1 import LabelVerbaliser
    from kgsemembed.verbalisation.v2 import AnnotationVerbaliser

    g = Graph()
    _label(g, _ENT, "Test")

    assert LabelVerbaliser().verbalise(g, _ENT, "instance") == "Test"
    assert AnnotationVerbaliser().verbalise(g, _ENT, "instance") == "Label: Test."


def test_exports_correct() -> None:
    """SchemaAwareVerbaliser is exported from the package-level __init__."""
    from kgsemembed.verbalisation import SchemaAwareVerbaliser as PkgV6
    from kgsemembed.verbalisation.v6 import SchemaAwareVerbaliser as ModV6

    assert PkgV6 is ModV6


def test_repository_conventions() -> None:
    """V6 follows repository naming and typing conventions."""
    sig = inspect.signature(SchemaAwareVerbaliser.__init__)
    assert sig.parameters["model_key"].default == "M2"

    sig_v = inspect.signature(SchemaAwareVerbaliser.verbalise)
    assert "graph" in sig_v.parameters
    assert "entity_uri" in sig_v.parameters
    assert "entity_type" in sig_v.parameters
