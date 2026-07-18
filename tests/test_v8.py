"""Tests for RelationalSignatureVerbaliser (V8)."""

import pytest
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, SKOS

from kgsemembed.verbalisation import (
    EXCLUDED_PREDICATES,
    RelationalSignatureVerbaliser,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_NS = "http://example.org/ont#"


def _u(name: str) -> URIRef:
    return URIRef(f"{_NS}{name}")


def _label(g: Graph, uri: URIRef, text: str) -> None:
    g.add((uri, RDFS.label, Literal(text, lang="en")))


def _v() -> RelationalSignatureVerbaliser:
    return RelationalSignatureVerbaliser()


def _line(out: str, prefix: str) -> str:
    for line in out.splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(f"No line starting with {prefix!r} in:\n{out}")


def _outgoing(out: str) -> str:
    return _line(out, "Outgoing relationships:")


def _incoming(out: str) -> str:
    return _line(out, "Incoming relationships:")


# ---------------------------------------------------------------------------
# Construction and dispatch
# ---------------------------------------------------------------------------


def test_constructor_takes_no_arguments() -> None:
    assert isinstance(_v(), RelationalSignatureVerbaliser)


@pytest.mark.parametrize("entity_type", ["class", "instance"])
def test_class_and_instance_produce_both_summaries(entity_type: str) -> None:
    g = Graph()
    subject = _u("Heart")
    g.add((subject, _u("partOf"), _u("Body")))
    g.add((_u("Aorta"), _u("connectsTo"), subject))
    out = _v().verbalise(g, subject, entity_type)
    assert _outgoing(out) == "Outgoing relationships: part of."
    assert _incoming(out) == "Incoming relationships: connects to."


def test_unsupported_entity_type_raises_value_error() -> None:
    g = Graph()
    with pytest.raises(ValueError, match="widget"):
        _v().verbalise(g, _u("Thing"), "widget")


# ---------------------------------------------------------------------------
# Outgoing relationships
# ---------------------------------------------------------------------------


def test_outgoing_distinct_types_sorted() -> None:
    g = Graph()
    subject = _u("Heart")
    g.add((subject, _u("regulates"), _u("Rhythm")))
    g.add((subject, _u("hasPart"), _u("Valve")))
    out = _v().verbalise(g, subject, "class")
    assert _outgoing(out) == "Outgoing relationships: has part, regulates."


def test_outgoing_deduplicates_repeated_predicate() -> None:
    g = Graph()
    subject = _u("Heart")
    g.add((subject, _u("hasPart"), _u("Valve")))
    g.add((subject, _u("hasPart"), _u("Chamber")))
    out = _v().verbalise(g, subject, "class")
    assert _outgoing(out) == "Outgoing relationships: has part."


def test_outgoing_uses_predicate_label_when_available() -> None:
    g = Graph()
    subject = _u("Heart")
    pred = _u("p1")
    _label(g, pred, "Regulates")
    g.add((subject, pred, _u("Rhythm")))
    out = _v().verbalise(g, subject, "class")
    assert _outgoing(out) == "Outgoing relationships: regulates."


# ---------------------------------------------------------------------------
# Incoming relationships
# ---------------------------------------------------------------------------


def test_incoming_distinct_types_sorted_and_deduplicated() -> None:
    g = Graph()
    obj = _u("Heart")
    g.add((_u("Aorta"), _u("connectsTo"), obj))
    g.add((_u("Vein"), _u("connectsTo"), obj))
    g.add((_u("Body"), _u("hasOrgan"), obj))
    out = _v().verbalise(g, obj, "class")
    assert _incoming(out) == "Incoming relationships: connects to, has organ."


# ---------------------------------------------------------------------------
# Exclusion rules
# ---------------------------------------------------------------------------


def test_excluded_predicates_omitted_from_both_directions() -> None:
    g = Graph()
    entity = _u("Heart")
    for uri in EXCLUDED_PREDICATES:
        g.add((entity, URIRef(uri), _u("Other")))
        g.add((_u("Other"), URIRef(uri), entity))
    g.add((entity, _u("hasPart"), _u("Valve")))
    g.add((_u("Body"), _u("contains"), entity))
    out = _v().verbalise(g, entity, "class")
    assert _outgoing(out) == "Outgoing relationships: has part."
    assert _incoming(out) == "Incoming relationships: contains."


def test_excluded_predicates_constant_unchanged() -> None:
    assert EXCLUDED_PREDICATES == [
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://www.w3.org/2000/01/rdf-schema#comment",
        "http://www.w3.org/2004/02/skos/core#prefLabel",
        "http://www.w3.org/2004/02/skos/core#altLabel",
        "http://www.w3.org/2004/02/skos/core#definition",
        "http://purl.obolibrary.org/obo/IAO_0000115",
    ]


# ---------------------------------------------------------------------------
# Empty signatures
# ---------------------------------------------------------------------------


def test_no_outgoing_triples_uses_none() -> None:
    g = Graph()
    entity = _u("Heart")
    g.add((_u("Body"), _u("contains"), entity))
    out = _v().verbalise(g, entity, "class")
    assert _outgoing(out) == "Outgoing relationships: none."
    assert _incoming(out) == "Incoming relationships: contains."


def test_no_incoming_triples_uses_none() -> None:
    g = Graph()
    entity = _u("Heart")
    g.add((entity, _u("hasPart"), _u("Valve")))
    out = _v().verbalise(g, entity, "class")
    assert _outgoing(out) == "Outgoing relationships: has part."
    assert _incoming(out) == "Incoming relationships: none."


def test_no_eligible_triples_uses_none_both() -> None:
    g = Graph()
    entity = _u("Heart")
    g.add((entity, RDF.type, _u("Organ")))
    g.add((entity, RDFS.label, Literal("Heart", lang="en")))
    out = _v().verbalise(g, entity, "instance")
    assert out == (
        "Outgoing relationships: none.\nIncoming relationships: none."
    )


def test_entity_with_no_triples_uses_none_both() -> None:
    g = Graph()
    out = _v().verbalise(g, _u("Ghost"), "class")
    assert out == (
        "Outgoing relationships: none.\nIncoming relationships: none."
    )


# ---------------------------------------------------------------------------
# Predicate entities
# ---------------------------------------------------------------------------


def _predicate_graph() -> tuple[Graph, URIRef]:
    g = Graph()
    pred = _u("hasPart")
    _label(g, pred, "has part")
    g.add((pred, RDFS.comment, Literal("Relates a whole to a part", lang="en")))
    g.add((pred, RDFS.domain, _u("Organism")))
    g.add((pred, RDFS.range, _u("Anatomy")))
    _label(g, _u("Organism"), "Organism")
    _label(g, _u("Anatomy"), "Anatomy")
    g.add((pred, RDFS.subPropertyOf, _u("relatedTo")))
    g.add((_u("Foo"), RDFS.subPropertyOf, pred))
    return g, pred


def test_predicate_entity_full_metadata() -> None:
    g, pred = _predicate_graph()
    out = _v().verbalise(g, pred, "predicate")
    assert out == (
        "Property: has part.\n"
        "Definition: Relates a whole to a part.\n"
        "Domain: Organism.\n"
        "Range: Anatomy.\n"
        "Outgoing relationships: domain, range, sub property of.\n"
        "Incoming relationships: sub property of."
    )


def test_predicate_entity_omits_definition_when_absent() -> None:
    g = Graph()
    pred = _u("hasPart")
    _label(g, pred, "has part")
    g.add((pred, RDFS.domain, _u("Organism")))
    _label(g, _u("Organism"), "Organism")
    out = _v().verbalise(g, pred, "predicate")
    assert "Definition:" not in out
    assert _line(out, "Domain:") == "Domain: Organism."
    assert "Range:" not in out


def test_predicate_entity_omits_domain_and_range_when_absent() -> None:
    g = Graph()
    pred = _u("hasPart")
    _label(g, pred, "has part")
    g.add((pred, RDFS.comment, Literal("A definition.", lang="en")))
    out = _v().verbalise(g, pred, "predicate")
    assert out.startswith("Property: has part.\nDefinition: A definition.")
    assert "Domain:" not in out
    assert "Range:" not in out
    assert _outgoing(out) == "Outgoing relationships: none."
    assert _incoming(out) == "Incoming relationships: none."


def test_predicate_definition_comes_from_rdfs_comment_only() -> None:
    g = Graph()
    pred = _u("hasPart")
    _label(g, pred, "has part")
    g.add((pred, SKOS.definition, Literal("A skos definition.", lang="en")))
    g.add(
        (pred, URIRef("http://purl.obolibrary.org/obo/IAO_0000115"),
         Literal("An obo definition.", lang="en"))
    )
    out = _v().verbalise(g, pred, "predicate")
    assert "Definition:" not in out


def test_predicate_definition_prefers_rdfs_comment() -> None:
    g = Graph()
    pred = _u("hasPart")
    _label(g, pred, "has part")
    g.add((pred, RDFS.comment, Literal("The real comment", lang="en")))
    g.add((pred, SKOS.definition, Literal("A skos definition", lang="en")))
    out = _v().verbalise(g, pred, "predicate")
    assert _line(out, "Definition:") == "Definition: The real comment."


def test_predicate_entity_minimal_falls_back_to_local_name() -> None:
    g = Graph()
    pred = _u("hasPart")
    out = _v().verbalise(g, pred, "predicate")
    assert out == (
        "Property: has Part.\n"
        "Outgoing relationships: none.\n"
        "Incoming relationships: none."
    )


# ---------------------------------------------------------------------------
# Lowercasing
# ---------------------------------------------------------------------------


def test_predicate_labels_always_lowercased() -> None:
    g = Graph()
    subject = _u("Heart")
    upper = _u("p1")
    _label(g, upper, "REGULATES")
    g.add((subject, upper, _u("Rhythm")))
    g.add((subject, _u("HasPart"), _u("Valve")))
    out = _v().verbalise(g, subject, "class")
    assert _outgoing(out) == "Outgoing relationships: has part, regulates."


# ---------------------------------------------------------------------------
# Determinism and content constraints
# ---------------------------------------------------------------------------


def test_output_deterministic_across_runs() -> None:
    g = Graph()
    subject = _u("Heart")
    g.add((subject, _u("regulates"), _u("Rhythm")))
    g.add((subject, _u("hasPart"), _u("Valve")))
    g.add((_u("Body"), _u("contains"), subject))
    first = _v().verbalise(g, subject, "class")
    second = _v().verbalise(g, subject, "class")
    assert first == second


def test_no_literal_values_or_neighbour_identities() -> None:
    g = Graph()
    subject = _u("Heart")
    g.add((subject, _u("hasWeight"), Literal("300g")))
    g.add((subject, _u("hasPart"), _u("Valve")))
    out = _v().verbalise(g, subject, "class")
    assert "300g" not in out
    assert "Valve" not in out
    assert _outgoing(out) == "Outgoing relationships: has part, has weight."


def test_opaque_uris_and_missing_labels_use_local_name() -> None:
    g = Graph()
    subject = URIRef("http://example.org/obo/ORG_0001")
    g.add((subject, URIRef("http://example.org/obo/RO_0002131"), _u("X")))
    out = _v().verbalise(g, subject, "class")
    assert _outgoing(out) == "Outgoing relationships: ro 0002131."


def test_malformed_graph_with_bnode_object_does_not_raise() -> None:
    g = Graph()
    subject = _u("Heart")
    g.add((subject, _u("hasPart"), BNode()))
    g.add((BNode(), _u("contains"), subject))
    out = _v().verbalise(g, subject, "class")
    assert _outgoing(out) == "Outgoing relationships: has part."
    assert _incoming(out) == "Incoming relationships: contains."


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_exports_are_correct() -> None:
    import kgsemembed.verbalisation as verb

    assert "RelationalSignatureVerbaliser" in verb.__all__
    assert "EXCLUDED_PREDICATES" in verb.__all__
    assert verb.RelationalSignatureVerbaliser is RelationalSignatureVerbaliser


def test_existing_verbalisers_still_importable() -> None:
    from kgsemembed.verbalisation import (
        AnnotationVerbaliser,
        HierarchicalContextVerbaliser,
        LabelVerbaliser,
        NeighbourhoodWalkVerbaliser,
        SchemaAwareVerbaliser,
        StructuredKVVerbaliser,
        TemplateNLVerbaliser,
    )

    assert LabelVerbaliser is not None
    assert AnnotationVerbaliser is not None
    assert TemplateNLVerbaliser is not None
    assert StructuredKVVerbaliser is not None
    assert NeighbourhoodWalkVerbaliser is not None
    assert SchemaAwareVerbaliser is not None
    assert HierarchicalContextVerbaliser is not None
