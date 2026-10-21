"""Tests for RelationalSignatureVerbaliser (V8)."""

import pytest
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, SKOS

from kgsemembed.verbalisation import (
    EXCLUDED_PREDICATES,
    RelationalSignatureVerbaliser,
)
from kgsemembed.verbalisation.v8 import exclusion_family, v8_exclusions_digest

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


_FMA = "http://purl.org/sig/ont/fma/"
_OBO_IN_OWL = "http://www.geneontology.org/formats/oboInOwl#"
_OWL = "http://www.w3.org/2002/07/owl#"

_ANNOTATION_EXAMPLES = [
    (f"{_FMA}synonym", "roles_v1_annotation"),
    (f"{_FMA}preferred_name", "roles_v1_annotation"),
    (f"{_FMA}FMAID", "roles_v1_annotation"),
    (f"{_FMA}definition", "roles_v1_annotation"),
    (f"{_FMA}non-English_equivalent", "roles_v1_annotation"),
    (f"{_FMA}homonym_of", "roles_v1_annotation"),
    (f"{_OBO_IN_OWL}hasRelatedSynonym", "roles_v1_annotation"),
    (f"{_OBO_IN_OWL}hasDefinition", "roles_v1_annotation"),
    (f"{_OBO_IN_OWL}hasDbXref", "roles_v1_annotation"),
    ("http://www.w3.org/2000/01/rdf-schema#seeAlso", "owl_annotation"),
    (f"{_OWL}annotatedSource", "owl_axiom_annotation"),
    ("http://www.w3.org/2004/02/skos/core#hiddenLabel", "skos_documentation"),
    ("http://purl.org/dc/elements/1.1/description", "dublin_core"),
    ("http://schema.org/description", "schema_org"),
    ("http://xmlns.com/foaf/0.1/givenName", "foaf"),
    ("http://dbkwik.webdatacommons.org/ontology/abstract", "dbkwik_page"),
    ("http://dbkwik.webdatacommons.org/ontology/wikiPageWikiLinkText", "dbkwik_page"),
]

_RELATION_EXAMPLES = [
    "http://www.w3.org/2000/01/rdf-schema#subClassOf",
    "http://www.w3.org/2000/01/rdf-schema#domain",
    f"{_OWL}disjointWith",
    f"{_OWL}someValuesFrom",
    "http://www.w3.org/2004/02/skos/core#broader",
    "http://purl.org/dc/terms/subject",
    "http://dbkwik.webdatacommons.org/ontology/wikiPageWikiLink",
    "http://dbkwik.webdatacommons.org/memory-alpha.wikia.com/property/species",
    "http://dbpedia.org/ontology/birthDate",
    "http://www.wikidata.org/entity/P577",
    f"{_NS}partOf",
]


@pytest.mark.parametrize(("predicate", "family"), _ANNOTATION_EXAMPLES)
def test_annotation_predicate_of_each_family_excluded(predicate: str, family: str) -> None:
    g = Graph()
    entity = _u("Heart")
    g.add((entity, URIRef(predicate), Literal("text")))
    g.add((_u("Axiom"), URIRef(predicate), entity))
    g.add((entity, RDFS.subClassOf, _u("Organ")))
    out = _v().verbalise(g, entity, "class")
    assert exclusion_family(predicate) == family
    assert _outgoing(out) == "Outgoing relationships: sub class of."
    assert _incoming(out) == "Incoming relationships: none."


@pytest.mark.parametrize("predicate", _RELATION_EXAMPLES)
def test_relation_predicates_kept(predicate: str) -> None:
    g = Graph()
    entity = _u("Heart")
    g.add((entity, URIRef(predicate), _u("Other")))
    g.add((_u("Other"), URIRef(predicate), entity))
    out = _v().verbalise(g, entity, "class")
    assert exclusion_family(predicate) is None
    assert _outgoing(out) != "Outgoing relationships: none."
    assert _outgoing(out).split(": ", 1)[1] == _incoming(out).split(": ", 1)[1]


def test_d2_annotation_properties_not_listed_as_relationships() -> None:
    g = Graph()
    entity = URIRef("http://human.owl#NCI_C12611")
    g.add((entity, URIRef(f"{_OBO_IN_OWL}hasDefinition"), BNode()))
    g.add((entity, URIRef(f"{_OBO_IN_OWL}hasRelatedSynonym"), BNode()))
    g.add((entity, RDFS.subClassOf, URIRef("http://human.owl#NCI_C12219")))
    out = _v().verbalise(g, entity, "class")
    assert _outgoing(out) == "Outgoing relationships: sub class of."


def test_every_annotation_role_of_roles_v1_excluded() -> None:
    from kgsemembed.verbalisation.predicate_roles import CORE, DISJOINTNESS, get_role_map

    roles = get_role_map("roles_v1").roles
    for predicate, role in roles.items():
        kept = role in (CORE, DISJOINTNESS) and predicate != str(RDF.type) and predicate != str(RDFS.label)
        assert (predicate not in EXCLUDED_PREDICATES) == kept, predicate


def test_v8_exclusions_digest_is_pinned() -> None:
    """A change to the exclusion families must be deliberate."""
    assert v8_exclusions_digest() == (
        "062039673a927c8a02b53bb5e926c5b833def7bf5b23c33331fc8e2dfdac52ac"
    )


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
