"""Tests for HierarchicalContextVerbaliser (V7)."""

from unittest.mock import patch

import pytest
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS

from kgsemembed.verbalisation import HierarchicalContextVerbaliser
from kgsemembed.verbalisation import v7 as v7_module
from kgsemembed.verbalisation.ppas import PPAS_BUDGETS, estimate_tokens
from kgsemembed.verbalisation.v7 import MAX_ANCESTOR_DEPTH, MAX_SIBLINGS

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_NS = "http://example.org/ont#"


def _u(name: str) -> URIRef:
    return URIRef(f"{_NS}{name}")


def _label(g: Graph, uri: URIRef, text: str) -> None:
    g.add((uri, RDFS.label, Literal(text, lang="en")))


def _v(model_key: str = "M3") -> HierarchicalContextVerbaliser:
    return HierarchicalContextVerbaliser(model_key=model_key)


def _class_chain(g: Graph, *names: str) -> list[URIRef]:
    """Add labelled classes ``n0 subClassOf n1 subClassOf ...`` and return them."""
    nodes = [_u(name.replace(" ", "")) for name in names]
    for node, text in zip(nodes, names):
        _label(g, node, text)
    for child, parent in zip(nodes, nodes[1:]):
        g.add((child, RDFS.subClassOf, parent))
    return nodes


def _parents_line(out: str) -> str | None:
    for line in out.splitlines():
        if line.startswith("Parents:"):
            return line
    return None


def _related_line(out: str) -> str | None:
    for line in out.splitlines():
        if line.startswith("Related types include:"):
            return line
    return None


# ---------------------------------------------------------------------------
# Constants -- part of the experimental protocol
# ---------------------------------------------------------------------------


def test_protocol_constants() -> None:
    assert MAX_ANCESTOR_DEPTH == 3
    assert MAX_SIBLINGS == 5


def test_default_model_key_is_m2() -> None:
    assert HierarchicalContextVerbaliser().model_key == "M2"


# ---------------------------------------------------------------------------
# Base annotation
# ---------------------------------------------------------------------------


def test_annotation_forms_prefix() -> None:
    g = Graph()
    child, parent = _class_chain(g, "Heart", "Organ")
    g.add((child, RDFS.comment, Literal("A hollow muscular organ.", lang="en")))

    out = _v().verbalise(g, child, "class")
    assert out.startswith(
        "Label: Heart. Definition: A hollow muscular organ."
    )


def test_hierarchy_augments_rather_than_replaces() -> None:
    g = Graph()
    child, parent = _class_chain(g, "Heart", "Organ")

    out = _v().verbalise(g, child, "class")
    assert out.startswith("Label: Heart.")
    assert "Parents: Organ." in out


def test_reuses_annotation_verbaliser() -> None:
    g = Graph()
    child, parent = _class_chain(g, "Heart", "Organ")

    with patch.object(
        v7_module.AnnotationVerbaliser, "verbalise", return_value="SENTINEL."
    ) as spy:
        out = _v().verbalise(g, child, "class")

    spy.assert_called_once()
    assert out.startswith("SENTINEL.")
    assert "Parents: Organ." in out


# ---------------------------------------------------------------------------
# Ancestor traversal
# ---------------------------------------------------------------------------


def test_class_hierarchy_traversal() -> None:
    g = Graph()
    heart, *_ = _class_chain(g, "Heart", "Organ", "Anatomical structure")

    out = _v().verbalise(g, heart, "class")
    assert _parents_line(out) == "Parents: Organ -> Anatomical structure."


def test_instance_hierarchy_traversal() -> None:
    g = Graph()
    organ, structure = _class_chain(g, "Organ", "Anatomical structure")
    heart = _u("myHeart")
    _label(g, heart, "My heart")
    g.add((heart, RDF.type, organ))

    out = _v().verbalise(g, heart, "instance")
    assert _parents_line(out) == "Parents: Organ -> Anatomical structure."


def test_predicate_hierarchy_traversal() -> None:
    g = Graph()
    part_of = _u("partOf")
    related = _u("relatedTo")
    connected = _u("connectedTo")
    _label(g, part_of, "part of")
    _label(g, related, "related to")
    _label(g, connected, "connected to")
    g.add((part_of, RDFS.subPropertyOf, related))
    g.add((related, RDFS.subPropertyOf, connected))

    out = _v().verbalise(g, part_of, "predicate")
    assert _parents_line(out) == "Parents: related to -> connected to."


def test_ancestor_depth_capped() -> None:
    g = Graph()
    heart, *_ = _class_chain(g, "Heart", "A", "B", "C", "D", "E")

    ancestors = _v()._get_ancestors(g, heart, "class")
    assert len(ancestors) == MAX_ANCESTOR_DEPTH
    assert [str(a) for a in ancestors] == [str(_u("A")), str(_u("B")), str(_u("C"))]


def test_ancestor_ordering_nearest_first() -> None:
    g = Graph()
    heart, *_ = _class_chain(g, "Heart", "Organ", "Structure")

    ancestors = _v()._get_ancestors(g, heart, "class")
    assert ancestors == [_u("Organ"), _u("Structure")]


def test_hierarchy_shorter_than_limit() -> None:
    g = Graph()
    heart, organ = _class_chain(g, "Heart", "Organ")

    out = _v().verbalise(g, heart, "class")
    assert _parents_line(out) == "Parents: Organ."


def test_no_superclass_omits_parents() -> None:
    g = Graph()
    _label(g, _u("Heart"), "Heart")

    out = _v().verbalise(g, _u("Heart"), "class")
    assert _parents_line(out) is None


def test_instance_without_type_omits_parents() -> None:
    g = Graph()
    _label(g, _u("myHeart"), "My heart")

    out = _v().verbalise(g, _u("myHeart"), "instance")
    assert _parents_line(out) is None
    assert _related_line(out) is None


def test_predicate_without_superproperty_omits_parents() -> None:
    g = Graph()
    _label(g, _u("partOf"), "part of")

    out = _v().verbalise(g, _u("partOf"), "predicate")
    assert _parents_line(out) is None


def test_multiple_parents_follow_smallest_identifier() -> None:
    g = Graph()
    heart = _u("Heart")
    _label(g, heart, "Heart")
    for name in ("Zeta", "Alpha", "Mu"):
        _label(g, _u(name), name)
        g.add((heart, RDFS.subClassOf, _u(name)))

    ancestors = _v()._get_ancestors(g, heart, "class")
    assert ancestors == [_u("Alpha")]


# ---------------------------------------------------------------------------
# Cycle handling
# ---------------------------------------------------------------------------


def test_two_node_cycle_terminates() -> None:
    g = Graph()
    a, b = _u("A"), _u("B")
    _label(g, a, "A")
    _label(g, b, "B")
    g.add((a, RDFS.subClassOf, b))
    g.add((b, RDFS.subClassOf, a))

    ancestors = _v()._get_ancestors(g, a, "class")
    assert ancestors == [b]


def test_three_node_cycle_terminates() -> None:
    g = Graph()
    a, b, c = _u("A"), _u("B"), _u("C")
    for node, text in ((a, "A"), (b, "B"), (c, "C")):
        _label(g, node, text)
    g.add((a, RDFS.subClassOf, b))
    g.add((b, RDFS.subClassOf, c))
    g.add((c, RDFS.subClassOf, a))

    ancestors = _v()._get_ancestors(g, a, "class")
    assert ancestors == [b, c]


def test_self_loop_terminates() -> None:
    g = Graph()
    a = _u("A")
    _label(g, a, "A")
    g.add((a, RDFS.subClassOf, a))

    assert _v()._get_ancestors(g, a, "class") == []


# ---------------------------------------------------------------------------
# Blank-node superclasses
# ---------------------------------------------------------------------------


def test_blank_node_superclass_terminates_silently() -> None:
    g = Graph()
    heart = _u("Heart")
    _label(g, heart, "Heart")
    g.add((heart, RDFS.subClassOf, BNode()))

    ancestors = _v()._get_ancestors(g, heart, "class")
    assert ancestors == []


def test_blank_node_not_included_in_output() -> None:
    g = Graph()
    heart = _u("Heart")
    _label(g, heart, "Heart")
    g.add((heart, RDFS.subClassOf, BNode()))

    out = _v().verbalise(g, heart, "class")
    assert _parents_line(out) is None
    assert out == "Label: Heart."


def test_blank_node_deeper_in_chain_stops_traversal() -> None:
    g = Graph()
    heart, organ = _class_chain(g, "Heart", "Organ")
    g.add((organ, RDFS.subClassOf, BNode()))

    ancestors = _v()._get_ancestors(g, heart, "class")
    assert ancestors == [organ]


def test_named_parent_preferred_over_coexisting_blank_node() -> None:
    g = Graph()
    heart, organ = _class_chain(g, "Heart", "Organ")
    g.add((heart, RDFS.subClassOf, BNode()))

    ancestors = _v()._get_ancestors(g, heart, "class")
    assert ancestors == [organ]
    assert _v()._get_siblings(g, heart, "class") == []


def test_blank_node_choice_deterministic_across_reparses() -> None:
    def build() -> tuple[Graph, URIRef]:
        g = Graph()
        heart, organ = _class_chain(g, "Heart", "Organ")
        g.add((heart, RDFS.subClassOf, BNode()))
        return g, heart

    g1, h1 = build()
    g2, h2 = build()
    assert _v().verbalise(g1, h1, "class") == _v().verbalise(g2, h2, "class")
    assert _parents_line(_v().verbalise(g1, h1, "class")) == "Parents: Organ."


# ---------------------------------------------------------------------------
# Sibling discovery
# ---------------------------------------------------------------------------


def _siblings_graph(g: Graph, parent: URIRef, *names: str) -> None:
    _label(g, parent, "Parent")
    for name in names:
        _label(g, _u(name), name)
        g.add((_u(name), RDFS.subClassOf, parent))


def test_sibling_discovery() -> None:
    g = Graph()
    parent = _u("Organ")
    _siblings_graph(g, parent, "Heart", "Lung", "Liver")

    out = _v().verbalise(g, _u("Heart"), "class")
    assert _related_line(out) == "Related types include: Liver, Lung."


def test_current_entity_excluded_from_siblings() -> None:
    g = Graph()
    parent = _u("Organ")
    _siblings_graph(g, parent, "Heart", "Lung")

    siblings = _v()._get_siblings(g, _u("Heart"), "class")
    assert "Heart" not in siblings
    assert siblings == ["Lung"]


def test_sibling_count_capped() -> None:
    g = Graph()
    parent = _u("Organ")
    _siblings_graph(
        g, parent, "Heart", "Lung", "Liver", "Kidney", "Spleen", "Brain", "Skin"
    )

    siblings = _v()._get_siblings(g, _u("Heart"), "class")
    assert len(siblings) == MAX_SIBLINGS


def test_sibling_ordering_deterministic() -> None:
    g = Graph()
    parent = _u("Organ")
    _siblings_graph(g, parent, "Heart", "Zebra", "Apple", "Mango")

    siblings = _v()._get_siblings(g, _u("Heart"), "class")
    assert siblings == ["Apple", "Mango", "Zebra"]


def test_no_siblings_omits_related_line() -> None:
    g = Graph()
    heart, organ = _class_chain(g, "Heart", "Organ")

    out = _v().verbalise(g, heart, "class")
    assert _related_line(out) is None


def test_instance_siblings_share_type() -> None:
    g = Graph()
    organ = _u("Organ")
    _label(g, organ, "Organ")
    for name in ("HeartA", "HeartB", "HeartC"):
        _label(g, _u(name), name)
        g.add((_u(name), RDF.type, organ))

    siblings = _v()._get_siblings(g, _u("HeartA"), "instance")
    assert siblings == ["HeartB", "HeartC"]


def test_predicate_siblings_share_superproperty() -> None:
    g = Graph()
    related = _u("relatedTo")
    _label(g, related, "related to")
    for name in ("partOf", "hasPart", "connectedTo"):
        _label(g, _u(name), name)
        g.add((_u(name), RDFS.subPropertyOf, related))

    siblings = _v()._get_siblings(g, _u("partOf"), "predicate")
    assert siblings == ["connectedTo", "hasPart"]


def test_blank_node_parent_yields_no_siblings() -> None:
    g = Graph()
    heart = _u("Heart")
    _label(g, heart, "Heart")
    g.add((heart, RDFS.subClassOf, BNode()))

    assert _v()._get_siblings(g, heart, "class") == []


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def test_full_output_structure() -> None:
    g = Graph()
    heart, organ = _class_chain(g, "Heart", "Organ")
    for name in ("Lung", "Liver"):
        _label(g, _u(name), name)
        g.add((_u(name), RDFS.subClassOf, organ))

    out = _v().verbalise(g, heart, "class")
    assert out == (
        "Label: Heart.\n\n"
        "Parents: Organ.\n"
        "Related types include: Liver, Lung."
    )


def test_blank_line_separates_annotation_and_hierarchy() -> None:
    g = Graph()
    heart, organ = _class_chain(g, "Heart", "Organ")

    out = _v().verbalise(g, heart, "class")
    assert "\n\n" in out
    annotation, hierarchy = out.split("\n\n", 1)
    assert annotation == "Label: Heart."
    assert hierarchy.startswith("Parents:")


def test_lines_terminated_with_periods() -> None:
    g = Graph()
    heart, organ = _class_chain(g, "Heart", "Organ")
    _label(g, _u("Lung"), "Lung")
    g.add((_u("Lung"), RDFS.subClassOf, organ))

    out = _v().verbalise(g, heart, "class")
    assert _parents_line(out).endswith(".")
    assert _related_line(out).endswith(".")


def test_no_trailing_whitespace() -> None:
    g = Graph()
    heart, organ = _class_chain(g, "Heart", "Organ")
    _label(g, _u("Lung"), "Lung")
    g.add((_u("Lung"), RDFS.subClassOf, organ))

    out = _v().verbalise(g, heart, "class")
    for line in out.splitlines():
        assert line == line.rstrip()


def test_output_deterministic() -> None:
    g = Graph()
    heart, organ, _structure = _class_chain(g, "Heart", "Organ", "Structure")
    for name in ("Lung", "Liver", "Kidney"):
        _label(g, _u(name), name)
        g.add((_u(name), RDFS.subClassOf, organ))

    assert _v().verbalise(g, heart, "class") == _v().verbalise(g, heart, "class")


def test_output_contains_no_raw_uris() -> None:
    g = Graph()
    heart, organ = _class_chain(g, "Heart", "Organ")
    g.add((_u("Lung"), RDFS.subClassOf, organ))

    out = _v().verbalise(g, heart, "class")
    assert "http://" not in out
    assert _NS not in out


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


def _budget_graph() -> tuple[Graph, URIRef]:
    """Class ``Heart`` with a three-level ancestor chain and five siblings."""
    g = Graph()
    heart, organ, body, thing = _class_chain(g, "Heart", "Organ", "Body", "Thing")
    for name in ("Lung", "Liver", "Kidney", "Spleen", "Brain"):
        _label(g, _u(name), name)
        g.add((_u(name), RDFS.subClassOf, organ))
    return g, heart


def test_m3_includes_full_hierarchy() -> None:
    g, heart = _budget_graph()
    out = _v("M3").verbalise(g, heart, "class")
    assert _parents_line(out) == "Parents: Organ -> Body -> Thing."
    assert _related_line(out) == (
        "Related types include: Brain, Kidney, Liver, Lung, Spleen."
    )


def test_m2_budget_never_exceeded() -> None:
    g, heart = _budget_graph()
    out = _v("M2").verbalise(g, heart, "class")
    assert estimate_tokens(out) <= PPAS_BUDGETS["M2"]


def test_siblings_removed_before_ancestors() -> None:
    g, heart = _budget_graph()
    with patch.dict(v7_module.PPAS_BUDGETS, {"M2": 15}):
        out = _v("M2").verbalise(g, heart, "class")
    assert _parents_line(out) == "Parents: Organ -> Body -> Thing."
    assert _related_line(out) is None


def test_ancestor_chain_shortened_when_needed() -> None:
    g, heart = _budget_graph()
    with patch.dict(v7_module.PPAS_BUDGETS, {"M2": 9}):
        out = _v("M2").verbalise(g, heart, "class")
    assert _parents_line(out) == "Parents: Organ -> Body."
    assert "Thing" not in out
    assert _related_line(out) is None


def test_annotation_never_truncated() -> None:
    g, heart = _budget_graph()
    with patch.dict(v7_module.PPAS_BUDGETS, {"M2": 1}):
        out = _v("M2").verbalise(g, heart, "class")
    assert out == "Label: Heart."


def test_budget_uses_estimate_tokens() -> None:
    g, heart = _budget_graph()
    with patch.dict(v7_module.PPAS_BUDGETS, {"M2": 9}):
        with patch.object(
            v7_module, "estimate_tokens", wraps=v7_module.estimate_tokens
        ) as spy:
            _v("M2").verbalise(g, heart, "class")
    assert spy.called


# ---------------------------------------------------------------------------
# Unsupported entity types
# ---------------------------------------------------------------------------


def test_unsupported_entity_type_raises() -> None:
    g = Graph()
    _label(g, _u("Heart"), "Heart")
    with pytest.raises(ValueError, match="relationship"):
        _v().verbalise(g, _u("Heart"), "relationship")


# ---------------------------------------------------------------------------
# Malformed graphs never raise
# ---------------------------------------------------------------------------


def test_empty_graph_returns_annotation_only() -> None:
    g = Graph()
    out = _v().verbalise(g, _u("Heart"), "class")
    assert out == "Label: Heart."


def test_literal_superclass_does_not_raise() -> None:
    g = Graph()
    heart = _u("Heart")
    _label(g, heart, "Heart")
    g.add((heart, RDFS.subClassOf, Literal("not a class")))

    out = _v().verbalise(g, heart, "class")
    assert out == "Label: Heart."


# ---------------------------------------------------------------------------
# Regression -- exports and existing verbalisers unaffected
# ---------------------------------------------------------------------------


def test_export_available() -> None:
    import kgsemembed.verbalisation as verb

    assert "HierarchicalContextVerbaliser" in verb.__all__
    assert verb.HierarchicalContextVerbaliser is HierarchicalContextVerbaliser


def test_existing_verbalisers_still_importable() -> None:
    from kgsemembed.verbalisation import (
        AnnotationVerbaliser,
        LabelVerbaliser,
        NeighbourhoodWalkVerbaliser,
        SchemaAwareVerbaliser,
        StructuredKVVerbaliser,
        TemplateNLVerbaliser,
    )

    assert all(
        cls is not None
        for cls in (
            AnnotationVerbaliser,
            LabelVerbaliser,
            NeighbourhoodWalkVerbaliser,
            SchemaAwareVerbaliser,
            StructuredKVVerbaliser,
            TemplateNLVerbaliser,
        )
    )
