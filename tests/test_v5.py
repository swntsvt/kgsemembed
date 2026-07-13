"""Tests for NeighbourhoodWalkVerbaliser (V5)."""

from unittest.mock import patch

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from kgsemembed.verbalisation import NeighbourhoodWalkVerbaliser
from kgsemembed.verbalisation import v5 as v5_module
from kgsemembed.verbalisation.v5 import (
    RANDOM_SEED,
    WALK_DEPTH,
    WALKS_PER_ENTITY,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_NS = "http://example.org/ont#"
_HEART = URIRef(f"{_NS}Heart")


def _u(name: str) -> URIRef:
    return URIRef(f"{_NS}{name}")


def _label(g: Graph, uri: URIRef, text: str) -> None:
    g.add((uri, RDFS.label, Literal(text, lang="en")))


_SEP = " | "


def _v(model_key: str = "M2") -> NeighbourhoodWalkVerbaliser:
    return NeighbourhoodWalkVerbaliser(model_key=model_key)


def _walks(text: str) -> list[str]:
    return text.split(_SEP)


def _chain(*labels: str) -> tuple[Graph, list[URIRef]]:
    """Build a single linear chain ``n0 -[linkedTo]-> n1 -> ...`` with labels."""
    g = Graph()
    link = _u("linkedTo")
    _label(g, link, "linked to")
    nodes = [_u(name.replace(" ", "")) for name in labels]
    for node, text in zip(nodes, labels):
        _label(g, node, text)
    for src, dst in zip(nodes, nodes[1:]):
        g.add((src, link, dst))
    return g, nodes


# ---------------------------------------------------------------------------
# Constants -- part of the experimental protocol
# ---------------------------------------------------------------------------


def test_protocol_constants() -> None:
    assert WALK_DEPTH == 4
    assert WALKS_PER_ENTITY == 10
    assert RANDOM_SEED == 42


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_repeated_calls_are_identical() -> None:
    g = Graph()
    for name in ("A", "B", "C", "D"):
        _label(g, _u(name), name)
        g.add((_HEART, _u("rel"), _u(name)))
        g.add((_u(name), _u("rel"), _HEART))
    _label(g, _HEART, "Heart")

    out1 = _v().verbalise(g, _HEART, "instance")
    out2 = _v().verbalise(g, _HEART, "instance")
    assert out1 == out2


def test_determinism_independent_of_prior_random_state() -> None:
    import random

    g, _ = _chain("Heart", "Cardiovascular system", "Blood vessel")
    baseline = _v().verbalise(g, _HEART, "instance")

    random.seed(999)
    random.random()  # perturb global RNG state
    assert _v().verbalise(g, _HEART, "instance") == baseline


def test_seed_called_exactly_once_per_call() -> None:
    g, _ = _chain("Heart", "Lung")
    with patch.object(v5_module.random, "seed", wraps=v5_module.random.seed) as spy:
        _v().verbalise(g, _HEART, "instance")
    spy.assert_called_once_with(RANDOM_SEED)


def test_seed_called_once_even_for_empty_neighbourhood() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    with patch.object(v5_module.random, "seed", wraps=v5_module.random.seed) as spy:
        _v().verbalise(g, _HEART, "instance")
    spy.assert_called_once_with(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Walk generation
# ---------------------------------------------------------------------------


def test_generates_fixed_number_of_walks() -> None:
    g, _ = _chain("Heart", "Cardiovascular system", "Blood vessel")
    out = _v("M3").verbalise(g, _HEART, "instance")
    assert len(_walks(out)) == WALKS_PER_ENTITY


def test_walk_respects_maximum_depth() -> None:
    labels = ["N0", "N1", "N2", "N3", "N4", "N5", "N6"]
    g, nodes = _chain(*labels)
    walk = _v()._single_walk(g, nodes[0], set())
    assert len(walk) == WALK_DEPTH


def test_walk_terminates_before_maximum_depth() -> None:
    g, nodes = _chain("Heart", "Lung")
    walk = _v()._single_walk(g, nodes[0], set())
    assert len(walk) == 1


def test_correct_node_and_predicate_sequence() -> None:
    g, _ = _chain("Heart", "Cardiovascular system", "Blood vessel")
    out = _walks(_v("M3").verbalise(g, _HEART, "instance"))[0]
    assert out == "Heart [linked to] Cardiovascular system [linked to] Blood vessel"


def test_bracket_notation_around_predicate() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    _label(g, _u("Cardiovascular"), "Cardiovascular system")
    part_of = _u("partOf")
    _label(g, part_of, "part of")
    g.add((_HEART, part_of, _u("Cardiovascular")))

    line = _walks(_v("M3").verbalise(g, _HEART, "instance"))[0]
    assert line == "Heart [part of] Cardiovascular system"


def test_predicate_falls_back_to_local_name() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    _label(g, _u("Vessel"), "Vessel")
    g.add((_HEART, _u("part_of"), _u("Vessel")))

    line = _walks(_v("M3").verbalise(g, _HEART, "instance"))[0]
    assert line == "Heart [part of] Vessel"


def test_node_falls_back_to_local_name() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    part_of = _u("partOf")
    _label(g, part_of, "part of")
    g.add((_HEART, part_of, _u("CardiovascularSystem")))

    line = _walks(_v("M3").verbalise(g, _HEART, "instance"))[0]
    assert line == "Heart [part of] Cardiovascular System"


def test_walks_joined_with_pipe_separator() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    _label(g, _u("Organ"), "Organ")
    _label(g, _u("Vessel"), "Vessel")
    g.add((_HEART, _u("rel"), _u("Organ")))
    g.add((_HEART, _u("rel"), _u("Vessel")))
    _label(g, _u("rel"), "rel")

    out = _v("M3").verbalise(g, _HEART, "instance")
    assert " | " in out
    assert len(out.split(" | ")) == WALKS_PER_ENTITY


def test_output_contains_no_raw_uris() -> None:
    g, _ = _chain("Heart", "Cardiovascular system", "Blood vessel")
    out = _v("M3").verbalise(g, _HEART, "instance")
    assert "http://" not in out
    assert _NS not in out


# ---------------------------------------------------------------------------
# Predicate filtering
# ---------------------------------------------------------------------------


def test_owl_sameas_excluded() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    g.add((_HEART, OWL.sameAs, _u("Other")))
    _label(g, _u("Other"), "Other")

    assert _v().verbalise(g, _HEART, "instance") == "Heart"


def test_owl_disjointwith_excluded() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    g.add((_HEART, OWL.disjointWith, _u("Bone")))
    _label(g, _u("Bone"), "Bone")

    assert _v().verbalise(g, _HEART, "instance") == "Heart"


def test_rdf_type_excluded_for_class_entities() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    g.add((_HEART, RDF.type, OWL.Class))

    assert _v().verbalise(g, _HEART, "class") == "Heart"


def test_rdf_type_traversed_for_non_class_entities() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    _label(g, _u("Organ"), "Organ")
    g.add((_HEART, RDF.type, _u("Organ")))

    line = _walks(_v("M3").verbalise(g, _HEART, "instance"))[0]
    assert line == "Heart [type] Organ"


def test_exclusions_apply_at_every_hop() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    _label(g, _u("Organ"), "Organ")
    g.add((_HEART, _u("rel"), _u("Organ")))
    _label(g, _u("rel"), "rel")
    g.add((_u("Organ"), RDF.type, _u("Deeper")))  # class exclusion at hop 2

    line = _walks(_v("M3").verbalise(g, _HEART, "class"))[0]
    assert line == "Heart [rel] Organ"


# ---------------------------------------------------------------------------
# Traversal -- only URIRef nodes
# ---------------------------------------------------------------------------


def test_literals_never_traversed() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    g.add((_HEART, _u("mass"), Literal("300g")))

    assert _v().verbalise(g, _HEART, "instance") == "Heart"


def test_blank_nodes_never_traversed() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    g.add((_HEART, _u("restriction"), BNode()))

    assert _v().verbalise(g, _HEART, "instance") == "Heart"


def test_only_uriref_neighbours_selected() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    _label(g, _u("Vessel"), "Vessel")
    g.add((_HEART, _u("rel"), Literal("noise")))
    g.add((_HEART, _u("rel"), BNode()))
    g.add((_HEART, _u("rel"), _u("Vessel")))

    steps = _v()._get_outgoing_triples(g, _HEART, set())
    assert steps == [(_u("rel"), _u("Vessel"))]


# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------


def _deep_graph() -> Graph:
    """Depth >= WALK_DEPTH chain with verbose labels to inflate token cost."""
    g = Graph()
    link = _u("isCloselyAssociatedWith")
    _label(g, link, "is closely associated with the neighbouring")
    names = [
        "Heart muscle tissue region",
        "Cardiovascular circulatory system network",
        "Major blood vessel structure",
        "Oxygenated red blood cell",
        "Cellular oxygen transport unit",
    ]
    nodes = [_HEART] + [_u(f"Node{i}") for i in range(1, len(names))]
    for node, text in zip(nodes, names):
        _label(g, node, text)
    for src, dst in zip(nodes, nodes[1:]):
        g.add((src, link, dst))
    return g


def test_m3_includes_all_walks() -> None:
    g = _deep_graph()
    out = _v("M3").verbalise(g, _HEART, "instance")
    assert len(_walks(out)) == WALKS_PER_ENTITY


def test_budget_never_exceeded() -> None:
    from kgsemembed.verbalisation.ppas import PPAS_BUDGETS, estimate_tokens

    g = _deep_graph()
    out = _v("M1").verbalise(g, _HEART, "instance")
    assert estimate_tokens(out) <= PPAS_BUDGETS["M1"]


def test_budget_truncates_relative_to_unbounded() -> None:
    g = _deep_graph()
    capped = _walks(_v("M1").verbalise(g, _HEART, "instance"))
    uncapped = _walks(_v("M3").verbalise(g, _HEART, "instance"))
    assert 0 < len(capped) < len(uncapped) == WALKS_PER_ENTITY


def test_budget_uses_estimate_tokens() -> None:
    g = _deep_graph()
    with patch.object(
        v5_module, "estimate_tokens", wraps=v5_module.estimate_tokens
    ) as spy:
        _v("M1").verbalise(g, _HEART, "instance")
    assert spy.called


# ---------------------------------------------------------------------------
# Empty neighbourhood / edge cases
# ---------------------------------------------------------------------------


def test_empty_graph_returns_local_name() -> None:
    g = Graph()
    assert _v().verbalise(g, _HEART, "instance") == "Heart"


def test_no_outgoing_triples_returns_label() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    assert _v().verbalise(g, _HEART, "instance") == "Heart"


def test_only_literal_targets_returns_label() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    g.add((_HEART, _u("mass"), Literal("300g")))
    g.add((_HEART, _u("colour"), Literal("red")))
    assert _v().verbalise(g, _HEART, "instance") == "Heart"


def test_only_skipped_predicates_returns_label() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    g.add((_HEART, OWL.sameAs, _u("A")))
    g.add((_HEART, OWL.disjointWith, _u("B")))
    assert _v().verbalise(g, _HEART, "instance") == "Heart"


def test_self_loop_terminates() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    g.add((_HEART, _u("rel"), _HEART))
    _label(g, _u("rel"), "rel")

    line = _walks(_v("M3").verbalise(g, _HEART, "instance"))[0]
    assert line == "Heart [rel] Heart [rel] Heart [rel] Heart [rel] Heart"


def test_cyclic_graph_is_depth_bounded() -> None:
    g = Graph()
    _label(g, _HEART, "Heart")
    _label(g, _u("Lung"), "Lung")
    g.add((_HEART, _u("rel"), _u("Lung")))
    g.add((_u("Lung"), _u("rel"), _HEART))
    _label(g, _u("rel"), "rel")

    walk = _v()._single_walk(g, _HEART, set())
    assert len(walk) == WALK_DEPTH


# ---------------------------------------------------------------------------
# Regression -- exports and existing verbalisers unaffected
# ---------------------------------------------------------------------------


def test_export_available() -> None:
    import kgsemembed.verbalisation as verb

    assert "NeighbourhoodWalkVerbaliser" in verb.__all__
    assert verb.NeighbourhoodWalkVerbaliser is NeighbourhoodWalkVerbaliser


def test_existing_verbalisers_still_importable() -> None:
    from kgsemembed.verbalisation import (
        AnnotationVerbaliser,
        LabelVerbaliser,
        SchemaAwareVerbaliser,
        StructuredKVVerbaliser,
        TemplateNLVerbaliser,
    )

    assert all(
        cls is not None
        for cls in (
            AnnotationVerbaliser,
            LabelVerbaliser,
            SchemaAwareVerbaliser,
            StructuredKVVerbaliser,
            TemplateNLVerbaliser,
        )
    )


def test_default_model_key_is_m2() -> None:
    assert NeighbourhoodWalkVerbaliser().model_key == "M2"
