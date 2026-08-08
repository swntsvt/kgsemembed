"""Tests for StructuredKVVerbaliser (V4)."""

import logging
from contextlib import contextmanager
from typing import Iterator
from unittest.mock import patch

from rdflib import BNode, Graph, Literal, URIRef

from kgsemembed.verbalisation.base import LABEL_PREDICATES
from kgsemembed.verbalisation.v4 import StructuredKVVerbaliser
from kgsemembed.verbalisation.ppas import (
    PPAS_BUDGETS,
    should_apply_ppas,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENT = URIRef("http://example.org/ont#Aspirin")
_DRUG = URIRef("http://example.org/ont#Drug")
_CHEM = URIRef("http://example.org/ont#ChemicalCompound")
_MOLECULE = URIRef("http://example.org/ont#MolecularFormula")
_CAS = URIRef("http://example.org/ont#CASNumber")
_AUTHOR = URIRef("http://example.org/ont#author")
_SAME_AS = URIRef("http://www.w3.org/2002/07/owl#sameAs")
_SELF = URIRef("http://example.org/ont#SelfRef")
_BLANK = BNode()


def _make_graph() -> Graph:
    """Return a graph with a typical instance entity."""
    g = Graph()
    g.add(
        (_ENT, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _DRUG)
    )
    g.add(
        (_ENT, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _CHEM)
    )
    g.add((_ENT, _CAS, Literal("50-78-2")))
    g.add((_ENT, _MOLECULE, Literal("C9H8O4")))
    return g


def _add_labels(g: Graph) -> Graph:
    """Add rdfs:labels to the entity and its URIRef objects."""
    g.add((_ENT, LABEL_PREDICATES[0], Literal("Aspirin", lang="en")))
    g.add((_DRUG, LABEL_PREDICATES[0], Literal("Drug", lang="en")))
    g.add((_CHEM, LABEL_PREDICATES[0], Literal("Chemical Compound", lang="en")))
    g.add(
        (_MOLECULE, LABEL_PREDICATES[0], Literal("Molecular Formula", lang="en"))
    )
    g.add((_CAS, LABEL_PREDICATES[0], Literal("CAS number", lang="en")))
    return g


# ---------------------------------------------------------------------------
# Core Behaviour
# ---------------------------------------------------------------------------


def test_normal_serialisation() -> None:
    """Entity with multiple predicates produces pipe-separated key-value pairs."""
    g = _make_graph()
    _add_labels(g)
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    assert isinstance(result, str)
    assert len(result) > 0
    assert " | " in result
    assert "type: Drug" in result
    assert "CAS number: 50-78-2" in result


def test_datatype_properties() -> None:
    """Literal (datatype) property values are serialised directly."""
    g = Graph()
    g.add((_ENT, _CAS, Literal("123-45-6")))
    _add_labels(g)
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    assert "CAS number: 123-45-6" in result


def test_uri_valued_properties() -> None:
    """URI-valued properties resolve to readable labels, not raw URIs."""
    g = Graph()
    g.add(
        (_ENT, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _DRUG)
    )
    _add_labels(g)
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    assert "Drug" in result
    assert "http://example.org/ont#Drug" not in result


def test_rdf_type_serialisation() -> None:
    """rdf:type declarations use the 'type' key."""
    g = Graph()
    g.add(
        (_ENT, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _DRUG)
    )
    g.add(
        (_ENT, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _CHEM)
    )
    _add_labels(g)
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    assert result.count("type:") == 2
    assert "type: Drug" in result
    assert "type: Chemical Compound" in result


def test_repeated_predicates() -> None:
    """Multiple values for one predicate repeat the predicate key."""
    g = Graph()
    author1 = URIRef("http://example.org/person/Alice")
    author2 = URIRef("http://example.org/person/Bob")
    g.add((_ENT, _AUTHOR, author1))
    g.add((_ENT, _AUTHOR, author2))
    g.add((_ENT, LABEL_PREDICATES[0], Literal("Paper", lang="en")))
    g.add((author1, LABEL_PREDICATES[0], Literal("Alice", lang="en")))
    g.add((author2, LABEL_PREDICATES[0], Literal("Bob", lang="en")))
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    assert result.count("author:") == 2
    assert "Alice" in result
    assert "Bob" in result


def test_predicate_label_resolution() -> None:
    """Predicate rdfs:label is used when available."""
    g = Graph()
    pred = URIRef("http://example.org/ont#pubDate")
    g.add((_ENT, pred, Literal("2024")))
    g.add((_ENT, LABEL_PREDICATES[0], Literal("Entity", lang="en")))
    g.add((pred, LABEL_PREDICATES[0], Literal("Publication date", lang="en")))
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    assert "Publication date: 2024" in result


def test_predicate_local_name_fallback() -> None:
    """When predicate has no label, local name from URI is used."""
    g = Graph()
    pred = URIRef("http://example.org/ont#customProp")
    g.add((_ENT, pred, Literal("value")))
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    # get_local_name converts camelCase -> spaces
    assert "custom Prop: value" in result


def test_object_label_resolution() -> None:
    """URIRef objects resolve via get_label_or_local (label then local name)."""
    g = Graph()
    g.add(
        (_ENT, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _DRUG)
    )
    # No label for _DRUG -- should fall back to local name
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    assert "type: Drug" in result


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_owl_sameas_excluded() -> None:
    """owl:sameAs triples are not included in the output."""
    g = Graph()
    g.add(
        (_ENT, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _DRUG)
    )
    g.add((_ENT, _SAME_AS, URIRef("http://dbpedia.org/resource/Aspirin")))
    _add_labels(g)
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    assert "sameAs" not in result
    assert "dbpedia.org" not in result


def test_blank_nodes_excluded() -> None:
    """Blank-node objects are silently excluded without raising."""
    g = Graph()
    g.add(
        (_ENT, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _DRUG)
    )
    g.add((_ENT, _MOLECULE, _BLANK))
    _add_labels(g)
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    assert "type: Drug" in result
    assert _BLANK.n3() not in result


def test_reflexive_triples_excluded() -> None:
    """Triples where object equals subject are excluded."""
    g = Graph()
    g.add(
        (_SELF, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _DRUG)
    )
    g.add((_SELF, _SELF, _SELF))
    g.add((_SELF, LABEL_PREDICATES[0], Literal("Self", lang="en")))
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _SELF, "instance")
    assert "Self" in result
    # The reflexive triple should not appear
    assert result.count("Self") == 1


# ---------------------------------------------------------------------------
# PPAS
# ---------------------------------------------------------------------------


def test_ppas_invoked_when_above_threshold_not_m3() -> None:
    """PPAS is applied when triple count > threshold and model is not M3."""
    g = Graph()
    g.add((_ENT, LABEL_PREDICATES[0], Literal("Entity", lang="en")))
    for i in range(25):
        pred = URIRef(f"http://example.org/ont#prop_{i}")
        g.add((_ENT, pred, Literal(f"value_{i}")))
        g.add((pred, LABEL_PREDICATES[0], Literal(f"Prop {i}", lang="en")))

    v = StructuredKVVerbaliser(model_key="M2")

    with patch(
        "kgsemembed.verbalisation.v4.should_apply_ppas", return_value=True
    ) as mock_apply:
        with patch(
            "kgsemembed.verbalisation.v4.ppas_sample",
            side_effect=lambda triples, tier_list, budget, fn: triples[:10],
        ) as mock_sample:
            v.verbalise(g, _ENT, "instance")
            mock_apply.assert_called_once()
            mock_sample.assert_called_once()


def test_ppas_not_invoked_when_below_threshold() -> None:
    """PPAS is not applied when triple count is at or below threshold."""
    g = Graph()
    g.add((_ENT, LABEL_PREDICATES[0], Literal("Entity", lang="en")))
    for i in range(15):
        pred = URIRef(f"http://example.org/ont#prop_{i}")
        g.add((_ENT, pred, Literal(f"value_{i}")))

    v = StructuredKVVerbaliser(model_key="M2")

    # Verify real should_apply_ppas returns False for 15 triples
    triples = list(g.triples((_ENT, None, None)))
    assert should_apply_ppas(triples, "M2") is False

    with patch("kgsemembed.verbalisation.v4.ppas_sample") as mock_sample:
        v.verbalise(g, _ENT, "instance")
        mock_sample.assert_not_called()


def test_ppas_not_invoked_for_m3() -> None:
    """M3 model never triggers PPAS regardless of triple count."""
    g = Graph()
    g.add((_ENT, LABEL_PREDICATES[0], Literal("Entity", lang="en")))
    for i in range(50):
        pred = URIRef(f"http://example.org/ont#prop_{i}")
        g.add((_ENT, pred, Literal(f"value_{i}")))

    v = StructuredKVVerbaliser(model_key="M3")
    # M3 has no budget cap, so should_apply_ppas returns False
    assert should_apply_ppas([URIRef("s")] * 100, "M3") is False
    result = v.verbalise(g, _ENT, "instance")
    assert len(result) > 0


def test_ppas_preserves_order_after_sampling() -> None:
    """PPAS-sampled triples preserve their original order."""
    g = Graph()
    g.add((_ENT, LABEL_PREDICATES[0], Literal("Entity", lang="en")))
    for i in range(25):
        pred = URIRef(f"http://example.org/ont#prop_{i}")
        g.add((_ENT, pred, Literal(f"value_{i}")))
        g.add((pred, LABEL_PREDICATES[0], Literal(f"Prop {i}", lang="en")))

    v = StructuredKVVerbaliser(model_key="M2")
    sampled = []

    def _track_sample(triples, tier_list, budget, fn):
        sampled.extend(triples)
        return triples

    with patch(
        "kgsemembed.verbalisation.v4.should_apply_ppas", return_value=True
    ):
        with patch(
            "kgsemembed.verbalisation.v4.ppas_sample", side_effect=_track_sample
        ):
            v.verbalise(g, _ENT, "instance")

    assert len(sampled) > 0


def test_ppas_never_applied_for_m3_with_many_triples() -> None:
    """M3 should never invoke PPAS even with many triples."""
    g = Graph()
    g.add((_ENT, LABEL_PREDICATES[0], Literal("Entity", lang="en")))
    for i in range(100):
        pred = URIRef(f"http://example.org/ont#prop_{i}")
        g.add((_ENT, pred, Literal(f"value_{i}")))

    v = StructuredKVVerbaliser(model_key="M3")
    assert PPAS_BUDGETS["M3"] is None
    assert should_apply_ppas([URIRef("s")] * 100, "M3") is False


def test_more_than_twenty_triples() -> None:
    """Entity with more than 20 triples triggers PPAS for non-M3 models."""
    g = Graph()
    g.add((_ENT, LABEL_PREDICATES[0], Literal("Entity", lang="en")))
    for i in range(25):
        pred = URIRef(f"http://example.org/ont#p{i}")
        g.add((_ENT, pred, Literal(f"v{i}")))

    # Verify real should_apply_ppas returns True for 25 triples
    triples = list(g.triples((_ENT, None, None)))
    assert should_apply_ppas(triples, "M1") is True

    v = StructuredKVVerbaliser(model_key="M1")
    with patch("kgsemembed.verbalisation.v4.ppas_sample") as mock_sample:
        v.verbalise(g, _ENT, "instance")
        mock_sample.assert_called_once()


def test_exactly_twenty_triples() -> None:
    """Exactly 20 triples does NOT trigger PPAS (at threshold)."""
    g = Graph()
     # 1 label + 19 properties = 20 total (at threshold, not above)
    g.add((_ENT, LABEL_PREDICATES[0], Literal("Entity", lang="en")))
    for i in range(19):
        pred = URIRef(f"http://example.org/ont#p{i}")
        g.add((_ENT, pred, Literal(f"v{i}")))

     # Verify real should_apply_ppas returns False for 20 triples
    triples = list(g.triples((_ENT, None, None)))
    assert len(triples) == 20
    assert should_apply_ppas(triples, "M1") is False

    v = StructuredKVVerbaliser(model_key="M1")
    with patch("kgsemembed.verbalisation.v4.ppas_sample") as mock_sample:
        result = v.verbalise(g, _ENT, "instance")
        mock_sample.assert_not_called()
        assert len(result) > 0


# ---------------------------------------------------------------------------
# PPAS Fallback for Untiered Predicates
# ---------------------------------------------------------------------------


def _make_untiered_graph(triple_count: int = 49) -> Graph:
    """Return a graph whose entity uses only predicates outside the tiers."""
    g = Graph()
    for i in range(triple_count):
        pred = URIRef(f"http://dbpedia.org/ontology/combatant_{i}")
        g.add((_ENT, pred, Literal(f"Force {i}")))
    return g


@contextmanager
def _capture_warnings() -> Iterator[list[logging.LogRecord]]:
    """Collect warnings from the V4 logger, bypassing global log config.

    ``init_logging`` disables propagation on the ``kgsemembed`` logger, so
    pytest's ``caplog`` fixture cannot see these records once any earlier test
    has initialised logging.
    """
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("kgsemembed.verbalisation.v4")
    handler = _Collector(level=logging.WARNING)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def test_fallback_when_all_predicates_untiered() -> None:
    """Entity with 49 untiered triples still produces a non-empty string."""
    g = _make_untiered_graph()
    triples = list(g.triples((_ENT, None, None)))
    assert len(triples) == 49
    assert should_apply_ppas(triples, "M2") is True

    v = StructuredKVVerbaliser(model_key="M2")
    result = v.verbalise(g, _ENT, "instance")

    assert result != ""
    assert result.count(" | ") == 48
    assert "Force 0" in result


def test_fallback_logs_warning() -> None:
    """A warning is logged whenever the fallback is triggered."""
    g = _make_untiered_graph()
    v = StructuredKVVerbaliser(model_key="M2")

    with _capture_warnings() as records:
        v.verbalise(g, _ENT, "instance")

    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert str(_ENT) in records[0].getMessage()
    assert "49" in records[0].getMessage()


def test_no_fallback_when_predicates_tiered() -> None:
    """Entity with tiered predicates keeps PPAS output and logs no warning."""
    g = Graph()
    g.add((_ENT, LABEL_PREDICATES[0], Literal("Aspirin", lang="en")))
    for i in range(30):
        g.add(
            (
                _ENT,
                URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"),
                URIRef(f"http://example.org/ont#Class_{i}"),
            )
        )

    v = StructuredKVVerbaliser(model_key="M2")

    with _capture_warnings() as records:
        result = v.verbalise(g, _ENT, "instance")

    assert "label: Aspirin" in result
    assert "type: Class 0" in result
    assert records == []


def test_fallback_preserves_ppas_selection_when_non_empty() -> None:
    """A non-empty PPAS result is used verbatim, never replaced by fallback."""
    g = _make_untiered_graph()
    v = StructuredKVVerbaliser(model_key="M2")

    with patch(
        "kgsemembed.verbalisation.v4.ppas_sample",
        side_effect=lambda t, tier_list, budget, fn: t[:3],
    ):
        with _capture_warnings() as records:
            result = v.verbalise(g, _ENT, "instance")

    assert result.count(" | ") == 2
    assert records == []


def test_fallback_output_matches_unsampled_verbalisation() -> None:
    """Fallback output equals the output produced with PPAS disabled."""
    g = _make_untiered_graph()

    fallback = StructuredKVVerbaliser(model_key="M2").verbalise(
        g, _ENT, "instance"
    )
    unsampled = StructuredKVVerbaliser(model_key="M3").verbalise(
        g, _ENT, "instance"
    )

    assert fallback == unsampled


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


def test_only_rdf_type() -> None:
    """Entity with only rdf:type produces valid non-empty output."""
    g = Graph()
    g.add(
        (_ENT, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), _DRUG)
    )
    g.add((_DRUG, LABEL_PREDICATES[0], Literal("Drug", lang="en")))
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    assert result == "type: Drug"


def test_predicate_without_label() -> None:
    """Predicate with no label falls back to local name from URI."""
    g = Graph()
    pred = URIRef("http://example.org/ont#customProp")
    g.add((_ENT, pred, Literal("val")))
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    # get_local_name converts camelCase -> spaces
    assert "custom Prop: val" in result


def test_uri_object_without_label() -> None:
    """URI object with no label falls back to local name from URI."""
    g = Graph()
    target = URIRef("http://example.org/ont#SomeClass")
    g.add(
        (_ENT, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), target)
    )
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    assert "type: Some Class" in result


def test_literal_object() -> None:
    """Literal objects are serialised as their string value."""
    g = Graph()
    g.add((_ENT, _CAS, Literal("99-99-9")))
    g.add((_ENT, LABEL_PREDICATES[0], Literal("Entity", lang="en")))
    g.add((_CAS, LABEL_PREDICATES[0], Literal("CAS", lang="en")))
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    assert "CAS: 99-99-9" in result


def test_empty_graph() -> None:
    """Entity in empty graph returns empty string."""
    g = Graph()
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    assert result == ""


def test_entity_with_minimal_info() -> None:
    """Entity with only a label returns valid output (label key-value)."""
    g = Graph()
    g.add((_ENT, LABEL_PREDICATES[0], Literal("Minimal", lang="en")))
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    assert "label: Minimal" in result


def test_duplicate_predicate_labels() -> None:
    """Multiple predicates with the same rdfs:label produce duplicate keys."""
    g = Graph()
    pred_a = URIRef("http://example.org/ont#a")
    pred_b = URIRef("http://example.org/ont#b")
    g.add((_ENT, pred_a, Literal("val1")))
    g.add((_ENT, pred_b, Literal("val2")))
    g.add((_ENT, LABEL_PREDICATES[0], Literal("Entity", lang="en")))
    g.add((pred_a, LABEL_PREDICATES[0], Literal("SameLabel", lang="en")))
    g.add((pred_b, LABEL_PREDICATES[0], Literal("SameLabel", lang="en")))
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    assert result.count("SameLabel:") == 2


def test_duplicate_object_values() -> None:
    """Duplicate object values for different predicates are both included."""
    g = Graph()
    g.add((_ENT, _CAS, Literal("50-78-2")))
    g.add((_ENT, _MOLECULE, Literal("50-78-2")))
    g.add((_ENT, LABEL_PREDICATES[0], Literal("Entity", lang="en")))
    g.add((_CAS, LABEL_PREDICATES[0], Literal("CAS", lang="en")))
    g.add((_MOLECULE, LABEL_PREDICATES[0], Literal("Mol", lang="en")))
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    assert result.count("50-78-2") == 2


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_existing_verbalisers_unaffected() -> None:
    """V1 and V2 still work correctly after adding V4."""
    from kgsemembed.verbalisation.v1 import LabelVerbaliser
    from kgsemembed.verbalisation.v2 import AnnotationVerbaliser

    g = Graph()
    g.add((_ENT, LABEL_PREDICATES[0], Literal("Test", lang="en")))

    v1 = LabelVerbaliser()
    assert v1.verbalise(g, _ENT, "instance") == "Test"

    v2 = AnnotationVerbaliser()
    assert v2.verbalise(g, _ENT, "instance") == "Label: Test."


def test_exports_correct() -> None:
    """StructuredKVVerbaliser is exported from the package-level __init__."""
    from kgsemembed.verbalisation import StructuredKVVerbaliser as PKG_SV4
    from kgsemembed.verbalisation.v4 import (
        StructuredKVVerbaliser as ModSV4,
    )

    assert PKG_SV4 is ModSV4


def test_repository_conventions() -> None:
    """V4 follows repository naming and typing conventions."""
    import inspect

    sig = inspect.signature(StructuredKVVerbaliser.__init__)
    assert "model_key" in sig.parameters
    assert sig.parameters["model_key"].default == "M2"

    sig_v = inspect.signature(StructuredKVVerbaliser.verbalise)
    assert "graph" in sig_v.parameters
    assert "entity_uri" in sig_v.parameters
    assert "entity_type" in sig_v.parameters


def test_deterministic_output() -> None:
    """Same input always produces the same output."""
    g = _make_graph()
    _add_labels(g)
    v = StructuredKVVerbaliser()
    result1 = v.verbalise(g, _ENT, "instance")
    result2 = v.verbalise(g, _ENT, "instance")
    assert result1 == result2


def test_pipe_separator_format() -> None:
    """Output uses ' | ' (space-pipe-space) as separator."""
    g = _make_graph()
    _add_labels(g)
    v = StructuredKVVerbaliser()
    result = v.verbalise(g, _ENT, "instance")
    pairs = result.split(" | ")
    assert len(pairs) > 1
    for pair in pairs:
        assert ": " in pair
