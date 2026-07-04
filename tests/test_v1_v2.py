"""Tests for LabelVerbaliser (V1) and AnnotationVerbaliser (V2)."""

from rdflib import Graph, Literal, URIRef

from kgsemembed.verbalisation.base import (
    DEFINITION_PREDICATES,
    LABEL_PREDICATES,
    SYNONYM_PREDICATES,
)
from kgsemembed.verbalisation.v1 import LabelVerbaliser
from kgsemembed.verbalisation.v2 import AnnotationVerbaliser


# ---------------------------------------------------------------------------
# LabelVerbaliser (V1)
# ---------------------------------------------------------------------------


def _make_graph_with_label_and_synonyms() -> Graph:
    g = Graph()
    ent = URIRef("http://example.org/ont#HeartDisease")
    g.add((ent, LABEL_PREDICATES[0], Literal("Heart Disease", lang="en")))
    g.add((ent, SYNONYM_PREDICATES[0], Literal("cardiac disease", lang="en")))
    g.add((ent, SYNONYM_PREDICATES[1], Literal("heart ailment", lang="en")))
    return g


def test_v1_label_with_synonyms() -> None:
    """Entity with label and two synonyms returns them joined by ' ; '."""
    g = _make_graph_with_label_and_synonyms()
    v = LabelVerbaliser()
    ent = URIRef("http://example.org/ont#HeartDisease")
    result = v.verbalise(g, ent, "class")
    assert result == "Heart Disease ; cardiac disease ; heart ailment"


def test_v1_label_only() -> None:
    """Entity with only rdfs:label returns just the label."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Cancer")
    g.add((ent, LABEL_PREDICATES[0], Literal("Cancer")))
    v = LabelVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert result == "Cancer"


def test_v1_fallback_to_local_name() -> None:
    """Entity with no label returns the local name from its URI."""
    g = Graph()
    ent = URIRef("http://example.org/ont#MyEntity")
    v = LabelVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert result == "My Entity"


def test_v1_fallback_with_synonyms() -> None:
    """Entity with no label but with synonyms uses local name as primary."""
    g = Graph()
    ent = URIRef("http://example.org/ont#MyEntity")
    g.add((ent, SYNONYM_PREDICATES[2], Literal("syn1", lang="en")))
    v = LabelVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert result == "My Entity ; syn1"


def test_v1_prefers_rdfs_label() -> None:
    """rdfs:label is preferred over skos:prefLabel."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    g.add((ent, LABEL_PREDICATES[0], Literal("from rdfs")))
    g.add((ent, LABEL_PREDICATES[1], Literal("from skos")))
    v = LabelVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert result == "from rdfs"


def test_v1_prefers_english_over_untagged() -> None:
    """English label is preferred over untagged."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    g.add((ent, LABEL_PREDICATES[0], Literal("heart disease", lang="en")))
    g.add((ent, LABEL_PREDICATES[0], Literal("disease du coeur")))
    v = LabelVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert result == "heart disease"


def test_v1_deduplicates_synonyms() -> None:
    """Duplicate synonym values across predicates appear only once."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    g.add((ent, LABEL_PREDICATES[0], Literal("Label")))
      # Use obo:hasSynonym (index 2) to avoid skos:altLabel overlapping with LABEL_PREDICATES
    g.add((ent, SYNONYM_PREDICATES[2], Literal("syn", lang="en")))
    g.add((ent, SYNONYM_PREDICATES[3], Literal("syn", lang="en")))
    v = LabelVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert result == "Label ; syn"


def test_v1_ignores_non_english_synonyms() -> None:
    """French synonyms are excluded from output."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    g.add((ent, LABEL_PREDICATES[0], Literal("Label")))
    g.add((ent, SYNONYM_PREDICATES[0], Literal("synonyme", lang="fr")))
    v = LabelVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert result == "Label"


def test_v1_entity_type_ignored() -> None:
    """Output is the same regardless of entity_type value."""
    g = _make_graph_with_label_and_synonyms()
    v = LabelVerbaliser()
    ent = URIRef("http://example.org/ont#HeartDisease")
    result_class = v.verbalise(g, ent, "class")
    result_inst = v.verbalise(g, ent, "instance")
    result_pred = v.verbalise(g, ent, "predicate")
    assert result_class == result_inst == result_pred


def test_v1_output_strips_whitespace() -> None:
    """Leading/trailing whitespace in label is stripped."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    g.add((ent, LABEL_PREDICATES[0], Literal("  Label  ")))
    v = LabelVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert result == "Label"


def test_v1_returns_str_not_none() -> None:
    """verbalise always returns a non-empty string."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Unknown")
    v = LabelVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# AnnotationVerbaliser (V2)
# ---------------------------------------------------------------------------


def _make_graph_with_all_fields() -> Graph:
    g = Graph()
    ent = URIRef("http://example.org/ont#HeartDisease")
    g.add((ent, LABEL_PREDICATES[0], Literal("Heart Disease", lang="en")))
    g.add((ent, DEFINITION_PREDICATES[0], Literal("A cardiovascular condition", lang="en")))
    g.add((ent, SYNONYM_PREDICATES[0], Literal("cardiac disease", lang="en")))
    return g


def test_v2_all_fields() -> None:
    """Entity with label, definition, and synonyms returns all three fields."""
    g = _make_graph_with_all_fields()
    v = AnnotationVerbaliser()
    ent = URIRef("http://example.org/ont#HeartDisease")
    result = v.verbalise(g, ent, "class")
    assert result == "Label: Heart Disease. Definition: A cardiovascular condition. Synonyms: cardiac disease."


def test_v2_label_only() -> None:
    """Entity with only a label returns 'Label: {label}.' with no empty fields."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Cancer")
    g.add((ent, LABEL_PREDICATES[0], Literal("Cancer")))
    v = AnnotationVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert result == "Label: Cancer."


def test_v2_label_and_def() -> None:
    """Entity with label and definition but no synonyms omits Synonyms field."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Diabetes")
    g.add((ent, LABEL_PREDICATES[0], Literal("Diabetes")))
    g.add((ent, DEFINITION_PREDICATES[1], Literal("A metabolic disorder")))
    v = AnnotationVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert result == "Label: Diabetes. Definition: A metabolic disorder."


def test_v2_label_and_syn() -> None:
    """Entity with label and synonyms but no definition omits Definition field."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Asthma")
    g.add((ent, LABEL_PREDICATES[0], Literal("Asthma")))
    g.add((ent, SYNONYM_PREDICATES[2], Literal("bronchial asthma", lang="en")))
    v = AnnotationVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert result == "Label: Asthma. Synonyms: bronchial asthma."


def test_v2_fallback_label() -> None:
    """Entity with no label uses local name fallback."""
    g = Graph()
    ent = URIRef("http://example.org/ont#UnknownEntity")
    v = AnnotationVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert result == "Label: Unknown Entity."


def test_v2_fallback_with_fields() -> None:
    """Entity with no label but with definition and synonyms uses local name for label."""
    g = Graph()
    ent = URIRef("http://example.org/ont#UnknownEntity")
    g.add((ent, DEFINITION_PREDICATES[0], Literal("A condition")))
        # Use obo:hasSynonym to avoid skos:altLabel overlapping with LABEL_PREDICATES
    g.add((ent, SYNONYM_PREDICATES[2], Literal("syn", lang="en")))
    v = AnnotationVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert result == "Label: Unknown Entity. Definition: A condition. Synonyms: syn."


def test_v2_def_order_first_wins() -> None:
    """First definition in DEFINITION_PREDICATES order is used."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    g.add((ent, LABEL_PREDICATES[0], Literal("Test")))
    g.add((ent, DEFINITION_PREDICATES[0], Literal("first def")))
    g.add((ent, DEFINITION_PREDICATES[1], Literal("second def")))
    v = AnnotationVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert "first def" in result
    assert "second def" not in result


def test_v2_syn_separator() -> None:
    """Synonyms are joined with '; ' (semicolon-space)."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    g.add((ent, LABEL_PREDICATES[0], Literal("Test")))
    g.add((ent, SYNONYM_PREDICATES[0], Literal("syn1", lang="en")))
    g.add((ent, SYNONYM_PREDICATES[1], Literal("syn2", lang="en")))
    g.add((ent, SYNONYM_PREDICATES[2], Literal("syn3", lang="en")))
    v = AnnotationVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert "syn1; syn2; syn3" in result


def test_v2_fields_fixed_order() -> None:
    """Fields appear in fixed order: Label, Definition, Synonyms."""
    g = _make_graph_with_all_fields()
    v = AnnotationVerbaliser()
    ent = URIRef("http://example.org/ont#HeartDisease")
    result = v.verbalise(g, ent, "class")
    label_pos = result.index("Label:")
    def_pos = result.index("Definition:")
    syn_pos = result.index("Synonyms:")
    assert label_pos < def_pos < syn_pos


def test_v2_no_empty_fields() -> None:
    """Entity with only synonyms has no Definition field in output."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    g.add((ent, SYNONYM_PREDICATES[0], Literal("syn", lang="en")))
    v = AnnotationVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert "Definition:" not in result
    assert "Label:" in result
    assert "Synonyms:" in result


def test_v2_output_strips_whitespace() -> None:
    """Leading/trailing whitespace in label is stripped."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    g.add((ent, LABEL_PREDICATES[0], Literal("  Label  ")))
    v = AnnotationVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert result == "Label: Label."


def test_v2_entity_type_ignored() -> None:
    """Output is the same regardless of entity_type value."""
    g = _make_graph_with_all_fields()
    v = AnnotationVerbaliser()
    ent = URIRef("http://example.org/ont#HeartDisease")
    result_class = v.verbalise(g, ent, "class")
    result_inst = v.verbalise(g, ent, "instance")
    result_pred = v.verbalise(g, ent, "predicate")
    assert result_class == result_inst == result_pred


def test_v2_deduplicates_synonyms() -> None:
    """Duplicate synonym values across predicates appear only once."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    g.add((ent, LABEL_PREDICATES[0], Literal("Test")))
        # Use obo:hasSynonym to avoid skos:altLabel overlapping with LABEL_PREDICATES
    g.add((ent, SYNONYM_PREDICATES[2], Literal("syn", lang="en")))
    g.add((ent, SYNONYM_PREDICATES[3], Literal("syn", lang="en")))
    v = AnnotationVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert result.count("syn") == 1


def test_v2_multiple_defs_first_wins() -> None:
    """Only the first definition is used when multiple exist."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    g.add((ent, LABEL_PREDICATES[0], Literal("Test")))
    g.add((ent, DEFINITION_PREDICATES[0], Literal("first")))
    g.add((ent, DEFINITION_PREDICATES[1], Literal("second")))
    g.add((ent, DEFINITION_PREDICATES[2], Literal("third")))
    v = AnnotationVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert "first" in result
    assert "second" not in result
    assert "third" not in result


def test_v2_returns_str_not_none() -> None:
    """verbalise always returns a non-empty string."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Unknown")
    v = AnnotationVerbaliser()
    result = v.verbalise(g, ent, "class")
    assert isinstance(result, str)
    assert len(result) > 0
