"""Tests for LabelVerbaliser (V1) and AnnotationVerbaliser (V2)."""

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import XSD

from kgsemembed.verbalisation.base import (
    DEFINITION_PREDICATES,
    LABEL_PREDICATES,
    SYNONYM_PREDICATES,
    annotation_predicates_digest,
)
from kgsemembed.verbalisation.predicate_roles import get_role_map
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


# ---------------------------------------------------------------------------
# D1/D2 annotation predicates (#136)
# ---------------------------------------------------------------------------

_FMA = "http://purl.org/sig/ont/fma/"
_OBO_IN_OWL = "http://www.geneontology.org/formats/oboInOwl#"
_RDFS_LABEL = URIRef("http://www.w3.org/2000/01/rdf-schema#label")


def _roles_v1_predicates(role: str) -> set[str]:
    roles = get_role_map("roles_v1").roles
    return {predicate for predicate, assigned in roles.items() if assigned == role}


def _fma_class() -> tuple[Graph, URIRef]:
    g = Graph()
    ent = URIRef("http://purl.org/sig/ont/fma.owl#fma7088")
    g.add((ent, _RDFS_LABEL, Literal("Heart", lang="en")))
    g.add((ent, URIRef(f"{_FMA}synonym"), Literal("Cor", datatype=XSD.string)))
    g.add((ent, URIRef(f"{_FMA}definition"), Literal("Organ pumping blood", datatype=XSD.string)))
    return g, ent


def _anatomy_class() -> tuple[Graph, URIRef]:
    g = Graph()
    ent = URIRef("http://human.owl#NCI_C12611")
    g.add((ent, _RDFS_LABEL, Literal("Antibody_Producing_Cell", datatype=XSD.string)))
    for node, predicate, text in [
        ("genid1", "hasRelatedSynonym", "Immunoglobulin-Producing Cells"),
        ("genid2", "hasRelatedSynonym", "Plasma Cell"),
        ("genid3", "hasDefinition", "A cell that secretes antibodies"),
    ]:
        node_uri = URIRef(f"http://human.owl#{node}")
        g.add((ent, URIRef(f"{_OBO_IN_OWL}{predicate}"), node_uri))
        g.add((node_uri, _RDFS_LABEL, Literal(text, datatype=XSD.string)))
    return g, ent


def test_synonym_predicates_equal_the_roles_v1_synonym_role() -> None:
    """V1/V2 read exactly the predicates the #131 role map calls synonyms."""
    assert {str(p) for p in SYNONYM_PREDICATES} == _roles_v1_predicates("synonym")


def test_definition_predicates_equal_the_roles_v1_definition_role() -> None:
    """V2 reads exactly the predicates the #131 role map calls definitions."""
    assert {str(p) for p in DEFINITION_PREDICATES} == _roles_v1_predicates("definition")


def test_definition_priority_keeps_the_original_three_first() -> None:
    """Appending predicates leaves first-definition-wins unchanged elsewhere."""
    assert [str(p) for p in DEFINITION_PREDICATES[:3]] == [
        "http://www.w3.org/2000/01/rdf-schema#comment",
        "http://www.w3.org/2004/02/skos/core#definition",
        "http://purl.obolibrary.org/obo/IAO_0000115",
    ]


def test_v1_reads_fma_synonyms() -> None:
    """An FMA class's fma:synonym literals follow its label."""
    g, ent = _fma_class()
    assert LabelVerbaliser().verbalise(g, ent, "class") == "Heart ; Cor"


def test_v2_reads_fma_definition_and_synonyms() -> None:
    """An FMA class's fma:definition and fma:synonym reach V2."""
    g, ent = _fma_class()
    assert AnnotationVerbaliser().verbalise(g, ent, "class") == (
        "Label: Heart. Definition: Organ pumping blood. Synonyms: Cor."
    )


def test_v1_resolves_reified_anatomy_synonyms() -> None:
    """oboInOwl synonym nodes contribute their rdfs:label, never their URI."""
    g, ent = _anatomy_class()
    assert LabelVerbaliser().verbalise(g, ent, "class") == (
        "Antibody_Producing_Cell ; Immunoglobulin-Producing Cells ; Plasma Cell"
    )


def test_v2_resolves_reified_anatomy_definition_and_synonyms() -> None:
    """oboInOwl:hasDefinition nodes resolve like synonym nodes."""
    g, ent = _anatomy_class()
    assert AnnotationVerbaliser().verbalise(g, ent, "class") == (
        "Label: Antibody_Producing_Cell. Definition: A cell that secretes antibodies. "
        "Synonyms: Immunoglobulin-Producing Cells; Plasma Cell."
    )


def test_regional_english_tags_count_as_english() -> None:
    """SNOMED's en-gb and en-us values are English; other languages are not."""
    g = Graph()
    ent = URIRef("http://snomed.info/id/80891009")
    g.add((ent, LABEL_PREDICATES[0], Literal("Heart structure")))
    g.add((ent, SYNONYM_PREDICATES[0], Literal("Cardiac structure", lang="en-gb")))
    g.add((ent, SYNONYM_PREDICATES[0], Literal("Heart, NOS", lang="en-US")))
    g.add((ent, SYNONYM_PREDICATES[0], Literal("Herz", lang="de")))
    g.add((ent, SYNONYM_PREDICATES[0], Literal("cœur", lang="fr-ca")))
    result = LabelVerbaliser().verbalise(g, ent, "class")
    assert result.startswith("Heart structure ; ")
    assert set(result.split(" ; ")[1:]) == {"Cardiac structure", "Heart, NOS"}


def test_reified_node_labels_follow_the_language_rule() -> None:
    """A synonym node's non-English labels are dropped."""
    g = Graph()
    ent = URIRef("http://example.org/ont#Test")
    node = URIRef("http://example.org/ont#genid9")
    g.add((ent, LABEL_PREDICATES[0], Literal("Label")))
    g.add((ent, SYNONYM_PREDICATES[4], node))
    g.add((node, _RDFS_LABEL, Literal("Etikett", lang="de")))
    assert LabelVerbaliser().verbalise(g, ent, "class") == "Label"


def test_annotation_predicates_digest_is_pinned() -> None:
    """A change to the predicate lists or value rules must be deliberate."""
    assert annotation_predicates_digest() == (
        "e6c9f1514dfec094f37159d75b26bb24d6993f852b30af7644a1d9adc984bdec"
    )
