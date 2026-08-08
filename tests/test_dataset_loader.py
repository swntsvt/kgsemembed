"""Tests for graph loading, EDOAL reference parsing, and the D1-D5 enumerators.

Tests marked ``integration`` read the real ``data/`` tree and are skipped when
the corresponding dataset directory is absent, so a checkout without the OAEI
downloads still runs the rest of the suite.
"""

from pathlib import Path

import pytest
from rdflib import Graph

from kgsemembed.datasets.loader import (
    _parse_alignment_refs,
    _resolve_entity_type,
    _split_80_10_10,
    _split_val_test_20_80,
    extract_entity_uris,
    load_dataset,
    load_graph,
    load_oaei_dataset,
    load_pair_from_dir,
)
from kgsemembed.utils.errors import DataError

_EDOAL_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<rdf:RDF xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<Alignment>
  <map>
    <Cell>
      <entity1 rdf:resource="http://ex/s1"/>
      <entity2 rdf:resource="http://ex/t1"/>
      <relation>=</relation>
    </Cell>
  </map>
  <map>
    <Cell>
      <entity1 rdf:resource="http://ex/s2"/>
      <entity2 rdf:resource="http://ex/t2"/>
      <relation>&lt;</relation>
    </Cell>
  </map>
  <map>
    <Cell>
      <entity1 rdf:resource="http://ex/s3"/>
      <entity2 rdf:resource="http://ex/t3"/>
      <relation>=</relation>
    </Cell>
  </map>
</Alignment>
</rdf:RDF>
"""

_OWL_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Class rdf:about="http://ex/Alpha"/>
  <owl:ObjectProperty rdf:about="http://ex/relatesTo"/>
  <rdf:Description rdf:about="http://ex/thing1">
    <rdf:type rdf:resource="http://ex/Alpha"/>
  </rdf:Description>
</rdf:RDF>
"""


def _requires(dataset_dir: str) -> None:
    if not Path("data", dataset_dir).exists():
        pytest.skip(f"dataset directory missing: data/{dataset_dir}")


def _ontology_xml(prefix: str, count: int) -> str:
    body = "\n".join(
        f'  <owl:Class rdf:about="http://ex/{prefix}{i:02d}">'
        f'<rdfs:label xml:lang="en">{prefix} {i:02d}</rdfs:label></owl:Class>'
        for i in range(count)
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"\n'
        '         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"\n'
        '         xmlns:owl="http://www.w3.org/2002/07/owl#">\n'
        f"{body}\n</rdf:RDF>\n"
    )


def _edoal_xml(count: int) -> str:
    cells = "\n".join(
        "  <map><Cell>"
        f'<entity1 rdf:resource="http://ex/src{i:02d}"/>'
        f'<entity2 rdf:resource="http://ex/tgt{i:02d}"/>'
        "<relation>=</relation></Cell></map>"
        for i in range(count)
    )
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<rdf:RDF xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment"\n'
        '         xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        f"<Alignment>\n{cells}\n</Alignment>\n</rdf:RDF>\n"
    )


def build_sample_pair_dir(tmp_path: Path, count: int = 10) -> Path:
    """Write a source/target/reference triple in the real OAEI EDOAL layout."""
    pair_dir = tmp_path / "pair"
    pair_dir.mkdir()
    (pair_dir / "source.rdf").write_text(_ontology_xml("src", count), encoding="utf-8")
    (pair_dir / "target.rdf").write_text(_ontology_xml("tgt", count), encoding="utf-8")
    (pair_dir / "reference.rdf").write_text(_edoal_xml(count), encoding="utf-8")
    return pair_dir


def _assert_str_pairs(refs) -> None:
    assert all(
        isinstance(ref, tuple) and len(ref) == 2 and isinstance(ref[0], str) and isinstance(ref[1], str)
        for ref in refs
    )


# ---------------------------------------------------------------------------
# Graph loading utilities
# ---------------------------------------------------------------------------


def test_load_rdfxml_graph() -> None:
    loaded = load_graph("data/sample/source.rdf")
    assert loaded.format == "xml"
    assert len(loaded.triples) > 0
    assert any("MA_" in entity for entity in loaded.entities)


def test_load_ttl_graph(tmp_path: Path) -> None:
    ttl_path = tmp_path / "mini.ttl"
    ttl_path.write_text(
        """
@prefix ex: <http://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:A rdfs:label "Alpha"@en .
""".strip()
    )
    loaded = load_graph(ttl_path)
    assert loaded.format == "turtle"
    assert loaded.labels["http://example.org/A"] == "Alpha"


def test_unsupported_extension_raises_data_error(tmp_path: Path) -> None:
    bad_path = tmp_path / "graph.nt"
    bad_path.write_text("<a> <b> <c> .")
    with pytest.raises(DataError):
        load_graph(bad_path)


def test_label_priority_and_fallback(tmp_path: Path) -> None:
    ttl_path = tmp_path / "labels.ttl"
    ttl_path.write_text(
        """
@prefix ex: <http://example.org/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .

ex:WithBoth rdfs:label "RDFS Label"@en ;
    skos:prefLabel "SKOS Label"@en .

ex:WithFrench rdfs:label "Nom"@fr ;
    rdfs:label "Name"@en .

ex:NoLabel ex:relatedTo ex:WithBoth .
""".strip()
    )
    loaded = load_graph(ttl_path)

    assert loaded.labels["http://example.org/WithBoth"] == "RDFS Label"
    assert loaded.labels["http://example.org/WithFrench"] == "Name"
    assert loaded.labels["http://example.org/NoLabel"] == "NoLabel"


def test_load_oaei_dataset_bundle() -> None:
    bundle = load_oaei_dataset(
        source_path="data/sample/source.rdf",
        target_path="data/sample/target.rdf",
        alignment_path="data/sample/reference.rdf",
    )
    assert len(bundle.source.triples) > 0
    assert len(bundle.target.triples) > 0
    assert len(bundle.alignment.triples) > 0


def test_missing_path_raises_data_error() -> None:
    with pytest.raises(DataError) as exc:
        load_oaei_dataset("missing.rdf", "data/sample/target.rdf", "data/sample/reference.rdf")
    assert "missing.rdf" in str(exc.value)


# ---------------------------------------------------------------------------
# EDOAL reference parser
# ---------------------------------------------------------------------------


def test_parse_alignment_refs_keeps_equivalence_only(tmp_path: Path) -> None:
    path = tmp_path / "reference.rdf"
    path.write_text(_EDOAL_TEMPLATE, encoding="utf-8")

    refs = _parse_alignment_refs(path)

    assert refs == [("http://ex/s1", "http://ex/t1"), ("http://ex/s3", "http://ex/t3")]
    _assert_str_pairs(refs)


def test_parse_alignment_refs_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DataError):
        _parse_alignment_refs(tmp_path / "absent.rdf")


def test_parse_alignment_refs_malformed_xml_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.rdf"
    path.write_text("<rdf:RDF><unclosed>", encoding="utf-8")
    with pytest.raises(DataError):
        _parse_alignment_refs(path)


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------


def test_split_80_10_10_slices_sorted_by_source() -> None:
    refs = [(f"s{i:02d}", f"t{i:02d}") for i in range(10)]
    train, val, test = _split_80_10_10(list(reversed(refs)))
    assert train == refs[:8]
    assert val == [("s08", "t08")]
    assert test == [("s09", "t09")]


def test_split_val_test_20_80_leaves_train_empty() -> None:
    refs = [(f"s{i:02d}", f"t{i:02d}") for i in range(10)]
    train, val, test = _split_val_test_20_80(list(reversed(refs)))
    assert train == []
    assert val == [("s00", "t00"), ("s01", "t01")]
    assert len(test) == 8


def test_split_val_test_20_80_keeps_one_val_for_tiny_inputs() -> None:
    train, val, test = _split_val_test_20_80([("s0", "t0"), ("s1", "t1")])
    assert train == []
    assert val == [("s0", "t0")]
    assert test == [("s1", "t1")]


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def test_extract_entity_uris_mixed_is_classes_plus_predicates(tmp_path: Path) -> None:
    path = tmp_path / "onto.rdf"
    path.write_text(_OWL_TEMPLATE, encoding="utf-8")
    graph = Graph()
    graph.parse(str(path), format="xml")

    classes = extract_entity_uris(graph, "class")
    predicates = extract_entity_uris(graph, "predicate")
    mixed = extract_entity_uris(graph, "mixed")

    assert classes == ["http://ex/Alpha"]
    assert predicates == ["http://ex/relatesTo"]
    assert mixed == sorted(classes + predicates)
    assert len(mixed) == len(set(mixed))
    assert "http://ex/thing1" not in mixed


def test_extract_entity_uris_rejects_unknown_type() -> None:
    with pytest.raises(DataError):
        extract_entity_uris(Graph(), "not-a-type")


def test_load_dataset_rejects_unknown_dataset() -> None:
    with pytest.raises(DataError):
        load_dataset("D9", "data/")


# ---------------------------------------------------------------------------
# Real-data integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_load_pair_from_dir_reads_sample() -> None:
    _requires("sample")
    pair = load_pair_from_dir("data/sample")

    assert pair.source_graph is not None
    assert len(pair.source_graph) > 0
    assert len(pair.source_entities) > 0
    _assert_str_pairs(pair.val_refs)


def test_load_pair_from_dir_splits_edoal_reference(tmp_path: Path) -> None:
    pair_dir = build_sample_pair_dir(tmp_path)
    pair = load_pair_from_dir(pair_dir)

    assert len(pair.source_graph) > 0
    assert len(pair.source_entities) == 10
    assert len(pair.val_refs) > 0
    assert len(pair.test_refs) > 0
    _assert_str_pairs(pair.val_refs)
    _assert_str_pairs(pair.test_refs)


@pytest.mark.integration
def test_load_dataset_d1_returns_single_pair() -> None:
    _requires("d1_snomed_fma")
    pairs = load_dataset("D1")

    assert len(pairs) == 1
    pair = pairs[0]
    assert len(pair.source_graph) > 0
    assert len(pair.source_entities) > 0
    assert len(pair.val_refs) > 0
    assert len(pair.test_refs) > 0
    _assert_str_pairs(pair.val_refs)


@pytest.mark.integration
def test_load_dataset_d2_split_is_deterministic() -> None:
    _requires("d2_anatomy")
    first = load_dataset("D2")
    second = load_dataset("D2")

    assert len(first) == 1
    assert first[0].val_refs == second[0].val_refs
    assert first[0].test_refs == second[0].test_refs
    assert len(first[0].val_refs) > 0
    _assert_str_pairs(first[0].val_refs)


@pytest.mark.integration
def test_load_dataset_d3_returns_21_conference_pairs() -> None:
    _requires("d3_conference")
    pairs = load_dataset("D3")

    assert len(pairs) == 21
    names = [pair.pair_name for pair in pairs]
    assert not any("confTool" in name or "sofsem" in name for name in names)
    for pair in pairs:
        assert pair.entity_type == "mixed"
        assert pair.train_refs == []
        assert len(pair.val_refs) > 0
        _assert_str_pairs(pair.val_refs)


@pytest.mark.integration
def test_load_dataset_d4_splits_schema_and_instance() -> None:
    _requires("d4_kgtrack")
    pairs = load_dataset("D4")

    assert len(pairs) == 2
    schema, instance = pairs
    assert schema.source_graph is instance.source_graph
    assert schema.target_graph is instance.target_graph
    assert schema.entity_type == "mixed"
    assert instance.entity_type == "instance"
    assert len(schema.val_refs) > 0
    assert len(instance.val_refs) > 0


@pytest.mark.integration
def test_load_dataset_d4_reference_entity_types_are_consistent() -> None:
    _requires("d4_kgtrack")
    schema, instance = load_dataset("D4")
    graph = schema.source_graph

    for uri, _ in schema.train_refs[:10]:
        assert _resolve_entity_type(graph, uri) in ("class", "predicate")
    for uri, _ in instance.train_refs[:10]:
        assert _resolve_entity_type(graph, uri) == "instance"


@pytest.mark.integration
def test_load_dataset_d5_builds_graphs_from_triple_files() -> None:
    _requires("d5_openea")
    pairs = load_dataset("D5")

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.source_graph is not None
    assert len(pair.source_graph) > 0
    assert len(pair.val_refs) > 0
    assert len(pair.test_refs) > 0
    assert pair.train_refs == []
    _assert_str_pairs(pair.val_refs)
