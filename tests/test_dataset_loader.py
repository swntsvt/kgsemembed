from pathlib import Path

import pytest

from kgsemembed.datasets.loader import load_graph, load_oaei_dataset
from kgsemembed.utils.errors import DataError


def test_load_rdfxml_graph() -> None:
    loaded = load_graph("data/sample/source.rdf")
    assert loaded.format == "xml"
    assert len(loaded.triples) > 0
    assert any("EntityA" in entity for entity in loaded.entities)


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
