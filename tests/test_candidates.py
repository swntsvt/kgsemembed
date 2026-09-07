from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDFS

from kgsemembed.candidates.generator import (
    _char_ngrams,
    build_candidate_table,
    generate_candidates,
)
from kgsemembed.candidates.ngram import (
    dump_candidates,
    get_entity_label,
    get_entity_text_from_attributes,
    load_candidates,
)
from kgsemembed.pipeline.run_experiment import run_experiment
from kgsemembed.utils.errors import ConfigurationError, DataError


def test_char_ngrams_for_multiple_n() -> None:
    assert _char_ngrams("abcd", 2) == ["ab", "bc", "cd"]
    assert _char_ngrams("abcd", 3) == ["abc", "bcd"]
    assert _char_ngrams("abcd", 4) == ["abcd"]


def test_cosine_and_jaccard_scores_generate_pairs() -> None:
    source = {"s1": "alpha"}
    target = {"t1": "alpha", "t2": "beta"}

    cosine_pairs = generate_candidates(source, target, n=3, metric="cosine", top_k=2)
    jaccard_pairs = generate_candidates(source, target, n=3, metric="jaccard", top_k=2)

    assert cosine_pairs[0].target_uri == "t1"
    assert jaccard_pairs[0].target_uri == "t1"
    assert cosine_pairs[0].score >= cosine_pairs[1].score
    assert jaccard_pairs[0].score >= jaccard_pairs[1].score


def test_deterministic_tie_breaking_and_top_k() -> None:
    source = {"s1": "aaaa"}
    target = {"t2": "aaaa", "t1": "aaaa", "t3": "aaaa"}

    pairs = generate_candidates(source, target, n=2, metric="jaccard", top_k=2)

    assert [p.target_uri for p in pairs] == ["t1", "t2"]
    assert [p.rank for p in pairs] == [1, 2]


def test_build_candidate_table_schema() -> None:
    pairs = [
        generate_candidates({"s1": "a"}, {"t1": "a"}, n=1, metric="jaccard", top_k=1)[0]
    ]
    table = build_candidate_table(pairs)
    assert list(table.columns) == ["source_uri", "target_uri", "score", "rank"]


def test_invalid_metric_and_invalid_params_raise() -> None:
    with pytest.raises(ConfigurationError):
        generate_candidates({"s": "a"}, {"t": "a"}, n=1, metric="bad", top_k=1)
    with pytest.raises(ConfigurationError):
        generate_candidates({"s": "a"}, {"t": "a"}, n=0, metric="cosine", top_k=1)
    with pytest.raises(ConfigurationError):
        generate_candidates({"s": "a"}, {"t": "a"}, n=1, metric="cosine", top_k=0)


def test_reproducible_output_across_runs() -> None:
    source = {f"s{i}": f"entity {i}" for i in range(20)}
    target = {f"t{i}": f"entity {i}" for i in range(40)}

    run1 = generate_candidates(source, target, n=3, metric="cosine", top_k=5)
    run2 = generate_candidates(source, target, n=3, metric="cosine", top_k=5)

    assert run1 == run2


def test_dump_candidates_round_trips_through_load_candidates(tmp_path: Path) -> None:
    candidates = {
        "http://ex/s1": ["http://ex/t1", "http://ex/t2"],
        "http://ex/s2": ["http://ex/t3"],
    }
    path = tmp_path / "candidates" / "D1" / "d1_pair_candidates.json"

    dump_candidates(candidates, path)

    assert load_candidates("D1", "d1_pair", tmp_path) == candidates


def test_dump_candidates_creates_missing_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "nested" / "pair_candidates.json"

    dump_candidates({"http://ex/s1": ["http://ex/t1"]}, path)

    assert path.exists()


def test_dump_candidates_writes_empty_map_readable_by_loader(tmp_path: Path) -> None:
    path = tmp_path / "candidates" / "D2" / "empty_candidates.json"

    dump_candidates({}, path)

    assert load_candidates("D2", "empty", tmp_path) == {}


def test_load_candidates_rejects_file_without_candidates_key(tmp_path: Path) -> None:
    path = tmp_path / "candidates" / "D3" / "bad_candidates.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"not_candidates": {}}')

    with pytest.raises(DataError):
        load_candidates("D3", "bad", tmp_path)


def _openea_graph() -> Graph:
    """Graph mimicking the D5 loader: values arrive as plain untyped Literals."""
    graph = Graph()
    entity = URIRef("http://dbpedia.org/resource/E291085")
    graph.add((entity, URIRef("http://dbpedia.org/ontology/birthName"), Literal("Ada Lovelace")))
    graph.add((entity, URIRef("http://dbpedia.org/ontology/alias"), Literal("Ada Byron")))
    return graph


def test_attribute_text_concatenates_qualifying_literals() -> None:
    text = get_entity_text_from_attributes(
        _openea_graph(), "http://dbpedia.org/resource/E291085"
    )

    assert text == "Ada Byron Ada Lovelace"


def test_attribute_text_strips_openea_inlined_datatype_suffix() -> None:
    """The datatype URI must never reach the index as shared n-gram boilerplate."""
    graph = Graph()
    entity = URIRef("http://dbpedia.org/resource/E1")
    graph.add((entity, URIRef("http://dbpedia.org/ontology/birthDate"),
               Literal('"1955-03-02"^^<http://www.w3.org/2001/XMLSchema#date>')))
    graph.add((entity, URIRef("http://dbpedia.org/ontology/title"),
               Literal('"The Hoodlum Saint"^^<http://www.w3.org/2001/XMLSchema#string>')))

    text = get_entity_text_from_attributes(graph, "http://dbpedia.org/resource/E1")

    assert "XMLSchema" not in text
    assert text == "1955-03-02 The Hoodlum Saint"


def test_attribute_text_drops_numeric_and_out_of_range_values() -> None:
    graph = Graph()
    entity = URIRef("http://dbpedia.org/resource/E2")
    graph.add((entity, URIRef("http://dbpedia.org/ontology/runtime"), Literal("6480.0")))
    graph.add((entity, URIRef("http://dbpedia.org/ontology/code"), Literal("ab")))
    graph.add((entity, URIRef("http://dbpedia.org/ontology/abstract"), Literal("x" * 201)))
    graph.add((entity, URIRef("http://dbpedia.org/ontology/rel"), URIRef("http://example.org/o")))

    assert get_entity_text_from_attributes(graph, "http://dbpedia.org/resource/E2") == ""


def test_attribute_text_keeps_date_shaped_values() -> None:
    """Dates align across D5 sides, so they survive the bare-quantity filter."""
    graph = Graph()
    entity = URIRef("http://dbpedia.org/resource/E4")
    graph.add((entity, URIRef("http://dbpedia.org/ontology/birthDate"),
               Literal('"1955-03-02"^^<http://www.w3.org/2001/XMLSchema#date>')))

    text = get_entity_text_from_attributes(graph, "http://dbpedia.org/resource/E4")

    assert text == "1955-03-02"


def test_attribute_text_keeps_dates_while_dropping_years_and_decimals() -> None:
    graph = Graph()
    entity = URIRef("http://dbpedia.org/resource/E5")
    graph.add((entity, URIRef("http://dbpedia.org/ontology/birthYear"),
               Literal('"1955"^^<http://www.w3.org/2001/XMLSchema#gYear>')))
    graph.add((entity, URIRef("http://dbpedia.org/ontology/budget"), Literal("14000000")))
    graph.add((entity, URIRef("http://dbpedia.org/ontology/releaseDate"),
               Literal('"1946-04-11"^^<http://www.w3.org/2001/XMLSchema#date>')))
    graph.add((entity, URIRef("http://dbpedia.org/ontology/runtime"), Literal("6480.0")))

    text = get_entity_text_from_attributes(graph, "http://dbpedia.org/resource/E5")

    assert text == "1946-04-11"


def test_entity_label_uses_date_attributes_for_opaque_ids() -> None:
    graph = Graph()
    entity = URIRef("http://www.wikidata.org/entity/Q1108721")
    graph.add((entity, URIRef("http://www.wikidata.org/entity/P569"),
               Literal('"1955-03-02"^^<http://www.w3.org/2001/XMLSchema#date>')))

    assert get_entity_label(graph, str(entity)) == "1955-03-02"


def test_attribute_text_rejects_malformed_date_like_values() -> None:
    graph = Graph()
    entity = URIRef("http://dbpedia.org/resource/E6")
    graph.add((entity, URIRef("http://dbpedia.org/ontology/p1"), Literal("1955-3-2")))
    graph.add((entity, URIRef("http://dbpedia.org/ontology/p2"), Literal("19550302")))
    graph.add((entity, URIRef("http://dbpedia.org/ontology/p3"), Literal("1955-03")))

    assert get_entity_text_from_attributes(graph, "http://dbpedia.org/resource/E6") == ""


def test_attribute_text_respects_max_values() -> None:
    graph = Graph()
    entity = URIRef("http://dbpedia.org/resource/E3")
    for index in range(5):
        graph.add((entity, URIRef(f"http://dbpedia.org/ontology/p{index}"), Literal(f"name{index}")))

    text = get_entity_text_from_attributes(graph, "http://dbpedia.org/resource/E3", max_values=2)

    assert text == "name0 name1"


def test_attribute_text_returns_empty_string_for_none_graph() -> None:
    assert get_entity_text_from_attributes(None, "http://dbpedia.org/resource/E1") == ""


def test_entity_label_uses_attribute_text_for_opaque_ids() -> None:
    label = get_entity_label(_openea_graph(), "http://dbpedia.org/resource/E291085")

    assert label == "Ada Byron Ada Lovelace"


def test_entity_label_falls_back_to_local_name_when_no_attributes_qualify() -> None:
    graph = Graph()
    graph.add((URIRef("http://www.wikidata.org/entity/Q7"),
               URIRef("http://www.wikidata.org/entity/P2047"), Literal("99")))

    assert get_entity_label(graph, "http://www.wikidata.org/entity/Q7") == "Q7"


def test_entity_label_prefers_rdfs_label_over_attribute_text() -> None:
    graph = _openea_graph()
    graph.add((URIRef("http://dbpedia.org/resource/E291085"), RDFS.label, Literal("Countess")))

    assert get_entity_label(graph, "http://dbpedia.org/resource/E291085") == "Countess"


def test_entity_label_leaves_bare_numeric_local_names_untouched() -> None:
    graph = Graph()
    entity = URIRef("http://dbkwik.webdatacommons.org/memory-alpha.wikia.com/resource/9")
    graph.add((entity, URIRef("http://dbkwik.org/ontology/name"), Literal("Fiat XI/9")))

    assert get_entity_label(graph, str(entity)) == "9"


def test_pipeline_persists_candidate_csv_with_expected_schema(tmp_path: Path) -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "candidates.persist=true",
                f"experiment.output_dir={tmp_path}",
                "candidates.output_file=candidates.csv",
                "candidates.top_k=2",
            ],
        )

    assert run_experiment(cfg) == 0
    csv_path = tmp_path / "candidates.csv"
    assert csv_path.exists()

    table = pd.read_csv(csv_path)
    assert list(table.columns) == ["source_uri", "target_uri", "score", "rank"]


def test_pipeline_cli_override_changes_candidate_behavior(tmp_path: Path) -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "candidates.metric=jaccard",
                "candidates.n=2",
                "candidates.top_k=1",
                "candidates.persist=true",
                f"experiment.output_dir={tmp_path}",
                "candidates.output_file=override.csv",
            ],
        )
    assert run_experiment(cfg) == 0
    table = pd.read_csv(tmp_path / "override.csv")
    assert int(table["rank"].max()) == 1
