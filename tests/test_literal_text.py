"""Tests for removing OpenEA's inlined datatype suffix from literal text (#183)."""

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import XSD

from kgsemembed.candidates.ngram import _literal_text as stage1_literal_text
from kgsemembed.verbalisation import build_verbaliser
from kgsemembed.verbalisation.base import literal_text

_ENT = URIRef("http://dbpedia.org/resource/E000038")
_XSD = "http://www.w3.org/2001/XMLSchema#"


@pytest.mark.parametrize(
    "raw, expected",
    [
        (f'"2008-10-29"^^<{_XSD}date>', "2008-10-29"),
        (f'"1985"^^<{_XSD}gYear>', "1985"),
        (f'"1082.0"^^<{_XSD}double>', "1082.0"),
        (f'"99"^^<{_XSD}decimal>', "99"),
        ('"1.5E8"^^<http://dbpedia.org/datatype/usDollar>', "1.5E8"),
        (f'"true"^^<{_XSD}boolean>', "true"),
        (f'""^^<{_XSD}string>', ""),
        (f'"line one\nline two"^^<{_XSD}string>', "line one\nline two"),
    ],
)
def test_inlined_suffix_and_quotes_are_removed(raw: str, expected: str) -> None:
    assert literal_text(Literal(raw)) == expected


def test_escaped_value_is_kept_verbatim() -> None:
    raw = f'"He said \\"hi\\" \\\\ there"^^<{_XSD}string>'
    assert literal_text(Literal(raw)) == 'He said \\"hi\\" \\\\ there'


@pytest.mark.parametrize(
    "raw",
    [
        "Robert Clifford Jones",
        '\\"Jerusalem of the East\\',
        'The value "5"^^<http://example.org/int> appears in text',
        '"quoted" but no datatype',
        '"5"^^<http://example.org/int> trailing words',
        "tt0067946",
    ],
)
def test_other_text_is_unchanged(raw: str) -> None:
    assert literal_text(Literal(raw)) == raw


def test_rdflib_typed_and_language_tagged_literals_are_unchanged() -> None:
    assert literal_text(Literal("5", datatype=XSD.integer)) == "5"
    assert literal_text(Literal("2008-10-29", datatype=XSD.date)) == "2008-10-29"
    tagged = Literal(f'"x"^^<{_XSD}string>', lang="en")
    assert literal_text(tagged) == f'"x"^^<{_XSD}string>'
    assert literal_text(Literal("Heart", lang="en")) == "Heart"


@pytest.mark.parametrize(
    "raw",
    [f'"2008-10-29"^^<{_XSD}date>', '"1.5E8"^^<http://dbpedia.org/datatype/usDollar>', "plain", '"a"'],
)
def test_same_text_as_stage1(raw: str) -> None:
    assert literal_text(Literal(raw)) == stage1_literal_text(Literal(raw))


def _d5_entity() -> Graph:
    g = Graph()
    g.add((_ENT, URIRef("http://dbpedia.org/ontology/releaseDate"), Literal(f'"2008-10-29"^^<{_XSD}date>')))
    g.add((_ENT, URIRef("http://dbpedia.org/ontology/runtime"), Literal(f'"1082.0"^^<{_XSD}double>')))
    return g


@pytest.mark.parametrize("strategy", ["V3", "V4", "V6", "V2+V6", "V6+V3", "V4+V6"])
@pytest.mark.parametrize("model_key", ["M2", "M3"])
def test_strategies_write_no_suffix(strategy: str, model_key: str) -> None:
    text = build_verbaliser(strategy, model_key, "source").verbalise(_d5_entity(), _ENT, "instance")
    assert "^^" not in text and "XMLSchema" not in text
    assert "2008-10-29" in text and "1082.0" in text


def test_budget_counts_the_cleaned_text() -> None:
    verbaliser = build_verbaliser("V6", "M2", "source")
    text = verbaliser.verbalise(_d5_entity(), _ENT, "instance")
    assert text == "release Date: 2008-10-29 | runtime: 1082.0"
    assert verbaliser.budget.count(text) < 25
