"""Tests for CombinedVerbaliser and the verbaliser registry factory."""

import pytest
from rdflib import Graph, URIRef

from kgsemembed.verbalisation import (
    AnnotationVerbaliser,
    CombinedVerbaliser,
    HierarchicalContextVerbaliser,
    LabelVerbaliser,
    NeighbourhoodWalkVerbaliser,
    RelationalSignatureVerbaliser,
    SchemaAwareVerbaliser,
    StructuredKVVerbaliser,
    TemplateNLVerbaliser,
    VALID_STRATEGY_NAMES,
    VerbaliserBase,
    build_verbaliser,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

_ENTITY = URIRef("http://example.org/ont#Entity")


class StubVerbaliser(VerbaliserBase):
    """Return a fixed output and record the order of invocation."""

    def __init__(self, output: str, call_log: list[str], name: str) -> None:
        self.output = output
        self.call_log = call_log
        self.name = name

    def verbalise(
        self, graph: Graph, entity_uri: URIRef, entity_type: str
    ) -> str:
        self.call_log.append(self.name)
        return self.output


def _stub(output: str, call_log: list[str], name: str = "stub") -> StubVerbaliser:
    return StubVerbaliser(output, call_log, name)


# Expected component structure of every registry strategy. Individual
# strategies map to a single type; composites map to their ordered component
# types. The registry, not this table, drives construction tests; this table
# only asserts what each name should build.
_EXPECTED_STRUCTURE: dict[str, list[type]] = {
    "V1": [LabelVerbaliser],
    "V2": [AnnotationVerbaliser],
    "V3": [TemplateNLVerbaliser],
    "V4": [StructuredKVVerbaliser],
    "V5": [NeighbourhoodWalkVerbaliser],
    "V6": [SchemaAwareVerbaliser],
    "V7": [HierarchicalContextVerbaliser],
    "V8": [RelationalSignatureVerbaliser],
    "V2+V6": [AnnotationVerbaliser, SchemaAwareVerbaliser],
    "V2+V8": [AnnotationVerbaliser, RelationalSignatureVerbaliser],
    "V2+V7": [AnnotationVerbaliser, HierarchicalContextVerbaliser],
    "V2+V8+V7": [
        AnnotationVerbaliser,
        RelationalSignatureVerbaliser,
        HierarchicalContextVerbaliser,
    ],
    "V6+V3": [SchemaAwareVerbaliser, TemplateNLVerbaliser],
    "V4+V6": [StructuredKVVerbaliser, SchemaAwareVerbaliser],
    "V8+V6": [RelationalSignatureVerbaliser, SchemaAwareVerbaliser],
}

_MODEL_AWARE_TYPES = (
    TemplateNLVerbaliser,
    StructuredKVVerbaliser,
    NeighbourhoodWalkVerbaliser,
    SchemaAwareVerbaliser,
    HierarchicalContextVerbaliser,
)


# ---------------------------------------------------------------------------
# Registry construction (driven by VALID_STRATEGY_NAMES)
# ---------------------------------------------------------------------------


def test_valid_strategy_names_match_expected_structure() -> None:
    assert set(VALID_STRATEGY_NAMES) == set(_EXPECTED_STRUCTURE)


def test_valid_strategy_names_has_no_duplicates() -> None:
    assert len(VALID_STRATEGY_NAMES) == len(set(VALID_STRATEGY_NAMES))


def test_valid_strategy_names_is_deterministic() -> None:
    assert VALID_STRATEGY_NAMES == list(VALID_STRATEGY_NAMES)
    assert VALID_STRATEGY_NAMES == [name for name in _EXPECTED_STRUCTURE]


@pytest.mark.parametrize("strategy_name", VALID_STRATEGY_NAMES)
def test_build_verbaliser_returns_verbaliser_base(strategy_name: str) -> None:
    verbaliser = build_verbaliser(strategy_name)
    assert isinstance(verbaliser, VerbaliserBase)


@pytest.mark.parametrize("strategy_name", VALID_STRATEGY_NAMES)
def test_build_verbaliser_constructs_expected_structure(
    strategy_name: str,
) -> None:
    expected = _EXPECTED_STRUCTURE[strategy_name]
    verbaliser = build_verbaliser(strategy_name)

    if len(expected) == 1:
        assert not isinstance(verbaliser, CombinedVerbaliser)
        assert isinstance(verbaliser, expected[0])
    else:
        assert isinstance(verbaliser, CombinedVerbaliser)
        actual = [type(s) for s in verbaliser.strategies]
        assert actual == expected


# ---------------------------------------------------------------------------
# Model-key propagation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy_name", VALID_STRATEGY_NAMES)
def test_model_aware_components_receive_model_key(strategy_name: str) -> None:
    verbaliser = build_verbaliser(strategy_name, model_key="M4")
    components = (
        verbaliser.strategies
        if isinstance(verbaliser, CombinedVerbaliser)
        else [verbaliser]
    )
    for component in components:
        if isinstance(component, _MODEL_AWARE_TYPES):
            assert component.model_key == "M4"


def test_model_independent_components_have_no_model_key() -> None:
    for strategy_name in ("V1", "V2", "V8"):
        verbaliser = build_verbaliser(strategy_name, model_key="M4")
        assert not hasattr(verbaliser, "model_key")


def test_default_model_key_is_m2() -> None:
    verbaliser = build_verbaliser("V3")
    assert verbaliser.model_key == "M2"


def test_arbitrary_model_key_is_forwarded() -> None:
    verbaliser = build_verbaliser("V6+V3", model_key="anything")
    for component in verbaliser.strategies:
        assert component.model_key == "anything"


# ---------------------------------------------------------------------------
# Invalid registry names
# ---------------------------------------------------------------------------


def test_unknown_strategy_raises_value_error() -> None:
    with pytest.raises(ValueError):
        build_verbaliser("V99")


def test_unknown_strategy_error_names_invalid_value() -> None:
    with pytest.raises(ValueError, match="V99"):
        build_verbaliser("V99")


def test_unknown_strategy_error_lists_every_valid_strategy() -> None:
    with pytest.raises(ValueError) as excinfo:
        build_verbaliser("nope")
    message = str(excinfo.value)
    for name in VALID_STRATEGY_NAMES:
        assert name in message


# ---------------------------------------------------------------------------
# CombinedVerbaliser: composition behaviour
# ---------------------------------------------------------------------------


def test_combined_preserves_strategy_order() -> None:
    call_log: list[str] = []
    combined = CombinedVerbaliser(
        [
            _stub("first", call_log, "a"),
            _stub("second", call_log, "b"),
            _stub("third", call_log, "c"),
        ]
    )
    combined.verbalise(Graph(), _ENTITY, "class")
    assert call_log == ["a", "b", "c"]


def test_combined_joins_with_single_newline() -> None:
    call_log: list[str] = []
    combined = CombinedVerbaliser(
        [_stub("alpha", call_log), _stub("beta", call_log)]
    )
    assert combined.verbalise(Graph(), _ENTITY, "class") == "alpha\nbeta"


def test_combined_strips_surrounding_whitespace() -> None:
    call_log: list[str] = []
    combined = CombinedVerbaliser(
        [_stub("  alpha  ", call_log), _stub("\nbeta\n", call_log)]
    )
    assert combined.verbalise(Graph(), _ENTITY, "class") == "alpha\nbeta"


def test_combined_omits_empty_output() -> None:
    call_log: list[str] = []
    combined = CombinedVerbaliser(
        [_stub("alpha", call_log), _stub("", call_log), _stub("gamma", call_log)]
    )
    assert combined.verbalise(Graph(), _ENTITY, "class") == "alpha\ngamma"


def test_combined_omits_whitespace_only_output() -> None:
    call_log: list[str] = []
    combined = CombinedVerbaliser(
        [_stub("alpha", call_log), _stub("   \n\t ", call_log), _stub("gamma", call_log)]
    )
    assert combined.verbalise(Graph(), _ENTITY, "class") == "alpha\ngamma"


def test_combined_all_empty_returns_empty_string() -> None:
    call_log: list[str] = []
    combined = CombinedVerbaliser(
        [_stub("", call_log), _stub("   ", call_log)]
    )
    assert combined.verbalise(Graph(), _ENTITY, "class") == ""


def test_combined_no_leading_or_trailing_newline() -> None:
    call_log: list[str] = []
    combined = CombinedVerbaliser(
        [_stub("", call_log), _stub("middle", call_log), _stub("", call_log)]
    )
    result = combined.verbalise(Graph(), _ENTITY, "class")
    assert result == "middle"
    assert not result.startswith("\n")
    assert not result.endswith("\n")


def test_combined_supports_two_strategies() -> None:
    call_log: list[str] = []
    combined = CombinedVerbaliser(
        [_stub("a", call_log), _stub("b", call_log)]
    )
    assert combined.verbalise(Graph(), _ENTITY, "class") == "a\nb"


def test_combined_supports_three_or_more_strategies() -> None:
    call_log: list[str] = []
    combined = CombinedVerbaliser(
        [_stub("a", call_log), _stub("b", call_log), _stub("c", call_log)]
    )
    assert combined.verbalise(Graph(), _ENTITY, "class") == "a\nb\nc"


def test_combined_preserves_internal_component_content() -> None:
    call_log: list[str] = []
    combined = CombinedVerbaliser(
        [
            _stub("  Label: Heart.\nDefinition: An organ.  ", call_log),
            _stub("Outgoing: part of.\nIncoming: none.", call_log),
        ]
    )
    result = combined.verbalise(Graph(), _ENTITY, "class")
    assert result == (
        "Label: Heart.\nDefinition: An organ.\n"
        "Outgoing: part of.\nIncoming: none."
    )


# ---------------------------------------------------------------------------
# CombinedVerbaliser: constructor validation
# ---------------------------------------------------------------------------


def test_combined_zero_strategies_raises() -> None:
    with pytest.raises(ValueError):
        CombinedVerbaliser([])


def test_combined_one_strategy_raises() -> None:
    call_log: list[str] = []
    with pytest.raises(ValueError):
        CombinedVerbaliser([_stub("solo", call_log)])


def test_combined_stores_strategies_unmodified() -> None:
    call_log: list[str] = []
    strategies = [_stub("a", call_log), _stub("b", call_log)]
    combined = CombinedVerbaliser(strategies)
    assert combined.strategies == strategies


# ---------------------------------------------------------------------------
# Regression: exports remain correct
# ---------------------------------------------------------------------------


def test_public_exports_present() -> None:
    import kgsemembed.verbalisation as verbalisation

    for name in ("CombinedVerbaliser", "build_verbaliser", "VALID_STRATEGY_NAMES"):
        assert name in verbalisation.__all__
        assert hasattr(verbalisation, name)
