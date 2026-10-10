"""Tests for the Phase 2 experiment condition registry."""

import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from kgsemembed.embeddings.models import MODEL_REGISTRY
from kgsemembed.pipeline import (
    EXPERIMENT_CONDITIONS,
    ExperimentCondition,
    get_condition,
    get_conditions_for_dataset,
    get_conditions_for_group,
)
from kgsemembed.pipeline.conditions import _validate_registry
from kgsemembed.verbalisation import (
    AnnotationVerbaliser,
    CombinedVerbaliser,
    RelationalSignatureVerbaliser,
    VerbaliserBase,
)
from kgsemembed.verbalisation.registry import VALID_STRATEGY_NAMES, build_verbaliser

_NON_PPAS_IDS = {"C14", "C15", "C19"}

_ALL = ["D1", "D2", "D3", "D4", "D5"]

_CANONICAL_MATRIX = [
    ("C1", "V1", "M1", _ALL, True, "A"),
    ("C2", "V2", "M1", _ALL, True, "A"),
    ("C17", "V1", "M4", ["D1", "D2"], True, "A"),
    ("C3", "V2+V6", "M2", _ALL, True, "B"),
    ("C9", "V8", "M2", _ALL, True, "B"),
    ("C10", "V2+V8", "M2", _ALL, True, "B"),
    ("C12", "V8", "M2", ["D3", "D4"], True, "B"),
    ("C18", "V2+V8", "M1", _ALL, True, "B"),
    ("C4", "V3", "M2", ["D1", "D2", "D3"], True, "C"),
    ("C5", "V4", "M2", ["D4", "D5"], True, "C"),
    ("C6", "V2+V7", "M4", ["D1", "D2"], True, "C"),
    ("C7", "V6+V3", "M2", ["D3", "D4"], True, "C"),
    ("C8", "V5", "M2", ["D4", "D5"], True, "C"),
    ("C11", "V2+V8", "M4", ["D1", "D2"], True, "C"),
    ("C13", "V2+V8", "M5", _ALL, True, "C"),
    ("C16", "V2+V8+V7", "M4", ["D1", "D2"], True, "C"),
    ("C14", "V4+V6", "M3", ["D5"], False, "D"),
    ("C15", "V2+V8", "M3", ["D1"], False, "D"),
    ("C19", "V2+V8", "M2_uncapped", ["D1"], False, "D"),
    ("C20", "V2", "M2", _ALL, True, "E"),
]


def _make_condition(**overrides) -> ExperimentCondition:
    defaults = dict(
        condition_id="C1",
        strategy_name="V1",
        model_key="M1",
        datasets=["D1"],
        apply_ppas=True,
        description="synthetic condition.",
        ablation_group="A",
    )
    defaults.update(overrides)
    return ExperimentCondition(**defaults)


def test_registry_holds_twenty_conditions():
    assert len(EXPERIMENT_CONDITIONS) == 20


def test_registry_matches_canonical_matrix_in_order():
    actual = [
        (
            c.condition_id,
            c.strategy_name,
            c.model_key,
            c.datasets,
            c.apply_ppas,
            c.ablation_group,
        )
        for c in EXPERIMENT_CONDITIONS
    ]
    assert actual == _CANONICAL_MATRIX


def test_every_condition_has_a_description():
    for condition in EXPERIMENT_CONDITIONS:
        assert condition.description.strip()


def test_condition_identifiers_are_unique():
    ids = [c.condition_id for c in EXPERIMENT_CONDITIONS]
    assert len(ids) == len(set(ids))


def test_every_strategy_is_registered():
    for condition in EXPERIMENT_CONDITIONS:
        assert condition.strategy_name in VALID_STRATEGY_NAMES


def test_every_model_is_registered():
    for condition in EXPERIMENT_CONDITIONS:
        assert condition.model_key in MODEL_REGISTRY


def test_ppas_disabled_only_for_the_non_ppas_conditions():
    disabled = {c.condition_id for c in EXPERIMENT_CONDITIONS if not c.apply_ppas}
    assert disabled == _NON_PPAS_IDS


def test_all_other_conditions_enable_ppas():
    for condition in EXPERIMENT_CONDITIONS:
        if condition.condition_id not in _NON_PPAS_IDS:
            assert condition.apply_ppas is True


def test_get_condition_returns_matching_condition():
    condition = get_condition("C1")
    assert isinstance(condition, ExperimentCondition)
    assert condition.condition_id == "C1"


def test_get_condition_strategy_lookup():
    assert get_condition("C10").strategy_name == "V2+V8"


def test_get_condition_ppas_flags():
    assert get_condition("C14").apply_ppas is False
    assert get_condition("C15").apply_ppas is False
    assert get_condition("C19").apply_ppas is False
    assert get_condition("C1").apply_ppas is True


def test_c19_uses_the_uncapped_twin_of_the_c10_model():
    assert get_condition("C19").model_key == "M2_uncapped"
    assert get_condition("C19").strategy_name == get_condition("C10").strategy_name
    assert get_condition("C19").datasets == ["D1"]


def test_c19_verbalises_identically_to_c10():
    """
    C19 is a null ablation by design: V2+V8 never consults a token budget, so
    swapping M2 for its uncapped twin cannot change the verbalised text.  This
    pins that fact, so making V2 or V8 budget-aware fails here rather than
    silently turning C19 into a different experiment.
    """
    from rdflib import Graph, Literal, RDFS, URIRef

    from kgsemembed.verbalisation.registry import build_verbaliser

    graph = Graph()
    entity = URIRef("http://example.org/e1")
    graph.add((entity, RDFS.label, Literal("Heart", lang="en")))
    for index in range(200):
        graph.add((entity, URIRef(f"http://example.org/p{index}"), Literal(index)))

    strategy = get_condition("C19").strategy_name
    capped = build_verbaliser(strategy, get_condition("C10").model_key)
    uncapped = build_verbaliser(strategy, get_condition("C19").model_key)
    assert capped.verbalise(graph, entity, "class") == uncapped.verbalise(
        graph, entity, "class"
    )


def _v8_ablation_graph() -> tuple[Graph, dict[str, URIRef]]:
    ex = "http://example.org/"
    graph = Graph()
    heart, organ, part_of, aorta = (
        URIRef(f"{ex}{name}") for name in ("Heart", "Organ", "partOf", "aorta1")
    )
    graph.add((heart, RDF.type, OWL.Class))
    graph.add((heart, RDFS.label, Literal("Heart", lang="en")))
    graph.add((heart, RDFS.comment, Literal("A muscular organ.")))
    graph.add((heart, RDFS.subClassOf, organ))
    graph.add((part_of, RDF.type, OWL.ObjectProperty))
    graph.add((part_of, RDFS.label, Literal("part of")))
    graph.add((part_of, RDFS.domain, heart))
    graph.add((part_of, RDFS.range, organ))
    graph.add((aorta, RDFS.label, Literal("Aorta")))
    graph.add((aorta, part_of, heart))
    return graph, {"class": heart, "predicate": part_of, "instance": aorta}


def _condition_verbaliser(condition_id: str) -> VerbaliserBase:
    condition = get_condition(condition_id)
    return build_verbaliser(condition.strategy_name, condition.model_key)


def test_c20_differs_from_c10_only_in_strategy() -> None:
    c20, c10 = get_condition("C20"), get_condition("C10")
    assert (c20.strategy_name, c10.strategy_name) == ("V2", "V2+V8")
    for field in ("model_key", "datasets", "apply_ppas"):
        assert getattr(c20, field) == getattr(c10, field)


def test_c20_is_the_sole_member_of_its_own_ablation_group() -> None:
    assert get_condition("C20").ablation_group == "E"
    assert [c.condition_id for c in get_conditions_for_group("E")] == ["C20"]


def test_c20_builds_the_bare_annotation_verbaliser() -> None:
    assert type(_condition_verbaliser("C20")) is AnnotationVerbaliser


def test_c10_builds_annotation_then_relational_signature() -> None:
    verbaliser = _condition_verbaliser("C10")
    assert isinstance(verbaliser, CombinedVerbaliser)
    assert [type(s) for s in verbaliser.strategies] == [
        AnnotationVerbaliser,
        RelationalSignatureVerbaliser,
    ]


@pytest.mark.parametrize("entity_type", ["class", "predicate", "instance"])
def test_c10_text_extends_c20_text_with_v8(entity_type: str) -> None:
    """
    C20 and C10 must differ only by V8, so the C10 text is the C20 text with
    the V8 signature appended and nothing of the annotation text replaced.
    """
    graph, entities = _v8_ablation_graph()
    entity = entities[entity_type]
    v2_text = _condition_verbaliser("C20").verbalise(graph, entity, entity_type)
    v8_text = RelationalSignatureVerbaliser().verbalise(graph, entity, entity_type)
    combined = _condition_verbaliser("C10").verbalise(graph, entity, entity_type)
    assert combined == f"{v2_text}\n{v8_text}"


def test_get_condition_unknown_raises_key_error():
    with pytest.raises(KeyError):
        get_condition("C999")


def test_get_condition_uses_exact_string_matching():
    with pytest.raises(KeyError):
        get_condition("c1")
    with pytest.raises(KeyError):
        get_condition("C1 ")


def test_get_conditions_for_dataset_includes_expected():
    ids = [c.condition_id for c in get_conditions_for_dataset("D5")]
    assert "C1" in ids
    assert "C14" in ids


def test_get_conditions_for_dataset_preserves_registry_order():
    expected = [c for c in EXPERIMENT_CONDITIONS if "D5" in c.datasets]
    assert get_conditions_for_dataset("D5") == expected


def test_get_conditions_for_dataset_unknown_returns_empty():
    assert get_conditions_for_dataset("D99") == []


def test_get_conditions_for_group_d_holds_every_non_ppas_condition():
    ids = [c.condition_id for c in get_conditions_for_group("D")]
    assert ids == ["C14", "C15", "C19"]


@pytest.mark.parametrize(
    "group, expected_ids",
    [
        ("A", ["C1", "C2", "C17"]),
        ("B", ["C3", "C9", "C10", "C12", "C18"]),
        ("C", ["C4", "C5", "C6", "C7", "C8", "C11", "C13", "C16"]),
    ],
)
def test_get_conditions_for_group_membership(group, expected_ids):
    ids = [c.condition_id for c in get_conditions_for_group(group)]
    assert ids == expected_ids


def test_get_conditions_for_group_preserves_registry_order():
    for group in {c.ablation_group for c in EXPERIMENT_CONDITIONS}:
        expected = [c for c in EXPERIMENT_CONDITIONS if c.ablation_group == group]
        assert get_conditions_for_group(group) == expected


def test_get_conditions_for_group_unknown_returns_empty():
    assert get_conditions_for_group("Z") == []


def test_validation_rejects_wrong_count():
    with pytest.raises(ValueError):
        _validate_registry([_make_condition()])


def test_validation_rejects_duplicate_ids():
    conditions = [_make_condition() for _ in range(19)]
    with pytest.raises(ValueError):
        _validate_registry(conditions)


def test_validation_rejects_unknown_strategy():
    conditions = [_make_condition(condition_id=f"C{i}") for i in range(19)]
    conditions[0] = _make_condition(condition_id="C0", strategy_name="V999")
    with pytest.raises(ValueError):
        _validate_registry(conditions)


def test_validation_rejects_unknown_model():
    conditions = [_make_condition(condition_id=f"C{i}") for i in range(19)]
    conditions[0] = _make_condition(condition_id="C0", model_key="M999")
    with pytest.raises(ValueError):
        _validate_registry(conditions)


def test_validation_rejects_incorrect_ppas_assignment():
    conditions = [_make_condition(condition_id=f"C{i}") for i in range(19)]
    conditions[0] = _make_condition(condition_id="C0", apply_ppas=False)
    with pytest.raises(ValueError):
        _validate_registry(conditions)


def test_package_exports_are_available():
    import kgsemembed.pipeline as pipeline

    for name in (
        "EXPERIMENT_CONDITIONS",
        "ExperimentCondition",
        "get_condition",
        "get_conditions_for_dataset",
        "get_conditions_for_group",
    ):
        assert hasattr(pipeline, name)
