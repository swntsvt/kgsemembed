"""Tests for the predicate vocabularies of the controlled PPAS ablation (#126, #128)."""

import pytest

from kgsemembed.verbalisation.ppas import (
    CLASS_TIER_LIST,
    INSTANCE_TIER_LIST,
    PREDICATE_TIER_LIST,
)
from kgsemembed.verbalisation.predicate_config import (
    IDENTIFIER,
    LEXICAL,
    PER_PREDICATE_CONFIGS,
    PredicateVocabulary,
    expand_predicate,
    get_predicate_config,
)
from kgsemembed.verbalisation.selection import tier_config_digest

FMA = "http://purl.org/sig/ont/fma/"
FMA_PREFERRED_NAME = f"{FMA}preferred_name"
FMA_SYNONYM = f"{FMA}synonym"
FMA_FMAID = f"{FMA}FMAID"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"


def _tiered(vocabulary: PredicateVocabulary, entity_type: str) -> set[str]:
    return {p for tier in vocabulary.tier_list_for(entity_type) for p in tier}


# ---------------------------------------------------------------------------
# Registry and provenance
# ---------------------------------------------------------------------------


def test_registry_holds_base_its_correction_and_the_two_extended_vocabularies():
    assert sorted(PER_PREDICATE_CONFIGS) == ["base", "base_v2", "fma_extended", "obo_extended"]


def test_unknown_vocabulary_raises():
    with pytest.raises(KeyError):
        get_predicate_config("covered_only")


def test_vocabularies_have_distinct_config_and_tier_hashes():
    base, extended = get_predicate_config("base"), get_predicate_config("fma_extended")
    assert base.config_hash != extended.config_hash
    assert base.tiers_hash() != extended.tiers_hash()


def test_base_tiers_hash_equals_the_legacy_tier_digest():
    assert get_predicate_config("base").tiers_hash() == tier_config_digest()


def test_config_hash_tracks_the_tiers_and_cannot_be_passed_in():
    base = get_predicate_config("base")
    changed = {**base.tier_lists, "class": [[RDFS_LABEL]]}
    other = PredicateVocabulary("base", "", base.covered_predicates, changed)
    assert other.config_hash != base.config_hash
    with pytest.raises(TypeError):
        PredicateVocabulary("x", "", base.covered_predicates, base.tier_lists, {}, "forged")


# ---------------------------------------------------------------------------
# Base preserves the legacy PPAS tiers
# ---------------------------------------------------------------------------


def test_base_tiers_equal_the_legacy_ppas_tier_lists():
    base = get_predicate_config("base")
    assert base.tier_list_for("class") == CLASS_TIER_LIST
    assert base.tier_list_for("instance") == INSTANCE_TIER_LIST
    assert base.tier_list_for("predicate") == PREDICATE_TIER_LIST


def test_base_tiers_are_copies_of_the_module_constants():
    base = get_predicate_config("base")
    assert base.tier_list_for("class") is not CLASS_TIER_LIST
    assert base.tier_list_for("class")[0] is not CLASS_TIER_LIST[0]


def test_base_covers_exactly_its_tiered_predicates():
    base = get_predicate_config("base")
    tiered = set().union(*(_tiered(base, t) for t in ("class", "instance", "predicate")))
    assert base.covered_predicates == tiered


def test_unsupported_entity_type_raises():
    with pytest.raises(ValueError):
        get_predicate_config("base").tier_list_for("individual")


# ---------------------------------------------------------------------------
# FMA extension (#128: namespace and roles)
# ---------------------------------------------------------------------------


def test_fma_predicates_use_the_namespace_declared_by_d1():
    extended = get_predicate_config("fma_extended")
    assert set(extended.predicate_roles) == {FMA_PREFERRED_NAME, FMA_SYNONYM, FMA_FMAID}
    assert not any("oasis" in uri for uri in extended.covered_predicates)


def test_fma_lexical_predicates_join_class_tier_zero():
    extended = get_predicate_config("fma_extended")
    assert {FMA_PREFERRED_NAME, FMA_SYNONYM} <= set(extended.tier_list_for("class")[0])
    assert extended.predicate_roles[FMA_PREFERRED_NAME] == LEXICAL
    assert extended.predicate_roles[FMA_SYNONYM] == LEXICAL


def test_fmaid_is_covered_but_never_tiered():
    extended = get_predicate_config("fma_extended")
    assert extended.predicate_roles[FMA_FMAID] == IDENTIFIER
    assert FMA_FMAID in extended.covered_predicates
    for entity_type in ("class", "instance", "predicate"):
        assert FMA_FMAID not in _tiered(extended, entity_type)


def test_fma_extension_leaves_other_tiers_equal_to_base():
    base, extended = get_predicate_config("base"), get_predicate_config("fma_extended")
    assert extended.tier_list_for("class")[1:] == base.tier_list_for("class")[1:]
    assert extended.tier_list_for("instance") == base.tier_list_for("instance")
    assert extended.covered_predicates - base.covered_predicates == set(extended.predicate_roles)


def test_base_covers_no_fma_predicate():
    base = get_predicate_config("base")
    assert not any(uri.startswith(FMA) for uri in base.covered_predicates)


# ---------------------------------------------------------------------------
# Prefix expansion and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("fma:synonym", FMA_SYNONYM),
    ("rdfs:label", RDFS_LABEL),
    (FMA_FMAID, FMA_FMAID),
])
def test_expand_predicate_resolves_prefixes_and_keeps_full_uris(name, expected):
    assert expand_predicate(name) == expected


@pytest.mark.parametrize("name", ["xal:synonym", "fma:", "synonym"])
def test_expand_predicate_rejects_unknown_or_empty_names(name):
    with pytest.raises(ValueError):
        expand_predicate(name)


def test_vocabulary_rejects_prefixed_names():
    base = get_predicate_config("base")
    with pytest.raises(ValueError):
        PredicateVocabulary("x", "", base.covered_predicates | {"fma:synonym"}, base.tier_lists)


def test_vocabulary_rejects_tiered_predicates_outside_the_covered_set():
    base = get_predicate_config("base")
    with pytest.raises(ValueError):
        PredicateVocabulary("x", "", frozenset(), base.tier_lists)


# ---------------------------------------------------------------------------
# OBO-in-OWL extension (D2)
# ---------------------------------------------------------------------------

OBO = "http://www.geneontology.org/formats/oboInOwl#"


def test_obo_predicates_use_the_namespace_declared_by_d2():
    extended = get_predicate_config("obo_extended")
    assert all(uri.startswith(OBO) for uri in extended.predicate_roles)
    assert f"{OBO}hasRelatedSynonym" in extended.covered_predicates


def test_obo_synonyms_and_definition_join_class_tier_zero():
    extended = get_predicate_config("obo_extended")
    lexical = {uri for uri, role in extended.predicate_roles.items() if role == LEXICAL}
    assert lexical == {f"{OBO}{name}" for name in (
        "hasExactSynonym", "hasRelatedSynonym", "hasNarrowSynonym", "hasBroadSynonym",
        "hasSynonym", "hasDefinition")}
    assert lexical <= set(extended.tier_list_for("class")[0])


def test_obo_identifiers_are_covered_but_never_tiered():
    extended = get_predicate_config("obo_extended")
    identifiers = {uri for uri, role in extended.predicate_roles.items() if role == IDENTIFIER}
    assert identifiers == {f"{OBO}hasAlternativeId", f"{OBO}hasDbXref"}
    for entity_type in ("class", "instance", "predicate"):
        assert not identifiers & _tiered(extended, entity_type)


def test_obo_extension_leaves_base_and_fma_extended_unchanged():
    base, obo = get_predicate_config("base"), get_predicate_config("obo_extended")
    assert obo.tier_list_for("class")[1:] == base.tier_list_for("class")[1:]
    assert obo.covered_predicates - base.covered_predicates == set(obo.predicate_roles)
    assert get_predicate_config("fma_extended").config_hash.startswith("1ebca7d18c8f")
    assert base.config_hash.startswith("eb299c6ac2d9")


def test_base_still_lists_the_unused_obo_synonym_uris():
    base = get_predicate_config("base")
    assert "http://purl.obolibrary.org/obo/hasExactSynonym" in base.covered_predicates
    assert not any(uri.startswith(OBO) for uri in base.covered_predicates)


# ---------------------------------------------------------------------------
# base_v2: corrected OBO-in-OWL synonym URIs (#130)
# ---------------------------------------------------------------------------

_HISTORICAL_HASHES = {
    "base": ("eb299c6ac2d9cc54735618672736ad573ad3c71c725543037a2e86b8457558bb",
             "34f65847c146167e7356959bfc6fe5717a43a4be05c9993fcc01a898aa12776d"),
    "fma_extended": ("1ebca7d18c8f2fa2308e4db164401a0f41b819f990504fd187a6f3a3ff52b29d",
                     "2de92887126e996a83490c58b060121437d56186c2859b0af408bd14b64d0c00"),
    "obo_extended": ("55448ec69c984b349304e7cd32f4fdcfd0def120424fd258cdf78e7297524592",
                     "ed5a6e4dd9b9f474cdbba24936161806b1d47b77f71d8ba278d34ae2a02ef0c6"),
}
_WRONG = {"http://purl.obolibrary.org/obo/hasExactSynonym", "http://purl.obolibrary.org/obo/hasSynonym"}
_RIGHT = {f"{OBO}hasExactSynonym", f"{OBO}hasSynonym"}
_FIXES = {
    "http://purl.obolibrary.org/obo/hasExactSynonym": f"{OBO}hasExactSynonym",
    "http://purl.obolibrary.org/obo/hasSynonym": f"{OBO}hasSynonym",
}


@pytest.mark.parametrize("name", sorted(_HISTORICAL_HASHES))
def test_historical_vocabularies_keep_their_hashes(name):
    vocabulary = get_predicate_config(name)
    assert (vocabulary.config_hash, vocabulary.tiers_hash()) == _HISTORICAL_HASHES[name]


def test_base_v2_remaps_exactly_the_two_obo_synonym_uris():
    from kgsemembed.verbalisation.predicate_config import BASE_V2_CORRECTIONS

    assert BASE_V2_CORRECTIONS == _FIXES
    base, corrected = get_predicate_config("base"), get_predicate_config("base_v2")
    assert _RIGHT <= corrected.covered_predicates and not _WRONG & corrected.covered_predicates
    assert corrected.covered_predicates - _RIGHT == base.covered_predicates - _WRONG


def test_base_v2_keeps_every_tier_position():
    base, corrected = get_predicate_config("base"), get_predicate_config("base_v2")
    for entity_type in ("class", "instance", "predicate"):
        remapped = [[_FIXES.get(uri, uri) for uri in tier] for tier in base.tier_list_for(entity_type)]
        assert corrected.tier_list_for(entity_type) == remapped


def test_base_v2_adds_no_other_synonym_scope():
    corrected = get_predicate_config("base_v2")
    for scope in ("hasRelatedSynonym", "hasNarrowSynonym", "hasBroadSynonym"):
        assert f"{OBO}{scope}" not in corrected.covered_predicates
    assert corrected.predicate_roles == {}


def test_base_v2_has_its_own_hash_and_leaves_the_ppas_lists_unchanged():
    assert get_predicate_config("base_v2").config_hash not in {h for h, _ in _HISTORICAL_HASHES.values()}
    assert set(CLASS_TIER_LIST[0]) & _WRONG == _WRONG
    assert get_predicate_config("base").tiers_hash() == tier_config_digest()
