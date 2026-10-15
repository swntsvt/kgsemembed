"""Tests for predicate vocabulary and eligibility configurations (#126)."""

import pytest
from rdflib import URIRef

from kgsemembed.verbalisation.predicate_config import (
    PER_PREDICATE_CONFIGS,
    PredicateVocabulary,
    get_predicate_config,
)


class TestPredicateConfigRegistry:
    """Test the predicate configuration registry."""

    def test_base_config_exists(self):
        """Base configuration must be registered."""
        config = get_predicate_config("base")
        assert config.name == "base"
        assert len(config.covered_predicates) > 0
        assert len(config.tier_lists) == 3

    def test_covered_only_config_exists(self):
        """Covered-only configuration must be registered."""
        config = get_predicate_config("covered_only")
        assert config.name == "covered_only"
        assert len(config.covered_predicates) > 0

    def test_fma_extended_config_exists(self):
        """FMA-extended configuration must be registered."""
        config = get_predicate_config("fma_extended")
        assert config.name == "fma_extended"
        assert len(config.covered_predicates) > 0
        assert len(config.fma_special_predicates) == 3

    def test_get_unknown_config_raises(self):
        """Requesting a non-existent configuration raises KeyError."""
        with pytest.raises(KeyError):
            get_predicate_config("nonexistent")

    def test_all_configs_have_unique_names(self):
        """All registered configurations have distinct names."""
        names = [c.name for c in PER_PREDICATE_CONFIGS.values()]
        assert len(names) == len(set(names))

    def test_all_configs_have_unique_hashes(self):
        """All registered configurations have distinct config hashes."""
        hashes = [c.config_hash for c in PER_PREDICATE_CONFIGS.values()]
        assert len(hashes) == len(set(hashes))


class TestPredicateURINormalisation:
    """Test URI normalisation for predicate comparison."""

    def test_uri_refs_are_normalised(self):
        """URIRef objects are normalised to strings."""
        config = get_predicate_config("base")
        uris = list(config.covered_predicates)
        assert all(isinstance(uri, str) for uri in uris)
        assert all(uri.startswith("http://") for uri in uris)

    def test_full_uris_match_predicates(self):
        """Predicates in tier lists use full URIs."""
        config = get_predicate_config("base")
        for entity_type, tiers in config.tier_lists.items():
            for tier in tiers:
                for predicate in tier:
                    assert isinstance(predicate, str)
                    assert predicate.startswith("http://")
                    assert predicate in config.covered_predicates


class TestFMASpecialPredicates:
    """Test FMA-specific predicate handling."""

    def test_fma_predicates_not_in_base(self):
        """FMA predicates are absent from base config."""
        base = get_predicate_config("base")
        fma_extended = get_predicate_config("fma_extended")
        # Base should not have FMA special predicates
        for fma_pred in fma_extended.fma_special_predicates.keys():
            # Extract full URI from key
            if fma_pred.startswith("fma:"):
                fma_uri = f"http://purl.org/oasis/names/tc/ciq/xal/3.0#{fma_pred.split(':')[1]}"
                assert fma_uri not in base.covered_predicates

    def test_fma_extended_contains_special_predicates(self):
        """FMA-extended config documents its special predicates."""
        config = get_predicate_config("fma_extended")
        assert "fma:preferred_name" in config.fma_special_predicates
        assert "fma:synonym" in config.fma_special_predicates
        assert "fma:FMAID" in config.fma_special_predicates

    def test_fma_predicates_have_documented_roles(self):
        """FMA special predicates have role descriptions."""
        config = get_predicate_config("fma_extended")
        for pred, role in config.fma_special_predicates.items():
            assert isinstance(role, str)
            assert len(role) > 0


class TestConfigConsistency:
    """Test that configurations are internally consistent."""

    def test_all_tiered_predicates_are_covered(self):
        """Every predicate in tier lists must be in covered_predicates."""
        for config in PER_PREDICATE_CONFIGS.values():
            for entity_type, tiers in config.tier_lists.items():
                for tier in tiers:
                    for predicate in tier:
                        assert (
                            predicate in config.covered_predicates
                        ), f"{config.name}/{entity_type}: {predicate} not in covered_predicates"

    def test_each_entity_type_has_tier_list(self):
        """Each config must have tiers for class, instance, predicate."""
        expected_types = {"class", "instance", "predicate"}
        for config in PER_PREDICATE_CONFIGS.values():
            assert set(config.tier_lists.keys()) == expected_types

    def test_base_config_is_complete(self):
        """Base config must have all three entity types configured."""
        config = get_predicate_config("base")
        for entity_type in ("class", "instance", "predicate"):
            assert entity_type in config.tier_lists
            assert len(config.tier_lists[entity_type]) > 0


class TestConfigDifferences:
    """Test intentional differences between configurations."""

    def test_covered_only_has_same_or_fewer_predicates(self):
        """Covered-only should have the same or fewer than base."""
        base = get_predicate_config("base")
        covered_only = get_predicate_config("covered_only")
        assert len(covered_only.covered_predicates) <= len(base.covered_predicates)

    def test_fma_extended_has_more_than_base(self):
        """FMA-extended should have more predicates than base."""
        base = get_predicate_config("base")
        fma_extended = get_predicate_config("fma_extended")
        assert len(fma_extended.covered_predicates) > len(base.covered_predicates)

    def test_fma_extended_has_fma_tier_entries(self):
        """FMA-extended should add FMA predicates to class tier 0."""
        fma_extended = get_predicate_config("fma_extended")
        class_tier_0 = fma_extended.tier_lists["class"][0]
        fma_preds = {p for p in class_tier_0 if "xal" in p or "preferred_name" in p or "synonym" in p}
        assert len(fma_preds) > 0
