"""Predicate vocabulary and eligibility configuration for controlled PPAS ablation (#126).

Separates predicate filtering (eligibility) from ordering (priority), enabling
independent measurement of which effects explain PPAS results.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional

from rdflib import URIRef

# ---------------------------------------------------------------------------
# Standard namespace URIs and FMA-specific predicates
# ---------------------------------------------------------------------------
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
_SKOS = "http://www.w3.org/2004/02/skos/core#"
_OBO = "http://purl.obolibrary.org/obo/"
_OWL = "http://www.w3.org/2002/07/owl#"
_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_FMA = "http://purl.org/oasis/names/tc/ciq/xal/3.0#"  # FMA namespace base

# FMA-specific predicates not in the base tier configuration
# These are documented in Issue #126 as important for D1 (SNOMED-FMA Body)
_FMA_PREFERRED_NAME = f"{_FMA}preferred_name"
_FMA_SYNONYM = f"{_FMA}synonym"
_FMA_FMAID = f"{_FMA}FMAID"

# Note: FMA uses multiple namespaces; this is the OBO Foundry standard.
# If datasets use alternate FMA URI schemes, they must be normalized first.


@dataclass(frozen=True)
class PredicateVocabulary:
    """
    Explicit predicate eligibility and priority configuration.

    Separates which predicates are eligible (covered_predicates) from how
    eligible predicates are ordered (tier_lists). This allows independent
    testing of filtering and prioritization effects.

    Attributes
    ----------
    name : str
        Configuration identifier (e.g., "base", "covered_only", "fma_extended").
    description : str
        Human-readable documentation of the configuration.
    covered_predicates : frozenset[str]
        Fully-qualified predicate URIs that are eligible for selection.
        When filter_mode="covered", only these predicates are included.
        When filter_mode="all", this set is ignored (all predicates eligible).
    tier_lists : dict[str, list[list[str]]]
        Predicate tiers by entity type ("class", "instance", "predicate").
        Each entity type maps to tiers [tier_0, tier_1, ...] where tier_0 is
        highest priority. Predicates not in any tier are untiered (lowest).
    fma_special_predicates : dict[str, str]
        FMA-specific predicates and their documented role.
        Example: {"fma:preferred_name": "lexical", "fma:FMAID": "identifier"}
    config_hash : str
        SHA-256 of the JSON-serialized configuration (name, tier_lists,
        covered_predicates, fma_special_predicates). Frozen at creation.
    """

    name: str
    description: str
    covered_predicates: FrozenSet[str]
    tier_lists: Dict[str, list[list[str]]]
    fma_special_predicates: Dict[str, str]
    config_hash: str

    def __post_init__(self) -> None:
        """Validate configuration consistency (called after dataclass init)."""
        # Ensure all tier-listed predicates are in covered_predicates
        if not self._all_tiered_predicates_covered():
            raise ValueError(
                f"Config {self.name!r}: tier lists contain predicates absent from "
                f"covered_predicates. Tier lists must be a subset of covered predicates."
            )

    def _all_tiered_predicates_covered(self) -> bool:
        """Check that every predicate in tier_lists is in covered_predicates."""
        for entity_type, tiers in self.tier_lists.items():
            for tier in tiers:
                for predicate in tier:
                    if predicate not in self.covered_predicates:
                        return False
        return True


def _normalise_predicate_uri(uri: str) -> str:
    """
    Normalise a predicate URI to a canonical form.

    Supports full URIs (e.g. "http://...") and rdflib.URIRef objects.

    Parameters
    ----------
    uri : str
        Predicate URI.

    Returns
    -------
    str
        Canonical string form of the URI.
    """
    return str(URIRef(uri))


def _make_config_hash(
    name: str,
    tier_lists: Dict[str, list[list[str]]],
    covered_predicates: FrozenSet[str],
    fma_special_predicates: Dict[str, str],
) -> str:
    """
    Create a SHA-256 fingerprint of the configuration.

    Parameters
    ----------
    name : str
        Configuration name.
    tier_lists : dict[str, list[list[str]]]
        Tier lists by entity type.
    covered_predicates : frozenset[str]
        Eligible predicates.
    fma_special_predicates : dict[str, str]
        FMA predicate roles.

    Returns
    -------
    str
        SHA-256 hex digest.
    """
    payload = {
        "name": name,
        "tier_lists": tier_lists,
        "covered_predicates": sorted(covered_predicates),
        "fma_special_predicates": fma_special_predicates,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _make_base_config() -> PredicateVocabulary:
    """
    Create the base configuration.

    Includes all predicates from the current PPAS tier lists (CLASS_TIER_LIST,
    INSTANCE_TIER_LIST, PREDICATE_TIER_LIST). Untiered predicates are
    eligible but receive lowest priority.

    This configuration represents the existing PPAS behavior and is the
    baseline for P2 (all-ppas ordering).

    Returns
    -------
    PredicateVocabulary
        Base configuration.
    """
    # All tier-listed predicates from ppas.py (as of Issue #122)
    tier_lists = {
        "class": [
            [  # Tier 0: core lexical
                f"{_RDFS}label",
                f"{_SKOS}prefLabel",
                f"{_SKOS}altLabel",
                f"{_OBO}hasExactSynonym",
                f"{_OBO}hasSynonym",
                f"{_OBO}IAO_0000115",
                f"{_RDFS}comment",
                f"{_SKOS}definition",
            ],
            [  # Tier 1: structural type
                f"{_RDF}type",
                f"{_RDFS}subClassOf",
                f"{_RDFS}domain",
                f"{_RDFS}range",
            ],
            [  # Tier 2: equivalence
                f"{_OWL}equivalentClass",
                f"{_SKOS}exactMatch",
            ],
            [],  # Tier 3 - populated dynamically per dataset
            [],  # Tier 4
            [],  # Tier 5
        ],
        "instance": [
            [  # Tier 0
                f"{_RDF}type",
                f"{_RDFS}label",
                f"{_SKOS}prefLabel",
            ],
            [],  # Tier 1 -- datatype properties / functional object properties
            [],  # Tier 2
            [  # Tier 3
                f"{_OWL}sameAs",
                f"{_SKOS}exactMatch",
            ],
        ],
        "predicate": [
            [  # Tier 0
                f"{_RDFS}label",
                f"{_SKOS}prefLabel",
                f"{_RDFS}comment",
            ],
            [  # Tier 1
                f"{_RDFS}domain",
                f"{_RDFS}range",
            ],
            [  # Tier 2
                f"{_RDFS}subPropertyOf",
                f"{_OWL}inverseOf",
                f"{_OWL}equivalentProperty",
            ],
        ],
    }

    # Collect all tier-listed predicates as the "covered" set
    covered = set()
    for entity_type, tiers in tier_lists.items():
        for tier in tiers:
            covered.update(tier)

    config_hash = _make_config_hash("base", tier_lists, frozenset(covered), {})

    return PredicateVocabulary(
        name="base",
        description=(
            "All predicates from PPAS tier lists (CLASS, INSTANCE, PREDICATE). "
            "Untiered predicates are eligible but lowest priority. "
            "Represents existing PPAS behavior (P2 baseline)."
        ),
        covered_predicates=frozenset(covered),
        tier_lists=tier_lists,
        fma_special_predicates={},
        config_hash=config_hash,
    )


def _make_covered_only_config() -> PredicateVocabulary:
    """
    Create the covered-only configuration.

    Same tier lists as base, but untiered predicates are ineligible.
    This allows isolation of the filtering effect (P3 vs P1).

    Returns
    -------
    PredicateVocabulary
        Covered-only configuration.
    """
    base = _make_base_config()
    # Reuse tier_lists as-is; filter is applied at selection time
    config_hash = _make_config_hash(
        "covered_only",
        base.tier_lists,
        base.covered_predicates,
        {},
    )
    return PredicateVocabulary(
        name="covered_only",
        description=(
            "Only predicates in the PPAS tier lists (no untiered). "
            "Used to isolate filtering effect: P3 (covered-det) vs P1 (all-det), "
            "and P4 (covered-ppas) vs P3 (covered-det)."
        ),
        covered_predicates=base.covered_predicates,
        tier_lists=base.tier_lists,
        fma_special_predicates={},
        config_hash=config_hash,
    )


def _make_fma_extended_config() -> PredicateVocabulary:
    """
    Create the FMA-extended configuration.

    Extends base with FMA-specific predicates that were absent from the
    original tier lists but are semantically important for D1 (SNOMED-FMA Body).

    FMA-specific predicates:
    - fma:preferred_name: Lexical/label predicate (similar to rdfs:label)
    - fma:synonym: Lexical/label predicate (similar to skos:altLabel)
    - fma:FMAID: Identifier predicate (documented separately, not auto-labeled)

    Returns
    -------
    PredicateVocabulary
        FMA-extended configuration.
    """
    base = _make_base_config()

    # Extend tier lists: add FMA predicates to appropriate tiers
    tier_lists = {
        entity_type: [tier[:] for tier in tiers]  # Deep copy
        for entity_type, tiers in base.tier_lists.items()
    }

    # Add FMA predicates to class tier 0 (lexical)
    tier_lists["class"][0] = (
        tier_lists["class"][0]
        + [_FMA_PREFERRED_NAME, _FMA_SYNONYM]
    )

    # Extend covered_predicates
    covered = set(base.covered_predicates)
    covered.update({_FMA_PREFERRED_NAME, _FMA_SYNONYM, _FMA_FMAID})

    # Document FMA special cases
    fma_special = {
        "fma:preferred_name": "lexical (similar to rdfs:label)",
        "fma:synonym": "lexical (similar to skos:altLabel)",
        "fma:FMAID": (
            "identifier (numeric or alphanumeric ID); not automatically treated "
            "as a label; included for completeness but may not contribute to matching"
        ),
    }

    config_hash = _make_config_hash(
        "fma_extended",
        tier_lists,
        frozenset(covered),
        fma_special,
    )

    return PredicateVocabulary(
        name="fma_extended",
        description=(
            "Base configuration extended with FMA-specific predicates "
            "(fma:preferred_name, fma:synonym, fma:FMAID). "
            "Evaluates whether vocabulary coverage improves matching on D1. "
            "Used for P5 (extended-ppas)."
        ),
        covered_predicates=frozenset(covered),
        tier_lists=tier_lists,
        fma_special_predicates=fma_special,
        config_hash=config_hash,
    )


# Global registry of predicate configurations
# Keyed by configuration name; frozen at module import time
PER_PREDICATE_CONFIGS: Dict[str, PredicateVocabulary] = {
    config.name: config
    for config in [
        _make_base_config(),
        _make_covered_only_config(),
        _make_fma_extended_config(),
    ]
}


def get_predicate_config(name: str) -> PredicateVocabulary:
    """
    Retrieve a registered predicate configuration by name.

    Parameters
    ----------
    name : str
        Configuration name (e.g., "base", "covered_only", "fma_extended").

    Returns
    -------
    PredicateVocabulary
        The requested configuration.

    Raises
    ------
    KeyError
        If the configuration name is not registered.
    """
    if name not in PER_PREDICATE_CONFIGS:
        available = sorted(PER_PREDICATE_CONFIGS.keys())
        raise KeyError(
            f"Predicate configuration {name!r} not found. "
            f"Available: {available}."
        )
    return PER_PREDICATE_CONFIGS[name]
