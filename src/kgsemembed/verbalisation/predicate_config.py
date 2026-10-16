"""Predicate vocabularies for the controlled PPAS ablation (#126, #128).

A vocabulary fixes two things independently of each other:

* **eligibility** -- ``covered_predicates``, the predicates kept when a policy
  filters with ``filter_mode="covered"``; under ``filter_mode="all"`` every
  predicate is eligible and this set is not consulted;
* **priority** -- ``tier_lists``, the per-entity-type tiers the PPAS order
  visits, highest priority first.

Fallback policy for predicates absent from every tier: they are visited after
the last tier, in canonical ``(predicate URI, object term)`` order.  This holds
for unknown predicates and for covered-but-untiered ones alike.

Vocabularies
------------
``base``
    The legacy PPAS tier lists of :mod:`kgsemembed.verbalisation.ppas`, copied
    at import time, so its ordering is the #122 PPAS ordering by construction.
    Its covered set is the union of the tiered predicates.
``fma_extended``
    ``base`` plus three FMA annotation predicates, assigned by their declared
    meaning in the FMA ontology, never by alignment outcomes:

    * ``fma:preferred_name`` -- the term's preferred English name, the role
      ``skos:prefLabel`` plays elsewhere: class tier 0 (lexical);
    * ``fma:synonym`` -- an alternative English name, the role
      ``skos:altLabel`` plays elsewhere: class tier 0 (lexical);
    * ``fma:FMAID`` -- an opaque numeric identifier with no counterpart in
      SNOMED CT, so it carries no cross-ontology lexical signal.  It is covered
      (a covered filter keeps it) but untiered, so it falls back after every
      tier rather than sharing the lexical tier.

    The FMA namespace is the one D1's ``target.rdf`` declares for its
    annotation properties, ``http://purl.org/sig/ont/fma/``.  FMA class URIs use
    a different namespace (``http://purl.org/sig/ont/fma.owl#``), which is why
    predicate URIs are matched in full and never by prefix name.
"""

import copy
import hashlib
import json
from dataclasses import dataclass, field

from kgsemembed.verbalisation.ppas import (
    CLASS_TIER_LIST,
    INSTANCE_TIER_LIST,
    PREDICATE_TIER_LIST,
)

ENTITY_TYPES = ("class", "instance", "predicate")

PREFIXES: dict[str, str] = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "obo": "http://purl.obolibrary.org/obo/",
    "fma": "http://purl.org/sig/ont/fma/",
}

LEXICAL = "lexical"
IDENTIFIER = "identifier"


def expand_predicate(name: str) -> str:
    """
    Expand a ``prefix:local`` name to a full URI; full URIs pass through.

    Parameters
    ----------
    name : str
        A prefixed name such as ``"fma:synonym"`` or a full ``http(s)`` URI.

    Returns
    -------
    str
        The full predicate URI.

    Raises
    ------
    ValueError
        If the prefix is not in :data:`PREFIXES`.
    """
    if name.startswith(("http://", "https://")):
        return name
    prefix, _, local = name.partition(":")
    if prefix not in PREFIXES or not local:
        raise ValueError(f"Cannot expand predicate name {name!r}; known prefixes: {sorted(PREFIXES)}.")
    return PREFIXES[prefix] + local


@dataclass(frozen=True)
class PredicateVocabulary:
    """
    Predicate eligibility and priority, fingerprinted by ``config_hash``.

    Attributes
    ----------
    name : str
        Vocabulary identifier, e.g. ``"base"`` or ``"fma_extended"``.
    description : str
        Human-readable rationale.
    covered_predicates : frozenset[str]
        Full predicate URIs kept by ``filter_mode="covered"``.
    tier_lists : dict[str, list[list[str]]]
        Full predicate URIs grouped by tier for each entity type.
    predicate_roles : dict[str, str]
        Role (``"lexical"`` or ``"identifier"``) of each predicate the
        vocabulary adds to ``base``, keyed by full URI.
    config_hash : str
        SHA-256 of name, tiers, covered set and roles; derived, never passed.
    """

    name: str
    description: str
    covered_predicates: frozenset[str]
    tier_lists: dict[str, list[list[str]]]
    predicate_roles: dict[str, str] = field(default_factory=dict)
    config_hash: str = field(init=False)

    def __post_init__(self) -> None:
        _validate_vocabulary(self)
        object.__setattr__(self, "config_hash", _config_hash(self))

    def tier_list_for(self, entity_type: str) -> list[list[str]]:
        """
        Return this vocabulary's tiers for an entity type.

        Parameters
        ----------
        entity_type : str
            One of :data:`ENTITY_TYPES`.

        Returns
        -------
        list[list[str]]
            Predicate URIs grouped by tier, highest priority first.

        Raises
        ------
        ValueError
            If ``entity_type`` is not supported.
        """
        if entity_type not in self.tier_lists:
            raise ValueError(
                f"Unsupported entity type: {entity_type!r}. Expected one of {sorted(self.tier_lists)}."
            )
        return self.tier_lists[entity_type]

    def tiers_hash(self) -> str:
        """
        Fingerprint only the tier lists, the input to PPAS ordering.

        Returns
        -------
        str
            SHA-256 hex digest of ``tier_lists``.
        """
        return _digest(self.tier_lists)


def _tiered(tier_lists: dict[str, list[list[str]]]) -> set[str]:
    """Return every predicate that appears in some tier."""
    return {p for tiers in tier_lists.values() for tier in tiers for p in tier}


def _validate_vocabulary(vocabulary: PredicateVocabulary) -> None:
    """Raise if tiers, covered set and roles are inconsistent or not full URIs."""
    if set(vocabulary.tier_lists) != set(ENTITY_TYPES):
        raise ValueError(f"{vocabulary.name}: tier lists must cover exactly {ENTITY_TYPES}.")
    tiered = _tiered(vocabulary.tier_lists)
    if not tiered <= vocabulary.covered_predicates:
        raise ValueError(f"{vocabulary.name}: tiered predicates absent from covered_predicates.")
    uris = vocabulary.covered_predicates | set(vocabulary.predicate_roles)
    if any(expand_predicate(uri) != uri for uri in uris):
        raise ValueError(f"{vocabulary.name}: predicates must be full URIs.")
    if not set(vocabulary.predicate_roles) <= vocabulary.covered_predicates:
        raise ValueError(f"{vocabulary.name}: every role-annotated predicate must be covered.")


def _digest(payload: object) -> str:
    """Return the SHA-256 of a canonical JSON encoding."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _config_hash(vocabulary: PredicateVocabulary) -> str:
    """Fingerprint everything that determines a vocabulary's selections."""
    return _digest({
        "name": vocabulary.name,
        "tier_lists": vocabulary.tier_lists,
        "covered_predicates": sorted(vocabulary.covered_predicates),
        "predicate_roles": vocabulary.predicate_roles,
    })


def _legacy_tier_lists() -> dict[str, list[list[str]]]:
    """Deep-copy the module-level PPAS tier lists of :mod:`ppas`."""
    return {
        "class": copy.deepcopy(CLASS_TIER_LIST),
        "instance": copy.deepcopy(INSTANCE_TIER_LIST),
        "predicate": copy.deepcopy(PREDICATE_TIER_LIST),
    }


def _make_base() -> PredicateVocabulary:
    """Build the legacy vocabulary: PPAS tiers, covered set = tiered predicates."""
    tier_lists = _legacy_tier_lists()
    return PredicateVocabulary(
        name="base",
        description="Legacy PPAS tier lists; covered predicates are exactly the tiered ones.",
        covered_predicates=frozenset(_tiered(tier_lists)),
        tier_lists=tier_lists,
    )


FMA_EXTENSION: dict[str, str] = {
    expand_predicate("fma:preferred_name"): LEXICAL,
    expand_predicate("fma:synonym"): LEXICAL,
    expand_predicate("fma:FMAID"): IDENTIFIER,
}


def _make_fma_extended() -> PredicateVocabulary:
    """Build ``base`` plus FMA lexical predicates in class tier 0 and FMAID untiered."""
    tier_lists = _legacy_tier_lists()
    lexical = [uri for uri, role in FMA_EXTENSION.items() if role == LEXICAL]
    tier_lists["class"][0] = tier_lists["class"][0] + lexical
    return PredicateVocabulary(
        name="fma_extended",
        description=(
            "Base plus fma:preferred_name and fma:synonym in class tier 0 (lexical) "
            "and fma:FMAID covered but untiered (identifier)."
        ),
        covered_predicates=frozenset(_tiered(tier_lists) | set(FMA_EXTENSION)),
        tier_lists=tier_lists,
        predicate_roles=dict(FMA_EXTENSION),
    )


PER_PREDICATE_CONFIGS: dict[str, PredicateVocabulary] = {
    vocabulary.name: vocabulary for vocabulary in (_make_base(), _make_fma_extended())
}


def get_predicate_config(name: str) -> PredicateVocabulary:
    """
    Retrieve a registered predicate vocabulary by name.

    Parameters
    ----------
    name : str
        Vocabulary name, ``"base"`` or ``"fma_extended"``.

    Returns
    -------
    PredicateVocabulary
        The requested vocabulary.

    Raises
    ------
    KeyError
        If the name is not registered.
    """
    if name not in PER_PREDICATE_CONFIGS:
        raise KeyError(
            f"Predicate configuration {name!r} not found. Available: {sorted(PER_PREDICATE_CONFIGS)}."
        )
    return PER_PREDICATE_CONFIGS[name]
