"""Predicate roles for the predicate-role ablation (#131).

A role map assigns every predicate that occurs in D1 or D2 one role, fixed from
the property's declaration, its ``rdfs:label`` and sample values in the source
and target graphs -- never from alignments or matching results.  The ablation
removes items by role, so a predicate missing from the map is an error, not a
silent default.

Roles (``roles_v1``)
--------------------
``core``
    ``rdfs:label``, ``rdf:type``, ``rdfs:subClassOf``: never removed.
``preferred_name``
    ``skos:prefLabel`` (SNOMED) and ``fma:preferred_name``.  On D1 these equal
    ``rdfs:label`` for 99% of entities after case and underscore
    normalisation, so removing them mainly removes a duplicated name.
``synonym``
    ``skos:altLabel``, ``fma:synonym``, ``fma:slot_synonym`` and the
    OBO-in-OWL synonym family.
``definition``
    ``skos:definition``, ``fma:definition``, ``oboInOwl:hasDefinition``,
    ``obo:IAO_0000115`` and ``rdfs:comment`` (whose D1 values read
    "Old definition: ...").
``identifier``
    FMA and atlas identifiers and codes, ``oboInOwl:hasAlternativeId`` and
    ``oboInOwl:hasDbXref``.
``non_english``
    ``fma:non-English_equivalent``.
``disjointness``
    ``owl:disjointWith``.
``other``
    Rare FMA annotations (atlas region names, homonym links, eponyms), under
    0.1% of D1 candidate items; removed only with every non-core role.
"""

import hashlib
import json
from dataclasses import dataclass, field

from kgsemembed.verbalisation.predicate_config import expand_predicate

CORE = "core"
PREFERRED_NAME = "preferred_name"
SYNONYM = "synonym"
DEFINITION = "definition"
IDENTIFIER = "identifier"
NON_ENGLISH = "non_english"
DISJOINTNESS = "disjointness"
OTHER = "other"
ROLES = (CORE, PREFERRED_NAME, SYNONYM, DEFINITION, IDENTIFIER, NON_ENGLISH, DISJOINTNESS, OTHER)
NON_CORE_ROLES = tuple(role for role in ROLES if role != CORE)

_ROLES_V1_NAMES: dict[str, tuple[str, ...]] = {
    CORE: ("rdfs:label", "rdf:type", "rdfs:subClassOf"),
    PREFERRED_NAME: ("skos:prefLabel", "fma:preferred_name"),
    SYNONYM: (
        "skos:altLabel", "fma:synonym", "fma:slot_synonym",
        "oboInOwl:hasExactSynonym", "oboInOwl:hasRelatedSynonym", "oboInOwl:hasNarrowSynonym",
        "oboInOwl:hasBroadSynonym", "oboInOwl:hasSynonym",
    ),
    DEFINITION: (
        "skos:definition", "fma:definition", "oboInOwl:hasDefinition", "obo:IAO_0000115", "rdfs:comment",
    ),
    IDENTIFIER: (
        "fma:FMAID", "fma:RadLex_ID", "fma:Talairach", "fma:AAL", "fma:DK__Freesurfer_", "fma:Neurolex",
        "fma:fma0329122", "fma:CMA_label", "oboInOwl:hasAlternativeId", "oboInOwl:hasDbXref",
    ),
    NON_ENGLISH: ("fma:non-English_equivalent",),
    DISJOINTNESS: ("owl:disjointWith",),
    OTHER: (
        "fma:JHU_DTI-81", "fma:JHU_White-Matter_Tractography_Atlas", "fma:homonym_for",
        "fma:homonym_of", "fma:Eponym",
    ),
}


class UnmappedPredicateError(KeyError):
    """Raised when an item's predicate has no role in the active role map."""


@dataclass(frozen=True)
class PredicateRoleMap:
    """
    One role per full predicate URI, fingerprinted by ``config_hash``.

    Attributes
    ----------
    name : str
        Role-map identifier, e.g. ``"roles_v1"``.
    roles : dict[str, str]
        Full predicate URI to one of :data:`ROLES`.
    config_hash : str
        SHA-256 of name and roles; derived, never passed.
    """

    name: str
    roles: dict[str, str]
    config_hash: str = field(init=False)

    def __post_init__(self) -> None:
        unknown = sorted(set(self.roles.values()) - set(ROLES))
        if unknown:
            raise ValueError(f"{self.name}: unknown roles {unknown}.")
        if any(expand_predicate(uri) != uri for uri in self.roles):
            raise ValueError(f"{self.name}: predicates must be full URIs.")
        encoded = json.dumps({"name": self.name, "roles": self.roles}, sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "config_hash", hashlib.sha256(encoded.encode("utf-8")).hexdigest())

    def role_of(self, predicate: str) -> str:
        """
        Return a predicate's role.

        Parameters
        ----------
        predicate : str
            Full predicate URI.

        Returns
        -------
        str
            One of :data:`ROLES`.

        Raises
        ------
        UnmappedPredicateError
            If the predicate has no role.
        """
        try:
            return self.roles[predicate]
        except KeyError:
            raise UnmappedPredicateError(f"{self.name}: no role for predicate {predicate!r}.") from None

    def unmapped(self, predicates: set[str]) -> list[str]:
        """Return the given predicates that have no role, sorted."""
        return sorted(p for p in predicates if p not in self.roles)


def _make_roles_v1() -> PredicateRoleMap:
    """Build the role map fixed before any #131 run."""
    roles = {expand_predicate(name): role for role, names in _ROLES_V1_NAMES.items() for name in names}
    return PredicateRoleMap(name="roles_v1", roles=roles)


ROLE_MAPS: dict[str, PredicateRoleMap] = {m.name: m for m in (_make_roles_v1(),)}


def get_role_map(name: str) -> PredicateRoleMap:
    """
    Retrieve a registered role map by name.

    Parameters
    ----------
    name : str
        Role-map name, e.g. ``"roles_v1"``.

    Returns
    -------
    PredicateRoleMap
        The requested role map.

    Raises
    ------
    KeyError
        If the name is not registered.
    """
    if name not in ROLE_MAPS:
        raise KeyError(f"Role map {name!r} not found. Available: {sorted(ROLE_MAPS)}.")
    return ROLE_MAPS[name]


@dataclass(frozen=True)
class RoleFilter:
    """
    Which roles a selector removes, under which role map.

    Attributes
    ----------
    role_map : PredicateRoleMap
        Role of every predicate.
    excluded_roles : frozenset[str]
        Roles whose items are removed; never ``core``.
    """

    role_map: PredicateRoleMap
    excluded_roles: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if CORE in self.excluded_roles:
            raise ValueError("The core role is never removed.")
        unknown = sorted(self.excluded_roles - set(ROLES))
        if unknown:
            raise ValueError(f"Unknown roles {unknown}.")

    def keeps(self, predicate: str) -> bool:
        """Whether an item with this predicate survives the filter; raises if unmapped."""
        return self.role_map.role_of(predicate) not in self.excluded_roles
