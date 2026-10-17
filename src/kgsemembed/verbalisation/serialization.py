"""Serialization orders for the text-serialization-order experiment (#134).

An order ranks an entity's items by a group key, keeping canonical order
(predicate URI, then object) within each group.  The same orders serve two
purposes: a *selection* order is the sequence in which a budget greedily keeps
items, and a *render* order is the sequence in which the kept items are
written.  The four orders were fixed in #134 before any run:

``uri``
    Canonical order, the behaviour of every condition before #134.
``core``
    Core items (``rdf:type``, ``rdfs:label``, ``rdfs:subClassOf``), then the rest.
``label``
    ``rdfs:label``, then the rest.
``role``
    ``rdfs:label``, preferred_name, synonym, definition, ``rdfs:subClassOf``,
    ``rdf:type``, disjointness, non_english, identifier, other.

:func:`order_definitions_sha256` fingerprints these definitions; results and
the freeze record carry it, so a changed definition cannot pass as the frozen one.
"""

import hashlib
import json
from typing import Protocol, Sequence

from kgsemembed.verbalisation.predicate_config import expand_predicate
from kgsemembed.verbalisation.predicate_roles import (
    CORE,
    DEFINITION,
    DISJOINTNESS,
    IDENTIFIER,
    NON_ENGLISH,
    OTHER,
    PREFERRED_NAME,
    SYNONYM,
    PredicateRoleMap,
)

RENDER_URI = "uri"
RENDER_CORE = "core"
RENDER_LABEL = "label"
RENDER_ROLE = "role"
SERIALIZATION_ORDERS = (RENDER_URI, RENDER_CORE, RENDER_LABEL, RENDER_ROLE)

RDFS_LABEL = expand_predicate("rdfs:label")
RDFS_SUBCLASS_OF = expand_predicate("rdfs:subClassOf")
RDF_TYPE = expand_predicate("rdf:type")

ROLE_PRIORITY: tuple[tuple[str, str], ...] = (
    ("predicate", RDFS_LABEL),
    ("role", PREFERRED_NAME),
    ("role", SYNONYM),
    ("role", DEFINITION),
    ("predicate", RDFS_SUBCLASS_OF),
    ("predicate", RDF_TYPE),
    ("role", DISJOINTNESS),
    ("role", NON_ENGLISH),
    ("role", IDENTIFIER),
    ("role", OTHER),
)
"""The ``role`` order: each entry names a single predicate or a whole role."""

ORDER_DEFINITIONS: dict[str, object] = {
    RENDER_URI: "canonical (predicate URI, object)",
    RENDER_CORE: [["role", CORE], ["rest"]],
    RENDER_LABEL: [["predicate", RDFS_LABEL], ["rest"]],
    RENDER_ROLE: [list(entry) for entry in ROLE_PRIORITY],
}
"""Every order as data, the input of :func:`order_definitions_sha256`."""


class HasPredicate(Protocol):
    """Anything with a full predicate URI, e.g. an ``EligibleItem``."""

    predicate: str


def order_definitions_sha256() -> str:
    """
    Fingerprint the four order definitions.

    Returns
    -------
    str
        SHA-256 hex digest of :data:`ORDER_DEFINITIONS` as canonical JSON.
    """
    encoded = json.dumps(ORDER_DEFINITIONS, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _role_priority_group(predicate: str, role_map: PredicateRoleMap) -> int:
    """Return a predicate's position in :data:`ROLE_PRIORITY`; raise if it has none."""
    role = role_map.role_of(predicate)
    for position, (kind, name) in enumerate(ROLE_PRIORITY):
        if (kind == "predicate" and name == predicate) or (kind == "role" and name == role):
            return position
    raise ValueError(f"Predicate {predicate!r} (role {role!r}) has no place in the role-priority order.")


def order_group(predicate: str, order: str, role_map: PredicateRoleMap) -> int:
    """
    Return the group key of a predicate under one order; lower comes first.

    Parameters
    ----------
    predicate : str
        Full predicate URI.
    order : str
        One of :data:`SERIALIZATION_ORDERS`.
    role_map : PredicateRoleMap
        Role of every predicate.

    Returns
    -------
    int
        Group key; ties keep canonical order.
    """
    if order == RENDER_URI:
        return 0
    if order == RENDER_CORE:
        return int(role_map.role_of(predicate) != CORE)
    if order == RENDER_LABEL:
        return int(predicate != RDFS_LABEL)
    if order == RENDER_ROLE:
        return _role_priority_group(predicate, role_map)
    raise ValueError(f"Unknown serialization order {order!r}; expected {SERIALIZATION_ORDERS}.")


def serialization_order(
    items: Sequence[HasPredicate], indices: Sequence[int], order: str, role_map: PredicateRoleMap
) -> list[int]:
    """
    Arrange item indices in one order, canonical within each group.

    Parameters
    ----------
    items : Sequence[HasPredicate]
        Items in canonical order.
    indices : Sequence[int]
        Indices into ``items`` to arrange, in any order.
    order : str
        One of :data:`SERIALIZATION_ORDERS`.
    role_map : PredicateRoleMap
        Role of every predicate.

    Returns
    -------
    list[int]
        ``indices`` sorted by ``(group, canonical index)``.
    """
    return sorted(indices, key=lambda i: (order_group(items[i].predicate, order, role_map), i))
