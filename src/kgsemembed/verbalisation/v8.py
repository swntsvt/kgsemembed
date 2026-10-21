"""Relational signature verbalisation strategy (V8).

Represents an entity by the inventory of relationship types in which it
participates rather than by its annotation content.  Class and instance
entities are described purely by their distinct outgoing and incoming
predicate types::

    Outgoing relationships: has part, part of.
    Incoming relationships: regulates.

Predicate entities additionally carry a descriptive property header with their
definition, domain, and range when those are available.  The output is
inherently bounded by the number of distinct predicate types in the graph, so
no token-budget management (PPAS) is required and the result is fully
deterministic.

Excluded predicates (#145)
--------------------------
``rdf:type``, ``rdfs:label`` and every annotation predicate are excluded in
both directions, so the signature captures structural role rather than the
presence of lexical content.  The set is fixed from predicate semantics, never
from alignments, as named families (:data:`EXCLUSION_FAMILIES`):

``type_and_label``
    ``rdf:type`` and ``rdfs:label``.
``roles_v1_annotation``
    Every predicate of an annotation role of the ``roles_v1`` role map (#131):
    preferred name, synonym, definition, identifier, non-English and other.
    These cover the FMA, SNOMED and OBO-in-OWL annotations of D1 and D2.  The
    ``disjointness`` role (``owl:disjointWith``) is a logical axiom and stays.
``owl_annotation``
    The annotation properties OWL 2 builds in (``rdfs:comment``,
    ``rdfs:seeAlso``, ``rdfs:isDefinedBy``, ``owl:versionInfo``, ...).
``owl_axiom_annotation``
    ``owl:annotatedSource``/``Property``/``Target``, which reify annotations on
    axioms; every FMA class is the ``annotatedSource`` of one.
``skos_documentation``, ``dublin_core``, ``schema_org``, ``foaf``
    Labels, names, notes, descriptions, titles, rights and depictions.
``dbkwik_page``
    DBkWik (D4) page text and links: abstract, thumbnail, wiki-link anchor
    text and external links.

Every other predicate is a relation or attribute and is kept: OWL and RDFS
structure (``subClassOf``, ``domain``, ``disjointWith``, restriction syntax),
SKOS ``broader``/``related``, ``dcterms:subject``, wiki links, infobox
properties (D4) and DBpedia and Wikidata properties (D5).
"""

import hashlib
import json
from typing import Iterable

from rdflib import Graph, URIRef
from rdflib.namespace import RDFS

from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.predicate_roles import CORE, DISJOINTNESS, get_role_map

_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
_OWL = "http://www.w3.org/2002/07/owl#"
_SKOS = "http://www.w3.org/2004/02/skos/core#"
_DC = "http://purl.org/dc/elements/1.1/"
_DCTERMS = "http://purl.org/dc/terms/"
_SCHEMA = "http://schema.org/"
_FOAF = "http://xmlns.com/foaf/0.1/"
_DBKWIK = "http://dbkwik.webdatacommons.org/ontology/"

_KEPT_ROLES = (CORE, DISJOINTNESS)


def _roles_v1_annotations() -> tuple[str, ...]:
    """Return the predicates of every annotation role of ``roles_v1``, sorted."""
    roles = get_role_map("roles_v1").roles
    return tuple(sorted(p for p, role in roles.items() if role not in _KEPT_ROLES))


def _names(namespace: str, *locals_: str) -> tuple[str, ...]:
    """Return full URIs of *locals_* in *namespace*."""
    return tuple(namespace + local for local in locals_)


EXCLUSION_FAMILIES: dict[str, tuple[str, ...]] = {
    "type_and_label": (f"{_RDF}type", f"{_RDFS}label"),
    "roles_v1_annotation": _roles_v1_annotations(),
    "owl_annotation": (
        *_names(_RDFS, "comment", "seeAlso", "isDefinedBy"),
        *_names(_OWL, "versionInfo", "deprecated", "priorVersion", "backwardCompatibleWith", "incompatibleWith"),
    ),
    "owl_axiom_annotation": _names(_OWL, "annotatedSource", "annotatedProperty", "annotatedTarget"),
    "skos_documentation": _names(
        _SKOS, "prefLabel", "altLabel", "hiddenLabel", "definition", "note", "scopeNote",
        "editorialNote", "changeNote", "historyNote", "example",
    ),
    "dublin_core": (*_names(_DC, "title", "description", "rights"), *_names(_DCTERMS, "title", "description", "rights")),
    "schema_org": _names(_SCHEMA, "name", "alternateName", "description"),
    "foaf": _names(_FOAF, "name", "givenName", "familyName", "nick", "depiction", "thumbnail"),
    "dbkwik_page": _names(_DBKWIK, "abstract", "thumbnail", "wikiPageWikiLinkText", "wikiPageExternalLink"),
}

EXCLUDED_PREDICATES = frozenset(p for family in EXCLUSION_FAMILIES.values() for p in family)


def exclusion_family(predicate: str) -> str | None:
    """
    Return the first family that excludes a predicate, or ``None`` if V8 keeps it.

    Parameters
    ----------
    predicate : str
        Full predicate URI.

    Returns
    -------
    str | None
        A key of :data:`EXCLUSION_FAMILIES`, or ``None`` for a kept relation.
    """
    return next((name for name, family in EXCLUSION_FAMILIES.items() if predicate in family), None)


def v8_exclusions_digest() -> str:
    """
    Return the SHA-256 of the V8 exclusion families.

    Recorded in every result so that a change to what V8 excludes is visible
    in the results, not only in the code.

    Returns
    -------
    str
        Hex digest of the family names and their predicates, in order.
    """
    encoded = json.dumps(EXCLUSION_FAMILIES, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

_RELATIONSHIP_ENTITY_TYPES = ("class", "instance")


class RelationalSignatureVerbaliser(VerbaliserBase):
    """Verbalise entities by their outgoing and incoming predicate types.

    Class and instance entities yield two relationship summaries; predicate
    entities are prefixed with a property description carrying their optional
    definition, domain, and range.  ``rdf:type``, ``rdfs:label`` and the
    annotation predicates of :data:`EXCLUSION_FAMILIES` never appear in either
    direction; the remaining predicate labels are lowercased, deduplicated
    by predicate type, and sorted, so the output is deterministic across
    repeated executions and graph re-parses.
    """

    def __init__(self) -> None:
        """Initialise the verbaliser; the strategy is stateless."""

    def verbalise(
        self, graph: Graph, entity_uri: URIRef, entity_type: str
    ) -> str:
        """
        Convert an entity into its relational signature.

        Parameters
        ----------
        graph : Graph
            RDF graph containing entity triples.
        entity_uri : URIRef
            URI of the entity to verbalise.
        entity_type : str
            One of ``"class"``, ``"instance"``, ``"predicate"``.

        Returns
        -------
        str
            Relationship summaries, prefixed with a property description for
            predicate entities.

        Raises
        ------
        ValueError
            If *entity_type* is not a supported entity type.
        """
        if entity_type in _RELATIONSHIP_ENTITY_TYPES:
            return "\n".join(self._relationship_lines(graph, entity_uri))
        if entity_type == "predicate":
            return self._verbalise_predicate_entity(graph, entity_uri)
        raise ValueError(
            f"Unsupported entity type: {entity_type!r}. Expected one of "
            "'class', 'instance', 'predicate'."
        )

    def _get_outgoing_predicate_labels(
        self, graph: Graph, entity_uri: URIRef
    ) -> list[str]:
        """Return sorted, lowercased labels of distinct outgoing predicates.

        Inspects triples where *entity_uri* is the subject, drops
        :data:`EXCLUDED_PREDICATES`, and resolves each remaining predicate to its
        label or local-name fallback.  Returns an empty list when no eligible
        predicate exists.
        """
        return self._predicate_labels(
            graph, graph.predicates(subject=entity_uri)
        )

    def _get_incoming_predicate_labels(
        self, graph: Graph, entity_uri: URIRef
    ) -> list[str]:
        """Return sorted, lowercased labels of distinct incoming predicates.

        Applies the same exclusion, resolution, lowercasing, deduplication,
        and sorting rules as :meth:`_get_outgoing_predicate_labels`, inspecting
        triples where *entity_uri* is the object.
        """
        return self._predicate_labels(
            graph, graph.predicates(object=entity_uri)
        )

    def _predicate_labels(
        self, graph: Graph, predicates: Iterable[URIRef]
    ) -> list[str]:
        """Resolve *predicates* to sorted, deduplicated, lowercased labels.

        Predicate URIs are deduplicated before resolution so each distinct
        predicate type is labelled once regardless of occurrence frequency.
        """
        distinct = {
            pred for pred in predicates if str(pred) not in EXCLUDED_PREDICATES
        }
        labels = {
            self.get_label_or_local(graph, pred).lower() for pred in distinct
        }
        return sorted(labels)

    def _verbalise_predicate_entity(
        self, graph: Graph, entity_uri: URIRef
    ) -> str:
        """Return the property description followed by relationship summaries.

        The definition, domain, and range lines are emitted only when the
        corresponding information exists in the graph.
        """
        parts = [f"Property: {self.get_label_or_local(graph, entity_uri)}."]
        definition = self._get_definition(graph, entity_uri)
        if definition:
            parts.append(f"Definition: {definition}.")
        domain = self._resolve_schema_labels(graph, entity_uri, RDFS.domain)
        if domain:
            parts.append(f"Domain: {domain}.")
        object_range = self._resolve_schema_labels(graph, entity_uri, RDFS.range)
        if object_range:
            parts.append(f"Range: {object_range}.")
        parts.extend(self._relationship_lines(graph, entity_uri))
        return "\n".join(parts)

    def _relationship_lines(
        self, graph: Graph, entity_uri: URIRef
    ) -> list[str]:
        """Return the outgoing and incoming relationship summary lines."""
        outgoing = self._get_outgoing_predicate_labels(graph, entity_uri)
        incoming = self._get_incoming_predicate_labels(graph, entity_uri)
        return [
            self._format_relationships("Outgoing", outgoing),
            self._format_relationships("Incoming", incoming),
        ]

    def _get_definition(self, graph: Graph, entity_uri: URIRef) -> str | None:
        """Return the first English/untagged ``rdfs:comment`` literal, else None."""
        comments = self.get_all_literals(graph, entity_uri, [RDFS.comment])
        return comments[0].strip() if comments else None

    def _resolve_schema_labels(
        self, graph: Graph, entity_uri: URIRef, predicate: URIRef
    ) -> str:
        """Return sorted, comma-joined labels of named *predicate* objects."""
        labels = {
            self.get_label_or_local(graph, obj)
            for obj in graph.objects(entity_uri, predicate)
            if isinstance(obj, URIRef)
        }
        return ", ".join(sorted(labels))

    @staticmethod
    def _format_relationships(direction: str, labels: list[str]) -> str:
        """Format a relationship summary line, using ``none`` when empty."""
        body = ", ".join(labels) if labels else "none"
        return f"{direction} relationships: {body}."
