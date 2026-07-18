"""Relational signature verbalisation strategy (V8).

Represents an entity by the inventory of relationship types in which it
participates rather than by its annotation content.  Class and instance
entities are described purely by their distinct outgoing and incoming
predicate types::

    Outgoing relationships: has part, part of.
    Incoming relationships: regulates.

Predicate entities additionally carry a descriptive property header with their
definition, domain, and range when those are available.  Annotation predicates
(labels, comments, definitions) are always excluded so the signature captures
structural role rather than lexical content.  The output is inherently bounded
by the number of distinct predicate types in the graph, so no token-budget
management (PPAS) is required and the result is fully deterministic.
"""

from typing import Iterable

from rdflib import Graph, URIRef
from rdflib.namespace import RDFS

from kgsemembed.verbalisation.base import VerbaliserBase

EXCLUDED_PREDICATES = [
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "http://www.w3.org/2004/02/skos/core#altLabel",
    "http://www.w3.org/2004/02/skos/core#definition",
    "http://purl.obolibrary.org/obo/IAO_0000115",
]

_RELATIONSHIP_ENTITY_TYPES = ("class", "instance")


class RelationalSignatureVerbaliser(VerbaliserBase):
    """Verbalise entities by their outgoing and incoming predicate types.

    Class and instance entities yield two relationship summaries; predicate
    entities are prefixed with a property description carrying their optional
    definition, domain, and range.  Annotation predicates never appear, labels
    are lowercased, deduplicated by predicate type, and sorted, so the output
    is deterministic across repeated executions and graph re-parses.
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

        Inspects triples where *entity_uri* is the subject, drops excluded
        annotation predicates, and resolves each remaining predicate to its
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
