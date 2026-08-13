"""Abstract base class for all verbalisation strategies."""

from abc import ABC, abstractmethod
import re

from rdflib import BNode, Graph, URIRef, Literal
from rdflib.namespace import RDFS, SKOS
from typing import Optional, List

# Standard predicate constants used by all strategies
LABEL_PREDICATES = [
    RDFS.label,
    SKOS.prefLabel,
    URIRef("http://www.w3.org/2004/02/skos/core#altLabel"),
]

DEFINITION_PREDICATES = [
    RDFS.comment,
    SKOS.definition,
    URIRef("http://purl.obolibrary.org/obo/IAO_0000115"),
]

SYNONYM_PREDICATES = [
    URIRef("http://www.w3.org/2004/02/skos/core#altLabel"),
    URIRef("http://purl.obolibrary.org/obo/hasExactSynonym"),
    URIRef("http://purl.obolibrary.org/obo/hasSynonym"),
    URIRef("http://purl.obolibrary.org/obo/hasRelatedSynonym"),
]

# Predicates and objects excluded from structured key-value serialisation
_EXCLUDED_PREDICATES = [
    "http://www.w3.org/2002/07/owl#sameAs",
]

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


class VerbaliserBase(ABC):
    """Base class for all KG entity verbalisation strategies.

    Subclasses implement ``verbalise`` to define how entity metadata is
    converted into human-readable text for embedding.
    """

    @abstractmethod
    def verbalise(self, graph: Graph, entity_uri: URIRef, entity_type: str) -> str:
        """Convert entity metadata into a textual description.

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
            Non-empty string for entities with at least one label.
            Empty string only for entities with no retrievable info.
        """

    def get_label(self, graph: Graph, entity_uri: URIRef) -> Optional[str]:
        """Return the first English or untagged label from LABEL_PREDICATES.

        Parameters
        ----------
        graph : Graph
            RDF graph containing entity triples.
        entity_uri : URIRef
            URI of the entity.

        Returns
        -------
        Optional[str]
            Label string or None if nothing found.
        """
        for pred in LABEL_PREDICATES:
            for obj in graph.objects(entity_uri, pred):
                if isinstance(obj, Literal):
                    lang = str(obj.language) if obj.language else ""
                    if lang == "en":
                        return str(obj)
        # Second pass: untagged labels
        for pred in LABEL_PREDICATES:
            for obj in graph.objects(entity_uri, pred):
                if isinstance(obj, Literal) and obj.language is None:
                    return str(obj)
        return None

    def get_local_name(self, uri: URIRef) -> str:
        """Extract a human-readable local name from a URI.

        Splits on ``#`` first, then the last ``/``. Converts camelCase and
        snake_case into space-separated words.

        Parameters
        ----------
        uri : URIRef
            The URI to parse.

        Returns
        -------
        str
            Extracted and formatted local name.
        """
        local = str(uri)
        if "#" in local:
            local = local.split("#")[-1]
        else:
            last_slash = local.rfind("/")
            if last_slash != -1:
                local = local[last_slash + 1:]

        # Convert snake_case to spaces
        local = local.replace("_", " ")
        # Insert space before uppercase letters (camelCase)
        local = re.sub(r"([a-z])([A-Z])", r"\1 \2", local)
        return local

    def get_label_or_local(self, graph: Graph, uri: URIRef) -> str:
        """Return get_label() if found, otherwise get_local_name().

        Parameters
        ----------
        graph : Graph
            RDF graph containing entity triples.
        uri : URIRef
            URI of the entity.

        Returns
        -------
        str
            Label string or local name fallback.
        """
        label = self.get_label(graph, uri)
        if label:
            return label
        return self.get_local_name(uri)

    def get_all_literals(self, graph: Graph, entity_uri: URIRef, predicates: List[URIRef]) -> List[str]:
        """Return all English or untagged literal values for given predicates.

        Parameters
        ----------
        graph : Graph
            RDF graph containing entity triples.
        entity_uri : URIRef
            URI of the entity.
        predicates : List[URIRef]
            List of predicate URIs to query.

        Returns
        -------
        List[str]
            Deduplicated literal strings preserving first-occurrence order.
        """
        seen = set()
        results = []
        for pred in predicates:
            for obj in graph.objects(entity_uri, pred):
                if isinstance(obj, Literal):
                    lang = str(obj.language) if obj.language else ""
                    if lang == "en" or obj.language is None:
                        val = str(obj)
                        if val not in seen:
                            seen.add(val)
                            results.append(val)
        return results

    # ------------------------------------------------------------------
    # Structured key-value serialisation helpers (shared by V4 and V6)
    # ------------------------------------------------------------------

    def _collect_triples(
        self, graph: Graph, entity_uri: URIRef
    ) -> list[tuple]:
        """Return structured-KV triples for *entity_uri*, filtering noise.

        Parameters
        ----------
        graph : Graph
            RDF graph containing entity triples.
        entity_uri : URIRef
            URI of the entity whose triples are collected.

        Returns
        -------
        list of tuple
            Triples excluding ``owl:sameAs``, blank-node objects, and
            reflexive statements, sorted by ``(subject, predicate, object)``
            so that budget-based filtering receives a stable ordering.
        """
        triples = [
            t
            for t in graph.triples((entity_uri, None, None))
            if not self._should_exclude(*t)
        ]
        return sorted(triples, key=lambda t: (str(t[0]), str(t[1]), str(t[2])))

    @staticmethod
    def _should_exclude(
        subj: URIRef, pred: URIRef, obj: URIRef | BNode | Literal
    ) -> bool:
        """Return True if the triple should be excluded from serialisation."""
        if str(pred) in _EXCLUDED_PREDICATES:
            return True
        if isinstance(obj, BNode):
            return True
        if subj == obj:
            return True
        return False

    def _resolve_predicate_from_graph(
        self, graph: Graph, pred: URIRef
    ) -> str:
        """Resolve a predicate's label from *graph*, else its local name.

        Parameters
        ----------
        graph : Graph
            RDF graph to query for predicate labels.
        pred : URIRef
            Predicate URI to resolve.

        Returns
        -------
        str
            Human-readable predicate label or local-name fallback.
        """
        for obj in graph.objects(pred, RDFS.label):
            if isinstance(obj, Literal) and obj.language in ("en", None):
                return str(obj)
        return self.get_local_name(pred)

    @staticmethod
    def _verbalise_triple(
        _subj: URIRef, pred: URIRef, obj: URIRef | BNode | Literal
    ) -> str:
        """Return ``"pred_local: obj"`` for fast token-cost estimation."""
        pred_label = pred.split("#")[-1].split("/")[-1]
        return f"{pred_label}: {obj}"

    def _format_pair(
        self,
        graph: Graph,
        pred: URIRef,
        obj: URIRef | BNode | Literal,
    ) -> str:
        """Format a single triple as ``"key: value"``.

        Parameters
        ----------
        graph : Graph
            RDF graph used to resolve predicate and object labels.
        pred : URIRef
            Predicate URI of the triple.
        obj : URIRef | BNode | Literal
            Object of the triple.

        Returns
        -------
        str
            ``"key: value"`` string, or empty string for non-serialisable
            objects.
        """
        is_type = str(pred) == _RDF_TYPE
        key = "type" if is_type else self._resolve_predicate_from_graph(graph, pred)

        if isinstance(obj, Literal):
            value = str(obj)
        elif isinstance(obj, URIRef):
            value = self.get_label_or_local(graph, obj)
        else:
            return ""

        return f"{key}: {value}"
