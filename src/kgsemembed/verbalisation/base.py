"""Abstract base class for all verbalisation strategies."""

from abc import ABC, abstractmethod
import re

from rdflib import Graph, URIRef, Literal
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
