"""Annotation-based verbalisation strategy for KG entity annotation.

Concatenates label, definition, and synonyms into a structured
sentence-like string with optional fields.
"""

from rdflib import Graph, Literal, URIRef

from kgsemembed.verbalisation.base import (
    DEFINITION_PREDICATES,
    VerbaliserBase,
)

_PRIMARY_LABEL_PREDICATES = [
    URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
    URIRef("http://www.w3.org/2004/02/skos/core#prefLabel"),
]

_ALT_LABEL_PREDICATES = [
    URIRef("http://www.w3.org/2004/02/skos/core#altLabel"),
    URIRef("http://purl.obolibrary.org/obo/hasExactSynonym"),
    URIRef("http://purl.obolibrary.org/obo/hasSynonym"),
]


class AnnotationVerbaliser(VerbaliserBase):
    """Convert entity metadata into a structured annotation string.

    Fields appear in fixed order (Label, Definition, Synonyms) and
    empty fields are omitted.  If no label exists, the URI local name
    is used as a fallback.

    Example
    -------
    >>> entity with label "Myocardial infarction",
    >>> definition "Necrosis of the myocardium",
    >>> synonyms ["heart attack", "MI"]
    >>> "Label: Myocardial infarction. Definition: Necrosis of the myocardium. Synonyms: heart attack; MI."
    """

    def verbalise(self, graph: Graph, entity_uri: URIRef, entity_type: str) -> str:
        """
        Convert entity metadata into a structured annotation string.

        Parameters
        ----------
        graph : Graph
            RDF graph containing entity triples.
        entity_uri : URIRef
            URI of the entity to verbalise.
        entity_type : str
            One of ``"class"``, ``"instance"``, ``"predicate"`` (ignored).

        Returns
        -------
        str
            Non-empty string with label, and optionally definition and synonyms.
        """
        label = self._get_primary_label(graph, entity_uri)

        if label is None:
            label = self.get_label_or_local(graph, entity_uri)

        defs = self.get_all_literals(graph, entity_uri, DEFINITION_PREDICATES)
        definition = defs[0] if defs else None

        synonyms = self._get_alt_labels(graph, entity_uri)

        parts = [f"Label: {label.strip()}"]

        if definition:
            parts.append(f"Definition: {definition.strip()}")

        if synonyms:
            parts.append(f"Synonyms: {'; '.join(s.strip() for s in synonyms)}")

        return ". ".join(parts) + "."

    def _get_primary_label(self, graph: Graph, entity_uri: URIRef) -> str | None:
        """Return the first English or untagged label from primary predicates."""
        for pred in _PRIMARY_LABEL_PREDICATES:
            for obj in graph.objects(entity_uri, pred):
                if isinstance(obj, Literal):
                    lang = str(obj.language) if obj.language else ""
                    if lang == "en":
                        return str(obj)
        for pred in _PRIMARY_LABEL_PREDICATES:
            for obj in graph.objects(entity_uri, pred):
                if isinstance(obj, Literal) and obj.language is None:
                    return str(obj)
        return None

    def _get_alt_labels(self, graph: Graph, entity_uri: URIRef) -> list[str]:
        """Return deduplicated English/untagged alt labels."""
        seen: set[str] = set()
        results: list[str] = []
        for pred in _ALT_LABEL_PREDICATES:
            for obj in graph.objects(entity_uri, pred):
                if isinstance(obj, Literal):
                    lang = str(obj.language) if obj.language else ""
                    if lang == "en" or obj.language is None:
                        val = str(obj)
                        if val not in seen:
                            seen.add(val)
                            results.append(val)
        return results
