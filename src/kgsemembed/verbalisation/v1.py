"""Label-only verbalisation strategy for KG entity annotation.

Produces a flat semicolon-separated string of the primary label and all
synonyms for a given entity.
"""

from rdflib import Graph, Literal, URIRef

from kgsemembed.verbalisation.base import VerbaliserBase

_PRIMARY_LABEL_PREDICATES = [
    URIRef("http://www.w3.org/2000/01/rdf-schema#label"),
    URIRef("http://www.w3.org/2004/02/skos/core#prefLabel"),
]

_ALT_LABEL_PREDICATES = [
    URIRef("http://www.w3.org/2004/02/skos/core#altLabel"),
    URIRef("http://purl.obolibrary.org/obo/hasExactSynonym"),
    URIRef("http://purl.obolibrary.org/obo/hasSynonym"),
]


class LabelVerbaliser(VerbaliserBase):
    """Convert entity metadata into a label-only textual description.

    The output is the primary label followed by any synonyms,
    joined with ``" ; "``.  If no label exists, the URI local name
    is used as a fallback.

    Example
    -------
    >>> entity with rdfs:label "Heart" and obo:hasSynonym "Cardiac muscle"
    >>> "Heart ; Cardiac muscle"
    """

    def verbalise(self, graph: Graph, entity_uri: URIRef, entity_type: str) -> str:
        """
        Convert entity metadata into a label-only string.

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
            Non-empty string with label and optional synonyms.
        """
        primary = self._get_primary_label(graph, entity_uri)

        if primary is None:
            primary = self.get_label_or_local(graph, entity_uri)

        alts = self._get_alt_labels(graph, entity_uri)
        alt_str = " ; ".join(alts)

        if alt_str:
            result = f"{primary} ; {alt_str}"
        else:
            result = primary

        return result.strip()

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
