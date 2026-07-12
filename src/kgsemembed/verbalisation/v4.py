"""Structured key-value verbalisation strategy for instance entities.

Serialises an entity as deterministic pipe-separated predicate-value pairs.
Integrates with PPAS for entities exceeding the triple count threshold.
"""

from rdflib import Graph, URIRef

from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.ppas import (
    INSTANCE_TIER_LIST,
    PPAS_BUDGETS,
    ppas_sample,
    should_apply_ppas,
)


class StructuredKVVerbaliser(VerbaliserBase):
    """Serialise instance entities as structured key-value pairs.

    Output format::

        predicate_label: value | predicate_label: value

    RDF type declarations use the key ``"type"``.  Blank-node objects,
    ``owl:sameAs`` triples, and reflexive triples are silently excluded.

    Example
    -------
    >>> entity with rdfs:label "Aspirin", type "Drug", CAS "50-78-2"
    >>> "type: Drug | CAS number: 50-78-2"
    """

    def __init__(self, model_key: str = "M2") -> None:
        """
        Parameters
        ----------
        model_key : str
            Model key used to determine whether PPAS applies (e.g. "M1", "M2").
        """
        self.model_key = model_key

    def verbalise(
        self, graph: Graph, entity_uri: URIRef, entity_type: str
    ) -> str:
        """
        Convert entity triples into pipe-separated key-value pairs.

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
            Non-empty key-value string for entities with retrievable data;
            empty string only for entities with no retrievable information.
        """
        triples = self._collect_triples(graph, entity_uri)

        if should_apply_ppas(triples, self.model_key):
            triples = ppas_sample(
                triples,
                INSTANCE_TIER_LIST,
                PPAS_BUDGETS[self.model_key],
                self._verbalise_triple,
            )

        pairs: list[str] = []
        for subj, pred, obj in triples:
            pair = self._format_pair(graph, pred, obj)
            if pair:
                pairs.append(pair)

        return " | ".join(pairs)
