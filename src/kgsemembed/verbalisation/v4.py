"""Structured key-value verbalisation strategy for instance entities.

Serialises an entity as deterministic pipe-separated predicate-value pairs.
Integrates with PPAS for entities exceeding the triple count threshold.
"""

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDFS

from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.ppas import (
    INSTANCE_TIER_LIST,
    PPAS_BUDGETS,
    ppas_sample,
    should_apply_ppas,
)

_EXCLUDED_PREDICATES = [
    "http://www.w3.org/2002/07/owl#sameAs",
]


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

    # ------------------------------------------------------------------
    # Triple collection & filtering
    # ------------------------------------------------------------------

    def _collect_triples(
        self, graph: Graph, entity_uri: URIRef
    ) -> list[tuple]:
        """Return filtered triples for *entity_uri*."""
        return [
            t
            for t in graph.triples((entity_uri, None, None))
            if not self._should_exclude(*t)
        ]

    @staticmethod
    def _should_exclude(
        subj: URIRef, pred: URIRef, obj: URIRef | BNode | Literal
    ) -> bool:
        """Return True if the triple should be excluded from verbalisation."""
        if str(pred) in _EXCLUDED_PREDICATES:
            return True
        if isinstance(obj, BNode):
            return True
        if subj == obj:
            return True
        return False

    # ------------------------------------------------------------------
    # Predicate / object resolution
    # ------------------------------------------------------------------

    def _resolve_predicate_from_graph(
        self, graph: Graph, pred: URIRef
    ) -> str:
        """Resolve predicate label from the given graph.

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
        """Fast triple verbaliser used by PPAS for token-cost estimation.

        Returns ``"pred_local: obj_str"`` using the predicate local name
        and the object's string representation.
        """
        pred_label = pred.split("#")[-1].split("/")[-1]
        return f"{pred_label}: {obj}"

    # ------------------------------------------------------------------
    # Pair formatting
    # ------------------------------------------------------------------

    def _format_pair(
        self,
        graph: Graph,
        pred: URIRef,
        obj: URIRef | BNode | Literal,
    ) -> str:
        """Format a single triple as ``"key: value"``."""
        is_type = (
            str(pred) == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        )
        key = "type" if is_type else self._resolve_predicate_from_graph(graph, pred)

        if isinstance(obj, Literal):
            value = str(obj)
        elif isinstance(obj, URIRef):
            value = self.get_label_or_local(graph, obj)
        else:
            return ""

        return f"{key}: {value}"
