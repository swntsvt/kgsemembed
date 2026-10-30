"""Structured key-value verbalisation strategy for instance entities.

Serialises an entity as deterministic pipe-separated predicate-value pairs.
Pairs keep graph order unless the entity exceeds the PPAS triple-count
threshold or its text does not fit the model token budget; then they are
taken in PPAS tier order, untiered predicates last, and kept while they fit.
The budget holds for every entity, counted with the model's tokenizer (#183).
"""

from functools import cached_property
from typing import Optional

from rdflib import Graph, URIRef

from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.budget import TokenBudget, fit_items, token_budget
from kgsemembed.verbalisation.ppas import (
    INSTANCE_TIER_LIST,
    should_apply_ppas,
    tier_order,
)

_PAIR_SEPARATOR = " | "


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

    def __init__(self, model_key: str = "M2", role: str = "candidate") -> None:
        """
        Parameters
        ----------
        model_key : str
            Model key used to determine whether PPAS applies (e.g. "M1", "M2").
        role : str
            ``"source"`` or ``"candidate"``: the encoder role whose input the
            budget counts.
        """
        self.model_key = model_key
        self.role = role

    @cached_property
    def budget(self) -> Optional[TokenBudget]:
        """The model's token budget for this role, or ``None`` when uncapped."""
        return token_budget(self.model_key, self.role)

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
        return self.verbalise_after(graph, entity_uri, entity_type, "")

    def verbalise_after(
        self, graph: Graph, entity_uri: URIRef, entity_type: str, preceding: str
    ) -> str:
        """
        Render pairs that fit what ``preceding`` left of the budget.

        Parameters
        ----------
        graph : Graph
            RDF graph containing entity triples.
        entity_uri : URIRef
            URI of the entity to verbalise.
        entity_type : str
            One of ``"class"``, ``"instance"``, ``"predicate"`` (ignored).
        preceding : str
            Joined text of the earlier components of a composite, or ``""``.

        Returns
        -------
        str
            The kept pairs joined with `` | ``.
        """
        triples = self._collect_triples(graph, entity_uri)
        pairs = self._pairs(graph, triples)
        if self.budget is None:
            return _PAIR_SEPARATOR.join(pairs)
        if should_apply_ppas(triples, self.model_key) or not self.budget.fits(_PAIR_SEPARATOR.join(pairs), preceding):
            pairs = self._pairs(graph, tier_order(triples, INSTANCE_TIER_LIST))
        return _PAIR_SEPARATOR.join(fit_items(pairs, _PAIR_SEPARATOR, self.budget, preceding))

    def _pairs(self, graph: Graph, triples: list[tuple]) -> list[str]:
        """Render each triple as a ``key: value`` pair, dropping empty ones."""
        return [pair for _subj, pred, obj in triples if (pair := self._format_pair(graph, pred, obj))]
