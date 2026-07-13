"""Predicate-aware random walk verbalisation strategy (V5).

Represents an entity as a corpus of deterministic random walks through the
knowledge graph.  Each walk alternates between entity labels and predicate
labels, capturing local neighbourhood structure in a form suitable for
embedding models::

    Heart [part of] Cardiovascular system [contains] Blood vessel

Walks traverse only ``URIRef`` neighbours reached through eligible predicates;
literal-valued and blank-node edges are never followed.  A fixed random seed is
applied once per call so repeated verbalisation of the same entity is fully
reproducible.  Concatenated walks are capped to the model token budget (models
without a budget, e.g. ``"M3"``, retain every walk).
"""

import random

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF

from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.ppas import PPAS_BUDGETS, estimate_tokens

WALK_DEPTH = 4
WALKS_PER_ENTITY = 10
RANDOM_SEED = 42

_ALWAYS_SKIPPED = {str(OWL.sameAs), str(OWL.disjointWith)}
_WALK_SEPARATOR = " | "


class NeighbourhoodWalkVerbaliser(VerbaliserBase):
    """Verbalise entities as predicate-aware random walks.

    Output format::

        Entity [predicate] Neighbour ... | Entity [predicate] Neighbour ...

    ``WALKS_PER_ENTITY`` independent walks of at most ``WALK_DEPTH`` hops are
    generated per entity, rendered with square-bracketed predicate labels, then
    joined with `` | `` and capped to the model token budget.
    """

    def __init__(self, model_key: str = "M2") -> None:
        """
        Parameters
        ----------
        model_key : str
            Model key used to look up the token budget (e.g. "M1", "M2").
        """
        self.model_key = model_key

    def verbalise(
        self, graph: Graph, entity_uri: URIRef, entity_type: str
    ) -> str:
        """
        Convert an entity's neighbourhood into predicate-aware random walks.

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
            Budget-capped walks joined with `` | ``, or the entity's readable
            label when it has no eligible outgoing relationships.
        """
        random.seed(RANDOM_SEED)
        skipped = self._skipped_predicates(entity_type)
        if not self._get_outgoing_triples(graph, entity_uri, skipped):
            return self.get_label_or_local(graph, entity_uri)

        walks = [
            self._single_walk(graph, entity_uri, skipped)
            for _ in range(WALKS_PER_ENTITY)
        ]
        texts = [self._render_walk(graph, entity_uri, walk) for walk in walks]
        return self._apply_budget(texts)

    @staticmethod
    def _skipped_predicates(entity_type: str) -> set[str]:
        """Return the predicate URIs excluded from traversal for *entity_type*."""
        skipped = set(_ALWAYS_SKIPPED)
        if entity_type == "class":
            skipped.add(str(RDF.type))
        return skipped

    def _get_outgoing_triples(
        self, graph: Graph, node: URIRef, skipped: set[str]
    ) -> list[tuple[URIRef, URIRef]]:
        """Return eligible ``(predicate, object)`` steps leaving *node*.

        Only ``URIRef`` objects reached through non-skipped predicates are
        eligible; literal and blank-node objects are excluded.  Results are
        sorted so walk selection is reproducible across runs.
        """
        steps = [
            (pred, obj)
            for pred, obj in graph.predicate_objects(node)
            if str(pred) not in skipped and isinstance(obj, URIRef)
        ]
        return sorted(steps, key=lambda step: (str(step[0]), str(step[1])))

    def _single_walk(
        self, graph: Graph, start: URIRef, skipped: set[str]
    ) -> list[tuple[URIRef, URIRef]]:
        """Perform one random walk of at most ``WALK_DEPTH`` hops from *start*.

        At each step one eligible outgoing triple is chosen uniformly at random
        and its ``URIRef`` object becomes the next node.  The walk terminates
        early when the current node has no eligible outgoing triples.
        """
        steps: list[tuple[URIRef, URIRef]] = []
        current = start
        for _ in range(WALK_DEPTH):
            eligible = self._get_outgoing_triples(graph, current, skipped)
            if not eligible:
                break
            pred, obj = random.choice(eligible)
            steps.append((pred, obj))
            current = obj
        return steps

    def _render_walk(
        self, graph: Graph, start: URIRef, steps: list[tuple[URIRef, URIRef]]
    ) -> str:
        """Render a walk as ``"Entity [predicate] Node [predicate] Node"``."""
        parts = [self.get_label_or_local(graph, start)]
        for pred, obj in steps:
            parts.append(f"[{self.get_label_or_local(graph, pred)}]")
            parts.append(self.get_label_or_local(graph, obj))
        return " ".join(parts)

    def _apply_budget(self, walks: list[str]) -> str:
        """Concatenate *walks*, stopping before the model token budget is exceeded.

        Models without a configured budget (e.g. ``"M3"``) retain every walk.
        """
        budget = PPAS_BUDGETS.get(self.model_key)
        if budget is None:
            return _WALK_SEPARATOR.join(walks)

        selected: list[str] = []
        for walk in walks:
            candidate = _WALK_SEPARATOR.join(selected + [walk])
            if estimate_tokens(candidate) > budget:
                break
            selected.append(walk)
        return _WALK_SEPARATOR.join(selected)
