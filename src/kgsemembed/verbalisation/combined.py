"""Composite verbalisation strategy.

Runs several verbalisation strategies in sequence and concatenates their
non-empty outputs into a single description.  Each component is given the
text of the components before it, so a budgeted component fits what they left
of the budget (#183: components are filled in their registered order).
"""

from rdflib import Graph, URIRef

from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.budget import join_parts

_MIN_STRATEGIES = 2


class CombinedVerbaliser(VerbaliserBase):
    """Combine several verbalisation strategies into one description.

    Component strategies are executed in the supplied order and their
    stripped, non-empty outputs are joined with a single newline.  Empty and
    whitespace-only component outputs are dropped so no blank lines or leading
    and trailing newlines appear in the result.

    Example
    -------
    >>> Annotation text.
    >>> Outgoing relationships: part of, located in.
    """

    def __init__(self, strategies: list[VerbaliserBase]) -> None:
        """
        Parameters
        ----------
        strategies : list[VerbaliserBase]
            Component strategies executed in order. At least two are required.

        Raises
        ------
        ValueError
            If fewer than two strategies are supplied.
        """
        if len(strategies) < _MIN_STRATEGIES:
            raise ValueError(
                f"CombinedVerbaliser requires at least {_MIN_STRATEGIES} "
                f"strategies, got {len(strategies)}."
            )
        self.strategies = strategies

    def verbalise(
        self, graph: Graph, entity_uri: URIRef, entity_type: str
    ) -> str:
        """
        Combine component verbalisations into a single description.

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
            Non-empty component outputs joined with a single newline, or an
            empty string when every component produces no output.
        """
        parts: list[str] = []
        for strategy in self.strategies:
            preceding = join_parts(parts)
            parts.append(strategy.verbalise_after(graph, entity_uri, entity_type, preceding))
        return join_parts(parts)
