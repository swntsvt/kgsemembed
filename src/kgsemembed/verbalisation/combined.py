"""Composite verbalisation strategy.

Runs several verbalisation strategies in sequence and concatenates their
non-empty outputs into a single description.  Composition is purely
structural: each component strategy is invoked exactly as it was configured
and its result is combined without altering the component's own behaviour.
"""

from rdflib import Graph, URIRef

from kgsemembed.verbalisation.base import VerbaliserBase

_MIN_STRATEGIES = 2
_SEPARATOR = "\n"


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
        parts = [
            stripped
            for strategy in self.strategies
            if (stripped := strategy.verbalise(graph, entity_uri, entity_type).strip())
        ]
        return _SEPARATOR.join(parts)
