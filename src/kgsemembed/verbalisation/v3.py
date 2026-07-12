"""Template-based natural language verbalisation strategy (V3).

Converts schema-relevant RDF triples into deterministic English sentences using
predicate-specific templates.  Triple selection reuses the PPAS predicate-tier
mechanism (:mod:`kgsemembed.verbalisation.ppas`) so the same schema-aware
prioritisation and token-budget behaviour that governs V6 also decides which
triples are rendered as sentences.

Predicates carrying a dedicated template (e.g. ``rdfs:subClassOf``) are rendered
with that template.  Remaining literal-valued triples use the datatype-property
template and remaining URI-valued triples use the generic fallback template.
Blank-node objects and non-English literals produce no output.
"""

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.ppas import (
    CLASS_TIER_LIST,
    INSTANCE_TIER_LIST,
    PPAS_BUDGETS,
    PREDICATE_TIER_LIST,
    estimate_tokens,
    ppas_sample,
)

_TIER_LISTS: dict[str, list[list[str]]] = {
    "class": CLASS_TIER_LIST,
    "instance": INSTANCE_TIER_LIST,
    "predicate": PREDICATE_TIER_LIST,
}

_SENTENCE_TEMPLATES: dict[str, str] = {
    str(RDFS.subClassOf): "{entity} is a subclass of {object}.",
    str(RDF.type): "{entity} is a {object}.",
    str(OWL.equivalentClass): "{entity} is equivalent to {object}.",
    str(OWL.disjointWith): "{entity} is disjoint from {object}.",
    str(RDFS.domain): "The domain of {entity} is {object}.",
    str(RDFS.range): "The range of {entity} is {object}.",
    str(RDFS.label): "{entity} is called {object}.",
    str(RDFS.comment): "{entity}: {object}.",
    str(OWL.inverseOf): "{entity} is the inverse of {object}.",
    str(RDFS.subPropertyOf): "{entity} is a sub-property of {object}.",
}

_DATATYPE_TEMPLATE = "{entity} has {predicate} {object}."
_FALLBACK_TEMPLATE = "{entity} {predicate} {object}."


class TemplateNLVerbaliser(VerbaliserBase):
    """Verbalise entities as templated natural language sentences.

    Output format::

        <Entity> is a subclass of <Parent>. <Entity> is called <Label>.

    Triples are selected in schema-aware predicate-tier order and capped to the
    model token budget (models without a budget, e.g. ``"M3"``, retain every
    tier-ordered triple), then each surviving triple is rendered with its
    predicate-specific template.
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
        Convert entity triples into templated English sentences.

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
            Space-separated English sentences, or an empty string for entities
            with no retrievable information.

        Raises
        ------
        ValueError
            If *entity_type* is not a supported entity type.
        """
        tier_list = self._tier_list_for(entity_type)
        triples = self._collect_triples(graph, entity_uri)
        selected = self._select_triples(triples, tier_list)

        entity = self.get_label_or_local(graph, entity_uri)
        sentences = [
            sentence
            for _subj, pred, obj in selected
            if (sentence := self._render(graph, entity, pred, obj))
        ]
        return " ".join(sentences)

    @staticmethod
    def _tier_list_for(entity_type: str) -> list[list[str]]:
        """Return the predicate-tier list for *entity_type*."""
        if entity_type not in _TIER_LISTS:
            raise ValueError(
                f"Unsupported entity type: {entity_type!r}. Expected one of "
                "'class', 'instance', 'predicate'."
            )
        return _TIER_LISTS[entity_type]

    def _select_triples(
        self, triples: list[tuple], tier_list: list[list[str]]
    ) -> list[tuple]:
        """Order triples by predicate tier and cap them to the model budget."""
        if not triples:
            return []
        augmented = tier_list + [self._untiered_predicates(triples, tier_list)]
        budget = PPAS_BUDGETS.get(self.model_key)
        if budget is None:
            budget = self._total_cost(triples)
        return ppas_sample(triples, augmented, budget, self._verbalise_triple)

    @staticmethod
    def _untiered_predicates(
        triples: list[tuple], tier_list: list[list[str]]
    ) -> list[str]:
        """Return predicates present in *triples* but absent from any tier."""
        tiered = {predicate for tier in tier_list for predicate in tier}
        untiered: list[str] = []
        for _subj, pred, _obj in triples:
            key = str(pred)
            if key not in tiered and key not in untiered:
                untiered.append(key)
        return untiered

    def _total_cost(self, triples: list[tuple]) -> int:
        """Return the summed token cost of *triples* for the uncapped path."""
        return sum(
            estimate_tokens(self._verbalise_triple(*triple))
            for triple in triples
        )

    def _render(
        self,
        graph: Graph,
        entity: str,
        pred: URIRef,
        obj: URIRef | Literal,
    ) -> str | None:
        """Render a single triple as a sentence, or None if it is skipped."""
        if isinstance(obj, Literal):
            if obj.language not in (None, "en"):
                return None
            value = str(obj)
        else:
            value = self.get_label_or_local(graph, obj)

        template = _SENTENCE_TEMPLATES.get(str(pred))
        if template is not None:
            return template.format(entity=entity, object=value)

        predicate = self._resolve_predicate_from_graph(graph, pred)
        fallback = _DATATYPE_TEMPLATE if isinstance(obj, Literal) else _FALLBACK_TEMPLATE
        return fallback.format(entity=entity, predicate=predicate, object=value)
