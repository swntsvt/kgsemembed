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

from functools import cached_property
from typing import Optional

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS

from kgsemembed.verbalisation.base import VerbaliserBase, literal_text
from kgsemembed.verbalisation.budget import TokenBudget, fit_items, token_budget
from kgsemembed.verbalisation.ppas import (
    CLASS_TIER_LIST,
    INSTANCE_TIER_LIST,
    PREDICATE_TIER_LIST,
    tier_order,
)

_SENTENCE_SEPARATOR = " "

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

    Triples are rendered in schema-aware predicate-tier order (untiered
    predicates last) and the sentences are kept in that order while they fit
    the model token budget, counted with the model's tokenizer (#183); models
    without a budget, e.g. ``"M3"``, retain every sentence.
    """

    def __init__(self, model_key: str = "M2", role: str = "candidate") -> None:
        """
        Parameters
        ----------
        model_key : str
            Model key used to look up the token budget (e.g. "M1", "M2").
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
        return self.verbalise_after(graph, entity_uri, entity_type, "")

    def verbalise_after(
        self, graph: Graph, entity_uri: URIRef, entity_type: str, preceding: str
    ) -> str:
        """
        Render tier-ordered sentences that fit what ``preceding`` left of the budget.

        Parameters
        ----------
        graph : Graph
            RDF graph containing entity triples.
        entity_uri : URIRef
            URI of the entity to verbalise.
        entity_type : str
            One of ``"class"``, ``"instance"``, ``"predicate"``.
        preceding : str
            Joined text of the earlier components of a composite, or ``""``.

        Returns
        -------
        str
            The kept sentences joined with single spaces.
        """
        tier_list = self._tier_list_for(entity_type)
        ordered = tier_order(self._collect_triples(graph, entity_uri), tier_list)
        entity = self.get_label_or_local(graph, entity_uri)
        sentences = [
            sentence
            for _subj, pred, obj in ordered
            if (sentence := self._render(graph, entity, pred, obj))
        ]
        return _SENTENCE_SEPARATOR.join(fit_items(sentences, _SENTENCE_SEPARATOR, self.budget, preceding))

    @staticmethod
    def _tier_list_for(entity_type: str) -> list[list[str]]:
        """Return the predicate-tier list for *entity_type*."""
        if entity_type not in _TIER_LISTS:
            raise ValueError(
                f"Unsupported entity type: {entity_type!r}. Expected one of "
                "'class', 'instance', 'predicate'."
            )
        return _TIER_LISTS[entity_type]

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
            value = literal_text(obj)
        else:
            value = self.get_label_or_local(graph, obj)

        template = _SENTENCE_TEMPLATES.get(str(pred))
        if template is not None:
            return template.format(entity=entity, object=value)

        predicate = self._resolve_predicate_from_graph(graph, pred)
        fallback = _DATATYPE_TEMPLATE if isinstance(obj, Literal) else _FALLBACK_TEMPLATE
        return fallback.format(entity=entity, predicate=predicate, object=value)
