"""Abstract base class for all verbalisation strategies."""

from abc import ABC, abstractmethod
import hashlib
import json
import re

from rdflib import BNode, Graph, URIRef, Literal
from rdflib.namespace import RDFS, SKOS
from typing import Optional, List

# Standard predicate constants used by all strategies
LABEL_PREDICATES = [
    RDFS.label,
    SKOS.prefLabel,
    URIRef("http://www.w3.org/2004/02/skos/core#altLabel"),
]

_FMA = "http://purl.org/sig/ont/fma/"
_OBO_IN_OWL = "http://www.geneontology.org/formats/oboInOwl#"

# The definition and synonym roles of the ``roles_v1`` role map (#131), so the
# annotation verbalisers read the predicates D1 (SNOMED, FMA) and D2 (MA, NCI)
# actually use.  New definition predicates follow the original three, keeping
# the first-definition-wins choice unchanged wherever those three occur.
DEFINITION_PREDICATES = [
    RDFS.comment,
    SKOS.definition,
    URIRef("http://purl.obolibrary.org/obo/IAO_0000115"),
    URIRef(f"{_FMA}definition"),
    URIRef(f"{_OBO_IN_OWL}hasDefinition"),
]

SYNONYM_PREDICATES = [
    URIRef("http://www.w3.org/2004/02/skos/core#altLabel"),
    URIRef(f"{_FMA}synonym"),
    URIRef(f"{_FMA}slot_synonym"),
    URIRef(f"{_OBO_IN_OWL}hasExactSynonym"),
    URIRef(f"{_OBO_IN_OWL}hasRelatedSynonym"),
    URIRef(f"{_OBO_IN_OWL}hasNarrowSynonym"),
    URIRef(f"{_OBO_IN_OWL}hasBroadSynonym"),
    URIRef(f"{_OBO_IN_OWL}hasSynonym"),
]

_ANNOTATION_VALUE_RULES = (
    "literal objects tagged en, en-* or untagged; non-literal objects "
    "resolve to their own rdfs:label literals under the same language rule"
)

# Predicates and objects excluded from structured key-value serialisation
_EXCLUDED_PREDICATES = [
    "http://www.w3.org/2002/07/owl#sameAs",
]

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


def annotation_predicates_digest() -> str:
    """
    Return the SHA-256 of the annotation predicate lists and value rules.

    Recorded in every result so that a change to what V1/V2 read is visible
    in the results, not only in the code.

    Returns
    -------
    str
        Hex digest of the definition and synonym predicates, in order, and
        the language and node-resolution rules.
    """
    payload = {
        "definition": [str(p) for p in DEFINITION_PREDICATES],
        "synonym": [str(p) for p in SYNONYM_PREDICATES],
        "rules": _ANNOTATION_VALUE_RULES,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_english_or_untagged(literal: Literal) -> bool:
    """
    Return whether ``literal`` is untagged or tagged English (``en``, ``en-*``).

    Parameters
    ----------
    literal : Literal
        Literal to test.

    Returns
    -------
    bool
        ``True`` for untagged literals and tags ``en`` or ``en-<region>``.
    """
    if literal.language is None:
        return True
    language = str(literal.language).lower()
    return language == "en" or language.startswith("en-")


class VerbaliserBase(ABC):
    """Base class for all KG entity verbalisation strategies.

    Subclasses implement ``verbalise`` to define how entity metadata is
    converted into human-readable text for embedding.
    """

    @abstractmethod
    def verbalise(self, graph: Graph, entity_uri: URIRef, entity_type: str) -> str:
        """Convert entity metadata into a textual description.

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
            Non-empty string for entities with at least one label.
            Empty string only for entities with no retrievable info.
        """

    def verbalise_after(
        self, graph: Graph, entity_uri: URIRef, entity_type: str, preceding: str
    ) -> str:
        """Verbalise as a component that follows ``preceding`` in a composite.

        Budgeted strategies fit their text into what ``preceding`` left of
        the budget (#183); every other strategy ignores it.

        Parameters
        ----------
        graph : Graph
            RDF graph containing entity triples.
        entity_uri : URIRef
            URI of the entity to verbalise.
        entity_type : str
            One of ``"class"``, ``"instance"``, ``"predicate"``.
        preceding : str
            Joined text of the earlier components, or ``""``.

        Returns
        -------
        str
            The component's text.
        """
        return self.verbalise(graph, entity_uri, entity_type)

    def get_label(self, graph: Graph, entity_uri: URIRef) -> Optional[str]:
        """Return the first English or untagged label from LABEL_PREDICATES.

        Parameters
        ----------
        graph : Graph
            RDF graph containing entity triples.
        entity_uri : URIRef
            URI of the entity.

        Returns
        -------
        Optional[str]
            Label string or None if nothing found.
        """
        for pred in LABEL_PREDICATES:
            for obj in graph.objects(entity_uri, pred):
                if isinstance(obj, Literal):
                    lang = str(obj.language) if obj.language else ""
                    if lang == "en":
                        return str(obj)
        # Second pass: untagged labels
        for pred in LABEL_PREDICATES:
            for obj in graph.objects(entity_uri, pred):
                if isinstance(obj, Literal) and obj.language is None:
                    return str(obj)
        return None

    def get_local_name(self, uri: URIRef) -> str:
        """Extract a human-readable local name from a URI.

        Splits on ``#`` first, then the last ``/``. Converts camelCase and
        snake_case into space-separated words.

        Parameters
        ----------
        uri : URIRef
            The URI to parse.

        Returns
        -------
        str
            Extracted and formatted local name.
        """
        local = str(uri)
        if "#" in local:
            local = local.split("#")[-1]
        else:
            last_slash = local.rfind("/")
            if last_slash != -1:
                local = local[last_slash + 1:]

        # Convert snake_case to spaces
        local = local.replace("_", " ")
        # Insert space before uppercase letters (camelCase)
        local = re.sub(r"([a-z])([A-Z])", r"\1 \2", local)
        return local

    def get_label_or_local(self, graph: Graph, uri: URIRef) -> str:
        """Return get_label() if found, otherwise get_local_name().

        Parameters
        ----------
        graph : Graph
            RDF graph containing entity triples.
        uri : URIRef
            URI of the entity.

        Returns
        -------
        str
            Label string or local name fallback.
        """
        label = self.get_label(graph, uri)
        if label:
            return label
        return self.get_local_name(uri)

    def get_all_literals(self, graph: Graph, entity_uri: URIRef, predicates: List[URIRef]) -> List[str]:
        """Return all English or untagged literal values for given predicates.

        Parameters
        ----------
        graph : Graph
            RDF graph containing entity triples.
        entity_uri : URIRef
            URI of the entity.
        predicates : List[URIRef]
            List of predicate URIs to query.

        Returns
        -------
        List[str]
            Deduplicated literal strings preserving first-occurrence order.
        """
        seen = set()
        results = []
        for pred in predicates:
            for obj in graph.objects(entity_uri, pred):
                if isinstance(obj, Literal):
                    lang = str(obj.language) if obj.language else ""
                    if lang == "en" or obj.language is None:
                        val = str(obj)
                        if val not in seen:
                            seen.add(val)
                            results.append(val)
        return results

    def get_annotation_values(
        self, graph: Graph, entity_uri: URIRef, predicates: List[URIRef]
    ) -> List[str]:
        """Return English or untagged annotation values for given predicates.

        Unlike :meth:`get_all_literals`, regional English tags (``en-gb``)
        count as English, and a non-literal object -- such as the reified
        ``oboInOwl:Synonym`` nodes of the OAEI Anatomy ontologies -- yields
        its own ``rdfs:label`` values.

        Parameters
        ----------
        graph : Graph
            RDF graph containing entity triples.
        entity_uri : URIRef
            URI of the entity.
        predicates : List[URIRef]
            Predicate URIs to query, in priority order.

        Returns
        -------
        List[str]
            Deduplicated values preserving first-occurrence order.
        """
        values = (
            value
            for pred in predicates
            for obj in graph.objects(entity_uri, pred)
            for value in self._annotation_literals(graph, obj)
        )
        return list(dict.fromkeys(str(value) for value in values))

    @staticmethod
    def _annotation_literals(graph: Graph, obj: URIRef | BNode | Literal) -> List[Literal]:
        """Return ``obj`` itself, or a node's labels, when English or untagged."""
        candidates = [obj] if isinstance(obj, Literal) else list(graph.objects(obj, RDFS.label))
        return [
            value
            for value in candidates
            if isinstance(value, Literal) and is_english_or_untagged(value)
        ]

    # ------------------------------------------------------------------
    # Structured key-value serialisation helpers (shared by V4 and V6)
    # ------------------------------------------------------------------

    def _collect_triples(
        self, graph: Graph, entity_uri: URIRef
    ) -> list[tuple]:
        """Return structured-KV triples for *entity_uri*, filtering noise.

        Parameters
        ----------
        graph : Graph
            RDF graph containing entity triples.
        entity_uri : URIRef
            URI of the entity whose triples are collected.

        Returns
        -------
        list of tuple
            Triples excluding ``owl:sameAs``, blank-node objects, and
            reflexive statements, sorted by ``(subject, predicate, object)``
            so that budget-based filtering receives a stable ordering.
        """
        triples = [
            t
            for t in graph.triples((entity_uri, None, None))
            if not self._should_exclude(*t)
        ]
        return sorted(triples, key=lambda t: (str(t[0]), str(t[1]), str(t[2])))

    @staticmethod
    def _should_exclude(
        subj: URIRef, pred: URIRef, obj: URIRef | BNode | Literal
    ) -> bool:
        """Return True if the triple should be excluded from serialisation."""
        if str(pred) in _EXCLUDED_PREDICATES:
            return True
        if isinstance(obj, BNode):
            return True
        if subj == obj:
            return True
        return False

    def _resolve_predicate_from_graph(
        self, graph: Graph, pred: URIRef
    ) -> str:
        """Resolve a predicate's label from *graph*, else its local name.

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
        """Return ``"pred_local: obj"`` for fast token-cost estimation."""
        pred_label = pred.split("#")[-1].split("/")[-1]
        return f"{pred_label}: {obj}"

    def _format_pair(
        self,
        graph: Graph,
        pred: URIRef,
        obj: URIRef | BNode | Literal,
    ) -> str:
        """Format a single triple as ``"key: value"``.

        Parameters
        ----------
        graph : Graph
            RDF graph used to resolve predicate and object labels.
        pred : URIRef
            Predicate URI of the triple.
        obj : URIRef | BNode | Literal
            Object of the triple.

        Returns
        -------
        str
            ``"key: value"`` string, or empty string for non-serialisable
            objects.
        """
        is_type = str(pred) == _RDF_TYPE
        key = "type" if is_type else self._resolve_predicate_from_graph(graph, pred)

        if isinstance(obj, Literal):
            value = str(obj)
        elif isinstance(obj, URIRef):
            value = self.get_label_or_local(graph, obj)
        else:
            return ""

        return f"{key}: {value}"
