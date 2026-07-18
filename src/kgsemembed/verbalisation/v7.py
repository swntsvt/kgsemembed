"""Hierarchical context verbalisation strategy (V7).

Extends the annotation-based representation (V2) with taxonomic context.  The
V2 annotation is treated as the canonical entity description and is augmented
with an ancestor chain and a bounded set of sibling entities drawn from the
appropriate ontology hierarchy::

    Label: Heart. Definition: A hollow muscular organ.

    Parents: Organ -> Anatomical structure -> Material entity.
    Related types include: Kidney, Liver, Lung.

Hierarchy traversal follows ``rdfs:subClassOf`` for classes, ``rdf:type`` then
``rdfs:subClassOf`` for instances, and ``rdfs:subPropertyOf`` for predicates.
Traversal is depth-bounded, cycle-safe, and never raises for malformed graphs.
Only named (``URIRef``) parents are followed -- blank-node parents are ignored,
so a class whose only parent is anonymous terminates the chain safely, while a
class that also has a named parent still yields its taxonomic ancestors.  When
multiple named parents exist, the lexicographically smallest is followed,
keeping the chain linear and deterministic across graph re-parses.  When the
combined output exceeds the model token budget,
siblings are dropped first and the ancestor chain is then shortened from its
most distant end; the V2 annotation text is never truncated (models without a
budget, e.g. ``"M3"``, retain the full hierarchy).
"""

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS

from kgsemembed.verbalisation.base import VerbaliserBase
from kgsemembed.verbalisation.ppas import PPAS_BUDGETS, estimate_tokens
from kgsemembed.verbalisation.v2 import AnnotationVerbaliser

MAX_ANCESTOR_DEPTH = 3
MAX_SIBLINGS = 5

_HIERARCHY_PREDICATES: dict[str, tuple[URIRef, URIRef]] = {
    "class": (RDFS.subClassOf, RDFS.subClassOf),
    "instance": (RDF.type, RDFS.subClassOf),
    "predicate": (RDFS.subPropertyOf, RDFS.subPropertyOf),
}


class HierarchicalContextVerbaliser(VerbaliserBase):
    """Augment the V2 annotation with ancestor and sibling context.

    Output format::

        <V2 annotation>

        Parents: Parent1 -> Parent2 -> Parent3.
        Related types include: Sibling1, Sibling2.

    Either contextual line is omitted when the corresponding hierarchy is
    empty.  The whole output is deterministic and capped to the model token
    budget without ever truncating the annotation text.
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
        Verbalise an entity as its V2 annotation plus hierarchical context.

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
            The V2 annotation augmented with budget-fitted ancestor and
            sibling context.

        Raises
        ------
        ValueError
            If *entity_type* is not a supported entity type.
        """
        self._predicates_for(entity_type)
        annotation = AnnotationVerbaliser().verbalise(
            graph, entity_uri, entity_type
        )
        ancestors = self._get_ancestors(graph, entity_uri, entity_type)
        ancestor_labels = [self.get_label_or_local(graph, a) for a in ancestors]
        siblings = self._get_siblings(graph, entity_uri, entity_type)
        return self._apply_budget(annotation, ancestor_labels, siblings)

    @staticmethod
    def _predicates_for(entity_type: str) -> tuple[URIRef, URIRef]:
        """Return the ``(immediate-parent, climb)`` predicates for a type."""
        if entity_type not in _HIERARCHY_PREDICATES:
            raise ValueError(
                f"Unsupported entity type: {entity_type!r}. Expected one of "
                "'class', 'instance', 'predicate'."
            )
        return _HIERARCHY_PREDICATES[entity_type]

    def _get_ancestors(
        self, graph: Graph, entity_uri: URIRef, entity_type: str
    ) -> list[URIRef]:
        """Return the ancestor chain, nearest parent first.

        Traversal stops at ``MAX_ANCESTOR_DEPTH``, when no named parent exists
        (including when the only parent is anonymous), or when a cycle is
        detected.  Malformed hierarchies never raise.
        """
        first_pred, climb_pred = self._predicates_for(entity_type)
        ancestors: list[URIRef] = []
        visited: set[URIRef] = {entity_uri}
        current = self._first_parent(graph, entity_uri, first_pred)
        while current is not None and len(ancestors) < MAX_ANCESTOR_DEPTH:
            if current in visited:
                break
            ancestors.append(current)
            visited.add(current)
            current = self._first_parent(graph, current, climb_pred)
        return ancestors

    def _get_siblings(
        self, graph: Graph, entity_uri: URIRef, entity_type: str
    ) -> list[str]:
        """Return up to ``MAX_SIBLINGS`` labels sharing the immediate parent.

        The current entity is excluded and siblings are sorted by label then
        URI so selection and ordering are deterministic.
        """
        first_pred, _ = self._predicates_for(entity_type)
        parent = self._first_parent(graph, entity_uri, first_pred)
        if parent is None:
            return []
        candidates = {
            subject
            for subject in graph.subjects(first_pred, parent)
            if isinstance(subject, URIRef) and subject != entity_uri
        }
        labelled = sorted(
            (self.get_label_or_local(graph, s), str(s)) for s in candidates
        )
        return [label for label, _ in labelled[:MAX_SIBLINGS]]

    @staticmethod
    def _first_parent(
        graph: Graph, node: URIRef, pred: URIRef
    ) -> URIRef | None:
        """Return the smallest named parent, ignoring blank-node parents.

        Blank-node (anonymous) parents are skipped so that named taxonomic
        ancestors are never dropped and selection stays deterministic across
        graph re-parses.  Returns ``None`` when no named parent exists.
        """
        named = sorted(
            (
                obj
                for obj in graph.objects(node, pred)
                if isinstance(obj, URIRef)
            ),
            key=str,
        )
        return named[0] if named else None

    def _apply_budget(
        self,
        annotation: str,
        ancestor_labels: list[str],
        sibling_labels: list[str],
    ) -> str:
        """Fit the output to the model budget, preserving the annotation.

        Siblings are removed first, then the ancestor chain is shortened from
        its most distant end.  Models without a budget keep the full hierarchy.
        """
        budget = PPAS_BUDGETS.get(self.model_key)
        output = self._format(annotation, ancestor_labels, sibling_labels)
        if budget is None or estimate_tokens(output) <= budget:
            return output

        ancestors = list(ancestor_labels)
        while True:
            output = self._format(annotation, ancestors, [])
            if not ancestors or estimate_tokens(output) <= budget:
                return output
            ancestors.pop()

    @staticmethod
    def _format(
        annotation: str,
        ancestor_labels: list[str],
        sibling_labels: list[str],
    ) -> str:
        """Append the hierarchy lines to *annotation*, omitting empty ones."""
        lines: list[str] = []
        if ancestor_labels:
            lines.append(f"Parents: {' -> '.join(ancestor_labels)}.")
        if sibling_labels:
            lines.append(f"Related types include: {', '.join(sibling_labels)}.")
        if not lines:
            return annotation
        return annotation + "\n\n" + "\n".join(lines)
