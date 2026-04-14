"""OAEI dataset loaders for RDF/XML and Turtle graphs.

Note: Phase 2 datasets are treated as English-only. Label selection prefers
English literals when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDFS, SKOS

from kgsemembed.utils.errors import DataError


@dataclass(frozen=True)
class LoadedGraphData:
    path: Path
    format: str
    triples: list[tuple[str, str, str]]
    entities: set[str]
    labels: dict[str, str]


@dataclass(frozen=True)
class LoadedDatasetBundle:
    source: LoadedGraphData
    target: LoadedGraphData
    alignment: LoadedGraphData


def _detect_rdflib_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".rdf", ".xml"}:
        return "xml"
    if suffix == ".ttl":
        return "turtle"
    raise DataError(f"Unsupported RDF format for path: {path}")


def _uri_local_name(uri: str) -> str:
    if "#" in uri:
        return unquote(uri.rsplit("#", 1)[-1])
    parsed = urlparse(uri)
    if parsed.path:
        return unquote(parsed.path.rstrip("/").rsplit("/", 1)[-1])
    return uri


def _choose_literal(candidates: list[Literal]) -> str | None:
    if not candidates:
        return None
    # Phase 2 assumption: datasets are English-only. Prefer explicit 'en'.
    for lit in candidates:
        if (lit.language or "").lower() == "en":
            return str(lit)
    for lit in candidates:
        if lit.language in {None, ""}:
            return str(lit)
    return str(candidates[0])


def _extract_entities(graph: Graph) -> set[str]:
    entities: set[str] = set()
    for subj, _, obj in graph:
        if isinstance(subj, URIRef):
            entities.add(str(subj))
        if isinstance(obj, URIRef):
            entities.add(str(obj))
    return entities


def _extract_labels(graph: Graph, entities: set[str]) -> dict[str, str]:
    labels: dict[str, str] = {}

    for entity in entities:
        entity_ref = URIRef(entity)
        rdfs_candidates = [obj for obj in graph.objects(entity_ref, RDFS.label) if isinstance(obj, Literal)]
        chosen = _choose_literal(rdfs_candidates)
        if chosen is not None:
            labels[entity] = chosen
            continue

        skos_candidates = [obj for obj in graph.objects(entity_ref, SKOS.prefLabel) if isinstance(obj, Literal)]
        chosen = _choose_literal(skos_candidates)
        if chosen is not None:
            labels[entity] = chosen
            continue

        labels[entity] = _uri_local_name(entity)

    return labels


def load_graph(path: str | Path) -> LoadedGraphData:
    graph_path = Path(path)
    if not graph_path.exists():
        raise DataError(f"Dataset graph not found: {graph_path}")

    rdf_format = _detect_rdflib_format(graph_path)
    graph = Graph()
    try:
        graph.parse(graph_path, format=rdf_format)
    except Exception as exc:
        raise DataError(f"Failed to parse RDF graph at {graph_path}: {exc}") from exc

    entities = _extract_entities(graph)
    labels = _extract_labels(graph, entities)
    triples = [(str(s), str(p), str(o)) for s, p, o in graph]

    return LoadedGraphData(
        path=graph_path,
        format=rdf_format,
        triples=triples,
        entities=entities,
        labels=labels,
    )


def load_oaei_dataset(
    source_path: str | Path,
    target_path: str | Path,
    alignment_path: str | Path,
) -> LoadedDatasetBundle:
    return LoadedDatasetBundle(
        source=load_graph(source_path),
        target=load_graph(target_path),
        alignment=load_graph(alignment_path),
    )
