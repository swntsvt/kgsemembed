"""OAEI dataset loaders for RDF/XML and Turtle graphs.

Note: Phase 2 datasets are treated as English-only. Label selection prefers
English literals when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from kgsemembed.utils.errors import DataError

EntityPair = tuple[str, str]

_DATASET_ROOT = "datasets"
_RDF_SUFFIX_FORMATS = {".rdf": "xml", ".xml": "xml", ".owl": "xml", ".ttl": "turtle"}
_ONTOLOGY_ENTITY_TYPES = (OWL.Class, SKOS.Concept)


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


@dataclass(frozen=True)
class AlignmentPair:
    """
    A single source/target ontology pair with its held-out reference splits.

    Attributes
    ----------
    dataset_id : str
        OAEI track identifier, e.g. ``"D1"``.
    pair_name : str
        Unique name of the ontology pair, e.g. ``"d1_snomed_fma"``.
    source_graph : Graph
        Parsed RDF graph of the source ontology, used for verbalisation.
    target_graph : Graph
        Parsed RDF graph of the target ontology, used for verbalisation.
    source_entities : list[str]
        Source entity URIs to be matched, in deterministic order.
    val_refs : list[EntityPair]
        Validation reference pairs, used only for threshold tuning.
    test_refs : list[EntityPair]
        Test reference pairs, used only for final metric computation.
    """

    dataset_id: str
    pair_name: str
    source_graph: Graph
    target_graph: Graph
    source_entities: list[str]
    val_refs: list[EntityPair]
    test_refs: list[EntityPair]


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


def _load_graph_file(pair_dir: Path, stem: str) -> Graph:
    for suffix, rdf_format in _RDF_SUFFIX_FORMATS.items():
        path = pair_dir / f"{stem}{suffix}"
        if not path.exists():
            continue
        graph = Graph()
        try:
            graph.parse(path, format=rdf_format)
        except Exception as exc:
            raise DataError(f"Failed to parse RDF graph at {path}: {exc}") from exc
        return graph
    raise DataError(f"No RDF graph named {stem!r} found under {pair_dir}")


def _ontology_entities(graph: Graph) -> list[str]:
    typed = {
        str(subject)
        for entity_type in _ONTOLOGY_ENTITY_TYPES
        for subject in graph.subjects(RDF.type, entity_type)
        if isinstance(subject, URIRef)
    }
    if typed:
        return sorted(typed)
    return sorted({str(s) for s in graph.subjects() if isinstance(s, URIRef)})


def _read_ref_rows(path: Path) -> list[EntityPair]:
    if not path.exists():
        raise DataError(f"Reference file not found: {path}")
    rows: list[EntityPair] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        columns = line.split()
        if len(columns) >= 2:
            rows.append((columns[0], columns[1]))
    return rows


def _sorted_by_source(refs: list[EntityPair]) -> list[EntityPair]:
    return sorted(refs, key=lambda pair: (pair[0], pair[1]))


def _split_80_10_10(refs: list[EntityPair]) -> tuple[list[EntityPair], list[EntityPair]]:
    ordered = _sorted_by_source(refs)
    count = len(ordered)
    return ordered[int(count * 0.8) : int(count * 0.9)], ordered[int(count * 0.9) :]


def _split_first_fraction(
    refs: list[EntityPair], fraction: float
) -> tuple[list[EntityPair], list[EntityPair]]:
    ordered = _sorted_by_source(refs)
    cut = int(len(ordered) * fraction)
    return ordered[:cut], ordered[cut:]


def _refs_d1(pair_dir: Path) -> tuple[list[EntityPair], list[EntityPair]]:
    equiv = pair_dir / "refs_equiv"
    return _read_ref_rows(equiv / "val.tsv"), _read_ref_rows(equiv / "test.tsv")


def _refs_d2(pair_dir: Path) -> tuple[list[EntityPair], list[EntityPair]]:
    return _split_80_10_10(_read_ref_rows(pair_dir / "refs.tsv"))


def _refs_d3(pair_dir: Path) -> tuple[list[EntityPair], list[EntityPair]]:
    return _split_first_fraction(_read_ref_rows(pair_dir / "refs.tsv"), 0.2)


def _refs_d4(pair_dir: Path) -> tuple[list[EntityPair], list[EntityPair]]:
    val_schema, test_schema = _split_80_10_10(_read_ref_rows(pair_dir / "refs_schema.tsv"))
    val_instance, test_instance = _split_80_10_10(
        _read_ref_rows(pair_dir / "refs_instance.tsv")
    )
    return val_schema + val_instance, test_schema + test_instance


def _refs_d5(pair_dir: Path) -> tuple[list[EntityPair], list[EntityPair]]:
    fold = pair_dir / "721_5fold" / "1"
    return _read_ref_rows(fold / "valid_links"), _read_ref_rows(fold / "test_links")


_RefLoader = Callable[[Path], tuple[list[EntityPair], list[EntityPair]]]

_REF_LOADERS: dict[str, _RefLoader] = {
    "D1": _refs_d1,
    "D2": _refs_d2,
    "D3": _refs_d3,
    "D4": _refs_d4,
    "D5": _refs_d5,
}


def _load_alignment_pair(
    dataset_id: str, pair_dir: Path, ref_loader: _RefLoader
) -> AlignmentPair:
    source_graph = _load_graph_file(pair_dir, "source")
    target_graph = _load_graph_file(pair_dir, "target")
    val_refs, test_refs = ref_loader(pair_dir)
    return AlignmentPair(
        dataset_id=dataset_id,
        pair_name=pair_dir.name,
        source_graph=source_graph,
        target_graph=target_graph,
        source_entities=_ontology_entities(source_graph),
        val_refs=val_refs,
        test_refs=test_refs,
    )


def load_dataset(dataset_id: str, data_dir: str | Path = "data/") -> list[AlignmentPair]:
    """
    Load every alignment pair for ``dataset_id`` with dataset-specific splits.

    Ontology pairs live under ``{data_dir}/datasets/{dataset_id}/{pair_name}/``
    with ``source`` and ``target`` RDF graphs and per-dataset reference files.
    Validation and test references are split according to the fixed OAEI-track
    convention for each dataset and are never re-mixed.

    Parameters
    ----------
    dataset_id : str
        OAEI track identifier, one of ``"D1"`` through ``"D5"``.
    data_dir : str | Path
        Dataset root directory (default ``"data/"``).

    Returns
    -------
    list[AlignmentPair]
        Alignment pairs in deterministic ``pair_name`` order.

    Raises
    ------
    DataError
        If ``dataset_id`` is unknown or its directory does not exist.
    """
    ref_loader = _REF_LOADERS.get(dataset_id)
    if ref_loader is None:
        raise DataError(f"Unknown dataset identifier: {dataset_id!r}.")
    dataset_dir = Path(data_dir) / _DATASET_ROOT / dataset_id
    if not dataset_dir.is_dir():
        raise DataError(f"Dataset directory not found: {dataset_dir}")
    pair_dirs = sorted(child for child in dataset_dir.iterdir() if child.is_dir())
    return [_load_alignment_pair(dataset_id, pair_dir, ref_loader) for pair_dir in pair_dirs]
