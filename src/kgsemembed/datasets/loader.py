"""OAEI dataset loaders for RDF/XML graphs and EDOAL reference alignments.

Note: Phase 2 datasets are treated as English-only. Label selection prefers
English literals when available.

All OAEI ontology files shipped with this project are RDF/XML with a ``.rdf``
suffix; every :meth:`rdflib.Graph.parse` call on dataset files therefore uses
``format="xml"``. The generic :func:`load_graph` utility still sniffs the
suffix so ad-hoc Turtle files remain loadable outside the D1-D5 flow.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from kgsemembed.utils.errors import DataError

logger = logging.getLogger(__name__)

EntityPair = tuple[str, str]
SplitFn = Callable[[list[EntityPair]], tuple[list[EntityPair], list[EntityPair], list[EntityPair]]]

_ALIGNMENT_NS = "http://knowledgeweb.semanticweb.org/heterogeneity/alignment"
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_EQUIVALENCE_RELATION = "="

_CLASS_TYPES = (OWL.Class, RDFS.Class, SKOS.Concept)
_PREDICATE_TYPES = (OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property)


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
        OAEI track identifier, e.g. ``"D1"``. D4 splits into ``"D4_schema"``
        and ``"D4_instance"``.
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
    train_refs : list[EntityPair]
        Training reference pairs; empty for tracks with no train slice.
    source_id : str
        Short identifier of the source ontology, e.g. ``"mouse"``.
    target_id : str
        Short identifier of the target ontology, e.g. ``"human"``.
    entity_type : str
        Pair-level descriptor: ``"class"``, ``"instance"``, ``"predicate"``,
        or ``"mixed"``. ``"mixed"`` never reaches a verbaliser or embedder;
        individual entities are always resolved to one of the first three.
    kgstore : object | None
        Optional Phase 3 graph store handle; unused in Phase 2.
    """

    dataset_id: str
    pair_name: str
    source_graph: Graph
    target_graph: Graph
    source_entities: list[str]
    val_refs: list[EntityPair]
    test_refs: list[EntityPair]
    train_refs: list[EntityPair] = field(default_factory=list)
    source_id: str = ""
    target_id: str = ""
    entity_type: str = "class"
    kgstore: object | None = None


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


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def _typed_subjects(graph: Graph, rdf_types: tuple[URIRef, ...]) -> list[str]:
    return sorted(
        {
            str(subject)
            for rdf_type in rdf_types
            for subject in graph.subjects(RDF.type, rdf_type)
            if isinstance(subject, URIRef)
        }
    )


def extract_class_uris(graph: Graph) -> list[str]:
    """
    Return sorted URIs of every class-typed subject in ``graph``.

    Parameters
    ----------
    graph : Graph
        RDF graph to inspect.

    Returns
    -------
    list[str]
        Sorted, deduplicated class URIs.
    """
    return _typed_subjects(graph, _CLASS_TYPES)


def extract_predicate_uris(graph: Graph) -> list[str]:
    """
    Return sorted URIs of every property-typed subject in ``graph``.

    Parameters
    ----------
    graph : Graph
        RDF graph to inspect.

    Returns
    -------
    list[str]
        Sorted, deduplicated predicate URIs.
    """
    return _typed_subjects(graph, _PREDICATE_TYPES)


def extract_instance_uris(graph: Graph) -> list[str]:
    """
    Return sorted URIs of every subject that is neither class nor predicate.

    Parameters
    ----------
    graph : Graph
        RDF graph to inspect.

    Returns
    -------
    list[str]
        Sorted, deduplicated instance URIs.
    """
    schema = set(extract_class_uris(graph)) | set(extract_predicate_uris(graph))
    subjects = {str(s) for s in graph.subjects() if isinstance(s, URIRef)}
    return sorted(subjects - schema)


def extract_entity_uris(graph: Graph, entity_type: str) -> list[str]:
    """
    Return the source URIs of ``graph`` matching a pair-level entity type.

    Parameters
    ----------
    graph : Graph
        RDF graph to inspect.
    entity_type : str
        One of ``"class"``, ``"predicate"``, ``"instance"``, ``"mixed"``.

    Returns
    -------
    list[str]
        Sorted, deduplicated entity URIs.

    Raises
    ------
    DataError
        If ``entity_type`` is not a recognised descriptor.
    """
    if entity_type == "class":
        return extract_class_uris(graph)
    if entity_type == "predicate":
        return extract_predicate_uris(graph)
    if entity_type == "mixed":
        return sorted(set(extract_class_uris(graph)) | set(extract_predicate_uris(graph)))
    if entity_type == "instance":
        return extract_instance_uris(graph)
    raise DataError(f"Unknown entity type: {entity_type!r}.")


def _resolve_entity_type(graph: Graph, uri: str) -> str:
    """
    Determine whether ``uri`` is a class, predicate, or instance.

    Parameters
    ----------
    graph : Graph
        Graph supplying the ``rdf:type`` triples for ``uri``.
    uri : str
        Entity URI to classify.

    Returns
    -------
    str
        One of ``"class"``, ``"predicate"``, ``"instance"``.
    """
    types = set(graph.objects(URIRef(uri), RDF.type))
    if OWL.Class in types or RDFS.Class in types:
        return "class"
    if OWL.ObjectProperty in types or OWL.DatatypeProperty in types or RDF.Property in types:
        return "predicate"
    return "instance"


# ---------------------------------------------------------------------------
# Reference parsing
# ---------------------------------------------------------------------------


def _cell_pair(cell: ET.Element) -> EntityPair | None:
    relation = cell.find(f"{{{_ALIGNMENT_NS}}}relation")
    if relation is None or (relation.text or "").strip() != _EQUIVALENCE_RELATION:
        return None
    entity1 = cell.find(f"{{{_ALIGNMENT_NS}}}entity1")
    entity2 = cell.find(f"{{{_ALIGNMENT_NS}}}entity2")
    if entity1 is None or entity2 is None:
        return None
    source = entity1.get(f"{{{_RDF_NS}}}resource")
    target = entity2.get(f"{{{_RDF_NS}}}resource")
    if not source or not target:
        return None
    return (str(source), str(target))


def _parse_alignment_refs(path: Path) -> list[EntityPair]:
    """
    Parse an OAEI EDOAL RDF/XML alignment file into reference pairs.

    Uses :mod:`xml.etree.ElementTree` rather than rdflib: the OAEI alignment
    namespace ends in neither ``#`` nor ``/``, so rdflib's namespace
    concatenation yields malformed element names.

    Only equivalence relations are returned; subsumption cells are dropped.
    This filter is mandatory and not configurable.

    Parameters
    ----------
    path : Path
        Path to the EDOAL alignment file.

    Returns
    -------
    list[EntityPair]
        ``(source_uri, target_uri)`` string tuples.

    Raises
    ------
    DataError
        If the file is missing or is not well-formed XML.
    """
    if not Path(path).exists():
        raise DataError(f"Reference alignment not found: {path}")
    try:
        root = ET.parse(str(path)).getroot()
    except ET.ParseError as exc:
        raise DataError(f"Malformed alignment XML at {path}: {exc}") from exc

    pairs = (_cell_pair(cell) for cell in root.iter(f"{{{_ALIGNMENT_NS}}}Cell"))
    return [pair for pair in pairs if pair is not None]


def load_tsv_alignment(path: str | Path) -> list[EntityPair]:
    """
    Read a whitespace-delimited alignment file into reference pairs.

    Parameters
    ----------
    path : str | Path
        Path to a two-column links file, e.g. OpenEA ``valid_links``.

    Returns
    -------
    list[EntityPair]
        ``(source_uri, target_uri)`` string tuples.

    Raises
    ------
    DataError
        If the file does not exist.
    """
    links_path = Path(path)
    if not links_path.exists():
        raise DataError(f"Reference file not found: {links_path}")
    rows: list[EntityPair] = []
    for line in links_path.read_text(encoding="utf-8").splitlines():
        columns = line.split()
        if len(columns) >= 2:
            rows.append((str(columns[0]), str(columns[1])))
    return rows


# ---------------------------------------------------------------------------
# Deterministic splits
# ---------------------------------------------------------------------------


def _sorted_by_source(refs: list[EntityPair]) -> list[EntityPair]:
    return sorted(refs, key=lambda pair: (pair[0], pair[1]))


def _split_80_10_10(
    refs: list[EntityPair],
) -> tuple[list[EntityPair], list[EntityPair], list[EntityPair]]:
    """
    Split references 80/10/10 by sorted source URI, with no shuffling.

    Parameters
    ----------
    refs : list[EntityPair]
        Reference pairs in any order.

    Returns
    -------
    tuple[list[EntityPair], list[EntityPair], list[EntityPair]]
        Train, validation, and test slices.
    """
    ordered = _sorted_by_source(refs)
    count = len(ordered)
    n_train = int(count * 0.8)
    n_val = int(count * 0.1)
    return ordered[:n_train], ordered[n_train : n_train + n_val], ordered[n_train + n_val :]


def _split_val_test_20_80(
    refs: list[EntityPair],
) -> tuple[list[EntityPair], list[EntityPair], list[EntityPair]]:
    """
    Split references into an empty train slice, 20% validation, 80% test.

    Used for D3 Conference pairs, which are too small for a 10% val slice.

    Parameters
    ----------
    refs : list[EntityPair]
        Reference pairs in any order.

    Returns
    -------
    tuple[list[EntityPair], list[EntityPair], list[EntityPair]]
        Empty train slice, validation, and test slices.
    """
    ordered = _sorted_by_source(refs)
    n_val = max(1, int(len(ordered) * 0.2)) if ordered else 0
    return [], ordered[:n_val], ordered[n_val:]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _parse_rdf_xml(path: Path) -> Graph:
    if not path.exists():
        raise DataError(f"Ontology file not found: {path}")
    graph = Graph()
    try:
        graph.parse(str(path), format="xml")
    except Exception as exc:
        raise DataError(f"Failed to parse RDF graph at {path}: {exc}") from exc
    return graph


def _triple_object(value: str) -> URIRef | Literal:
    if value.startswith(("http://", "https://")):
        return URIRef(value)
    return Literal(value)


def _graph_from_triple_files(rel_path: Path, attr_path: Path) -> Graph:
    """
    Build a graph from OpenEA relation and attribute triple files.

    Both files are tab-separated triples. Attribute objects starting with an
    HTTP scheme become :class:`URIRef`; everything else becomes a plain
    :class:`Literal`, which is sufficient because verbalisers use ``str(obj)``.

    Parameters
    ----------
    rel_path : Path
        Path to a ``rel_triples_*`` file.
    attr_path : Path
        Path to an ``attr_triples_*`` file.

    Returns
    -------
    Graph
        Graph containing every well-formed triple from both files.
    """
    graph = Graph()
    for path, as_object in ((rel_path, URIRef), (attr_path, _triple_object)):
        if not path.exists():
            raise DataError(f"Triple file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            columns = line.strip().split("\t")
            if len(columns) == 3:
                graph.add((URIRef(columns[0]), URIRef(columns[1]), as_object(columns[2])))
    return graph


def _build_pair(
    dataset_id: str,
    pair_name: str,
    graphs: tuple[Graph, Graph],
    ids: tuple[str, str],
    entity_type: str,
    splits: tuple[list[EntityPair], list[EntityPair], list[EntityPair]],
) -> AlignmentPair:
    source_graph, target_graph = graphs
    train_refs, val_refs, test_refs = splits
    return AlignmentPair(
        dataset_id=dataset_id,
        pair_name=pair_name,
        source_graph=source_graph,
        target_graph=target_graph,
        source_entities=extract_entity_uris(source_graph, entity_type),
        val_refs=val_refs,
        test_refs=test_refs,
        train_refs=train_refs,
        source_id=ids[0],
        target_id=ids[1],
        entity_type=entity_type,
        kgstore=None,
    )


def _load_rdf_pair(
    pair_dir: Path,
    pair_name: str,
    ids: tuple[str, str],
    dataset_id: str,
    entity_type: str,
    split_fn: SplitFn,
) -> AlignmentPair:
    """
    Load one alignment pair from a ``source.rdf`` / ``target.rdf`` directory.

    Parameters
    ----------
    pair_dir : Path
        Directory holding ``source.rdf``, ``target.rdf``, ``reference.rdf``.
    pair_name : str
        Name recorded on the resulting pair.
    ids : tuple[str, str]
        Source and target ontology identifiers.
    dataset_id : str
        OAEI track identifier.
    entity_type : str
        Pair-level entity descriptor.
    split_fn : SplitFn
        Deterministic train/val/test splitter.

    Returns
    -------
    AlignmentPair
        Fully populated alignment pair.
    """
    source_graph = _parse_rdf_xml(pair_dir / "source.rdf")
    target_graph = _parse_rdf_xml(pair_dir / "target.rdf")
    splits = split_fn(_parse_alignment_refs(pair_dir / "reference.rdf"))
    return _build_pair(
        dataset_id, pair_name, (source_graph, target_graph), ids, entity_type, splits
    )


def load_pair_from_dir(
    pair_dir: str | Path,
    pair_name: str = "sample",
    ids: tuple[str, str] = ("source", "target"),
    dataset_id: str = "sample",
    entity_type: str = "class",
) -> AlignmentPair:
    """
    Load any directory holding ``source.rdf``, ``target.rdf``, ``reference.rdf``.

    Lets smoke tests load ``data/sample/`` without it being treated as one of
    the named D1-D5 datasets.

    Parameters
    ----------
    pair_dir : str | Path
        Directory containing the three RDF/XML files.
    pair_name : str
        Name recorded on the resulting pair.
    ids : tuple[str, str]
        Source and target ontology identifiers.
    dataset_id : str
        Track identifier recorded on the resulting pair.
    entity_type : str
        Pair-level entity descriptor.

    Returns
    -------
    AlignmentPair
        Alignment pair split 80/10/10 by sorted source URI.
    """
    return _load_rdf_pair(
        Path(pair_dir), pair_name, ids, dataset_id, entity_type, _split_80_10_10
    )


# ---------------------------------------------------------------------------
# Per-dataset enumerators
# ---------------------------------------------------------------------------


def _load_d1(data_dir: Path) -> list[AlignmentPair]:
    pair_dir = data_dir / "d1_snomed_fma"
    return [
        _load_rdf_pair(
            pair_dir, "d1_snomed_fma", ("snomed", "fma"), "D1", "class", _split_80_10_10
        )
    ]


def _load_d2(data_dir: Path) -> list[AlignmentPair]:
    pair_dir = data_dir / "d2_anatomy"
    return [
        _load_rdf_pair(
            pair_dir, "d2_anatomy", ("mouse", "human"), "D2", "class", _split_80_10_10
        )
    ]


def _load_d3(data_dir: Path) -> list[AlignmentPair]:
    dataset_dir = data_dir / "d3_conference"
    if not dataset_dir.is_dir():
        raise DataError(f"Dataset directory not found: {dataset_dir}")
    pairs: list[AlignmentPair] = []
    for pair_dir in sorted(child for child in dataset_dir.iterdir() if child.is_dir()):
        ids = pair_dir.name.split("-", 1)
        if len(ids) != 2:
            logger.warning("Skipping unexpected D3 folder: %s", pair_dir.name)
            continue
        pairs.append(
            _load_rdf_pair(
                pair_dir, pair_dir.name, (ids[0], ids[1]), "D3", "mixed", _split_val_test_20_80
            )
        )
    if len(pairs) != 21:
        logger.warning("Expected 21 D3 pairs, found %d", len(pairs))
    return pairs


def _partition_d4_refs(
    graph: Graph, refs: list[EntityPair]
) -> tuple[list[EntityPair], list[EntityPair]]:
    schema_refs: list[EntityPair] = []
    instance_refs: list[EntityPair] = []
    for source_uri, target_uri in refs:
        bucket = (
            schema_refs
            if _resolve_entity_type(graph, source_uri) in {"class", "predicate"}
            else instance_refs
        )
        bucket.append((source_uri, target_uri))
    return schema_refs, instance_refs


def _load_d4(data_dir: Path) -> list[AlignmentPair]:
    dataset_dir = data_dir / "d4_kgtrack"
    source_graph = _parse_rdf_xml(dataset_dir / "ontologies" / "memoryalpha.rdf")
    target_graph = _parse_rdf_xml(dataset_dir / "ontologies" / "stexpanded.rdf")
    refs = _parse_alignment_refs(dataset_dir / "references" / "memoryalpha-stexpanded.rdf")
    schema_refs, instance_refs = _partition_d4_refs(source_graph, refs)

    graphs = (source_graph, target_graph)
    ids = ("memoryalpha", "stexpanded")
    return [
        _build_pair(
            "D4_schema", "memoryalpha-stexpanded-schema", graphs, ids, "mixed",
            _split_80_10_10(schema_refs),
        ),
        _build_pair(
            "D4_instance", "memoryalpha-stexpanded-instance", graphs, ids, "instance",
            _split_80_10_10(instance_refs),
        ),
    ]


def _load_d5(data_dir: Path) -> list[AlignmentPair]:
    dataset_dir = data_dir / "d5_openea" / "D_W_15K_V2"
    source_graph = _graph_from_triple_files(
        dataset_dir / "rel_triples_1", dataset_dir / "attr_triples_1"
    )
    target_graph = _graph_from_triple_files(
        dataset_dir / "rel_triples_2", dataset_dir / "attr_triples_2"
    )
    fold_dir = dataset_dir / "721_5fold" / "1"
    splits = (
        [],
        load_tsv_alignment(fold_dir / "valid_links"),
        load_tsv_alignment(fold_dir / "test_links"),
    )
    return [
        _build_pair(
            "D5", "D_W_15K_V2", (source_graph, target_graph),
            ("dbpedia", "wikidata"), "instance", splits,
        )
    ]


_DATASET_LOADERS: dict[str, Callable[[Path], list[AlignmentPair]]] = {
    "D1": _load_d1,
    "D2": _load_d2,
    "D3": _load_d3,
    "D4": _load_d4,
    "D5": _load_d5,
}


def load_dataset(dataset_id: str, data_dir: str | Path = "data/") -> list[AlignmentPair]:
    """
    Load every alignment pair for ``dataset_id`` with dataset-specific splits.

    Each track has its own on-disk layout: D1, D2, and every D3 subdirectory
    hold ``source.rdf``, ``target.rdf``, and an EDOAL ``reference.rdf``; D4
    keeps its ontologies and references in sibling directories; D5 ships
    OpenEA triple files. Splits follow the fixed per-track convention and are
    never re-mixed.

    Parameters
    ----------
    dataset_id : str
        OAEI track identifier, one of ``"D1"`` through ``"D5"``.
    data_dir : str | Path
        Dataset root directory (default ``"data/"``).

    Returns
    -------
    list[AlignmentPair]
        Alignment pairs in deterministic order. D4 returns two pairs sharing
        the same source and target graph objects.

    Raises
    ------
    DataError
        If ``dataset_id`` is unknown or its files cannot be read.
    """
    loader = _DATASET_LOADERS.get(dataset_id)
    if loader is None:
        raise DataError(f"Unknown dataset identifier: {dataset_id!r}.")
    return loader(Path(data_dir))
