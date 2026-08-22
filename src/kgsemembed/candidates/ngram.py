"""Inverted-index character n-gram candidate generation and loading.

Stage-1 lexical filtering scores every source label against every target label.
Doing that pairwise is quadratic and does not scale to the instance-level
tracks, so this module builds an inverted index over target n-grams — a sparse
term-document matrix whose columns are n-grams — and scores a whole block of
sources per matrix product. Pairs sharing no n-gram score zero and are never
materialised.

Scores and ordering match the pairwise implementation in
:mod:`kgsemembed.candidates.generator`: tokenisation is shared with it, ties
break on ascending target URI, and sources with fewer than ``top_k`` non-zero
matches are padded with the lexicographically smallest remaining targets.

Generated candidate lists are persisted as JSON under
``{data_dir}/candidates/{dataset_id}/{pair_name}_candidates.json`` and read
back into the Phase 2 embedding pipeline by :func:`load_candidates`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np
import orjson
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDFS, SKOS
from scipy import sparse

from kgsemembed.candidates.generator import CandidatePair, _build_tokens
from kgsemembed.utils.errors import ConfigurationError, DataError

CandidateMap = Dict[str, List[str]]

_CANDIDATE_ROOT = "candidates"
_BLOCK_ROWS = 256

_LABEL_PREDICATES = (RDFS.label, SKOS.prefLabel)
_OPAQUE_ID_PATTERN = re.compile(r"^[EQ]\d+$")
_TYPED_LITERAL_PATTERN = re.compile(r'^"(.*)"\^\^<[^>]*>$', re.DOTALL)


@dataclass(frozen=True)
class NgramConfig:
    """Character n-gram scoring settings shared by index build and query."""

    n: int = 3
    metric: str = "cosine"
    top_k: int = 20
    strip_punctuation: bool = True


@dataclass(frozen=True)
class NgramIndex:
    """
    Inverted index over the character n-grams of a target label set.

    Attributes
    ----------
    uris : list[str]
        Target URIs in ascending order; row ``i`` of ``matrix`` describes
        ``uris[i]``, so a row index doubles as the URI sort key.
    vocabulary : dict[str, int]
        Mapping from n-gram to its column in ``matrix``.
    matrix : sparse.csr_matrix
        Target n-gram count matrix.
    norms : np.ndarray
        Per-target L2 norm of the count vector, for cosine scoring.
    distinct : np.ndarray
        Per-target count of distinct n-grams, for Jaccard scoring.
    """

    uris: List[str]
    vocabulary: Dict[str, int]
    matrix: sparse.csr_matrix
    norms: np.ndarray
    distinct: np.ndarray


def _literal_text(obj: Literal) -> str:
    """Return a literal's lexical form, stripping any inlined datatype suffix.

    OpenEA triple files store typed literals as ``"1955"^^<...#gYear>`` inside a
    single tab-separated column, so :func:`_graph_from_triple_files` preserves
    the datatype URI as part of the literal's own string.
    """
    match = _TYPED_LITERAL_PATTERN.match(str(obj).strip())
    return (match.group(1) if match else str(obj)).strip()


def _is_informative(text: str, min_length: int, max_length: int) -> bool:
    """Return whether *text* is a usable length and not a bare quantity."""
    if not min_length <= len(text) <= max_length:
        return False
    return not text.replace(".", "").replace("-", "").replace(" ", "").isnumeric()


def get_entity_text_from_attributes(
    graph: Graph | None,
    entity_uri: str,
    max_values: int = 10,
    min_length: int = 3,
    max_length: int = 200,
) -> str:
    """
    Build a text representation of an entity from its Literal-valued triples.

    Fallback for entities carrying no label, notably D5 OpenEA entities whose
    ``rdfs:label`` triples were deliberately removed. Literal objects are taken
    in ascending predicate then value order so the result does not depend on
    graph store iteration order.

    Parameters
    ----------
    graph : Graph | None
        Graph supplying the entity's triples. ``None`` yields an empty string.
    entity_uri : str
        URI of the entity to describe.
    max_values : int
        Maximum number of literal values to concatenate.
    min_length : int
        Shortest accepted value, in characters.
    max_length : int
        Longest accepted value, in characters.

    Returns
    -------
    str
        Space-joined literal values, or ``""`` if none qualify.
    """
    if graph is None:
        return ""
    triples = list(graph.triples((URIRef(entity_uri), None, None)))
    values = sorted(
        (str(predicate), _literal_text(obj))
        for _, predicate, obj in triples
        if isinstance(obj, Literal)
    )
    kept = [text for _, text in values if _is_informative(text, min_length, max_length)]
    return " ".join(kept[:max_values])


def _local_name(uri: str) -> str:
    """Return the trailing URI segment with underscores read as spaces."""
    tail = uri.rsplit("#", 1)[-1] if "#" in uri else uri.rstrip("/").rsplit("/", 1)[-1]
    return tail.replace("_", " ")


def _first_label(graph: Graph, entity_uri: str) -> str | None:
    """Return the first English ``rdfs:label`` or ``skos:prefLabel`` literal."""
    for predicate in _LABEL_PREDICATES:
        for obj in graph.objects(URIRef(entity_uri), predicate):
            if isinstance(obj, Literal) and (obj.language or "en").lower() == "en":
                return str(obj)
    return None


def get_entity_label(graph: Graph | None, entity_uri: str) -> str:
    """
    Resolve the text used to represent an entity in the n-gram index.

    Prefers an English label. Failing that, falls back to the URI local name,
    except where that local name is an opaque identifier such as a DBpedia
    ``E291085`` or a Wikidata ``Q1108721``, which carries no lexical signal; in
    that case attribute literals stand in when any qualify.

    Parameters
    ----------
    graph : Graph | None
        Graph supplying label and attribute triples for ``entity_uri``.
    entity_uri : str
        URI of the entity to describe.

    Returns
    -------
    str
        Label, attribute text, or URI local name, in that order of preference.
    """
    label = _first_label(graph, entity_uri) if graph is not None else None
    if label:
        return label
    local_name = _local_name(str(entity_uri))
    if _OPAQUE_ID_PATTERN.match(local_name):
        return get_entity_text_from_attributes(graph, str(entity_uri)) or local_name
    return local_name


def _token_map(labels: Dict[str, str], config: NgramConfig) -> Dict[str, List[str]]:
    return _build_tokens(labels, n=config.n, strip_punctuation=config.strip_punctuation)


def _counts_by_uri(token_map: Dict[str, List[str]], uris: List[str]) -> List[Counter]:
    return [Counter(token_map[uri]) for uri in uris]


def _norms(counts: List[Counter]) -> np.ndarray:
    return np.array(
        [np.sqrt(sum(value * value for value in count.values())) for count in counts],
        dtype=np.float64,
    )


def _distinct(counts: List[Counter]) -> np.ndarray:
    return np.array([len(count) for count in counts], dtype=np.float64)


def _matrix_from_counts(
    counts: List[Counter], vocabulary: Dict[str, int], grow: bool
) -> sparse.csr_matrix:
    rows: List[int] = []
    columns: List[int] = []
    values: List[float] = []
    for row, count in enumerate(counts):
        for gram, value in count.items():
            column = vocabulary.setdefault(gram, len(vocabulary)) if grow else vocabulary.get(gram)
            if column is None:
                continue
            rows.append(row)
            columns.append(column)
            values.append(value)
    shape = (len(counts), max(len(vocabulary), 1))
    return sparse.csr_matrix((values, (rows, columns)), shape=shape, dtype=np.float64)


def build_ngram_index(labels: Dict[str, str], config: NgramConfig) -> NgramIndex:
    """
    Build an inverted n-gram index over a target label set.

    Parameters
    ----------
    labels : dict[str, str]
        Mapping from target URI to its label text.
    config : NgramConfig
        n-gram size and normalisation settings.

    Returns
    -------
    NgramIndex
        Index whose ``uris`` are sorted ascending.
    """
    token_map = _token_map(labels, config)
    uris = sorted(token_map)
    counts = _counts_by_uri(token_map, uris)
    vocabulary: Dict[str, int] = {}
    matrix = _matrix_from_counts(counts, vocabulary, grow=True)
    return NgramIndex(uris, vocabulary, matrix, _norms(counts), _distinct(counts))


def _binarised(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    binary = matrix.copy()
    binary.data = np.ones_like(binary.data)
    return binary


def _cosine_block(
    block: sparse.csr_matrix, norms: np.ndarray, index: NgramIndex
) -> np.ndarray:
    products = np.asarray((block @ index.matrix.T).todense(), dtype=np.float64)
    denominator = np.outer(norms, index.norms)
    return np.divide(products, denominator, out=np.zeros_like(products), where=denominator > 0)


def _jaccard_block(
    block: sparse.csr_matrix, distinct: np.ndarray, index: NgramIndex
) -> np.ndarray:
    intersection = np.asarray(
        (_binarised(block) @ _binarised(index.matrix).T).todense(), dtype=np.float64
    )
    union = distinct[:, None] + index.distinct[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _score_block(
    block: sparse.csr_matrix,
    stats: tuple[np.ndarray, np.ndarray],
    index: NgramIndex,
    metric: str,
) -> np.ndarray:
    if metric == "cosine":
        return _cosine_block(block, stats[0], index)
    if metric == "jaccard":
        return _jaccard_block(block, stats[1], index)
    raise ConfigurationError(f"Unsupported candidates.metric: {metric}")


def _top_k_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Return the best ``top_k`` column indices, ties broken on ascending index."""
    positive = np.flatnonzero(scores > 0)
    if positive.size >= top_k:
        best = positive[np.argpartition(-scores[positive], top_k - 1)[:top_k]]
        tied = positive[scores[positive] >= scores[best].min()]
        return tied[np.lexsort((tied, -scores[tied]))][:top_k]

    ordered = positive[np.lexsort((positive, -scores[positive]))]
    chosen = set(ordered.tolist())
    padding = list(
        islice((i for i in range(scores.shape[0]) if i not in chosen), top_k - ordered.size)
    )
    if not padding:
        return ordered
    return np.concatenate([ordered, np.array(padding, dtype=int)])


def _row_pairs(
    source_uri: str, scores: np.ndarray, index: NgramIndex, top_k: int
) -> Iterator[CandidatePair]:
    for rank, column in enumerate(_top_k_indices(scores, top_k), start=1):
        yield CandidatePair(
            source_uri=source_uri,
            target_uri=index.uris[int(column)],
            score=float(scores[column]),
            rank=rank,
        )


def _project_block(counts: List[Counter], index: NgramIndex):
    matrix = _matrix_from_counts(counts, dict(index.vocabulary), grow=False)
    return matrix, (_norms(counts), _distinct(counts))


def generate_candidates_for_pair(
    source_labels: Dict[str, str],
    target: Dict[str, str] | NgramIndex,
    config: NgramConfig,
) -> List[CandidatePair]:
    """
    Rank target labels against every source label via an inverted index.

    Parameters
    ----------
    source_labels : dict[str, str]
        Mapping from source URI to its label text.
    target : dict[str, str] | NgramIndex
        Target labels, or a prebuilt index to reuse across calls.
    config : NgramConfig
        n-gram size, metric, and ``top_k`` settings.

    Returns
    -------
    list[CandidatePair]
        Candidates for every source URI, in ascending source URI order.

    Raises
    ------
    ConfigurationError
        If ``config.top_k`` is not positive or the metric is unsupported.
    """
    if config.top_k <= 0:
        raise ConfigurationError("candidates.top_k must be a positive integer")

    index = target if isinstance(target, NgramIndex) else build_ngram_index(target, config)
    token_map = _token_map(source_labels, config)
    uris = sorted(token_map)

    pairs: List[CandidatePair] = []
    for start in range(0, len(uris), _BLOCK_ROWS):
        block_uris = uris[start : start + _BLOCK_ROWS]
        block, stats = _project_block(_counts_by_uri(token_map, block_uris), index)
        scores = _score_block(block, stats, index, config.metric)
        for offset, source_uri in enumerate(block_uris):
            pairs.extend(_row_pairs(source_uri, scores[offset], index, config.top_k))
    return pairs


def query_ngram_index(
    index: NgramIndex, label: str, config: NgramConfig
) -> List[CandidatePair]:
    """
    Score a single label against an index and return its ranked candidates.

    Parameters
    ----------
    index : NgramIndex
        Index built over the target label set.
    label : str
        Source label text to score.
    config : NgramConfig
        Scoring settings; ``top_k`` bounds the returned list.

    Returns
    -------
    list[CandidatePair]
        Candidates ranked by descending score, ties on ascending target URI.
    """
    return generate_candidates_for_pair({"": label}, index, config)


def _candidates_path(dataset_id: str, pair_name: str, data_dir: str | Path) -> Path:
    return (
        Path(data_dir)
        / _CANDIDATE_ROOT
        / dataset_id
        / f"{pair_name}_candidates.json"
    )


def dump_candidates(candidates: CandidateMap, output_path: str | Path) -> None:
    """
    Write candidate target URIs to the JSON format :func:`load_candidates` reads.

    Public counterpart to :func:`load_candidates` and the canonical write path
    for candidate files. Missing parent directories are created.

    Parameters
    ----------
    candidates : CandidateMap
        Mapping from each source URI to its ordered candidate target URIs.
    output_path : str | Path
        Destination file, conventionally
        ``{data_dir}/candidates/{dataset_id}/{pair_name}_candidates.json``.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "candidates": {
            str(source): [str(target) for target in targets]
            for source, targets in candidates.items()
        }
    }
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def _parse_candidate_map(payload: dict) -> CandidateMap:
    candidates = payload.get("candidates")
    if not isinstance(candidates, dict):
        raise DataError("Candidate file must contain a 'candidates' object.")
    return {str(source): [str(target) for target in targets] for source, targets in candidates.items()}


def load_candidates(
    dataset_id: str, pair_name: str, data_dir: str | Path = "data/"
) -> CandidateMap:
    """
    Load pre-generated candidate target URIs for a single alignment pair.

    Parameters
    ----------
    dataset_id : str
        OAEI track identifier, e.g. ``"D1"``.
    pair_name : str
        Ontology pair name, e.g. ``"d1_snomed_fma"``.
    data_dir : str | Path
        Dataset root directory (default ``"data/"``).

    Returns
    -------
    CandidateMap
        Mapping from each source URI to its ordered candidate target URIs.

    Raises
    ------
    DataError
        If the candidate file is missing or is not valid candidate JSON.
    """
    path = _candidates_path(dataset_id, pair_name, data_dir)
    if not path.exists():
        raise DataError(f"Candidate file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataError(f"Malformed candidate JSON at {path}: {exc}") from exc
    return _parse_candidate_map(payload)
