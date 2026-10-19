"""Describe the reference alignments of the five datasets as the pipeline uses them (#138).

Two levels, both read-only:

* **Reference files**, parsed directly from the EDOAL XML or OpenEA link file
  rather than through the loader: cells, relation types, ``measure`` values
  below 1, malformed cells, duplicate equivalence pairs, and the pairs and
  sources two files of one dataset share (D1 ``input.rdf``/``reference.rdf``,
  D5's splits and ``ent_links``).
* **Loaded pairs**, through :func:`load_dataset` exactly as every experiment
  loads them: reference cardinality, the entity kind of each reference end,
  whether it lies in the stage-1 source set and target pool, the validation
  and test populations against the stored top-20 candidate lists, and the
  known pairs.

Nothing is written under ``data/``.

Usage
-----
python scripts/reference_statistics.py --data_dir data/ \\
    --output artifacts/issue-138/reference_statistics.md
"""

from __future__ import annotations

import argparse
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdflib import RDF, Graph, URIRef

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kgsemembed.candidates.ngram import load_candidates  # noqa: E402
from kgsemembed.datasets import AlignmentPair, load_dataset  # noqa: E402
from kgsemembed.datasets.loader import _resolve_entity_type, extract_entity_uris  # noqa: E402
from kgsemembed.evaluation.population import build_populations, without_known_pairs  # noqa: E402
from scripts.dataset_provenance import TrackedFile, tracked_files  # noqa: E402
from scripts.dataset_provenance import _table as _provenance_table  # noqa: E402

_ALIGNMENT_NS = "{http://knowledgeweb.semanticweb.org/heterogeneity/alignment}"
_RDF_RESOURCE = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
_ALL_DATASETS = ("D1", "D2", "D3", "D4", "D5")
_D5_SPLITS = ("train_links", "valid_links", "test_links")

EntityPair = Tuple[str, str]
CandidateMap = Dict[str, List[str]]
Cell = Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]


@dataclass(frozen=True)
class FileAudit:
    """
    What one reference file contains before the loader filters it.

    Attributes
    ----------
    local : str
        Path under the dataset root.
    cells : int
        EDOAL ``Cell`` elements, or non-empty lines of a link file.
    relations : Dict[str, int]
        Cells per ``relation`` value (``"="`` for every link-file line).
    partial_measure : int
        Cells whose ``measure`` is below 1.
    malformed : int
        Cells without a relation, ``entity1`` or ``entity2`` resource.
    equivalences : Tuple[EntityPair, ...]
        Every equivalence pair in file order, duplicates included.
    """

    local: str
    cells: int
    relations: Dict[str, int]
    partial_measure: int
    malformed: int
    equivalences: Tuple[EntityPair, ...]


@dataclass(frozen=True)
class PopulationAudit:
    """
    One evaluation population of one pair, against its stored candidates.

    Attributes
    ----------
    refs : int
        Reference pairs scored in the population.
    ref_sources : int
        Distinct reference sources.
    gold_per_source : Dict[str, int]
        Reference sources with 1, 2, or 3+ gold targets.
    unmatched : int
        Sources without any reference drawn into the population.
    unranked : int
        Reference sources without a non-empty candidate list.
    gold_listed : int
        Reference sources with at least one gold target among their candidates.
    refs_listed : int
        Reference pairs whose target is among the source's candidates.
    """

    refs: int
    ref_sources: int
    gold_per_source: Dict[str, int]
    unmatched: int
    unranked: int
    gold_listed: int
    refs_listed: int


@dataclass(frozen=True)
class KnownAudit:
    """
    Known (training) pairs of one pair and their effect on the populations.

    Attributes
    ----------
    pairs : int
        Known pairs.
    sources : int
        Distinct known-pair sources.
    excluded : int
        Known-pair sources with no validation or test reference.
    kept_val : int
        Known-pair sources scored on validation.
    kept_test : int
        Known-pair sources scored on test.
    removed : int
        Known pairs removed from the candidate lists of scored sources.
    """

    pairs: int
    sources: int
    excluded: int
    kept_val: int
    kept_test: int
    removed: int


@dataclass(frozen=True)
class PairAudit:
    """
    Reference structure and evaluation populations of one loaded pair.

    Attributes
    ----------
    dataset_id, pair_name : str
        Pair identity as written in results.
    source_entities, target_pool : int
        Sources ranked by stage 1 and the targets it ranks them against.
    list_lengths : Tuple[int, int]
        Shortest and longest stored candidate list.
    refs : Tuple[EntityPair, ...]
        Validation plus test references, in loader order.
    kinds : Dict[str, int]
        Reference pairs per ``source kind -> target kind``.
    outside_sources, outside_targets : int
        References whose source is not a stage-1 source, or whose target is
        not in the target pool.
    populations : Dict[str, PopulationAudit]
        ``"validation"`` and ``"test"``.
    known : Optional[KnownAudit]
        ``None`` when the pair has no known pairs.
    """

    dataset_id: str
    pair_name: str
    source_entities: int
    target_pool: int
    list_lengths: Tuple[int, int]
    refs: Tuple[EntityPair, ...]
    kinds: Dict[str, int]
    outside_sources: int
    outside_targets: int
    populations: Dict[str, PopulationAudit]
    known: Optional[KnownAudit]


# ---------------------------------------------------------------------------
# Reference files
# ---------------------------------------------------------------------------


def _resource(cell: ET.Element, tag: str) -> Optional[str]:
    element = cell.find(f"{_ALIGNMENT_NS}{tag}")
    return element.get(_RDF_RESOURCE) if element is not None else None


def _text(cell: ET.Element, tag: str) -> Optional[str]:
    element = cell.find(f"{_ALIGNMENT_NS}{tag}")
    return (element.text or "").strip() if element is not None else None


def _is_partial(measure: Optional[str]) -> bool:
    return bool(measure) and float(measure) < 1.0


def alignment_cells(data: bytes) -> List[Cell]:
    """
    Read every cell of an EDOAL alignment in document order, unfiltered.

    Parameters
    ----------
    data : bytes
        Alignment file content.

    Returns
    -------
    List[Cell]
        ``(entity1, entity2, relation, measure)`` per cell; a missing element
        or resource is ``None``.
    """
    cells = ET.fromstring(data).iter(f"{_ALIGNMENT_NS}Cell")
    return [
        (_resource(c, "entity1"), _resource(c, "entity2"), _text(c, "relation"), _text(c, "measure"))
        for c in cells
    ]


def audit_edoal(path: Path, local: str) -> FileAudit:
    """
    Count the cells of an EDOAL alignment file without filtering them.

    Parameters
    ----------
    path : Path
        Alignment file.
    local : str
        Path recorded in the report.

    Returns
    -------
    FileAudit
        Raw cell counts and every equivalence pair.
    """
    cells = alignment_cells(path.read_bytes())
    relations = Counter(relation or "" for _, _, relation, _ in cells)
    malformed = sum(1 for s, t, relation, _ in cells if not (s and t and relation))
    equivalences = tuple((s, t) for s, t, relation, _ in cells if s and t and relation == "=")
    partial = sum(1 for *_, measure in cells if _is_partial(measure))
    return FileAudit(local, len(cells), dict(relations), partial, malformed, equivalences)


def audit_links(path: Path, local: str) -> FileAudit:
    """
    Count the lines of an OpenEA link file.

    Parameters
    ----------
    path : Path
        Tab-separated link file.
    local : str
        Path recorded in the report.

    Returns
    -------
    FileAudit
        One equivalence per well-formed line; others are malformed.
    """
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs = tuple((row[0], row[1]) for row in rows if len(row) >= 2)
    return FileAudit(local, len(rows), {"=": len(pairs)}, 0, len(rows) - len(pairs), pairs)


def audit_reference_files(data_dir: Path, datasets: Sequence[str]) -> List[Tuple[str, FileAudit]]:
    """
    Audit every reference file the loaders read for ``datasets``.

    Parameters
    ----------
    data_dir : Path
        Dataset root.
    datasets : Sequence[str]
        Dataset identifiers, ``"D1"`` to ``"D5"``.

    Returns
    -------
    List[Tuple[str, FileAudit]]
        ``(dataset_id, audit)`` in the order of
        :func:`scripts.dataset_provenance.tracked_files`.
    """
    files = [f for f in tracked_files(data_dir) if f.is_reference and f.dataset_id in datasets]
    return [(f.dataset_id, _audit_file(data_dir, f)) for f in files]


def _audit_file(data_dir: Path, item: TrackedFile) -> FileAudit:
    path = data_dir / item.local
    return audit_edoal(path, item.local) if path.suffix == ".rdf" else audit_links(path, item.local)


def file_overlaps(audits: Sequence[FileAudit]) -> List[Tuple[str, str, int, int]]:
    """
    Count the equivalence pairs and sources every two files share.

    Parameters
    ----------
    audits : Sequence[FileAudit]
        Reference files of one dataset (one D3 pair's single file gives none).

    Returns
    -------
    List[Tuple[str, str, int, int]]
        ``(file_a, file_b, shared_pairs, shared_sources)`` per file pair.
    """
    rows = []
    for a, b in combinations(audits, 2):
        pairs_a, pairs_b = set(a.equivalences), set(b.equivalences)
        sources_a, sources_b = {s for s, _ in pairs_a}, {s for s, _ in pairs_b}
        rows.append((a.local, b.local, len(pairs_a & pairs_b), len(sources_a & sources_b)))
    return rows


def d5_split_union(audits: Sequence[FileAudit]) -> Optional[Tuple[int, int, bool]]:
    """
    Compare the union of D5's three split files with ``ent_links``.

    Parameters
    ----------
    audits : Sequence[FileAudit]
        D5's reference-file audits.

    Returns
    -------
    Optional[Tuple[int, int, bool]]
        ``(union size, ent_links size, equal)``, or ``None`` when a file is
        missing from ``audits``.
    """
    by_name = {Path(a.local).name: set(a.equivalences) for a in audits}
    if "ent_links" not in by_name or not all(name in by_name for name in _D5_SPLITS):
        return None
    union = set().union(*(by_name[name] for name in _D5_SPLITS))
    return len(union), len(by_name["ent_links"]), union == by_name["ent_links"]


# ---------------------------------------------------------------------------
# Loaded pairs
# ---------------------------------------------------------------------------


def entity_kind(graph: Graph, uri: str) -> str:
    """
    Classify a reference end the way the D4 partition does, plus two checks.

    Parameters
    ----------
    graph : Graph
        Graph the URI should be described in.
    uri : str
        Reference end.

    Returns
    -------
    str
        ``"absent"`` when the URI is the subject of no triple, ``"untyped"``
        when it has triples but no ``rdf:type`` (the loader treats it as an
        instance), else the loader's ``"class"``, ``"predicate"`` or
        ``"instance"``.
    """
    node = URIRef(uri)
    if (node, None, None) not in graph:
        return "absent"
    if (node, RDF.type, None) not in graph:
        return "untyped"
    return _resolve_entity_type(graph, uri)


def cardinality(refs: Iterable[EntityPair]) -> Dict[str, int]:
    """
    Classify every distinct reference pair by the degree of its two ends.

    Parameters
    ----------
    refs : Iterable[EntityPair]
        Reference pairs; duplicates are counted once.

    Returns
    -------
    Dict[str, int]
        Pairs per class: ``"1:1"``, ``"1:N"`` (source has several targets),
        ``"N:1"`` (target has several sources), ``"N:M"`` (both).
    """
    distinct = set(refs)
    out_degree = Counter(source for source, _ in distinct)
    in_degree = Counter(target for _, target in distinct)
    labels = {(False, False): "1:1", (True, False): "1:N", (False, True): "N:1", (True, True): "N:M"}
    counts = Counter(labels[(out_degree[s] > 1, in_degree[t] > 1)] for s, t in distinct)
    return {label: counts.get(label, 0) for label in labels.values()}


def _gold_bucket(n_targets: int) -> str:
    return str(n_targets) if n_targets < 3 else "3+"


def audit_population(
    refs: Sequence[EntityPair], sources: FrozenSet[str], candidates: CandidateMap
) -> PopulationAudit:
    """
    Describe one population's reference sources against their candidate lists.

    Parameters
    ----------
    refs : Sequence[EntityPair]
        References of the population's slice.
    sources : FrozenSet[str]
        The population's sources (reference and unmatched).
    candidates : CandidateMap
        Candidate lists with known pairs already removed.

    Returns
    -------
    PopulationAudit
        Counts for the population.
    """
    gold: Dict[str, set] = {}
    for source, target in refs:
        gold.setdefault(source, set()).add(target)
    listed = {s: set(candidates.get(s, [])) for s in gold}
    return PopulationAudit(
        refs=len(refs),
        ref_sources=len(gold),
        gold_per_source=dict(Counter(_gold_bucket(len(t)) for t in gold.values())),
        unmatched=len(sources - set(gold)),
        unranked=sum(1 for s in gold if not listed[s]),
        gold_listed=sum(1 for s, targets in gold.items() if targets & listed[s]),
        refs_listed=sum(1 for s, t in refs if t in listed[s]),
    )


def audit_known(pair: AlignmentPair, raw: CandidateMap, scored: FrozenSet[str]) -> Optional[KnownAudit]:
    """
    Describe a pair's known pairs and how the populations treat them.

    Parameters
    ----------
    pair : AlignmentPair
        Pair supplying ``train_refs`` and the validation and test references.
    raw : CandidateMap
        Stored candidate lists, known pairs included.
    scored : FrozenSet[str]
        Sources of the validation and test populations.

    Returns
    -------
    Optional[KnownAudit]
        ``None`` when the pair has no known pairs.
    """
    if not pair.train_refs:
        return None
    sources = {s for s, _ in pair.train_refs}
    val = {s for s, _ in pair.val_refs}
    test = {s for s, _ in pair.test_refs}
    known = set(pair.train_refs)
    removed = sum(1 for s in scored for t in raw.get(s, []) if (s, t) in known)
    return KnownAudit(
        pairs=len(pair.train_refs), sources=len(sources), excluded=len(sources - val - test),
        kept_val=len(sources & (val - test)), kept_test=len(sources & test), removed=removed,
    )


def _filtered_candidates(pair: AlignmentPair, raw: CandidateMap) -> CandidateMap:
    ranked = [[(source, target, 0.0) for target in targets] for source, targets in raw.items()]
    kept = without_known_pairs(pair, ranked)
    return {source: [t for _, t, _ in rows] for source, rows in zip(raw, kept)}


def _kinds(pair: AlignmentPair, refs: Sequence[EntityPair]) -> Dict[str, int]:
    counts = Counter(
        f"{entity_kind(pair.source_graph, s)} -> {entity_kind(pair.target_graph, t)}" for s, t in refs
    )
    return dict(sorted(counts.items()))


def audit_pair(pair: AlignmentPair, raw: CandidateMap) -> PairAudit:
    """
    Describe one loaded pair against its stored candidate lists.

    Parameters
    ----------
    pair : AlignmentPair
        Pair as :func:`load_dataset` returns it.
    raw : CandidateMap
        Stored candidate lists for the pair.

    Returns
    -------
    PairAudit
        Reference structure, populations and known pairs.
    """
    refs = tuple(pair.val_refs + pair.test_refs)
    pool = set(extract_entity_uris(pair.target_graph, pair.entity_type))
    sources = set(pair.source_entities)
    populations = build_populations(pair)
    candidates = _filtered_candidates(pair, raw)
    lengths = [len(targets) for targets in raw.values()] or [0]
    return PairAudit(
        dataset_id=pair.dataset_id, pair_name=pair.pair_name,
        source_entities=len(sources), target_pool=len(pool), list_lengths=(min(lengths), max(lengths)),
        refs=refs, kinds=_kinds(pair, refs),
        outside_sources=sum(1 for s, _ in refs if s not in sources),
        outside_targets=sum(1 for _, t in refs if t not in pool),
        populations={
            "validation": audit_population(pair.val_refs, populations.val_sources, candidates),
            "test": audit_population(pair.test_refs, populations.test_sources, candidates),
        },
        known=audit_known(pair, raw, populations.val_sources | populations.test_sources),
    )


def audit_dataset(dataset_id: str, data_dir: Path) -> List[PairAudit]:
    """
    Load one dataset and audit each of its pairs.

    Parameters
    ----------
    dataset_id : str
        ``"D1"`` to ``"D5"``; ``"D4"`` yields both D4 pairs.
    data_dir : Path
        Dataset root holding ``candidates/``.

    Returns
    -------
    List[PairAudit]
        One audit per pair, in loader order.
    """
    return [
        audit_pair(pair, load_candidates(pair.dataset_id, pair.pair_name, data_dir))
        for pair in load_dataset(dataset_id, data_dir)
    ]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _n(value: int) -> str:
    return f"{value:,}"


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    return _provenance_table(header, rows) + [""]


def _with_total(rows: List[Tuple[List[str], List[int]]], label: List[str]) -> List[List[str]]:
    formatted = [keys + [_n(v) for v in values] for keys, values in rows]
    if len(rows) < 2:
        return formatted
    totals = [sum(column) for column in zip(*(values for _, values in rows))]
    return formatted + [label + [_n(v) for v in totals]]


def _relations(relations: Dict[str, int]) -> str:
    return ", ".join(f"`{name}` {_n(count)}" for name, count in sorted(relations.items()))


def _file_row(dataset_id: str, audit: FileAudit) -> List[str]:
    eq = len(audit.equivalences)
    return [
        dataset_id, audit.local, _n(audit.cells), _relations(audit.relations),
        _n(audit.partial_measure), _n(audit.malformed), _n(eq - len(set(audit.equivalences))),
    ]


def file_section(files: List[Tuple[str, FileAudit]]) -> List[str]:
    """
    Render the raw reference-file table and the per-dataset file overlaps.

    Parameters
    ----------
    files : List[Tuple[str, FileAudit]]
        Output of :func:`audit_reference_files`.

    Returns
    -------
    List[str]
        Markdown lines.
    """
    header = ["Dataset", "File", "Cells", "Relations", "measure < 1", "Malformed", "Duplicate `=` pairs"]
    lines = ["## 1. Reference files", "", *_table(header, [_file_row(d, a) for d, a in files])]
    return lines + _overlap_lines(files)


def _overlap_lines(files: List[Tuple[str, FileAudit]]) -> List[str]:
    by_dataset: Dict[str, List[FileAudit]] = {}
    for dataset_id, audit in files:
        if dataset_id in {"D1", "D5"}:
            by_dataset.setdefault(dataset_id, []).append(audit)
    rows = [[d, a, b, _n(p), _n(s)] for d, audits in by_dataset.items() for a, b, p, s in file_overlaps(audits)]
    if not rows:
        return []
    lines = ["Files of one dataset sharing pairs or sources:", ""]
    lines += _table(["Dataset", "File A", "File B", "Shared pairs", "Shared sources"], rows)
    union = d5_split_union(by_dataset.get("D5", []))
    if union is not None:
        lines += [f"D5: train ∪ valid ∪ test = {_n(union[0])} pairs; `ent_links` = {_n(union[1])}; "
                  f"equal: {'yes' if union[2] else 'no'}.", ""]
    return lines


def _structure_row(audit: PairAudit) -> Tuple[List[str], List[int]]:
    distinct = set(audit.refs)
    card = cardinality(audit.refs)
    values = [
        len(audit.refs), len(audit.refs) - len(distinct), len({s for s, _ in distinct}),
        len({t for _, t in distinct}), card["1:1"], card["1:N"], card["N:1"], card["N:M"],
        audit.outside_sources, audit.outside_targets,
    ]
    return [audit.dataset_id, audit.pair_name], values


def structure_section(audits: List[PairAudit]) -> List[str]:
    """
    Render reference structure per pair, with a D3 total.

    Parameters
    ----------
    audits : List[PairAudit]
        Audits of every loaded pair.

    Returns
    -------
    List[str]
        Markdown lines.
    """
    header = [
        "Column", "Pair", "Refs (val+test)", "Duplicates", "Sources", "Targets",
        "1:1", "1:N", "N:1", "N:M", "Source not ranked", "Target not in pool",
    ]
    lines = ["## 2. Reference structure (as loaded, validation + test)", ""]
    return lines + _table(header, _grouped_rows(audits, _structure_row))


def _grouped_rows(audits: List[PairAudit], row_fn) -> List[List[str]]:
    rows: List[List[str]] = []
    for dataset_id in dict.fromkeys(a.dataset_id for a in audits):
        group = [row_fn(a) for a in audits if a.dataset_id == dataset_id]
        rows += _with_total(group, [dataset_id, "**total**"])
    return rows


def kinds_section(audits: List[PairAudit]) -> List[str]:
    """
    Render the entity kinds of both ends of every reference pair.

    Parameters
    ----------
    audits : List[PairAudit]
        Audits of every loaded pair.

    Returns
    -------
    List[str]
        Markdown lines.
    """
    rows = [
        [a.dataset_id, a.pair_name, "; ".join(f"{k} {_n(v)}" for k, v in a.kinds.items())]
        for a in audits
    ]
    return ["## 3. Entity kinds of reference ends (source -> target)", "", *_table(["Column", "Pair", "Kinds"], rows)]


def _population_row(audit: PairAudit, name: str) -> Tuple[List[str], List[int]]:
    pop = audit.populations[name]
    gold = [pop.gold_per_source.get(bucket, 0) for bucket in ("1", "2", "3+")]
    values = [
        pop.refs, pop.ref_sources, *gold, pop.unmatched, pop.unranked,
        pop.gold_listed, pop.ref_sources - pop.gold_listed, pop.refs_listed,
    ]
    return [audit.dataset_id, audit.pair_name], values


def population_section(audits: List[PairAudit]) -> List[str]:
    """
    Render the validation and test populations against the stored candidates.

    Parameters
    ----------
    audits : List[PairAudit]
        Audits of every loaded pair.

    Returns
    -------
    List[str]
        Markdown lines.
    """
    header = [
        "Column", "Pair", "Refs", "Ref sources", "1 gold", "2 gold", "3+ gold", "Unmatched",
        "Unranked ref sources", "Gold in list", "No gold in list", "Refs in list",
    ]
    lines = ["## 4. Evaluation populations against the stored candidate lists", ""]
    for name in ("validation", "test"):
        rows = _grouped_rows(audits, lambda a, n=name: _population_row(a, n))
        lines += [f"### {name.capitalize()}", "", *_table(header, rows)]
    return lines


def known_section(audits: List[PairAudit]) -> List[str]:
    """
    Render the known (training) pairs of every pair that has them.

    Parameters
    ----------
    audits : List[PairAudit]
        Audits of every loaded pair.

    Returns
    -------
    List[str]
        Markdown lines, empty when no pair has known pairs.
    """
    rows = [
        [a.dataset_id, *(_n(v) for v in (k.pairs, k.sources, k.excluded, k.kept_val, k.kept_test, k.removed))]
        for a in audits if (k := a.known) is not None
    ]
    if not rows:
        return []
    header = ["Column", "Known pairs", "Sources", "Excluded sources", "Kept (val)", "Kept (test)", "Removed from lists"]
    return ["## 5. Known (training) pairs", "", *_table(header, rows)]


def search_space_section(audits: List[PairAudit]) -> List[str]:
    """
    Render the stage-1 source set, target pool and candidate-list lengths.

    Parameters
    ----------
    audits : List[PairAudit]
        Audits of every loaded pair.

    Returns
    -------
    List[str]
        Markdown lines.
    """
    rows = [
        [a.dataset_id, a.pair_name, _n(a.source_entities), _n(a.target_pool),
         f"{a.list_lengths[0]}–{a.list_lengths[1]}"]
        for a in audits
    ]
    header = ["Column", "Pair", "Stage-1 sources", "Target pool", "Candidates per source"]
    return ["## 6. Stage-1 search space", "", *_table(header, rows)]


_PREAMBLE = [
    "# Reference statistics",
    "",
    "Generated by `scripts/reference_statistics.py`; definitions:",
    "",
    "- **Reference files** are parsed directly. The loader keeps only `=` cells with both",
    "  entity resources, ignores `measure`, and keeps duplicates as separate references.",
    "- **Refs** are the loaded validation and test references (known pairs excluded);",
    "  D5's training links and D1's `input.rdf` appear only in section 5.",
    "- **Source not ranked**: the reference source is not a stage-1 source of the pair's",
    "  entity type, so it never gets a candidate list. **Target not in pool**: the target",
    "  is not a target of that type, so no candidate list can contain it.",
    "- **Kinds**: `absent` = subject of no triple; `untyped` = no `rdf:type` (treated as an",
    "  instance by the loader); otherwise the loader's class / predicate / instance.",
    "- **Unranked ref sources**: reference sources without a non-empty candidate list",
    "  (MRR 0, all their references missed). **No gold in list** includes them.",
    "- Candidate lists are the stored top-20 files with known pairs removed, as scored.",
    "- **Unmatched**: population sources without any reference (false positives under a",
    "  complete reference; on D4 only judgeable when their top target is a reference target).",
    "",
]


def build_report(files: List[Tuple[str, FileAudit]], audits: List[PairAudit]) -> List[str]:
    """
    Assemble the full report.

    Parameters
    ----------
    files : List[Tuple[str, FileAudit]]
        Raw reference-file audits.
    audits : List[PairAudit]
        Loaded-pair audits.

    Returns
    -------
    List[str]
        Markdown lines.
    """
    return [
        *_PREAMBLE, *file_section(files), *structure_section(audits), *kinds_section(audits),
        *population_section(audits), *known_section(audits), *search_space_section(audits),
    ]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Describe reference alignments and evaluation populations.")
    parser.add_argument("--datasets", nargs="+", default=list(_ALL_DATASETS), choices=_ALL_DATASETS)
    parser.add_argument("--data_dir", default="data/", help="Dataset root directory.")
    parser.add_argument("--output", default="artifacts/issue-138/reference_statistics.md", help="Output path.")
    return parser


def main() -> None:
    """Write the reference statistics report."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    data_dir = Path(args.data_dir)
    files = audit_reference_files(data_dir, args.datasets)
    audits = [audit for dataset_id in args.datasets for audit in audit_dataset(dataset_id, data_dir)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(build_report(files, audits)), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
