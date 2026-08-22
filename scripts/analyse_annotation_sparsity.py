"""Quantify annotation sparsity across the D3 Conference ontologies.

Tests whether annotation sparsity predicts the V8 advantage over V6 by
correlating each alignment pair's minimum ontology annotation density with the
C10 (V2+V8/M2) minus C3 (V2+V6/M2) F1 difference.

The analysis is read-only: dataset graphs and result JSON files are consumed
but never written back. Pairs missing a C10 or C3 result are skipped with a
warning rather than aborting the run.

Usage
-----
python scripts/analyse_annotation_sparsity.py \\
    --data_dir data/ \\
    --results_dir data/results/ \\
    --output data/analysis/annotation_sparsity.md
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median

import numpy as np
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS
from scipy.stats import spearmanr

from kgsemembed.datasets.loader import AlignmentPair, load_dataset
from kgsemembed.verbalisation.base import LABEL_PREDICATES

_LOGGER = logging.getLogger("kgsemembed.scripts.analyse_annotation_sparsity")

_DATASET_ID = "D3"
_V8_CONDITION = "C10"
_V6_CONDITION = "C3"

_ANNOTATION_PREDICATES: tuple[URIRef, ...] = tuple(LABEL_PREDICATES) + (
    RDFS.comment,
    SKOS.definition,
    OWL.equivalentClass,
    OWL.equivalentProperty,
)
_LABEL_PREDICATES: tuple[URIRef, ...] = (RDFS.label, SKOS.prefLabel)
_ENTITY_TYPES: tuple[URIRef, ...] = (OWL.Class, OWL.ObjectProperty)
_HIERARCHY_PREDICATES: tuple[URIRef, ...] = (RDFS.subClassOf, RDFS.subPropertyOf)


@dataclass(frozen=True)
class OntologyStats:
    """
    Annotation coverage summary for a single ontology.

    Attributes
    ----------
    ontology_id : str
        Short ontology identifier, e.g. ``"cmt"``.
    n_entities : int
        Number of class and object-property entities considered.
    density : float
        Mean number of the seven annotation predicates present per entity.
    label_coverage : float
        Fraction of entities carrying an ``rdfs:label`` or ``skos:prefLabel``.
    """

    ontology_id: str
    n_entities: int
    density: float
    label_coverage: float


@dataclass(frozen=True)
class PairRow:
    """
    One D3 alignment pair joined to its ontology density and F1 scores.

    Attributes
    ----------
    pair_name : str
        Pair identifier, e.g. ``"cmt-edas"``.
    source : OntologyStats
        Density statistics of the source ontology.
    target : OntologyStats
        Density statistics of the target ontology.
    c10_f1 : float
        Test F1 of condition C10 (V2+V8/M2).
    c3_f1 : float
        Test F1 of condition C3 (V2+V6/M2).
    """

    pair_name: str
    source: OntologyStats
    target: OntologyStats
    c10_f1: float
    c3_f1: float

    @property
    def min_density(self) -> float:
        """Return the sparser of the two ontology densities."""
        return min(self.source.density, self.target.density)

    @property
    def delta_f1(self) -> float:
        """Return the C10 minus C3 F1 difference."""
        return self.c10_f1 - self.c3_f1

    @property
    def v8_wins(self) -> bool:
        """Return True when V8 outperforms V6 on this pair."""
        return self.delta_f1 > 0


@dataclass(frozen=True)
class SparsityCorrelation:
    """
    Outcome of the sparsity hypothesis test over the analysed pairs.

    Attributes
    ----------
    n_pairs : int
        Number of pairs with both C10 and C3 results.
    rho : float
        Spearman rank correlation between ``min_density`` and ``delta_f1``.
    p_value : float
        Two-sided p-value of the correlation.
    median_min_density : float
        Median ``min_density`` used to split sparse from annotation-rich pairs.
    sparse_mean_delta : float
        Mean ``delta_f1`` over pairs below the median density.
    rich_mean_delta : float
        Mean ``delta_f1`` over pairs at or above the median density.
    """

    n_pairs: int
    rho: float
    p_value: float
    median_min_density: float
    sparse_mean_delta: float
    rich_mean_delta: float


def collect_ontology_graphs(pairs: list[AlignmentPair]) -> dict[str, Graph]:
    """
    Deduplicate the ontology graphs reached through every alignment pair.

    Each pair parses both of its ontologies independently, so a given ontology
    appears in several pairs; only the first graph seen per identifier is kept.

    Parameters
    ----------
    pairs : list[AlignmentPair]
        Alignment pairs of a single dataset.

    Returns
    -------
    dict[str, Graph]
        One graph per ontology identifier, keyed by short ontology id.
    """
    graphs: dict[str, Graph] = {}
    for pair in pairs:
        graphs.setdefault(pair.source_id, pair.source_graph)
        graphs.setdefault(pair.target_id, pair.target_graph)
    return graphs


def _annotatable_entities(graph: Graph) -> list[URIRef]:
    """
    Return the class and object-property URIs whose annotations are counted.

    Parameters
    ----------
    graph : Graph
        Ontology graph to inspect.

    Returns
    -------
    list[URIRef]
        Sorted, deduplicated entity URIs.
    """
    typed = {
        subject
        for entity_type in _ENTITY_TYPES
        for subject in graph.subjects(RDF.type, entity_type)
        if isinstance(subject, URIRef)
    }
    hierarchical = {
        subject
        for predicate in _HIERARCHY_PREDICATES
        for subject in graph.subjects(predicate, None)
        if isinstance(subject, URIRef)
    }
    return sorted(typed | hierarchical)


def _annotation_count(graph: Graph, entity: URIRef) -> int:
    """
    Count how many of the seven annotation predicates ``entity`` carries.

    Parameters
    ----------
    graph : Graph
        Ontology graph holding the entity triples.
    entity : URIRef
        Entity URI to score.

    Returns
    -------
    int
        Number of distinct annotation predicates present, in ``[0, 7]``.
    """
    return sum(
        1
        for predicate in _ANNOTATION_PREDICATES
        if (entity, predicate, None) in graph
    )


def _has_label(graph: Graph, entity: URIRef) -> bool:
    """
    Return True when ``entity`` carries an ``rdfs:label`` or ``skos:prefLabel``.

    Parameters
    ----------
    graph : Graph
        Ontology graph holding the entity triples.
    entity : URIRef
        Entity URI to check.

    Returns
    -------
    bool
        True if at least one label triple exists.
    """
    return any((entity, predicate, None) in graph for predicate in _LABEL_PREDICATES)


def compute_ontology_stats(ontology_id: str, graph: Graph) -> OntologyStats:
    """
    Compute annotation density and label coverage for one ontology.

    Parameters
    ----------
    ontology_id : str
        Short ontology identifier, e.g. ``"cmt"``.
    graph : Graph
        Parsed ontology graph.

    Returns
    -------
    OntologyStats
        Entity count, mean annotation density, and label coverage.
    """
    entities = _annotatable_entities(graph)
    if not entities:
        _LOGGER.warning("Ontology %s exposes no class or property entities", ontology_id)
        return OntologyStats(ontology_id, 0, 0.0, 0.0)

    density = fmean(_annotation_count(graph, entity) for entity in entities)
    coverage = fmean(float(_has_label(graph, entity)) for entity in entities)
    return OntologyStats(ontology_id, len(entities), density, coverage)


def load_condition_f1(results_dir: Path, condition_id: str) -> dict[str, float]:
    """
    Read every D3 result JSON of one condition into per-pair F1 scores.

    Parameters
    ----------
    results_dir : Path
        Root results directory containing ``<condition>/<dataset>`` folders.
    condition_id : str
        Ablation condition identifier, e.g. ``"C10"``.

    Returns
    -------
    dict[str, float]
        Test F1 keyed by pair name; empty when the condition has no results.
    """
    condition_dir = results_dir / condition_id / _DATASET_ID
    if not condition_dir.is_dir():
        _LOGGER.warning("No %s results directory at %s", condition_id, condition_dir)
        return {}

    scores: dict[str, float] = {}
    for path in sorted(condition_dir.glob("*_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        pair_name = payload.get("pair_name", path.stem.removesuffix("_results"))
        scores[pair_name] = float(payload["metrics"]["f1"])
    return scores


def _pair_row(
    pair: AlignmentPair,
    stats: dict[str, OntologyStats],
    scores: tuple[dict[str, float], dict[str, float]],
) -> PairRow | None:
    """
    Join one alignment pair to its ontology stats and both condition scores.

    Parameters
    ----------
    pair : AlignmentPair
        Alignment pair to describe.
    stats : dict[str, OntologyStats]
        Density statistics keyed by ontology identifier.
    scores : tuple[dict[str, float], dict[str, float]]
        C10 and C3 F1 scores keyed by pair name.

    Returns
    -------
    PairRow | None
        Joined row, or None when either condition lacks a result.
    """
    c10_scores, c3_scores = scores
    if pair.pair_name not in c10_scores or pair.pair_name not in c3_scores:
        _LOGGER.warning("Skipping %s: missing C10 or C3 result", pair.pair_name)
        return None
    return PairRow(
        pair_name=pair.pair_name,
        source=stats[pair.source_id],
        target=stats[pair.target_id],
        c10_f1=c10_scores[pair.pair_name],
        c3_f1=c3_scores[pair.pair_name],
    )


def build_pair_rows(
    pairs: list[AlignmentPair],
    stats: dict[str, OntologyStats],
    scores: tuple[dict[str, float], dict[str, float]],
) -> list[PairRow]:
    """
    Build the pair-level analysis rows, sorted from sparsest pair upwards.

    Parameters
    ----------
    pairs : list[AlignmentPair]
        Alignment pairs of the D3 track.
    stats : dict[str, OntologyStats]
        Density statistics keyed by ontology identifier.
    scores : tuple[dict[str, float], dict[str, float]]
        C10 and C3 F1 scores keyed by pair name.

    Returns
    -------
    list[PairRow]
        Rows with both condition results, ascending by ``min_density``.
    """
    rows = [_pair_row(pair, stats, scores) for pair in pairs]
    return sorted(
        (row for row in rows if row is not None),
        key=lambda row: (row.min_density, row.pair_name),
    )


def compute_correlation(rows: list[PairRow]) -> SparsityCorrelation:
    """
    Correlate pair sparsity with the V8 minus V6 F1 difference.

    Parameters
    ----------
    rows : list[PairRow]
        Pair-level analysis rows.

    Returns
    -------
    SparsityCorrelation
        Spearman statistics and the sparse versus annotation-rich group means.
        Correlation fields are NaN when fewer than three pairs are available.
    """
    densities = [row.min_density for row in rows]
    deltas = [row.delta_f1 for row in rows]
    if len(rows) < 3:
        _LOGGER.warning("Only %d pairs available; correlation not computed", len(rows))
        return SparsityCorrelation(len(rows), float("nan"), float("nan"), float("nan"),
                                   float("nan"), float("nan"))

    result = spearmanr(densities, deltas)
    cut = median(densities)
    sparse = [row.delta_f1 for row in rows if row.min_density < cut]
    rich = [row.delta_f1 for row in rows if row.min_density >= cut]
    return SparsityCorrelation(
        n_pairs=len(rows),
        rho=float(result.statistic),
        p_value=float(result.pvalue),
        median_min_density=cut,
        sparse_mean_delta=fmean(sparse) if sparse else float("nan"),
        rich_mean_delta=fmean(rich) if rich else float("nan"),
    )


def _render_table(header: list[str], rows: list[list[str]]) -> str:
    """
    Render a Markdown table from a header and pre-formatted cell rows.

    Parameters
    ----------
    header : list[str]
        Column titles.
    rows : list[list[str]]
        Cell values, one list per row.

    Returns
    -------
    str
        Markdown table text without a trailing newline.
    """
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _render_density_table(stats: dict[str, OntologyStats]) -> str:
    """
    Render the per-ontology annotation density table, sparsest first.

    Parameters
    ----------
    stats : dict[str, OntologyStats]
        Density statistics keyed by ontology identifier.

    Returns
    -------
    str
        Markdown table text.
    """
    ordered = sorted(stats.values(), key=lambda item: (item.density, item.ontology_id))
    rows = [
        [item.ontology_id, str(item.n_entities), f"{item.density:.3f}", f"{item.label_coverage:.3f}"]
        for item in ordered
    ]
    return _render_table(["Ontology", "N entities", "Density (0-7)", "Label coverage"], rows)


def _render_pair_table(rows: list[PairRow]) -> str:
    """
    Render the pair-level analysis table, sparsest pair first.

    Parameters
    ----------
    rows : list[PairRow]
        Pair-level analysis rows.

    Returns
    -------
    str
        Markdown table text.
    """
    cells = [
        [
            row.pair_name,
            f"{row.min_density:.3f}",
            f"{row.c10_f1:.4f}",
            f"{row.c3_f1:.4f}",
            f"{row.delta_f1:+.4f}",
            "yes" if row.v8_wins else "no",
        ]
        for row in rows
    ]
    header = ["Pair", "Min density", "C10 F1", "C3 F1", "Delta F1", "V8 wins"]
    return _render_table(header, cells)


def _interpretation(correlation: SparsityCorrelation) -> str:
    """
    Describe in one sentence what the observed correlation implies.

    Parameters
    ----------
    correlation : SparsityCorrelation
        Computed Spearman statistics.

    Returns
    -------
    str
        Interpretation sentence.
    """
    if np.isnan(correlation.rho):
        return "Too few pairs with both C10 and C3 results to test the hypothesis."
    if correlation.p_value >= 0.05:
        return (
            "The correlation is not significant at alpha = 0.05, so annotation "
            "sparsity does not reliably predict the V8 advantage on D3."
        )
    direction = "supports" if correlation.rho < 0 else "contradicts"
    return (
        f"The correlation is significant and {direction} the hypothesis that V8 "
        "gains most where annotation coverage is sparsest."
    )


def _render_correlation(correlation: SparsityCorrelation) -> str:
    """
    Render the correlation summary section body.

    Parameters
    ----------
    correlation : SparsityCorrelation
        Computed Spearman statistics.

    Returns
    -------
    str
        Markdown paragraph text.
    """
    return "\n".join(
        [
            f"- Pairs analysed: {correlation.n_pairs}",
            f"- Spearman r: {correlation.rho:.4f}",
            f"- p-value: {correlation.p_value:.4f}",
            f"- Median min density: {correlation.median_min_density:.3f}",
            f"- Mean delta F1, sparse pairs (below median): {correlation.sparse_mean_delta:+.4f}",
            f"- Mean delta F1, annotation-rich pairs (at or above median): "
            f"{correlation.rich_mean_delta:+.4f}",
            "",
            _interpretation(correlation),
        ]
    )


def _finding_statement(correlation: SparsityCorrelation, rows: list[PairRow]) -> str:
    """
    Draft the paper-ready finding paragraph citing the Spearman correlation.

    Parameters
    ----------
    correlation : SparsityCorrelation
        Computed Spearman statistics.
    rows : list[PairRow]
        Pair-level analysis rows.

    Returns
    -------
    str
        One-paragraph finding statement.
    """
    wins = sum(1 for row in rows if row.v8_wins)
    verdict = "is consistent with" if correlation.rho < 0 else "does not support"
    return (
        f"Across the {correlation.n_pairs} Conference alignment pairs for which both "
        f"conditions completed, the annotation-independent relational signature of V8 "
        f"(C10) outperformed the structured key-value verbalisation of V6 (C3) on "
        f"{wins} pairs. Ranking each pair by the annotation density of its sparser "
        f"ontology, measured as the mean number of seven annotation predicates present "
        f"per class or object property, yields a Spearman correlation of r = "
        f"{correlation.rho:.3f} (p = {correlation.p_value:.3f}) against the per-pair F1 "
        f"difference. Pairs below the median density gain {correlation.sparse_mean_delta:+.3f} "
        f"F1 on average from V8, against {correlation.rich_mean_delta:+.3f} for pairs at or "
        f"above the median. This {verdict} the hypothesis that V8's relational signature "
        f"compensates for missing lexical annotation."
    )


def render_report(
    stats: dict[str, OntologyStats],
    rows: list[PairRow],
    correlation: SparsityCorrelation,
) -> str:
    """
    Assemble the full Markdown analysis report.

    Parameters
    ----------
    stats : dict[str, OntologyStats]
        Density statistics keyed by ontology identifier.
    rows : list[PairRow]
        Pair-level analysis rows.
    correlation : SparsityCorrelation
        Computed Spearman statistics.

    Returns
    -------
    str
        Complete Markdown document.
    """
    sections = [
        "# Annotation Sparsity and the V8 Advantage on D3",
        "## 1. Ontology Annotation Density",
        _render_density_table(stats),
        "## 2. Pair-Level Analysis (C10 vs C3)",
        _render_pair_table(rows),
        "## 3. Correlation Summary",
        _render_correlation(correlation),
        "## 4. Finding Statement",
        _finding_statement(correlation, rows),
        "",
    ]
    return "\n\n".join(sections)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the sparsity analysis."""
    parser = argparse.ArgumentParser(
        description="Correlate D3 annotation sparsity with the V8 over V6 F1 advantage."
    )
    parser.add_argument("--data_dir", default="data/", help="Dataset root directory.")
    parser.add_argument(
        "--results_dir", default="data/results/", help="Root directory of result JSON files."
    )
    parser.add_argument(
        "--output",
        default="data/analysis/annotation_sparsity.md",
        help="Destination Markdown analysis path.",
    )
    return parser


def main() -> None:
    """Run the annotation sparsity analysis and write the Markdown report."""
    random.seed(42)
    np.random.seed(42)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _build_arg_parser().parse_args()

    pairs = load_dataset(_DATASET_ID, args.data_dir)
    stats = {
        ontology_id: compute_ontology_stats(ontology_id, graph)
        for ontology_id, graph in collect_ontology_graphs(pairs).items()
    }
    results_dir = Path(args.results_dir)
    scores = (
        load_condition_f1(results_dir, _V8_CONDITION),
        load_condition_f1(results_dir, _V6_CONDITION),
    )
    rows = build_pair_rows(pairs, stats, scores)
    correlation = compute_correlation(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(stats, rows, correlation), encoding="utf-8")

    print(_render_correlation(correlation))
    print()
    print(_finding_statement(correlation, rows))
    print(f"\nAnalysis written to {output_path}")


if __name__ == "__main__":
    main()
