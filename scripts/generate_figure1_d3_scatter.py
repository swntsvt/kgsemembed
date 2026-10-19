"""Command-line entry point for Figure 1, the D3 per-pair scatter plot.

Plots a per-pair metric of C3 (V2+V6 / M2) against C10 (V2+V8 / M2) over
the 21 Conference pairs: re-ranking MRR by default, the paper's primary
metric, or ``--metric hits_at_1`` / ``--metric f1``.  Pairs involving CMT or EDAS, the annotation-sparse
ontologies, are highlighted to show that V8's advantage over V6 concentrates
on them.  Points above the diagonal are pairs where C10 wins.

Writes ``fig1_d3_scatter.pdf`` (and a ``.png`` preview) to the output
directory for MRR, ``fig1_d3_scatter_{metric}.pdf`` otherwise, and prints a
per-pair table with win counts.

Usage
-----
python scripts/generate_figure1_d3_scatter.py \\
    --results_dir data/results/ --output_dir figures/ --metric mrr
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from kgsemembed.evaluation.reference_counts import (  # noqa: E402
    DEFAULT_COUNTS_PATH,
    load_reference_counts,
    result_metric,
)

PairScores = Dict[str, Tuple[float, float]]

_DATASET_ID = "D3"
_BASELINE_CONDITION = "C3"
_VARIANT_CONDITION = "C10"
_OUTPUT_STEM = "fig1_d3_scatter"
_PRIMARY_METRIC = "mrr"
_METRIC_LABELS = {"mrr": "MRR", "hits_at_1": "Hits@1", "f1": "F1"}
_SPARSE_COLOUR = "#e05c36"
_RICH_COLOUR = "#5b7fa6"
SPARSE_PAIRS = frozenset({
    "cmt-conference", "cmt-confof", "cmt-edas", "cmt-ekaw",
    "cmt-iasted", "cmt-sigkdd",
    "confof-edas", "conference-edas", "edas-ekaw",
    "edas-iasted", "edas-sigkdd",
})
_LABEL_OFFSETS = {
    "cmt-edas": (-0.03, 0.05),
    "cmt-iasted": (0.06, 0.10),
    "cmt-ekaw": (0.03, 0.05),
}
_ANNOTATION_STYLE = {
    "fontsize": 7,
    "color": "#222222",
    "arrowprops": {
        "arrowstyle": "->", "color": "#666666", "lw": 0.8, "shrinkA": 3, "shrinkB": 3,
    },
    "bbox": {
        "boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88,
        "edgecolor": "#cccccc", "linewidth": 0.5,
    },
}


def _counts_for(metric: str) -> Optional[dict]:
    """Load the reference counts when the metric may need them, else ``None``."""
    if metric != "hits_at_1" or not DEFAULT_COUNTS_PATH.exists():
        return None
    return load_reference_counts(DEFAULT_COUNTS_PATH)


def load_d3_scores(results_dir: Path, condition_id: str, metric: str) -> Dict[str, float]:
    """
    Load one overall metric of every D3 pair for one condition.

    Parameters
    ----------
    results_dir : Path
        Root directory of the experiment result files.
    condition_id : str
        Registered condition identifier, e.g. ``"C3"``.
    metric : str
        ``mrr``, ``hits_at_1`` or ``f1``.

    Returns
    -------
    Dict[str, float]
        Pair name mapped to that pair's metric value.
    """
    counts = _counts_for(metric)
    scores = {}
    for path in (results_dir / condition_id / _DATASET_ID).glob("*_results.json"):
        result = json.loads(path.read_text())
        scores[result["pair_name"]] = result_metric(result, metric, counts)
    return scores


def pair_scores(results_dir: Path, metric: str = _PRIMARY_METRIC) -> PairScores:
    """
    Pair the C3 and C10 metric of every D3 pair present under both conditions.

    Parameters
    ----------
    results_dir : Path
        Root directory of the experiment result files.
    metric : str
        ``mrr`` (default), ``hits_at_1`` or ``f1``.

    Returns
    -------
    PairScores
        Pair name mapped to ``(C3 value, C10 value)``, sorted by pair name.

    Raises
    ------
    ValueError
        If no pair has results under both conditions.
    """
    baseline = load_d3_scores(results_dir, _BASELINE_CONDITION, metric)
    variant = load_d3_scores(results_dir, _VARIANT_CONDITION, metric)
    shared = sorted(set(baseline) & set(variant))
    if not shared:
        raise ValueError(f"No D3 pairs shared by C3 and C10 under {results_dir}")
    return {pair: (baseline[pair], variant[pair]) for pair in shared}


def _axis_limit(scores: PairScores) -> float:
    """Return the shared upper axis limit, leaving headroom above the top point."""
    return max(max(pair) for pair in scores.values()) * 1.18


def _plot_points(ax: Axes, scores: PairScores) -> None:
    """Scatter the pairs, coloured by annotation sparsity, over the diagonal."""
    x_vals, y_vals = zip(*scores.values())
    colours = [_SPARSE_COLOUR if p in SPARSE_PAIRS else _RICH_COLOUR for p in scores]
    ax.scatter(x_vals, y_vals, c=colours, s=52, zorder=3,
               linewidths=0.5, edgecolors="white")
    lim = _axis_limit(scores)
    ax.plot([0, lim], [0, lim], color="#888888", linewidth=0.8,
            linestyle="--", zorder=1)


def _annotate_pairs(ax: Axes, scores: PairScores) -> None:
    """Label the pairs named in the paper with an offset arrow callout."""
    for pair, (dx, dy) in _LABEL_OFFSETS.items():
        if pair not in scores:
            continue
        x, y = scores[pair]
        ax.annotate(pair, xy=(x, y), xytext=(x + dx, y + dy), **_ANNOTATION_STYLE)


def _style_axes(ax: Axes, lim: float, metric: str) -> None:
    """Set axis labels, square limits and the background grid."""
    name = _METRIC_LABELS[metric]
    ax.set_xlabel(f"C3 {name}  (V2+V6 / M2)", fontsize=9)
    ax.set_ylabel(f"C10 {name}  (V2+V8 / M2)", fontsize=9)
    ax.set_xlim(-0.01, lim)
    ax.set_ylim(-0.01, lim)
    ax.tick_params(labelsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5, zorder=0)


def _add_legend_and_regions(ax: Axes, lim: float) -> None:
    """Add the sparsity legend and the 'wins' labels either side of the diagonal."""
    handles = [
        mpatches.Patch(color=_SPARSE_COLOUR, label="CMT or EDAS pair (annotation-sparse)"),
        mpatches.Patch(color=_RICH_COLOUR, label="Other Conference pair"),
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.9)
    region_style = {"fontsize": 7.5, "color": "#888888", "ha": "center",
                    "style": "italic", "rotation": 45}
    ax.text(lim * 0.22, lim * 0.78, "C10 wins", **region_style)
    ax.text(lim * 0.78, lim * 0.22, "C3 wins", **region_style)


def build_figure(scores: PairScores, metric: str = _PRIMARY_METRIC) -> Figure:
    """
    Draw the C3-versus-C10 per-pair scatter plot.

    Parameters
    ----------
    scores : PairScores
        Pair name mapped to ``(C3 value, C10 value)``.
    metric : str
        Metric plotted, for the axis labels.

    Returns
    -------
    Figure
        The finished figure, not yet saved.
    """
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    lim = _axis_limit(scores)
    _plot_points(ax, scores)
    _annotate_pairs(ax, scores)
    _style_axes(ax, lim, metric)
    _add_legend_and_regions(ax, lim)
    fig.tight_layout()
    return fig


def _output_stem(metric: str) -> str:
    """Name the primary-metric figure plainly and suffix any other metric."""
    return _OUTPUT_STEM if metric == _PRIMARY_METRIC else f"{_OUTPUT_STEM}_{metric}"


def save_figure(fig: Figure, output_dir: Path, metric: str = _PRIMARY_METRIC) -> Path:
    """
    Save the figure as a print PDF and a PNG preview.

    Parameters
    ----------
    fig : Figure
        Figure returned by :func:`build_figure`.
    output_dir : Path
        Directory to write into; created if missing.
    metric : str
        Metric plotted; any but the primary one suffixes the file name.

    Returns
    -------
    Path
        Path of the written PDF.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _output_stem(metric)
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    return pdf_path


def _winner(c3_f1: float, c10_f1: float) -> str:
    """Name the condition with the higher value, or ``"tie"``."""
    if c10_f1 > c3_f1:
        return "C10"
    return "C3" if c3_f1 > c10_f1 else "tie"


def summary_lines(scores: PairScores, metric: str = _PRIMARY_METRIC) -> List[str]:
    """
    Render the per-pair metric table and win counts printed after plotting.

    Parameters
    ----------
    scores : PairScores
        Pair name mapped to ``(C3 value, C10 value)``.
    metric : str
        Metric of the values, for the column headings.

    Returns
    -------
    List[str]
        Table lines, sparse pairs starred, followed by overall and sparse-pair
        win counts out of the number of pairs plotted.
    """
    name = _METRIC_LABELS[metric]
    lines = [f"{'Pair':<35} {'C3 ' + name:>7} {'C10 ' + name:>7} {'Winner'}", "-" * 58]
    winners = {pair: _winner(*f1s) for pair, f1s in scores.items()}
    for pair, (c3_f1, c10_f1) in scores.items():
        label = pair + (" *" if pair in SPARSE_PAIRS else "")
        lines.append(f"{label:<35} {c3_f1:>7.4f} {c10_f1:>7.4f}  {winners[pair]}")
    sparse = [pair for pair in scores if pair in SPARSE_PAIRS]
    count = list(winners.values()).count
    sparse_c10 = sum(winners[pair] == "C10" for pair in sparse)
    lines.append(f"\nC10 wins: {count('C10')}/{len(scores)}, "
                 f"C3 wins: {count('C3')}/{len(scores)}")
    lines.append(f"C10 wins on sparse pairs: {sparse_c10}/{len(sparse)}")
    return lines


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for Figure 1."""
    parser = argparse.ArgumentParser(
        description="Plot a D3 per-pair metric of C3 against C10 (Figure 1)."
    )
    parser.add_argument(
        "--results_dir", type=Path, default=Path("data/results/"),
        help="Root directory of result files.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("figures/"),
        help="Directory for the PDF and PNG output.",
    )
    parser.add_argument(
        "--metric", default=_PRIMARY_METRIC, choices=tuple(_METRIC_LABELS),
        help="Metric to plot (default: the primary metric, re-ranking MRR).",
    )
    return parser


def main() -> None:
    """Plot Figure 1 from the D3 results and print the per-pair summary."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    scores = pair_scores(args.results_dir, args.metric)
    pdf_path = save_figure(build_figure(scores, args.metric), args.output_dir, args.metric)
    print(f"Saved: {pdf_path}\n\nPair-level results:")
    print("\n".join(summary_lines(scores, args.metric)))


if __name__ == "__main__":
    main()
