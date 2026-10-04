"""Command-line entry point for Figure 1, the D3 per-pair scatter plot.

Plots per-pair F1 of C3 (V2+V6 / M2) against C10 (V2+V8 / M2) over the 21
Conference pairs.  Pairs involving CMT or EDAS, the annotation-sparse
ontologies, are highlighted to show that V8's advantage over V6 concentrates
on them.  Points above the diagonal are pairs where C10 wins.

Writes ``fig1_d3_scatter.pdf`` (and a ``.png`` preview) to the output
directory and prints a per-pair table with win counts.

Usage
-----
python scripts/generate_figure1_d3_scatter.py \\
    --results_dir data/results/ --output_dir figures/
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

PairScores = Dict[str, Tuple[float, float]]

_DATASET_ID = "D3"
_BASELINE_CONDITION = "C3"
_VARIANT_CONDITION = "C10"
_OUTPUT_STEM = "fig1_d3_scatter"
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


def load_d3_f1(results_dir: Path, condition_id: str) -> Dict[str, float]:
    """
    Load the overall F1 of every D3 pair for one condition.

    Parameters
    ----------
    results_dir : Path
        Root directory of the experiment result files.
    condition_id : str
        Registered condition identifier, e.g. ``"C3"``.

    Returns
    -------
    Dict[str, float]
        Pair name mapped to that pair's F1.
    """
    scores = {}
    for path in (results_dir / condition_id / _DATASET_ID).glob("*_results.json"):
        result = json.loads(path.read_text())
        scores[result["pair_name"]] = result["metrics"]["f1"]
    return scores


def pair_scores(results_dir: Path) -> PairScores:
    """
    Pair the C3 and C10 F1 of every D3 pair present under both conditions.

    Parameters
    ----------
    results_dir : Path
        Root directory of the experiment result files.

    Returns
    -------
    PairScores
        Pair name mapped to ``(C3 F1, C10 F1)``, sorted by pair name.

    Raises
    ------
    ValueError
        If no pair has results under both conditions.
    """
    baseline = load_d3_f1(results_dir, _BASELINE_CONDITION)
    variant = load_d3_f1(results_dir, _VARIANT_CONDITION)
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


def _style_axes(ax: Axes, lim: float) -> None:
    """Set axis labels, square limits and the background grid."""
    ax.set_xlabel("C3 F1  (V2+V6 / M2)", fontsize=9)
    ax.set_ylabel("C10 F1  (V2+V8 / M2)", fontsize=9)
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


def build_figure(scores: PairScores) -> Figure:
    """
    Draw the C3-versus-C10 per-pair scatter plot.

    Parameters
    ----------
    scores : PairScores
        Pair name mapped to ``(C3 F1, C10 F1)``.

    Returns
    -------
    Figure
        The finished figure, not yet saved.
    """
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    lim = _axis_limit(scores)
    _plot_points(ax, scores)
    _annotate_pairs(ax, scores)
    _style_axes(ax, lim)
    _add_legend_and_regions(ax, lim)
    fig.tight_layout()
    return fig


def save_figure(fig: Figure, output_dir: Path) -> Path:
    """
    Save the figure as a print PDF and a PNG preview.

    Parameters
    ----------
    fig : Figure
        Figure returned by :func:`build_figure`.
    output_dir : Path
        Directory to write into; created if missing.

    Returns
    -------
    Path
        Path of the written PDF.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{_OUTPUT_STEM}.pdf"
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / f"{_OUTPUT_STEM}.png", dpi=200, bbox_inches="tight")
    return pdf_path


def _winner(c3_f1: float, c10_f1: float) -> str:
    """Name the condition with the higher F1, or ``"tie"``."""
    if c10_f1 > c3_f1:
        return "C10"
    return "C3" if c3_f1 > c10_f1 else "tie"


def summary_lines(scores: PairScores) -> List[str]:
    """
    Render the per-pair F1 table and win counts printed after plotting.

    Parameters
    ----------
    scores : PairScores
        Pair name mapped to ``(C3 F1, C10 F1)``.

    Returns
    -------
    List[str]
        Table lines, sparse pairs starred, followed by overall and sparse-pair
        win counts out of the number of pairs plotted.
    """
    lines = [f"{'Pair':<35} {'C3 F1':>7} {'C10 F1':>7} {'Winner'}", "-" * 58]
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
        description="Plot D3 per-pair F1 of C3 against C10 (Figure 1)."
    )
    parser.add_argument(
        "--results_dir", type=Path, default=Path("data/results/"),
        help="Root directory of result files.",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("figures/"),
        help="Directory for the PDF and PNG output.",
    )
    return parser


def main() -> None:
    """Plot Figure 1 from the D3 results and print the per-pair summary."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    scores = pair_scores(args.results_dir)
    pdf_path = save_figure(build_figure(scores), args.output_dir)
    print(f"Saved: {pdf_path}\n\nPair-level results:")
    print("\n".join(summary_lines(scores)))


if __name__ == "__main__":
    main()
