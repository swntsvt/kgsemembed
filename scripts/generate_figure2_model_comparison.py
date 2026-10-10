"""Command-line entry point for Figure 2, the D1/D2 model comparison chart.

Plots F1 on D1 and D2 for each of the five embedding models with the V2+V8
strategy held constant (C18, C10, C15, C11, C13), alongside the C17 reference
bar (V1 on BioLORD).  The chart shows that domain-specific pre-training
dominates biomedical class matching regardless of verbalisation strategy.

Each result file's ``strategy`` and ``model_key`` are checked against the bar
it is plotted under, so a bar can never carry the wrong label.  A missing
result, such as C15 on D2 which is not run, is drawn as ``n/a`` rather than
as a zero-height bar.

Writes ``fig2_model_comparison.pdf`` (and a ``.png`` preview) to the output
directory and prints the F1 values plotted.

Usage
-----
python scripts/generate_figure2_model_comparison.py \\
    --results_dir data/results/ --output_dir figures/
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


@dataclass(frozen=True)
class BarSpec:
    """One model group on the x-axis and the condition that supplies it."""

    label: str
    condition_id: str
    strategy: str
    model_key: str
    colour: str


@dataclass(frozen=True)
class DatasetSpec:
    """One bar within each model group."""

    dataset_id: str
    pair_name: str
    label: str
    alpha: float


ModelScores = Dict[str, List[Optional[float]]]

_OUTPUT_STEM = "fig2_model_comparison"
_CONSTANT_STRATEGY = "V2+V8"
BARS = (
    BarSpec("M1\n(MiniLM)", "C18", _CONSTANT_STRATEGY, "M1", "#7fb3d3"),
    BarSpec("M2\n(bge-large)", "C10", _CONSTANT_STRATEGY, "M2", "#5b7fa6"),
    BarSpec("M3\n(bge-m3)", "C15", _CONSTANT_STRATEGY, "M3", "#3a5f8a"),
    BarSpec("M4\n(BioLORD)", "C11", _CONSTANT_STRATEGY, "M4", "#e07b54"),
    BarSpec("M5\n(stella)", "C13", _CONSTANT_STRATEGY, "M5", "#9b59b6"),
    BarSpec("M4 V1\n(BioLORD,\nC17 ref)", "C17", "V1", "M4", "#c0392b"),
)
DATASETS = (
    DatasetSpec("D1", "d1_snomed_fma", "D1 (SNOMED-FMA)", 0.45),
    DatasetSpec("D2", "d2_anatomy", "D2 (Anatomy MA-NCI)", 1.0),
)
_BAR_WIDTH = 0.35
_MISSING_LABEL = "n/a"


def _check_condition(result: dict, bar: BarSpec) -> None:
    """Raise if a result file was produced by a different strategy or model."""
    found = (result["strategy"], result["model_key"])
    if found != (bar.strategy, bar.model_key):
        raise ValueError(
            f"{bar.condition_id} result has strategy/model {found}, "
            f"expected {(bar.strategy, bar.model_key)}"
        )


def load_f1(results_dir: Path, bar: BarSpec, dataset: DatasetSpec) -> Optional[float]:
    """
    Load the overall F1 of one condition on one dataset.

    Parameters
    ----------
    results_dir : Path
        Root directory of the experiment result files.
    bar : BarSpec
        Bar whose condition, strategy and model the result must match.
    dataset : DatasetSpec
        Dataset whose single pair is read.

    Returns
    -------
    Optional[float]
        The pair's F1, or ``None`` if the result file does not exist.

    Raises
    ------
    ValueError
        If the result's strategy or model differs from the bar's.
    """
    path = (results_dir / bar.condition_id / dataset.dataset_id
            / f"{dataset.pair_name}_results.json")
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    _check_condition(result, bar)
    return result["metrics"]["f1"]


def collect_scores(results_dir: Path) -> ModelScores:
    """
    Load the F1 of every bar on every dataset.

    Parameters
    ----------
    results_dir : Path
        Root directory of the experiment result files.

    Returns
    -------
    ModelScores
        Dataset identifier mapped to F1 values aligned with ``BARS``; a
        missing result is ``None``.

    Raises
    ------
    ValueError
        If no result file is found for any bar.
    """
    scores = {d.dataset_id: [load_f1(results_dir, b, d) for b in BARS] for d in DATASETS}
    if all(f1 is None for values in scores.values() for f1 in values):
        raise ValueError(f"No D1 or D2 results found under {results_dir}")
    return scores


def _y_limit(scores: ModelScores) -> float:
    """Return the upper y limit, leaving headroom for the value labels."""
    present = [f1 for values in scores.values() for f1 in values if f1 is not None]
    return max(present) * 1.25


def _label_bar(ax: Axes, x: float, f1: Optional[float]) -> None:
    """Write the F1 above a bar, or ``n/a`` where the result is missing."""
    if f1 is None:
        ax.text(x, 0.01, _MISSING_LABEL, ha="center", va="bottom",
                fontsize=6.5, color="#888888", rotation=90)
        return
    ax.text(x, f1 + 0.008, f"{f1:.3f}", ha="center", va="bottom",
            fontsize=6.5, rotation=90 if f1 < 0.12 else 0)


def _plot_dataset(ax: Axes, values: List[Optional[float]], dataset: DatasetSpec,
                  offset: float) -> None:
    """Draw one dataset's bar in every model group, with its value labels."""
    x = np.arange(len(BARS)) + offset
    heights = [0.0 if f1 is None else f1 for f1 in values]
    colours = [bar.colour for bar in BARS]
    ax.bar(x, heights, _BAR_WIDTH, color=colours, alpha=dataset.alpha,
           edgecolor=colours, linewidth=0.8, zorder=2)
    for xi, f1 in zip(x, values):
        _label_bar(ax, xi, f1)


def _add_reference_divider(ax: Axes, y_top: float) -> None:
    """Separate the V2+V8 groups from the C17 reference bar."""
    divider_x = len(BARS) - 1.5
    ax.axvline(x=divider_x, color="#aaaaaa", linewidth=0.8, linestyle=":")
    ax.text(divider_x, y_top * 0.98, "V1 ref", ha="center", va="top",
            fontsize=6.5, color="#888888",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ax.text((len(BARS) - 2) / 2, -0.17, f"Strategy: {_CONSTANT_STRATEGY} (constant)",
            ha="center", transform=ax.get_xaxis_transform(),
            fontsize=7.5, color="#444444", style="italic")


def _style_axes(ax: Axes, y_top: float) -> None:
    """Set tick labels, limits, the dataset legend and the background grid."""
    ax.set_ylabel("F1", fontsize=9)
    ax.set_xticks(np.arange(len(BARS)))
    ax.set_xticklabels([bar.label for bar in BARS], fontsize=8)
    ax.set_ylim(0, y_top)
    ax.tick_params(axis="y", labelsize=8)
    handles = [mpatches.Patch(color="#666666", alpha=d.alpha, label=d.label)
               for d in DATASETS]
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


def build_figure(scores: ModelScores) -> Figure:
    """
    Draw the grouped bar chart of F1 per model on D1 and D2.

    Parameters
    ----------
    scores : ModelScores
        Output of :func:`collect_scores`.

    Returns
    -------
    Figure
        The finished figure, not yet saved.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    offsets = (-_BAR_WIDTH / 2, _BAR_WIDTH / 2)
    for dataset, offset in zip(DATASETS, offsets):
        _plot_dataset(ax, scores[dataset.dataset_id], dataset, offset)
    y_top = _y_limit(scores)
    _style_axes(ax, y_top)
    _add_reference_divider(ax, y_top)
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


def _format_f1(f1: Optional[float]) -> str:
    """Format an F1 for the summary table, ``n/a`` when missing."""
    return f"{_MISSING_LABEL:>8}" if f1 is None else f"{f1:>8.4f}"


def summary_lines(scores: ModelScores) -> List[str]:
    """
    Render the table of F1 values printed after plotting.

    Parameters
    ----------
    scores : ModelScores
        Output of :func:`collect_scores`.

    Returns
    -------
    List[str]
        Header, rule, and one row per bar giving its condition, strategy,
        model and F1 on each dataset.
    """
    header = f"{'Condition':<10} {'Strategy':<8} {'Model':<6}"
    header += "".join(f" {d.dataset_id + ' F1':>8}" for d in DATASETS)
    lines = [header, "-" * len(header)]
    for i, bar in enumerate(BARS):
        row = f"{bar.condition_id:<10} {bar.strategy:<8} {bar.model_key:<6}"
        row += "".join(f" {_format_f1(scores[d.dataset_id][i])}" for d in DATASETS)
        lines.append(row)
    return lines


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for Figure 2."""
    parser = argparse.ArgumentParser(
        description="Plot D1/D2 F1 per model with V2+V8 held constant (Figure 2)."
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
    """Plot Figure 2 from the D1/D2 results and print the values used."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    scores = collect_scores(args.results_dir)
    pdf_path = save_figure(build_figure(scores), args.output_dir)
    print(f"Saved: {pdf_path}\n\nData used:")
    print("\n".join(summary_lines(scores)))


if __name__ == "__main__":
    main()
