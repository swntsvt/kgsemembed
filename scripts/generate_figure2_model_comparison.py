"""Command-line entry point for Figure 2, the D1/D2 model comparison chart.

Plots one metric on D1 and D2 for each of the five embedding models with the
V2+V8 strategy held constant (C18, C10, C15, C11, C13), alongside the C17
reference bar (V1 on BioLORD): re-ranking MRR by default, the paper's primary
metric, or ``--metric hits_at_1`` / ``--metric f1``.  Holding the strategy
fixed isolates the model's contribution on the two biomedical datasets.  The
figure's claims rest on the paired intervals of ``scripts/figure_claims.py``
(#149, ``artifacts/issue-149/``), not on the bar heights: under MRR, M5 leads
the V2+V8 models on D1 and ties M4 on D2, and C17 is above every V2+V8 bar on
both datasets.

Each result file's ``strategy`` and ``model_key`` are checked against the bar
it is plotted under, so a bar can never carry the wrong label.  A missing
result is drawn as ``n/a`` rather than as a zero-height bar.

Writes ``fig2_model_comparison.pdf`` (and a ``.png`` preview) to the output
directory for MRR, ``fig2_model_comparison_{metric}.pdf`` otherwise, and
prints the values plotted.

Usage
-----
python scripts/generate_figure2_model_comparison.py \\
    --results_dir data/results/ --output_dir figures/ --metric mrr
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
from scripts.figure_specs import (  # noqa: E402
    BARS,
    CONSTANT_STRATEGY,
    DATASETS,
    BarSpec,
    DatasetSpec,
    check_condition,
)

ModelScores = Dict[str, List[Optional[float]]]

_OUTPUT_STEM = "fig2_model_comparison"
_PRIMARY_METRIC = "mrr"
_METRIC_LABELS = {"mrr": "MRR", "hits_at_1": "Hits@1", "f1": "F1"}
_BAR_WIDTH = 0.35
_MISSING_LABEL = "n/a"


def _counts_for(metric: str) -> Optional[dict]:
    """Load the reference counts when the metric may need them, else ``None``."""
    if metric != "hits_at_1" or not DEFAULT_COUNTS_PATH.exists():
        return None
    return load_reference_counts(DEFAULT_COUNTS_PATH)


def load_score(
    results_dir: Path, bar: BarSpec, dataset: DatasetSpec, metric: str = _PRIMARY_METRIC
) -> Optional[float]:
    """
    Load one overall metric of one condition on one dataset.

    Parameters
    ----------
    results_dir : Path
        Root directory of the experiment result files.
    bar : BarSpec
        Bar whose condition, strategy and model the result must match.
    dataset : DatasetSpec
        Dataset whose single pair is read.
    metric : str
        ``mrr`` (default), ``hits_at_1`` or ``f1``.

    Returns
    -------
    Optional[float]
        The pair's metric value, or ``None`` if the result file does not exist.

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
    check_condition(result, bar)
    return result_metric(result, metric, _counts_for(metric))


def collect_scores(results_dir: Path, metric: str = _PRIMARY_METRIC) -> ModelScores:
    """
    Load one metric of every bar on every dataset.

    Parameters
    ----------
    results_dir : Path
        Root directory of the experiment result files.
    metric : str
        ``mrr`` (default), ``hits_at_1`` or ``f1``.

    Returns
    -------
    ModelScores
        Dataset identifier mapped to values aligned with ``BARS``; a
        missing result is ``None``.

    Raises
    ------
    ValueError
        If no result file is found for any bar.
    """
    scores = {d.dataset_id: [load_score(results_dir, b, d, metric) for b in BARS] for d in DATASETS}
    if all(f1 is None for values in scores.values() for f1 in values):
        raise ValueError(f"No D1 or D2 results found under {results_dir}")
    return scores


def _y_limit(scores: ModelScores) -> float:
    """Return the upper y limit, leaving headroom for the value labels."""
    present = [f1 for values in scores.values() for f1 in values if f1 is not None]
    return max(present) * 1.25


def _label_bar(ax: Axes, x: float, f1: Optional[float]) -> None:
    """Write the value above a bar, or ``n/a`` where the result is missing."""
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
    ax.text((len(BARS) - 2) / 2, -0.17, f"Strategy: {CONSTANT_STRATEGY} (constant)",
            ha="center", transform=ax.get_xaxis_transform(),
            fontsize=7.5, color="#444444", style="italic")


def _style_axes(ax: Axes, y_top: float, metric: str) -> None:
    """Set tick labels, limits, the dataset legend and the background grid."""
    ax.set_ylabel(_METRIC_LABELS[metric], fontsize=9)
    ax.set_xticks(np.arange(len(BARS)))
    ax.set_xticklabels([bar.label for bar in BARS], fontsize=8)
    ax.set_ylim(0, y_top)
    ax.tick_params(axis="y", labelsize=8)
    handles = [mpatches.Patch(color="#666666", alpha=d.alpha, label=d.label)
               for d in DATASETS]
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)


def build_figure(scores: ModelScores, metric: str = _PRIMARY_METRIC) -> Figure:
    """
    Draw the grouped bar chart of one metric per model on D1 and D2.

    Parameters
    ----------
    scores : ModelScores
        Output of :func:`collect_scores`.
    metric : str
        Metric plotted, for the axis label.

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
    _style_axes(ax, y_top, metric)
    _add_reference_divider(ax, y_top)
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


def _format_f1(f1: Optional[float]) -> str:
    """Format a value for the summary table, ``n/a`` when missing."""
    return f"{_MISSING_LABEL:>8}" if f1 is None else f"{f1:>8.4f}"


def summary_lines(scores: ModelScores, metric: str = _PRIMARY_METRIC) -> List[str]:
    """
    Render the table of metric values printed after plotting.

    Parameters
    ----------
    scores : ModelScores
        Output of :func:`collect_scores`.
    metric : str
        Metric of the values, for the column headings.

    Returns
    -------
    List[str]
        Header, rule, and one row per bar giving its condition, strategy,
        model and value on each dataset.
    """
    name = _METRIC_LABELS[metric]
    header = f"{'Condition':<10} {'Strategy':<8} {'Model':<6}"
    header += "".join(f" {d.dataset_id + ' ' + name:>9}" for d in DATASETS)
    lines = [header, "-" * len(header)]
    for i, bar in enumerate(BARS):
        row = f"{bar.condition_id:<10} {bar.strategy:<8} {bar.model_key:<6}"
        row += "".join(f"  {_format_f1(scores[d.dataset_id][i])}" for d in DATASETS)
        lines.append(row)
    return lines


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for Figure 2."""
    parser = argparse.ArgumentParser(
        description="Plot a D1/D2 metric per model with V2+V8 held constant (Figure 2)."
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
    """Plot Figure 2 from the D1/D2 results and print the values used."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args()
    scores = collect_scores(args.results_dir, args.metric)
    pdf_path = save_figure(build_figure(scores, args.metric), args.output_dir, args.metric)
    print(f"Saved: {pdf_path}\n\nData used:")
    print("\n".join(summary_lines(scores, args.metric)))


if __name__ == "__main__":
    main()
