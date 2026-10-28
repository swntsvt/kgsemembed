"""Cost report of one column: table, LaTeX table and cost-vs-MRR figure (#181).

Reads the timing records of ``scripts/encode_timing.py``, the encoded input
size of ``scripts/encoded_input_size.py`` and the stored MRR of every
condition and of B0, and writes, per column:

- ``cost_{column}.md`` and ``cost_{column}.tex``: per condition, model size,
  texts and tokens encoded, verbalisation, encoding (median of the timed
  repeats, with its range) and stage-2 seconds, cost relative to the cheapest
  condition, MRR and its difference from B0; B0's row is stage 1 alone, which
  every condition also runs;
- ``cost_vs_mrr_{column}.{pdf,png}``: MRR against stage-2 seconds (log scale),
  one point per condition, B0 as a reference line.

Reads only; regenerates byte-identically from the same inputs (PDF metadata
dates are fixed).

Usage
-----
python scripts/generate_cost_report.py --column D1
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kgsemembed.pipeline.conditions import EXPERIMENT_CONDITIONS  # noqa: E402

MODEL_STYLE = {
    "M1": ("#2a78d6", "o"), "M2": ("#eb6834", "s"), "M3": ("#1baf7a", "D"),
    "M4": ("#eda100", "^"), "M5": ("#e87ba4", "v"),
}
MODEL_NAMES = {
    "M1": "M1 MiniLM-L6", "M2": "M2 bge-large", "M3": "M3 bge-m3",
    "M4": "M4 BioLORD", "M5": "M5 stella 1.5B",
}
_REGISTRY_ORDER = [condition.condition_id for condition in EXPERIMENT_CONDITIONS]
_INK, _MUTED, _GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def read_json(path: Path) -> object:
    """
    Read one JSON file.

    Parameters
    ----------
    path : Path
        File to read.

    Returns
    -------
    object
        The parsed content.
    """
    return json.loads(path.read_text())


def stage2_row(record: Dict[str, object], sizes: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """
    Summarise one condition's timing record as a table row.

    Parameters
    ----------
    record : Dict[str, object]
        Timing record of ``encode_timing.py``.
    sizes : Dict[str, Dict[str, object]]
        Encoded input size records of the column, by condition.

    Returns
    -------
    Dict[str, object]
        Condition identity, model size, texts, tokens and seconds per stage.
    """
    passes = [s + c for s, c in zip(record["source_encode_seconds"], record["candidate_encode_seconds"])]
    encode = statistics.median(passes)
    return {
        "condition_id": record["condition_id"], "strategy": record["strategy"],
        "model_key": record["model_key"], "n_parameters": record["model"]["n_parameters"],
        "n_texts": record["n_source_texts"] + record["n_candidate_texts"],
        "encoded_tokens": sizes[record["condition_id"]]["total"]["encoded_tokens"],
        "verbalise": record["verbalise_seconds"], "encode": encode,
        "encode_range": (min(passes), max(passes)), "rank": record["rank_evaluate_seconds"],
        "stage2": record["verbalise_seconds"] + encode + record["rank_evaluate_seconds"],
        "mrr": record["metrics"]["mrr"],
    }


def load_rows(args: argparse.Namespace) -> List[Dict[str, object]]:
    """
    Build the condition rows of the column, in registry order.

    Parameters
    ----------
    args : argparse.Namespace
        ``timing_dir``, ``input_size`` and ``column``.

    Returns
    -------
    List[Dict[str, object]]
        One row per timed condition.
    """
    sizes = {r["condition_id"]: r for r in read_json(Path(args.input_size)) if r["column"] == args.column}
    records = [read_json(path) for path in sorted(Path(args.timing_dir).glob(f"C*_{args.column}_*.json"))]
    rows = [stage2_row(record, sizes) for record in records]
    return sorted(rows, key=lambda row: _REGISTRY_ORDER.index(row["condition_id"]))


def baseline_row(args: argparse.Namespace) -> Dict[str, object]:
    """
    Build B0's row: stage-1 seconds and its stored MRR.

    Parameters
    ----------
    args : argparse.Namespace
        ``timing_dir``, ``baseline_dir`` and ``column``.

    Returns
    -------
    Dict[str, object]
        Median and range of the stage-1 seconds, and B0's MRR.
    """
    (record,) = [read_json(p) for p in Path(args.timing_dir).glob(f"stage1_{args.column}_*.json")]
    stored = read_json(Path(args.baseline_dir) / "B0" / args.column / f"{record['pair_name']}_results.json")
    seconds = record["score_seconds"]
    return {"stage1": statistics.median(seconds), "stage1_range": (min(seconds), max(seconds)),
            "mrr": stored["metrics"]["mrr"], "n_sources": record["n_sources"]}


def machine_note(args: argparse.Namespace) -> str:
    """
    Describe the machine, commit and repeats behind the timings.

    Parameters
    ----------
    args : argparse.Namespace
        ``timing_dir`` and ``column``.

    Returns
    -------
    str
        One sentence naming chip, memory, device, torch version, commit and repeats.
    """
    record = read_json(sorted(Path(args.timing_dir).glob(f"C*_{args.column}_*.json"))[0])
    machine = record["machine"]
    return (
        f"{machine['chip']}, {machine['memory_bytes'] // 2**30} GB, device {record['model']['device']}, "
        f"torch {machine['versions']['torch']}, commit {record['git_commit'][:7]}, "
        f"{record['repeats']} timed encoding passes after a warm-up"
    )


def _markdown_line(row: Dict[str, object], cheapest: float, b0_mrr: float) -> str:
    low, high = row["encode_range"]
    cells = [
        row["condition_id"], row["strategy"], row["model_key"], f"{row['n_parameters'] / 1e6:,.0f}",
        f"{row['n_texts']:,}", f"{row['encoded_tokens'] / 1e6:.2f}", f"{row['verbalise']:.1f}",
        f"{row['encode']:.1f} ({low:.1f}–{high:.1f})", f"{row['stage2']:.1f}",
        f"{row['stage2'] / cheapest:.1f}×", f"{row['mrr']:.4f}", f"{row['mrr'] - b0_mrr:+.4f}",
    ]
    return "| " + " | ".join(cells) + " |"


def render_markdown(rows: List[Dict[str, object]], b0: Dict[str, object], args: argparse.Namespace) -> str:
    """
    Render the cost table as Markdown.

    Parameters
    ----------
    rows : List[Dict[str, object]]
        Condition rows.
    b0 : Dict[str, object]
        B0's row.
    args : argparse.Namespace
        ``column`` and ``timing_dir``.

    Returns
    -------
    str
        The Markdown document.
    """
    cheapest = min(row["stage2"] for row in rows)
    low, high = b0["stage1_range"]
    lines = [
        f"# Computational cost on {args.column} (#181)", "",
        "Generated by `scripts/generate_cost_report.py` from `scripts/encode_timing.py` and "
        f"`scripts/encoded_input_size.py`. Machine: {machine_note(args)}. Seconds are wall-clock "
        "and descriptive: they hold for this implementation and machine (MPS, batches padded to a "
        "multiple of 64 tokens, batch size reduced for long texts), not in general. Every timed run "
        "reproduced its stored metrics exactly, and every encoding pass returned identical embeddings.",
        "",
        f"Stage 1 (n-gram scoring over all targets, every condition's shared first stage and B0's "
        f"whole cost): {b0['stage1']:.1f} s ({low:.1f}–{high:.1f}) for {b0['n_sources']:,} sources; "
        f"B0 MRR {b0['mrr']:.4f}. Stage 2 below comes on top of it; dataset loading is excluded.", "",
        "| Cond. | Strategy | Model | Params (M) | Texts | Encoded tokens (M) | Verbalise s "
        "| Encode s (range) | Stage 2 s | × cheapest | MRR | Δ MRR vs B0 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    return "\n".join(lines + [_markdown_line(row, cheapest, b0["mrr"]) for row in rows]) + "\n"


def _latex_line(row: Dict[str, object], cheapest: float, b0_mrr: float) -> str:
    cells = [
        row["condition_id"], row["strategy"], row["model_key"], f"{row['n_parameters'] / 1e6:,.0f}",
        f"{row['encoded_tokens'] / 1e6:.2f}", f"{row['stage2'] / 60:.1f}",
        f"{row['stage2'] / cheapest:.0f}", f"{row['mrr']:.3f}", f"{row['mrr'] - b0_mrr:+.3f}".replace("-", "$-$"),
    ]
    return " & ".join(cells).replace(",", "{,}") + r" \\"


def render_latex(rows: List[Dict[str, object]], b0: Dict[str, object], args: argparse.Namespace) -> str:
    """
    Render the cost table as LaTeX (threeparttable, as the results tables).

    Parameters
    ----------
    rows : List[Dict[str, object]]
        Condition rows.
    b0 : Dict[str, object]
        B0's row.
    args : argparse.Namespace
        ``column`` and ``timing_dir``.

    Returns
    -------
    str
        The LaTeX table.
    """
    cheapest = min(row["stage2"] for row in rows)
    head = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\begin{threeparttable}",
        rf"\caption{{Computational cost of stage~2 on {args.column}, one run per condition. Tokens: "
        r"total encoded tokens (sources and unique candidates, each capped at the model's limit). "
        r"Time: verbalisation, encoding (median of three passes) and ranking, in minutes; "
        r"$\times$: relative to the cheapest condition. Stage~1, shared by every condition and "
        rf"B0's whole cost, takes {b0['stage1']:.0f}\,s.}}",
        rf"\label{{tab:cost_{args.column.lower()}}}", r"\begin{tabular}{lllrrrrrr}", r"\hline",
        r"\textbf{Cond.} & \textbf{Text} & \textbf{Model} & \textbf{Params (M)} & \textbf{Tokens (M)} "
        r"& \textbf{Time (min)} & $\times$ & \textbf{MRR} & $\Delta$\textbf{B0} \\", r"\hline",
        rf"B0 & -- & -- & -- & -- & -- & -- & {b0['mrr']:.3f} & -- \\",
    ]
    tail = [r"\hline", r"\end{tabular}", r"\begin{tablenotes}\footnotesize",
            rf"\item Timings: {machine_note(args)}; descriptive, specific to this machine and implementation.",
            r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    return "\n".join(head + [_latex_line(row, cheapest, b0["mrr"]) for row in rows] + tail) + "\n"


def _style_axes(axes: plt.Axes, column: str) -> None:
    axes.set_xscale("log")
    axes.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes.xaxis.set_minor_formatter(NullFormatter())
    axes.set_xlabel("Stage-2 wall-clock time (s, log scale)", color=_INK)
    axes.set_ylabel(f"{column} re-ranking MRR", color=_INK)
    axes.grid(True, which="major", color=_GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(_MUTED)
    axes.tick_params(colors=_MUTED)


def plot(rows: List[Dict[str, object]], b0: Dict[str, object], column: str, stem: Path) -> None:
    """
    Plot MRR against stage-2 seconds, one labelled point per condition.

    Parameters
    ----------
    rows : List[Dict[str, object]]
        Condition rows.
    b0 : Dict[str, object]
        B0's row (reference line).
    column : str
        Column name for the axis label.
    stem : Path
        Output path without suffix; ``.pdf`` and ``.png`` are written.
    """
    figure, axes = plt.subplots(figsize=(6.4, 4.0))
    _style_axes(axes, column)
    axes.axhline(b0["mrr"], color=_MUTED, linestyle="--", linewidth=1.2)
    axes.annotate(f"B0, stage 1 only ({b0['stage1']:.3g} s)", (0.01, b0["mrr"]), xycoords=("axes fraction", "data"),
                  xytext=(0, 4), textcoords="offset points", color=_MUTED, fontsize=8)
    for key, (colour, marker) in MODEL_STYLE.items():
        own = [row for row in rows if row["model_key"] == key]
        if not own:
            continue
        axes.scatter([r["stage2"] for r in own], [r["mrr"] for r in own], s=46, color=colour, marker=marker,
                     edgecolors="white", linewidths=1.5, label=MODEL_NAMES[key], zorder=3)
        for row in own:
            axes.annotate(row["condition_id"], (row["stage2"], row["mrr"]), xytext=(5, 3),
                          textcoords="offset points", fontsize=7.5, color=_INK)
    axes.legend(frameon=False, fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(stem.with_suffix(".png"), dpi=200, metadata={"Software": None})
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Input and output locations and the column.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--column", default="D1")
    parser.add_argument("--timing_dir", default="artifacts/issue-181/timing/")
    parser.add_argument("--input_size", default="artifacts/issue-181/encoded_input_size.json")
    parser.add_argument("--baseline_dir", default="data/results_ngram_baseline/")
    parser.add_argument("--output_dir", default="artifacts/issue-181/")
    return parser.parse_args()


def main() -> None:
    """Write the cost table (Markdown, LaTeX) and figure of one column."""
    np.random.seed(42)
    random.seed(42)
    args = parse_args()
    rows, b0 = load_rows(args), baseline_row(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"cost_{args.column}.md").write_text(render_markdown(rows, b0, args))
    (output / f"cost_{args.column}.tex").write_text(render_latex(rows, b0, args))
    plot(rows, b0, args.column, output / f"cost_vs_mrr_{args.column}")


if __name__ == "__main__":
    main()
