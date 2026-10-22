"""Per-dataset paired intervals behind the claims of Figures 1 and 2 (#149).

Both figures plot point estimates; this script computes the #140 paired
intervals their claims must rest on, from the per-source sidecars.

- **Figure 1** (D3, C3 V2+V6/M2 vs C10 V2+V8/M2): C10 − C3 over all 21 pairs,
  over the CMT/EDAS pairs the figure highlights and over the other pairs, the
  difference between those two subgroups, and each pair on its own.  The
  subgroup difference gets two intervals: resampling test sources within each
  pair (as every #140 interval does) and resampling whole pairs within each
  subgroup, which also reflects how much pairs differ from one another.
- **Figure 2** (D1/D2, V2+V8 per model, plus C17 V1/M4): every pair of plotted
  conditions on each dataset, B − A with A the earlier bar.  These are
  exploratory: the figure's own family per dataset gets Holm-adjusted MRR
  p-values, separate from the registered comparisons in ``report.md``.

Every interval is a 95% percentile bootstrap and every p-value a paired
randomisation test, with ``paired_tests.N_REPLICATES`` replicates and seed
``paired_tests.SEED``.  Writes ``figure_claims.json`` and ``figure_claims.md``.

Usage
-----
python scripts/figure_claims.py --results_dir data/results/ \\
    --output_dir artifacts/issue-149/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kgsemembed.evaluation.paired_tests import (  # noqa: E402
    CONFIDENCE,
    DECISION_METRIC,
    METRICS,
    PairArrays,
    SidecarCache,
    TestSettings,
    _bootstrap_deltas,
    aligned_arrays,
    check_same_references,
    holm_adjust,
    paired_test,
    statistic,
)
from scripts.generate_figure1_d3_scatter import SPARSE_PAIRS  # noqa: E402
from scripts.generate_figure2_model_comparison import BARS, DATASETS  # noqa: E402

FIG1_A, FIG1_B, FIG1_DATASET = "C3", "C10", "D3"
SUBGROUPS = ("all", "cmt_edas", "other")
_METRIC_NAMES = {"mrr": "MRR", "hits_at_1": "Hits@1", "f1": "F1"}
_TAIL = 100 * (1 - CONFIDENCE) / 2


def column_arrays(
    cache: SidecarCache, conditions: Tuple[str, str], dataset_id: str, metric: str
) -> Dict[str, PairArrays]:
    """
    Align two conditions' unit values on every pair of a column.

    Parameters
    ----------
    cache : SidecarCache
        Sidecar cache of the results directory.
    conditions : Tuple[str, str]
        Conditions A and B.
    dataset_id : str
        Dataset column.
    metric : str
        One of ``paired_tests.METRICS``.

    Returns
    -------
    Dict[str, PairArrays]
        Pair name mapped to its aligned ``(a, b)`` arrays.

    Raises
    ------
    DataError
        If the two conditions' sidecars describe different references or setups.
    """
    condition_a, condition_b = conditions
    check_same_references(cache.sidecars(condition_a, dataset_id),
                          cache.sidecars(condition_b, dataset_id))
    values_a = cache.values(condition_a, dataset_id, metric)
    values_b = cache.values(condition_b, dataset_id, metric)
    return {name: aligned_arrays(values_a[name], values_b[name], metric) for name in sorted(values_a)}


def subgroup_pairs(arrays: Dict[str, PairArrays], subgroup: str) -> Dict[str, PairArrays]:
    """Return the pairs of a Figure 1 subgroup: ``all``, ``cmt_edas`` or ``other``."""
    if subgroup == "all":
        return arrays
    keep = subgroup == "cmt_edas"
    return {name: pair for name, pair in arrays.items() if (name in SPARSE_PAIRS) == keep}


def pair_delta(pair: PairArrays, metric: str) -> float:
    """Return one pair's observed B − A value."""
    a, b = pair
    return float(statistic(b.sum(axis=0), metric) - statistic(a.sum(axis=0), metric))


def _interval(deltas: np.ndarray) -> Tuple[float, float]:
    """Return the percentile interval of bootstrap replicates."""
    low, high = np.percentile(deltas, [_TAIL, 100 - _TAIL])
    return float(low), float(high)


def pair_level_interval(
    deltas_x: List[float], deltas_y: List[float], settings: TestSettings
) -> Tuple[float, float]:
    """
    Interval of the difference of two subgroup means, resampling whole pairs.

    Parameters
    ----------
    deltas_x, deltas_y : List[float]
        Observed per-pair deltas of the two subgroups.
    settings : TestSettings
        Replicates and seed.

    Returns
    -------
    Tuple[float, float]
        Percentile interval of ``mean(x*) − mean(y*)``.
    """
    rng = np.random.default_rng(settings.seed)
    x, y = np.asarray(deltas_x), np.asarray(deltas_y)
    means_x = x[rng.integers(0, len(x), (settings.n_replicates, len(x)))].mean(axis=1)
    means_y = y[rng.integers(0, len(y), (settings.n_replicates, len(y)))].mean(axis=1)
    return _interval(means_x - means_y)


def subgroup_difference(
    arrays: Dict[str, PairArrays], metric: str, settings: TestSettings
) -> dict:
    """
    Compare the CMT/EDAS subgroup's mean delta with the other pairs'.

    Parameters
    ----------
    arrays : Dict[str, PairArrays]
        Every D3 pair's aligned arrays.
    metric : str
        One of ``paired_tests.METRICS``.
    settings : TestSettings
        Replicates and seed.

    Returns
    -------
    dict
        ``delta`` (CMT/EDAS mean delta minus the others'), ``ci_sources``
        (test sources resampled within pairs) and ``ci_pairs`` (pairs
        resampled within each subgroup).
    """
    sparse = list(subgroup_pairs(arrays, "cmt_edas").values())
    other = list(subgroup_pairs(arrays, "other").values())
    deltas_s = [pair_delta(pair, metric) for pair in sparse]
    deltas_o = [pair_delta(pair, metric) for pair in other]
    within = _bootstrap_deltas(sparse, metric, settings) - _bootstrap_deltas(other, metric, settings)
    return {"delta": float(np.mean(deltas_s) - np.mean(deltas_o)),
            "ci_sources": _interval(within),
            "ci_pairs": pair_level_interval(deltas_s, deltas_o, settings)}


def _direction(row: dict) -> str:
    """Name the condition an interval favours, or ``none`` when it covers 0."""
    if row["ci_low"] > 0:
        return "B"
    return "A" if row["ci_high"] < 0 else "none"


def per_pair_rows(arrays: Dict[str, PairArrays], metric: str, settings: TestSettings) -> List[dict]:
    """Test every D3 pair on its own and name the condition its interval favours."""
    rows = []
    for name, pair in arrays.items():
        row = {"pair": name, "cmt_edas": name in SPARSE_PAIRS, **paired_test([pair], metric, settings)}
        rows.append({**row, "favours": _direction(row)})
    return rows


def figure1(cache: SidecarCache, settings: TestSettings) -> Dict[str, dict]:
    """
    Compute every Figure 1 interval for every metric.

    Parameters
    ----------
    cache : SidecarCache
        Sidecar cache of the results directory.
    settings : TestSettings
        Replicates and seed.

    Returns
    -------
    Dict[str, dict]
        Metric mapped to ``subgroups`` (C10 − C3 over all, CMT/EDAS and other
        pairs), ``difference`` (:func:`subgroup_difference`) and ``pairs``
        (:func:`per_pair_rows`).
    """
    out = {}
    for metric in METRICS:
        arrays = column_arrays(cache, (FIG1_A, FIG1_B), FIG1_DATASET, metric)
        out[metric] = {
            "subgroups": {s: paired_test(list(subgroup_pairs(arrays, s).values()), metric, settings)
                          for s in SUBGROUPS},
            "difference": subgroup_difference(arrays, metric, settings),
            "pairs": per_pair_rows(arrays, metric, settings),
        }
    return out


def figure2_conditions(results_dir: Path, dataset_id: str) -> List[str]:
    """Return the Figure 2 conditions with results on a dataset, in bar order."""
    return [bar.condition_id for bar in BARS
            if any((results_dir / bar.condition_id / dataset_id).glob("*_results.json"))]


def _comparison_rows(cache: SidecarCache, column: dict, settings: TestSettings) -> List[dict]:
    """Test every pair of a column's Figure 2 conditions for one metric."""
    rows, metric = [], column["metric"]
    for condition_a, condition_b in combinations(column["conditions"], 2):
        arrays = column_arrays(cache, (condition_a, condition_b), column["dataset_id"], metric)
        result = paired_test(list(arrays.values()), metric, settings)
        rows.append({"dataset_id": column["dataset_id"], "condition_a": condition_a,
                     "condition_b": condition_b, "metric": metric, **result})
    return rows


def add_holm(rows: List[dict]) -> List[dict]:
    """Add Holm-adjusted p-values to the decision metric's rows of one dataset."""
    decision = [row for row in rows if row["metric"] == DECISION_METRIC]
    for row, adjusted in zip(decision, holm_adjust([row["p_value"] for row in decision])):
        row["p_holm"] = adjusted
    return rows


def figure2(results_dir: Path, cache: SidecarCache, settings: TestSettings) -> List[dict]:
    """
    Compute every Figure 2 pairwise interval on each dataset and metric.

    Parameters
    ----------
    results_dir : Path
        Root of the result tree, to find which conditions ran on each dataset.
    cache : SidecarCache
        Sidecar cache of the same directory.
    settings : TestSettings
        Replicates and seed.

    Returns
    -------
    List[dict]
        One row per dataset, comparison and metric; MRR rows carry ``p_holm``
        (Holm within the dataset's Figure 2 comparisons).
    """
    rows = []
    for dataset in DATASETS:
        conditions = figure2_conditions(results_dir, dataset.dataset_id)
        columns = [{"dataset_id": dataset.dataset_id, "conditions": conditions, "metric": metric}
                   for metric in METRICS]
        rows += add_holm([row for column in columns
                          for row in _comparison_rows(cache, column, settings)])
    return rows


def _fmt(value: float) -> str:
    """Format a signed metric difference."""
    return f"{value:+.4f}"


def _ci(low: float, high: float) -> str:
    """Format an interval."""
    return f"[{low:+.4f}, {high:+.4f}]"


def _table(header: Iterable[str], rows: Iterable[Iterable[str]]) -> List[str]:
    """Render a Markdown table."""
    header = list(header)
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def figure1_markdown(fig1: Dict[str, dict]) -> List[str]:
    """Render the Figure 1 subgroup and subgroup-difference table."""
    rows = []
    for metric, block in fig1.items():
        for subgroup in SUBGROUPS:
            r = block["subgroups"][subgroup]
            rows.append([_METRIC_NAMES[metric], subgroup, str(r["n_pairs"]), f"{r['value_a']:.4f}",
                         f"{r['value_b']:.4f}", _fmt(r["delta"]), _ci(r["ci_low"], r["ci_high"]),
                         f"{r['p_value']:.4f}"])
        d = block["difference"]
        rows.append([_METRIC_NAMES[metric], "cmt_edas − other", "", "", "", _fmt(d["delta"]),
                     f"sources {_ci(*d['ci_sources'])}; pairs {_ci(*d['ci_pairs'])}", ""])
    return _table(["Metric", "Pairs", "n", "C3", "C10", "Δ (C10 − C3)", "95% CI", "p"], rows)


def pair_count_markdown(fig1: Dict[str, dict]) -> List[str]:
    """Render, per metric, how many pairs' own intervals favour each condition."""
    rows = []
    for metric, block in fig1.items():
        for subgroup in ("cmt_edas", "other"):
            pairs = [p for p in block["pairs"] if p["cmt_edas"] == (subgroup == "cmt_edas")]
            count = [p["favours"] for p in pairs].count
            rows.append([_METRIC_NAMES[metric], subgroup, str(len(pairs)),
                         str(sum(p["delta"] > 0 for p in pairs)), str(count("B")), str(count("A"))])
    return _table(["Metric", "Pairs", "n", "C10 ahead (point)", "CI favours C10",
                   "CI favours C3"], rows)


def _holm_cell(row: dict) -> str:
    """Format a Holm p-value, or a dash for metrics without a decision."""
    return f"{row['p_holm']:.4f}" if "p_holm" in row else "—"


def figure2_markdown(rows: List[dict]) -> List[str]:
    """Render the Figure 2 pairwise table."""
    cells = [[r["dataset_id"], _METRIC_NAMES[r["metric"]], f"{r['condition_a']} vs {r['condition_b']}",
              f"{r['value_a']:.4f}", f"{r['value_b']:.4f}", _fmt(r["delta"]),
              _ci(r["ci_low"], r["ci_high"]), f"{r['p_value']:.4f}", _holm_cell(r)] for r in rows]
    return _table(["Dataset", "Metric", "A vs B", "A", "B", "Δ (B − A)", "95% CI", "p",
                   "p (Holm)"], cells)


def render_markdown(fig1: Dict[str, dict], fig2: List[dict]) -> str:
    """
    Render both figures' intervals as a Markdown report.

    Parameters
    ----------
    fig1 : Dict[str, dict]
        Output of :func:`figure1`.
    fig2 : List[dict]
        Output of :func:`figure2`.

    Returns
    -------
    str
        The report.
    """
    lines = ["# Figure claims: paired intervals (#149)", "",
             "_Generated by `scripts/figure_claims.py`; 95% percentile bootstrap, paired "
             "randomisation p, 10,000 replicates, seed 42 (#140 settings)._", "",
             "## Figure 1: D3, C10 (V2+V8/M2) − C3 (V2+V6/M2)", ""]
    lines += figure1_markdown(fig1) + ["", "Pairs whose own interval excludes 0:", ""]
    lines += pair_count_markdown(fig1)
    lines += ["", "## Figure 2: D1/D2, pairwise (exploratory; Holm within dataset, MRR only)", ""]
    return "\n".join(lines + figure2_markdown(fig2)) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(description="Paired intervals behind Figures 1 and 2 (#149).")
    parser.add_argument("--results_dir", type=Path, default=Path("data/results/"),
                        help="Root of the result tree, with per-source sidecars.")
    parser.add_argument("--output_dir", type=Path, default=Path("artifacts/issue-149/"),
                        help="Directory for figure_claims.json and figure_claims.md.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Compute both figures' intervals and write them as JSON and Markdown."""
    random.seed(42)
    np.random.seed(42)
    args = _build_arg_parser().parse_args(argv)
    cache, settings = SidecarCache(args.results_dir), TestSettings()
    fig1, fig2 = figure1(cache, settings), figure2(args.results_dir, cache, settings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"figure1": fig1, "figure2": fig2}
    (args.output_dir / "figure_claims.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.output_dir / "figure_claims.md").write_text(render_markdown(fig1, fig2))
    print(render_markdown(fig1, fig2))


if __name__ == "__main__":
    main()
